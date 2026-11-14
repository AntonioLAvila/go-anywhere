import math
from types import SimpleNamespace

import mjlab  # noqa: F401
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from tensordict import TensorDict

from go_anywhere.mdp.observations import foot_contact_forces_b
from go_anywhere.rl.symmetry import (
  FOOT_PERM,
  JOINT_PERM,
  JOINT_SIGN,
  TERM_RULES,
  mirror_lr,
)


def test_joint_mirror_is_an_involution():
  perm = torch.tensor(JOINT_PERM)
  sign = torch.tensor(JOINT_SIGN)
  x = torch.randn(12)
  y = x[perm] * sign
  z = y[perm] * sign
  assert torch.allclose(x, z)
  assert sorted(FOOT_PERM) == [0, 1, 2, 3]


def test_no_world_frame_terms_have_a_rule():
  assert not [name for name in TERM_RULES if name.endswith("_w")]


def test_foot_contact_force_rule_holds_at_any_attitude():
  """World-frame foot forces of a reflected robot (y -> -y, feet swapped) must
  map onto the mirror rule once rotated into the base frame."""
  n = 64
  quat = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
  force_w = 30.0 * torch.randn(n, 4, 3)
  quat_m = quat * torch.tensor([1.0, -1.0, 1.0, -1.0])
  force_w_m = force_w[:, FOOT_PERM] * torch.tensor([1.0, -1.0, 1.0])

  def observe(q, f):
    scene = {
      "robot": SimpleNamespace(data=SimpleNamespace(root_link_quat_w=q)),
      "feet": SimpleNamespace(data=SimpleNamespace(force=f)),
    }
    return foot_contact_forces_b(SimpleNamespace(scene=scene), "feet")

  perm, sign = TERM_RULES["foot_contact_forces"](None, 12)
  expected = observe(quat, force_w)[:, perm] * torch.tensor(sign)
  assert torch.allclose(observe(quat_m, force_w_m), expected, atol=1e-4)


@pytest.mark.gpu
def test_mirror_matches_a_physically_reflected_rollout(device):
  """Envs [N:] start as the reflection of envs [:N] about the world XZ plane
  (random yaw, so the plane is not the robot's sagittal plane) and receive the
  mirrored actions. Their observations must equal ``mirror_lr`` of the originals.

  The robots are compared in free fall: once feet touch the ground even two
  identical envs drift apart within a few steps (contact solver), which would
  hide a wrong rule. Contact terms are covered by the test above.
  """
  n = 16
  cfg = load_env_cfg("Go-Anywhere-Expert-Flat-Go2")
  cfg.scene.num_envs = 2 * n
  cfg.scene.terrain.terrain_generator.num_rows = 2
  cfg.scene.terrain.terrain_generator.border_width = 5.0
  # Per-env randomness would differ between an env and its twin.
  for group in ("proprio_hist", "terrain_hist"):
    cfg.observations[group].enable_corruption = False
  scan = cfg.observations["terrain_hist"].terms["height_scan"]
  scan.delay_min_lag = scan.delay_max_lag = 0
  cfg.events = {k: v for k, v in cfg.events.items() if k.startswith("reset_")}
  cfg.curriculum = {}
  cfg.terminations = {}
  goal_cfg = cfg.commands["goal"]
  goal_cfg.resample_on_success = False
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    robot = env.scene["robot"]
    goal = env.command_manager.get_term("goal")
    a_ids = torch.arange(n, device=device)
    b_ids = a_ids + n
    origins = env.scene.env_origins
    perm = torch.tensor(JOINT_PERM, device=device)
    sign = torch.tensor(JOINT_SIGN, device=device)
    flip_y = torch.tensor([1.0, -1.0, 1.0], device=device)

    gen = torch.Generator(device=device).manual_seed(0)

    def rand(*shape):
      return torch.rand(*shape, device=device, generator=gen) * 2 - 1

    yaw = rand(n) * math.pi
    quat = quat_from_euler_xyz(0.3 * rand(n), 0.3 * rand(n), yaw)
    rel = torch.cat([0.3 * rand(n, 2), torch.full((n, 1), 0.8, device=device)], dim=-1)
    joints = robot.data.default_joint_pos[a_ids] + 0.1 * rand(n, 12)
    joint_vel = 3.0 * rand(n, 12)
    lin_vel, ang_vel = rand(n, 3), rand(n, 3)
    goal_rel = torch.cat([3.0 * rand(n, 2), torch.zeros(n, 1, device=device)], dim=-1)
    goal_yaw = rand(n) * math.pi

    # Reflection y -> -y: polar vectors flip y, axial vectors flip x and z, the
    # quaternion (w, x, y, z) becomes (w, -x, y, -z).
    quat_m = quat * torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)
    for ids, p, q, v, w, j, jv in (
      (a_ids, rel, quat, lin_vel, ang_vel, joints, joint_vel),
      (
        b_ids,
        rel * flip_y,
        quat_m,
        lin_vel * flip_y,
        ang_vel * -flip_y,
        joints[:, perm] * sign,
        joint_vel[:, perm] * sign,
      ),
    ):
      robot.write_root_link_pose_to_sim(torch.cat([origins[ids] + p, q], -1), ids)
      robot.write_root_link_velocity_to_sim(torch.cat([v, w], -1), ids)
      robot.write_joint_state_to_sim(j, jv, env_ids=ids)
    goal.goal_pos_w[a_ids] = origins[a_ids] + goal_rel
    goal.goal_pos_w[b_ids] = origins[b_ids] + goal_rel * flip_y
    goal.goal_yaw_w[a_ids] = goal_yaw
    goal.goal_yaw_w[b_ids] = -goal_yaw
    for t in (goal.budget, goal.budget_left):
      t[b_ids] = t[a_ids]
    env.sim.forward()

    om = env.observation_manager
    for _ in range(5):
      act = 0.2 * rand(n, 12)
      obs, *_ = env.step(torch.cat([act, act[:, perm] * sign], dim=0))
    original = TensorDict({k: v[a_ids] for k, v in obs.items()}, batch_size=[n])
    mirrored, _ = mirror_lr(env, original)
    for group in obs.keys():
      expected, actual = mirrored[group][n:], obs[group][b_ids]
      if expected.dim() == 3:  # Older history frames predate the state override.
        expected, actual = expected[:, -1], actual[:, -1]
      offset = 0
      for term, shape in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        d = shape[-1]
        e, a = expected[:, offset : offset + d], actual[:, offset : offset + d]
        offset += d
        scale = max(1.0, a.abs().max().item())
        err = (e - a).abs().max().item()
        assert err < 0.01 * scale, f"{group}/{term}: err {err:.4f} vs scale {scale:.3f}"
  finally:
    env.close()
