"""Left-right mirroring for rsl_rl's symmetry extension.

The mirror is the reflection about the robot's own sagittal plane, applied to
quantities expressed in the base frame or the yaw-aligned base frame. It holds
for any heading and any goal direction because the command is mirrored with
the robot (a goal ahead-left becomes a goal ahead-right). It does NOT hold for
world-frame quantities: a fixed sign flip of a world-frame vector is a
reflection about a world plane, which matches the robot's sagittal plane only
at zero yaw. Every term in the table below must therefore be body- or
yaw-frame; ``tests/test_symmetry.py`` checks the table against the simulator
by comparing a rollout with its physically reflected twin.

Builds, once per env, a permutation + sign vector for every observation group
from the observation manager's term layout. Unknown term names raise so a
silently wrong mirror cannot slip into training.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from tensordict import TensorDict

from go_anywhere.mdp.commands import COMMAND_DIM

Rule = tuple[list[int], list[float]]

# Go2 joint order: FL(hip, thigh, calf), FR, RL, RR. Hip abduction flips sign.
JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
JOINT_SIGN = [-1.0, 1.0, 1.0] * 4
FOOT_PERM = [1, 0, 3, 2]  # (FL, FR, RL, RR) -> (FR, FL, RR, RL)
VECTOR_SIGN = [1.0, -1.0, 1.0]  # Polar vector (velocity, force, gravity).
AXIAL_SIGN = [-1.0, 1.0, -1.0]  # Axial vector (angular velocity).
# [dx, dy, dz, cos(heading err), sin(heading err), budget fraction, speed].
COMMAND_SIGN = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
assert len(COMMAND_SIGN) == COMMAND_DIM


def _joints(env: Any, dim: int) -> Rule:
  return JOINT_PERM, JOINT_SIGN


def _per_foot(env: Any, dim: int) -> Rule:
  return FOOT_PERM, [1.0] * 4


def _per_foot_vector(env: Any, dim: int) -> Rule:
  perm = [3 * f + i for f in FOOT_PERM for i in range(3)]
  return perm, VECTOR_SIGN * 4


def _invariant(env: Any, dim: int) -> Rule:
  return list(range(dim)), [1.0] * dim


def _terrain_scan(env: Any, dim: int) -> Rule:
  """Permutation of the yaw-aligned ray grid under y -> -y."""
  sensor = env.scene["terrain_scan"]
  if sensor.cfg.ray_alignment != "yaw":
    raise ValueError("The terrain scan must be yaw-aligned to be mirrored.")
  offsets, _ = sensor.cfg.pattern.generate_rays(None, "cpu")
  pts = offsets[:, :2]
  mirrored = pts * torch.tensor([1.0, -1.0])
  perm = torch.cdist(mirrored, pts).argmin(dim=1)
  if not torch.allclose(pts[perm], mirrored, atol=1e-4):
    raise ValueError("The terrain scan ray pattern is not left-right symmetric.")
  return perm.tolist(), [1.0] * len(perm)


# Observation term name -> mirror rule. Add an entry for every new term.
TERM_RULES: dict[str, Callable[[Any, int], Rule]] = {
  "base_ang_vel": lambda env, dim: ([0, 1, 2], AXIAL_SIGN),
  "base_lin_vel": lambda env, dim: ([0, 1, 2], VECTOR_SIGN),
  "projected_gravity": lambda env, dim: ([0, 1, 2], VECTOR_SIGN),
  "joint_pos": _joints,
  "joint_pos_clean": _joints,
  "joint_vel": _joints,
  "actions": _joints,
  "command": lambda env, dim: (list(range(COMMAND_DIM)), COMMAND_SIGN),
  "height_scan": _terrain_scan,
  "height_scan_clean": _terrain_scan,
  "foot_height": _per_foot,
  "foot_air_time": _per_foot,
  "foot_contact": _per_foot,
  "foot_contact_forces": _per_foot_vector,
  "base_height_above_feet": _invariant,
  "task_id": _invariant,
}


def _term_rule(env: Any, term: str, dim: int) -> Rule:
  """Permutation/sign for one observation term (indices local to the term)."""
  if term not in TERM_RULES:
    raise ValueError(f"No mirror rule for observation term '{term}' (dim {dim}).")
  perm, sign = TERM_RULES[term](env, dim)
  if sorted(perm) != list(range(dim)) or len(sign) != dim:
    raise ValueError(f"Mirror rule for '{term}' does not match its dim {dim}.")
  return perm, sign


class _MirrorSpec:
  def __init__(self, env: Any):
    om = env.observation_manager
    self.perm: dict[str, torch.Tensor] = {}
    self.sign: dict[str, torch.Tensor] = {}
    for group, terms in om.active_terms.items():
      dims = om.group_obs_term_dim[group]
      perm: list[int] = []
      sign: list[float] = []
      offset = 0
      for term, shape in zip(terms, dims, strict=True):
        d = int(shape[-1])
        p, s = _term_rule(env, term, d)
        perm += [offset + i for i in p]
        sign += list(s)
        offset += d
      self.perm[group] = torch.tensor(perm, device=env.device)
      self.sign[group] = torch.tensor(sign, device=env.device)
    self.action_perm = torch.tensor(JOINT_PERM, device=env.device)
    self.action_sign = torch.tensor(JOINT_SIGN, device=env.device)


def _spec(env: Any) -> _MirrorSpec:
  base = env.unwrapped if hasattr(env, "unwrapped") else env
  spec = getattr(base, "_go_anywhere_mirror", None)
  if spec is None:
    spec = _MirrorSpec(base)
    setattr(base, "_go_anywhere_mirror", spec)
  return spec


def mirror_lr(
  env: Any,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
  **_: Any,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Return (obs, actions) with mirrored copies appended along the batch dim."""
  spec = _spec(env)
  obs_out = None
  if obs is not None:
    mirrored = {}
    for group in obs.keys():
      assert isinstance(group, str)
      x = obs[group]
      perm, sign = spec.perm[group], spec.sign[group]
      mirrored[group] = torch.cat([x, x[..., perm] * sign], dim=0)
    obs_out = TensorDict(mirrored, batch_size=[obs.batch_size[0] * 2])
  act_out = None
  if actions is not None:
    act_out = torch.cat(
      [actions, actions[..., spec.action_perm] * spec.action_sign], dim=0
    )
  return obs_out, act_out
