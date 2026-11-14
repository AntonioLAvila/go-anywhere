"""Reset events: airborne drops, fallen poses, reference-state banks."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

from go_anywhere.assets.go2 import TUCK_JOINT_POS

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

_ROBOT = SceneEntityCfg("robot")


def _write_root(
  asset: Entity,
  env_ids: torch.Tensor,
  pos: torch.Tensor,
  quat: torch.Tensor,
  lin_vel: torch.Tensor,
  ang_vel: torch.Tensor,
) -> None:
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(
    torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids
  )


def _write_joints_noisy(
  asset: Entity,
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  base: torch.Tensor,
  noise: float,
) -> None:
  limits = asset.data.soft_joint_pos_limits[env_ids]
  jp = base + sample_uniform(-noise, noise, base.shape, env.device)
  jp = jp.clamp(limits[..., 0], limits[..., 1])
  jv = torch.zeros_like(jp)
  asset.write_joint_state_to_sim(jp, jv, env_ids=env_ids)


def reset_airborne(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  height_range: tuple[float, float] = (0.3, 0.8),
  roll_pitch_max: float = math.radians(30.0),
  ang_vel_max: float = 0.5,
  lin_vel_xy_max: float = 0.5,
  lin_vel_z_range: tuple[float, float] = (-1.0, 0.0),
  xy_jitter: float = 0.5,
  joint_noise: float = 0.3,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Release the robot in the air above its spawn origin (drops family).

  Envelope anchors: Jeon 2022 (pitch +-60 deg, roll +-30 deg, |w| 0.5 rad/s,
  |v_xy| 1.5 m/s), Roscia 2023 (Go1 0.35-1.0 m). Push the ranges with the
  ``event_param_stages`` curriculum. Actuator blackout (Ma 2023) is a Phase-1
  follow-up.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  n = len(env_ids)
  dev = env.device
  origins = env.scene.env_origins[env_ids]
  pos = origins.clone()
  pos[:, :2] += sample_uniform(-xy_jitter, xy_jitter, (n, 2), dev)
  pos[:, 2] += sample_uniform(*height_range, (n,), dev)
  roll = sample_uniform(-roll_pitch_max, roll_pitch_max, (n,), dev)
  pitch = sample_uniform(-roll_pitch_max, roll_pitch_max, (n,), dev)
  yaw = sample_uniform(-math.pi, math.pi, (n,), dev)
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  lin_vel = torch.zeros(n, 3, device=dev)
  lin_vel[:, :2] = sample_uniform(-lin_vel_xy_max, lin_vel_xy_max, (n, 2), dev)
  lin_vel[:, 2] = sample_uniform(*lin_vel_z_range, (n,), dev)
  ang_vel = sample_uniform(-ang_vel_max, ang_vel_max, (n, 3), dev)
  _write_root(asset, env_ids, pos, quat, lin_vel, ang_vel)
  _write_joints_noisy(
    asset, env, env_ids, asset.data.default_joint_pos[env_ids], joint_noise
  )


def reset_fallen(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mode: Literal["supine", "prone", "random"] = "supine",
  height: float = 0.15,
  xy_jitter: float = 0.3,
  joint_noise: float = 0.5,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Start on the ground upside-down / on the side / in a random attitude."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  n, dev = len(env_ids), env.device
  pos = env.scene.env_origins[env_ids].clone()
  pos[:, :2] += sample_uniform(-xy_jitter, xy_jitter, (n, 2), dev)
  pos[:, 2] += height
  yaw = sample_uniform(-math.pi, math.pi, (n,), dev)
  if mode == "supine":
    roll = math.pi + sample_uniform(-0.3, 0.3, (n,), dev)
    pitch = sample_uniform(-0.2, 0.2, (n,), dev)
  elif mode == "prone":
    roll = sample_uniform(-0.3, 0.3, (n,), dev)
    pitch = sample_uniform(-0.3, 0.3, (n,), dev)
  else:
    roll = sample_uniform(-math.pi, math.pi, (n,), dev)
    pitch = sample_uniform(-math.pi / 2, math.pi / 2, (n,), dev)
    pos[:, 2] += 0.25
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  zeros = torch.zeros(n, 3, device=dev)
  _write_root(asset, env_ids, pos, quat, zeros, zeros)
  # Joints between home and tuck.
  default = asset.data.default_joint_pos[env_ids]
  tuck = default.clone()
  _, names = asset.find_joints((".*",))
  for j, name in enumerate(names):
    for pattern, val in TUCK_JOINT_POS.items():
      if re.fullmatch(pattern, name):
        tuck[:, j] = val
  alpha = sample_uniform(0.0, 1.0, (n, 1), dev)
  base = default + alpha * (tuck - default)
  _write_joints_noisy(asset, env, env_ids, base, joint_noise)


class reset_from_reference_states:
  """Reverse-curriculum reset from a bank of reference states.

  Bank format (``.npz``): ``qpos [N, 7 + nj]`` (root pos, root quat wxyz,
  joints), ``qvel [N, 6 + nj]`` (root lin/ang vel in world frame, joint vel),
  optionally ``phase [N]`` in [0, 1] (fraction of the reference motion). Root
  XY are relative to the env origin; ``z`` is absolute above the spawn-cell
  ground. States are written as recorded (no yaw randomization: pose and
  velocity are tied to the obstacle they were recorded on). ``index_range``
  selects a window of phases; the ``event_param_stages`` curriculum moves the
  window backward over training.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    bank_path = Path(cfg.params["bank_path"])
    if not bank_path.exists():
      raise FileNotFoundError(f"Reference bank {bank_path} not found.")
    data = np.load(bank_path)
    dev = env.device
    self.qpos = torch.as_tensor(data["qpos"], device=dev, dtype=torch.float32)
    self.qvel = torch.as_tensor(data["qvel"], device=dev, dtype=torch.float32)
    if "phase" in data:
      self.phase = torch.as_tensor(data["phase"], device=dev, dtype=torch.float32)
    else:
      self.phase = torch.linspace(0.0, 1.0, self.qpos.shape[0], device=dev)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    bank_path: str,
    index_range: tuple[float, float] = (0.8, 1.0),
    asset_cfg: SceneEntityCfg = _ROBOT,
  ) -> None:
    del bank_path  # Consumed in __init__.
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    lo, hi = index_range
    candidates = ((self.phase >= lo) & (self.phase <= hi)).nonzero().flatten()
    if candidates.numel() == 0:
      raise ValueError(f"No reference states with phase in [{lo}, {hi}].")
    idx = candidates[
      torch.randint(0, candidates.numel(), (len(env_ids),), device=env.device)
    ]
    qpos, qvel = self.qpos[idx], self.qvel[idx]
    pos = qpos[:, :3] + env.scene.env_origins[env_ids]
    _write_root(asset, env_ids, pos, qpos[:, 3:7], qvel[:, :3], qvel[:, 3:6])
    asset.write_joint_state_to_sim(qpos[:, 7:], qvel[:, 6:], env_ids=env_ids)
