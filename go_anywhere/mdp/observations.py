"""Observation terms: teacher-routing skill id and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from go_anywhere.mdp.cells import get_cell_labels
from go_anywhere.terrains.tiles import NUM_SKILLS, Skill

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg

_ROBOT = SceneEntityCfg("robot")


class task_id:
  """One-hot skill id used to route the teacher ensemble.

  Priority: (1) state override to ``DROPS`` when the robot is airborne or not
  upright, (2) the tile label under the base, (3) ``FLAT`` (plane terrain, pits,
  outside the grid).
  Teacher-only: the student never sees this group.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.cells = get_cell_labels(env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    foot_height_sensor: str | None = None,
    airborne_height: float = 0.3,
    fallen_gravity_z: float = -0.3,
    asset_cfg: SceneEntityCfg = _ROBOT,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    n = env.num_envs
    skill = torch.full((n,), int(Skill.FLAT), device=env.device, dtype=torch.long)
    if self.cells.available:
      skill = self.cells.skill_at(asset.data.root_link_pos_w[:, :2])
      skill = torch.where(skill < 0, torch.full_like(skill, int(Skill.FLAT)), skill)
    # State override: airborne or not upright -> drops expert.
    gravity_z = asset.data.projected_gravity_b[:, 2]
    fallen = gravity_z > fallen_gravity_z
    airborne = torch.zeros_like(fallen)
    if foot_height_sensor is not None:
      sensor: TerrainHeightSensor = env.scene[foot_height_sensor]
      heights = sensor.data.heights
      if heights.dim() == 3:
        heights = heights.amin(dim=-1)
      airborne = heights.amin(dim=-1) > airborne_height
    skill = torch.where(
      fallen | airborne, torch.full_like(skill, int(Skill.DROPS)), skill
    )
    return torch.nn.functional.one_hot(skill, NUM_SKILLS).float()


def foot_contact_forces_b(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Net foot contact forces in the base frame, log-compressed, ``[B, F * 3]``.

  The contact sensor reports net forces in the world frame. A world-frame
  vector would make the policy depend on the robot's yaw and has no left-right
  mirror rule, so it is rotated into the base frame like every other vector
  observation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  force_w = sensor.data.force
  assert force_w is not None
  b, f = force_w.shape[:2]
  quat = asset.data.root_link_quat_w[:, None].expand(b, f, 4).reshape(-1, 4)
  force_b = quat_apply_inverse(quat, force_w.reshape(-1, 3)).view(b, f * 3)
  return torch.sign(force_b) * torch.log1p(force_b.abs())


def base_height_above_feet(
  env: ManagerBasedRlEnv, foot_height_sensor: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Base height above the mean ground hit under the feet (privileged), ``[B, 1]``."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: TerrainHeightSensor = env.scene[foot_height_sensor]
  heights = sensor.data.heights
  if heights.dim() == 3:
    heights = heights.amin(dim=-1)
  frame_z = sensor.data.frame_pos_w[:, :, 2]  # [B, F]
  ground_z = (frame_z - heights).mean(dim=-1, keepdim=True)
  return asset.data.root_link_pos_w[:, 2:3] - ground_z
