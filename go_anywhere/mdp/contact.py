"""Unified world-contact scheme.

All thresholds are expressed in body-weight units (BW) so they transfer across
robots. Literature anchors: ANYmal 700 N ~ 1.4 BW penalty onset and 1500 N ~ 3
BW termination (ANYmal Parkour, Zhang 2024); collision penalty per contacting
body -1 (legged_gym, ANYmal Parkour) to -10 (Extreme Parkour, PIE); stumble
``|F_xy| > 4 |F_z|``; feet within 5 cm of an edge (Extreme Parkour).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply

from go_anywhere.assets.go2 import BODY_WEIGHT_N, NON_FOOT_GEOMS
from go_anywhere.mdp.cells import get_cell_labels

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

IMPACT_ONSET_BW = 1.4
IMPACT_TERMINATE_BW = 3.0
CONTACT_FORCE_THRESHOLD_N = 1.0


def _force_magnitude(sensor: ContactSensor) -> torch.Tensor:
  """``[B, N]`` max contact force magnitude per primary over the history (if any)."""
  data = sensor.data
  if data.force_history is not None:
    return torch.norm(data.force_history, dim=-1).amax(dim=-1)
  assert data.force is not None
  return torch.norm(data.force, dim=-1)


def world_contact_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = CONTACT_FORCE_THRESHOLD_N,
) -> torch.Tensor:
  """Number of non-foot bodies touching the world this step."""
  sensor: ContactSensor = env.scene[sensor_name]
  return (_force_magnitude(sensor) > force_threshold).sum(dim=-1).float()


def stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  ratio: float = 4.0,
  force_threshold: float = CONTACT_FORCE_THRESHOLD_N,
) -> torch.Tensor:
  """1 if any foot has a mostly tangential contact force (hit an edge)."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  tangential = torch.norm(force[..., :2], dim=-1)
  normal = force[..., 2].abs()
  hit = (tangential > ratio * normal) & (torch.norm(force, dim=-1) > force_threshold)
  return hit.any(dim=-1).float()


def feet_edge(
  env: ManagerBasedRlEnv,
  ring_sensor_name: str,
  contact_sensor_name: str,
  drop: float = 0.05,
) -> torch.Tensor:
  """Count feet in contact whose surrounding ring shows a drop > ``drop``.

  ``ring_sensor_name`` must be a ``TerrainHeightSensorCfg`` with
  ``reduction="none"`` and one frame per foot in the same order as the contact
  sensor's primaries.
  """
  ring: TerrainHeightSensor = env.scene[ring_sensor_name]
  heights = ring.data.heights  # [B, F, N]
  assert heights.dim() == 3, "feet_edge needs reduction='none' on the ring sensor"
  spread = heights.amax(dim=-1) - heights.amin(dim=-1)
  contact: ContactSensor = env.scene[contact_sensor_name]
  found = contact.data.found
  assert found is not None
  on_edge = (spread > drop) & (found > 0)
  return on_edge.sum(dim=-1).float()


def foot_impact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  onset_bw: float = IMPACT_ONSET_BW,
  body_weight_n: float = BODY_WEIGHT_N,
) -> torch.Tensor:
  """Quadratic penalty on per-foot force above ``onset_bw`` body weights."""
  sensor: ContactSensor = env.scene[sensor_name]
  mag = _force_magnitude(sensor) / body_weight_n
  excess = (mag - onset_bw).clamp(min=0.0)
  return (excess**2).sum(dim=-1)


def peak_foot_impact_bw(
  env: ManagerBasedRlEnv, sensor_name: str, body_weight_n: float = BODY_WEIGHT_N
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return (_force_magnitude(sensor) / body_weight_n).amax(dim=-1)


def box_penetration_depth(
  points: torch.Tensor, centers: torch.Tensor, half_sizes: torch.Tensor
) -> torch.Tensor:
  """Penetration depth of ``points [B, P, 3]`` into boxes ``[B, K, 3]`` -> ``[B, P, K]``.

  Depth is the smallest axis clearance to a face (0 outside the box).
  """
  d = half_sizes[:, None] - (points[:, :, None, :] - centers[:, None]).abs()
  return d.amin(dim=-1).clamp(min=0.0)


class penetration_cost:
  """Soft-to-hard hurdle penalty (Zhuang et al. 2023).

  Samples points on the robot's non-foot collision geoms, computes their
  penetration depth into the hurdle boxes of the robot's current cell and
  scales by point speed so sprinting through a bar is never free.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get(
      "asset_cfg", SceneEntityCfg("robot", geom_names=NON_FOOT_GEOMS)
    )
    self.asset: Entity = env.scene[asset_cfg.name]
    geom_ids, _ = self.asset.find_geoms(asset_cfg.geom_names or NON_FOOT_GEOMS)
    self.geom_ids = torch.as_tensor(geom_ids, device=env.device, dtype=torch.long)
    mj_model: mujoco.MjModel = env.sim.mj_model
    global_ids = self.asset.indexing.geom_ids[self.geom_ids].cpu().numpy()
    offsets = []
    for gid in global_ids:
      gtype = int(mj_model.geom_type[gid])
      sx, sy, sz = (float(v) for v in mj_model.geom_size[gid])
      pts = [(0.0, 0.0, 0.0)]
      if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        for ax in (-0.9, 0.9):
          for ay in (-0.9, 0.9):
            for az in (-0.9, 0.9):
              pts.append((ax * sx, ay * sy, az * sz))
      elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        pts += [(0.0, 0.0, sy), (0.0, 0.0, -sy), (sx, 0.0, 0.0), (-sx, 0.0, 0.0)]
      elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        pts += [(sx, 0, 0), (-sx, 0, 0), (0, 0, sx), (0, 0, -sx)]
      offsets.append(torch.tensor(pts, device=env.device, dtype=torch.float32))
    self.max_pts = max(o.shape[0] for o in offsets)
    padded = torch.zeros(len(offsets), self.max_pts, 3, device=env.device)
    valid = torch.zeros(len(offsets), self.max_pts, dtype=torch.bool, device=env.device)
    for i, o in enumerate(offsets):
      padded[i, : o.shape[0]] = o
      valid[i, : o.shape[0]] = True
    self.offsets = padded  # [G, P, 3]
    self.valid = valid
    self.cells = get_cell_labels(env)

  def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
    if not self.cells.available:
      return torch.zeros(env.num_envs, device=env.device)
    pose = self.asset.data.geom_pose_w[:, self.geom_ids]  # [B, G, 7]
    vel = self.asset.data.geom_vel_w[:, self.geom_ids, :3]  # [B, G, 3]
    b, g, p = pose.shape[0], pose.shape[1], self.max_pts
    quat = pose[:, :, None, 3:7].expand(b, g, p, 4).reshape(-1, 4)
    off = self.offsets[None].expand(b, g, p, 3).reshape(-1, 3)
    pts = pose[:, :, None, :3] + quat_apply(quat, off).view(b, g, p, 3)
    speed = torch.norm(vel, dim=-1)[:, :, None].expand(b, g, p)
    pts = pts.reshape(b, g * p, 3)
    speed = speed.reshape(b, g * p)
    valid = self.valid[None].expand(b, g, p).reshape(b, g * p)

    base_xy = self.asset.data.root_link_pos_w[:, :2]
    boxes, hurdle = self.cells.boxes_at(base_xy)  # [B, K, 6], [B, K]
    depth = box_penetration_depth(pts, boxes[:, :, :3], boxes[:, :, 3:])  # [B, GP, K]
    depth = depth * hurdle[:, None, :].float()
    depth = depth.amax(dim=-1)  # [B, GP]
    cost = (depth * (speed + 1e-3) * valid.float()).sum(dim=-1)
    return cost


# Terminations.


def foot_impact_exceeded(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  limit_bw: float = IMPACT_TERMINATE_BW,
  body_weight_n: float = BODY_WEIGHT_N,
) -> torch.Tensor:
  return peak_foot_impact_bw(env, sensor_name, body_weight_n) > limit_bw


def trunk_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = CONTACT_FORCE_THRESHOLD_N,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return (_force_magnitude(sensor) > force_threshold).any(dim=-1)
