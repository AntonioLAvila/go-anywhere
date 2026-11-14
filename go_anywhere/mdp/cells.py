"""Env-side view of the terrain cell metadata (skill labels, boxes, routes)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from go_anywhere.terrains.raster import PARKED_Z, read_cells
from go_anywhere.terrains.tiles import Skill

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CellLabels:
  """Per-cell skill labels, boxes and waypoints as device tensors.

  Built from the ``CellMeta`` every go-anywhere cell writes into the terrain
  spec. ``available`` is False when the terrain is a plane.
  """

  def __init__(self, env: ManagerBasedRlEnv):
    self.device = dev = env.device
    self.available = False
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
      return
    assert terrain.terrain_origins is not None
    gen = terrain.cfg.terrain_generator
    rows, cols = terrain.terrain_origins.shape[:2]
    cells = read_cells(terrain.spec, gen.size, rows, cols)
    self.rows, self.cols = rows, cols
    self.size = torch.tensor(gen.size, device=dev, dtype=torch.float32)
    self.grid_offset = torch.tensor(
      [-rows * gen.size[0] * 0.5, -cols * gen.size[1] * 0.5],
      device=dev,
      dtype=torch.float32,
    )
    gy = max(len(m.skills) for m in cells.values())
    gx = max(len(m.skills[0]) for m in cells.values())
    kmax = max(1, max(len(m.boxes) for m in cells.values()))
    wmax = max(len(m.waypoints) for m in cells.values())
    labels = np.full((rows, cols, gy, gx), -1, dtype=np.int64)
    slot_size = np.ones((rows, cols, 2), dtype=np.float32)
    slots = np.ones((rows, cols, 2), dtype=np.int64)  # (nx, ny) per cell.
    boxes = np.zeros((rows, cols, kmax, 6), dtype=np.float32)
    boxes[..., 2] = PARKED_Z
    boxes[..., 3:] = 0.01
    hurdle = np.zeros((rows, cols, kmax), dtype=bool)
    waypoints = np.zeros((rows, cols, wmax, 3), dtype=np.float32)
    route_len = np.zeros((rows, cols), dtype=np.int64)
    ordered = np.zeros((rows, cols), dtype=bool)
    difficulty = np.zeros((rows, cols), dtype=np.float32)
    for (r, c), m in cells.items():
      grid = np.array(m.skills)
      labels[r, c, : grid.shape[0], : grid.shape[1]] = grid
      slot_size[r, c] = m.slot_size
      slots[r, c] = (grid.shape[1], grid.shape[0])
      if m.boxes:
        boxes[r, c, : len(m.boxes)] = m.boxes
        hurdle[r, c, : len(m.boxes)] = m.hurdle_mask
      wp = np.array(m.waypoints, dtype=np.float32)
      corner = (r - rows / 2) * gen.size[0], (c - cols / 2) * gen.size[1]
      wp[:, 0] += corner[0]
      wp[:, 1] += corner[1]
      waypoints[r, c, : len(wp)] = wp
      waypoints[r, c, len(wp) :] = wp[-1]
      route_len[r, c] = len(wp)
      ordered[r, c] = m.ordered
      difficulty[r, c] = m.difficulty
    self.labels = torch.as_tensor(labels, device=dev)
    self.slot_size = torch.as_tensor(slot_size, device=dev)
    self.slots = torch.as_tensor(slots, device=dev)
    self.boxes = torch.as_tensor(boxes, device=dev)
    self.hurdle = torch.as_tensor(hurdle, device=dev)
    self.waypoints = torch.as_tensor(waypoints, device=dev)
    """World-frame goals ``[rows, cols, W, 3]``, padded with the last one."""
    self.route_len = torch.as_tensor(route_len, device=dev)
    self.ordered = torch.as_tensor(ordered, device=dev)
    """Whether a cell's waypoints are a route to follow in order."""
    self.difficulty = torch.as_tensor(difficulty, device=dev)
    self.available = True

  def cell_index(
    self, xy: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(row, col, valid) for world XY ``[B, 2]``."""
    local = xy - self.grid_offset
    row = torch.floor(local[:, 0] / self.size[0]).long()
    col = torch.floor(local[:, 1] / self.size[1]).long()
    valid = (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)
    return row.clamp(0, self.rows - 1), col.clamp(0, self.cols - 1), valid

  def cell_corner(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    return self.grid_offset + torch.stack([row, col], dim=-1) * self.size

  def skill_at(self, xy: torch.Tensor) -> torch.Tensor:
    """Skill id ``[B]`` of the slot under each XY (``FLAT`` outside the grid)."""
    row, col, valid = self.cell_index(xy)
    local = xy - self.cell_corner(row, col)
    slot = torch.floor(local / self.slot_size[row, col]).long()
    slot = torch.minimum(slot.clamp(min=0), self.slots[row, col] - 1)
    out = self.labels[row, col, slot[:, 1], slot[:, 0]]
    return torch.where(valid, out, torch.full_like(out, int(Skill.FLAT)))

  def boxes_at(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame boxes ``[B, K, 6]`` and hurdle mask ``[B, K]`` of each XY's cell."""
    row, col, _ = self.cell_index(xy)
    boxes = self.boxes[row, col].clone()
    boxes[:, :, :2] += self.cell_corner(row, col)[:, None]
    return boxes, self.hurdle[row, col]


def get_cell_labels(env: ManagerBasedRlEnv) -> CellLabels:
  cached = getattr(env, "_go_anywhere_cells", None)
  if cached is None:
    cached = CellLabels(env)
    setattr(env, "_go_anywhere_cells", cached)
  return cached
