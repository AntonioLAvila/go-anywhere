"""Stones-Everywhere cell (Zhang, Rudin, Hoeller, Hutter 2024, Table I).

The whole cell is a grid of square stones over a pit. Per level (linear in
difficulty): stone size 92 -> 20 cm, gap 8 -> 26 cm, XY shift +-3.6 -> +-11.8 cm,
height +-1 -> +-10 cm. We add a small flat spawn platform at the cell centre
(the paper does not state its spawn procedure) and sample goals on stone tops
at a distance range (unordered goal candidates in the cell metadata).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainOutput,
)

from go_anywhere.terrains.raster import (
  CellMeta,
  add_heightfield,
  jittered_patches,
  write_cell_meta,
)
from go_anywhere.terrains.tiles import Skill

# Zhang et al. 2024 Table I, levels 0 and 9.
STONE_SIZE_RANGE = (0.92, 0.20)
STONE_GAP_RANGE = (0.08, 0.26)
STONE_SHIFT_RANGE = (0.036, 0.118)
STONE_HEIGHT_RANGE = (0.01, 0.10)


def stones_params(difficulty: float) -> dict[str, float]:
  d = float(np.clip(difficulty, 0.0, 1.0))

  def lin(r: tuple[float, float]) -> float:
    return r[0] + d * (r[1] - r[0])

  return {
    "stone": lin(STONE_SIZE_RANGE),
    "gap": lin(STONE_GAP_RANGE),
    "shift": lin(STONE_SHIFT_RANGE),
    "height": lin(STONE_HEIGHT_RANGE),
  }


def sparsity(params: dict[str, float]) -> float:
  """Fraction of non-steppable area at zero shift: 1 - (s / (s + g))^2."""
  s, g = params["stone"], params["gap"]
  return 1.0 - (s / (s + g)) ** 2


@dataclass(kw_only=True)
class StonesEverywhereTerrainCfg(SubTerrainCfg):
  resolution: float = 0.025
  floor_depth: float = 1.0
  platform_width: float = 1.0
  goal_distance_range: tuple[float, float] = (1.5, 3.5)
  max_goals: int = 32
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = field(
    default_factory=lambda: {
      "spawn": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.3),
    }
  )

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    p = stones_params(difficulty)
    res = self.resolution
    nx, ny = int(round(self.size[0] / res)), int(round(self.size[1] / res))
    cell = np.full((ny, nx), -self.floor_depth, dtype=np.float64)
    pitch = p["stone"] + p["gap"]
    n_x = max(1, int(np.floor(self.size[0] / pitch)))
    n_y = max(1, int(np.floor(self.size[1] / pitch)))
    off_x = 0.5 * (self.size[0] - n_x * pitch) + pitch / 2
    off_y = 0.5 * (self.size[1] - n_y * pitch) + pitch / 2
    centers: list[tuple[float, float, float]] = []
    half = p["stone"] / 2
    cx0, cy0 = self.size[0] / 2, self.size[1] / 2
    pw = self.platform_width / 2
    for i in range(n_x):
      for j in range(n_y):
        cx = off_x + i * pitch + rng.uniform(-p["shift"], p["shift"])
        cy = off_y + j * pitch + rng.uniform(-p["shift"], p["shift"])
        z = rng.uniform(-p["height"], p["height"])
        ix0, ix1 = int((cx - half) / res), int(np.ceil((cx + half) / res))
        iy0, iy1 = int((cy - half) / res), int(np.ceil((cy + half) / res))
        cell[max(0, iy0) : min(ny, iy1), max(0, ix0) : min(nx, ix1)] = z
        if not (abs(cx - cx0) < pw + half and abs(cy - cy0) < pw + half):
          centers.append((cx, cy, z))
    # Flat spawn platform in the centre.
    px0, px1 = int((cx0 - pw) / res), int(np.ceil((cx0 + pw) / res))
    py0, py1 = int((cy0 - pw) / res), int(np.ceil((cy0 + pw) / res))
    cell[py0:py1, px0:px1] = 0.0

    geom, _ = add_heightfield(spec, body, cell, self.size)
    origin = np.array([cx0, cy0, 0.0])

    # Goal candidates: stone tops within the distance range of the platform.
    cands = np.array(centers) if centers else origin[None, :]
    dist = np.linalg.norm(cands[:, :2] - origin[None, :2], axis=1)
    lo, hi = self.goal_distance_range
    sel = cands[(dist >= lo) & (dist <= hi)]
    if len(sel) == 0:
      sel = cands
    goals = sel[rng.permutation(len(sel))[: self.max_goals]]
    num = (self.flat_patch_sampling or {}).get("spawn", FlatPatchSamplingCfg())

    assert geom.geom is not None
    write_cell_meta(
      spec,
      geom.geom,
      CellMeta(
        difficulty=float(difficulty),
        slot_size=(float(self.size[0]), float(self.size[1])),
        skills=[[int(Skill.STEPS)]],
        waypoints=goals.tolist(),
        ordered=False,
        path=[((0, 0), "stones_everywhere", {**p, "sparsity": sparsity(p)})],
      ),
    )
    return TerrainOutput(
      origin=origin,
      geometries=[geom],
      flat_patches={"spawn": jittered_patches(origin, 0.3, num.num_patches, rng)},
    )
