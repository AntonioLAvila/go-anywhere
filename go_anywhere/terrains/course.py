"""Course cell: path-first tile grammar + model-synthesis fill + fBm.

One ``CourseTerrainCfg.function`` call produces a single 8x8 m heightfield, a
fixed budget of box primitives, ``spawn`` flat patches and a ``CellMeta``
(skill per tile slot, hurdle boxes, the route as ordered waypoints) written
into the spec so the env can route teachers, compute penetration costs and
command goals along the path.
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

from go_anywhere.terrains.grammar import Path, check_feasible, sample_path
from go_anywhere.terrains.noise import band_limited_roughness
from go_anywhere.terrains.raster import (
  CellMeta,
  add_boxes,
  add_heightfield,
  jittered_patches,
  write_cell_meta,
)
from go_anywhere.terrains.synthesis import fill_free_slots
from go_anywhere.terrains.tiles import (
  TILE_SKILL,
  BoxSpec,
  CapabilityEnvelope,
  Tile,
  TileKind,
  rasterize_tile,
  sample_tile,
)


def default_patch_sampling(num_patches: int = 8) -> dict[str, FlatPatchSamplingCfg]:
  return {"spawn": FlatPatchSamplingCfg(num_patches=num_patches, patch_radius=0.3)}


def route_waypoints(path: Path, tile_size: float) -> np.ndarray:
  """``[L, 3]`` goals along the path: the midpoint of each edge shared by
  consecutive path tiles (inside both tiles' flat edge strips), then the centre
  of the last tile. Consecutive waypoints are joined by a straight segment that
  stays on path tiles."""
  pts = []
  for p in path.placements:
    cx, cy = (p.slot[0] + 0.5) * tile_size, (p.slot[1] + 0.5) * tile_size
    if p.exit is None:
      pts.append((cx, cy, p.tile.exit_h))
    else:
      half = tile_size / 2
      pts.append((cx + p.exit[0] * half, cy + p.exit[1] * half, p.tile.exit_h))
  return np.array(pts)


def add_corridor_walls(
  cell: np.ndarray, path: Path, n: int, thickness_px: int, height: float
) -> np.ndarray:
  """Raise a wall along every path-tile edge the path does not cross, so the
  route cannot be left to walk around an obstacle. Returns the wall mask."""
  mask = np.zeros(cell.shape, dtype=bool)
  k = thickness_px
  for p in path.placements:
    i, j = p.slot
    rows, cols = slice(j * n, (j + 1) * n), slice(i * n, (i + 1) * n)
    open_edges = {p.exit, (-p.entry[0], -p.entry[1]) if p.entry else None}
    top = max(p.tile.entry_h, p.tile.exit_h) + height
    for edge, sel in (
      ((1, 0), (rows, slice((i + 1) * n - k, (i + 1) * n))),
      ((-1, 0), (rows, slice(i * n, i * n + k))),
      ((0, 1), (slice((j + 1) * n - k, (j + 1) * n), cols)),
      ((0, -1), (slice(j * n, j * n + k), cols)),
    ):
      if edge not in open_edges:
        cell[sel] = top
        mask[sel] = True
  return mask


def _place(
  cell: np.ndarray,
  tile_h: np.ndarray,
  slot: tuple[int, int],
  direction: tuple[int, int],
  n: int,
) -> None:
  """Write a local ``[iu, iv]`` tile into the cell array ``[iy, ix]``.

  Slot ``(i, j)`` maps to x = i * tile, y = j * tile. Direction (di, dj) is the
  travel direction in slot units; ``u`` runs along it.
  """
  di, dj = direction
  if (di, dj) == (1, 0):  # +x: u -> x, v -> y
    block = tile_h.T  # [iv, iu] -> [iy, ix]
  elif (di, dj) == (-1, 0):  # -x
    block = tile_h[::-1, :].T
  elif (di, dj) == (0, 1):  # +y: u -> y, v -> x (mirror v so frame stays right-handed)
    block = tile_h[:, ::-1]
  else:  # (0, -1): -y
    block = tile_h[::-1, :]
  i, j = slot
  cell[j * n : (j + 1) * n, i * n : (i + 1) * n] = block


def _box_to_cell(
  box: BoxSpec, slot: tuple[int, int], direction: tuple[int, int], tile_size: float
) -> BoxSpec:
  cu, cv, cz = box.center
  hu, hv, hz = box.half_size
  di, dj = direction
  i, j = slot
  x0, y0 = i * tile_size, j * tile_size
  if (di, dj) == (1, 0):
    c = (x0 + cu, y0 + cv, cz)
    h = (hu, hv, hz)
  elif (di, dj) == (-1, 0):
    c = (x0 + tile_size - cu, y0 + cv, cz)
    h = (hu, hv, hz)
  elif (di, dj) == (0, 1):
    c = (x0 + tile_size - cv, y0 + cu, cz)
    h = (hv, hu, hz)
  else:
    c = (x0 + cv, y0 + tile_size - cu, cz)
    h = (hv, hu, hz)
  return BoxSpec(center=c, half_size=h, collidable=box.collidable, tag=box.tag)


@dataclass(kw_only=True)
class CourseTerrainCfg(SubTerrainCfg):
  """Composite course cell (see module docstring)."""

  allowed_kinds: tuple[TileKind, ...] = (TileKind.FLAT, TileKind.ROUGH)
  kind_weights: dict[TileKind, float] | None = None
  filler_kinds: tuple[TileKind, ...] = (TileKind.FLAT, TileKind.ROUGH)
  tile_size: float = 2.0
  resolution: float = 0.05
  path_length_range: tuple[int, int] = (5, 9)
  floor_depth: float = 1.0
  envelope: CapabilityEnvelope = field(default_factory=CapabilityEnvelope)
  start_height: float = 0.0
  roughness_amplitude: tuple[float, float] = (0.0, 0.02)
  """Global micro-roughness amplitude range (meters), linear in difficulty."""
  hurdles_collidable: bool = True
  """False = stage A of the soft-to-hard hurdle curriculum (penetrable bars)."""
  max_boxes: int = 8
  filler_height_step: float = 0.1
  filler_max_step: float = 0.25
  """Max height difference between neighbouring slots off the path. 0 keeps every
  filler tile at the path height (use for the flat family)."""
  wall_probability: float = 0.0
  """Share of cells whose path is a walled corridor (Zhuang et al. 2023 tracks).
  The rest stay open so policies also see unwalled obstacles."""
  wall_height: float = 0.8
  """Above the higher end of each tile; must exceed what the robot can climb."""
  wall_thickness: float = 0.15
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = field(
    default_factory=default_patch_sampling
  )

  def _grid(self) -> tuple[int, int]:
    return (
      max(1, int(round(self.size[0] / self.tile_size))),
      max(1, int(round(self.size[1] / self.tile_size))),
    )

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    grid = self._grid()
    n = int(round(self.tile_size / self.resolution))
    ny, nx = grid[1] * n, grid[0] * n

    path = sample_path(
      difficulty,
      self.allowed_kinds,
      rng,
      self.envelope,
      self.tile_size,
      grid=grid,
      length_range=self.path_length_range,
      kind_weights=self.kind_weights,
      start_height=self.start_height,
    )
    ok, why = check_feasible(path, self.envelope)
    if not ok:  # The grammar only emits feasible paths; fail loudly otherwise.
      raise RuntimeError(f"Infeasible path sampled: {why}")

    levels = np.arange(
      self.envelope.min_height, self.envelope.max_height + 1e-9, self.filler_height_step
    )
    fill = fill_free_slots(
      path,
      grid,
      rng,
      levels,
      filler_kinds=self.filler_kinds,
      max_step=self.filler_max_step,
    )

    cell = np.zeros((ny, nx), dtype=np.float64)
    skills = np.full((grid[1], grid[0]), -1, dtype=int)  # [iy, ix] slots.
    boxes: list[BoxSpec] = []

    for p in path.placements:
      h, tile_boxes = rasterize_tile(
        p.tile, self.tile_size, self.resolution, rng, self.floor_depth
      )
      _place(cell, h, p.slot, p.direction, n)
      skills[p.slot[1], p.slot[0]] = int(TILE_SKILL[p.tile.kind])
      boxes += [
        _box_to_cell(b, p.slot, p.direction, self.tile_size) for b in tile_boxes
      ]

    for slot, a in fill.items():
      tile = sample_tile(
        a.kind, difficulty, rng, self.envelope, self.tile_size, a.base_h
      )
      assert tile is not None  # Filler kinds do not change height.
      h, _ = rasterize_tile(
        tile, self.tile_size, self.resolution, rng, self.floor_depth
      )
      direction = ((1, 0), (0, 1))[int(rng.integers(2))]
      _place(cell, h, slot, direction, n)
      skills[slot[1], slot[0]] = int(TILE_SKILL[tile.kind])

    # Tiles that need exact geometry are kept free of micro-roughness.
    exact = np.zeros((ny, nx), dtype=bool)
    for p in path.placements:
      if p.tile.kind in (TileKind.STONES, TileKind.HURDLE, TileKind.GAP):
        i, j = p.slot
        exact[j * n : (j + 1) * n, i * n : (i + 1) * n] = True
    walled = bool(rng.random() < self.wall_probability)
    if walled:
      thickness_px = max(1, int(round(self.wall_thickness / self.resolution)))
      exact |= add_corridor_walls(cell, path, n, thickness_px, self.wall_height)
    amp = self.roughness_amplitude[0] + difficulty * (
      self.roughness_amplitude[1] - self.roughness_amplitude[0]
    )
    if amp > 0:
      rough = band_limited_roughness((ny, nx), self.resolution, rng, amp)
      cell += rough * ~exact

    hf_geom, _ = add_heightfield(spec, body, cell, self.size)
    collidable_tags = frozenset({"hurdle"}) if self.hurdles_collidable else frozenset()
    box_geoms, box_arr, hurdle_mask = add_boxes(
      body, boxes, self.max_boxes, collidable_tags
    )

    first = path.placements[0]
    spawn_c = np.array(
      [
        (first.slot[0] + 0.5) * self.tile_size,
        (first.slot[1] + 0.5) * self.tile_size,
        first.tile.entry_h,
      ]
    )
    num = (self.flat_patch_sampling or default_patch_sampling())["spawn"].num_patches

    assert hf_geom.geom is not None
    write_cell_meta(
      spec,
      hf_geom.geom,
      CellMeta(
        difficulty=float(difficulty),
        slot_size=(self.tile_size, self.tile_size),
        skills=skills.tolist(),
        waypoints=route_waypoints(path, self.tile_size).tolist(),
        ordered=True,
        boxes=box_arr.tolist(),
        hurdle_mask=hurdle_mask.tolist(),
        walled=walled,
        path=[
          (p.slot, p.tile.kind.value, dict(p.tile.params)) for p in path.placements
        ],
      ),
    )
    return TerrainOutput(
      origin=spawn_c,
      geometries=[hf_geom] + box_geoms,
      flat_patches={"spawn": jittered_patches(spawn_c, 0.3, num, rng)},
    )


def make_tile(kind: TileKind, **params: float) -> Tile:
  """Convenience for tests."""
  return Tile(kind=kind, params=dict(params))
