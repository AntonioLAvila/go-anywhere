"""Heightfield + box emission shared by all go-anywhere cells.

MuJoCo heightfield conventions (``mjGEOM_HFIELD``): ``nrow`` samples run along
the geom's local Y axis and ``ncol`` along X; ``userdata`` is row-major with
values in [0, 1] mapped to ``[0, size[2]]``. ``size[3]`` is the thickness of
the base slab below z=0. We therefore keep height arrays indexed ``[iy, ix]``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainGeometry

from go_anywhere.terrains.tiles import BoxSpec

CELL_PREFIX = "ga_cell_"


@dataclass
class CellMeta:
  """What a cell exposes to the env side. Stored as JSON in a MuJoCo ``<text>``
  element of the terrain spec, so it travels with the model: no shared Python
  state and no dependence on the order the generator builds cells in.

  Coordinates are in the cell's corner frame (x, y in ``[0, size]``).
  """

  difficulty: float
  slot_size: tuple[float, float]
  """Size (x, y) of one label slot in meters."""
  skills: list[list[int]]
  """Skill id per slot, indexed ``[iy][ix]`` (-1 for no skill)."""
  waypoints: list[list[float]]
  """``[W, 3]`` goal positions: the route in travel order if ``ordered``, else
  interchangeable goal candidates."""
  ordered: bool
  boxes: list[list[float]] = field(default_factory=list)
  """``[K, 6]`` boxes (x, y, z, hx, hy, hz); parked boxes sit far below the floor."""
  hurdle_mask: list[bool] = field(default_factory=list)
  """``[K]`` which boxes are hurdle bars (penetration cost targets)."""
  walled: bool = False
  path: list = field(default_factory=list)
  """Path tiles as (slot, kind, params), for previews and analysis."""


def write_cell_meta(spec: mujoco.MjSpec, geom: mujoco.MjsGeom, meta: CellMeta) -> None:
  """Attach ``meta`` to the cell whose heightfield geom is ``geom``."""
  name = CELL_PREFIX + uuid.uuid4().hex
  geom.name = name
  spec.add_text(name=name, data=json.dumps(asdict(meta)))


def read_cells(
  spec: mujoco.MjSpec, cell_size: tuple[float, float], rows: int, cols: int
) -> dict[tuple[int, int], CellMeta]:
  """Recover every cell's metadata from a generated terrain spec.

  The generator moves each cell's geoms to their grid position, so (row, col)
  follows from the heightfield geom's position (grid centred on the world origin,
  rows along x, columns along y; see ``add_heightfield`` for the half-size offset).
  """
  out: dict[tuple[int, int], CellMeta] = {}
  for text in spec.texts:
    if not text.name.startswith(CELL_PREFIX):
      continue
    pos = spec.geom(text.name).pos
    row = int(round((pos[0] - cell_size[0] / 2) / cell_size[0] + rows / 2))
    col = int(round((pos[1] - cell_size[1] / 2) / cell_size[1] + cols / 2))
    data = json.loads(text.data)
    data["slot_size"] = tuple(data["slot_size"])
    out[(row, col)] = CellMeta(**data)
  expected = {(r, c) for r in range(rows) for c in range(cols)}
  if set(out) != expected:
    raise RuntimeError(
      f"Terrain spec holds metadata for {len(out)} cells, expected {rows}x{cols}; "
      "every go-anywhere sub-terrain must call write_cell_meta."
    )
  return out


FLOOR_Z = -2.5
"""Bottom of every cell's heightfield base. A MuJoCo heightfield is solid (for
collision and rendering) only between its surface and its own base, so cells
whose lowest point is high would float above lower neighbours with an open
hollow underneath. Extending every base to one common floor closes those sides."""


def add_heightfield(
  spec: mujoco.MjSpec,
  body: mujoco.MjsBody,
  heights: np.ndarray,
  size: tuple[float, float],
  min_thickness: float = 0.5,
  rgba: tuple[float, float, float, float] | None = None,
  floor_z: float = FLOOR_Z,
) -> tuple[TerrainGeometry, float]:
  """Add one heightfield geom covering ``size`` at the cell corner frame.

  Args:
    heights: ``[ny, nx]`` absolute heights in meters.
    floor_z: Bottom of the base slab (shared by all cells, see ``FLOOR_Z``).
  Returns:
    The terrain geometry and the z offset (min height) of the field.
  """
  z_min = float(heights.min())
  z_max = float(heights.max())
  z_range = max(z_max - z_min, 1e-3)
  normalized = ((heights - z_min) / z_range).astype(np.float32)
  thickness = max(min_thickness, z_min - floor_z)
  uid = uuid.uuid4().hex
  hfield = spec.add_hfield(
    name=f"ga_hfield_{uid}",
    size=[size[0] / 2, size[1] / 2, z_range, thickness],
    nrow=int(heights.shape[0]),
    ncol=int(heights.shape[1]),
    userdata=normalized.flatten().tolist(),
  )
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_HFIELD,
    hfieldname=hfield.name,
    pos=[size[0] / 2, size[1] / 2, z_min],
  )
  if rgba is not None:
    geom.rgba[:] = rgba
  return TerrainGeometry(geom=geom, hfield=hfield), z_min


PARKED_Z = -50.0


def add_boxes(
  body: mujoco.MjsBody,
  boxes: list[BoxSpec],
  max_boxes: int,
  collidable_tags: frozenset[str] | None = None,
) -> tuple[list[TerrainGeometry], np.ndarray, np.ndarray]:
  """Emit a fixed budget of box geoms (unused ones parked below the floor).

  Returns the geometries, a ``[max_boxes, 6]`` array of (cx, cy, cz, hx, hy, hz)
  and a bool ``[max_boxes]`` mask of hurdle bars.
  """
  arr = np.zeros((max_boxes, 6), dtype=np.float64)
  arr[:, 2] = PARKED_Z
  arr[:, 3:] = 0.01
  hurdle = np.zeros(max_boxes, dtype=bool)
  geoms: list[TerrainGeometry] = []
  if len(boxes) > max_boxes:
    raise ValueError(f"Cell needs {len(boxes)} boxes but max_boxes is {max_boxes}.")
  for i in range(max_boxes):
    if i < len(boxes):
      b = boxes[i]
      center, half = b.center, b.half_size
      collidable = b.collidable
      if collidable_tags is not None and b.tag:
        collidable = collidable and (b.tag in collidable_tags)
      arr[i] = (*center, *half)
      hurdle[i] = b.tag == "hurdle"
      rgba = (0.85, 0.35, 0.1, 1.0) if collidable else (0.85, 0.35, 0.1, 0.35)
    else:
      center, half, collidable = (0.0, 0.0, PARKED_Z), (0.01, 0.01, 0.01), False
      rgba = (0.0, 0.0, 0.0, 0.0)
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=list(half),
      pos=list(center),
      contype=1 if collidable else 0,
      conaffinity=1 if collidable else 0,
      rgba=rgba,
    )
    geoms.append(TerrainGeometry(geom=geom, color=rgba))
  return geoms, arr, hurdle


def jittered_patches(
  center: np.ndarray, radius: float, num: int, rng: np.random.Generator
) -> np.ndarray:
  """``[num, 3]`` points uniformly inside a disc around ``center`` (z kept)."""
  r = radius * np.sqrt(rng.random(num))
  th = rng.uniform(0, 2 * np.pi, num)
  out = np.repeat(center[None, :], num, axis=0)
  out[:, 0] += r * np.cos(th)
  out[:, 1] += r * np.sin(th)
  return out
