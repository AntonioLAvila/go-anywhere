"""Per-family terrain generators and the full distribution.

Every cell built here writes its ``CellMeta`` into the terrain spec, so the env
side can recover per-cell skill labels, boxes and routes with ``read_cells``.
"""

from __future__ import annotations

from typing import Callable

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

from go_anywhere.terrains.course import CourseTerrainCfg
from go_anywhere.terrains.stones_everywhere import StonesEverywhereTerrainCfg
from go_anywhere.terrains.tiles import CapabilityEnvelope, Skill, TileKind

CELL_SIZE = (8.0, 8.0)
NUM_ROWS = 10
BORDER_WIDTH = 20.0

FAMILIES: tuple[str, ...] = (
  "flat",
  "rough",
  "stairs",
  "steps",
  "gaps",
  "climb",
  "descend",
  "hurdles",
  "drops",
)

EXPERT_FOR_SKILL: dict[Skill, str] = {
  Skill.FLAT: "flat",
  Skill.ROUGH: "rough",
  Skill.STAIRS: "stairs",
  Skill.STEPS: "steps",
  Skill.GAPS: "gaps",
  Skill.CLIMB: "climb",
  Skill.DESCEND: "descend",
  Skill.HURDLES: "hurdles",
  Skill.DROPS: "drops",
}

_K = TileKind

# Share of walled-corridor cells for families whose obstacles could be walked around.
WALL_PROBABILITY = 0.75


def _family_cfg(family: str, hurdles_collidable: bool) -> SubTerrainCfg:
  """One cell cfg for a single-skill expert (that skill + flat connectors)."""
  if family == "flat":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT,),
      filler_kinds=(_K.FLAT,),
      roughness_amplitude=(0.0, 0.0),
      filler_max_step=0.0,
    )
  if family == "rough":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT, _K.ROUGH, _K.SLOPE),
      kind_weights={_K.ROUGH: 2.0, _K.SLOPE: 1.0},
      roughness_amplitude=(0.0, 0.04),
      filler_max_step=0.1,
    )
  if family == "stairs":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT, _K.STAIRS_UP, _K.STAIRS_DOWN),
      wall_probability=WALL_PROBABILITY,
    )
  if family == "steps":
    return StonesEverywhereTerrainCfg()
  if family == "gaps":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT, _K.GAP), wall_probability=WALL_PROBABILITY
    )
  if family == "climb":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT, _K.LEDGE_UP, _K.LEDGE_DOWN),
      kind_weights={_K.LEDGE_UP: 3.0, _K.LEDGE_DOWN: 1.0},
      wall_probability=WALL_PROBABILITY,
    )
  if family == "descend":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT, _K.LEDGE_DOWN, _K.STAIRS_DOWN),
      kind_weights={_K.LEDGE_DOWN: 3.0, _K.STAIRS_DOWN: 1.0},
      start_height=CapabilityEnvelope().max_height,
      wall_probability=WALL_PROBABILITY,
    )
  if family == "hurdles":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT, _K.HURDLE),
      hurdles_collidable=hurdles_collidable,
      wall_probability=WALL_PROBABILITY,
    )
  if family == "drops":
    return CourseTerrainCfg(
      allowed_kinds=(_K.FLAT, _K.ROUGH), roughness_amplitude=(0.0, 0.03)
    )
  raise ValueError(f"Unknown family {family}")


def _variants(
  name: str, make: Callable[[], SubTerrainCfg], variants: int, proportion: float = 1.0
) -> dict[str, SubTerrainCfg]:
  """``variants`` independent layouts of one sub-terrain.

  mjlab's curriculum mode builds one column per sub-terrain, i.e. one layout
  per difficulty level; registering a sub-terrain several times gives several.
  The spawn proportion is split so the group keeps its total weight.
  """
  out: dict[str, SubTerrainCfg] = {}
  for i in range(variants):
    cfg = make()
    cfg.proportion = proportion / variants
    out[f"{name}/{i}"] = cfg
  return out


def sub_terrain_group(name: str) -> str:
  """Group of a sub-terrain name (``gaps/3`` -> ``gaps``), for per-skill statistics."""
  return name.rsplit("/", 1)[0]


def family_sub_terrains(
  family: str, variants: int = 8, hurdles_collidable: bool = True
) -> dict[str, SubTerrainCfg]:
  return _variants(family, lambda: _family_cfg(family, hurdles_collidable), variants)


def family_terrain_cfg(
  family: str, variants: int = 8, hurdles_collidable: bool = True
) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=CELL_SIZE,
    border_width=BORDER_WIDTH,
    num_rows=NUM_ROWS,
    curriculum=True,
    sub_terrains=family_sub_terrains(family, variants, hurdles_collidable),
    add_lights=True,
  )


def full_distribution_cfg(hurdles_collidable: bool = True) -> TerrainGeneratorCfg:
  """Union of single-skill cells, mixed courses and natural cells."""
  subs: dict[str, SubTerrainCfg] = {}
  for fam in FAMILIES:
    subs.update(family_sub_terrains(fam, 2, hurdles_collidable))
  subs.update(
    _variants(
      "mixed",
      lambda: CourseTerrainCfg(
        allowed_kinds=tuple(_K),
        hurdles_collidable=hurdles_collidable,
        roughness_amplitude=(0.0, 0.03),
        wall_probability=0.5,
      ),
      variants=6,
      proportion=3.0,
    )
  )
  subs.update(
    _variants(
      "natural",
      lambda: CourseTerrainCfg(
        allowed_kinds=(_K.FLAT, _K.ROUGH, _K.SLOPE),
        kind_weights={_K.ROUGH: 3.0, _K.SLOPE: 2.0},
        filler_kinds=(_K.ROUGH,),
        roughness_amplitude=(0.02, 0.08),
      ),
      variants=3,
      proportion=1.5,
    )
  )
  return TerrainGeneratorCfg(
    size=CELL_SIZE,
    border_width=BORDER_WIDTH,
    num_rows=NUM_ROWS,
    curriculum=True,
    sub_terrains=subs,
    add_lights=True,
  )
