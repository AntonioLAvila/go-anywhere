import copy

import mujoco
import numpy as np
import pytest
from mjlab.terrains import TerrainEntity, TerrainEntityCfg

from go_anywhere.terrains import (
  FAMILIES,
  CapabilityEnvelope,
  CourseTerrainCfg,
  Skill,
  StonesEverywhereTerrainCfg,
  TileKind,
  check_feasible,
  read_cells,
  sample_path,
)
from go_anywhere.terrains.course import add_corridor_walls, route_waypoints
from go_anywhere.terrains.presets import (
  _family_cfg,
  family_terrain_cfg,
  full_distribution_cfg,
)
from go_anywhere.terrains.raster import CELL_PREFIX, add_heightfield
from go_anywhere.terrains.stones_everywhere import sparsity, stones_params
from go_anywhere.terrains.synthesis import fill_free_slots
from go_anywhere.terrains.tiles import (
  height_delta,
  rasterize_tile,
  sample_tile,
)

ALL_KINDS = tuple(TileKind)


def test_stones_everywhere_matches_zhang_table():
  for level, (stone, gap, shift, height, sp) in {
    0: (0.92, 0.08, 0.036, 0.01, 15.4),
    5: (0.52, 0.18, 0.082, 0.06, 44.8),
    9: (0.20, 0.26, 0.118, 0.10, 81.1),
  }.items():
    p = stones_params(level / 9)
    assert p["stone"] == pytest.approx(stone, abs=1e-6)
    assert p["gap"] == pytest.approx(gap, abs=1e-6)
    assert p["shift"] == pytest.approx(shift, abs=1e-3)
    assert p["height"] == pytest.approx(height, abs=1e-6)
    assert 100 * sparsity(p) == pytest.approx(sp, abs=0.2)


@pytest.mark.parametrize("seed", range(40))
def test_sample_path_is_feasible(seed):
  rng = np.random.default_rng(seed)
  env = CapabilityEnvelope()
  path = sample_path(rng.random(), ALL_KINDS, rng, env, tile_size=2.0)
  ok, why = check_feasible(path, env)
  assert ok, why
  assert path.placements[0].tile.kind == TileKind.FLAT
  assert path.placements[-1].tile.kind == TileKind.FLAT


def test_fill_never_touches_path_and_is_height_continuous():
  rng = np.random.default_rng(3)
  env = CapabilityEnvelope()
  path = sample_path(0.7, ALL_KINDS, rng, env, tile_size=2.0)
  levels = np.arange(env.min_height, env.max_height + 1e-9, 0.1)
  fill = fill_free_slots(path, (4, 4), rng, levels)
  assert set(fill) & set(path.slots) == set()
  assert len(fill) + len(path.slots) == 16


@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_every_tile_rasterizes(kind, difficulty):
  rng = np.random.default_rng(0)
  tile = sample_tile(
    kind, difficulty, rng, CapabilityEnvelope(), 2.0, entry_h=0.3, shrink=True
  )
  assert tile is not None
  assert tile.exit_h == pytest.approx(0.3 + height_delta(tile))
  h, boxes = rasterize_tile(tile, 2.0, 0.05, rng, 1.0)
  assert h.shape == (40, 40)
  assert np.isfinite(h).all()
  # Edge strips meet the entry / exit heights.
  assert np.allclose(h[0], tile.entry_h)
  assert np.allclose(h[-1], tile.exit_h)
  assert (len(boxes) > 0) == (kind == TileKind.HURDLE)


def _spec_with_terrain_body():
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  return spec


def test_heightfield_orientation_follows_mujoco_rows_along_y():
  spec = _spec_with_terrain_body()
  body = spec.body("terrain")
  heights = np.zeros((20, 40))
  heights[:, 20:] = 0.5  # Step up along +x (columns).
  add_heightfield(spec, body, heights, size=(4.0, 2.0))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  def ground_z(x, y):
    pnt = np.array([x, y, 5.0])
    vec = np.array([0.0, 0.0, -1.0])
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geomid)
    return 5.0 - dist

  assert ground_z(0.5, 1.0) == pytest.approx(0.0, abs=0.03)
  assert ground_z(3.5, 1.0) == pytest.approx(0.5, abs=0.03)


def _cell_meta(spec):
  (text,) = [t for t in spec.texts if t.name.startswith(CELL_PREFIX)]
  return read_cells(spec, (8.0, 8.0), 1, 1)[(0, 0)] if text else None


def _centred(spec):
  """Shift a lone cell's geoms the way the generator would for a 1x1 grid."""
  for geom in spec.body("terrain").geoms:
    geom.pos = np.array(geom.pos) + np.array([-4.0, -4.0, 0.0])


def test_course_cell_compiles_and_writes_meta():
  cfg = CourseTerrainCfg(allowed_kinds=ALL_KINDS, hurdles_collidable=False)
  cfg.size = (8.0, 8.0)
  spec = _spec_with_terrain_body()
  out = cfg.function(0.8, spec, np.random.default_rng(1))
  model = spec.compile()
  assert model.nhfield == 1
  assert model.ngeom == 1 + cfg.max_boxes
  assert set(out.flat_patches) == {"spawn"}
  _centred(spec)
  meta = _cell_meta(spec)
  skills = np.array(meta.skills)
  assert skills.shape == (4, 4)
  assert (skills >= 0).all() and (skills < len(Skill)).all()
  assert np.array(meta.boxes).shape == (cfg.max_boxes, 6)
  assert meta.ordered and 5 <= len(meta.waypoints) <= 9
  # Stage-A hurdle bars do not collide.
  for i in range(1, model.ngeom):
    assert model.geom_contype[i] == 0


def test_stones_cell_compiles():
  cfg = StonesEverywhereTerrainCfg()
  cfg.size = (8.0, 8.0)
  spec = _spec_with_terrain_body()
  out = cfg.function(1.0, spec, np.random.default_rng(0))
  spec.compile()
  assert out.flat_patches["spawn"].shape == (8, 3)
  _centred(spec)
  meta = _cell_meta(spec)
  assert meta.skills == [[int(Skill.STEPS)]]
  assert not meta.ordered and 1 <= len(meta.waypoints) <= cfg.max_goals


@pytest.mark.parametrize("curriculum", [True, False])
def test_cell_meta_survives_the_generator(curriculum):
  """Metadata is recovered by position, in curriculum and in random mode, and
  two terrains built from copies of one cfg share nothing."""
  gen = full_distribution_cfg()
  gen.num_rows, gen.border_width, gen.seed = 3, 2.0, 7
  gen.curriculum = curriculum
  cols = len(gen.sub_terrains) if curriculum else 4
  gen.num_cols = cols
  cells = []
  for cfg in (copy.deepcopy(gen), copy.deepcopy(gen)):
    terrain = TerrainEntity(
      TerrainEntityCfg(terrain_type="generator", terrain_generator=cfg), device="cpu"
    )
    cells.append(read_cells(terrain.spec, cfg.size, 3, cols))
  assert set(cells[0]) == {(r, c) for r in range(3) for c in range(cols)}
  assert cells[0] == cells[1]  # Same seed, independent builds.
  if curriculum:
    for (row, _), meta in cells[0].items():
      assert meta.difficulty == pytest.approx(row / 2)
    stones = list(gen.sub_terrains).index("steps/0")
    assert cells[0][(0, stones)].skills == [[int(Skill.STEPS)]]


SKILL_KINDS = {
  "rough": (TileKind.ROUGH, TileKind.SLOPE),
  "stairs": (TileKind.STAIRS_UP, TileKind.STAIRS_DOWN),
  "gaps": (TileKind.GAP,),
  "climb": (TileKind.LEDGE_UP, TileKind.LEDGE_DOWN),
  "descend": (TileKind.LEDGE_DOWN, TileKind.STAIRS_DOWN),
  "hurdles": (TileKind.HURDLE,),
}


@pytest.mark.parametrize("family", SKILL_KINDS)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_every_family_path_contains_its_skill(family, difficulty):
  cfg = _family_cfg(family, hurdles_collidable=True)
  counts = []
  for seed in range(100):
    path = sample_path(
      difficulty,
      cfg.allowed_kinds,
      np.random.default_rng(seed),
      cfg.envelope,
      cfg.tile_size,
      kind_weights=cfg.kind_weights,
      start_height=cfg.start_height,
    )
    assert check_feasible(path, cfg.envelope)[0]
    counts.append(sum(p.tile.kind in SKILL_KINDS[family] for p in path.placements))
  assert min(counts) >= 1
  assert np.mean(counts) >= 2.0


def test_slope_stays_within_the_envelope():
  env = CapabilityEnvelope()
  for seed in range(20):
    rng = np.random.default_rng(seed)
    tile = sample_tile(TileKind.SLOPE, 1.0, rng, env, 2.0)
    h, _ = rasterize_tile(tile, 2.0, 0.05, rng, 1.0)
    steepest = np.abs(np.diff(h[:, 0])).max() / 0.05
    assert np.degrees(np.arctan(steepest)) <= env.max_slope_deg + 0.5


@pytest.mark.parametrize("seed", range(20))
def test_route_stays_on_the_path_and_walls_leave_it_open(seed):
  rng = np.random.default_rng(seed)
  tile, n = 2.0, 40
  path = sample_path(0.5, ALL_KINDS, rng, CapabilityEnvelope(), tile)
  wp = route_waypoints(path, tile)
  assert len(wp) == len(path.placements)
  slots = set(path.slots)
  mask = add_corridor_walls(np.zeros((4 * n, 4 * n)), path, n, 3, 0.8)
  assert mask.any()
  start = (np.array(path.placements[0].slot) + 0.5) * tile
  for a, b in zip(np.vstack([start, wp[:-1, :2]]), wp[:, :2], strict=True):
    for t in np.linspace(0.0, 1.0, 41):
      x, y = a + t * (b - a)
      # Nudge off tile boundaries toward the segment's interior.
      xm, ym = a + 0.5 * (b - a)
      x, y = x + 1e-6 * np.sign(xm - x), y + 1e-6 * np.sign(ym - y)
      assert (int(x // tile), int(y // tile)) in slots
      assert not mask[int(y / tile * n), int(x / tile * n)]


def test_layout_variants_are_separate_columns():
  gen = family_terrain_cfg("gaps", variants=3)
  assert list(gen.sub_terrains) == ["gaps/0", "gaps/1", "gaps/2"]
  assert sum(c.proportion for c in gen.sub_terrains.values()) == pytest.approx(1.0)
  assert set(FAMILIES) <= {
    n.split("/")[0] for n in full_distribution_cfg().sub_terrains
  }


def test_heightfield_bases_share_a_common_floor():
  """Cells must be solid down to FLOOR_Z so cell-boundary cliffs have closed sides."""
  from go_anywhere.terrains.raster import FLOOR_Z

  gen = family_terrain_cfg("descend")  # Starts high up: previously floated.
  gen.num_rows = 3
  spec = _spec_with_terrain_body()
  for cfg in gen.sub_terrains.values():
    cfg.size = gen.size
    for row in range(gen.num_rows):
      cfg.function(row / 2, spec, np.random.default_rng(row))
  for hf, geom in zip(
    spec.hfields,
    [g for g in spec.body("terrain").geoms if g.type == mujoco.mjtGeom.mjGEOM_HFIELD],
    strict=True,
  ):
    z_min = geom.pos[2]
    assert z_min - hf.size[3] == pytest.approx(FLOOR_Z, abs=1e-6)


def test_flat_family_fillers_are_level():
  gen = family_terrain_cfg("flat")
  cfg = next(iter(gen.sub_terrains.values()))
  cfg.size = gen.size
  spec = _spec_with_terrain_body()
  cfg.function(1.0, spec, np.random.default_rng(0))
  hf = spec.hfields[-1]
  heights = np.array(hf.userdata) * hf.size[2]
  assert heights.max() - heights.min() < 1e-6
