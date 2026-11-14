"""Path-first course grammar.

``sample_path`` lays a self-avoiding walk of tile slots across the cell grid,
assigns a tile kind to every slot (flat at both ends, directional obstacles
only on straight segments) and threads entry/exit heights through the path.
Every transition is checked against the capability envelope, so a sampled
path is feasible by construction. ``check_feasible`` is the exact checker the
regret editor reuses (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go_anywhere.terrains.tiles import (
  DIRECTIONAL_KINDS,
  OPPOSITE_KIND,
  CapabilityEnvelope,
  Tile,
  TileKind,
  height_delta,
  sample_tile,
  within_envelope,
)

Dir = tuple[int, int]
DIRS: tuple[Dir, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class TilePlacement:
  slot: tuple[int, int]
  direction: Dir
  """Orientation of the tile's ``u`` axis (the travel direction through it)."""
  tile: Tile
  entry: Dir | None = None
  """Step from the previous path slot into this one (None for the first tile)."""
  exit: Dir | None = None
  """Step from this slot to the next one (None for the last tile)."""


@dataclass
class Path:
  placements: list[TilePlacement] = field(default_factory=list)
  start_height: float = 0.0

  @property
  def slots(self) -> list[tuple[int, int]]:
    return [p.slot for p in self.placements]


def _self_avoiding_walk(
  grid: tuple[int, int], length: int, rng: np.random.Generator, max_tries: int = 200
) -> list[tuple[int, int]] | None:
  ni, nj = grid
  for _ in range(max_tries):
    start = (int(rng.integers(ni)), int(rng.integers(nj)))
    walk = [start]
    visited = {start}
    while len(walk) < length:
      i, j = walk[-1]
      options = [
        (i + di, j + dj)
        for di, dj in DIRS
        if 0 <= i + di < ni and 0 <= j + dj < nj and (i + di, j + dj) not in visited
      ]
      if not options:
        break
      # Prefer continuing straight (longer directional segments).
      if len(walk) >= 2:
        pi, pj = walk[-2]
        straight = (2 * i - pi, 2 * j - pj)
        if straight in options and rng.random() < 0.6:
          nxt = straight
        else:
          nxt = options[int(rng.integers(len(options)))]
      else:
        nxt = options[int(rng.integers(len(options)))]
      walk.append(nxt)
      visited.add(nxt)
    if len(walk) == length:
      return walk
  return None


def _sample_walk_tiles(
  walk: list[tuple[int, int]],
  difficulty: float,
  allowed_kinds: tuple[TileKind, ...],
  kinds: list[TileKind],
  weights: np.ndarray,
  rng: np.random.Generator,
  envelope: CapabilityEnvelope,
  tile_size: float,
  start_height: float,
) -> Path:
  path = Path(start_height=start_height)
  h = start_height
  for idx, slot in enumerate(walk):
    entry = (slot[0] - walk[idx - 1][0], slot[1] - walk[idx - 1][1]) if idx else None
    last = idx == len(walk) - 1
    exit_ = None if last else (walk[idx + 1][0] - slot[0], walk[idx + 1][1] - slot[1])
    is_turn = entry is not None and exit_ is not None and entry != exit_

    tile: Tile | None = None
    if idx > 0 and not last:
      kind = kinds[int(rng.choice(len(kinds), p=weights))]
      # The drawn kind, then its mirror image; first at full size, then shrunk.
      options = [kind]
      if OPPOSITE_KIND.get(kind) in allowed_kinds:
        options.append(OPPOSITE_KIND[kind])
      for shrink in (False, True):
        for k in options:
          if tile is None and not (is_turn and k in DIRECTIONAL_KINDS):
            cand = sample_tile(k, difficulty, rng, envelope, tile_size, h, shrink)
            if cand is not None and within_envelope(cand, envelope)[0]:
              tile = cand
    if tile is None:  # Path ends, and slots where nothing allowed fits.
      filler = TileKind.ROUGH if TileKind.ROUGH in allowed_kinds else TileKind.FLAT
      if idx == 0 or last:
        filler = TileKind.FLAT
      tile = sample_tile(filler, difficulty, rng, envelope, tile_size, h)
      assert tile is not None
    h = tile.exit_h
    # ``u`` runs along the travel direction; turn slots hold non-directional tiles.
    direction = exit_ or entry or (1, 0)
    path.placements.append(
      TilePlacement(slot=slot, direction=direction, tile=tile, entry=entry, exit=exit_)
    )
  return path


def sample_path(
  difficulty: float,
  allowed_kinds: tuple[TileKind, ...],
  rng: np.random.Generator,
  envelope: CapabilityEnvelope,
  tile_size: float,
  grid: tuple[int, int] = (4, 4),
  length_range: tuple[int, int] = (5, 9),
  kind_weights: dict[TileKind, float] | None = None,
  start_height: float = 0.0,
  min_skill_tiles: int = 1,
  max_tries: int = 100,
) -> Path:
  """Sample a feasible skill-labelled path through the slot grid.

  The path holds at least ``min_skill_tiles`` tiles of the non-flat allowed
  kinds, so a cell never silently degenerates into a flat walk.
  """
  kinds = [k for k in allowed_kinds if k != TileKind.FLAT] or [TileKind.FLAT]
  weights = np.array([(kind_weights or {}).get(k, 1.0) for k in kinds], dtype=float)
  weights = weights / weights.sum()
  required = 0 if kinds == [TileKind.FLAT] else min_skill_tiles

  for _ in range(max_tries):
    length = int(rng.integers(length_range[0], length_range[1] + 1))
    length = min(length, grid[0] * grid[1])
    walk = _self_avoiding_walk(grid, length, rng)
    if walk is None:
      continue
    path = _sample_walk_tiles(
      walk,
      difficulty,
      allowed_kinds,
      kinds,
      weights,
      rng,
      envelope,
      tile_size,
      start_height,
    )
    if sum(p.tile.kind in kinds for p in path.placements) >= required:
      return path
  raise RuntimeError(
    f"No path with {required} tiles of {[k.value for k in kinds]} in {max_tries} tries "
    f"(difficulty {difficulty}, start height {start_height})."
  )


def check_feasible(path: Path, envelope: CapabilityEnvelope) -> tuple[bool, str]:
  """Exact feasibility check: envelope bounds, height continuity, height range."""
  if not path.placements:
    return False, "empty path"
  if path.placements[0].tile.kind != TileKind.FLAT:
    return False, "path must start on a flat tile"
  if path.placements[-1].tile.kind != TileKind.FLAT:
    return False, "path must end on a flat tile"
  h = path.start_height
  prev_slot = None
  for p in path.placements:
    if prev_slot is not None:
      di, dj = p.slot[0] - prev_slot[0], p.slot[1] - prev_slot[1]
      if abs(di) + abs(dj) != 1:
        return False, "slots not adjacent"
    ok, why = within_envelope(p.tile, envelope)
    if not ok:
      return False, why
    if abs(p.tile.entry_h - h) > 1e-6:
      return False, "height discontinuity"
    if abs(p.tile.exit_h - (p.tile.entry_h + height_delta(p.tile))) > 1e-6:
      return False, "exit height inconsistent"
    h = p.tile.exit_h
    if not (envelope.min_height - 1e-6 <= h <= envelope.max_height + 1e-6):
      return False, "height out of range"
    prev_slot = p.slot
  return True, ""
