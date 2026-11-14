"""Model-synthesis style fill of the non-path slots.

A small constraint solver in the spirit of Merrell's model synthesis with
"modifying in blocks": the domain of each free slot is (filler kind, base
height level); path slots are pinned; the hard constraint is height
continuity between 4-neighbours (|dh| <= max_step). Free slots are assigned in
breadth-first order from the pinned slots so every slot is constrained by an
already-assigned neighbour. A contradiction restarts the assignment with a new
draw (a bounded number of times); the final fallback snaps a slot to the
nearest feasible level so the fill always terminates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from go_anywhere.terrains.grammar import Path
from go_anywhere.terrains.tiles import FILLER_KINDS, TileKind

Slot = tuple[int, int]


@dataclass
class FillAssignment:
  kind: TileKind
  base_h: float


def _neighbors(slot: Slot, grid: tuple[int, int]) -> list[Slot]:
  i, j = slot
  out = []
  for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    a, b = i + di, j + dj
    if 0 <= a < grid[0] and 0 <= b < grid[1]:
      out.append((a, b))
  return out


def _bfs_order(
  pinned: dict[Slot, float], grid: tuple[int, int], rng: np.random.Generator
) -> list[Slot]:
  """Free slots ordered by distance from the pinned set (ties shuffled)."""
  seen = set(pinned)
  frontier = deque(pinned.keys())
  order: list[Slot] = []
  while frontier:
    layer: list[Slot] = []
    for _ in range(len(frontier)):
      slot = frontier.popleft()
      for nb in _neighbors(slot, grid):
        if nb not in seen:
          seen.add(nb)
          layer.append(nb)
    rng.shuffle(layer)
    order += layer
    frontier.extend(layer)
  # Slots unreachable from the path (no pinned slots at all) in scan order.
  for i in range(grid[0]):
    for j in range(grid[1]):
      if (i, j) not in seen:
        order.append((i, j))
  return order


def fill_free_slots(
  path: Path,
  grid: tuple[int, int],
  rng: np.random.Generator,
  height_levels: np.ndarray,
  max_step: float = 0.25,
  filler_kinds: tuple[TileKind, ...] = FILLER_KINDS,
  kind_weights: dict[TileKind, float] | None = None,
  max_tries: int = 20,
) -> dict[Slot, FillAssignment]:
  """Assign (kind, base height) to every slot not on the path."""
  pinned: dict[Slot, float] = {}
  for p in path.placements:
    # Use the mean tile height as the slot's representative height.
    pinned[p.slot] = 0.5 * (p.tile.entry_h + p.tile.exit_h)

  kinds = list(filler_kinds)
  w = np.array([(kind_weights or {}).get(k, 1.0) for k in kinds], dtype=float)
  w = w / w.sum()

  def domain(slot: Slot, assigned: dict[Slot, FillAssignment]) -> np.ndarray:
    ok = np.ones_like(height_levels, dtype=bool)
    for nb in _neighbors(slot, grid):
      if nb in pinned:
        hn = pinned[nb]
      elif nb in assigned:
        hn = assigned[nb].base_h
      else:
        continue
      ok &= np.abs(height_levels - hn) <= max_step + 1e-9
    return height_levels[ok]

  order = _bfs_order(pinned, grid, rng)
  assigned: dict[Slot, FillAssignment] = {}
  for _ in range(max_tries):
    assigned = {}
    failed = False
    for slot in order:
      dom = domain(slot, assigned)
      if dom.size == 0:
        failed = True
        break
      kind = kinds[int(rng.choice(len(kinds), p=w))]
      assigned[slot] = FillAssignment(kind=kind, base_h=float(rng.choice(dom)))
    if not failed:
      return assigned
    order = _bfs_order(pinned, grid, rng)

  # Fallback: complete the last attempt by snapping failed slots to the level
  # nearest the mean of their assigned neighbours.
  for slot in order:
    if slot in assigned:
      continue
    hs = []
    for nb in _neighbors(slot, grid):
      if nb in pinned:
        hs.append(pinned[nb])
      elif nb in assigned:
        hs.append(assigned[nb].base_h)
    target = float(np.mean(hs)) if hs else path.start_height
    base = float(height_levels[np.argmin(np.abs(height_levels - target))])
    assigned[slot] = FillAssignment(kind=TileKind.FLAT, base_h=base)
  return assigned
