"""Tile library for the course generator.

A course cell is a grid of square tiles. Each tile has a kind, a skill label,
difficulty-scaled parameters, an entry height and an exit height along its
travel direction. Feasibility is checked against a per-skill capability
envelope, in the spirit of ALLSTEPS (Xie et al. 2020) and Peng et al. 2016.

Local tile frame: ``u`` runs along the travel direction, ``v`` is lateral. A
rasterized tile is an array ``heights[iu, iv]`` in meters plus a list of box
primitives for shapes a heightfield cannot represent (hurdle bars).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class Skill(IntEnum):
  """Skill families. The index is the teacher-routing id."""

  FLAT = 0
  ROUGH = 1
  STAIRS = 2
  STEPS = 3
  GAPS = 4
  CLIMB = 5
  DESCEND = 6
  HURDLES = 7
  DROPS = 8


SKILL_NAMES: tuple[str, ...] = tuple(s.name.lower() for s in Skill)
NUM_SKILLS = len(Skill)


class TileKind(str, Enum):
  FLAT = "flat"
  ROUGH = "rough"
  SLOPE = "slope"
  STAIRS_UP = "stairs_up"
  STAIRS_DOWN = "stairs_down"
  STONES = "stones"
  GAP = "gap"
  LEDGE_UP = "ledge_up"
  LEDGE_DOWN = "ledge_down"
  HURDLE = "hurdle"


TILE_SKILL: dict[TileKind, Skill] = {
  TileKind.FLAT: Skill.FLAT,
  TileKind.ROUGH: Skill.ROUGH,
  TileKind.SLOPE: Skill.ROUGH,
  TileKind.STAIRS_UP: Skill.STAIRS,
  TileKind.STAIRS_DOWN: Skill.STAIRS,
  TileKind.STONES: Skill.STEPS,
  TileKind.GAP: Skill.GAPS,
  TileKind.LEDGE_UP: Skill.CLIMB,
  TileKind.LEDGE_DOWN: Skill.DESCEND,
  TileKind.HURDLE: Skill.HURDLES,
}

# Tiles whose geometry depends on the travel direction. They are only placed
# on straight path segments; turn slots get FLAT/ROUGH.
DIRECTIONAL_KINDS = frozenset(
  {
    TileKind.SLOPE,
    TileKind.STAIRS_UP,
    TileKind.STAIRS_DOWN,
    TileKind.GAP,
    TileKind.LEDGE_UP,
    TileKind.LEDGE_DOWN,
    TileKind.HURDLE,
  }
)

# Tiles that can be used to fill non-path slots.
FILLER_KINDS = (TileKind.FLAT, TileKind.ROUGH, TileKind.STONES)


@dataclass
class CapabilityEnvelope:
  """Upper bounds on obstacle parameters the robot is assumed to handle.

  Initial values come from Go1/A1-class parkour results (Extreme Parkour,
  Robot Parkour Learning, ANYmal Parkour scaled down, Zhang et al. 2024).
  Calibrate from ``scripts/eval.py``.
  """

  max_gap: float = 0.8
  max_climb: float = 0.5
  max_descend: float = 0.8
  max_hurdle: float = 0.3
  max_hurdle_thickness: float = 0.4
  min_stone: float = 0.2
  max_stone_gap: float = 0.26
  max_stair_rise: float = 0.23
  stair_width: float = 0.3
  max_slope_deg: float = 30.0
  max_rough: float = 0.10
  min_height: float = -0.6
  max_height: float = 1.0


@dataclass
class BoxSpec:
  """Axis-aligned box in the local tile frame (u, v, z) with half sizes."""

  center: tuple[float, float, float]
  half_size: tuple[float, float, float]
  collidable: bool = True
  tag: str = ""


@dataclass
class Tile:
  kind: TileKind
  params: dict[str, float] = field(default_factory=dict)
  entry_h: float = 0.0
  exit_h: float = 0.0

  @property
  def skill(self) -> Skill:
    return TILE_SKILL[self.kind]


def _lerp(lo: float, hi: float, t: float) -> float:
  return lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))


# Height-changing kinds and their mirror image, tried when a tile does not fit.
OPPOSITE_KIND: dict[TileKind, TileKind] = {
  TileKind.STAIRS_UP: TileKind.STAIRS_DOWN,
  TileKind.STAIRS_DOWN: TileKind.STAIRS_UP,
  TileKind.LEDGE_UP: TileKind.LEDGE_DOWN,
  TileKind.LEDGE_DOWN: TileKind.LEDGE_UP,
}
MIN_LEDGE = 0.1
MIN_STAIR_RISE = 0.05


def sample_tile(
  kind: TileKind,
  difficulty: float,
  rng: np.random.Generator,
  env: CapabilityEnvelope,
  tile_size: float,
  entry_h: float = 0.0,
  shrink: bool = False,
) -> Tile | None:
  """Sample a tile of ``kind`` entered at ``entry_h``, parameters linear in difficulty.

  Parameters are drawn uniformly from a band ``[p(d) - w, p(d)]`` so that a
  given difficulty level still shows variety (Peng et al. 2016). A tile whose
  exit height would leave ``[env.min_height, env.max_height]`` is returned as
  ``None``, or, with ``shrink``, reduced to the room that is left (fewer steps,
  lower ledge) if that is still a meaningful obstacle.
  """
  d = float(np.clip(difficulty, 0.0, 1.0))
  jitter = 0.2 * rng.random()  # Up to 20% below the nominal value.
  dj = max(0.0, d - jitter)
  room_up = env.max_height - entry_h
  room_down = entry_h - env.min_height
  eps = 1e-9
  p: dict[str, float] = {}
  if kind == TileKind.FLAT:
    pass
  elif kind == TileKind.ROUGH:
    p["amplitude"] = _lerp(0.02, env.max_rough, dj)
    p["cell"] = float(rng.uniform(0.15, 0.4))  # Peng's hard 20–40 cm band.
  elif kind == TileKind.SLOPE:
    # The ramp spans the tile between its two flat edge strips.
    angle = np.radians(_lerp(5.0, env.max_slope_deg, dj))
    delta = float((tile_size - 2 * EDGE_STRIP) * np.tan(angle))
    up = rng.random() < 0.5
    room, other = (room_up, room_down) if up else (room_down, room_up)
    if delta > room + eps:  # Go the other way if it has more room, then clamp.
      if other > room:
        up, room = not up, other
      delta = min(delta, room)
    p["delta"] = delta if up else -delta
  elif kind in (TileKind.STAIRS_UP, TileKind.STAIRS_DOWN):
    p["rise"] = _lerp(MIN_STAIR_RISE, env.max_stair_rise, dj)
    p["width"] = env.stair_width
    n = max(1, int(np.floor((tile_size - 2 * EDGE_STRIP) / env.stair_width)))
    room = room_up if kind == TileKind.STAIRS_UP else room_down
    fit = int(np.floor((room + eps) / p["rise"]))
    if fit < n:
      if not shrink or fit < 1:
        return None
      n = fit
    p["n_steps"] = float(n)
  elif kind == TileKind.STONES:
    # Zhang et al. 2024 Table I, linear in level.
    p["stone"] = _lerp(0.92, env.min_stone, dj)
    p["gap"] = _lerp(0.08, env.max_stone_gap, dj)
    p["shift"] = _lerp(0.036, 0.118, dj)
    p["height"] = _lerp(0.01, 0.10, dj)
  elif kind == TileKind.GAP:
    p["width"] = _lerp(0.1, env.max_gap, dj)
  elif kind in (TileKind.LEDGE_UP, TileKind.LEDGE_DOWN):
    up = kind == TileKind.LEDGE_UP
    p["height"] = _lerp(MIN_LEDGE, env.max_climb if up else env.max_descend, dj)
    room = room_up if up else room_down
    if p["height"] > room + eps:
      if not shrink or room < MIN_LEDGE:
        return None
      p["height"] = room
  elif kind == TileKind.HURDLE:
    p["height"] = _lerp(0.05, env.max_hurdle, dj)
    p["thickness"] = float(rng.uniform(0.05, min(0.2, env.max_hurdle_thickness)))
  else:
    raise ValueError(f"Unknown tile kind {kind}")
  tile = Tile(kind=kind, params=p, entry_h=entry_h)
  tile.exit_h = entry_h + height_delta(tile)
  return tile


def height_delta(tile: Tile) -> float:
  """Exit height minus entry height for a tile."""
  k, p = tile.kind, tile.params
  if k == TileKind.SLOPE:
    return p["delta"]
  if k == TileKind.STAIRS_UP:
    return p["rise"] * p["n_steps"]
  if k == TileKind.STAIRS_DOWN:
    return -p["rise"] * p["n_steps"]
  if k == TileKind.LEDGE_UP:
    return p["height"]
  if k == TileKind.LEDGE_DOWN:
    return -p["height"]
  return 0.0


def within_envelope(tile: Tile, env: CapabilityEnvelope) -> tuple[bool, str]:
  k, p = tile.kind, tile.params
  eps = 1e-6
  if k == TileKind.GAP and p["width"] > env.max_gap + eps:
    return False, "gap too wide"
  if k == TileKind.LEDGE_UP and p["height"] > env.max_climb + eps:
    return False, "ledge too high"
  if k == TileKind.LEDGE_DOWN and p["height"] > env.max_descend + eps:
    return False, "drop too high"
  if k == TileKind.HURDLE and (
    p["height"] > env.max_hurdle + eps
    or p["thickness"] > env.max_hurdle_thickness + eps
  ):
    return False, "hurdle out of envelope"
  if k == TileKind.STONES and (
    p["stone"] < env.min_stone - eps or p["gap"] > env.max_stone_gap + eps
  ):
    return False, "stones too sparse"
  if (
    k in (TileKind.STAIRS_UP, TileKind.STAIRS_DOWN)
    and p["rise"] > env.max_stair_rise + eps
  ):
    return False, "stair rise too high"
  if k == TileKind.ROUGH and p["amplitude"] > env.max_rough + eps:
    return False, "roughness too large"
  return True, ""


# Edge strips (meters) at the entry and exit of every tile are kept flat at the
# entry/exit height so neighbouring tiles always meet.
EDGE_STRIP = 0.25


def rasterize_tile(
  tile: Tile,
  tile_size: float,
  resolution: float,
  rng: np.random.Generator,
  floor_depth: float,
) -> tuple[np.ndarray, list[BoxSpec]]:
  """Rasterize a tile in its local frame.

  Returns ``heights[iu, iv]`` (meters, absolute) and box primitives.
  """
  n = int(round(tile_size / resolution))
  u = (np.arange(n) + 0.5) * resolution  # Along travel.
  h = np.full((n, n), tile.entry_h, dtype=np.float64)
  boxes: list[BoxSpec] = []
  k, p = tile.kind, tile.params
  e0, e1 = tile.entry_h, tile.exit_h
  pit = e0 - floor_depth

  if k == TileKind.FLAT:
    pass

  elif k == TileKind.ROUGH:
    cell = max(resolution, p["cell"])
    step = max(1, int(round(cell / resolution)))
    coarse = rng.uniform(
      -p["amplitude"], p["amplitude"], size=(n // step + 2, n // step + 2)
    )
    h += np.kron(coarse, np.ones((step, step)))[:n, :n]

  elif k == TileKind.SLOPE:
    t = np.clip((u - EDGE_STRIP) / max(1e-6, tile_size - 2 * EDGE_STRIP), 0.0, 1.0)
    h[:] = (e0 + (e1 - e0) * t)[:, None]

  elif k in (TileKind.STAIRS_UP, TileKind.STAIRS_DOWN):
    n_steps = int(p["n_steps"])
    rise = p["rise"] * (1.0 if k == TileKind.STAIRS_UP else -1.0)
    start = EDGE_STRIP
    for i in range(n_steps):
      lo = start + i * p["width"]
      h[u >= lo] = e0 + (i + 1) * rise
    h[u >= start + n_steps * p["width"]] = e1

  elif k == TileKind.STONES:
    # Stones-Everywhere inside the tile, over a pit; flat strips at both ends.
    h[:] = pit
    pitch = p["stone"] + p["gap"]
    n_u = max(1, int(np.floor((tile_size - 2 * EDGE_STRIP) / pitch)))
    n_v = max(1, int(np.floor(tile_size / pitch)))
    off_u = EDGE_STRIP + 0.5 * (tile_size - 2 * EDGE_STRIP - n_u * pitch) + pitch / 2
    off_v = 0.5 * (tile_size - n_v * pitch) + pitch / 2
    for i in range(n_u):
      for j in range(n_v):
        cu = off_u + i * pitch + rng.uniform(-p["shift"], p["shift"])
        cv = off_v + j * pitch + rng.uniform(-p["shift"], p["shift"])
        z = e0 + rng.uniform(-p["height"], p["height"])
        half = p["stone"] / 2
        iu0, iu1 = int((cu - half) / resolution), int(np.ceil((cu + half) / resolution))
        iv0, iv1 = int((cv - half) / resolution), int(np.ceil((cv + half) / resolution))
        h[max(0, iu0) : min(n, iu1), max(0, iv0) : min(n, iv1)] = z

  elif k == TileKind.GAP:
    w = p["width"]
    c = tile_size / 2
    mask = (u >= c - w / 2) & (u <= c + w / 2)
    h[mask] = pit

  elif k == TileKind.LEDGE_UP or k == TileKind.LEDGE_DOWN:
    h[u >= tile_size / 2] = e1

  elif k == TileKind.HURDLE:
    t = p["thickness"]
    hb = p["height"]
    boxes.append(
      BoxSpec(
        center=(tile_size / 2, tile_size / 2, e0 + hb / 2),
        half_size=(t / 2, tile_size / 2, hb / 2),
        tag="hurdle",
      )
    )

  else:
    raise ValueError(f"Unknown tile kind {k}")

  # Enforce flat edge strips so adjacent tiles meet exactly.
  h[u < EDGE_STRIP] = e0
  h[u > tile_size - EDGE_STRIP] = e1
  return h, boxes
