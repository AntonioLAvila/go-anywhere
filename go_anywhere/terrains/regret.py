"""Regret buffer over course parameters (Phase 5 interface).

PLR / ACCEL style: keep the highest-learning-potential path parameterizations,
mutate them locally, and re-verify with ``grammar.check_feasible`` (exact)
rather than relying on regret to filter unsolvable levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go_anywhere.terrains.grammar import Path, check_feasible
from go_anywhere.terrains.tiles import CapabilityEnvelope


@dataclass
class RegretEntry:
  path: Path
  score: float
  staleness: int = 0


@dataclass
class RegretBuffer:
  capacity: int = 256
  entries: list[RegretEntry] = field(default_factory=list)

  def add(self, path: Path, score: float) -> None:
    self.entries.append(RegretEntry(path=path, score=score))
    self.entries.sort(key=lambda e: e.score, reverse=True)
    del self.entries[self.capacity :]

  def sample(self, rng: np.random.Generator, temperature: float = 1.0) -> Path | None:
    if not self.entries:
      return None
    scores = np.array([e.score for e in self.entries], dtype=float)
    p = np.exp((scores - scores.max()) / max(temperature, 1e-6))
    p /= p.sum()
    return self.entries[int(rng.choice(len(self.entries), p=p))].path

  @staticmethod
  def verify(path: Path, envelope: CapabilityEnvelope) -> bool:
    return check_feasible(path, envelope)[0]
