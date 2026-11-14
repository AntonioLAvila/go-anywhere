"""Natural terrain patch library (Phase 2 interface).

Macro structure for "natural" cells should come from a precomputed library of
eroded / DEM-derived height patches (Musgrave 1989, Cordonnier 2016, Zhou
2007) rather than per-cell erosion. This module defines the on-disk format and
loader; building the library is a later phase.

Format: ``library.npz`` with ``patches`` float32 ``[N, H, W]`` in meters
(zero-mean), ``resolution`` scalar meters/pixel, optional ``tags`` str array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PatchLibrary:
  patches: np.ndarray
  resolution: float
  tags: tuple[str, ...] = ()

  @classmethod
  def load(cls, path: str | Path) -> "PatchLibrary":
    data = np.load(path, allow_pickle=False)
    tags = tuple(str(t) for t in data["tags"]) if "tags" in data else ()
    return cls(patches=data["patches"], resolution=float(data["resolution"]), tags=tags)

  def sample(self, rng: np.random.Generator) -> np.ndarray:
    return self.patches[int(rng.integers(len(self.patches)))]
