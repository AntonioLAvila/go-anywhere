"""Reference-state bank writer/reader (see ``mdp.events.reset_from_reference_states``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReferenceStateBank:
  qpos: np.ndarray
  """[N, 7 + nj] root pos, root quat (w, x, y, z), joint positions."""
  qvel: np.ndarray
  """[N, 6 + nj] root linear/angular velocity (world), joint velocities."""
  phase: np.ndarray
  """[N] in [0, 1]; 1 = end of the reference motion (closest to the goal)."""

  def save(self, path: str | Path) -> None:
    np.savez_compressed(path, qpos=self.qpos, qvel=self.qvel, phase=self.phase)

  @classmethod
  def load(cls, path: str | Path) -> "ReferenceStateBank":
    d = np.load(path)
    return cls(qpos=d["qpos"], qvel=d["qvel"], phase=d["phase"])

  @classmethod
  def from_trajectory(cls, qpos: np.ndarray, qvel: np.ndarray) -> "ReferenceStateBank":
    n = qpos.shape[0]
    return cls(qpos=qpos, qvel=qvel, phase=np.linspace(0.0, 1.0, n))
