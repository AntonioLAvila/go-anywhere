"""TaskSpec: a terrain + spawn + goal + budget tuple with optional reference states.

Produced by future trajectory-first generators (kinodynamic TO or replayed
policies) and by the regret editor; consumed by ``CourseTerrainCfg`` through a
fixed path (``grammar.Path``) instead of a sampled one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TaskSpec:
  family: str
  difficulty: float
  path: list[dict] = field(default_factory=list)
  """Serialized ``grammar.Path``: [{slot, direction, kind, params}, ...]."""
  spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)
  goal: tuple[float, float, float] = (0.0, 0.0, 0.0)
  time_budget_s: float | None = None
  reference_bank: str | None = None
  tags: tuple[str, ...] = ()

  def dump(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(self), indent=2))

  @classmethod
  def load(cls, path: str | Path) -> "TaskSpec":
    data = json.loads(Path(path).read_text())
    data["spawn"] = tuple(data["spawn"])
    data["goal"] = tuple(data["goal"])
    data["tags"] = tuple(data.get("tags", ()))
    return cls(**data)
