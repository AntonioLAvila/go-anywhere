"""Process-wide visible-context schedule for the LocoFormer-style curriculum.

The env-side curriculum term sets the value; ``LocoTransformer`` reads it on
every forward pass during training. ``None`` means the full window.
"""

from __future__ import annotations

_VISIBLE_CONTEXT: int | None = None


def set_visible_context(steps: int | None) -> None:
  global _VISIBLE_CONTEXT
  _VISIBLE_CONTEXT = None if steps is None else int(steps)


def get_visible_context() -> int | None:
  return _VISIBLE_CONTEXT
