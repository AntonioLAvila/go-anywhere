"""Task registration entry point (``mjlab.tasks`` entry-point group).

Importing this package registers every ``Go-Anywhere-*`` task with mjlab's
registry so the stock ``train`` / ``play`` / ``list-envs`` scripts find them.
"""

from go_anywhere.tasks import experts, student  # noqa: F401

EXPERT_TASKS = tuple(
  f"Go-Anywhere-Expert-{name}-Go2"
  for name in (
    "Flat",
    "Rough",
    "Stairs",
    "Steps",
    "Gaps",
    "Climb",
    "Descend",
    "Hurdles",
    "Hurdles-Hard",
    "Drops",
  )
)
STUDENT_TASKS = ("Go-Anywhere-Distill-Go2", "Go-Anywhere-Finetune-Go2")
ALL_TASKS = EXPERT_TASKS + STUDENT_TASKS
