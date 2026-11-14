"""flat expert. Family overrides live in common.make_expert_env_cfg."""

from go_anywhere.tasks.experts.common import register_expert

TASK_ID = register_expert("flat")
