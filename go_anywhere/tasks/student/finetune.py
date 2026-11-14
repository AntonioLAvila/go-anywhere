"""Go-Anywhere-Finetune-Go2: PPO finetune of the distilled student.

Precautions from Rudin et al. 2025: critic warm-up with the actor frozen,
reduced initial std, privileged critic, conservative LR. Warm-start with
``--agent.init-from <student_checkpoint.pt>``.
"""

from mjlab.tasks.registry import register_mjlab_task

from go_anywhere.rl import (
  GAUSSIAN,
  GoAnywhereOnPolicyRunner,
  GoAnywhereOnPolicyRunnerCfg,
  GoAnywherePpoAlgorithmCfg,
  LocoTransformerCfg,
)
from go_anywhere.tasks.experts.common import STUDENT_GROUPS, TEACHER_GROUPS
from go_anywhere.tasks.student.distill import distill_env_cfg

TASK_ID = "Go-Anywhere-Finetune-Go2"


def finetune_runner_cfg() -> GoAnywhereOnPolicyRunnerCfg:
  return GoAnywhereOnPolicyRunnerCfg(
    actor=LocoTransformerCfg(distribution_cfg={**GAUSSIAN, "init_std": 0.3}),
    critic=LocoTransformerCfg(),
    algorithm=GoAnywherePpoAlgorithmCfg(
      learning_rate=1.0e-4,
      schedule="adaptive",
      desired_kl=0.005,
      entropy_coef=0.002,
      critic_warmup_iters=200,
      bc_coef=0.0,
      symmetry_cfg=None,
    ),
    obs_groups={"actor": STUDENT_GROUPS, "critic": TEACHER_GROUPS + ("task_id",)},
    init_std=0.3,
    experiment_name="go2_finetune",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=distill_env_cfg(),
  play_env_cfg=distill_env_cfg(play=True),
  rl_cfg=finetune_runner_cfg(),
  runner_cls=GoAnywhereOnPolicyRunner,
)
