"""Go-Anywhere-Distill-Go2: DAgger distillation of the expert ensemble."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task

from go_anywhere.envs.base_env_cfg import make_go_anywhere_env_cfg
from go_anywhere.rl import (
  GAUSSIAN,
  DaggerAlgorithmCfg,
  GoAnywhereDistillationRunner,
  LocoTransformerCfg,
  RslRlDistillationRunnerCfg,
  TeacherEnsembleCfg,
)
from go_anywhere.terrains import full_distribution_cfg

TASK_ID = "Go-Anywhere-Distill-Go2"


def distill_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_go_anywhere_env_cfg(
    full_distribution_cfg(),
    play=play,
    trunk_contact_terminates=False,  # Mixed cells include recovery situations.
  )
  cfg.curriculum.pop("context_length", None)  # Student always sees the full window.
  return cfg


def distill_runner_cfg() -> RslRlDistillationRunnerCfg:
  return RslRlDistillationRunnerCfg(
    student=LocoTransformerCfg(distribution_cfg=dict(GAUSSIAN)),
    teacher=TeacherEnsembleCfg(manifest="configs/experts.yaml"),
    algorithm=DaggerAlgorithmCfg(),
    experiment_name="go2_distill",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=distill_env_cfg(),
  play_env_cfg=distill_env_cfg(play=True),
  rl_cfg=distill_runner_cfg(),
  runner_cls=GoAnywhereDistillationRunner,
)
