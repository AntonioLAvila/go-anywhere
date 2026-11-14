"""Shared expert construction: env per family + PPO runner cfg."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

from go_anywhere import mdp
from go_anywhere.envs.base_env_cfg import make_go_anywhere_env_cfg
from go_anywhere.rl import (
  GAUSSIAN,
  GoAnywhereOnPolicyRunner,
  GoAnywhereOnPolicyRunnerCfg,
  GoAnywherePpoAlgorithmCfg,
  LocoTransformerCfg,
  rnd_cfg,
  symmetry_cfg,
)
from go_anywhere.terrains import FAMILIES, family_terrain_cfg

# Experts see the student's terrain history too (input similarity for distillation)
# plus the clean current scan and privileged state.
TEACHER_GROUPS = (
  "proprio_hist",
  "terrain_hist",
  "terrain_priv",
  "command",
  "privileged",
)
STUDENT_GROUPS = ("proprio_hist", "terrain_hist", "command")

RND_FAMILIES = ("steps", "gaps", "hurdles")
# Novelty is measured on the current state and terrain. rsl_rl's RND concatenates
# its groups, so they must be flat (no history groups).
RND_STATE_GROUPS = ("privileged", "terrain_priv")

DROP_HEIGHT_STAGES = [
  {"step": 0, "height_range": (0.3, 0.6), "roll_pitch_max": math.radians(30)},
  {"step": 1500 * 24, "height_range": (0.4, 1.0), "roll_pitch_max": math.radians(60)},
  {
    "step": 3500 * 24,
    "height_range": (0.5, 1.5),
    "roll_pitch_max": math.pi,
    "ang_vel_max": 1.0,
    "lin_vel_xy_max": 1.5,
  },
]

REVERSE_CURRICULUM_STAGES = [
  {"step": 0, "index_range": (0.8, 1.0)},
  {"step": 1000 * 24, "index_range": (0.5, 1.0)},
  {"step": 2500 * 24, "index_range": (0.0, 1.0)},
]


def make_expert_env_cfg(
  family: str,
  play: bool = False,
  hurdles_collidable: bool = True,
  reference_bank: str | None = None,
) -> ManagerBasedRlEnvCfg:
  if family not in FAMILIES:
    raise ValueError(f"Unknown family {family}")
  gen = family_terrain_cfg(family, hurdles_collidable=hurdles_collidable)
  cfg = make_go_anywhere_env_cfg(
    gen,
    play=play,
    trunk_contact_terminates=family != "drops",
    foot_impact_terminates=family in ("descend", "gaps", "climb"),
    world_contact_weight=-10.0 if family in ("hurdles", "gaps") else -1.0,
  )

  if family == "flat":
    cfg.rewards.pop("feet_edge", None)

  if family == "drops":
    cfg.events["reset_base"] = EventTermCfg(
      func=mdp.reset_airborne, mode="reset", params=dict(DROP_HEIGHT_STAGES[0])
    )
    cfg.events["reset_base"].params.pop("step", None)
    cfg.events.pop("push_robot", None)
    cfg.curriculum["drop_height"] = CurriculumTermCfg(
      func=mdp.event_param_stages,
      params={"event_name": "reset_base", "stages": DROP_HEIGHT_STAGES},
    )
    goal_cfg = cfg.commands["goal"]
    assert isinstance(goal_cfg, mdp.GoalCommandCfg)
    goal_cfg.required_speed_range = (0.2, 0.8)
    goal_cfg.time_budget_range = (5.0, 12.0)
    cfg.rewards["foot_impact"].weight = -2.0
    cfg.rewards["world_contact"].weight = -0.5  # Recovery needs body contact.

  if family == "descend":
    cfg.rewards["foot_impact"].weight = -2.0
    cfg.rewards["soft_landing"] = RewardTermCfg(
      func=mdp.soft_landing, weight=-1e-4, params={"sensor_name": "feet_ground_contact"}
    )

  if family == "hurdles":
    cfg.rewards["penetration"] = RewardTermCfg(
      func=mdp.penetration_cost, weight=-0.5, params={}
    )

  if family in ("gaps", "climb", "hurdles") and reference_bank is not None:
    cfg.events["reset_reference"] = EventTermCfg(
      func=mdp.reset_from_reference_states,
      mode="reset",
      params={"bank_path": reference_bank, "index_range": (0.8, 1.0)},
    )
    cfg.curriculum["reverse_window"] = CurriculumTermCfg(
      func=mdp.event_param_stages,
      params={"event_name": "reset_reference", "stages": REVERSE_CURRICULUM_STAGES},
    )

  if play:
    cfg.curriculum = {}
  return cfg


def expert_ppo_runner_cfg(family: str) -> GoAnywhereOnPolicyRunnerCfg:
  algorithm = GoAnywherePpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=16,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    symmetry_cfg=symmetry_cfg(use_augmentation=True),
    rnd_cfg=rnd_cfg(weight=0.5) if family in RND_FAMILIES else None,
  )
  obs_groups: dict[str, tuple[str, ...]] = {
    "actor": TEACHER_GROUPS,
    "critic": TEACHER_GROUPS,
  }
  if algorithm.rnd_cfg is not None:
    obs_groups["rnd_state"] = RND_STATE_GROUPS
  return GoAnywhereOnPolicyRunnerCfg(
    actor=LocoTransformerCfg(distribution_cfg=dict(GAUSSIAN)),
    critic=LocoTransformerCfg(),
    algorithm=algorithm,
    obs_groups=obs_groups,
    experiment_name=f"go2_expert_{family}",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def task_id_for(family: str, suffix: str = "") -> str:
  return f"Go-Anywhere-Expert-{family.capitalize()}{suffix}-Go2"


def register_expert(family: str, suffix: str = "", **env_kwargs) -> str:
  task_id = task_id_for(family, suffix)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_expert_env_cfg(family, **env_kwargs),
    play_env_cfg=make_expert_env_cfg(family, play=True, **env_kwargs),
    rl_cfg=expert_ppo_runner_cfg(family),
    runner_cls=GoAnywhereOnPolicyRunner,
  )
  return task_id
