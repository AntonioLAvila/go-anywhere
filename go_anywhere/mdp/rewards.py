"""Goal-task rewards (after Zhang et al. 2024) and regularizers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from go_anywhere.assets.go2 import BODY_WEIGHT_N
from go_anywhere.mdp.commands import GoalCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _goal(env: ManagerBasedRlEnv, command_name: str) -> GoalCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GoalCommand)
  return term


def goal_arrival(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-off reward for reaching a goal: 1 per arrival within the budget, less
  when late (see ``GoalCommand.arrival``).

  Zhang et al. 2024 pay ``1 / (1 + d^2)`` during the last seconds of the budget,
  which integrates to 1 for a robot that arrives in time and waits. Goals here
  are replaced on arrival, so that integral is paid at once; paying it per step
  instead rewards hovering next to the goal until the budget runs out. Divided
  by dt so the weight reads per arrival (mjlab scales rewards by dt).
  """
  return _goal(env, command_name).arrival / env.step_dt


def heading_tracking(
  env: ManagerBasedRlEnv, command_name: str, near_radius: float = 2.5
) -> torch.Tensor:
  """``1 / (1 + heading_err^2)`` while within ``near_radius`` of the goal."""
  g = _goal(env, command_name)
  return (g.dist_xy < near_radius).float() / (1.0 + g.heading_err**2)


def goal_progress(
  env: ManagerBasedRlEnv, command_name: str, max_rate: float = 3.0
) -> torch.Tensor:
  """Decrease of the distance to the active goal per second (potential shaping)."""
  g = _goal(env, command_name)
  return (g.progress / env.step_dt).clamp(-max_rate, max_rate)


def dont_wait(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_threshold: float = 0.2,
  near_radius: float = 1.0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """1 when the robot is nearly stationary and not yet near the goal."""
  asset: Entity = env.scene[asset_cfg.name]
  g = _goal(env, command_name)
  speed = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
  return ((speed < speed_threshold) & (g.dist_xy > near_radius)).float()


def _joint_power(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Per-joint mechanical power ``tau * qdot`` (W). The Go2 entity sorts its
  actuators, so ``actuator_force`` is in joint order."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.actuator_force * asset.data.joint_vel


def mechanical_power(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Sum of |tau * qdot| (W); energy term from Zhuang et al. 2023."""
  return _joint_power(env, asset_cfg).abs().sum(dim=-1)


def cost_of_transport(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT,
  min_speed: float = 0.2,
  body_weight_n: float = BODY_WEIGHT_N,
) -> torch.Tensor:
  """Positive mechanical power / (m g v); dimensionless."""
  asset: Entity = env.scene[asset_cfg.name]
  power = _joint_power(env, asset_cfg).clamp(min=0.0).sum(dim=-1)
  v = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1).clamp(min=min_speed)
  return power / (body_weight_n * v)
