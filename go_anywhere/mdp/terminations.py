"""Terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.terminations import illegal_contact as illegal_contact

from go_anywhere.mdp.commands import GoalCommand
from go_anywhere.mdp.contact import foot_impact_exceeded as foot_impact_exceeded
from go_anywhere.mdp.contact import trunk_contact as trunk_contact

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def goal_timeout(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Fail the episode once the goal budget is overrun by the command's grace."""
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GoalCommand)
  return term.budget_left < -term.cfg.late_grace_s


def route_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """End the episode (as a time-out) once the last waypoint of a route is reached."""
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GoalCommand)
  return term.route_done


__all__ = [
  "goal_timeout",
  "route_complete",
  "illegal_contact",
  "foot_impact_exceeded",
  "trunk_contact",
]
