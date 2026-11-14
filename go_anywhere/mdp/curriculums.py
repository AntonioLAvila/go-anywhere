"""Curricula: terrain levels for the goal task, staged parameters, context length."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from go_anywhere.mdp.commands import GoalCommand
from go_anywhere.rl.context import set_visible_context
from go_anywhere.terrains.presets import sub_terrain_group

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def terrain_levels_goal(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  demote_progress_fraction: float = 0.1,
) -> dict[str, torch.Tensor]:
  """Promote on completing the task; relaxed demotion (Zhang et al. 2024).

  The task is the whole route on course cells and a single goal elsewhere. A
  robot is demoted only when it covered less of it than a random policy would
  (``demote_progress_fraction``), so it keeps interacting with a level until
  it overcomes it.
  """
  terrain = env.scene.terrain
  assert terrain is not None and terrain.cfg.terrain_generator is not None
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GoalCommand)

  reached = term.metrics["success"][env_ids] > 0.5
  move_up = torch.where(term.on_route[env_ids], term.route_done[env_ids], reached)
  move_down = (term.route_progress[env_ids] < demote_progress_fraction) & ~move_up
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)
  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  out: dict[str, torch.Tensor] = {"mean": levels.mean(), "max": levels.max()}
  names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
  if terrain.terrain_origins is not None and terrain.terrain_origins.shape[1] == len(
    names
  ):
    # One entry per group of layout variants (``gaps/0`` .. ``gaps/7`` -> ``gaps``).
    groups = [sub_terrain_group(name) for name in names]
    group_of_type = torch.tensor(
      [groups.index(g) for g in groups], device=levels.device
    )[terrain.terrain_types]
    for i in sorted(set(group_of_type.tolist())):
      out[groups[i]] = levels[group_of_type == i].mean()
  return out


def _apply_stage(target: dict[str, Any], stage: dict[str, Any]) -> None:
  for key, value in stage.items():
    if key == "step":
      continue
    target[key] = value


def event_param_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  stages: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
  """Set an event term's params from a step-indexed stage list.

  Used for the drop-height curriculum (``reset_airborne``) and the reverse
  curriculum window (``reset_from_reference_states``).
  """
  del env_ids
  cfg = env.event_manager.get_term_cfg(event_name)
  active = 0
  for i, stage in enumerate(stages):
    if env.common_step_counter >= stage["step"]:
      _apply_stage(cfg.params, stage)
      active = i
  return {"stage": torch.tensor(float(active))}


def command_speed_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
  """Widen the required-speed range (urgency) over training."""
  del env_ids
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GoalCommand)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      if "required_speed_range" in stage:
        term.cfg.required_speed_range = tuple(stage["required_speed_range"])
  lo, hi = term.cfg.required_speed_range
  return {"speed_min": torch.tensor(lo), "speed_max": torch.tensor(hi)}


def context_length_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
  """LocoFormer-style context curriculum: grow the visible proprio window."""
  del env_ids
  visible = None
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      visible = stage.get("visible_context")
  set_visible_context(visible)
  return {
    "visible_context": torch.tensor(float(visible if visible is not None else -1))
  }
