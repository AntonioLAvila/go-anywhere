"""Goal command: a target position (+ heading) with a time budget.

One command interface for every skill family. Goals come from the terrain
cell the robot is on: course cells give the route as ordered waypoints (the
next one becomes the goal on arrival, so a straight line to the goal always
lies on the feasible path), other cells give interchangeable goal candidates.
The budget is derived from the goal distance and a sampled required speed, so
urgency spans walking to galloping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  sample_uniform,
  wrap_to_pi,
  yaw_quat,
)

from go_anywhere.mdp.cells import get_cell_labels

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


COMMAND_DIM = 7
"""[dx, dy, dz (yaw frame), cos(heading err), sin(heading err),
budget_left / budget, required speed remaining]."""


class GoalCommand(CommandTerm):
  cfg: GoalCommandCfg

  def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    n, dev = self.num_envs, self.device
    self.goal_pos_w = torch.zeros(n, 3, device=dev)
    self.goal_yaw_w = torch.zeros(n, device=dev)
    self.budget = torch.ones(n, device=dev)
    self.budget_left = torch.ones(n, device=dev)
    self.initial_dist = torch.ones(n, device=dev)
    self.reached = torch.zeros(n, dtype=torch.bool, device=dev)
    self.dist_xy = torch.zeros(n, device=dev)
    self.heading_err = torch.zeros(n, device=dev)
    self.arrival = torch.zeros(n, device=dev)
    """Set for one step when a goal is reached: 1 if within the budget, decaying
    linearly to 0 over ``late_grace_s`` of overrun."""
    self.progress = torch.zeros(n, device=dev)
    """Decrease of the distance to the active goal over the last step (m)."""
    self._rel_yaw = torch.zeros(n, 3, device=dev)
    self._pending = torch.zeros(n, dtype=torch.bool, device=dev)
    self.waypoint_idx = torch.zeros(n, dtype=torch.long, device=dev)
    """Index of the active goal in an ordered route."""
    self.route_len = torch.ones(n, dtype=torch.long, device=dev)
    self.on_route = torch.zeros(n, dtype=torch.bool, device=dev)
    """The robot's cell has an ordered route (else goals are independent)."""
    self.route_done = torch.zeros(n, dtype=torch.bool, device=dev)
    self._heading_required = torch.ones(n, dtype=torch.bool, device=dev)
    self.cells = None
    if cfg.goal_source == "cell":
      self.cells = get_cell_labels(env)
      if not self.cells.available:
        raise ValueError("goal_source='cell' needs a go-anywhere terrain generator.")
    self._command = torch.zeros(n, COMMAND_DIM, device=dev)

    self.metrics["position_error"] = torch.zeros(n, device=dev)
    self.metrics["success"] = torch.zeros(n, device=dev)
    self.metrics["goals_reached"] = torch.zeros(n, device=dev)
    self.metrics["route_completed"] = torch.zeros(n, device=dev)
    self.metrics["budget_overrun"] = torch.zeros(n, device=dev)
    self.metrics["time_to_goal"] = torch.zeros(n, device=dev)
    # Episode outcomes captured at reset (for scripts/eval.py).
    self.last_episode_success = torch.zeros(n, device=dev)
    self.last_episode_goals = torch.zeros(n, device=dev)
    self.last_episode_time_to_goal = torch.zeros(n, device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    # On a route, success means reaching its end, not just the first waypoint.
    self.last_episode_success[env_ids] = torch.where(
      self.on_route[env_ids],
      self.route_done[env_ids].float(),
      self.metrics["success"][env_ids],
    )
    self.last_episode_goals[env_ids] = self.metrics["goals_reached"][env_ids]
    self.last_episode_time_to_goal[env_ids] = self.metrics["time_to_goal"][env_ids]
    return super().reset(env_ids)

  # Public helpers used by rewards / terminations / curricula.

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def progress_fraction(self) -> torch.Tensor:
    """Fraction of the distance to the active goal covered so far."""
    return 1.0 - self.dist_xy / self.initial_dist.clamp(min=1e-3)

  @property
  def route_progress(self) -> torch.Tensor:
    """Fraction of the whole task covered: waypoints passed plus progress toward
    the active one on a route, ``progress_fraction`` otherwise."""
    leg = self.progress_fraction.clamp(0.0, 1.0)
    along = (self.waypoint_idx + leg) / self.route_len
    along = torch.where(self.route_done, torch.ones_like(along), along)
    return torch.where(self.on_route, along, self.progress_fraction)

  # Resampling.

  def _sample_goals(self, env_ids: torch.Tensor) -> torch.Tensor:
    if self.cells is not None:
      terrain = self._env.scene.terrain
      assert terrain is not None
      levels = terrain.terrain_levels[env_ids]
      types = terrain.terrain_types[env_ids]
      num = self.cells.route_len[levels, types]
      ordered = self.cells.ordered[levels, types]
      self.route_len[env_ids] = num
      self.on_route[env_ids] = ordered
      idx = self.waypoint_idx[env_ids]
      random_idx = (torch.rand(len(env_ids), device=self.device) * num).long()
      k = torch.where(ordered, idx, random_idx).clamp(max=num - 1)
      # Intermediate waypoints are passed through; only the last needs the heading.
      self._heading_required[env_ids] = ~ordered | (idx >= num - 1)
      return self.cells.waypoints[levels, types, k]
    # Ring around the current robot position.
    pos = self.robot.data.root_link_pos_w[env_ids]
    lo, hi = self.cfg.ring_distance_range
    r = sample_uniform(lo, hi, (len(env_ids),), self.device)
    th = sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device)
    goal = pos.clone()
    goal[:, 0] += r * torch.cos(th)
    goal[:, 1] += r * torch.sin(th)
    return goal

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # The manager resamples during the env reset, before ``sim.forward()``: the
    # robot pose read here would still be the pre-reset one. Only mark the envs;
    # ``_update_command`` (always called after the forward) draws the goal.
    self._pending[env_ids] = True
    self.waypoint_idx[env_ids] = 0
    self.route_done[env_ids] = False

  def _draw_goals(self, env_ids: torch.Tensor) -> None:
    goal = self._sample_goals(env_ids)
    self.goal_pos_w[env_ids] = goal
    pos = self.robot.data.root_link_pos_w[env_ids]
    dist = torch.norm(goal[:, :2] - pos[:, :2], dim=-1)
    # Along a route the robot keeps facing its direction of travel.
    face_goal = torch.atan2(goal[:, 1] - pos[:, 1], goal[:, 0] - pos[:, 0])
    if self.cfg.heading_mode == "face_goal":
      yaw = face_goal
    else:
      yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device)
      yaw = torch.where(self.on_route[env_ids], face_goal, yaw)
    self.goal_yaw_w[env_ids] = yaw
    speed = sample_uniform(*self.cfg.required_speed_range, (len(env_ids),), self.device)
    budget = (dist / speed.clamp(min=1e-3)).clamp(*self.cfg.time_budget_range)
    self.budget[env_ids] = budget
    self.budget_left[env_ids] = budget
    self.initial_dist[env_ids] = dist.clamp(min=1e-3)
    self.dist_xy[env_ids] = dist  # Progress is measured against the new goal.
    self.reached[env_ids] = False

  # Per-step update.

  def _refresh_relative(self) -> None:
    """Goal position/heading relative to the robot, for the active goals."""
    pos = self.robot.data.root_link_pos_w
    quat = self.robot.data.root_link_quat_w
    rel_w = self.goal_pos_w - pos
    self._rel_yaw = quat_apply_inverse(yaw_quat(quat), rel_w).clamp(
      -self.cfg.max_relative_distance, self.cfg.max_relative_distance
    )
    self.dist_xy = torch.norm(rel_w[:, :2], dim=-1)
    if self.cfg.heading:
      _, _, yaw = euler_xyz_from_quat(quat)
      self.heading_err = wrap_to_pi(self.goal_yaw_w - yaw)
    else:
      self.heading_err = torch.zeros_like(self.dist_xy)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      self.budget_left -= self._env.step_dt
    pending = self._pending.nonzero().flatten()
    if len(pending) > 0:
      self._draw_goals(pending)
      self._pending[pending] = False

    prev_dist = self.dist_xy
    self._refresh_relative()
    self.progress = prev_dist - self.dist_xy

    newly = (self.dist_xy < self.cfg.success_radius) & (
      (self.heading_err.abs() < self.cfg.success_heading_tol) | ~self._heading_required
    )
    newly &= ~self.reached
    on_time = (1.0 + self.budget_left / self.cfg.late_grace_s).clamp(0.0, 1.0)
    self.arrival = newly.float() * on_time
    if newly.any():
      self.metrics["success"][newly] = 1.0
      self.metrics["goals_reached"][newly] += 1.0
      self.metrics["time_to_goal"][newly] = (self.budget - self.budget_left)[newly]
      self.reached |= newly
      # Routes advance to the next waypoint and end after the last one.
      advance = newly & self.on_route
      self.waypoint_idx[advance] += 1
      finished = advance & (self.waypoint_idx >= self.route_len)
      self.route_done |= finished
      self.metrics["route_completed"][finished] = 1.0
      next_goal = advance & ~finished
      if self.cfg.resample_on_success:
        next_goal |= newly & ~self.on_route
      if next_goal.any():
        self._draw_goals(next_goal.nonzero().flatten())
        self._refresh_relative()

    frac = (self.budget_left / self.budget.clamp(min=1e-3)).clamp(-1.0, 1.0)
    req_speed = (self.dist_xy / self.budget_left.clamp(min=0.2)).clamp(0.0, 5.0)
    self._command = torch.cat(
      [
        self._rel_yaw,
        torch.cos(self.heading_err).unsqueeze(-1),
        torch.sin(self.heading_err).unsqueeze(-1),
        frac.unsqueeze(-1),
        req_speed.unsqueeze(-1),
      ],
      dim=-1,
    )

  def _update_metrics(self) -> None:
    self.metrics["position_error"] = self.dist_xy.clone()
    self.metrics["budget_overrun"] = (self.budget_left < 0.0).float()

  # Debug visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    goals = self.goal_pos_w.cpu().numpy()
    base = self.robot.data.root_link_pos_w.cpu().numpy()
    reached = self.reached.cpu().numpy()
    for i in env_indices:
      color = (0.1, 0.9, 0.2, 0.9) if reached[i] else (0.9, 0.6, 0.1, 0.9)
      visualizer.add_sphere(goals[i] + (0.0, 0.0, 0.15), 0.08, color)
      visualizer.add_arrow(
        base[i], goals[i] + (0.0, 0.0, 0.15), (0.2, 0.6, 1.0, 0.8), 0.01
      )


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
  """Unused by the goal logic (goals resample on arrival / reset); kept huge."""
  goal_source: Literal["cell", "ring"] = "cell"
  """``cell``: waypoints of the robot's terrain cell. ``ring``: a random point
  around the robot (plane terrain)."""
  ring_distance_range: tuple[float, float] = (1.5, 4.0)
  heading: bool = True
  heading_mode: Literal["random", "face_goal"] = "random"
  required_speed_range: tuple[float, float] = (0.3, 1.5)
  """Sampled mean speed that sets the budget (urgency). Widen for gait studies."""
  time_budget_range: tuple[float, float] = (2.0, 12.0)
  success_radius: float = 0.25
  success_heading_tol: float = 0.5
  resample_on_success: bool = True
  """Draw a new goal on arrival in cells without a route (routes always advance)."""
  late_grace_s: float = 2.0
  """Overrun of the budget after which the goal counts as missed (``goal_timeout``);
  the arrival reward decays to zero over it."""
  max_relative_distance: float = 5.0

  def build(self, env: ManagerBasedRlEnv) -> GoalCommand:
    return GoalCommand(self, env)
