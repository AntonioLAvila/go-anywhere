import mjlab  # noqa: F401
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from go_anywhere.mdp import COMMAND_DIM, GoalCommand

pytestmark = pytest.mark.gpu


def test_goal_command_dims_and_arrival(device):
  cfg = load_env_cfg("Go-Anywhere-Expert-Flat-Go2")
  cfg.scene.num_envs = 4
  cfg.scene.terrain.terrain_generator.num_rows = 2
  cfg.scene.terrain.terrain_generator.border_width = 5.0
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    obs, _ = env.reset()
    goal = env.command_manager.get_term("goal")
    assert isinstance(goal, GoalCommand)
    assert obs["command"].shape == (4, COMMAND_DIM)
    assert (goal.budget > 0).all()
    # The goal is drawn from the post-reset pose, not the stale pre-reset one.
    pos = env.scene["robot"].data.root_link_pos_w
    dist = torch.norm(goal.goal_pos_w[:, :2] - pos[:, :2], dim=-1)
    assert torch.allclose(goal.initial_dist, dist, atol=1e-4)
    assert torch.allclose(goal.dist_xy, dist, atol=1e-4)
    # Teleport the robot onto its goal with the goal heading: arrival should latch.
    robot = env.scene["robot"]
    pos = robot.data.root_link_pos_w.clone()
    pos[:, :2] = goal.goal_pos_w[:, :2]
    yaw = goal.goal_yaw_w
    quat = torch.stack(
      [torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], dim=-1
    )
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
    env.step(torch.zeros(4, env.action_manager.total_action_dim, device=device))
    assert goal.metrics["goals_reached"].sum() >= 1
  finally:
    env.close()


def test_route_is_followed_waypoint_by_waypoint(device):
  cfg = load_env_cfg("Go-Anywhere-Expert-Flat-Go2")
  cfg.scene.num_envs = 4
  cfg.scene.terrain.terrain_generator.num_rows = 2
  cfg.scene.terrain.terrain_generator.border_width = 5.0
  cfg.terminations = {"route_complete": cfg.terminations["route_complete"]}
  cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    goal = env.command_manager.get_term("goal")
    assert isinstance(goal, GoalCommand)
    robot = env.scene["robot"]
    assert goal.on_route.all() and (goal.waypoint_idx == 0).all()
    route_len = goal.route_len.clone()
    assert ((route_len >= 5) & (route_len <= 9)).all()
    zero = torch.zeros(4, env.action_manager.total_action_dim, device=device)
    completed = torch.zeros(4, dtype=torch.bool, device=device)
    for hop in range(int(route_len.max())):
      # Each goal lies one waypoint further along the route.
      assert (goal.waypoint_idx[~completed] == hop).all()
      pos = goal.goal_pos_w.clone()
      pos[:, 2] += 0.3
      yaw = goal.goal_yaw_w
      quat = torch.stack(
        [torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], dim=-1
      )
      ids = (~completed).nonzero().flatten()
      robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1)[ids], ids)
      # Arrival is detected after the first step's physics; the termination that
      # reads it fires on the next step, as a time-out.
      env.step(zero)
      assert (goal.arrival[ids] == 1.0).all()  # On time: full arrival reward...
      _, _, _, time_outs, _ = env.step(zero)
      assert (goal.arrival[ids][~time_outs[ids]] == 0.0).all()  # ...paid once.
      assert (time_outs[ids] == (route_len[ids] == hop + 1)).all()
      completed |= time_outs
    assert completed.all()
    assert (goal.last_episode_success == 1.0).all()
  finally:
    env.close()
