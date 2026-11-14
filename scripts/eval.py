"""Evaluate a checkpoint per skill: success, time-to-goal, CoT, falls, peak impact.

Usage:
  bin/uv-run python scripts/eval.py Go-Anywhere-Expert-Flat-Go2 --checkpoint logs/.../model_5000.pt \
      --num-envs 512 --episodes 1000 [--csv out.csv]
Per sub-terrain group (layout variants pooled) and per difficulty row.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from go_anywhere import mdp
from go_anywhere.rl import GoAnywhereOnPolicyRunner
from go_anywhere.terrains import sub_terrain_group


@dataclass
class EvalConfig:
  task: str
  checkpoint: str
  num_envs: int = 256
  episodes: int = 512
  device: str = "cuda:0"
  curriculum_rows: bool = True
  """Keep the curriculum grid (one column per sub-terrain, rows = difficulty)."""
  csv: str | None = None


def main(cfg: EvalConfig) -> None:
  import go_anywhere.tasks  # noqa: F401

  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.curriculum = {}
  env_cfg.observations["proprio_hist"].enable_corruption = True
  if (
    env_cfg.scene.terrain is not None
    and env_cfg.scene.terrain.terrain_generator is not None
  ):
    env_cfg.scene.terrain.max_init_terrain_level = None
  agent_cfg = load_rl_cfg(cfg.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or GoAnywhereOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(
    cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )
  policy = runner.get_inference_policy(device=cfg.device)

  terrain = env.scene.terrain
  names = (
    list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if terrain and terrain.cfg.terrain_generator
    else ["plane"]
  )
  goal = env.command_manager.get_term("goal")
  assert isinstance(goal, mdp.GoalCommand)

  stats: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
    lambda: defaultdict(list)
  )
  peak_impact = torch.zeros(env.num_envs, device=cfg.device)
  energy = torch.zeros(env.num_envs, device=cfg.device)
  distance = torch.zeros(env.num_envs, device=cfg.device)
  prev_pos = env.scene["robot"].data.root_link_pos_w[:, :2].clone()
  done_count = 0
  obs = wrapped.get_observations()
  with torch.inference_mode():
    while done_count < cfg.episodes:
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      peak_impact = torch.maximum(
        peak_impact, mdp.peak_foot_impact_bw(env, "feet_ground_contact")
      )
      energy += mdp.mechanical_power(env) * env.step_dt
      pos = env.scene["robot"].data.root_link_pos_w[:, :2]
      distance += torch.norm(pos - prev_pos, dim=-1)
      prev_pos = pos.clone()
      done_ids = dones.nonzero().flatten()
      if len(done_ids) == 0:
        continue
      terminated = env.termination_manager.terminated[done_ids]
      col = (
        terrain.terrain_types[done_ids]
        if terrain and terrain.terrain_origins is not None
        else torch.zeros_like(done_ids)
      )
      row = (
        terrain.terrain_levels[done_ids]
        if terrain and terrain.terrain_origins is not None
        else torch.zeros_like(done_ids)
      )
      for i, e in enumerate(done_ids.tolist()):
        key = (sub_terrain_group(names[int(col[i])]), int(row[i]))
        s = stats[key]
        s["success"].append(float(goal.last_episode_success[e]))
        s["goals"].append(float(goal.last_episode_goals[e]))
        s["time_to_goal"].append(float(goal.last_episode_time_to_goal[e]))
        s["fall"].append(float(terminated[i]))
        s["peak_impact_bw"].append(float(peak_impact[e]))
        d = max(float(distance[e]), 1e-3)
        s["cot"].append(float(energy[e]) / (mdp.contact.BODY_WEIGHT_N * d))
      peak_impact[done_ids] = 0.0
      energy[done_ids] = 0.0
      distance[done_ids] = 0.0
      prev_pos[done_ids] = env.scene["robot"].data.root_link_pos_w[done_ids, :2]
      done_count += len(done_ids)

  rows = []
  print(
    f"{'terrain':28s} {'row':>3s} {'n':>5s} {'success':>8s} {'goals':>6s} {'ttg[s]':>7s} {'fall':>6s} {'impact[BW]':>10s} {'CoT':>6s}"
  )
  for (name, row), s in sorted(stats.items()):
    n = len(s["success"])
    rec = {
      "terrain": name,
      "row": row,
      "n": n,
      "success": sum(s["success"]) / n,
      "goals": sum(s["goals"]) / n,
      "time_to_goal": sum(s["time_to_goal"]) / n,
      "fall": sum(s["fall"]) / n,
      "peak_impact_bw": sum(s["peak_impact_bw"]) / n,
      "cot": sum(s["cot"]) / n,
    }
    rows.append(rec)
    print(
      f"{name:28s} {row:3d} {n:5d} {rec['success']:8.3f} {rec['goals']:6.2f} {rec['time_to_goal']:7.2f} "
      f"{rec['fall']:6.3f} {rec['peak_impact_bw']:10.2f} {rec['cot']:6.3f}"
    )
  if cfg.csv:
    with open(cfg.csv, "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      w.writeheader()
      w.writerows(rows)
    print(f"wrote {cfg.csv}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalConfig, config=mjlab.TYRO_FLAGS))
