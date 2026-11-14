import mjlab  # noqa: F401
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from go_anywhere.tasks import ALL_TASKS

pytestmark = pytest.mark.gpu


def _make_env(task, device, num_envs=8):
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.border_width = 5.0
  return ManagerBasedRlEnv(cfg=cfg, device=device)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_reset_and_step(task, device):
  env = _make_env(task, device)
  try:
    obs, _ = env.reset()
    rl_cfg = load_rl_cfg(task)
    for group_set, groups in rl_cfg.obs_groups.items():
      for g in groups:
        assert g in obs, (group_set, g)
    assert obs["proprio_hist"].dim() == 3 and obs["proprio_hist"].shape[1] == 25
    assert obs["terrain_hist"].dim() == 3 and obs["terrain_hist"].shape[1] == 50
    for _ in range(10):
      actions = (
        torch.rand(env.num_envs, env.action_manager.total_action_dim, device=device) * 2
        - 1
      )
      obs, rew, term, trunc, extras = env.step(actions)
      for g, v in obs.items():
        assert torch.isfinite(v).all(), g
      assert torch.isfinite(rew).all()
  finally:
    env.close()


def test_drops_reset_is_airborne(device):
  env = _make_env("Go-Anywhere-Expert-Drops-Go2", device)
  try:
    env.reset()
    z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert (z > 0.25).all(), z
  finally:
    env.close()
