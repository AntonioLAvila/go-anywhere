import copy

import mjlab  # noqa: F401  (loads entry points)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from go_anywhere.tasks import ALL_TASKS


def test_all_tasks_registered():
  tasks = set(list_tasks())
  missing = [t for t in ALL_TASKS if t not in tasks]
  assert not missing, missing


def test_cfgs_copy_cleanly():
  for task in ALL_TASKS:
    env_cfg = load_env_cfg(task)
    rl_cfg = load_rl_cfg(task)
    copy.deepcopy(env_cfg)
    copy.deepcopy(rl_cfg)
    assert env_cfg.decimation == 4 and env_cfg.sim.mujoco.timestep == 0.005  # 50 Hz
