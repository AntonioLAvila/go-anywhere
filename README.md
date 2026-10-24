# go-anywhere

RL locomotion for the Unitree Go2, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

```bash
git clone --recurse-submodules <repo-url>
# or, in an existing checkout:
git submodule update --init

uv sync
```

The robot MJCF comes from the [`go2_mjcf`](https://github.com/AntonioLAvila/go2_mjcf) submodule.

Run things through `bin/uv-run` (a thin `uv run` wrapper that puts `torch/lib` on
`LD_LIBRARY_PATH`, needed when another environment such as ROS has already set it):

```bash
bin/uv-run python scripts/train.py ...
```
