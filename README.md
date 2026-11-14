# go-anywhere

Near-arbitrary locomotion for the Unitree Go2, built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo-Warp + rsl_rl). Pipeline: privileged per-skill **experts** → DAgger **distillation** into one
student → PPO **finetune** on the full terrain distribution. Every policy is the same transformer over a
history window; control is 50 Hz joint-position targets.

## Setup

```bash
git clone --recurse-submodules <repo-url>   # or: git submodule update --init
uv sync
bin/uv-run list-envs                         # shows the Go-Anywhere-* tasks
bin/uv-run python scripts/check_go2.py       # compiles the Go2 asset
bin/uv-run pytest                            # CPU tests; add -m gpu on a CUDA machine
```

The robot MJCF comes from the [`go2_mjcf`](https://github.com/AntonioLAvila/go2_mjcf) submodule.
`bin/uv-run` is a thin `uv run` wrapper that puts `torch/lib` on `LD_LIBRARY_PATH`.

## Tasks

| Task id | What |
|---|---|
| `Go-Anywhere-Expert-{Flat,Rough,Stairs,Steps,Gaps,Climb,Descend,Drops}-Go2` | one privileged expert per skill family |
| `Go-Anywhere-Expert-Hurdles-Go2` / `-Hurdles-Hard-Go2` | hurdles, stage A (penetrable bars + penetration cost) / stage B (collidable) |
| `Go-Anywhere-Distill-Go2` | DAgger distillation of the experts listed in `configs/experts.yaml` |
| `Go-Anywhere-Finetune-Go2` | PPO finetune of the distilled student (critic warm-up, low init std) |

```bash
bin/uv-run train Go-Anywhere-Expert-Steps-Go2 --env.scene.num-envs 8192 --gpu-ids all
bin/uv-run play  Go-Anywhere-Expert-Steps-Go2
bin/uv-run python scripts/make_experts_manifest.py          # -> configs/experts.yaml
bin/uv-run train Go-Anywhere-Distill-Go2
bin/uv-run train Go-Anywhere-Finetune-Go2 --agent.init-from logs/rsl_rl/go2_distill/<run>/model_5000.pt
bin/uv-run python scripts/eval.py Go-Anywhere-Expert-Steps-Go2 --checkpoint <ckpt> --csv eval.csv
bin/uv-run python scripts/preview_terrain.py --preset full
```

## Layout

- `go_anywhere/assets/go2.py` — Go2 entity config (DC-motor PD actuators, collisions, head camera, body-weight constants).
- `go_anywhere/terrains/` — path-first tile grammar (`grammar.py`), model-synthesis fill (`synthesis.py`),
  fBm roughness (`noise.py`), course cell (`course.py`), Stones-Everywhere cell (Zhang et al. 2024), presets.
- `go_anywhere/mdp/` — goal command, contact scheme (body-weight thresholds), rewards, curricula, reset events.
- `go_anywhere/envs/base_env_cfg.py` — the shared goal-reaching environment.
- `go_anywhere/rl/` — `LocoTransformer`, `TeacherEnsemble`, DAgger distillation, finetune PPO, runners, symmetry.
- `go_anywhere/tasks/` — task registrations (entry point).
- `scripts/` — asset check, terrain preview, expert manifest, evaluation table.

## Phases

1. Scaffold (this repo): all tasks register and step; flat/rough/steps train end to end.
2. Expert tuning, capability-envelope calibration from `eval.py`, reference-state banks, hurdle stage B.
3. Distillation on mixed courses; natural-patch library; in-place heightfield refresh experiment.
4. Finetune; rendered depth from `head_camera`; context-length ablations.
5. Motion priors (AMP); gait-transition analysis. 6. Regret curricula over course parameters.
