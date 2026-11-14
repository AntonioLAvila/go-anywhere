# go-anywhere — conventions

- Run everything through `bin/uv-run` (wraps `uv run`, fixes torch `LD_LIBRARY_PATH`).
  Training/eval use mjlab's console scripts: `bin/uv-run train <TaskId> ...`, `bin/uv-run play <TaskId>`,
  `bin/uv-run list-envs`. Tasks are discovered through the `mjlab.tasks` entry point (`go_anywhere.tasks`).
- Control is 50 Hz PD joint-position targets (sim 0.005 s, decimation 4). Do not change without reason.
- Every policy is a `LocoTransformer` (go_anywhere/rl/models.py). Experts and student differ only in
  observation sets (`obs_groups` in the runner cfg), never in architecture.
- Observation groups are building blocks: `proprio_hist` (25×), `terrain_hist` (50×, strided to 10 Hz in the
  model), `terrain_priv`, `command`, `privileged`, `task_id` (teacher-only). Add terms to the groups, add
  a mirror rule in `go_anywhere/rl/symmetry.py` for each new term name.
- Vector observations must be base- or yaw-frame, never world-frame: world-frame vectors make the policy
  yaw-dependent and cannot be mirrored (`TERM_RULES` in `symmetry.py`).
- Contact thresholds live in `go_anywhere/mdp/contact.py` in body-weight units.
- Adding a skill family: a tile kind in `terrains/tiles.py` (+ rasterizer, envelope entry), a preset in
  `terrains/presets.py`, a `Skill` id, an expert module in `tasks/experts/`, and overrides in
  `tasks/experts/common.py`.
- Terrain cells must call `write_cell_meta` (`terrains/raster.py`; see `CourseTerrainCfg`,
  `StonesEverywhereTerrainCfg`): `task_id` routing, penetration costs and the goal command read per-cell
  labels, boxes and waypoints back from the terrain spec with `read_cells`.
- Tests: `bin/uv-run pytest` (CPU), `bin/uv-run pytest -m gpu` (needs CUDA). Style: ruff, 2-space indent.
- Distillation: the runner's ``teacher`` observation set must equal the experts' actor set exactly
  (same groups, same order); routing reads ``task_id`` from the env observations on its own.
- mjlab dumps ``params/agent.yaml`` with python-tuple tags: load it with ``yaml.FullLoader``.
