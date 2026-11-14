"""Render a terrain preset in the MuJoCo viewer and print cell metadata.

Usage: bin/uv-run python scripts/preview_terrain.py --preset steps [--rows 4] [--no-viewer]
Presets: a family name (flat, rough, stairs, steps, gaps, climb, descend, hurdles, drops) or ``full``.
"""

from __future__ import annotations

import argparse

from mjlab.terrains import TerrainEntity, TerrainEntityCfg

from go_anywhere.terrains import family_terrain_cfg, full_distribution_cfg, read_cells


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--preset", default="full")
  ap.add_argument("--rows", type=int, default=4)
  ap.add_argument("--no-viewer", action="store_true")
  args = ap.parse_args()
  gen = (
    full_distribution_cfg()
    if args.preset == "full"
    else family_terrain_cfg(args.preset)
  )
  gen.num_rows = args.rows
  gen.border_width = 5.0
  terrain = TerrainEntity(
    TerrainEntityCfg(terrain_type="generator", terrain_generator=gen), device="cpu"
  )
  model = terrain.spec.compile()
  cells = read_cells(terrain.spec, gen.size, gen.num_rows, len(gen.sub_terrains))
  names = list(gen.sub_terrains)
  print(f"cells={len(cells)} geoms={model.ngeom} hfields={model.nhfield}")
  for (row, col), meta in sorted(cells.items()):
    print(
      f"{names[col]:12s} row {row}: difficulty={meta.difficulty:.2f} "
      f"walled={meta.walled} waypoints={len(meta.waypoints)} "
      f"path={[kind for _, kind, _ in meta.path]}"
    )
  if not args.no_viewer:
    import mujoco.viewer as viewer

    viewer.launch(model)


if __name__ == "__main__":
  main()
