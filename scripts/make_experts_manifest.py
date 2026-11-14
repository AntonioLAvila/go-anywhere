"""Write configs/experts.yaml from the latest expert checkpoints under logs/.

Usage: bin/uv-run python scripts/make_experts_manifest.py [--log-root logs/rsl_rl] [--out configs/experts.yaml]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from go_anywhere.terrains import FAMILIES

DEFAULT_SKILLS = {fam: [fam] for fam in FAMILIES}


def latest_checkpoint(exp_dir: Path) -> Path | None:
  runs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
  for run in reversed(runs):
    ckpts = sorted(
      run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])
    )
    if ckpts:
      return ckpts[-1]
  return None


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--log-root", default="logs/rsl_rl")
  ap.add_argument("--out", default="configs/experts.yaml")
  args = ap.parse_args()
  root = Path(args.log_root)
  experts: dict[str, dict] = {}
  for fam in FAMILIES:
    exp = root / f"go2_expert_{fam}"
    if not exp.exists():
      continue
    ckpt = latest_checkpoint(exp)
    if ckpt is None:
      continue
    experts[fam] = {"checkpoint": str(ckpt), "skills": list(DEFAULT_SKILLS[fam])}
  if "drops" not in experts and "flat" in experts:
    experts["flat"]["skills"].append("drops")
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(yaml.safe_dump({"experts": experts}, sort_keys=False))
  print(f"wrote {args.out} with {len(experts)} experts: {list(experts)}")


if __name__ == "__main__":
  main()
