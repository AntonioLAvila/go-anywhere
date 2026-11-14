"""Runners: PPO (experts, finetune) and distillation, both with ONNX export."""

from __future__ import annotations

from typing import Any

import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner

from go_anywhere.rl.models import strip_none_model_cfg


def _export_onnx(runner: MjlabOnPolicyRunner, path: str) -> None:
  policy_dir, filename, _ = runner._get_export_paths(path)
  # mjlab's ONNX metadata helper assumes an observation group named "actor"; our
  # groups are composable, so no metadata is attached.
  try:
    runner.export_policy_to_onnx(str(policy_dir), filename)
  except Exception as e:  # A failed export must not kill a long training run.
    print(f"[WARN] ONNX export failed (training continues): {e}")


def _migrate_student_keys(loaded: dict) -> dict:
  if "actor_state_dict" not in loaded and "student_state_dict" in loaded:
    loaded = dict(loaded)
    loaded["actor_state_dict"] = loaded["student_state_dict"]
  return loaded


class GoAnywhereOnPolicyRunner(MjlabOnPolicyRunner):
  """PPO runner with warm-start (``init_from``), std override and ONNX export."""

  env: RslRlVecEnvWrapper

  def __init__(
    self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
  ) -> None:
    self.init_from: str | None = train_cfg.get("init_from")
    self.init_std: float | None = train_cfg.get("init_std")
    super().__init__(env, train_cfg, log_dir, device)
    if self.init_from:
      self.warm_start(self.init_from)
    if self.init_std is not None:
      self.set_action_std(self.init_std)

  def warm_start(self, path: str) -> None:
    loaded = torch.load(path, map_location=self.device, weights_only=False)
    loaded = _migrate_student_keys(loaded)
    self.alg.load(loaded, {"actor": True}, strict=True)
    print(f"[INFO] Warm-started actor from {path}")
    if getattr(self.alg, "bc_coef", 0.0) > 0:
      self.alg.set_bc_reference(self.alg.actor)

  def set_action_std(self, std: float) -> None:
    dist = getattr(self.alg.actor, "distribution", None)
    if dist is None:
      return
    with torch.no_grad():
      if hasattr(dist, "std_param"):
        dist.std_param.fill_(std)
      elif hasattr(dist, "log_std_param"):
        dist.log_std_param.fill_(float(torch.log(torch.tensor(std))))

  def save(self, path: str, infos: Any = None) -> None:
    super().save(path, infos)
    _export_onnx(self, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if "actor_state_dict" not in loaded and "student_state_dict" in loaded:
      # `play` on a distillation checkpoint: it only holds the student (actor).
      self.alg.load(_migrate_student_keys(loaded), {"actor": True}, strict)
      return loaded["infos"]
    return super().load(path, load_cfg, strict, map_location)


class GoAnywhereDistillationRunner(DistillationRunner, MjlabOnPolicyRunner):
  """rsl_rl DistillationRunner with mjlab checkpoint handling and ONNX export.

  The teacher is a ``TeacherEnsemble`` that loads its own expert checkpoints,
  so ``teacher_loaded`` is set after construction.
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
  ) -> None:
    for key in ("student", "teacher"):
      train_cfg[key] = strip_none_model_cfg(train_cfg[key])
    super().__init__(env, train_cfg, log_dir, device)
    self.alg.teacher_loaded = True

  def save(self, path: str, infos: Any = None) -> None:
    super().save(path, infos)
    _export_onnx(self, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if load_cfg is not None and "actor" in load_cfg and "student" not in load_cfg:
      load_cfg = {"student": load_cfg["actor"]}
    if load_cfg is None:
      load_cfg = {"student": True, "optimizer": True, "iteration": True}
    infos = super().load(path, load_cfg, strict, map_location)
    self.alg.teacher_loaded = True
    return infos
