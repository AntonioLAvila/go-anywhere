"""rsl_rl configuration dataclasses for go-anywhere.

They extend mjlab's dataclasses so the stock ``train``/``play`` CLI (tyro) can
override every field, e.g. ``--agent.actor.n-layers 6``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

LOCO_TRANSFORMER_CLS = "go_anywhere.rl.models:LocoTransformer"
TEACHER_ENSEMBLE_CLS = "go_anywhere.rl.models:TeacherEnsemble"
DAGGER_CLS = "go_anywhere.rl.algorithms:DaggerDistillation"
FINETUNE_PPO_CLS = "go_anywhere.rl.algorithms:FinetunePPO"
SYMMETRY_FUNC = "go_anywhere.rl.symmetry:mirror_lr"
WANDB_PROJECT = "go-anywhere"

ACTION_CLIP = 20.0
"""Bound on policy actions. One unit is ~0.15 rad of joint target, so this is far
outside any joint range; it only stops a wide action distribution from commanding
targets that destabilise the simulation."""

GAUSSIAN = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


@dataclass
class LocoTransformerCfg(RslRlModelCfg):
  """Transformer over proprio history tokens + terrain tokens + context tokens."""

  class_name: str = LOCO_TRANSFORMER_CLS
  hidden_dims: tuple[int, ...] = (256,)
  """Head MLP after the transformer readout."""
  activation: str = "elu"
  obs_normalization: bool = True
  d_model: int = 192
  n_layers: int = 4
  n_heads: int = 4
  mlp_ratio: int = 2
  dropout: float = 0.0
  proprio_group: str = "proprio_hist"
  terrain_group: str = "terrain_hist"
  terrain_stride: int = 5
  """Take every k-th frame of the terrain history (50 Hz buffer -> 10 Hz tokens)."""
  terrain_frames: int = 10
  bypass_current_frame: bool = True
  grad_checkpointing: bool = True
  """Recompute transformer activations in backward: ~5x less update memory for
  ~30% more update compute. Turn off on large-memory GPUs if update time matters."""


@dataclass
class TeacherEnsembleCfg(RslRlModelCfg):
  class_name: str = TEACHER_ENSEMBLE_CLS
  manifest: str = "configs/experts.yaml"
  routing_group: str = "task_id"
  obs_normalization: bool = False


@dataclass
class GoAnywherePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = FINETUNE_PPO_CLS
  entropy_coef: float = 0.005
  num_mini_batches: int = 16
  """Transformer update memory scales with minibatch size (~3.4 MB per augmented
  sample without checkpointing, ~0.6 MB with). 16 keeps 4096 envs under 16 GB."""
  rnd_cfg: dict[str, Any] | None = None
  symmetry_cfg: dict[str, Any] | None = None
  critic_warmup_iters: int = 0
  """Updates with the actor frozen (Rudin et al. 2025 finetune precaution)."""
  bc_coef: float = 0.0
  """Optional BC regularizer toward the initial (distilled) policy."""


def symmetry_cfg(
  use_augmentation: bool = True, mirror_loss_coeff: float = 0.0
) -> dict[str, Any]:
  return {
    "data_augmentation_func": SYMMETRY_FUNC,
    "use_data_augmentation": use_augmentation,
    "use_mirror_loss": mirror_loss_coeff > 0,
    "mirror_loss_coeff": mirror_loss_coeff,
  }


def rnd_cfg(weight: float = 0.5) -> dict[str, Any]:
  return {
    "weight": weight,
    "predictor_hidden_dims": (128, 128),
    "target_hidden_dims": (256, 256),
    "state_normalization": True,
    "reward_normalization": True,
    "learning_rate": 1e-3,
    "num_outputs": 64,
  }


@dataclass
class GoAnywhereOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  actor: RslRlModelCfg = field(
    default_factory=lambda: LocoTransformerCfg(distribution_cfg=dict(GAUSSIAN))
  )
  critic: RslRlModelCfg = field(default_factory=LocoTransformerCfg)
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=GoAnywherePpoAlgorithmCfg)
  logger: Literal["wandb", "tensorboard"] = "wandb"
  wandb_project: str = WANDB_PROJECT
  clip_actions: float | None = ACTION_CLIP
  init_from: str | None = None
  """Checkpoint to warm-start the actor from (expert or distilled student)."""
  init_std: float | None = None
  """Override the actor's initial action std after warm-start."""


@dataclass
class DaggerAlgorithmCfg:
  class_name: str = DAGGER_CLS
  num_learning_epochs: int = 2
  gradient_length: int = 15
  learning_rate: float = 5e-4
  max_grad_norm: float | None = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"
  beta_start: float = 0.5
  """Probability of executing the teacher action at the start (DAgger beta)."""
  beta_end: float = 0.0
  beta_decay_updates: int = 500
  action_noise_std: float = 0.05
  """Gaussian noise on executed actions (Rudin et al. 2025)."""
  rnd_cfg: dict[str, Any] | None = None
  symmetry_cfg: dict[str, Any] | None = None


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  logger: Literal["wandb", "tensorboard"] = "wandb"
  wandb_project: str = WANDB_PROJECT
  clip_actions: float | None = ACTION_CLIP
  student: RslRlModelCfg = field(
    default_factory=lambda: LocoTransformerCfg(distribution_cfg=dict(GAUSSIAN))
  )
  teacher: RslRlModelCfg = field(default_factory=TeacherEnsembleCfg)
  algorithm: DaggerAlgorithmCfg = field(default_factory=DaggerAlgorithmCfg)
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "student": ("proprio_hist", "terrain_hist", "command"),
      # Must equal the experts' actor set. Routing reads ``task_id`` from the env
      # observations directly, so it is not part of the teacher set.
      "teacher": (
        "proprio_hist",
        "terrain_hist",
        "terrain_priv",
        "command",
        "privileged",
      ),
    }
  )
