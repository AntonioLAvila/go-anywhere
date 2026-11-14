"""Policy models: LocoTransformer (all policies) and TeacherEnsemble (distillation)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from go_anywhere.rl.context import get_visible_context
from go_anywhere.terrains.tiles import NUM_SKILLS, Skill


class _Block(nn.Module):
  """Pre-LN transformer block built from export-friendly primitives."""

  def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.attn = nn.MultiheadAttention(
      d_model, n_heads, dropout=dropout, batch_first=True
    )
    self.ln2 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(
      nn.Linear(d_model, d_model * mlp_ratio),
      nn.GELU(),
      nn.Linear(d_model * mlp_ratio, d_model),
    )
    self.drop = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.ln1(x)
    a, _ = self.attn(h, h, h, need_weights=False)
    x = x + self.drop(a)
    x = x + self.drop(self.mlp(self.ln2(x)))
    return x


def _key(name: str) -> str:
  return name.replace(".", "_")


class LocoTransformer(MLPModel):
  """Transformer policy over a history window.

  Tokens: one per proprio history step (``proprio_group``, ``[B, H, D]``), one
  per strided terrain frame (``terrain_group``), and one per 1-D context group
  (command, privileged, task_id). The readout is the newest proprio token,
  concatenated with the normalized current frame (bypass) before the MLP head.
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256,),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    d_model: int = 192,
    n_layers: int = 4,
    n_heads: int = 4,
    mlp_ratio: int = 2,
    dropout: float = 0.0,
    proprio_group: str = "proprio_hist",
    terrain_group: str = "terrain_hist",
    terrain_stride: int = 5,
    terrain_frames: int = 10,
    bypass_current_frame: bool = True,
    grad_checkpointing: bool = True,
  ) -> None:
    # Plain attributes needed by the hooks MLPModel.__init__ calls.
    self._d_model = d_model
    self._proprio_group = proprio_group
    self._terrain_group = terrain_group
    self._terrain_stride = terrain_stride
    self._terrain_frames = terrain_frames
    self._bypass = bypass_current_frame
    self._group_shapes: dict[str, tuple[int, ...]] = {}
    self.grad_checkpointing = grad_checkpointing
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=False,
      distribution_cfg=distribution_cfg,
    )
    self.obs_normalization = obs_normalization
    self.normalizers = nn.ModuleDict()
    self.embed = nn.ModuleDict()
    self.modality = nn.ParameterDict()
    self.time_embed = nn.ParameterDict()
    for g, shape in self._group_shapes.items():
      d = shape[-1]
      self.normalizers[_key(g)] = (
        EmpiricalNormalization(d) if obs_normalization else nn.Identity()
      )
      self.embed[_key(g)] = nn.Linear(d, d_model)
      self.modality[_key(g)] = nn.Parameter(torch.zeros(d_model))
      if len(shape) == 2:
        self.time_embed[_key(g)] = nn.Parameter(0.02 * torch.randn(shape[0], d_model))
    self.blocks = nn.ModuleList(
      [_Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
    )
    self.ln_f = nn.LayerNorm(d_model)
    self.apply(self._init_weights)

  @staticmethod
  def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
      nn.init.normal_(m.weight, std=0.02)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  # MLPModel hooks.

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ):
    groups = list(obs_groups[obs_set])
    flat = 0
    for g in groups:
      shape = tuple(obs[g].shape[1:])
      if len(shape) not in (1, 2):
        raise ValueError(
          f"LocoTransformer supports [B, D] or [B, H, D] groups; got {obs[g].shape} for '{g}'"
        )
      self._group_shapes[g] = shape
      flat += int(torch.tensor(shape).prod())
    return groups, flat

  def _get_latent_dim(self) -> int:
    bypass = 0
    if self._bypass:
      for g, shape in self._group_shapes.items():
        bypass += shape[-1]  # Latest frame of sequences, full vector of 1-D groups.
    return self._d_model + bypass

  # Normalization.

  def _normalize(self, g: str, x: torch.Tensor) -> torch.Tensor:
    norm = self.normalizers[_key(g)]
    if isinstance(norm, nn.Identity):
      return x
    d = x.shape[-1]
    return norm(x.reshape(-1, d)).reshape(x.shape)

  def update_normalization(self, obs: TensorDict) -> None:
    if not self.obs_normalization:
      return
    for g, shape in self._group_shapes.items():
      x = obs[g]
      if len(shape) == 2:
        x = x[:, -1]
      norm = self.normalizers[_key(g)]
      assert isinstance(norm, EmpiricalNormalization)
      norm.update(x)

  # Token layout.

  def _proprio_indices(self, h: int) -> torch.Tensor:
    visible = get_visible_context() if self.training else None
    k = h if visible is None else max(1, min(h, int(visible)))
    return torch.arange(h - k, h)

  def _terrain_indices(self, h: int) -> torch.Tensor:
    idx = h - 1 - self._terrain_stride * torch.arange(self._terrain_frames)
    idx = idx[idx >= 0]
    return idx.flip(0)

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    tokens: list[torch.Tensor] = []
    bypass: list[torch.Tensor] = []
    readout_index = -1
    for g, shape in self._group_shapes.items():
      x = self._normalize(g, obs[g])
      k = _key(g)
      if len(shape) == 2:
        h = x.shape[1]
        if g == self._terrain_group:
          idx = self._terrain_indices(h).to(x.device)
        else:
          idx = self._proprio_indices(h).to(x.device)
        frames = x[:, idx]
        emb = self.embed[k](frames) + self.time_embed[k][idx] + self.modality[k]
        if g == self._proprio_group:
          readout_index = sum(t.shape[1] for t in tokens) + emb.shape[1] - 1
        tokens.append(emb)
        if self._bypass:
          bypass.append(x[:, -1])
      else:
        emb = (self.embed[k](x) + self.modality[k]).unsqueeze(1)
        tokens.append(emb)
        if self._bypass:
          bypass.append(x)
    seq = torch.cat(tokens, dim=1)
    use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
    for blk in self.blocks:
      if use_ckpt:
        # Store only block inputs; recompute activations in backward (memory for compute).
        seq = checkpoint(blk, seq, use_reentrant=False)
      else:
        seq = blk(seq)
    seq = self.ln_f(seq)
    read = seq[:, readout_index] if readout_index >= 0 else seq.mean(dim=1)
    if bypass:
      return torch.cat([read] + bypass, dim=-1)
    return read

  # Export.

  def flat_input_dim(self) -> int:
    return self.obs_dim

  def split_flat(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for g, shape in self._group_shapes.items():
      n = int(torch.tensor(shape).prod())
      out[g] = x[:, offset : offset + n].reshape(x.shape[0], *shape)
      offset += n
    return out

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxLocoTransformer(self, verbose)

  def as_jit(self) -> nn.Module:
    return _OnnxLocoTransformer(self, False)


def _safe_deepcopy(model: nn.Module) -> nn.Module:
  """Deep-copy a model whose distribution caches non-leaf tensors from the last
  forward pass (rsl_rl distributions keep ``_distribution``/``_alpha``/``_beta``)."""
  dist = getattr(model, "distribution", None)
  stash: dict[str, Any] = {}
  if dist is not None:
    for name, value in list(vars(dist).items()):
      if (
        name.startswith("_")
        and not isinstance(value, (nn.Module, nn.Parameter))
        and (torch.is_tensor(value) or hasattr(value, "mean"))
      ):
        stash[name] = value
        setattr(dist, name, None)
  try:
    return copy.deepcopy(model)
  finally:
    for name, value in stash.items():
      setattr(dist, name, value)


class _OnnxLocoTransformer(nn.Module):
  """Takes one flat observation vector (groups concatenated in set order)."""

  is_recurrent: bool = False

  def __init__(self, model: LocoTransformer, verbose: bool):
    super().__init__()
    self.verbose = verbose
    self.model = _safe_deepcopy(model).eval()
    self.input_size = model.flat_input_dim()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    obs = self.model.split_flat(x)
    return self.model(obs)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass


# Teacher ensemble.


def strip_none_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
  """Drop None-valued optional entries that ``MLPModel`` must not receive."""
  cfg = dict(cfg)
  for opt in ("cnn_cfg", "distribution_cfg"):
    if cfg.get(opt) is None:
      cfg.pop(opt, None)
  if cfg.get("rnn_type") is None:
    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      cfg.pop(opt, None)
  return cfg


class TeacherEnsemble(nn.Module):
  """Frozen per-skill experts routed by the ``task_id`` one-hot.

  Manifest (YAML)::

    experts:
      flat:
        checkpoint: logs/rsl_rl/go2_expert_flat/<run>/model_5000.pt
        agent_cfg: logs/rsl_rl/go2_expert_flat/<run>/params/agent.yaml  # optional
        skills: [flat, drops]   # skill names this expert answers for

  Every expert must have been trained with an actor observation set equal to
  the distillation ``teacher`` set (same groups, same order).
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    manifest: str = "configs/experts.yaml",
    routing_group: str = "task_id",
    **_: Any,
  ) -> None:
    super().__init__()
    self.routing_group = routing_group
    self.output_dim = output_dim
    self.experts = nn.ModuleList()
    self.names: list[str] = []
    skill_to_expert = torch.full((NUM_SKILLS,), -1, dtype=torch.long)
    path = Path(manifest)
    if not path.exists():
      raise FileNotFoundError(
        f"Expert manifest {path} not found. Create it with scripts/make_experts_manifest.py."
      )
    with open(path) as f:
      entries = (yaml.safe_load(f) or {}).get("experts", {})
    if not entries:
      raise ValueError(f"Expert manifest {path} lists no experts.")
    for name, entry in entries.items():
      ckpt = Path(entry["checkpoint"])
      agent_yaml = Path(entry.get("agent_cfg") or ckpt.parent / "params" / "agent.yaml")
      with open(agent_yaml) as f:
        # mjlab's dump_yaml emits python/tuple tags; FullLoader accepts them.
        agent = yaml.load(f, Loader=yaml.FullLoader)
      actor_cfg = copy.deepcopy(strip_none_model_cfg(agent["actor"]))
      cls = resolve_callable(actor_cfg.pop("class_name"))
      model = cls(obs, obs_groups, obs_set, output_dim, **actor_cfg)
      state = torch.load(ckpt, map_location="cpu", weights_only=False)
      sd = state.get("actor_state_dict") or state.get("student_state_dict")
      model.load_state_dict(sd, strict=True)
      model.eval()
      for p in model.parameters():
        p.requires_grad_(False)
      idx = len(self.experts)
      self.experts.append(model)
      self.names.append(name)
      for skill in entry.get("skills", [name]):
        skill_to_expert[int(Skill[skill.upper()])] = idx
    unassigned = [s.name.lower() for s in Skill if skill_to_expert[int(s)] < 0]
    if unassigned:
      print(
        f"[WARN] No expert listed for skills {unassigned}; '{self.names[0]}' labels "
        "them. List every skill in the manifest to make the routing explicit."
      )
      skill_to_expert[skill_to_expert < 0] = 0
    self.register_buffer("skill_to_expert", skill_to_expert)
    self.distribution = None

  def forward(
    self, obs, masks=None, hidden_state=None, stochastic_output: bool = False
  ) -> torch.Tensor:
    outs = torch.stack([e(obs) for e in self.experts], dim=0)  # [E, B, A]
    skill = obs[self.routing_group].argmax(dim=-1)
    idx = self.skill_to_expert[skill]
    return outs[idx, torch.arange(outs.shape[1], device=outs.device)]

  # MLPModel surface used by rsl_rl's Distillation.
  def reset(self, dones=None, hidden_state=None) -> None:
    pass

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None) -> None:
    pass

  def update_normalization(self, obs) -> None:
    pass

  def train(self, mode: bool = True):
    return super().train(False)
