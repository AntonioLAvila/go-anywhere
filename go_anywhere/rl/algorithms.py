"""Algorithms: DAgger-style distillation and finetune PPO."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO, Distillation
from tensordict import TensorDict


class DaggerDistillation(Distillation):
  """rsl_rl Distillation + beta-mixed rollouts + executed-action noise.

  rsl_rl's Distillation already lets the student act and labels with the
  teacher (DAgger with beta = 0). This adds a decaying probability ``beta`` of
  executing the teacher action instead (classic DAgger) and Gaussian noise on
  the executed action (Rudin et al. 2025) so the student is robust to the
  exploration noise RL finetuning adds later. Stored observations/labels are
  unchanged: the loss is always student(obs) vs teacher(obs).
  """

  def __init__(
    self,
    *args,
    beta_start: float = 0.5,
    beta_end: float = 0.0,
    beta_decay_updates: int = 500,
    action_noise_std: float = 0.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.beta_decay_updates = max(1, beta_decay_updates)
    self.action_noise_std = action_noise_std

  @property
  def beta(self) -> float:
    t = min(1.0, self.num_updates / self.beta_decay_updates)
    return self.beta_start + (self.beta_end - self.beta_start) * t

  def act(self, obs: TensorDict) -> torch.Tensor:
    student_actions = self.student(obs, stochastic_output=True).detach()
    teacher_actions = self.teacher(obs).detach()
    self.transition.actions = student_actions
    self.transition.privileged_actions = teacher_actions
    self.transition.observations = obs
    executed = student_actions
    if self.beta > 0:
      use_teacher = torch.rand(executed.shape[0], 1, device=executed.device) < self.beta
      executed = torch.where(use_teacher, teacher_actions, executed)
    if self.action_noise_std > 0:
      executed = executed + self.action_noise_std * torch.randn_like(executed)
    return executed

  def update(self) -> dict[str, float]:
    out = super().update()
    out["beta"] = self.beta
    return out


class FinetunePPO(PPO):
  """PPO with a critic warm-up (actor frozen) and an optional BC regularizer.

  With ``critic_warmup_iters=0`` and ``bc_coef=0`` this is plain PPO, so it is
  also used for the experts.
  """

  def __init__(
    self, *args, critic_warmup_iters: int = 0, bc_coef: float = 0.0, **kwargs
  ) -> None:
    super().__init__(*args, **kwargs)
    self.critic_warmup_iters = critic_warmup_iters
    self.bc_coef = bc_coef
    self.bc_reference: nn.Module | None = None
    self._num_updates = 0

  def set_bc_reference(self, model: nn.Module) -> None:
    self.bc_reference = copy.deepcopy(model).eval()
    for p in self.bc_reference.parameters():
      p.requires_grad_(False)

  def _set_actor_frozen(self, frozen: bool) -> None:
    for p in self.actor.parameters():
      p.requires_grad_(not frozen)

  def _bc_step(self) -> float:
    if self.bc_reference is None or self.bc_coef <= 0:
      return 0.0
    total, count = 0.0, 0
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, 1):
      with torch.no_grad():
        target = self.bc_reference(batch.observations)
      pred = self.actor(batch.observations)
      loss = self.bc_coef * nn.functional.mse_loss(pred, target)
      self.optimizer.zero_grad()
      loss.backward()
      if self.max_grad_norm:
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.optimizer.step()
      total += loss.item()
      count += 1
    return total / max(count, 1)

  def update(self) -> dict[str, float]:
    warm = self._num_updates < self.critic_warmup_iters
    self._set_actor_frozen(warm)
    bc_loss = 0.0 if warm else self._bc_step()
    out = super().update()
    self._set_actor_frozen(False)
    self._num_updates += 1
    out["critic_warmup"] = float(warm)
    if self.bc_coef > 0:
      out["bc"] = bc_loss
    return out
