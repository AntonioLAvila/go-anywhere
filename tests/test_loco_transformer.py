import torch
from tensordict import TensorDict

from go_anywhere.rl.context import set_visible_context
from go_anywhere.rl.models import LocoTransformer

B, H, D_P, HT, D_T, D_C = 6, 25, 42, 50, 176, 7
GROUPS = {"actor": ["proprio_hist", "terrain_hist", "command"]}


def _obs(seed=0):
  g = torch.Generator().manual_seed(seed)
  return TensorDict(
    {
      "proprio_hist": torch.randn(B, H, D_P, generator=g),
      "terrain_hist": torch.randn(B, HT, D_T, generator=g),
      "command": torch.randn(B, D_C, generator=g),
    },
    batch_size=[B],
  )


def _model(**kw):
  return LocoTransformer(
    _obs(),
    GROUPS,
    "actor",
    12,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    **kw,
  )


def test_forward_shape_and_param_count():
  m = _model()
  out = m(_obs())
  assert out.shape == (B, 12)
  n = sum(p.numel() for p in m.parameters())
  assert 1.0e6 < n < 2.5e6, n


def test_update_normalization_and_stochastic():
  m = _model()
  m.update_normalization(_obs(1))
  a = m(_obs(), stochastic_output=True)
  assert a.shape == (B, 12)
  assert m.output_std.shape[-1] == 12


def test_visible_context_changes_output_in_train_mode():
  m = _model().train()
  set_visible_context(None)
  full = m(_obs())
  set_visible_context(8)
  short = m(_obs())
  set_visible_context(None)
  assert not torch.allclose(full, short)
  m.eval()
  set_visible_context(8)
  assert torch.allclose(m(_obs()), full, atol=1e-6)
  set_visible_context(None)


def test_onnx_wrapper_matches_model():
  m = _model().eval()
  obs = _obs()
  flat = torch.cat(
    [obs["proprio_hist"].flatten(1), obs["terrain_hist"].flatten(1), obs["command"]],
    dim=1,
  )
  wrapper = m.as_onnx(False)
  assert wrapper.get_dummy_inputs()[0].shape[1] == flat.shape[1]
  assert torch.allclose(wrapper(flat), m(obs), atol=1e-6)


def test_onnx_export_runs(tmp_path):
  m = _model().eval()
  wrapper = m.as_onnx(False)
  torch.onnx.export(
    wrapper,
    wrapper.get_dummy_inputs(),
    str(tmp_path / "policy.onnx"),
    input_names=wrapper.input_names,
    output_names=wrapper.output_names,
    opset_version=18,
    dynamo=False,
  )
  assert (tmp_path / "policy.onnx").exists()


def test_onnx_wrapper_after_grad_forward():
  """Export must work after a training forward (distribution caches non-leaf tensors)."""
  m = _model().train()
  obs = _obs()
  a = m(obs, stochastic_output=True)
  m.get_output_log_prob(a).sum().backward()
  wrapper = m.as_onnx(False)
  flat = torch.cat(
    [obs["proprio_hist"].flatten(1), obs["terrain_hist"].flatten(1), obs["command"]],
    dim=1,
  )
  m.eval()
  assert torch.allclose(wrapper(flat), m(obs), atol=1e-6)
  assert m.distribution is not None and m.output_std.shape[-1] == 12
