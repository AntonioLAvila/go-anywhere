import copy

import torch
import yaml
from tensordict import TensorDict

from go_anywhere.rl.models import LocoTransformer, TeacherEnsemble
from go_anywhere.terrains import NUM_SKILLS, Skill

B = 5
GROUPS = {"teacher": ["proprio_hist", "command", "task_id"]}


def _obs(skill_ids):
  tid = torch.nn.functional.one_hot(torch.tensor(skill_ids), NUM_SKILLS).float()
  return TensorDict(
    {
      "proprio_hist": torch.randn(B, 25, 10),
      "command": torch.randn(B, 7),
      "task_id": tid,
    },
    batch_size=[B],
  )


def _save_expert(tmp_path, name, seed):
  torch.manual_seed(seed)
  cfg = dict(
    class_name="go_anywhere.rl.models:LocoTransformer",
    hidden_dims=[32],
    activation="elu",
    obs_normalization=True,
    d_model=16,
    n_layers=1,
    n_heads=2,
    mlp_ratio=1,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    cnn_cfg=None,
    rnn_type=None,
  )
  kwargs = {
    k: copy.deepcopy(v)
    for k, v in cfg.items()
    if k not in ("class_name", "cnn_cfg", "rnn_type")
  }
  m = LocoTransformer(_obs([0] * B), GROUPS, "teacher", 12, **kwargs)
  run = tmp_path / name / "run"
  (run / "params").mkdir(parents=True)
  (run / "params" / "agent.yaml").write_text(yaml.safe_dump({"actor": cfg}))
  torch.save(
    {"actor_state_dict": m.state_dict(), "iter": 0, "infos": {}}, run / "model_1.pt"
  )
  return m, run / "model_1.pt"


def test_routes_by_task_id(tmp_path):
  m_a, ck_a = _save_expert(tmp_path, "a", 1)
  m_b, ck_b = _save_expert(tmp_path, "b", 2)
  manifest = tmp_path / "experts.yaml"
  manifest.write_text(
    yaml.safe_dump(
      {
        "experts": {
          "a": {"checkpoint": str(ck_a), "skills": ["flat", "rough"]},
          "b": {"checkpoint": str(ck_b), "skills": ["gaps"]},
        }
      }
    )
  )
  ens = TeacherEnsemble(_obs([0] * B), GROUPS, "teacher", 12, manifest=str(manifest))
  obs = _obs(
    [
      int(Skill.FLAT),
      int(Skill.GAPS),
      int(Skill.ROUGH),
      int(Skill.GAPS),
      int(Skill.STAIRS),
    ]
  )
  out = ens(obs)
  ea, eb = m_a.eval()(obs), m_b.eval()(obs)
  assert torch.allclose(out[0], ea[0]) and torch.allclose(out[2], ea[2])
  assert torch.allclose(out[1], eb[1]) and torch.allclose(out[3], eb[3])
  assert torch.allclose(out[4], ea[4])  # Unassigned skill falls back to expert 0.
  assert not any(p.requires_grad for p in ens.parameters())
