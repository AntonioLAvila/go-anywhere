import torch

from go_anywhere.assets.go2 import BODY_WEIGHT_N
from go_anywhere.mdp.contact import (
  IMPACT_ONSET_BW,
  IMPACT_TERMINATE_BW,
  box_penetration_depth,
)


def test_box_penetration_depth():
  centers = torch.tensor([[[0.0, 0.0, 0.5]]])  # [B=1, K=1, 3]
  half = torch.tensor([[[0.1, 1.0, 0.5]]])
  pts = torch.tensor(
    [[[0.0, 0.0, 0.5], [0.05, 0.0, 0.5], [0.3, 0.0, 0.5], [0.0, 0.0, 1.2]]]
  )
  d = box_penetration_depth(pts, centers, half)[0, :, 0]
  assert d.tolist() == [0.1, 0.05000000074505806, 0.0, 0.0] or torch.allclose(
    d, torch.tensor([0.1, 0.05, 0.0, 0.0]), atol=1e-6
  )


def test_body_weight_thresholds_are_sane_for_go2():
  assert 120 < BODY_WEIGHT_N < 180
  assert IMPACT_ONSET_BW < IMPACT_TERMINATE_BW
