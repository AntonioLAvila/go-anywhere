import pytest
import torch


def pytest_collection_modifyitems(config, items):
  if torch.cuda.is_available():
    return
  skip = pytest.mark.skip(reason="CUDA not available")
  for item in items:
    if "gpu" in item.keywords:
      item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"
