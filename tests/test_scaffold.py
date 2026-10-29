"""
Scaffold tests that keep CI green until the real ones land.

Tests to add:
- data generators
- tensor shapes
- one tiny overfit check
"""

import torch

from deja_vu_tfm.utils import get_device, seed_everything


def test_device_is_detected() -> None:
  assert isinstance(get_device(), torch.device)


def test_seeding_repeats_exactly() -> None:
  seed_everything(0)
  a = torch.rand(3)
  seed_everything(0)
  b = torch.rand(3)
  assert torch.equal(a, b)
