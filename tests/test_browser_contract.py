"""Contracts the educational browser demo copies in JavaScript."""

import torch

from torchxrayvision.models import op_norm
from torchxrayvision.utils import normalize
import numpy as np


def test_operating_point_maps_to_one_half():
    thresholds = torch.tensor([[0.2, 0.8]])
    out = op_norm(thresholds.clone(), thresholds.clone())
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_op_norm_endpoints():
    thresholds = torch.tensor([[0.25]])
    zero = op_norm(torch.zeros(1, 1), thresholds)
    one = op_norm(torch.ones(1, 1), thresholds)
    assert torch.allclose(zero, torch.zeros(1, 1))
    assert torch.allclose(one, torch.ones(1, 1))


def test_normalize_8bit_endpoints():
    img = np.array([[0, 255]], dtype=np.uint8)
    scaled = normalize(img, 255)
    assert scaled.dtype == np.float32
    assert scaled[0, 0] == -1024
    assert scaled[0, 1] == 1024
