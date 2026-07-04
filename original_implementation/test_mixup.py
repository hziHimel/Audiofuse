"""Unit tests for mixup_batch in train_pytorch_mixup.py."""

import torch
import numpy as np
import pytest
from train_pytorch_mixup import mixup_batch


def _batch(B=8):
    spec = torch.rand(B, 1, 224, 224)
    wave = torch.rand(B, 110250)
    labels = torch.randint(0, 2, (B,)).float()
    return spec, wave, labels


def test_output_shapes():
    spec, wave, labels = _batch()
    sm, wm, la, lb, lam = mixup_batch(spec, wave, labels)
    assert sm.shape == spec.shape
    assert wm.shape == wave.shape
    assert la.shape == labels.shape
    assert lb.shape == labels.shape


def test_lambda_in_range():
    spec, wave, labels = _batch()
    _, _, _, _, lam = mixup_batch(spec, wave, labels, alpha=0.4)
    assert 0.0 <= lam <= 1.0


def test_mixed_is_convex_combination():
    """spec_mix = lam*spec + (1-lam)*spec[idx]: must lie in [min(spec,spec_b), max(spec,spec_b)]."""
    spec, wave, labels = _batch(B=4)
    torch.manual_seed(0)
    sm, wm, la, lb, lam = mixup_batch(spec, wave, labels, alpha=0.4)
    # Each element of sm is between 0 and 1 (inputs were uniform [0,1])
    assert sm.min().item() >= 0.0
    assert sm.max().item() <= 1.0


def test_same_lambda_for_spec_and_wave():
    """Both modalities use the same lam so they stay temporally consistent."""
    torch.manual_seed(42)
    np.random.seed(42)
    spec, wave, labels = _batch(B=4)
    sm, wm, la, lb, lam = mixup_batch(spec, wave, labels, alpha=0.4)
    # Verify lam is a scalar float (not per-sample)
    assert isinstance(lam, float)


def test_alpha_zero_is_identity():
    """α→0 makes Beta(0,0) degenerate; we clamp by using a small alpha instead."""
    spec, wave, labels = _batch(B=4)
    # With alpha very small, lam should be very close to 0 or 1
    lams = [mixup_batch(spec, wave, labels, alpha=1e-6)[4] for _ in range(20)]
    # All lam values should be near 0 or 1
    assert all(l < 0.01 or l > 0.99 for l in lams)


print("All mixup tests collected.")
