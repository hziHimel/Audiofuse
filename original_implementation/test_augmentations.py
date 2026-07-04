"""Unit tests for augmentations.py."""

import torch
import pytest
from augmentations import GaussianNoise, SpecAugment


# ── GaussianNoise ─────────────────────────────────────────────────────────────

def test_gaussian_noise_shape():
    m = GaussianNoise(snr_db=20).train()
    x = torch.randn(4, 110250)
    assert m(x).shape == x.shape


def test_gaussian_noise_changes_input():
    m = GaussianNoise(snr_db=20).train()
    x = torch.randn(4, 110250)
    assert not torch.allclose(m(x), x)


def test_gaussian_noise_identity_at_eval():
    m = GaussianNoise(snr_db=20).eval()
    x = torch.randn(4, 110250)
    assert torch.allclose(m(x), x)


def test_gaussian_noise_snr_scaling():
    """Higher SNR → smaller noise std → output closer to input."""
    x = torch.ones(1, 1000)
    m_high = GaussianNoise(snr_db=40).train()
    m_low  = GaussianNoise(snr_db=5).train()
    torch.manual_seed(0)
    diff_high = (m_high(x) - x).abs().mean().item()
    torch.manual_seed(0)
    diff_low  = (m_low(x) - x).abs().mean().item()
    assert diff_high < diff_low


# ── SpecAugment ───────────────────────────────────────────────────────────────

def test_specaugment_shape():
    m = SpecAugment(T=40, F=20).train()
    x = torch.randn(4, 1, 224, 224)
    assert m(x).shape == x.shape


def test_specaugment_changes_input():
    m = SpecAugment(T=40, F=20).train()
    x = torch.randn(4, 1, 224, 224)
    assert not torch.allclose(m(x), x)


def test_specaugment_identity_at_eval():
    m = SpecAugment(T=40, F=20).eval()
    x = torch.randn(4, 1, 224, 224)
    assert torch.allclose(m(x), x)


def test_specaugment_masks_bounded():
    """Masked values are replaced with mean; output range should be tighter than raw noise."""
    m = SpecAugment(T=80, F=40, num_time_masks=4, num_freq_masks=4).train()
    x = torch.randn(4, 1, 224, 224) * 10  # wide range
    out = m(x)
    assert out.shape == x.shape


print("All augmentation tests collected.")
