"""Unit tests for WaveformResCNN in train_pytorch_rescnn.py."""

import torch
import pytest
from train_pytorch_rescnn import ResBlock1d, WaveformResCNN, AudioFuseRes, C


def test_resblock_output_shape():
    block = ResBlock1d(1, 64, kernel_size=16, stride=4, padding=8)
    x = torch.randn(4, 1, 110250)
    out = block(x)
    assert out.shape[0] == 4
    assert out.shape[1] == 64


def test_resblock_skip_adds():
    """Output should differ from conv-only path (skip connection fires)."""
    block = ResBlock1d(1, 64, kernel_size=16, stride=4, padding=8)
    x = torch.randn(2, 1, 110250)
    out = block(x)
    conv_only = torch.nn.functional.relu(block.conv(x))
    assert not torch.allclose(out, conv_only)


def test_waveformrescnn_output_shape():
    model = WaveformResCNN()
    x = torch.randn(4, C.WAV_LEN)
    out = model(x)
    assert out.shape == (4, 64)


def test_waveformrescnn_output_nonneg():
    """Output passes through ReLU so must be >= 0."""
    model = WaveformResCNN()
    x = torch.randn(4, C.WAV_LEN)
    out = model(x)
    assert out.min().item() >= 0.0


def test_audiofuseres_output_shape():
    model = AudioFuseRes()
    spec = torch.randn(4, 1, 224, 224)
    wave = torch.randn(4, C.WAV_LEN)
    logits = model(spec, wave)
    assert logits.shape == (4,)


print("All rescnn tests collected.")
