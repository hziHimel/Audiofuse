"""Tests for train_pytorch_waveonly.py"""
import torch
import pytest
from train_pytorch_waveonly import WaveClassifier, WaveDataset
from train_pytorch import Config

C = Config()


def make_wave(B=4):
    return torch.randn(B, C.WAV_LEN)


def test_wave_classifier_output_shape():
    model = WaveClassifier()
    model.eval()
    with torch.no_grad():
        out = model(make_wave())
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_wave_classifier_single_sample():
    model = WaveClassifier()
    model.eval()
    with torch.no_grad():
        out = model(make_wave(B=1))
    assert out.shape == (1,)


def test_wave_classifier_output_range():
    model = WaveClassifier()
    model.eval()
    with torch.no_grad():
        logits = model(make_wave())
    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_wave_classifier_gradients():
    model = WaveClassifier()
    wave = make_wave()
    out = model(wave)
    loss = out.mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_wave_branch_weights_extractable():
    model = WaveClassifier()
    state = model.state_dict()
    wave_keys = [k for k in state if k.startswith("wave_branch.")]
    assert len(wave_keys) > 0, "wave_branch weights not found in state dict"
    head_keys = [k for k in state if k.startswith("head.")]
    assert len(head_keys) > 0, "head weights not found in state dict"
