"""Tests for train_pytorch_pretrained_init.py"""
import torch
import pytest
from train_pytorch import AudioFuse, Config
from train_pytorch_pretrained_init import (
    load_pretrained_branches, set_branches_frozen, make_balanced_sampler,
    make_optimizer, FREEZE_EPOCHS, BRANCH_LR, HEAD_LR
)
from train_pytorch_speconly import SpecClassifier
from train_pytorch_waveonly import WaveClassifier
import pandas as pd
import numpy as np
import tempfile
import os

C = Config()


def make_dummy_checkpoints(tmp_dir):
    """Save dummy SpecClassifier and WaveClassifier checkpoints."""
    spec_model = SpecClassifier()
    wave_model = WaveClassifier()
    spec_ckpt = os.path.join(tmp_dir, "spec.pt")
    wave_ckpt = os.path.join(tmp_dir, "wave.pt")
    torch.save(spec_model.state_dict(), spec_ckpt)
    torch.save(wave_model.state_dict(), wave_ckpt)
    return spec_ckpt, wave_ckpt


def test_load_pretrained_branches_no_error():
    with tempfile.TemporaryDirectory() as tmp:
        spec_ckpt, wave_ckpt = make_dummy_checkpoints(tmp)
        model = AudioFuse()
        load_pretrained_branches(model, spec_ckpt, wave_ckpt)  # should not raise


def test_branch_weights_actually_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        spec_model = SpecClassifier()
        wave_model = WaveClassifier()
        spec_ckpt = os.path.join(tmp, "spec.pt")
        wave_ckpt = os.path.join(tmp, "wave.pt")
        torch.save(spec_model.state_dict(), spec_ckpt)
        torch.save(wave_model.state_dict(), wave_ckpt)

        fusion_model = AudioFuse()
        load_pretrained_branches(fusion_model, spec_ckpt, wave_ckpt)

        # ViT branch weights should match
        for (k1, v1), (k2, v2) in zip(
            spec_model.spec_branch.state_dict().items(),
            fusion_model.spec_branch.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"ViT weight mismatch at {k1}"

        # CNN branch weights should match
        for (k1, v1), (k2, v2) in zip(
            wave_model.wave_branch.state_dict().items(),
            fusion_model.wave_branch.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"CNN weight mismatch at {k1}"


def test_set_branches_frozen():
    model = AudioFuse()
    set_branches_frozen(model, frozen=True)
    for p in model.spec_branch.parameters():
        assert not p.requires_grad
    for p in model.wave_branch.parameters():
        assert not p.requires_grad
    for p in model.head.parameters():
        assert p.requires_grad  # head should still be trainable


def test_set_branches_unfrozen():
    model = AudioFuse()
    set_branches_frozen(model, frozen=True)
    set_branches_frozen(model, frozen=False)
    for p in model.spec_branch.parameters():
        assert p.requires_grad
    for p in model.wave_branch.parameters():
        assert p.requires_grad


def test_make_optimizer_phase1_head_only():
    model = AudioFuse()
    set_branches_frozen(model, frozen=True)
    opt = make_optimizer(model, phase2=False)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == HEAD_LR


def test_make_optimizer_phase2_differential_lr():
    model = AudioFuse()
    opt = make_optimizer(model, phase2=True)
    assert len(opt.param_groups) == 3
    assert opt.param_groups[0]["lr"] == BRANCH_LR  # spec branch
    assert opt.param_groups[1]["lr"] == BRANCH_LR  # wave branch
    assert opt.param_groups[2]["lr"] == HEAD_LR    # head


def test_balanced_sampler_length():
    df = pd.DataFrame({
        "label": [0]*30 + [1]*10,
        "filepath": ["x.wav"]*40,
        "spec_path": ["x.npy"]*40,
    })
    sampler = make_balanced_sampler(df)
    assert len(sampler) == 40
