"""Tests for grad_flow.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_flow import branch_grad_norm, branch_grad_norms
from train_pytorch import AudioFuse, Config

C = Config()


def _backward_once(model):
    spec = torch.randn(2, 1, C.IMG_SIZE, C.IMG_SIZE)
    wave = torch.randn(2, C.WAV_LEN)
    labels = torch.tensor([0.0, 1.0])
    logits = model(spec, wave)
    F.binary_cross_entropy_with_logits(logits, labels).backward()


def test_grad_norm_zero_before_backward():
    m = nn.Linear(4, 1)
    assert branch_grad_norm(m) == 0.0


def test_grad_norm_positive_after_backward():
    m = nn.Linear(4, 1)
    out = m(torch.randn(3, 4)).sum()
    out.backward()
    assert branch_grad_norm(m) > 0.0


def test_grad_norm_frozen_branch_is_zero():
    model = AudioFuse()
    for p in model.spec_branch.parameters():
        p.requires_grad = False
    _backward_once(model)
    # frozen branch accumulates no grad
    assert branch_grad_norm(model.spec_branch) == 0.0
    assert branch_grad_norm(model.wave_branch) > 0.0


def test_branch_grad_norms_keys():
    model = AudioFuse()
    _backward_once(model)
    d = branch_grad_norms(model.spec_branch, model.wave_branch)
    assert set(d.keys()) == {"grad_norm_spec", "grad_norm_wave", "grad_ratio_wave_over_spec"}
    assert d["grad_norm_spec"] > 0.0
    assert d["grad_norm_wave"] > 0.0


def test_ratio_matches_manual():
    model = AudioFuse()
    _backward_once(model)
    d = branch_grad_norms(model.spec_branch, model.wave_branch)
    expected = d["grad_norm_wave"] / d["grad_norm_spec"]
    assert abs(d["grad_ratio_wave_over_spec"] - expected) < 1e-9


def test_ratio_inf_when_spec_zero():
    model = AudioFuse()
    for p in model.spec_branch.parameters():
        p.requires_grad = False
    _backward_once(model)
    d = branch_grad_norms(model.spec_branch, model.wave_branch)
    assert d["grad_ratio_wave_over_spec"] == float("inf")
