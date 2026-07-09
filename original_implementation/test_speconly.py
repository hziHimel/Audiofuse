"""Tests for train_pytorch_speconly.py"""
import torch
import pytest
from train_pytorch_speconly import SpecClassifier
from train_pytorch import Config

C = Config()


def make_spec(B=4):
    return torch.randn(B, 1, C.IMG_SIZE, C.IMG_SIZE)


def test_spec_classifier_output_shape():
    model = SpecClassifier()
    model.eval()
    with torch.no_grad():
        out = model(make_spec())
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_spec_classifier_single_sample():
    model = SpecClassifier()
    model.eval()
    with torch.no_grad():
        out = model(make_spec(B=1))
    assert out.shape == (1,)


def test_spec_classifier_output_range():
    model = SpecClassifier()
    model.eval()
    with torch.no_grad():
        logits = model(make_spec())
    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_spec_classifier_gradients():
    model = SpecClassifier()
    spec = make_spec()
    out = model(spec)
    out.mean().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_spec_branch_weights_extractable():
    model = SpecClassifier()
    state = model.state_dict()
    spec_keys = [k for k in state if k.startswith("spec_branch.")]
    assert len(spec_keys) > 0, "spec_branch weights not found in state dict"
    head_keys = [k for k in state if k.startswith("head.")]
    assert len(head_keys) > 0, "head weights not found in state dict"
