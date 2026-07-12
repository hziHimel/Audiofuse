"""Tests for modality_dropout.py"""
import torch
from modality_dropout import apply_modality_dropout


def _batch(B=64):
    spec = torch.randn(B, 1, 16, 16) + 5.0   # offset so zeros are distinguishable
    wave = torch.randn(B, 100) + 5.0
    return spec, wave


def test_p_zero_no_change():
    spec, wave = _batch()
    s, w = apply_modality_dropout(spec, wave, p=0.0)
    assert torch.equal(s, spec)
    assert torch.equal(w, wave)


def test_shapes_preserved():
    spec, wave = _batch()
    s, w = apply_modality_dropout(spec, wave, p=0.5)
    assert s.shape == spec.shape
    assert w.shape == wave.shape


def test_never_both_dropped():
    spec, wave = _batch(256)
    s, w = apply_modality_dropout(spec, wave, p=1.0)
    spec_zero = (s.flatten(1) == 0).all(dim=1)
    wave_zero = (w.flatten(1) == 0).all(dim=1)
    # no sample has BOTH modalities zeroed
    assert not (spec_zero & wave_zero).any()


def test_p_one_exactly_one_dropped():
    spec, wave = _batch(256)
    s, w = apply_modality_dropout(spec, wave, p=1.0)
    spec_zero = (s.flatten(1) == 0).all(dim=1)
    wave_zero = (w.flatten(1) == 0).all(dim=1)
    # with p=1 every sample drops exactly one modality
    assert (spec_zero ^ wave_zero).all()


def test_does_not_mutate_input():
    spec, wave = _batch()
    spec_ref = spec.clone()
    wave_ref = wave.clone()
    apply_modality_dropout(spec, wave, p=1.0)
    assert torch.equal(spec, spec_ref)
    assert torch.equal(wave, wave_ref)


def test_reproducible_with_generator():
    spec, wave = _batch()
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s1, w1 = apply_modality_dropout(spec, wave, p=0.5, generator=g1)
    s2, w2 = apply_modality_dropout(spec, wave, p=0.5, generator=g2)
    assert torch.equal(s1, s2)
    assert torch.equal(w1, w2)


def test_roughly_p_fraction_dropped():
    spec, wave = _batch(2000)
    s, w = apply_modality_dropout(spec, wave, p=0.5)
    spec_zero = (s.flatten(1) == 0).all(dim=1)
    wave_zero = (w.flatten(1) == 0).all(dim=1)
    frac = (spec_zero | wave_zero).float().mean().item()
    assert 0.4 < frac < 0.6   # ~50% of samples lose a modality
