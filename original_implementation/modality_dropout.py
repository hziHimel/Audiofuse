"""
Modality dropout for gradient dominance (Direction 3.1, remedy #3).

During training, each sample independently has (with probability p) exactly one
of its two modalities zeroed — either the spectrogram or the waveform, chosen at
random. This forces the fusion head to make correct predictions even when a
modality is absent, so BOTH branches must be independently informative. In
particular, samples where the waveform is dropped force the model to classify
from the ViT alone, directly counteracting ViT laziness.

At least one modality is always present (both are never dropped together). The
zeroed representation matches our branch-ablation convention (a zero input).
"""

import torch


def apply_modality_dropout(spec, wave, p: float, generator=None):
    """Per-sample modality dropout. With probability p a sample loses exactly
    one modality (50/50 spec vs wave). Returns (spec_out, wave_out) with the
    same shapes; inputs are not modified in place."""
    B = spec.size(0)
    dev = spec.device
    drop = torch.rand(B, generator=generator, device=dev) < p          # which samples drop
    drop_wave = torch.rand(B, generator=generator, device=dev) < 0.5   # among dropped: True→wave, False→spec

    spec_out = spec.clone()
    wave_out = wave.clone()

    drop_spec_mask = drop & (~drop_wave)
    drop_wave_mask = drop & drop_wave

    spec_out[drop_spec_mask] = 0.0
    wave_out[drop_wave_mask] = 0.0
    return spec_out, wave_out
