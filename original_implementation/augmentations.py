"""
On-the-fly audio augmentation utilities for AudioFuse (Direction 1.4).

Applied during training only; val set is never augmented.
All ops are pure PyTorch tensor ops so they run on MPS/CUDA without CPU round-trips.

Augmentations implemented:
  - Gaussian noise on waveform (SNR-controlled)
  - SpecAugment: random time masking + frequency masking on spectrogram

Not implemented here (require librosa, too slow for on-the-fly):
  - Time stretch (±10%), pitch shift (±2 st) — apply offline as a preprocessing step if needed.
"""

import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """Add zero-mean Gaussian noise to waveform. std scales with signal RMS for SNR control."""
    def __init__(self, snr_db: float = 20.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return wave
        rms = wave.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        snr_linear = 10 ** (self.snr_db / 20.0)
        noise_std = rms / snr_linear
        return wave + torch.randn_like(wave) * noise_std


class SpecAugment(nn.Module):
    """SpecAugment: random time + frequency masking on log-Mel spectrogram.

    Masks are filled with the per-sample mean value (less disruptive than zeros).
    T: max time steps to mask per stripe. F: max freq bins to mask per stripe.
    num_time_masks, num_freq_masks: number of independent stripes.
    """
    def __init__(self, T: int = 40, F: int = 20,
                 num_time_masks: int = 2, num_freq_masks: int = 2):
        super().__init__()
        self.T = T
        self.F = F
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (B, C, H, W) — H=freq bins, W=time steps."""
        if not self.training:
            return spec
        B, C, H, W = spec.shape
        out = spec.clone()
        mean = spec.mean(dim=(-1, -2), keepdim=True)

        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.T + 1, (B,))
            t0 = torch.randint(0, max(W - self.T, 1), (B,))
            for b in range(B):
                out[b, :, :, t0[b]:t0[b] + t[b]] = mean[b]

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.F + 1, (B,))
            f0 = torch.randint(0, max(H - self.F, 1), (B,))
            for b in range(B):
                out[b, :, f0[b]:f0[b] + f[b], :] = mean[b]

        return out
