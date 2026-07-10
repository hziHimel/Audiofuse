"""
Gradient-flow instrumentation for diagnosing gradient dominance (Direction 3.1).

Computes per-branch gradient L2 norms after loss.backward(). Used to empirically
demonstrate that the waveform CNN dominates gradients early in joint training
while the ViT branch receives vanishing signal.

The same instrumentation is reused to show how each gradient-dominance remedy
(OGM-GE, modality dropout, etc.) rebalances the per-branch gradient magnitudes.
"""

import torch
import torch.nn as nn


def branch_grad_norm(module: nn.Module) -> float:
    """Total L2 norm of all gradients in a module (call after loss.backward()).

    Returns 0.0 if no parameter has a gradient (e.g. branch is frozen)."""
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def branch_grad_norms(spec_branch: nn.Module, wave_branch: nn.Module) -> dict:
    """Per-branch gradient norms plus their ratio (wave / spec).

    A ratio >> 1 indicates the wave branch dominates the gradient signal."""
    g_spec = branch_grad_norm(spec_branch)
    g_wave = branch_grad_norm(wave_branch)
    ratio = g_wave / g_spec if g_spec > 1e-12 else float("inf")
    return {"grad_norm_spec": g_spec, "grad_norm_wave": g_wave, "grad_ratio_wave_over_spec": ratio}
