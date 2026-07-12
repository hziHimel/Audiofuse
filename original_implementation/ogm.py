"""
On-the-fly Gradient Modulation (OGM-GE) for gradient dominance (Direction 3.1).

Adapts Peng et al. (CVPR 2022) "Balanced Multimodal Learning via On-the-fly
Gradient Modulation" to AudioFuse's ViT + 1D-CNN dual branch.

Core idea: at each step, estimate how much each branch contributes to the
correct prediction, then damp the *dominant* branch's gradient by a coefficient
k = 1 - tanh(alpha * relu(ratio - 1)), leaving the weaker branch untouched. GE
(Generalization Enhancement) optionally adds Gaussian noise to the modulated
gradients to compensate for their reduced dynamics.

Adaptation note: the original OGM decomposes a single linear fusion classifier
into per-modality logit contributions. AudioFuse uses a non-linear 2-layer
fusion head, so we instead estimate each branch's standalone predictive
confidence via zeroed-branch forward passes (analogous to our branch ablation),
which is a faithful, interpretable proxy for per-branch contribution.
"""

import math


def true_class_confidence(probs, labels) -> float:
    """Sum of confidence in the TRUE class: p if y==1 else (1-p). Higher = the
    branch predicts the correct label more confidently."""
    conf = 0.0
    for p, y in zip(probs, labels):
        conf += p if y == 1 else (1.0 - p)
    return conf


def modulation_coeffs(conf_spec: float, conf_wave: float, alpha: float):
    """Per-branch gradient modulation coefficients (k_spec, k_wave).

    Only the dominant branch (higher confidence) is damped; the weaker branch
    keeps k=1. Returns values in (0, 1]. alpha controls damping strength;
    alpha=0 disables modulation (both k=1)."""
    eps = 1e-8
    if conf_wave >= conf_spec:
        ratio = conf_wave / (conf_spec + eps)          # wave dominates
        k_wave = 1.0 - math.tanh(alpha * max(0.0, ratio - 1.0))
        k_spec = 1.0
    else:
        ratio = conf_spec / (conf_wave + eps)          # spec dominates
        k_spec = 1.0 - math.tanh(alpha * max(0.0, ratio - 1.0))
        k_wave = 1.0
    return k_spec, k_wave
