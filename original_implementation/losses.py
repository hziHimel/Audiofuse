"""
Loss functions for AudioFuse extension experiments.
"""

import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss — down-weights easy negatives to focus on hard examples.

    Replaces BCE+pos_weight for imbalanced datasets. alpha=0.25 and gamma=2.0
    are the values from the original RetinaNet paper (Lin et al. 2017).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def bce_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 0.1,
) -> torch.Tensor:
    """BCE with label smoothing — softens 0/1 targets to reduce overconfidence.

    eps=0.1 is the standard value from Szegedy et al. (2016). Smoothed targets
    become eps/2 for negatives and 1 - eps/2 for positives.
    """
    smooth_targets = targets * (1.0 - eps) + (1.0 - targets) * eps
    return F.binary_cross_entropy_with_logits(logits, smooth_targets)
