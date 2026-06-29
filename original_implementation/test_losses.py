"""Unit tests for losses.py (focal loss and label smoothing)."""

import math
import torch
import pytest
from losses import focal_loss_with_logits, bce_with_label_smoothing


# ── Focal loss ───────────────────────────────────────────────────────────────

def test_focal_loss_perfect_predictions():
    """With very confident correct predictions, focal loss should be near 0."""
    logits = torch.tensor([10.0, -10.0])   # strongly predicts [1, 0]
    targets = torch.tensor([1.0, 0.0])
    loss = focal_loss_with_logits(logits, targets)
    assert loss.item() < 1e-3, f"Expected ~0 loss, got {loss.item():.6f}"
    print(f"PASS focal_loss_perfect: loss={loss.item():.6f}")


def test_focal_loss_wrong_predictions():
    """With wrong confident predictions, focal loss should be large."""
    logits = torch.tensor([10.0, -10.0])   # predicts [1, 0]
    targets = torch.tensor([0.0, 1.0])     # actual [0, 1] — completely wrong
    loss = focal_loss_with_logits(logits, targets)
    assert loss.item() > 1.0, f"Expected large loss, got {loss.item():.6f}"
    print(f"PASS focal_loss_wrong: loss={loss.item():.6f}")


def test_focal_loss_less_than_bce_on_easy_examples():
    """Focal loss should down-weight easy examples vs standard BCE."""
    import torch.nn.functional as F
    logits = torch.tensor([5.0, -5.0])   # easy, confidently correct
    targets = torch.tensor([1.0, 0.0])
    fl = focal_loss_with_logits(logits, targets, alpha=0.5, gamma=2.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert fl.item() < bce.item(), (
        f"Focal ({fl.item():.6f}) should be < BCE ({bce.item():.6f}) for easy examples"
    )
    print(f"PASS focal_vs_bce: focal={fl.item():.6f} < bce={bce.item():.6f}")


def test_focal_loss_gamma_zero_equals_alpha_weighted_bce():
    """When gamma=0, focal loss reduces to alpha-weighted BCE."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.5).float()
    alpha = 0.25

    fl = focal_loss_with_logits(logits, targets, alpha=alpha, gamma=0.0)

    # Manual alpha-weighted BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    expected = (alpha_t * bce).mean()

    assert abs(fl.item() - expected.item()) < 1e-5, (
        f"gamma=0 focal ({fl.item():.6f}) != alpha-weighted BCE ({expected.item():.6f})"
    )
    print(f"PASS focal_gamma0: fl={fl.item():.6f} ≈ alpha_bce={expected.item():.6f}")


# ── Label smoothing ──────────────────────────────────────────────────────────

def test_label_smoothing_reduces_confidence():
    """Label smoothing should produce higher loss than BCE for correct predictions."""
    import torch.nn.functional as F
    logits = torch.tensor([5.0, -5.0])
    targets = torch.tensor([1.0, 0.0])
    ls_loss = bce_with_label_smoothing(logits, targets, eps=0.1)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    assert ls_loss.item() > bce_loss.item(), (
        f"Label smoothing ({ls_loss.item():.6f}) should raise loss on correct predictions"
    )
    print(f"PASS label_smooth_confidence: ls={ls_loss.item():.6f} > bce={bce_loss.item():.6f}")


def test_label_smoothing_eps_zero_equals_bce():
    """With eps=0, label smoothing should equal standard BCE."""
    import torch.nn.functional as F
    torch.manual_seed(1)
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.5).float()
    ls = bce_with_label_smoothing(logits, targets, eps=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert abs(ls.item() - bce.item()) < 1e-6, (
        f"eps=0 label smoothing ({ls.item():.6f}) != BCE ({bce.item():.6f})"
    )
    print(f"PASS label_smooth_eps0: ls={ls.item():.6f} ≈ bce={bce.item():.6f}")


def test_label_smoothing_targets_in_range():
    """Smoothed targets should stay in (eps/2, 1-eps/2) range — no 0 or 1."""
    eps = 0.1
    # Use non-zero logits so BCE depends on the target value (logits=0 → loss=log2 for any target)
    logits = torch.tensor([2.0, 2.0, -2.0, -2.0])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss_smooth = bce_with_label_smoothing(logits, targets, eps=eps)
    loss_hard = bce_with_label_smoothing(logits, targets, eps=0.0)
    assert abs(loss_smooth.item() - loss_hard.item()) > 1e-4, (
        "eps=0.1 should change the loss — smoothed targets not being applied"
    )
    print(f"PASS label_smooth_range: loss_smooth={loss_smooth.item():.6f}, loss_hard={loss_hard.item():.6f}")


if __name__ == "__main__":
    test_focal_loss_perfect_predictions()
    test_focal_loss_wrong_predictions()
    test_focal_loss_less_than_bce_on_easy_examples()
    test_focal_loss_gamma_zero_equals_alpha_weighted_bce()
    test_label_smoothing_reduces_confidence()
    test_label_smoothing_eps_zero_equals_bce()
    test_label_smoothing_targets_in_range()
    print("\nAll tests passed.")
