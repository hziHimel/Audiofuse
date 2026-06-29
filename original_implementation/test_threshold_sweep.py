"""
Tests for the threshold sweep utility in train_pytorch.py.
Run: python -m pytest test_threshold_sweep.py -v
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from train_pytorch import sweep_threshold


def test_perfect_separation():
    # Probabilities perfectly separated — any threshold in (0.4, 0.6) should give F1=1.0
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thresh, f1 = sweep_threshold(y_true, y_prob)
    assert f1 == pytest.approx(1.0), f"Expected F1=1.0, got {f1}"
    assert 0.1 <= thresh <= 0.9


def test_optimal_threshold_beats_05():
    # Imbalanced: 3:1 negative-to-positive; optimum should shift threshold below 0.5
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 75 + [1] * 25)
    # Positive class scores between 0.3–0.6, negative between 0.1–0.5 — overlapping
    y_prob = np.concatenate([
        rng.uniform(0.1, 0.5, 75),
        rng.uniform(0.3, 0.6, 25),
    ])
    thresh, f1_opt = sweep_threshold(y_true, y_prob)
    from sklearn.metrics import f1_score
    f1_at_05 = f1_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)
    assert f1_opt >= f1_at_05, "Sweep should find threshold at least as good as 0.5"


def test_sweep_range():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    thresh, _ = sweep_threshold(y_true, y_prob, lo=0.1, hi=0.9, step=0.05)
    assert 0.1 <= thresh <= 0.9


def test_all_positive_predictions():
    y_true = np.array([1, 1, 1, 0])
    y_prob = np.array([0.9, 0.95, 0.85, 0.1])
    thresh, f1 = sweep_threshold(y_true, y_prob)
    assert f1 > 0.0
