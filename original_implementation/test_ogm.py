"""Tests for ogm.py"""
from ogm import true_class_confidence, modulation_coeffs


def test_confidence_perfect_predictions():
    # y=1 with p=1.0 and y=0 with p=0.0 → full confidence each
    conf = true_class_confidence([1.0, 0.0], [1, 0])
    assert abs(conf - 2.0) < 1e-9


def test_confidence_wrong_predictions():
    conf = true_class_confidence([0.0, 1.0], [1, 0])
    assert abs(conf - 0.0) < 1e-9


def test_equal_confidence_no_modulation():
    k_spec, k_wave = modulation_coeffs(5.0, 5.0, alpha=0.5)
    # wave >= spec branch taken; ratio=1 → tanh(0)=0 → k_wave=1
    assert abs(k_spec - 1.0) < 1e-9
    assert abs(k_wave - 1.0) < 1e-9


def test_wave_dominates_damps_wave_only():
    k_spec, k_wave = modulation_coeffs(conf_spec=2.0, conf_wave=10.0, alpha=0.5)
    assert k_spec == 1.0            # weaker branch untouched
    assert 0.0 < k_wave < 1.0       # dominant branch damped


def test_spec_dominates_damps_spec_only():
    k_spec, k_wave = modulation_coeffs(conf_spec=10.0, conf_wave=2.0, alpha=0.5)
    assert k_wave == 1.0
    assert 0.0 < k_spec < 1.0


def test_larger_ratio_more_damping():
    _, k_small = modulation_coeffs(5.0, 6.0, alpha=0.5)
    _, k_large = modulation_coeffs(5.0, 50.0, alpha=0.5)
    assert k_large < k_small         # bigger dominance → stronger damping


def test_alpha_zero_disables():
    k_spec, k_wave = modulation_coeffs(2.0, 100.0, alpha=0.0)
    assert abs(k_spec - 1.0) < 1e-9
    assert abs(k_wave - 1.0) < 1e-9


def test_coeffs_bounded():
    # coefficient range is [0, 1]: saturates to ~0 for extreme dominance,
    # =1 for the weaker branch. Realistic ratios keep it well above 0.
    for cs, cw in [(1.0, 100.0), (100.0, 1.0), (3.0, 3.1), (1.0, 1.0)]:
        k_spec, k_wave = modulation_coeffs(cs, cw, alpha=1.0)
        assert 0.0 <= k_spec <= 1.0
        assert 0.0 <= k_wave <= 1.0


def test_realistic_ratio_stays_positive():
    # batch-summed confidence ratios are small in practice; k stays well > 0
    k_spec, k_wave = modulation_coeffs(conf_spec=18.0, conf_wave=24.0, alpha=0.5)
    assert k_spec == 1.0
    assert k_wave > 0.5
