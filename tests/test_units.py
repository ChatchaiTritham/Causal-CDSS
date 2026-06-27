"""Focused unit tests for pure/deterministic functions in basics_cdss.metrics.

These tests target hand-computable cases so the expected values are derived by
hand, not echoed from the implementation. They cover calibration, harm-aware,
selective-prediction, and classification-performance metrics.
"""

import os
import sys

import numpy as np
import pytest

# Make the src/ layout importable without installing the package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from basics_cdss.metrics import calibration, coverage_risk, harm, performance


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def test_brier_score_perfect_and_known_value():
    # Perfect predictions -> Brier score exactly 0.
    y_true = np.array([1, 0, 1, 0])
    assert calibration.brier_score(y_true, y_true.astype(float)) == 0.0

    # Hand-computed: mean of (0.9-1)^2,(0.1-0)^2,(0.8-1)^2 = (0.01+0.01+0.04)/3
    y_true2 = np.array([1, 0, 1])
    y_prob2 = np.array([0.9, 0.1, 0.8])
    expected = (0.01 + 0.01 + 0.04) / 3.0
    assert calibration.brier_score(y_true2, y_prob2) == pytest.approx(expected)


def test_ece_perfectly_calibrated_extremes_is_zero():
    # Probabilities at 0/1 that exactly match labels -> 0 calibration error.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    assert calibration.expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.0)


def test_ece_empty_input_returns_zero():
    assert calibration.expected_calibration_error(np.array([]), np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# Harm-aware metrics
# --------------------------------------------------------------------------- #
def test_weighted_harm_loss_known_value():
    # One high-risk error (w=10) and one low-risk error (w=1) over 4 samples.
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])  # errors at index 1 (low) and 2 (high)
    risk_tiers = np.array(["high", "low", "high", "low"])
    loss = harm.weighted_harm_loss(y_true, y_pred, risk_tiers)
    # (w_low*1 + w_high*1) / N = (1 + 10) / 4
    assert loss == pytest.approx((1.0 + 10.0) / 4.0)


def test_weighted_harm_loss_zero_when_no_errors():
    y_true = np.array([1, 0, 1, 0])
    risk_tiers = np.array(["high", "low", "high", "low"])
    assert harm.weighted_harm_loss(y_true, y_true.copy(), risk_tiers) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Coverage-risk / selective prediction
# --------------------------------------------------------------------------- #
def test_abstention_rate_known_fraction():
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    # Below 0.5: 0.3 and 0.2 -> 2 of 5.
    assert coverage_risk.abstention_rate(y_prob, threshold=0.5) == pytest.approx(2 / 5)


def test_aurc_linear_risk_trapezoid():
    # Risk rising linearly from 0 to 0.2 over coverage 0..1 -> area = 0.1.
    coverages = np.array([0.0, 0.5, 1.0])
    risks = np.array([0.0, 0.1, 0.2])
    assert coverage_risk.area_under_risk_coverage_curve(coverages, risks) == pytest.approx(0.1)


def test_coverage_risk_curve_monotone_coverage():
    # Coverage must be non-increasing as the acceptance threshold rises.
    y_true = np.array([1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    coverages, risks, thresholds = coverage_risk.coverage_risk_curve(
        y_true, y_prob, n_thresholds=20
    )
    assert coverages[0] == pytest.approx(1.0)  # threshold 0 accepts all
    assert np.all(np.diff(coverages) <= 1e-9)


# --------------------------------------------------------------------------- #
# Classification performance
# --------------------------------------------------------------------------- #
def test_confusion_matrix_counts_and_prevalence():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = performance.confusion_matrix(y_true, y_pred)
    # TN=1 (idx0), FP=1 (idx1), TP=2 (idx2,3), FN=1 (idx4)
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 1, 2)
    assert cm.total == 5
    assert cm.prevalence == pytest.approx(3 / 5)


def test_performance_metrics_hand_computed():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = performance.compute_performance_metrics(y_true, y_pred)
    # TP=2,FP=1,FN=1,TN=1
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.specificity == pytest.approx(1 / 2)
    assert m.f1_score == pytest.approx(2 / 3)


def test_bootstrap_ci_seed42_reproducible():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.2, 0.85, 0.15])
    out_a = performance.bootstrap_confidence_interval(
        y_true, y_pred, y_prob, metric="f1_score", n_bootstrap=200, seed=42
    )
    out_b = performance.bootstrap_confidence_interval(
        y_true, y_pred, y_prob, metric="f1_score", n_bootstrap=200, seed=42
    )
    assert out_a == pytest.approx(out_b)  # deterministic under fixed seed
    point, lower, upper = out_a
    assert lower <= point <= upper
