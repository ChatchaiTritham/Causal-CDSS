"""Checks for scripts/default_loss_audit.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import default_loss_audit as dl  # noqa: E402


def test_interval_score_is_width_when_covered():
    s = dl.iscore(np.array([0.0]), np.array([0.1]), 0.05)
    assert np.isclose(s[0], 2 * dl.Z * 0.1)


def test_interval_score_penalises_miss():
    covered = dl.iscore(np.array([0.0]), np.array([0.1]), 0.0)[0]
    missed = dl.iscore(np.array([0.0]), np.array([0.1]), 1.0)[0]
    assert missed > covered + 30


def test_auc_bounds():
    assert dl.auc(np.array([1.0, 2.0]), np.array([0, 1])) == 1.0
    assert dl.auc(np.array([2.0, 1.0]), np.array([0, 1])) == 0.0
