"""Checks for scripts/msm_bounds.py (marginal sensitivity model bounds)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import msm_bounds as mb  # noqa: E402
import unmeasured_confounding as uc  # noqa: E402


def _cohort(seed=3, n=4000):
    return uc.simulate(uc.DOMAINS["sepsis"], n, np.random.default_rng(seed))


def test_lambda_one_collapses_to_hajek_ipw():
    d = _cohort()
    e = mb.logit_fit(np.column_stack([d["age"], d["z"], d["w"]]), d["a"])
    lo, hi, _, _ = mb.ate_bounds(d, 1.0, e)
    t = d["a"] == 1
    w1, w0 = 1 / e[t], 1 / (1 - e[~t])
    hajek = (w1 * d["y"][t]).sum() / w1.sum() - (w0 * d["y"][~t]).sum() / w0.sum()
    assert np.isclose(lo, hajek) and np.isclose(hi, hajek)


def test_bounds_nested_in_lambda():
    d = _cohort()
    prev = None
    for lam in (1.0, 1.2, 1.5, 2.0, 3.0):
        lo, hi, _, _ = mb.ate_bounds(d, lam)
        if prev is not None:
            assert lo <= prev[0] + 1e-12 and hi >= prev[1] - 1e-12
        prev = (lo, hi)


def test_arm_bounds_are_extreme_over_admissible_weights():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) < 0.4).astype(float)
    base = 1 + rng.exponential(1.0, 200)
    lam = 1.7
    dn, up = mb.arm_bounds(y, base, lam)
    for _ in range(300):
        f = rng.uniform(1 / lam, lam, 200)
        w = 1 + (base - 1) * f
        m = (w * y).sum() / w.sum()
        assert dn - 1e-12 <= m <= up + 1e-12


def test_smallest_lambda_is_one_at_point_estimate():
    d = _cohort()
    lo, _, _, _ = mb.ate_bounds(d, 1.0)
    assert mb.smallest_lambda(d, lo) == 1.0


def test_iscore90_width_when_covered():
    import sim_review_checks as sr
    s = sr.iscore90(np.array([0.0]), np.array([0.1]), 0.0)
    assert np.isclose(s[0], 2 * sr.Z90 * 0.1)
