"""Checks for scripts/v110_extensions.py (sharp bounds, regularised q bridge, propositions)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unmeasured_confounding as uc  # noqa: E402
import v110_extensions as v  # noqa: E402


def _cohort(seed=3, n=4000):
    d = uc.simulate(uc.DOMAINS["sepsis"], n, np.random.default_rng(seed))
    return np.column_stack([d["age"], d["z"], d["w"]]), d["a"], d["y"]


def test_sharp_bounds_collapse_at_one_and_nest():
    X, a, y = _cohort()
    b = v.sharp_bounds(X, a, y, (1.0, 1.2, 1.5, 2.0))
    assert np.isclose(b[0][0], b[0][1])
    for (l0, h0), (l1, h1) in zip(b, b[1:]):
        assert l1 <= l0 + 1e-12 and h1 >= h0 - 1e-12


def test_sharp_bounds_narrower_than_tan():
    X, a, y = _cohort()
    for (ls, hs), (lt, ht) in zip(v.sharp_bounds(X, a, y, (1.3, 1.5, 2.0)), v.tan_bounds(X, a, y, (1.3, 1.5, 2.0))):
        assert hs - ls < ht - lt


def test_sharp_tilt_is_extreme_binary():
    # the tilted mean of a Bernoulli(mu) under likelihood ratios in [c, c L^2] is maximised by
    # putting the large ratio on y = 1
    mu, L = 0.3, 1.7
    best = mu * L ** 2 / (mu * L ** 2 + 1 - mu)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        r1, r0 = rng.uniform(1, L ** 2, 2)
        assert mu * r1 / (mu * r1 + (1 - mu) * r0) <= best + 1e-12


def test_regularised_q_equals_plain_when_solved():
    d = uc.simulate(uc.DOMAINS["ards"], 4000, np.random.default_rng(11))
    n = len(d["a"])
    F = np.column_stack([np.ones(n), d["z"], d["age"]])
    G = np.column_stack([np.ones(n), d["w"], d["age"]])
    qp, ip = v.q_bridge(F, G, d["a"], "plain")
    qr, ir = v.q_bridge(F, G, d["a"], "reg")
    assert max(i[0] for i in ip) < v.TOL
    assert np.allclose(qp, qr)


def test_proposition1_limit():
    rows = v.prop1_check(np.random.default_rng(1))
    # n = 2e6; the fourth design has small Var(T | R), so allow 1% relative error
    assert all(abs(r["diff"]) < max(0.005, 0.01 * abs(r["formula"])) for r in rows)
