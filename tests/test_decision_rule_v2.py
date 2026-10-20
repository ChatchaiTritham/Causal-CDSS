"""Unit checks for the redesigned decision rule (scripts/decision_rule_v2.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decision_rule_v2 as v2  # noqa: E402
import review_round2 as r2  # noqa: E402
import unmeasured_confounding as uc  # noqa: E402


def test_ar_projection_contains_wald_point_under_strong_valid_proxies():
    p = dict(v2.WEAK_BASE, sev_to_z=1.0, sev_to_w=1.0)
    d = r2.sim_general(p, 4000, np.random.default_rng(1))
    lo, hi = v2.ar_projection(d)
    est, _ = uc.proximal_se(d)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= est <= hi


def test_v2_always_reports_dr():
    R = dict(F=np.array([5.0, 50.0, 500.0]), ok=np.array([1, 0, 1]),
             dr=np.array([0.1, 0.1, 0.1]), tsls=np.array([0.1, 0.1, 0.9]),
             dr_se=np.array([0.01] * 3), tsls_se=np.array([0.01] * 3),
             gamma_el=np.zeros(3), delta_el=np.zeros(3))
    dec, flagged = v2.decide(R, dict(gamma=0.0, delta=0.0))
    assert set(dec["v2 (DR default + flags)"]) <= {"dr", "dr_flag"}
    assert flagged.tolist() == [True, True, True]


def test_design_sets_differ():
    a, b = v2.design(2027), v2.design(2031)
    assert a[0] != b[0] and a[0] == r2.lhs_design()[0]
