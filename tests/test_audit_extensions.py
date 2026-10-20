"""Checks for the v1.0.3 audit extensions against closed-form or algebraic facts."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import proximal_audit_extensions as px  # noqa: E402
import unmeasured_confounding as uc  # noqa: E402


def _cohort(seed=1, n=4000):
    return uc.simulate(uc.DOMAINS["sepsis"], n, np.random.default_rng(seed))


def test_partial_f_is_symmetric_for_scalar_proxies():
    # For one Z and one W the partial F is a function of their partial correlation only,
    # so relevance of Z for W and of W for Z coincide.
    d = _cohort()
    assert np.isclose(uc.first_stage_f(d), px.reverse_f(d), rtol=1e-8)


def test_dr_point_estimate_equals_proximal_ipw_with_linear_bridge():
    # With h linear in (1, T, W, A) and g = (1, W, A), the q-bridge moments cancel the
    # augmentation term, so the DR form and proximal IPW agree to solver precision.
    for seed in range(10, 20):
        d = _cohort(seed, 8000)
        est, _, _, ok, _, _ = px.pipw_sandwich(d)
        if ok:
            break
    assert ok
    dr, _, ipw, _ = uc.proximal_dr_se(d)
    assert abs(dr - ipw) < 1e-5 and abs(est - ipw) < 1e-8


def test_dr_consistent_when_outcome_bridge_correct_and_q_wrong():
    cfg = uc.DOMAINS["sepsis"]
    d = px.simulate_ms(cfg, 100_000, np.random.default_rng(3), h_ok=True, q_ok=False, tau=-0.16)
    est = px.pipw_sandwich(d)[0]
    assert abs(est + 0.16) < 0.02


def test_audit_card_has_cds_hooks_required_fields():
    path = os.path.join(os.path.dirname(HERE), "results", "audit_card.json")
    cards = json.load(open(path, encoding="utf-8"))
    for c in cards:
        card = c["card"]
        for k in ("uuid", "summary", "indicator", "source"):
            assert k in card
        assert isinstance(card["detail"], str)
        assert len(card["summary"]) <= 140
        assert card["indicator"] in ("info", "warning", "critical")
        assert "label" in card["source"] and "topic" in card["source"]
