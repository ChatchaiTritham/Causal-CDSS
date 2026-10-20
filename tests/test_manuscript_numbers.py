"""Every number quoted in the KBS manuscript's abstract and highlights, checked against the
result files (v1.0.4). If a rerun changes a CSV, this test fails until the text is updated."""
import os

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
rd = lambda name: pd.read_csv(os.path.join(RES, name))


def rng(series, d):
    return round(float(series.min()), d), round(float(series.max()), d)


def test_ate_numbers():
    e = rd("proximal_estimates.csv").set_index(["domain", "estimator"])
    age = e.xs("backdoor_observed", level=1)
    assert rng(age.abs_bias, 3) == (0.133, 0.185)
    prox = e.xs("backdoor_proxy", level=1)
    assert round(prox.abs_bias.mean(), 2) == 0.05
    assert prox.ci95_coverage.max() <= 0.005
    assert e.xs("proximal", level=1).abs_bias.max() <= 0.0165
    dr = e.xs("proximal_dr", level=1)
    assert dr.abs_bias.max() <= 0.005 + 1e-9
    assert rng(dr.ci95_coverage, 2) == (0.95, 0.96)


def test_crossing_numbers_dense_grid():
    c = rd("r2_crossing_dense.csv")
    c = c[c.comparator == "linear proxy adjustment"]
    g2 = c[(c.estimator == "2sls") & (c.restriction == "gamma")].crossing.astype(float)
    d2 = c[(c.estimator == "2sls") & (c.restriction == "delta")].crossing.astype(float)
    ddr = c[(c.estimator == "dr") & (c.restriction == "delta")].crossing.astype(float)
    gdr = c[(c.estimator == "dr") & (c.restriction == "gamma")].crossing
    assert rng(g2, 2) == (0.20, 0.25)
    assert rng(d2, 2) == (0.17, 0.20)
    assert (round(ddr.min(), 3), round(ddr.max(), 2)) == (0.215, 0.24)
    assert set(gdr) == {">0.30"}


def test_lhs_accuracy():
    d = rd("r2_lhs_design.csv")
    w = 1 - d.share_inconclusive
    acc = (d.decision_correct.fillna(0) * w).sum() / w.sum()
    assert round(acc, 2) == 0.36


def test_policy_value_numbers():
    pv = rd("proximal_policy_value.csv")
    ident = pv[pv.identified == 1]
    assert rng(ident.prox_coverage, 2) == (0.94, 0.98)
    other = pv[pv.identified == 0]
    assert rng(other.prox_coverage, 2) == (0.90, 0.98)
    cov = rd("policy_value_coverage.csv")
    rec = cov[cov.nuisance == "logistic_recorded"].dr_ci95_coverage
    assert round(rec.min(), 3) == 0.005 and round(rec.max(), 3) == 0.925
    v = rd("proximal_policy_violation.csv")
    assert v[(v.policy == "knowledge") & (v.violation == "gamma")].prox_coverage.min() == 0.30
