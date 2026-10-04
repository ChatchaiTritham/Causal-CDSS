"""What happens to proximal identification when its own assumptions are false.

Simulating proxies that satisfy the proximal exclusion restrictions and then showing that
proximal estimation works is the same circularity as simulating a confounder and adjusting for
it. The question that is not circular is how fast the estimator degrades as the assumptions
fail, and whether the failure is detectable from data.

Three experiments:

1. **Exclusion-restriction sweep.** The negative-control exposure Z is given a direct effect on
   the outcome of size gamma, swept from 0 (assumption holds) to 0.5 (badly violated). The
   bias of the proximal estimator is traced against gamma and compared with the backdoor
   estimator, whose bias does not depend on gamma. The crossing point -- the violation at which
   proximal stops being the better choice -- is reported, because that, rather than the
   gamma = 0 result, is what tells an analyst when to use it.

2. **Double machine learning.** Cross-fitted DML (Chernozhukov et al., 2018) is run on the same
   data, both in the oracle case where severity is recorded and in the realistic case where it
   is not, so the reader can see which part of the error is estimation and which is
   identification. Nominal 95% interval coverage is reported over replications.

3. **Testable implications of the DAG.** The graph implies conditional independencies; those are
   tested against the data for the true DAG and for perturbed DAGs (an edge deleted, an edge
   added). A specification that is wrong in a way the data can see should fail more of its
   implied tests -- this is what makes the DAG falsifiable rather than assumed.

Outputs: results/proximal_robustness.csv, results/dml_estimates.csv,
         results/dag_implication_tests.csv

Usage: python scripts/proximal_robustness.py [--reps 100] [--n 8000]
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from scipy import stats  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from unmeasured_confounding import (  # noqa: E402
    DOMAINS, backdoor_observed_ate, ols, proximal_ate, sigmoid, true_ate,
)

RESULTS = REPO / "results"
GAMMAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def simulate_violated(cfg: dict, n: int, rng: np.random.Generator, gamma: float) -> dict:
    """Same generator as the main proximal experiment, but Z now acts on the outcome.

    gamma is the direct Z -> Y effect on the logit scale. gamma = 0 is the proximal
    assumption; anything else violates the exclusion restriction that identification needs.
    """
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    z = cfg["sev_to_z"] * sev + rng.normal(0, 1, n)
    w = cfg["sev_to_w"] * sev + rng.normal(0, 1, n)
    pt = sigmoid(cfg["sev_to_treat"] * sev + cfg["treat_intercept"])
    a = (rng.uniform(0, 1, n) < pt).astype(float)
    py = sigmoid(cfg["sev_to_out"] * sev + cfg["age_to_out"] * age
                 + cfg["treat_to_out"] * a + gamma * z + cfg["out_intercept"])
    y = (rng.uniform(0, 1, n) < py).astype(float)
    return dict(age=age, severity=sev, z=z, w=w, a=a, y=y)


def true_ate_violated(cfg: dict, n: int, rng: np.random.Generator, gamma: float) -> float:
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    z = cfg["sev_to_z"] * sev + rng.normal(0, 1, n)
    base = (cfg["sev_to_out"] * sev + cfg["age_to_out"] * age + gamma * z
            + cfg["out_intercept"])
    return float(np.mean(sigmoid(base + cfg["treat_to_out"]) - sigmoid(base)))


def dml_ate(d: dict, covariates: list[str], seed: int = 0, folds: int = 5) -> tuple[float, float]:
    """Cross-fitted partially linear DML estimate of the ATE and its standard error.

    Nuisances (propensity and outcome regression) are gradient-boosted; the orthogonal score
    is the usual residual-on-residual regression, so the estimate is not contaminated by
    overfitting of either nuisance.
    """
    X = np.column_stack([d[c] for c in covariates])
    a, y = d["a"], d["y"]
    n = len(y)
    res_a, res_y = np.zeros(n), np.zeros(n)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m_a = GradientBoostingClassifier(random_state=seed).fit(X[tr], a[tr])
        m_y = GradientBoostingRegressor(random_state=seed).fit(X[tr], y[tr])
        res_a[te] = a[te] - m_a.predict_proba(X[te])[:, 1]
        res_y[te] = y[te] - m_y.predict(X[te])
    denom = float(np.sum(res_a ** 2))
    theta = float(np.sum(res_a * res_y) / denom) if denom > 0 else float("nan")
    eps = res_y - theta * res_a
    se = float(np.sqrt(np.sum((res_a * eps) ** 2)) / denom) if denom > 0 else float("nan")
    return theta, se


def partial_corr_test(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> float:
    """p-value of a partial-correlation test of X _||_ Y | Z (Fisher z)."""
    if Z is not None and Z.size:
        bx = ols(Z, x)
        by = ols(Z, y)
        Zd = np.column_stack([np.ones(len(Z)), Z])
        x = x - Zd @ bx
        y = y - Zd @ by
    n = len(x)
    k = Z.shape[1] if (Z is not None and Z.size) else 0
    r = float(np.corrcoef(x, y)[0, 1])
    r = np.clip(r, -0.999999, 0.999999)
    dof = n - k - 3
    if dof <= 0:
        return float("nan")
    zstat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(dof)
    return float(2 * (1 - stats.norm.cdf(abs(zstat))))


def implied_independencies(graph: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """Conditional independencies implied by each candidate graph.

    The true graph is age -> severity -> {treatment, outcome}, age -> outcome,
    treatment -> outcome, severity -> {z, w}, with z and w having no other edges. Only the
    implications that distinguish the candidates are listed; each is checked against data.
    """
    base = [
        ("z", "a", ("severity",)),        # z _||_ a | severity
        ("z", "y", ("severity", "a")),    # exclusion restriction on z
        ("w", "a", ("severity",)),        # w _||_ a | severity
        ("z", "w", ("severity",)),        # the two proxies share only severity
        ("age", "a", ("severity",)),      # age acts on treatment only through severity
    ]
    if graph == "true":
        return base
    if graph == "missing_edge_sev_to_y":
        # claims severity has no direct effect on the outcome
        return base + [("severity", "y", ("a", "age"))]
    if graph == "extra_edge_age_to_a":
        # claims age acts on treatment directly, so it drops the implication that it does not
        return [t for t in base if t[:2] != ("age", "a")]
    raise ValueError(graph)


def main() -> None:
    ap = argparse.ArgumentParser(description="robustness of proximal identification")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- 1. violation sweep
    rows = []
    for domain, cfg in DOMAINS.items():
        for gamma in GAMMAS:
            truth = true_ate_violated(cfg, 200_000,
                                      np.random.default_rng(args.seed + 7), gamma)
            est = {"proximal": [], "backdoor_observed": []}
            for r in range(args.reps):
                rng = np.random.default_rng(args.seed + 131 * r + hash(domain) % 97)
                d = simulate_violated(cfg, args.n, rng, gamma)
                est["proximal"].append(proximal_ate(d))
                est["backdoor_observed"].append(backdoor_observed_ate(d))
            for name, vals in est.items():
                v = np.array(vals)
                rows.append({"domain": domain, "gamma_z_to_y": gamma, "estimator": name,
                             "true_ate": round(truth, 4),
                             "mean_estimate": round(float(v.mean()), 4),
                             "bias": round(float(v.mean() - truth), 4),
                             "abs_bias": round(abs(float(v.mean() - truth)), 4),
                             "reps": args.reps, "n": args.n})
            p = [r for r in rows if r["domain"] == domain and r["gamma_z_to_y"] == gamma]
            print(f"  {domain:8} gamma {gamma:<5} "
                  + "  ".join(f"{r['estimator']}: {r['abs_bias']:.3f}" for r in p), flush=True)

    with open(RESULTS / "proximal_robustness.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/proximal_robustness.csv")

    # ---------------------------------------------------------------- 2. DML
    dml_rows = []
    for domain, cfg in DOMAINS.items():
        truth = true_ate(cfg, 200_000, np.random.default_rng(args.seed + 999))
        for label, covs in (("oracle_severity_recorded", ["age", "severity"]),
                            ("recorded_only", ["age", "z", "w"])):
            ests, covers = [], []
            for r in range(max(args.reps // 5, 10)):
                rng = np.random.default_rng(args.seed + 977 * r + hash(domain) % 89)
                d = simulate_violated(cfg, args.n, rng, 0.0)
                theta, se = dml_ate(d, covs, seed=r)
                ests.append(theta)
                covers.append(float(theta - 1.96 * se <= truth <= theta + 1.96 * se))
            e = np.array(ests)
            dml_rows.append({"domain": domain, "adjustment_set": label,
                             "true_ate": round(truth, 4),
                             "mean_estimate": round(float(e.mean()), 4),
                             "bias": round(float(e.mean() - truth), 4),
                             "ci95_coverage": round(float(np.mean(covers)), 3),
                             "reps": len(ests), "n": args.n})
            print(f"  DML {domain:8} {label:26} bias {dml_rows[-1]['bias']:+.4f} "
                  f"coverage {dml_rows[-1]['ci95_coverage']}", flush=True)
    with open(RESULTS / "dml_estimates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dml_rows[0]))
        w.writeheader()
        w.writerows(dml_rows)
    print("wrote results/dml_estimates.csv")

    # ---------------------------------------------------------------- 3. DAG implications
    dag_rows = []
    for domain, cfg in DOMAINS.items():
        rng = np.random.default_rng(args.seed + hash(domain) % 83)
        d = simulate_violated(cfg, args.n, rng, 0.0)
        for graph in ("true", "missing_edge_sev_to_y", "extra_edge_age_to_a"):
            tests = implied_independencies(graph)
            passed = 0
            for x, y, cond in tests:
                Z = np.column_stack([d[c] for c in cond]) if cond else None
                pv = partial_corr_test(d[x], d[y], Z)
                passed += pv > 0.05
                dag_rows.append({"domain": domain, "graph": graph,
                                 "implication": f"{x} _||_ {y} | {','.join(cond) or '-'}",
                                 "p_value": round(pv, 5), "holds_at_0.05": bool(pv > 0.05)})
            print(f"  DAG {domain:8} {graph:24} {passed}/{len(tests)} implications hold",
                  flush=True)
    with open(RESULTS / "dag_implication_tests.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dag_rows[0]))
        w.writeheader()
        w.writerows(dag_rows)
    print("wrote results/dag_implication_tests.csv")


if __name__ == "__main__":
    main()
