"""Proximal policy value when an exclusion restriction fails.

Repeats the policy-value comparison of scripts/proximal_policy_subgroup.py on cohorts drawn from
the violated generators of scripts/proximal_robustness.py: a direct Z -> Y effect gamma (logit
scale) or a direct T -> W effect delta. For each rule and violation it reports bias and 95%
interval coverage of the proximal DR value estimator and of the standard DR estimator (logistic
nuisances on recorded covariates) against the interventional value on the realised cohort.

Output: results/proximal_policy_violation.csv
Usage:  python scripts/proximal_policy_violation.py [--reps 100] [--n 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402
import proximal_robustness as pr  # noqa: E402
from proximal_policy_subgroup import proximal_value  # noqa: E402

LEVELS = (0.0, 0.1, 0.2, 0.3, 0.5)
POLICIES = ("age_rule", "cdss_top50", "knowledge")


def value_truth(cfg, d, pi, gamma):
    base = (cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"]
            + gamma * d["z"] + cfg["out_intercept"])
    return float(np.mean(uc.sigmoid(base + cfg["treat_to_out"] * pi)))


def run(job):
    domain, kind, level, n, reps, seed = job
    cfg = uc.DOMAINS[domain]
    acc = {(p, k): [] for p in POLICIES for k in ("pb", "pc", "db", "dc")}
    for r in range(reps):
        off = seed + 131 * r + uc.domain_offset(domain, 97)
        if kind == "gamma":
            d = pr.simulate_violated(cfg, n, np.random.default_rng(off), level)
            train = pr.simulate_violated(cfg, n, np.random.default_rng(off + 500_000), level)
            g = level
        else:
            d = pr.simulate_violated_w(cfg, n, np.random.default_rng(off), level)
            train = pr.simulate_violated_w(cfg, n, np.random.default_rng(off + 500_000), level)
            g = 0.0
        pols = {"age_rule": (d["age"] > 0).astype(float),
                "cdss_top50": uc.policy_cdss(d, 0.5, train),
                "knowledge": uc.policy_knowledge(d, cfg)}
        b = uc.outcome_bridge(d)
        q = uc.treatment_bridge(d)
        nuis = uc.fit_nuisances(d, "logistic_recorded")
        for p, pi in pols.items():
            v = value_truth(cfg, d, pi, g)
            th, se = proximal_value(d, pi, b, q)
            o = uc.ope_estimates(d, pi, nuis)
            acc[(p, "pb")].append(th - v)
            acc[(p, "pc")].append(float(abs(th - v) <= 1.96 * se))
            acc[(p, "db")].append(o["dr"] - v)
            acc[(p, "dc")].append(float(abs(o["dr"] - v) <= 1.96 * o["dr_se"]))
    rows = []
    for p in POLICIES:
        rows.append({"domain": domain, "violation": kind, "level": level, "policy": p,
                     "prox_bias": round(float(np.mean(acc[(p, "pb")])), 4),
                     "prox_coverage": round(float(np.mean(acc[(p, "pc")])), 3),
                     "dr_bias": round(float(np.mean(acc[(p, "db")])), 4),
                     "dr_coverage": round(float(np.mean(acc[(p, "dc")])), 3), "reps": reps})
    print(f"  {domain} {kind}={level} done", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    jobs = [(dm, k, lv, a.n, a.reps, a.seed) for dm in uc.DOMAINS for k in ("gamma", "delta")
            for lv in LEVELS]
    with ProcessPoolExecutor() as ex:
        rows = [r for block in ex.map(run, jobs) for r in block]
    with open(uc.RESULTS / "proximal_policy_violation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/proximal_policy_violation.csv")


if __name__ == "__main__":
    main()
