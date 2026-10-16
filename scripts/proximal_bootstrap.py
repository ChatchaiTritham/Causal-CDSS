"""Nonparametric bootstrap inference for the proximal estimators.

The plug-in standard error of the doubly robust proximal estimator (proximal_dr_se) treats the
outcome and treatment bridges as known. This driver re-estimates both bridges inside every
bootstrap resample, so the interval propagates bridge-estimation uncertainty, and compares the
coverage of the plug-in, bootstrap-SE (Wald) and percentile intervals against the interventional
truth. Replication seeds match scripts/unmeasured_confounding.py, so replication r here is the
same cohort as replication r in results/proximal_estimates.csv.

Output: results/proximal_bootstrap.csv (per replication), results/proximal_bootstrap_summary.csv

Usage: python scripts/proximal_bootstrap.py [--reps 100] [--boot 200] [--n 8000] [--seed 42]
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

KEYS = ("age", "z", "w", "a", "y")


def run_domain(args_tuple):
    domain, reps, boot, n, seed = args_tuple
    cfg = uc.DOMAINS[domain]
    truth = uc.true_ate(cfg, 200_000, np.random.default_rng(seed + 999))
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed + 1000 * r + uc.domain_offset(domain, 997))
        d = uc.simulate(cfg, n, rng)
        d = {k: d[k] for k in KEYS}
        tsls, tsls_se = uc.proximal_se(d)
        dr, dr_se = uc.proximal_dr_only(d)
        brng = np.random.default_rng(seed + 7_000_000 + 1000 * r + uc.domain_offset(domain, 997))
        bt, bd = np.empty(boot), np.empty(boot)
        for b in range(boot):
            idx = brng.integers(0, n, n)
            db = {k: v[idx] for k, v in d.items()}
            bt[b] = uc.proximal_se(db)[0]
            bd[b] = uc.proximal_dr_only(db)[0]
        for name, est, se, bs in (("proximal_2sls", tsls, tsls_se, bt),
                                  ("proximal_dr", dr, dr_se, bd)):
            bse = float(bs.std(ddof=1))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(domain=domain, rep=r, estimator=name, truth=truth, estimate=est,
                             plugin_se=se, boot_se=bse,
                             cover_plugin=int(abs(est - truth) <= 1.96 * se),
                             cover_boot_wald=int(abs(est - truth) <= 1.96 * bse),
                             cover_boot_pct=int(lo <= truth <= hi)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    jobs = [(dom, a.reps, a.boot, a.n, a.seed) for dom in uc.DOMAINS]
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = [row for part in ex.map(run_domain, jobs) for row in part]
    out = uc.RESULTS / "proximal_bootstrap.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for dom in uc.DOMAINS:
        for est in ("proximal_2sls", "proximal_dr"):
            s = [x for x in rows if x["domain"] == dom and x["estimator"] == est]
            m = lambda k: float(np.mean([x[k] for x in s]))
            summ.append(dict(domain=dom, estimator=est, reps=len(s),
                             emp_sd=round(float(np.std([x["estimate"] for x in s], ddof=1)), 5),
                             mean_plugin_se=round(m("plugin_se"), 5),
                             mean_boot_se=round(m("boot_se"), 5),
                             cov_plugin=round(m("cover_plugin"), 3),
                             cov_boot_wald=round(m("cover_boot_wald"), 3),
                             cov_boot_pct=round(m("cover_boot_pct"), 3)))
    with open(uc.RESULTS / "proximal_bootstrap_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
