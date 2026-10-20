"""Sample-size sensitivity of the two strategies available when severity is unrecorded.

Re-runs proxy adjustment (backdoor on age, Z, W) and the proximal estimators (two-stage and
doubly robust) at several cohort sizes. Replication seeds follow scripts/unmeasured_confounding.py,
so the n = 8000 rows reproduce results/proximal_estimates.csv for these estimators.

Output: results/sample_size.csv
Usage:  python scripts/sample_size.py [--reps 200] [--sizes 1000 2000 4000 8000 16000] [--seed 42]
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

ESTIMATORS = ("backdoor_proxy", "proximal", "proximal_dr")


def run(job):
    domain, n, reps, seed = job
    cfg = uc.DOMAINS[domain]
    truth = uc.true_ate(cfg, 200_000, np.random.default_rng(seed + 999))
    est = {k: [] for k in ESTIMATORS}
    cov = {k: [] for k in ESTIMATORS}
    for r in range(reps):
        rng = np.random.default_rng(seed + 1000 * r + uc.domain_offset(domain, 997))
        d = uc.simulate(cfg, n, rng)
        fits = {"backdoor_proxy": uc.backdoor_proxy_se(d), "proximal": uc.proximal_se(d),
                "proximal_dr": uc.proximal_dr_only(d)}
        for k, (th, se) in fits.items():
            est[k].append(th)
            cov[k].append(float(th - 1.96 * se <= truth <= th + 1.96 * se))
    rows = []
    for k in ESTIMATORS:
        v = np.array(est[k])
        ok = np.isfinite(v)
        c = float(np.mean(np.array(cov[k])[ok]))
        rows.append({"domain": domain, "n": n, "estimator": k, "true_ate": round(truth, 4),
                     "bias": round(float(v[ok].mean() - truth), 4),
                     "rmse": round(float(np.sqrt(np.mean((v[ok] - truth) ** 2))), 4),
                     "sd": round(float(v[ok].std(ddof=1)), 4),
                     "ci95_coverage": round(c, 3),
                     "mcse_coverage": round(float(np.sqrt(c * (1 - c) / ok.sum())), 3),
                     "reps": int(ok.sum())})
    print(f"  {domain:6} n={n:6d} done", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    jobs = [(dm, n, a.reps, a.seed) for dm in uc.DOMAINS for n in a.sizes]
    with ProcessPoolExecutor() as ex:
        rows = [r for block in ex.map(run, jobs) for r in block]
    out = uc.RESULTS / "sample_size.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
