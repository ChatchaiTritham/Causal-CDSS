"""Checks requested in the Statistics in Medicine pre-submission review (v1.0.8). No new simulation.

From the stored held-out replications (results/v2_rule_reps.csv) and generator summaries
(results/default_loss_generators.csv):
  1. threshold-free association between the generator-level flag rate and DR coverage
     (Spearman rank correlation, cluster-bootstrap 95% interval) and the flag AUC at
     coverage thresholds 0.70, 0.80 and 0.90 (label noise from 40 replications attenuates a
     single-threshold AUC);
  2. the 90% interval score (width + 20 x miss) for the three fixed families, to show how the
     ranking depends on the miss penalty;
  3. the share of replications in which the DR treatment bridge did not solve.
Writes results/sim_review_checks.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import default_loss_audit as dl  # noqa: E402
import decision_rule_v2 as v2  # noqa: E402
import review_round2 as r2  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
Z90 = 1.644854


def iscore90(est, se, t):
    lo, hi = est - Z90 * se, est + Z90 * se
    return (hi - lo) + 20 * np.maximum(lo - t, 0) + 20 * np.maximum(t - hi, 0)


def main() -> None:
    gens = list(csv.DictReader(open(RES / "default_loss_generators.csv")))
    reps = list(csv.DictReader(open(RES / "v2_rule_reps.csv")))
    rng = np.random.default_rng(13)
    out = []
    for sname, seed in v2.SETS.items():
        G = [g for g in gens if g["set"] == sname]
        fr = np.array([float(g["flag_rate"]) for g in G])
        cov = np.array([float(g["dr_cov"]) for g in G])
        rho = stats.spearmanr(fr, cov).statistic
        idx = rng.integers(0, len(G), (2000, len(G)))
        boots = [stats.spearmanr(fr[i], cov[i]).statistic for i in idx]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out.append(dict(set=sname, check="spearman flag_rate vs dr_cov", value=round(float(rho), 3),
                        ci=f"{lo:.2f} to {hi:.2f}", n=len(G)))
        for thr in (0.70, 0.80, 0.90):
            lab = (cov < thr).astype(int)
            out.append(dict(set=sname, check=f"flag AUC, dr_cov < {thr:.2f}",
                            value=round(dl.auc(fr, lab), 3), ci="", n=int(lab.sum())))
        des = v2.design(seed)
        R = [x for x in reps if x["set"] == sname]
        fam_is = {f: [] for f in dl.FAM}
        for g in range(v2.N_GEN):
            rr = [x for x in R if int(x["gen"]) == g]
            t = r2.truth_general(des[g])
            for f in dl.FAM:
                e = np.array([float(x[f]) for x in rr])
                s = np.array([float(x[f + "_se"]) for x in rr])
                fam_is[f].append(float(np.mean(iscore90(e, s, t))))
        M = np.column_stack([fam_is[f] for f in dl.FAM])
        win = M.argmin(axis=1)
        for j, f in enumerate(dl.FAM):
            out.append(dict(set=sname, check=f"90% interval score, {f}", value=round(float(M[:, j].mean()), 4),
                            ci=f"best share {np.mean(win == j):.2f}", n=v2.N_GEN))
        ok = np.array([int(x["ok"]) for x in R])
        out.append(dict(set=sname, check="share of replications with unsolved q", value=round(float(1 - ok.mean()), 4),
                        ci="", n=len(ok)))
    dl.write(RES / "sim_review_checks.csv", out)


if __name__ == "__main__":
    main()
