"""Round-3 re-analysis of the held-out generator replications (results/v2_rule_reps.csv).

No new simulation: every quantity is recomputed from the stored replications of
``decision_rule_v2.py`` against the interventional truth of each generator.

Loss declared before this analysis was run (see REPRODUCIBILITY.md):
  primary   = 95% interval score (Gneiting & Raftery 2007): width plus 40x the distance
              by which the interval misses the truth;
  secondary = RMSE.
All three fixed families (proxy adjustment, proximal 2SLS, proximal DR) are scored on both
losses. Also reported: DR coverage stratified by U2 and rho_zw, a bias/SE decomposition of
DR undercoverage, the discrimination of the v2 flags (AUC for generator-level DR coverage
< 0.80), a disagreement-widened DR interval (heuristic, not a partial-identification bound),
and the bounded share / length of Anderson-Rubin sets by first-stage F bin.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_round2 as r2  # noqa: E402
import decision_rule_v2 as v2  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
FAM = ("proxy", "tsls", "dr")
Z = 1.959964
BOOT = 2000


def iscore(est, se, t):
    lo, hi = est - Z * se, est + Z * se
    return (hi - lo) + 40 * np.maximum(lo - t, 0) + 40 * np.maximum(t - hi, 0)


def auc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def wilson(x, m):
    if m == 0:
        return "nan"
    p = x / m
    c = Z * np.sqrt(p * (1 - p) / m + Z * Z / (4 * m * m)) / (1 + Z * Z / m)
    mid = (p + Z * Z / (2 * m)) / (1 + Z * Z / m)
    return f"{mid - c:.2f}-{mid + c:.2f}"


def write(path, rows):
    keys = []
    for x in rows:
        keys += [k for k in x if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rows = list(csv.DictReader(open(RES / "v2_rule_reps.csv")))
    rng = np.random.default_rng(11)
    summ, strat, arlen, gens = [], [], [], []
    for sname, seed in v2.SETS.items():
        des = v2.design(seed)
        G = []
        for g in range(v2.N_GEN):
            rr = [x for x in rows if x["set"] == sname and int(x["gen"]) == g]
            R = {k: np.array([float(x[k]) for x in rr]) for k in rr[0] if k != "set"}
            t = r2.truth_general(des[g])
            rec = dict(set=sname, gen=g, u2=des[g]["u2"], rho=des[g]["rho_zw"])
            for f in FAM:
                e, s = R[f], R[f + "_se"]
                rec[f + "_rmse"] = float(np.sqrt(np.mean((e - t) ** 2)))
                rec[f + "_is"] = float(np.mean(iscore(e, s, t)))
                rec[f + "_cov"] = float(np.mean(np.abs(e - t) <= Z * s))
                rec[f + "_w"] = float(np.mean(2 * Z * s))
            e, s = R["dr"], R["dr_se"]
            rec["dr_bias"] = float(np.mean(e) - t)
            rec["dr_sd"] = float(np.std(e, ddof=1))
            rec["dr_mse"] = float(np.median(s))
            wide = Z * s + np.abs(R["dr"] - R["tsls"])
            rec["wide_cov"] = float(np.mean(np.abs(e - t) <= wide))
            rec["wide_w"] = float(np.mean(2 * wide))
            dis = np.abs(R["dr"] - R["tsls"]) > Z * np.sqrt(R["dr_se"] ** 2 + R["tsls_se"] ** 2)
            flag = (R["F"] < v2.F_MIN) | (R["ok"] == 0) | dis
            rec["flag_rate"] = float(flag.mean())
            rec["dr_bad"] = int(rec["dr_cov"] < 0.80)
            for lab, lo_f, hi_f in (("<16", 0, v2.F_MIN), ("16-104.7", v2.F_MIN, v2.F_TF),
                                    (">=104.7", v2.F_TF, np.inf)):
                m = (R["F"] >= lo_f) & (R["F"] < hi_f)
                fin = m & np.isfinite(R["ar_lo"]) & np.isfinite(R["ar_hi"])
                arlen.append(dict(F_bin=lab, n=int(m.sum()), bounded=int(fin.sum()),
                                  lens=list(R["ar_hi"][fin] - R["ar_lo"][fin]),
                                  hits=int(((R["ar_lo"] <= t) & (t <= R["ar_hi"]) & fin).sum())))
            G.append(rec)
        A = {k: np.array([x[k] for x in G], float) for k in G[0] if k != "set"}
        idx = rng.integers(0, v2.N_GEN, (BOOT, v2.N_GEN))

        def ci(v):
            return np.nanpercentile(np.nanmean(v[idx], axis=1), [2.5, 97.5])

        for loss in ("is", "rmse"):
            M = np.column_stack([A[f + "_" + loss] for f in FAM])
            win = M.argmin(axis=1)
            for j, f in enumerate(FAM):
                w = (win == j).astype(float)
                lo, hi = ci(w)
                d_lo, d_hi = ci(M[:, j] - M[:, 2])
                summ.append(dict(set=sname, loss=loss, family=f,
                                 mean_loss=round(float(M[:, j].mean()), 4),
                                 diff_vs_dr_ci=f"{d_lo:+.4f} to {d_hi:+.4f}",
                                 best_share=round(float(w.mean()), 3),
                                 best_share_ci=f"{lo:.2f}-{hi:.2f}",
                                 coverage=round(float(A[f + "_cov"].mean()), 3),
                                 mean_width=round(float(A[f + "_w"].mean()), 4)))
        u2m, rhom = np.median(A["u2"]), np.median(A["rho"])
        for lab, m in (("u2<=median", A["u2"] <= u2m), ("u2>median", A["u2"] > u2m),
                       ("rho<=median", A["rho"] <= rhom), ("rho>median", A["rho"] > rhom),
                       ("all", np.ones(v2.N_GEN, bool))):
            strat.append(dict(set=sname, stratum=lab, n_gen=int(m.sum()),
                              dr_cov=round(float(A["dr_cov"][m].mean()), 3),
                              sd_over_se=round(float(np.median(A["dr_sd"][m] / A["dr_mse"][m])), 2),
                              absbias_over_sd=round(float(np.median(np.abs(A["dr_bias"][m]) / A["dr_sd"][m])), 2),
                              wide_cov=round(float(A["wide_cov"][m].mean()), 3),
                              wide_width=round(float(A["wide_w"][m].mean()), 4),
                              dr_width=round(float(A["dr_w"][m].mean()), 4),
                              median_u2=round(float(u2m), 3), median_rho=round(float(rhom), 3)))
        bad = A["dr_bad"].astype(int)
        a = auc(A["flag_rate"], bad)
        boots = [auc(A["flag_rate"][i], bad[i]) for i in idx[:500]]
        b_lo, b_hi = np.nanpercentile(boots, [2.5, 97.5])
        strat.append(dict(set=sname, stratum="flag_auc_dr_cov_below_0.80", n_gen=v2.N_GEN,
                          auc=round(a, 2), auc_ci=f"{b_lo:.2f}-{b_hi:.2f}", n_bad=int(bad.sum())))
        gens += G
    write(RES / "default_loss_summary.csv", summ)
    write(RES / "default_loss_strata.csv", strat)
    write(RES / "default_loss_generators.csv", gens)
    agg = {}
    for x in arlen:
        a_ = agg.setdefault(x["F_bin"], dict(n=0, bounded=0, lens=[], hits=0))
        for kk in ("n", "bounded", "hits"):
            a_[kk] += x[kk]
        a_["lens"] += x["lens"]
    out = []
    for k, a_ in agg.items():
        n, b = a_["n"], a_["bounded"]
        out.append(dict(F_bin=k, reps=n, bounded_share=round(b / n, 3) if n else "nan",
                        bounded_share_ci=wilson(b, n),
                        median_len_bounded=round(float(np.median(a_["lens"])), 3) if a_["lens"] else "nan",
                        cov_among_bounded=round(a_["hits"] / b, 3) if b else "nan",
                        cov_among_bounded_ci=wilson(a_["hits"], b)))
    write(RES / "ar_bounded.csv", out)


if __name__ == "__main__":
    main()
