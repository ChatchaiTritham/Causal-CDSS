"""Redesigned audit decision rule (v1.0.5), scored on held-out generators.

Background: on the 60 Latin-hypercube generators of ``review_round2.py`` (set A, seed 2027)
the v1 threshold rule of Algorithm 1 (choose proximal DR when the elicited gamma, delta lie
below half the design-stage crossing points, else proxy adjustment) picked the
lower-RMSE estimator in 36% of conclusive replications, against 65% for always choosing
proximal DR. This script evaluates a redesigned rule (v2) that drops the threshold branch:

  v2  report proximal DR by default; return INCONCLUSIVE (DR still reported, flagged) when
      (i) the treatment-bridge solve fails, (ii) the first-stage F < 16, or
      (iii) the two proximal estimators disagree: |DR - 2SLS| > 1.96 * sqrt(se_DR^2 + se_2SLS^2)
      (a bridge-agreement check; both are consistent when both bridges are correct).
      Below F = 104.7 the reported interval for the 2SLS step is an Anderson-Rubin
      projection interval instead of the Wald interval.

The rule was written AFTER seeing set A, so set A is not a clean test. It is therefore
re-scored on a FRESH set B (seed 2031, never used before) with the same ranges and the
same, unchanged constants. Comparators: always proxy adjustment, always 2SLS, always DR,
v1 with oracle (gamma, delta) and v1 with noisy elicited bounds (multiplicative lognormal
error, sd 0.35 on the log scale, about 0.5x-2x at 95%).

Scoring is per replication on all replications (inconclusive counted as its own category)
and cluster-bootstrapped over generators for CIs. Regret is RMSE(chosen) - min(RMSE_DR,
RMSE_proxy) per generator.

Usage: python scripts/decision_rule_v2.py [--workers 8] [--part rule|weak]
Writes results/v2_rule_reps.csv, v2_rule_summary.csv, v2_rule_flag.csv, v2_ar_coverage.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402
import review_round2 as r2  # noqa: E402

SETS = {"A (seed 2027, used in v1 test)": 2027, "B (seed 2031, fresh)": 2031}
N_GEN, REPS, SIZE = 60, 40, 4000
F_MIN, F_TF = 16.0, 104.7
ELICIT_SD = 0.35
BOOT = 2000


def design(seed):
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(r2.LHS_RANGES), seed=seed).random(N_GEN)
    keys = list(r2.LHS_RANGES)
    return [{k: r2.LHS_RANGES[k][0] + u[i, j] * (r2.LHS_RANGES[k][1] - r2.LHS_RANGES[k][0])
             for j, k in enumerate(keys)} for i in range(N_GEN)]


def ar_projection(d, level=0.95):
    """Anderson-Rubin projection interval for the treatment coefficient of the proximal 2SLS
    structural equation Y = c + bA A + bW W + bX age + e with excluded instrument Z.

    Bonferroni split: an AR confidence set for bW at 1 - alpha/2, then the union over accepted
    bW of the HC0 Wald interval for bA at 1 - alpha/2 from the regression of Y - bW W on
    (1, A, age, Z). Returns (lo, hi); infinite if the AR set reaches the grid edge.
    """
    n = len(d["y"])
    X = np.column_stack([np.ones(n), d["a"], d["age"], d["z"]])
    bread = np.linalg.inv(X.T @ X)
    P = bread @ X.T
    cy, cw = P @ d["y"], P @ d["w"]
    ey, ew = d["y"] - X @ cy, d["w"] - X @ cw
    M0 = (X * (ey * ey)[:, None]).T @ X
    M1 = (X * (ey * ew)[:, None]).T @ X
    M2 = (X * (ew * ew)[:, None]).T @ X
    b2, _ = uc.proximal_se(d)
    bW = r2.bridges(d)[0][2]
    grid = bW + np.linspace(-1, 1, 801) * max(5 * abs(bW), 2.0)
    a2 = 1 - level
    crit_ar = stats.chi2.ppf(1 - a2 / 2, 1)
    z = stats.norm.ppf(1 - a2 / 4)
    lo, hi, acc = np.inf, -np.inf, np.zeros(len(grid), bool)
    for i, g in enumerate(grid):
        coef = cy - g * cw
        V = bread @ (M0 - 2 * g * M1 + g * g * M2) @ bread
        if coef[3] ** 2 / V[3, 3] <= crit_ar:
            acc[i] = True
            s = np.sqrt(V[1, 1])
            lo, hi = min(lo, coef[1] - z * s), max(hi, coef[1] + z * s)
    if acc[0] or acc[-1]:
        return -np.inf, np.inf
    if not acc.any():
        return np.nan, np.nan
    return float(lo), float(hi)


def rep(job):
    sname, cell, r = job
    p = design(SETS[sname])[cell]
    sd = SETS[sname] * 1000 + 5_000_000 + 10_000 * cell + r
    d = r2.sim_general(p, SIZE, np.random.default_rng(sd))
    out = dict(set=sname, gen=cell, rep=r, F=uc.first_stage_f(d))
    out["proxy"], out["proxy_se"] = uc.backdoor_proxy_se(d)
    out["tsls"], out["tsls_se"] = uc.proximal_se(d)
    b, q, ok = r2.bridges(d)
    out["dr"], out["dr_se"] = r2.dr_ate(d, b, q)
    out["ok"] = int(ok)
    out["ar_lo"], out["ar_hi"] = ar_projection(d)
    e = np.random.default_rng(sd + 17).normal(0, ELICIT_SD, 2)
    out["gamma_el"], out["delta_el"] = p["gamma"] * np.exp(e[0]), p["delta"] * np.exp(e[1])
    return out


def decide(R, p):
    """Decisions of each strategy for one generator's replications (arrays)."""
    weak = (R["F"] < F_MIN) | (R["ok"] == 0)
    disagree = np.abs(R["dr"] - R["tsls"]) > 1.96 * np.sqrt(R["dr_se"] ** 2 + R["tsls_se"] ** 2)
    v1 = lambda g, dl: np.where(weak, "inconclusive",
                                np.where((g < r2.GSTAR_DR / 2) & (dl < r2.DSTAR_DR / 2), "dr", "proxy"))
    n = len(R["F"])
    return {
        "always proxy": np.full(n, "proxy"),
        "always 2SLS": np.full(n, "tsls"),
        "always DR": np.full(n, "dr"),
        "v1 oracle bounds": v1(np.full(n, p["gamma"]), np.full(n, p["delta"])),
        "v1 elicited bounds": v1(R["gamma_el"], R["delta_el"]),
        "v2 (DR default + flags)": np.where(weak | disagree, "dr_flag", "dr"),
    }, weak | disagree


WEAK_LOADINGS = (0.15, 0.20, 0.30, 0.40, 0.50)
WEAK_BASE = dict(age_to_sev=0.55, sev_to_treat=1.0, treat_intercept=-0.25, sev_to_out=1.0,
                 age_to_out=0.3, treat_to_out=-0.5, out_intercept=-0.75, u2=0.0, rho_zw=0.0,
                 gamma=0.0, delta=0.0)


def rep_weak(job):
    lam, r = job
    p = dict(WEAK_BASE, sev_to_z=lam, sev_to_w=lam)
    d = r2.sim_general(p, SIZE, np.random.default_rng(9_000_000 + int(lam * 1000) * 1000 + r))
    out = dict(lam=lam, rep=r, F=uc.first_stage_f(d))
    out["tsls"], out["tsls_se"] = uc.proximal_se(d)
    b, q, ok = r2.bridges(d)
    out["dr"], out["dr_se"] = r2.dr_ate(d, b, q)
    out["ok"] = int(ok)
    out["ar_lo"], out["ar_hi"] = ar_projection(d)
    return out


def main_weak(workers):
    """Valid proximal assumptions, weak proxies: Wald vs AR-projection coverage by F bin."""
    jobs = [(lam, r) for lam in WEAK_LOADINGS for r in range(200)]
    with ProcessPoolExecutor(workers) as ex:
        reps = list(ex.map(rep_weak, jobs, chunksize=8))
    rows = []
    for lam in WEAK_LOADINGS:
        t = r2.truth_general(dict(WEAK_BASE, sev_to_z=lam, sev_to_w=lam))
        R = {k: np.array([x[k] for x in reps if x["lam"] == lam], float) for k in reps[0]}
        for lo_f, hi_f, lab in ((0, F_MIN, "<16"), (F_MIN, F_TF, "16-104.7"), (F_TF, np.inf, ">=104.7")):
            m = (R["F"] >= lo_f) & (R["F"] < hi_f)
            if not m.any():
                continue
            ok = m & (R["ok"] == 1)
            rows.append(dict(loading=lam, F_bin=lab, reps=int(m.sum()),
                             F_median=round(float(np.median(R["F"][m])), 1),
                             bias_2sls=round(float(np.median(R["tsls"][m]) - t), 4),
                             wald_2sls_cov=round(float(np.mean(np.abs(R["tsls"][m] - t) <= 1.96 * R["tsls_se"][m])), 3),
                             ar_cov=round(float(np.mean((R["ar_lo"][m] <= t) & (t <= R["ar_hi"][m]))), 3),
                             ar_infinite=round(float(np.mean(np.isinf(R["ar_lo"][m]))), 3),
                             dr_cov_converged=round(float(np.mean(np.abs(R["dr"][ok] - t) <= 1.96 * R["dr_se"][ok])), 3) if ok.any() else "",
                             q_nonconverged=int((m & (R["ok"] == 0)).sum())))
    r2.write(rows, "v2_weak_valid.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--part", default="rule", choices=["rule", "weak"])
    a = ap.parse_args()
    if a.part == "weak":
        main_weak(a.workers)
        return
    jobs = [(s, c, r) for s in SETS for c in range(N_GEN) for r in range(REPS)]
    with ProcessPoolExecutor(a.workers) as ex:
        reps = list(ex.map(rep, jobs, chunksize=8))
    r2.write(reps, "v2_rule_reps.csv")
    rng = np.random.default_rng(7)
    summ, flag, arc = [], [], []
    for sname, seed in SETS.items():
        des = design(seed)
        per = {}   # strategy -> list over generators of (correct_share, inconc_share, rmse, regret, cov)
        gen_flag = []
        for g in range(N_GEN):
            rows = [x for x in reps if x["set"] == sname and x["gen"] == g]
            R = {k: np.array([x[k] for x in rows], float) for k in rows[0] if k != "set"}
            t = r2.truth_general(des[g])
            rm = {k: float(np.sqrt(np.mean((R[k] - t) ** 2))) for k in ("dr", "proxy", "tsls")}
            better = "dr" if rm["dr"] < rm["proxy"] else "proxy"
            best = min(rm["dr"], rm["proxy"])
            dec, flagged = decide(R, des[g])
            cov_dr = np.abs(R["dr"] - t) <= 1.96 * R["dr_se"]
            for sname2, dd in dec.items():
                fam = np.where(dd == "dr_flag", "dr", dd)
                inc = dd == "inconclusive"
                est = np.where(fam == "proxy", R["proxy"], np.where(fam == "tsls", R["tsls"], R["dr"]))
                se = np.where(fam == "proxy", R["proxy_se"], np.where(fam == "tsls", R["tsls_se"], R["dr_se"]))
                est = np.where(inc, R["dr"], est)          # inconclusive still reports DR
                se = np.where(inc, R["dr_se"], se)
                rmse = float(np.sqrt(np.mean((est - t) ** 2)))
                correct = np.where(inc, np.nan, (np.where(inc, "", fam) == better).astype(float))
                if sname2 == "always 2SLS":
                    correct = np.full(len(fam), np.nan)
                per.setdefault(sname2, []).append(dict(
                    correct=float(np.nanmean(correct)) if np.isfinite(correct).any() else np.nan,
                    correct_all=float(np.nansum(correct) / len(fam)),
                    inconc=float(inc.mean()), rmse=rmse, regret=rmse - best,
                    cov=float(np.mean(np.abs(est - t) <= 1.96 * se))))
            # flag diagnostics for v2
            dr_bad = np.mean(cov_dr) < 0.80
            ar_in = (R["ar_lo"] <= t) & (t <= R["ar_hi"])
            for lo_f, hi_f, lab in ((F_MIN, F_TF, "16-104.7"), (F_TF, np.inf, ">=104.7"), (0, F_MIN, "<16")):
                m = (R["F"] >= lo_f) & (R["F"] < hi_f)
                tc = np.abs(R["tsls"] - t) <= 1.96 * R["tsls_se"]
                arc.append(dict(set=sname, gen=g, F_bin=lab, n=int(m.sum()),
                                wald_hits=int(tc[m].sum()), ar_hits=int(ar_in[m].sum()),
                                ar_infinite=int(np.isinf(R["ar_lo"][m]).sum())))
            gen_flag.append(dict(set=sname, gen=g, dr_cov=round(float(cov_dr.mean()), 3),
                                 dr_bad=int(dr_bad), flag_rate=round(float(flagged.mean()), 3),
                                 u2=round(des[g]["u2"], 3), rho_zw=round(des[g]["rho_zw"], 3),
                                 gamma=round(des[g]["gamma"], 3), delta=round(des[g]["delta"], 3),
                                 cov_dr_unflagged=round(float(cov_dr[~flagged].mean()), 3) if (~flagged).any() else "",
                                 cov_dr_flagged=round(float(cov_dr[flagged].mean()), 3) if flagged.any() else ""))
        flag += gen_flag
        base = np.array([x["regret"] for x in per["always DR"]])
        base_c = np.array([x["correct_all"] for x in per["always DR"]])
        for strat, L in per.items():
            A = {k: np.array([x[k] for x in L], float) for k in L[0]}
            idx = rng.integers(0, N_GEN, (BOOT, N_GEN))
            ci = lambda v: np.nanpercentile(np.nanmean(v[idx], axis=1), [2.5, 97.5])
            c_lo, c_hi = ci(A["correct"])
            d_lo, d_hi = ci(A["regret"] - base)
            ca_lo, ca_hi = ci(A["correct_all"] - base_c)
            a_lo, a_hi = ci(A["correct_all"])
            summ.append(dict(set=sname, strategy=strat,
                             correct_conclusive=round(float(np.nanmean(A["correct"])), 3),
                             correct_ci=f"{c_lo:.2f}-{c_hi:.2f}",
                             correct_all_reps=round(float(np.mean(A["correct_all"])), 3),
                             correct_all_ci=f"{a_lo:.2f}-{a_hi:.2f}",
                             diff_correct_all_vs_dr_ci=f"{ca_lo:+.2f} to {ca_hi:+.2f}",
                             inconclusive=round(float(A["inconc"].mean()), 3),
                             mean_rmse=round(float(A["rmse"].mean()), 4),
                             mean_regret=round(float(A["regret"].mean()), 4),
                             regret_minus_dr_ci=f"{d_lo:+.4f} to {d_hi:+.4f}",
                             coverage=round(float(A["cov"].mean()), 3)))
    r2.write(summ, "v2_rule_summary.csv")
    r2.write(flag, "v2_rule_flag.csv")
    r2.write(arc, "v2_ar_coverage.csv")


if __name__ == "__main__":
    main()
