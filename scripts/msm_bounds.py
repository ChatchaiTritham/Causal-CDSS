"""Marginal-sensitivity-model (MSM) bounds for the ATE and the treat-all / treat-none values.

Added in v1.0.7 for the Statistics in Medicine submission. The proximal estimators give no
uncertainty statement that survives a failure of the bridge assumptions; this script asks
how large a departure from no unmeasured confounding (given the recorded covariates age, Z, W)
an auditor would have to allow before partial-identification bounds contain the truth.

Model (Tan 2006): the odds of treatment given (X, U) differ from the odds given X by a
factor in [1/Lambda, Lambda]. For the Hajek (normalised) IPW mean of a binary outcome in arm t
the bounds are closed form: the extreme weight vector puts the largest admissible inverse
weight on units with Y = 1 and the smallest on units with Y = 0 (and the reverse for the lower
bound). The ATE bounds are [L1 - U0, U1 - L0]. Sampling uncertainty is added by the percentile
bootstrap of Zhao, Small and Bhattacharya (2019), refitting the propensity model in every
resample.

Per replication we record
  lambda_star : smallest Lambda (grid 1.00-6.00, step 0.01) whose point bounds contain the
                interventional truth (needs the truth: an oracle quantity),
  lambda_zero : smallest Lambda whose point bounds contain 0 (observable; the sensitivity
                value an auditor would report),
and, for the main generators, the bootstrap coverage of the truth at Lambda in LAMBDAS.

Usage: python scripts/msm_bounds.py [--reps 100] [--boot 100] [--heldout-reps 10]
Writes results/msm_bounds_main.csv, results/msm_bounds_heldout.csv, results/msm_bounds_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402
import review_round2 as r2  # noqa: E402
import decision_rule_v2 as v2  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
GRID = np.round(np.arange(1.0, 6.0001, 0.01), 2)
LAMBDAS = (1.0, 1.5, 2.0, 3.0)


def logit_fit(X, t, iters=25):
    """Unpenalised logistic regression by Newton-Raphson; returns fitted P(T=1|X)."""
    Xd = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = uc.sigmoid(Xd @ b)
        H = (Xd * (p * (1 - p))[:, None]).T @ Xd
        step = np.linalg.solve(H, Xd.T @ (t - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return np.clip(uc.sigmoid(Xd @ b), 1e-6, 1 - 1e-6)


def arm_bounds(y, base_w, lam):
    """Bounds on the Hajek mean of binary y with inverse weights 1 + (base_w - 1) * [1/lam, lam].

    base_w is the estimated inverse probability of the received arm (>= 1)."""
    lo_w = 1 + (base_w - 1) / lam
    hi_w = 1 + (base_w - 1) * lam
    one = y == 1
    up = hi_w[one].sum() / (hi_w[one].sum() + lo_w[~one].sum())
    dn = lo_w[one].sum() / (lo_w[one].sum() + hi_w[~one].sum())
    return dn, up


def ate_bounds(d, lam, e=None):
    if e is None:
        e = logit_fit(np.column_stack([d["age"], d["z"], d["w"]]), d["a"])
    t = d["a"] == 1
    l1, u1 = arm_bounds(d["y"][t], 1 / e[t], lam)
    l0, u0 = arm_bounds(d["y"][~t], 1 / (1 - e[~t]), lam)
    return l1 - u0, u1 - l0, (l1, u1), (l0, u0)


def smallest_lambda(d, target):
    e = logit_fit(np.column_stack([d["age"], d["z"], d["w"]]), d["a"])
    for lam in GRID:
        lo, hi, _, _ = ate_bounds(d, lam, e)
        if lo <= target <= hi:
            return float(lam)
    return float("inf")


def boot_interval(d, lam, B, rng):
    """Zhao-Small-Bhattacharya percentile interval: 2.5% of lower and 97.5% of upper bounds."""
    n = len(d["y"])
    los, his = np.empty(B), np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        db = {k: d[k][i] for k in ("age", "z", "w", "a", "y")}
        los[b], his[b], _, _ = ate_bounds(db, lam)
    return float(np.percentile(los, 2.5)), float(np.percentile(his, 97.5))


def write(path, rows):
    keys = []
    for x in rows:
        keys += [k for k in x if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--heldout-reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main_rows, summ = [], []
    for domain, cfg in uc.DOMAINS.items():
        truth = uc.true_ate(cfg, 200_000, np.random.default_rng(args.seed + 999))
        for r in range(args.reps):
            # same cohorts as the main study (unmeasured_confounding.py, first `reps` seeds)
            d = uc.simulate(cfg, 8000, np.random.default_rng(
                args.seed + 1000 * r + uc.domain_offset(domain, 997)))
            rec = dict(domain=domain, rep=r, truth=round(truth, 5),
                       lambda_star=smallest_lambda(d, truth),
                       lambda_zero=smallest_lambda(d, 0.0))
            brng = np.random.default_rng(args.seed + 77_000 + 1000 * r + uc.domain_offset(domain, 997))
            for lam in LAMBDAS:
                lo, hi = boot_interval(d, lam, args.boot, brng)
                rec[f"lo_{lam}"], rec[f"hi_{lam}"] = round(lo, 5), round(hi, 5)
                rec[f"cov_{lam}"] = int(lo <= truth <= hi)
                rec[f"excl0_{lam}"] = int(hi < 0 or lo > 0)
            main_rows.append(rec)
        R = [x for x in main_rows if x["domain"] == domain]
        s = dict(analysis="main", domain=domain, n_rep=len(R), truth=round(truth, 4),
                 lambda_star_median=round(q([x["lambda_star"] for x in R], 50), 2),
                 lambda_star_q10=round(q([x["lambda_star"] for x in R], 10), 2),
                 lambda_star_q90=round(q([x["lambda_star"] for x in R], 90), 2),
                 lambda_zero_median=round(q([x["lambda_zero"] for x in R], 50), 2),
                 lambda_zero_q10=round(q([x["lambda_zero"] for x in R], 10), 2),
                 lambda_zero_q90=round(q([x["lambda_zero"] for x in R], 90), 2))
        for lam in LAMBDAS:
            s[f"cov_{lam}"] = round(float(np.mean([x[f"cov_{lam}"] for x in R])), 3)
            s[f"excl0_{lam}"] = round(float(np.mean([x[f"excl0_{lam}"] for x in R])), 3)
            s[f"width_{lam}"] = round(float(np.mean([x[f"hi_{lam}"] - x[f"lo_{lam}"] for x in R])), 4)
        summ.append(s)

    held = []
    sname = "B (seed 2031, fresh)"
    des = v2.design(v2.SETS[sname])
    for g, p in enumerate(des):
        t = r2.truth_general(p)
        ls, lz = [], []
        for r in range(args.heldout_reps):
            # same seeds as decision_rule_v2.rep, first `heldout-reps` replications
            sd = v2.SETS[sname] * 1000 + 5_000_000 + 10_000 * g + r
            d = r2.sim_general(p, v2.SIZE, np.random.default_rng(sd))
            ls.append(smallest_lambda(d, t))
            lz.append(smallest_lambda(d, 0.0))
        held.append(dict(gen=g, truth=round(t, 5), u2=round(p["u2"], 3), rho_zw=round(p["rho_zw"], 3),
                         gamma=round(p["gamma"], 3), delta=round(p["delta"], 3),
                         lambda_star_median=float(np.median(ls)),
                         lambda_zero_median=float(np.median(lz)),
                         share_sign_lost_first=float(np.mean(np.array(lz) <= np.array(ls)))))
    ls = np.array([x["lambda_star_median"] for x in held])
    lz = np.array([x["lambda_zero_median"] for x in held])
    nonnull = np.array([abs(x["truth"]) >= 0.02 for x in held])
    summ.append(dict(analysis="heldout set B", domain="60 generators", n_rep=args.heldout_reps,
                     lambda_star_median=round(q(ls, 50), 2), lambda_star_q10=round(q(ls, 10), 2),
                     lambda_star_q90=round(q(ls, 90), 2),
                     lambda_zero_median=round(q(lz, 50), 2), lambda_zero_q10=round(q(lz, 10), 2),
                     lambda_zero_q90=round(q(lz, 90), 2),
                     share_gen_lambda_star_le_1_5=round(float(np.mean(ls <= 1.5)), 3),
                     share_gen_lambda_star_le_2=round(float(np.mean(ls <= 2.0)), 3),
                     share_gen_lambda_star_inf=round(float(np.mean(~np.isfinite(ls))), 3),
                     n_gen_nonnull=int(nonnull.sum()),
                     share_nonnull_sign_lost_before_truth=round(
                         float(np.mean([x["share_sign_lost_first"] for x, m in zip(held, nonnull) if m])), 3)))
    write(RES / "msm_bounds_main.csv", main_rows)
    write(RES / "msm_bounds_heldout.csv", held)
    write(RES / "msm_bounds_summary.csv", summ)


def make_table() -> None:
    """Write tables/msm_bounds.tex (LaTeX rows) from results/msm_bounds_summary.csv."""
    names = {"sepsis": "Sepsis", "ards": "ARDS", "cardiac": "STEMI"}
    out = []
    for x in csv.DictReader(open(RES / "msm_bounds_summary.csv")):
        if x["analysis"] != "main":
            continue
        out.append(f"{names[x['domain']]} & {float(x['truth']):.3f} & "
                   f"{float(x['lambda_star_median']):.2f} ({float(x['lambda_star_q10']):.2f}--{float(x['lambda_star_q90']):.2f}) & "
                   f"{float(x['lambda_zero_median']):.2f} ({float(x['lambda_zero_q10']):.2f}--{float(x['lambda_zero_q90']):.2f}) & "
                   f"{float(x['cov_1.0']):.2f} & {float(x['cov_1.5']):.2f} & {float(x['width_1.5']):.2f} & "
                   f"{float(x['excl0_1.5']):.2f} \\\\")
    (RES.parent / "tables").mkdir(exist_ok=True)
    (RES.parent / "tables" / "msm_bounds.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        make_table()
    else:
        main()
        make_table()
