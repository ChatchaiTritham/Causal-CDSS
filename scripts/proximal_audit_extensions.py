"""Additional audit analyses requested in review (v1.0.3).

Part A -- bridge misspecification (2 x 2). The parametric treatment bridge
q_t(Z, A) = 1 + exp(theta'(1, Z, A)) is exactly correct under the main generator (Gaussian
proxies, logistic treatment in U). Here it is broken by adding a quadratic severity term to the
treatment link; the outcome bridge h is made exactly correct by a linear (continuous) outcome.
Cells: {h correct, h wrong} x {q correct, q wrong}. Estimators: proximal 2SLS (h only) and the
DR form of Eq. (pdr), whose point estimate equals proximal IPW (q only) when h is linear and
g = (1, W, A). For the DR form both the plug-in SE (bridges known) and a stacked
estimating-equation sandwich SE (theta_1, theta_0, ATE) are reported.

Part B -- reverse relevance diagnostic: partial F of W in Z ~ (T, W, age), next to the usual
first-stage F of Z in W ~ (T, Z, age).

Part C -- misspecified knowledge base. The knowledge-driven rule is rebuilt from a perturbed
declared model (outcome coefficients x0.7 or x1.3, proxy loadings x0.7, age->Y edge dropped)
and evaluated against the true generator: treated fraction, true value, regret against the
best rule studied (treat-all, optimal here), standard and proximal DR value coverage, and
benefit recall in the youngest age third at benefit thresholds 0.10/0.15/0.20.

Part D -- estimand and convergence checks on the main design: coverage of the proximal value
against the population value V(pi) (200,000-draw super-population) as well as against the
sample-conditional value, for rules that do not depend on a fitted model; DR ATE coverage with
non-converged bridge replications excluded; and an auditor-side subgroup quantity, the proximal
DR contrast V_s(pi) - V_s(treat none) in the youngest third, with its coverage.

Outputs: results/bridge_misspecification.csv, results/kb_misspecification.csv,
         results/estimand_checks.csv
Usage:   python scripts/proximal_audit_extensions.py [--reps 200] [--n 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402

CUT = 0.4307
Q_WRONG = 0.8        # coefficient of (U^2 - 1) in the treatment logit when q is broken
LIN_SCALE = 0.15     # scale of severity/age in the linear (h-correct) outcome
KB_VARIANTS = {
    "declared (correct)": {},
    "outcome coefs x0.7": {"scale_out": 0.7},
    "outcome coefs x1.3": {"scale_out": 1.3},
    "proxy loadings x0.7": {"scale_load": 0.7},
    "age->Y edge dropped": {"drop_age": True},
}


# ----------------------------------------------------------------------------- generators
def simulate_ms(cfg, n, rng, h_ok, q_ok, tau):
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    z = cfg["sev_to_z"] * sev + rng.normal(0, 1, n)
    w = cfg["sev_to_w"] * sev + rng.normal(0, 1, n)
    lin_t = cfg["sev_to_treat"] * sev + cfg["treat_intercept"]
    if not q_ok:
        lin_t = lin_t + Q_WRONG * (sev ** 2 - np.mean(sev ** 2))
    a = (rng.uniform(0, 1, n) < uc.sigmoid(lin_t)).astype(float)
    if h_ok:
        y = tau * a + LIN_SCALE * (cfg["sev_to_out"] * sev + cfg["age_to_out"] * age) \
            + rng.normal(0, 0.5, n)
    else:
        py = uc.sigmoid(cfg["sev_to_out"] * sev + cfg["age_to_out"] * age
                        + cfg["treat_to_out"] * a + cfg["out_intercept"])
        y = (rng.uniform(0, 1, n) < py).astype(float)
    return dict(age=age, severity=sev, z=z, w=w, a=a, y=y)


# ----------------------------------------------------------------------------- bridges
def fit_q(d):
    """Treatment-bridge parameters (theta_1, theta_0) as in uc.treatment_bridge; also flags."""
    n = len(d["a"])
    F = np.column_stack([np.ones(n), d["z"], d["age"]])
    G = np.column_stack([np.ones(n), d["w"], d["age"]])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(F[:, 1:], d["a"])
    eta = np.r_[lr.intercept_, lr.coef_.ravel()]
    th, ok = {}, True
    for arm, start in ((1, -eta), (0, eta)):
        ind = (d["a"] == arm).astype(float)

        def mom(t):
            return ((ind * (1.0 + np.exp(np.clip(F @ t, -50, 50))))[:, None] * G - G).mean(0)

        best = None
        for t0 in (start, np.zeros(3)):
            s = least_squares(mom, t0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.max(np.abs(s.fun)) < np.max(np.abs(best.fun)):
                best = s
            if np.max(np.abs(s.fun)) < 1e-6:
                break
        ok = ok and np.max(np.abs(best.fun)) < 1e-6
        th[arm] = best.x
    return th, F, G, ok


def pipw_sandwich(d):
    """DR/IPW proximal ATE with plug-in and stacked M-estimation sandwich SEs."""
    th, F, G, ok = fit_q(d)
    a, y, n = d["a"], d["y"], len(d["a"])

    def moments(p):
        t1, t0, psi = p[:3], p[3:6], p[6]
        q1 = 1.0 + np.exp(np.clip(F @ t1, -50, 50))
        q0 = 1.0 + np.exp(np.clip(F @ t0, -50, 50))
        m1 = (a * q1)[:, None] * G - G
        m0 = ((1 - a) * q0)[:, None] * G - G
        mp = a * q1 * y - (1 - a) * q0 * y - psi
        return np.column_stack([m1, m0, mp])

    q1 = 1.0 + np.exp(F @ th[1])
    q0 = 1.0 + np.exp(F @ th[0])
    psi_i = a * q1 * y - (1 - a) * q0 * y
    est = float(psi_i.mean())
    plug = float(psi_i.std(ddof=1) / np.sqrt(n))
    p0 = np.r_[th[1], th[0], est]
    J = np.empty((7, 7))
    for j in range(7):
        e = np.zeros(7)
        h = 1e-5 * max(1.0, abs(p0[j]))
        e[j] = h
        J[:, j] = (moments(p0 + e).mean(0) - moments(p0 - e).mean(0)) / (2 * h)
    M = moments(p0)
    Omega = M.T @ M / n
    Ji = np.linalg.pinv(J)
    V = Ji @ Omega @ Ji.T / n
    return est, plug, float(np.sqrt(max(V[6, 6], 0.0))), ok, q1, q0


def reverse_f(d):
    """Partial F of W in Z ~ (T, W, age): relevance in the direction the q-bridge needs."""
    n = len(d["z"])
    full = np.column_stack([np.ones(n), d["a"], d["w"], d["age"]])
    red = np.column_stack([np.ones(n), d["a"], d["age"]])
    rss = lambda X: float(np.sum((d["z"] - X @ np.linalg.lstsq(X, d["z"], rcond=None)[0]) ** 2))
    return (rss(red) - rss(full)) / (rss(full) / (n - full.shape[1]))


def summarise(est, se, truth):
    e, s = np.asarray(est), np.asarray(se)
    cov = float(np.mean(np.abs(e - truth) <= 1.96 * s))
    return dict(bias=round(float(e.mean() - truth), 4),
                mcse=round(float(e.std(ddof=1) / np.sqrt(len(e))), 4),
                emp_sd=round(float(e.std(ddof=1)), 4), median_se=round(float(np.median(s)), 4),
                coverage=round(cov, 3))


# ----------------------------------------------------------------------------- part A
def run_misspec(job):
    domain, n, reps, seed = job
    cfg = uc.DOMAINS[domain]
    truth_logit = uc.true_ate(cfg, 200_000, np.random.default_rng(seed + 7))
    tau = truth_logit
    rows = []
    for h_ok in (True, False):
        for q_ok in (True, False):
            r2, s2, rd, sp, ss, oks = [], [], [], [], [], []
            fails = 0
            for r in range(reps):
                rng = np.random.default_rng(seed + 31 * r + uc.domain_offset(domain, 991)
                                            + 100_000 * (2 * h_ok + q_ok))
                d = simulate_ms(cfg, n, rng, h_ok, q_ok, tau)
                th, se = uc.proximal_se(d)
                est, plug, sand, ok, _, _ = pipw_sandwich(d)
                fails += int(not ok)
                oks.append(bool(ok))
                r2.append(th); s2.append(se)
                rd.append(est); sp.append(plug); ss.append(sand)
            truth = tau if h_ok else truth_logit
            base = dict(domain=domain, h_bridge="correct" if h_ok else "wrong",
                        q_bridge="correct" if q_ok else "wrong", truth=round(truth, 4),
                        reps=reps, q_nonconverged=fails)
            rows.append({**base, "estimator": "proximal_2sls", "se_type": "HC0 two-stage",
                         **summarise(r2, s2, truth)})
            rows.append({**base, "estimator": "proximal_dr", "se_type": "plug-in",
                         **summarise(rd, sp, truth)})
            # the stacked sandwich needs a solved q-bridge; score it on converged replications
            m = np.asarray(oks)
            rows.append({**base, "estimator": "proximal_dr", "se_type": "sandwich",
                         **summarise(np.asarray(rd)[m], np.asarray(ss)[m], truth)})
    print(f"  misspec {domain} done", flush=True)
    return rows


# ----------------------------------------------------------------------------- part C, D
def kd_rule(d, cfg, scale_out=1.0, scale_load=1.0, drop_age=False):
    c = dict(cfg)
    c["sev_to_z"] *= scale_load
    c["sev_to_w"] *= scale_load
    for k in ("sev_to_out", "age_to_out", "treat_to_out"):
        c[k] *= scale_out
    if drop_age:
        c["age_to_out"] = 0.0
    return uc.policy_knowledge(d, c)


def pop_value(cfg, pi_fn, big):
    return uc.true_policy_value(cfg, big, pi_fn(big))


def run_policy(job):
    domain, n, reps, seed = job
    cfg = uc.DOMAINS[domain]
    big = uc.simulate(cfg, 200_000, np.random.default_rng(seed + 4242))
    kb = {name: (lambda dd, kw=kw: kd_rule(dd, cfg, **kw)) for name, kw in KB_VARIANTS.items()}
    fixed = {"treat_all": lambda dd: np.ones(len(dd["a"])),
             "treat_none": lambda dd: np.zeros(len(dd["a"])),
             "age_rule": lambda dd: (dd["age"] > 0).astype(float), **kb}
    vpop = {p: pop_value(cfg, fn, big) for p, fn in fixed.items()}
    acc = {p: {k: [] for k in ("vs", "prox", "prox_se", "dr", "dr_se", "treated",
                                "rec10", "rec15", "rec20", "ctr_est", "ctr_se", "ctr_true")}
           for p in fixed}
    ate = {"est": [], "se": [], "ok": []}
    ff, rf = [], []
    for r in range(reps):
        off = seed + 1000 * r + uc.domain_offset(domain, 997)
        d = uc.simulate(cfg, n, np.random.default_rng(off))
        b = uc.outcome_bridge(d)
        before = uc.BRIDGE_FAILS["count"]
        q = uc.treatment_bridge(d)
        ok = uc.BRIDGE_FAILS["count"] == before
        h = lambda t: b[0] + b[1] * t + b[2] * d["w"] + b[3] * d["age"]
        psi_ate = (2 * d["a"] - 1) * q * (d["y"] - h(d["a"])) + b[1]
        ate["est"].append(psi_ate.mean())
        ate["se"].append(psi_ate.std(ddof=1) / np.sqrt(n))
        ate["ok"].append(ok)
        ff.append(uc.first_stage_f(d)); rf.append(reverse_f(d))
        nuis = uc.fit_nuisances(d, "logistic_recorded")
        young = d["age"] < -CUT
        red = uc.sigmoid(cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"]
                         + cfg["out_intercept"]) - uc.sigmoid(
            cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"]
            + cfg["out_intercept"] + cfg["treat_to_out"])
        for p, fn in fixed.items():
            pi = fn(d)
            psi = h(pi) + (d["a"] == pi) * q * (d["y"] - h(d["a"]))
            psi0 = h(0.0) + (d["a"] == 0) * q * (d["y"] - h(d["a"]))
            o = uc.ope_estimates(d, pi, nuis)
            A = acc[p]
            A["vs"].append(uc.true_policy_value(cfg, d, pi))
            A["prox"].append(psi.mean()); A["prox_se"].append(psi.std(ddof=1) / np.sqrt(n))
            A["dr"].append(o["dr"]); A["dr_se"].append(o["dr_se"])
            A["treated"].append(pi.mean())
            for thr, key in ((0.10, "rec10"), (0.15, "rec15"), (0.20, "rec20")):
                m = young & (red > thr)
                A[key].append(pi[m].mean() if m.any() else np.nan)
            # auditor-side contrast in the youngest third: V_s(pi) - V_s(none), proximal DR
            dlt = (psi - psi0)[young]
            A["ctr_est"].append(dlt.mean()); A["ctr_se"].append(dlt.std(ddof=1) / np.sqrt(young.sum()))
            ys = {k: v[young] for k, v in d.items()}
            A["ctr_true"].append(uc.true_policy_value(cfg, ys, pi[young])
                                 - uc.true_policy_value(cfg, ys, np.zeros(young.sum())))
    best = min(vpop.values())
    rows_kb, rows_est = [], []
    for p in fixed:
        A = {k: np.asarray(v, float) for k, v in acc[p].items()}
        cov = lambda e, s, t: round(float(np.mean(np.abs(e - t) <= 1.96 * s)), 3)
        row = dict(domain=domain, policy=p, reps=reps,
                   treated=round(float(A["treated"].mean()), 3),
                   true_value_pop=round(vpop[p], 4), regret=round(vpop[p] - best, 4),
                   prox_cov_sample=cov(A["prox"], A["prox_se"], A["vs"]),
                   prox_cov_pop=cov(A["prox"], A["prox_se"], vpop[p]),
                   prox_bias=round(float(np.mean(A["prox"] - A["vs"])), 4),
                   dr_cov_sample=cov(A["dr"], A["dr_se"], A["vs"]),
                   dr_cov_pop=cov(A["dr"], A["dr_se"], vpop[p]),
                   recall_young_010=round(float(np.nanmean(A["rec10"])), 3),
                   recall_young_015=round(float(np.nanmean(A["rec15"])), 3),
                   recall_young_020=round(float(np.nanmean(A["rec20"])), 3),
                   recall_young_015_sd=round(float(np.nanstd(A["rec15"], ddof=1)), 3),
                   young_contrast_true=round(float(A["ctr_true"].mean()), 4),
                   young_contrast_bias=round(float(np.mean(A["ctr_est"] - A["ctr_true"])), 4),
                   young_contrast_cov=cov(A["ctr_est"], A["ctr_se"], A["ctr_true"]))
        (rows_kb if p in KB_VARIANTS else rows_est).append(row)
    truth = uc.true_ate(cfg, 200_000, np.random.default_rng(seed + 7))
    e, s, ok = (np.asarray(ate[k]) for k in ("est", "se", "ok"))
    cov_all = float(np.mean(np.abs(e - truth) <= 1.96 * s))
    cov_ok = float(np.mean(np.abs(e[ok] - truth) <= 1.96 * s[ok]))
    rows_est.append(dict(domain=domain, policy="ATE (DR proximal)", reps=reps,
                         treated="", true_value_pop=round(truth, 4), regret="",
                         prox_cov_sample=round(cov_all, 3), prox_cov_pop=round(cov_ok, 3),
                         prox_bias=round(float(e.mean() - truth), 4),
                         dr_cov_sample="", dr_cov_pop="",
                         recall_young_010=f"F={np.mean(ff):.0f}",
                         recall_young_015=f"reverseF={np.mean(rf):.0f}",
                         recall_young_020=f"nonconv={int((~ok).sum())}",
                         recall_young_015_sd="", young_contrast_true="",
                         young_contrast_bias="", young_contrast_cov=""))
    print(f"  policy {domain} done", flush=True)
    return rows_kb, rows_est


def write(rows, name):
    with open(uc.RESULTS / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    jobs = [(dm, a.n, a.reps, a.seed) for dm in uc.DOMAINS]
    with ProcessPoolExecutor() as ex:
        ms = list(ex.map(run_misspec, jobs))
        pol = list(ex.map(run_policy, jobs))
    write([r for o in ms for r in o], "bridge_misspecification.csv")
    write([r for o in pol for r in o[0]], "kb_misspecification.csv")
    write([r for o in pol for r in o[1]], "estimand_checks.csv")


if __name__ == "__main__":
    main()
