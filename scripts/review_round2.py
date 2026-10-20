"""Second-round review analyses (v1.0.4).

Each part answers a specific reviewer request; all numbers in the paper's new supplementary
sections S28-S40 come from the CSVs written here. Parts (``--parts`` selects a subset):

  nlh      bridge misspecification with a correct but NONLINEAR outcome bridge (h quadratic
           in W), so the IPW/DR identity of the linear case breaks; worst-case coverage that
           counts non-converged q solves as interval failures; within-arm first-stage F.
  pvmain   policy value on the main design: proximal coverage with failed q solves treated as
           'inconclusive' (Algorithm 1), q-weight tails, cross-fitted (K=5) standard DR at three
           clip levels with recorded and oracle nuisances, IPW/SNIPW/direct-method bias,
           withhold share among guideline-indicated patients, the violated-restriction runs
           with the same inconclusive rule, and per-third subgroup diagnostics with paired
           knowledge-driven-minus-risk-rule differences (per-replication values saved).
  pvboot   nonparametric bootstrap of the proximal policy value refitting both bridges.
  calib    trial-calibrated effect sizes and baseline mortality, with default and with
           AUROC-calibrated proxy loadings, plus the gamma sweep on the calibrated design.
  harm     an age-dependent harm term (effects change sign in the oldest third): harm recall,
           net benefit / decision curve, and its proximal estimate.
  lhs      a held-out Latin-hypercube family of 60 generators (second confounder, correlated
           proxy noise, joint violations, null and harmful effects) on which Algorithm 1's
           decisions are scored; plus three fixed stress arms on the main domains.
  falsify  empirical size and power of Fisher-z and of the generalised covariance measure
           (GCM, Shah & Peters 2020) for the implications of the declared graphs.
  dml      repeated cross-fitting (5 repetitions, median aggregation), nuisance diagnostics,
           three learners incl. a tuned one, and the true overlap-weighted PLR estimand.
  ovb      DoubleML omitted-variable-bias bounds (Chernozhukov et al.) with age and Z as
           benchmarks, and cross-checks of the hand-written AIPW (DoubleML) and proximal 2SLS
           (linearmodels IV2SLS).
  cross    crossing points on a dense grid (0.10-0.30, step 0.02, 200 replications) against
           both the linear proxy-adjusted and a cross-fitted AIPW comparator, bootstrap CIs.
  overid   an over-identified design (two Z) and the Hansen J test's power against gamma/delta.
  binbr    a logistic outcome bridge for the binary outcome (GMM), alone and in DR form.
  shift    transport of site-A bridges to a site B with shifted age or proxy loadings.
  diffnoise ARDS with group-differential measurement error in W.
  prop     Proposition 1 with general noise variances: exact population check, simulation
           check, and the linear-analogue analytic crossing points.
  conv     non-convergence counts per cell for the weak-proxy and nonlinear sweeps.

Usage: python scripts/review_round2.py [--parts all] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import least_squares, brentq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402
import proximal_robustness as pr  # noqa: E402
import proximal_audit_extensions as ext  # noqa: E402

SEED = 42
CUT = 0.4307
GSTAR_DR, DSTAR_DR = 0.28, 0.23     # design-stage DR crossing points (min over domains)
F_MIN = 16.0
LR = lambda: LogisticRegression(C=1e6, max_iter=1000)


# =========================================================================== helpers
def off(domain, mod=997):
    return uc.domain_offset(domain, mod)


def bridges(d):
    """Linear outcome bridge, treatment bridge, and whether the q solve converged."""
    b = uc.outcome_bridge(d)
    before = uc.BRIDGE_FAILS["count"]
    q = uc.treatment_bridge(d)
    return b, q, uc.BRIDGE_FAILS["count"] == before


def h_lin(b, d, t):
    return b[0] + b[1] * t + b[2] * d["w"] + b[3] * d["age"]


def dr_ate(d, b, q):
    psi = (2 * d["a"] - 1) * q * (d["y"] - h_lin(b, d, d["a"])) + b[1]
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


def cf_nuis(d, cols, seed, folds=5):
    X = np.column_stack([d[c] for c in cols])
    a, y = d["a"], d["y"]
    e, mu1, mu0 = (np.zeros(len(a)) for _ in range(3))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        e[te] = LR().fit(X[tr], a[tr]).predict_proba(X[te])[:, 1]
        t1, t0 = tr[a[tr] == 1], tr[a[tr] == 0]
        mu1[te] = LR().fit(X[t1], y[t1]).predict_proba(X[te])[:, 1]
        mu0[te] = LR().fit(X[t0], y[t0]).predict_proba(X[te])[:, 1]
    return e, mu1, mu0


def dr_value(d, pi, e, mu1, mu0, clip):
    e = np.clip(e, clip, 1 - clip)
    w = (d["a"] == pi) / (pi * e + (1 - pi) * (1 - e))
    psi = pi * mu1 + (1 - pi) * mu0 + w * (d["y"] - (d["a"] * mu1 + (1 - d["a"]) * mu0))
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


def aipw_ate(d, e, mu1, mu0, clip=0.01):
    e = np.clip(e, clip, 1 - clip)
    a, y = d["a"], d["y"]
    psi = mu1 - mu0 + a * (y - mu1) / e - (1 - a) * (y - mu0) / (1 - e)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


def cover(est, se, truth):
    est, se = np.asarray(est, float), np.asarray(se, float)
    return float(np.mean(np.abs(est - truth) <= 1.96 * se))


def r4(x):
    return round(float(x), 4)


def write(rows, name):
    with open(uc.RESULTS / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name, flush=True)


def truth_mc(sim_fn, n=200_000, seed=SEED + 7):
    """E[Y(1)] - E[Y(0)] for a generator exposing its outcome linear predictor."""
    return sim_fn(n, np.random.default_rng(seed))


# =========================================================================== part nlh
C2 = 0.10


def simulate_nlh(cfg, n, rng, q_ok, tau):
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    z = cfg["sev_to_z"] * sev + rng.normal(0, 1, n)
    w = cfg["sev_to_w"] * sev + rng.normal(0, 1, n)
    lin_t = cfg["sev_to_treat"] * sev + cfg["treat_intercept"]
    if not q_ok:
        lin_t = lin_t + ext.Q_WRONG * (sev ** 2 - np.mean(sev ** 2))
    a = (rng.uniform(0, 1, n) < uc.sigmoid(lin_t)).astype(float)
    y = (tau * a + ext.LIN_SCALE * (cfg["sev_to_out"] * sev + cfg["age_to_out"] * age)
         + C2 * sev ** 2 + rng.normal(0, 0.5, n))
    return dict(age=age, severity=sev, z=z, w=w, a=a, y=y)


def quad_bridge(d):
    """h(W,T,A) = b0 + bT T + bW W + b2 W^2 + bA A by 2SLS, instruments (1,T,Z,Z^2,A)."""
    n = len(d["y"])
    Zm = np.column_stack([np.ones(n), d["a"], d["z"], d["z"] ** 2, d["age"]])
    X = np.column_stack([np.ones(n), d["a"], d["w"], d["w"] ** 2, d["age"]])
    Xh = Zm @ np.linalg.lstsq(Zm, X, rcond=None)[0]
    b = np.linalg.lstsq(Xh, d["y"], rcond=None)[0]
    e = d["y"] - X @ b
    bread = np.linalg.inv(Xh.T @ Xh)
    se = np.sqrt((bread @ ((Xh * e[:, None] ** 2).T @ Xh) @ bread)[1, 1])
    return b, float(se)


def within_arm_f(d):
    out = []
    for t in (1.0, 0.0):
        m = d["a"] == t
        n = int(m.sum())
        y = d["w"][m]
        full = np.column_stack([np.ones(n), d["z"][m], d["age"][m]])
        red = np.column_stack([np.ones(n), d["age"][m]])
        rss = lambda X: float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
        out.append((rss(red) - rss(full)) / (rss(full) / (n - 3)))
    return out


def rep_nlh(domain, r, cell):
    cfg = uc.DOMAINS[domain]
    tau = uc.true_ate(cfg, 200_000, np.random.default_rng(SEED + 7))
    kind, q_ok = cell
    if kind == "orig":   # the four cells of S16 (same seeds), for worst-case coverage
        h_ok, q_ok2 = q_ok
        rng = np.random.default_rng(SEED + 31 * r + off(domain, 991) + 100_000 * (2 * h_ok + q_ok2))
        d = ext.simulate_ms(cfg, 8000, rng, h_ok, q_ok2, tau)
        est, plug, sand, ok, _, _ = ext.pipw_sandwich(d)
        return dict(est=est, plug=plug, sand=sand, ok=ok, truth=tau)
    if kind == "armF":
        rng = np.random.default_rng(SEED + 1000 * r + off(domain))
        d = uc.simulate(cfg, 8000, rng) if q_ok else ext.simulate_ms(cfg, 8000, rng, False, False, tau)
        f1, f0 = within_arm_f(d)
        return dict(pooled=uc.first_stage_f(d), f1=f1, f0=f0)
    rng = np.random.default_rng(SEED + 37 * r + off(domain, 991) + 300_000 * (1 + q_ok))
    d = simulate_nlh(cfg, 8000, rng, q_ok, tau)
    lin, lin_se = uc.proximal_se(d)
    bq, bq_se = quad_bridge(d)
    th, F, G, ok = ext.fit_q(d)
    q = np.where(d["a"] == 1, 1 + np.exp(F @ th[1]), 1 + np.exp(F @ th[0]))
    h = lambda t: bq[0] + bq[1] * t + bq[2] * d["w"] + bq[3] * d["w"] ** 2 + bq[4] * d["age"]
    sgn = 2 * d["a"] - 1
    psi = sgn * q * (d["y"] - h(d["a"])) + bq[1]
    ipw = sgn * q * d["y"]
    rn = np.sqrt(len(psi))
    return dict(truth=tau, lin=lin, lin_se=lin_se, quad=bq[1], quad_se=bq_se,
                dr=psi.mean(), dr_se=psi.std(ddof=1) / rn, ipw=ipw.mean(),
                ipw_se=ipw.std(ddof=1) / rn, ok=ok)


def agg_nlh(res):
    rows = []
    for (domain, cell), reps in sorted(res.items(), key=lambda x: str(x[0])):
        kind, q = cell
        if kind == "armF":
            P = {k: np.array([x[k] for x in reps]) for k in ("pooled", "f1", "f0")}
            rows.append(dict(domain=domain, cell=f"F by arm ({'main' if q else 'q-wrong'} generator)",
                             estimator="first-stage F (median)", se_type="",
                             bias=round(float(np.median(P["pooled"])), 1),
                             emp_sd=round(float(np.median(P["f1"])), 1),
                             median_se=round(float(np.median(P["f0"])), 1),
                             coverage="", coverage_worst_case="", nonconverged="", reps=len(reps)))
            continue
        if kind == "orig":
            h_ok, q_ok = q
            ok = np.array([x["ok"] for x in reps])
            est = np.array([x["est"] for x in reps])
            t = reps[0]["truth"]
            for se_type in ("plug", "sand"):
                se = np.array([x[se_type] for x in reps])
                hit = np.abs(est - t) <= 1.96 * se
                rows.append(dict(domain=domain,
                                 cell=f"h {'correct (linear)' if h_ok else 'wrong'} / q {'correct' if q_ok else 'wrong'}",
                                 estimator="proximal DR", se_type="plug-in" if se_type == "plug" else "sandwich",
                                 bias=r4(est.mean() - t), emp_sd=r4(est.std(ddof=1)),
                                 median_se=r4(np.median(se)),
                                 coverage=round(float(hit[ok].mean()), 3),
                                 coverage_worst_case=round(float((hit & ok).mean()), 3),
                                 nonconverged=int((~ok).sum()), reps=len(reps)))
            continue
        t = reps[0]["truth"]
        ok = np.array([x["ok"] for x in reps])
        for name, lab in (("lin", "2SLS, linear h (wrong)"), ("quad", "2SLS, quadratic h (correct)"),
                          ("ipw", "proximal IPW (q only)"), ("dr", "proximal DR, quadratic h")):
            est = np.array([x[name] for x in reps])
            se = np.array([x[name + "_se"] for x in reps])
            hit = np.abs(est - t) <= 1.96 * se
            rows.append(dict(domain=domain, cell=f"h correct (quadratic in W) / q {'correct' if q else 'wrong'}",
                             estimator=lab, se_type="plug-in/HC0", bias=r4(est.mean() - t),
                             emp_sd=r4(est.std(ddof=1)), median_se=r4(np.median(se)),
                             coverage=round(float(hit[ok].mean()), 3) if name in ("ipw", "dr") else round(float(hit.mean()), 3),
                             coverage_worst_case=round(float((hit & ok).mean()), 3) if name in ("ipw", "dr") else "",
                             nonconverged=int((~ok).sum()), reps=len(reps)))
    write(rows, "r2_bridge_nonlinear.csv")


# =========================================================================== part pvmain
POLS = ("age_rule", "treat_all", "treat_none", "cdss_top50", "cdss_top25", "knowledge")


def true_value_g(cfg, d, pi, gamma=0.0):
    base = (cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"]
            + gamma * d["z"] + cfg["out_intercept"])
    return float(np.mean(uc.sigmoid(base + cfg["treat_to_out"] * pi)))


def rep_pvmain(domain, r, cell):
    cfg = uc.DOMAINS[domain]
    o = SEED + 1000 * r + off(domain)
    gamma = cell
    if gamma == 0.0:
        d = uc.simulate(cfg, 8000, np.random.default_rng(o))
    else:
        d = pr.simulate_violated(cfg, 8000, np.random.default_rng(o + 77), gamma)
    train = uc.simulate(cfg, 8000, np.random.default_rng(o + 500_000))
    pols = uc.make_policies(d, train, cfg)
    pols["age_rule"] = (d["age"] > 0).astype(float)
    b, q, ok = bridges(d)
    out = dict(ok=ok, qmax=float(q.max()), q99=float(np.quantile(q, 0.99)))
    h = lambda t: h_lin(b, d, t)
    psi0 = h(0.0) + (d["a"] == 0) * q * (d["y"] - h(d["a"]))
    if gamma == 0.0:
        nin = uc.fit_nuisances(d, "logistic_recorded")
        cfr = cf_nuis(d, ["age", "z", "w"], r)
        cfo = cf_nuis(d, ["age", "z", "w", "severity"], r)
        ind = d["severity"] > np.quantile(d["severity"], 0.7)
    for p in POLS:
        pi = pols[p]
        v = true_value_g(cfg, d, pi, gamma)
        psi = h(pi) + (d["a"] == pi) * q * (d["y"] - h(d["a"]))
        rec = dict(truth=v, prox=psi.mean(), prox_se=psi.std(ddof=1) / np.sqrt(len(psi)))
        if gamma == 0.0:
            oo = uc.ope_estimates(d, pi, nin)
            rec.update(dr_in=oo["dr"], dr_in_se=oo["dr_se"], ipw=oo["ipw"], snipw=oo["snipw"],
                       dm=oo["outcome_only"], withhold=float(np.mean(pi[ind] == 0)),
                       treated=float(pi.mean()))
            for clip in (0.01, 0.02, 0.05):
                rec[f"cf_rec_{clip}"], rec[f"cf_rec_{clip}_se"] = dr_value(d, pi, *cfr, clip)
            rec["cf_orc"], rec["cf_orc_se"] = dr_value(d, pi, *cfo, 0.01)
        out[p] = rec
    if gamma != 0.0:
        return out
    # ---- subgroup diagnostics (age thirds)
    g = np.where(d["age"] < -CUT, 0, np.where(d["age"] > CUT, 2, 1))
    red = (uc.sigmoid(cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"] + cfg["out_intercept"])
           - uc.sigmoid(cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"]
                        + cfg["out_intercept"] + cfg["treat_to_out"]))
    sg = []
    for k in range(3):
        m = g == k
        ds = {kk: vv[m] for kk, vv in d.items()}
        _, _, ok_s = bridges(ds)
        f_s = uc.first_stage_f(ds)
        ben = red[m] > uc.KB_MIN_REDUCTION
        row = dict(third=k, n=int(m.sum()), ok=ok_s, F=f_s)
        for p in ("knowledge", "cdss_top50"):
            pi = pols[p]
            v_s = (true_value_g(cfg, ds, np.zeros(m.sum())) - true_value_g(cfg, ds, pi[m]))
            psi = h(pi) + (d["a"] == pi) * q * (d["y"] - h(d["a"]))
            dl = (psi - psi0)[m]
            row[p + "_recall"] = float(pi[m][ben].mean()) if ben.any() else np.nan
            row[p + "_gain"] = v_s
            row[p + "_ntreated"] = int(pi[m].sum())
            row[p + "_ctr_cov"] = float(abs(dl.mean() + v_s) <= 1.96 * dl.std(ddof=1) / np.sqrt(m.sum()))
        sg.append(row)
    out["sg"] = sg
    return out


def agg_pvmain(res):
    rows, sgrows, sgsum, viol = [], [], [], []
    for (domain, gamma), reps in sorted(res.items()):
        ok = np.array([x["ok"] for x in reps])
        for p in POLS:
            R = {k: np.array([x[p][k] for x in reps], float) for k in reps[0][p]}
            t = R["truth"]
            hit = np.abs(R["prox"] - t) <= 1.96 * R["prox_se"]
            base = dict(domain=domain, gamma=gamma, policy=p, reps=len(reps),
                        inconclusive_rate=round(float((~ok).mean()), 3),
                        prox_cov_all=round(float(hit.mean()), 3),
                        prox_cov_conclusive=round(float(hit[ok].mean()), 3),
                        prox_bias_conclusive=r4(np.mean(R["prox"][ok] - t[ok])))
            if gamma != 0.0:
                viol.append(base)
                continue
            base.update(
                q_max_median=round(float(np.median([x["qmax"] for x in reps])), 1),
                q_p99_median=round(float(np.median([x["q99"] for x in reps])), 2),
                treated=round(float(R["treated"].mean()), 3),
                withhold_share_indicated=round(float(R["withhold"].mean()), 3),
                dr_insample_cov=round(cover(R["dr_in"], R["dr_in_se"], t), 3),
                cf_rec_cov_clip001=round(cover(R["cf_rec_0.01"], R["cf_rec_0.01_se"], t), 3),
                cf_rec_cov_clip002=round(cover(R["cf_rec_0.02"], R["cf_rec_0.02_se"], t), 3),
                cf_rec_cov_clip005=round(cover(R["cf_rec_0.05"], R["cf_rec_0.05_se"], t), 3),
                cf_oracle_cov_clip001=round(cover(R["cf_orc"], R["cf_orc_se"], t), 3),
                bias_ipw=r4(np.mean(R["ipw"] - t)), bias_snipw=r4(np.mean(R["snipw"] - t)),
                bias_dm=r4(np.mean(R["dm"] - t)), bias_dr=r4(np.mean(R["dr_in"] - t)),
                rmse_ipw=r4(np.sqrt(np.mean((R["ipw"] - t) ** 2))),
                rmse_snipw=r4(np.sqrt(np.mean((R["snipw"] - t) ** 2))),
                rmse_dm=r4(np.sqrt(np.mean((R["dm"] - t) ** 2))),
                rmse_dr=r4(np.sqrt(np.mean((R["dr_in"] - t) ** 2))))
            rows.append(base)
        if gamma != 0.0:
            continue
        for r, x in enumerate(reps):
            for s in x["sg"]:
                sgrows.append(dict(domain=domain, rep=r, **{k: (round(v, 5) if isinstance(v, float) else v)
                                                            for k, v in s.items()}))
        for k, lab in enumerate(("youngest", "middle", "oldest")):
            S = [x["sg"][k] for x in reps]
            dre = np.array([s["knowledge_recall"] - s["cdss_top50_recall"] for s in S])
            dga = np.array([s["knowledge_gain"] - s["cdss_top50_gain"] for s in S])
            fin = np.isfinite(dre)
            sgsum.append(dict(
                domain=domain, third=lab, n_median=int(np.median([s["n"] for s in S])),
                nonconverged=int(sum(not s["ok"] for s in S)),
                F_median=round(float(np.median([s["F"] for s in S])), 0),
                kd_treated_min=int(min(s["knowledge_ntreated"] for s in S)),
                rr_treated_min=int(min(s["cdss_top50_ntreated"] for s in S)),
                recall_diff_mean=round(float(dre[fin].mean()), 3) if fin.any() else "",
                recall_diff_lo=round(float(np.percentile(dre[fin], 2.5)), 3) if fin.any() else "",
                recall_diff_hi=round(float(np.percentile(dre[fin], 97.5)), 3) if fin.any() else "",
                gain_diff_mean=r4(dga.mean()), gain_diff_lo=r4(np.percentile(dga, 2.5)),
                gain_diff_hi=r4(np.percentile(dga, 97.5)),
                kd_contrast_cov=round(float(np.mean([s["knowledge_ctr_cov"] for s in S])), 3),
                rr_contrast_cov=round(float(np.mean([s["cdss_top50_ctr_cov"] for s in S])), 3)))
    write(rows, "r2_policy_value_main.csv")
    write(viol, "r2_policy_value_violation.csv")
    write(sgsum, "r2_subgroup_diagnostics.csv")
    write(sgrows, "r2_subgroup_per_replication.csv")


# =========================================================================== part pvboot
B_BOOT = 100


def rep_pvboot(domain, r, cell):
    cfg = uc.DOMAINS[domain]
    o = SEED + 1000 * r + off(domain)
    d = uc.simulate(cfg, 8000, np.random.default_rng(o))
    train = uc.simulate(cfg, 8000, np.random.default_rng(o + 500_000))
    pols = uc.make_policies(d, train, cfg)
    pols["age_rule"] = (d["age"] > 0).astype(float)
    b, q, ok = bridges(d)
    rec = {}
    for p in POLS:
        pi = pols[p]
        psi = h_lin(b, d, pi) + (d["a"] == pi) * q * (d["y"] - h_lin(b, d, d["a"]))
        rec[p] = dict(truth=uc.true_policy_value(cfg, d, pi), est=psi.mean(),
                      plug=psi.std(ddof=1) / np.sqrt(len(psi)), boot=[])
    rng = np.random.default_rng(o + 9_000_000)
    n = len(d["y"])
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        db = {k: v[idx] for k, v in d.items()}
        bb, qb, _ = bridges(db)
        for p in POLS:
            pi = pols[p][idx]
            psi = h_lin(bb, db, pi) + (db["a"] == pi) * qb * (db["y"] - h_lin(bb, db, db["a"]))
            rec[p]["boot"].append(psi.mean())
    rec["ok"] = ok
    return rec


def agg_pvboot(res):
    rows = []
    for (domain, _), reps in sorted(res.items()):
        for p in POLS:
            t = np.array([x[p]["truth"] for x in reps])
            e = np.array([x[p]["est"] for x in reps])
            plug = np.array([x[p]["plug"] for x in reps])
            bse = np.array([np.std(x[p]["boot"], ddof=1) for x in reps])
            lo = np.array([np.percentile(x[p]["boot"], 2.5) for x in reps])
            hi = np.array([np.percentile(x[p]["boot"], 97.5) for x in reps])
            rows.append(dict(domain=domain, policy=p, reps=len(reps), resamples=B_BOOT,
                             plugin_cov=round(cover(e, plug, t), 3),
                             boot_wald_cov=round(cover(e, bse, t), 3),
                             boot_pct_cov=round(float(np.mean((lo <= t) & (t <= hi))), 3),
                             se_ratio_boot_over_plugin=round(float(np.median(bse / plug)), 3),
                             emp_sd=r4(np.std(e - t, ddof=1)), median_boot_se=r4(np.median(bse))))
    write(rows, "r2_policy_value_bootstrap.csv")


# =========================================================================== part calib
# Trial anchors (absolute 28/30-day or in-hospital mortality): sepsis/septic shock baseline 35%,
# ARR 0.03 (about three hours' delay at OR 1.04/h, Seymour 2017); ARDS prone positioning,
# PROSEVA (Guerin 2013) 32.8% -> 16.0%; STEMI primary PCI vs fibrinolysis 9% -> 7% (Keeley 2003).
ANCHORS = {"sepsis": (0.35, 0.03), "ards": (0.328, 0.168), "cardiac": (0.09, 0.02)}


def calibrate(cfg, p0, arr, seed=SEED + 55):
    rng = np.random.default_rng(seed)
    age = rng.normal(0, 1, 200_000)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, 200_000)
    lin = cfg["sev_to_out"] * sev + cfg["age_to_out"] * age
    k = brentq(lambda k: np.mean(uc.sigmoid(lin + k)) - p0, -15, 10)
    bt = brentq(lambda b: np.mean(uc.sigmoid(lin + k + b)) - (p0 - arr), -10, 0)
    return dict(cfg, out_intercept=k, treat_to_out=bt)


def proxy_auc_scale(cfg, target=0.70, seed=SEED + 56):
    from sklearn.metrics import roc_auc_score

    def auc(s):
        c = dict(cfg, sev_to_z=cfg["sev_to_z"] * s, sev_to_w=cfg["sev_to_w"] * s)
        d = uc.simulate(c, 100_000, np.random.default_rng(seed))
        return roc_auc_score(d["y"], d["w"]) - target
    lo, hi = 0.02, 3.0
    return brentq(auc, lo, hi) if auc(lo) < 0 < auc(hi) else 1.0


from functools import lru_cache


@lru_cache(maxsize=None)
def calib_cfg(domain, regime):
    cfg = calibrate(uc.DOMAINS[domain], *ANCHORS[domain])
    if regime == "calibrated effect + proxy AUROC 0.70":
        s = proxy_auc_scale(cfg)
        cfg = dict(cfg, sev_to_z=cfg["sev_to_z"] * s, sev_to_w=cfg["sev_to_w"] * s, scale=s)
    return cfg


def rep_calib(domain, r, cell):
    regime, gamma = cell
    cfg = calib_cfg(domain, regime)
    truth = pr.true_ate_violated(cfg, 200_000, np.random.default_rng(SEED + 7), gamma)
    d = pr.simulate_violated(cfg, 8000, np.random.default_rng(SEED + 613 * r + off(domain, 97)), gamma)
    out = dict(truth=truth, F=uc.first_stage_f(d))
    out["proxy"], out["proxy_se"] = uc.backdoor_proxy_se(d)
    out["2sls"], out["2sls_se"] = uc.proximal_se(d)
    b, q, ok = bridges(d)
    out["dr"], out["dr_se"] = dr_ate(d, b, q)
    out["ok"] = ok
    return out


def agg_calib(res):
    rows, cross = [], []
    for (domain, (regime, gamma)), reps in sorted(res.items()):
        t = reps[0]["truth"]
        cfg = calib_cfg(domain, regime)
        for e in ("proxy", "2sls", "dr"):
            est = np.array([x[e] for x in reps]); se = np.array([x[e + "_se"] for x in reps])
            rows.append(dict(domain=domain, regime=regime, gamma=gamma, estimator=e,
                             baseline_mortality=ANCHORS[domain][0], target_arr=ANCHORS[domain][1],
                             beta_T=round(cfg["treat_to_out"], 3), intercept=round(cfg["out_intercept"], 3),
                             loading_scale=round(cfg.get("scale", 1.0), 3),
                             true_ate=r4(t), bias=r4(est.mean() - t),
                             bias_pct_of_effect=round(100 * float(est.mean() - t) / abs(t), 1),
                             abs_bias=r4(abs(est.mean() - t)), mcse=r4(est.std(ddof=1) / np.sqrt(len(est))),
                             rmse=r4(np.sqrt(np.mean((est - t) ** 2))), coverage=round(cover(est, se, t), 3),
                             F_median=round(float(np.median([x["F"] for x in reps])), 1),
                             nonconverged=int(sum(not x["ok"] for x in reps)), reps=len(reps)))
    for domain in uc.DOMAINS:
        for regime in {k[1][0] for k in res if k[0] == domain}:
            sub = [r for r in rows if r["domain"] == domain and r["regime"] == regime]
            grid = sorted({r["gamma"] for r in sub})
            if len(grid) < 3:
                continue
            for e in ("2sls", "dr"):
                fake = [dict(gamma_z_to_y=g, estimator=("proximal" if e == "2sls" else "proximal_dr"),
                             abs_bias=next(r["abs_bias"] for r in sub if r["gamma"] == g and r["estimator"] == e))
                        for g in grid]
                fake += [dict(gamma_z_to_y=g, estimator="backdoor_proxy",
                              abs_bias=next(r["abs_bias"] for r in sub if r["gamma"] == g and r["estimator"] == "proxy"))
                         for g in grid]
                gs = pr.crossing(fake, "gamma_z_to_y", "proximal" if e == "2sls" else "proximal_dr")
                cross.append(dict(domain=domain, regime=regime, estimator=e,
                                  gamma_star=gs if gs is not None else "none in [0,0.5]"))
    write(rows, "r2_calibrated.csv")
    if cross:
        write(cross, "r2_calibrated_crossing.csv")


# =========================================================================== part harm
COSTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


def harm_cfg(domain):
    cfg = dict(uc.DOMAINS[domain])
    cfg["treat_age"] = -cfg["treat_to_out"] / 0.8    # logit effect changes sign at age 0.8
    return cfg


def harm_logit(cfg, d, t):
    return (cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"] + cfg["out_intercept"]
            + (cfg["treat_to_out"] + cfg["treat_age"] * d["age"]) * t)


def simulate_harm(cfg, n, rng):
    d = uc.simulate(cfg, n, rng)
    d["y"] = (rng.uniform(0, 1, n) < uc.sigmoid(harm_logit(cfg, d, d["a"]))).astype(float)
    return d


def bridge_interact(d):
    n = len(d["y"])
    Zm = np.column_stack([np.ones(n), d["a"], d["z"], d["age"], d["a"] * d["age"]])
    X = np.column_stack([np.ones(n), d["a"], d["w"], d["age"], d["a"] * d["age"]])
    Xh = Zm @ np.linalg.lstsq(Zm, X, rcond=None)[0]
    b = np.linalg.lstsq(Xh, d["y"], rcond=None)[0]
    return lambda t: b[0] + b[1] * t + b[2] * d["w"] + b[3] * d["age"] + b[4] * t * d["age"]


def rep_harm(domain, r, cell):
    cfg = harm_cfg(domain)
    o = SEED + 1000 * r + off(domain) + 7_000_000
    d = simulate_harm(cfg, 8000, np.random.default_rng(o))
    train = simulate_harm(cfg, 8000, np.random.default_rng(o + 500_000))
    ytrue0 = uc.sigmoid(harm_logit(cfg, d, 0.0))
    ytrue1 = uc.sigmoid(harm_logit(cfg, d, 1.0))
    arr = ytrue0 - ytrue1
    sev_hat = uc.posterior_severity(cfg, d)
    dd = dict(d, severity=sev_hat)
    pred_arr_aware = uc.sigmoid(harm_logit(cfg, dd, 0.0)) - uc.sigmoid(harm_logit(cfg, dd, 1.0))
    pols = {"treat_all": np.ones(8000), "treat_none": np.zeros(8000),
            "kd_ignores_harm": uc.policy_knowledge(d, cfg),
            "cdss_top50": uc.policy_cdss(d, 0.5, train)}
    for c in COSTS:
        pols[f"kd_harm_aware@{c:.2f}"] = (pred_arr_aware > max(c, 1e-9)).astype(float)
    hq = bridge_interact(d)
    before = uc.BRIDGE_FAILS["count"]
    q = uc.treatment_bridge(d)
    ok = uc.BRIDGE_FAILS["count"] == before
    psi0 = hq(0.0) + (d["a"] == 0) * q * (d["y"] - hq(d["a"]))
    harmed = arr < 0
    out = dict(ok=ok, harmed_share=float(harmed.mean()))
    for p, pi in pols.items():
        gain_true = float(np.mean(ytrue0) - np.mean(np.where(pi == 1, ytrue1, ytrue0)))
        psi = hq(pi) + (d["a"] == pi) * q * (d["y"] - hq(d["a"]))
        dl = psi0 - psi
        out[p] = dict(treated=float(pi.mean()), harm_recall=float(pi[harmed].mean()),
                      gain=gain_true, gain_est=float(dl.mean()),
                      gain_se=float(dl.std(ddof=1) / np.sqrt(len(dl))),
                      value=float(np.mean(np.where(pi == 1, ytrue1, ytrue0))))
    return out


def agg_harm(res):
    rows = []
    for (domain, _), reps in sorted(res.items()):
        for p in [k for k in reps[0] if isinstance(reps[0][k], dict)]:
            R = {k: np.array([x[p][k] for x in reps]) for k in reps[0][p]}
            for c in COSTS:
                if p.startswith("kd_harm_aware") and not p.endswith(f"{c:.2f}"):
                    continue
                nb_true = R["gain"] - c * R["treated"]
                nb_est = R["gain_est"] - c * R["treated"]
                rows.append(dict(domain=domain, policy=p.split("@")[0], cost_threshold=c,
                                 treated=round(float(R["treated"].mean()), 3),
                                 harmed_share=round(float(np.mean([x["harmed_share"] for x in reps])), 3),
                                 harm_recall=round(float(R["harm_recall"].mean()), 3),
                                 true_value=r4(R["value"].mean()),
                                 net_benefit_true=r4(nb_true.mean()),
                                 net_benefit_proximal_bias=r4(np.mean(nb_est - nb_true)),
                                 net_benefit_proximal_cov=round(cover(nb_est, R["gain_se"], nb_true), 3),
                                 nonconverged=int(sum(not x["ok"] for x in reps)), reps=len(reps)))
    write(rows, "r2_harm_decision_curve.csv")


# =========================================================================== part lhs
LHS_N, LHS_REPS, LHS_SIZE = 60, 40, 4000
LHS_RANGES = dict(age_to_sev=(0.3, 0.8), sev_to_treat=(0.4, 1.6), treat_intercept=(-1.0, 0.5),
                  sev_to_out=(0.6, 1.4), age_to_out=(0.1, 0.5), treat_to_out=(-1.4, 0.4),
                  out_intercept=(-1.5, 0.0), sev_to_z=(0.2, 1.3), sev_to_w=(0.2, 1.3),
                  u2=(0.0, 0.6), rho_zw=(0.0, 0.3), gamma=(0.0, 0.4), delta=(0.0, 0.4))


def lhs_design():
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(LHS_RANGES), seed=2027).random(LHS_N)
    keys = list(LHS_RANGES)
    return [{k: LHS_RANGES[k][0] + u[i, j] * (LHS_RANGES[k][1] - LHS_RANGES[k][0])
             for j, k in enumerate(keys)} for i in range(LHS_N)]


def sim_general(p, n, rng):
    age = rng.normal(0, 1, n)
    sev = p["age_to_sev"] * age + rng.normal(0, 1, n)
    u2 = rng.normal(0, 1, n)
    e0 = rng.normal(0, 1, n)
    rho = p.get("rho_zw", 0.0)
    ez = np.sqrt(1 - rho) * rng.normal(0, 1, n) + np.sqrt(rho) * e0
    ew = np.sqrt(1 - rho) * rng.normal(0, 1, n) + np.sqrt(rho) * e0
    z = p["sev_to_z"] * sev + ez
    a = (rng.uniform(0, 1, n) < uc.sigmoid(p["sev_to_treat"] * sev + p.get("u2", 0) * u2
                                           + p["treat_intercept"])).astype(float)
    w = p["sev_to_w"] * sev + p.get("delta", 0) * a + ew
    eta = (p["sev_to_out"] * sev + p["age_to_out"] * age + p.get("gamma", 0) * z
           + p.get("u2", 0) * u2 + p["out_intercept"])
    y = (rng.uniform(0, 1, n) < uc.sigmoid(eta + p["treat_to_out"] * a)).astype(float)
    return dict(age=age, severity=sev, z=z, w=w, a=a, y=y, eta=eta)


def truth_general(p):
    d = sim_general(p, 400_000, np.random.default_rng(SEED + 7))
    return float(np.mean(uc.sigmoid(d["eta"] + p["treat_to_out"]) - uc.sigmoid(d["eta"])))


def arm_params(domain, arm):
    cfg = dict(uc.DOMAINS[domain])
    cross = {r["domain"]: r for r in csv.DictReader(open(uc.RESULTS / "proximal_crossing.csv"))}[domain]
    if arm == "second confounder U2 (0.5)":
        cfg["u2"] = 0.5
    elif arm == "Z-W residual correlation 0.3":
        cfg["rho_zw"] = 0.3
    elif arm == "joint gamma*/2 + delta*/2 (DR)":
        cfg["gamma"] = float(cross["gamma_star_dr"]) / 2
        cfg["delta"] = float(cross["delta_star_dr"]) / 2
    return cfg


def rep_lhs(key, r, cell):
    if key == "lhs":
        p, n, sd = lhs_design()[cell], LHS_SIZE, SEED + 5_000_000 + 10_000 * cell + r
    else:
        p, n, sd = arm_params(key, cell), 8000, SEED + 1000 * r + off(key) + 3_000_000
    d = sim_general(p, n, np.random.default_rng(sd))
    out = dict(F=uc.first_stage_f(d))
    out["proxy"], out["proxy_se"] = uc.backdoor_proxy_se(d)
    out["2sls"], out["2sls_se"] = uc.proximal_se(d)
    b, q, ok = bridges(d)
    out["dr"], out["dr_se"] = dr_ate(d, b, q)
    out["ok"] = ok
    return out


def agg_lhs(res):
    design = lhs_design()
    rows, arms = [], []
    for (key, cell), reps in sorted(res.items(), key=lambda x: str(x[0])):
        if key == "lhs":
            p = design[cell]
            t = truth_general(p)
            E = {k: np.array([x[k] for x in reps], float) for k in reps[0]}
            ok = E["ok"].astype(bool)
            rm_dr = float(np.sqrt(np.mean((E["dr"][ok] - t) ** 2))) if ok.any() else np.nan
            rm_px = float(np.sqrt(np.mean((E["proxy"] - t) ** 2)))
            better = "proximal" if rm_dr < rm_px else "proxy"
            dec = np.where((E["F"] < F_MIN) | ~ok, "inconclusive",
                           np.where((p["gamma"] < GSTAR_DR / 2) & (p["delta"] < DSTAR_DR / 2),
                                    "proximal", "proxy"))
            conc = dec != "inconclusive"
            chosen_cov = np.where(dec == "proximal", np.abs(E["dr"] - t) <= 1.96 * E["dr_se"],
                                  np.abs(E["proxy"] - t) <= 1.96 * E["proxy_se"])
            rows.append(dict(dgp=cell, **{k: round(v, 3) for k, v in p.items()}, true_ate=r4(t),
                             F_median=round(float(np.median(E["F"])), 1),
                             rmse_dr=r4(rm_dr), rmse_proxy=r4(rm_px), rmse_2sls=r4(np.sqrt(np.mean((E["2sls"] - t) ** 2))),
                             cov_dr=round(cover(E["dr"][ok], E["dr_se"][ok], t), 3) if ok.any() else "",
                             cov_proxy=round(cover(E["proxy"], E["proxy_se"], t), 3),
                             better=better, share_proximal=round(float(np.mean(dec == "proximal")), 3),
                             share_proxy=round(float(np.mean(dec == "proxy")), 3),
                             share_inconclusive=round(float(np.mean(~conc)), 3),
                             decision_correct=round(float(np.mean(dec[conc] == better)), 3) if conc.any() else "",
                             chosen_cov=round(float(chosen_cov[conc].mean()), 3) if conc.any() else "",
                             nonconverged=int((~ok).sum()), reps=len(reps)))
        else:
            p = arm_params(key, cell)
            t = truth_general(p)
            for e in ("proxy", "2sls", "dr"):
                est = np.array([x[e] for x in reps]); se = np.array([x[e + "_se"] for x in reps])
                arms.append(dict(domain=key, arm=cell, estimator=e, true_ate=r4(t), bias=r4(est.mean() - t),
                                 mcse=r4(est.std(ddof=1) / np.sqrt(len(est))),
                                 rmse=r4(np.sqrt(np.mean((est - t) ** 2))), coverage=round(cover(est, se, t), 3),
                                 F_median=round(float(np.median([x["F"] for x in reps])), 1),
                                 nonconverged=int(sum(not x["ok"] for x in reps)), reps=len(reps)))
    if rows:
        write(rows, "r2_lhs_design.csv")
        # meta-model: log RMSE ratio (DR / proxy) on standardised design parameters
        keys = list(LHS_RANGES)
        X = np.array([[r[k] for k in keys] for r in rows], float)
        yv = np.log(np.array([r["rmse_dr"] / r["rmse_proxy"] for r in rows], float))
        okm = np.isfinite(yv)
        Xs = (X - X.mean(0)) / X.std(0)
        beta = np.linalg.lstsq(np.column_stack([np.ones(okm.sum()), Xs[okm]]), yv[okm], rcond=None)[0]
        res_ = yv[okm] - np.column_stack([np.ones(okm.sum()), Xs[okm]]) @ beta
        r2 = 1 - res_.var() / yv[okm].var()
        write([dict(parameter=k, std_coef=round(float(b), 3)) for k, b in zip(["intercept"] + keys, beta)]
              + [dict(parameter="R2", std_coef=round(float(r2), 3))], "r2_lhs_metamodel.csv")
    if arms:
        write(arms, "r2_stress_arms.csv")


# =========================================================================== part falsify
def gcm_p(x, y, Zc, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(x)
    rx, ry = np.zeros(n), np.zeros(n)
    for tr, te in KFold(2, shuffle=True, random_state=seed).split(Zc):
        rx[te] = x[te] - HistGradientBoostingRegressor(max_iter=100, random_state=seed).fit(Zc[tr], x[tr]).predict(Zc[te])
        ry[te] = y[te] - HistGradientBoostingRegressor(max_iter=100, random_state=seed).fit(Zc[tr], y[tr]).predict(Zc[te])
    R = rx * ry
    tstat = np.sqrt(n) * R.mean() / R.std()
    return float(2 * (1 - stats.norm.cdf(abs(tstat))))


def rep_falsify(domain, r, cell):
    cfg = uc.DOMAINS[domain]
    d = pr.simulate_violated(cfg, 8000, np.random.default_rng(SEED + 811 * r + off(domain, 83)), 0.0)
    tests = pr.implied_independencies("missing_edge_sev_to_y") + pr.implied_independencies_recorded("missing_edge_sev_to_y")
    out = {}
    for x, y, cond in tests:
        Zc = np.column_stack([d[c] for c in cond])
        key = f"{x} _||_ {y} | {','.join(cond)}"
        out[key] = (pr.partial_corr_test(d[x], d[y], Zc), gcm_p(d[x], d[y], Zc, r))
    return out


def agg_falsify(res):
    rows = []
    true_set = {f"{x} _||_ {y} | {','.join(c)}" for x, y, c in pr.implied_independencies("true")}
    for (domain, _), reps in sorted(res.items()):
        for key in reps[0]:
            pf = np.array([x[key][0] for x in reps]); pg = np.array([x[key][1] for x in reps])
            holds = key in true_set
            rows.append(dict(domain=domain, implication=key,
                             holds_under_truth=holds, quantity="size" if holds else "power",
                             fisher_z_reject_005=round(float(np.mean(pf < 0.05)), 3),
                             fisher_z_reject_bonf=round(float(np.mean(pf < 0.05 / 9)), 3),
                             gcm_reject_005=round(float(np.mean(pg < 0.05)), 3),
                             gcm_reject_bonf=round(float(np.mean(pg < 0.05 / 9)), 3), reps=len(reps)))
        # family-wise: true graph passes all 8 tests
        for lab, idx in (("fisher_z", 0), ("gcm", 1)):
            fw = np.mean([all(x[k][idx] > 0.05 / 8 for k in true_set) for x in reps])
            rows.append(dict(domain=domain, implication=f"true graph passes all 8 ({lab})",
                             holds_under_truth=True, quantity="1 - familywise error",
                             fisher_z_reject_005="", fisher_z_reject_bonf=round(float(fw), 3) if idx == 0 else "",
                             gcm_reject_005="", gcm_reject_bonf=round(float(fw), 3) if idx == 1 else "",
                             reps=len(reps)))
    write(rows, "r2_falsification_size_power.csv")


# =========================================================================== part dml
DML_REPS, DML_R = 40, 5
HGB_CONFIGS = [dict(), dict(max_depth=3, learning_rate=0.05, max_iter=200),
               dict(max_leaf_nodes=15, learning_rate=0.1, l2_regularization=1.0),
               dict(max_depth=2, learning_rate=0.1, max_iter=150)]


def tuned_config(domain, cols):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    d = uc.simulate(uc.DOMAINS[domain], 8000, np.random.default_rng(SEED + 99_999))
    X = np.column_stack([d[c] for c in cols])
    sc = [cross_val_score(HistGradientBoostingClassifier(random_state=0, **c), np.column_stack([X, d["a"]]),
                          d["y"], cv=3, scoring="neg_log_loss").mean() for c in HGB_CONFIGS]
    return int(np.argmax(sc))


def learners(kind, cfgi):
    from sklearn.ensemble import HistGradientBoostingClassifier as HC
    if kind == "logistic":
        return lambda: LR()
    c = HGB_CONFIGS[0 if kind == "hgb_default" else cfgi]
    return lambda: HC(random_state=0, **c)


def rep_dml(domain, r, cell):
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    kind, cols, cfgi = cell
    cfg = uc.DOMAINS[domain]
    d = pr.simulate_violated(cfg, 8000, np.random.default_rng(SEED + 977 * r + off(domain, 89)), 0.0)
    X = np.column_stack([d[c] for c in cols])
    a, y, n = d["a"], d["y"], len(d["y"])
    mk = learners(kind, cfgi)
    th_a, se_a, th_p, se_p, diag = [], [], [], [], []
    for rr in range(DML_R):
        e, m1, m0, my = (np.zeros(n) for _ in range(4))
        for tr, te in KFold(5, shuffle=True, random_state=1000 * r + rr).split(X):
            e[te] = mk().fit(X[tr], a[tr]).predict_proba(X[te])[:, 1]
            t1, t0 = tr[a[tr] == 1], tr[a[tr] == 0]
            m1[te] = mk().fit(X[t1], y[t1]).predict_proba(X[te])[:, 1]
            m0[te] = mk().fit(X[t0], y[t0]).predict_proba(X[te])[:, 1]
            my[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        clipped = float(np.mean((e < 0.01) | (e > 0.99)))
        ec = np.clip(e, 0.01, 0.99)
        psi = m1 - m0 + a * (y - m1) / ec - (1 - a) * (y - m0) / (1 - ec)
        th_a.append(psi.mean()); se_a.append(psi.std(ddof=1) / np.sqrt(n))
        ra, ry = a - ec, y - my
        th = np.sum(ra * ry) / np.sum(ra ** 2)
        th_p.append(th); se_p.append(np.sqrt(np.sum((ra * (ry - th * ra)) ** 2)) / np.sum(ra ** 2))
        mu_obs = np.where(a == 1, m1, m0)
        diag.append((roc_auc_score(a, e), brier_score_loss(a, ec), log_loss(y, np.clip(mu_obs, 1e-6, 1 - 1e-6)),
                     clipped))

    def agg(th, se):
        th, se = np.array(th), np.array(se)
        m = np.median(th)
        return float(m), float(np.sqrt(np.median(se ** 2 + (th - m) ** 2)))
    out = dict(zip(("aipw", "aipw_se"), agg(th_a, se_a)))
    out.update(zip(("plr", "plr_se"), agg(th_p, se_p)))
    out["single_aipw"], out["single_aipw_se"] = th_a[0], se_a[0]
    out.update(zip(("auc", "brier", "logloss", "clipped"), np.mean(diag, axis=0)))
    return out


def plr_truth(cfg):
    rng = np.random.default_rng(SEED + 999)
    age = rng.normal(0, 1, 400_000)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, 400_000)
    e = uc.sigmoid(cfg["sev_to_treat"] * sev + cfg["treat_intercept"])
    eta = cfg["sev_to_out"] * sev + cfg["age_to_out"] * age + cfg["out_intercept"]
    tau = uc.sigmoid(eta + cfg["treat_to_out"]) - uc.sigmoid(eta)
    return float(np.sum(e * (1 - e) * tau) / np.sum(e * (1 - e)))


def agg_dml(res):
    rows = []
    for (domain, (kind, cols, cfgi)), reps in sorted(res.items(), key=lambda x: str(x[0])):
        cfg = uc.DOMAINS[domain]
        ate = uc.true_ate(cfg, 200_000, np.random.default_rng(SEED + 999))
        plr_t = plr_truth(cfg)
        R = {k: np.array([x[k] for x in reps]) for k in reps[0]}
        setname = "oracle (age, severity)" if "severity" in cols else "recorded (age, Z, W)"
        for score, t, lab in (("aipw", ate, "ATE"), ("plr", plr_t, "overlap-weighted PLR estimand"),
                              ("plr", ate, "ATE"), ("single_aipw", ate, "ATE (single split)")):
            rows.append(dict(domain=domain, learner=kind + (f" (config {cfgi})" if kind == "hgb_tuned" else ""),
                             adjustment_set=setname, score=score, target=lab, truth=r4(t),
                             bias=r4(R[score].mean() - t), mcse=r4(R[score].std(ddof=1) / np.sqrt(len(reps))),
                             coverage=round(cover(R[score], R[score + "_se"], t), 3),
                             prop_auc=round(float(R["auc"].mean()), 3), prop_brier=round(float(R["brier"].mean()), 4),
                             outcome_logloss=round(float(R["logloss"].mean()), 4),
                             share_clipped=round(float(R["clipped"].mean()), 4), reps=len(reps),
                             cross_fit_repetitions=DML_R))
    write(rows, "r2_dml_repeated.csv")


# =========================================================================== part ovb
def rep_ovb(domain, r, cell):
    import pandas as pd
    import doubleml as dml
    cfg = uc.DOMAINS[domain]
    d = pr.simulate_violated(cfg, 8000, np.random.default_rng(SEED + 977 * r + off(domain, 89)), 0.0)
    df = pd.DataFrame({k: d[k] for k in ("age", "z", "w", "a", "y")})
    data = dml.DoubleMLData(df, "y", "a", x_cols=["age", "z", "w"])
    obj = dml.DoubleMLIRM(data, ml_g=LR(), ml_m=LR(), n_folds=5, trimming_threshold=0.01)
    np.random.seed(r)
    obj.fit()
    out = dict(theta=float(obj.coef[0]), se=float(obj.se[0]))
    for bench in ("age", "z"):
        bm = obj.sensitivity_benchmark(benchmarking_set=[bench])
        cfy, cfd, rho = (float(np.clip(bm[c].iloc[0], 1e-4, 0.99)) for c in ("cf_y", "cf_d", "rho"))
        obj.sensitivity_analysis(cf_y=cfy, cf_d=cfd, rho=abs(rho))
        sp = obj.sensitivity_params
        out[bench] = dict(cf_y=cfy, cf_d=cfd, rho=rho, lo=float(np.ravel(sp["theta"]["lower"])[0]),
                          hi=float(np.ravel(sp["theta"]["upper"])[0]),
                          ci_lo=float(np.ravel(sp["ci"]["lower"])[0]), ci_hi=float(np.ravel(sp["ci"]["upper"])[0]),
                          rv=float(np.ravel(sp["rv"])[0]), rva=float(np.ravel(sp["rva"])[0]))
    if r == 0:   # cross-checks on replication 0
        smpls = obj.smpls[0]
        e, m1, m0 = (np.zeros(8000) for _ in range(3))
        X = df[["age", "z", "w"]].to_numpy()
        for tr, te in smpls:
            e[te] = LR().fit(X[tr], d["a"][tr]).predict_proba(X[te])[:, 1]
            t1, t0 = tr[d["a"][tr] == 1], tr[d["a"][tr] == 0]
            m1[te] = LR().fit(X[t1], d["y"][t1]).predict_proba(X[te])[:, 1]
            m0[te] = LR().fit(X[t0], d["y"][t0]).predict_proba(X[te])[:, 1]
        out["own_aipw"] = aipw_ate(d, e, m1, m0, 0.01)
        from linearmodels.iv import IV2SLS
        df["const"] = 1.0
        iv = IV2SLS(df["y"], df[["const", "a", "age"]], df[["w"]], df[["z"]]).fit(cov_type="robust")
        out["iv"] = (float(iv.params["a"]), float(iv.std_errors["a"]))
        out["own_2sls"] = uc.proximal_se(d)
        import linearmodels, doubleml as _d
        out["versions"] = f"doubleml {_d.__version__}; linearmodels {linearmodels.__version__}"
    return out


def agg_ovb(res):
    rows, chk = [], []
    for (domain, _), reps in sorted(res.items()):
        t = uc.true_ate(uc.DOMAINS[domain], 200_000, np.random.default_rng(SEED + 999))
        for bench in ("age", "z"):
            B = {k: np.array([x[bench][k] for x in reps]) for k in reps[0][bench]}
            rows.append(dict(domain=domain, benchmark=bench, true_ate=r4(t),
                             theta_mean=r4(np.mean([x["theta"] for x in reps])),
                             cf_y=r4(B["cf_y"].mean()), cf_d=r4(B["cf_d"].mean()), rho=r4(B["rho"].mean()),
                             bound_lo=r4(B["lo"].mean()), bound_hi=r4(B["hi"].mean()),
                             share_bounds_contain_truth=round(float(np.mean((B["lo"] <= t) & (t <= B["hi"]))), 3),
                             share_ci_bounds_contain_truth=round(float(np.mean((B["ci_lo"] <= t) & (t <= B["ci_hi"]))), 3),
                             robustness_value=r4(B["rv"].mean()), rva=r4(B["rva"].mean()), reps=len(reps)))
        x = reps[0]
        chk.append(dict(domain=domain, doubleml_irm=round(x["theta"], 4), own_aipw_same_folds=round(x["own_aipw"][0], 4),
                        doubleml_se=round(x["se"], 4), own_aipw_se=round(x["own_aipw"][1], 4),
                        linearmodels_iv2sls=round(x["iv"][0], 4), own_proximal_2sls=round(x["own_2sls"][0], 4),
                        linearmodels_se=round(x["iv"][1], 4), own_2sls_se=round(x["own_2sls"][1], 4),
                        versions=x["versions"]))
    write(rows, "r2_dml_ovb_bounds.csv")
    write(chk, "r2_reference_crosscheck.csv")


# =========================================================================== part cross
DENSE = tuple(np.round(np.arange(0.10, 0.3001, 0.02), 2))


def rep_cross(domain, r, cell):
    kind, v = cell
    cfg = uc.DOMAINS[domain]
    if kind == "gamma":
        d = pr.simulate_violated(cfg, 8000, np.random.default_rng(SEED + 1301 * r + off(domain, 97) + int(v * 1000)), v)
    else:
        d = pr.simulate_violated_w(cfg, 8000, np.random.default_rng(SEED + 1303 * r + off(domain, 97) + int(v * 1000)), v)
    out = {}
    out["2sls"] = uc.proximal_se(d)[0]
    b, q, ok = bridges(d)
    out["dr"] = dr_ate(d, b, q)[0]
    out["ols"] = uc.backdoor_proxy_se(d)[0]
    out["aipw"] = aipw_ate(d, *cf_nuis(d, ["age", "z", "w"], r), 0.01)[0]
    out["ok"] = ok
    return out


def agg_cross(res):
    rows = []
    rng = np.random.default_rng(2028)
    for domain in uc.DOMAINS:
        cfg = uc.DOMAINS[domain]
        for kind in ("gamma", "delta"):
            truths = {v: (pr.true_ate_violated(cfg, 200_000, np.random.default_rng(SEED + 7), v) if kind == "gamma"
                          else uc.true_ate(cfg, 200_000, np.random.default_rng(SEED + 7))) for v in DENSE}
            E = {v: {k: np.array([x[k] for x in res[(domain, (kind, v))]], float) for k in ("2sls", "dr", "ols", "aipw")}
                 for v in DENSE}
            nrep = len(E[DENSE[0]]["2sls"])

            def crossing(idx, est, comp):
                g = np.array(DENSE)
                dif = np.array([abs(E[v][est][idx].mean() - truths[v]) - abs(E[v][comp][idx].mean() - truths[v]) for v in DENSE])
                if dif[0] >= 0:
                    return -np.inf
                for i in range(len(g) - 1):
                    if dif[i] < 0 <= dif[i + 1]:
                        return g[i] + (g[i + 1] - g[i]) * (-dif[i]) / (dif[i + 1] - dif[i])
                return np.inf
            for est in ("2sls", "dr"):
                for comp in ("ols", "aipw"):
                    pt = crossing(np.arange(nrep), est, comp)
                    bs = np.array([crossing(rng.integers(0, nrep, nrep), est, comp) for _ in range(1000)])
                    fin = np.isfinite(bs)
                    fmt = lambda x: ("<0.10" if x == -np.inf else ">0.30" if x == np.inf else round(float(x), 3))
                    rows.append(dict(domain=domain, restriction=kind, estimator=est,
                                     comparator="linear proxy adjustment" if comp == "ols" else "cross-fitted AIPW (logistic)",
                                     crossing=fmt(pt),
                                     ci_lo=round(float(np.percentile(bs[fin], 2.5)), 3) if fin.mean() > 0.9 else "",
                                     ci_hi=round(float(np.percentile(bs[fin], 97.5)), 3) if fin.mean() > 0.9 else "",
                                     comparator_bias_at_0_10=r4(E[DENSE[0]][comp].mean() - truths[DENSE[0]]),
                                     reps=nrep, grid="0.10-0.30 step 0.02"))
    write(rows, "r2_crossing_dense.csv")


# =========================================================================== part overid
OV_GRID = (0.0, 0.1, 0.2, 0.3, 0.5)


def rep_overid(domain, r, cell):
    kind, v = cell
    cfg = uc.DOMAINS[domain]
    rng = np.random.default_rng(SEED + 1409 * r + off(domain, 97) + int(v * 1000) + (0 if kind == "gamma" else 50_000))
    n = 8000
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    z1 = cfg["sev_to_z"] * sev + rng.normal(0, 1, n)
    z2 = cfg["sev_to_z"] * sev + rng.normal(0, 1, n)
    a = (rng.uniform(0, 1, n) < uc.sigmoid(cfg["sev_to_treat"] * sev + cfg["treat_intercept"])).astype(float)
    w = cfg["sev_to_w"] * sev + (v * a if kind == "delta" else 0) + rng.normal(0, 1, n)
    eta = (cfg["sev_to_out"] * sev + cfg["age_to_out"] * age + cfg["treat_to_out"] * a
           + (v * z1 if kind == "gamma" else 0) + cfg["out_intercept"])
    y = (rng.uniform(0, 1, n) < uc.sigmoid(eta)).astype(float)
    Zm = np.column_stack([np.ones(n), a, z1, z2, age])
    X = np.column_stack([np.ones(n), a, w, age])
    W0 = np.linalg.inv(Zm.T @ Zm / n)
    A_ = X.T @ Zm / n
    b1 = np.linalg.solve(A_ @ W0 @ A_.T, A_ @ W0 @ (Zm.T @ y / n))
    g = Zm * (y - X @ b1)[:, None]
    S = np.linalg.inv(g.T @ g / n)
    b2 = np.linalg.solve(A_ @ S @ A_.T, A_ @ S @ (Zm.T @ y / n))
    gb = (Zm * (y - X @ b2)[:, None]).mean(0)
    J = float(n * gb @ S @ gb)
    return dict(J_p=float(1 - stats.chi2.cdf(J, 1)), b_T=float(b2[1]))


def agg_overid(res):
    rows = []
    for (domain, (kind, v)), reps in sorted(res.items(), key=lambda x: str(x[0])):
        p = np.array([x["J_p"] for x in reps])
        rows.append(dict(domain=domain, violation=kind, strength=v,
                         hansen_J_reject_005=round(float(np.mean(p < 0.05)), 3),
                         mean_gmm_bT=r4(np.mean([x["b_T"] for x in reps])), reps=len(reps)))
    write(rows, "r2_overid_hansen_j.csv")


# =========================================================================== part binbr
def logistic_bridge(d):
    n = len(d["y"])
    Zm = np.column_stack([np.ones(n), d["a"], d["z"], d["age"]])
    X = np.column_stack([np.ones(n), d["a"], d["w"], d["age"]])
    start = np.r_[LR().fit(X[:, 1:], d["y"]).intercept_, LR().fit(X[:, 1:], d["y"]).coef_.ravel()]
    sol = least_squares(lambda b: ((d["y"] - uc.sigmoid(X @ b))[:, None] * Zm).mean(0), start,
                        method="lm", xtol=1e-12, ftol=1e-12)
    ok = np.max(np.abs(sol.fun)) < 1e-7
    return sol.x, ok


def rep_binbr(domain, r, cell):
    cfg = uc.DOMAINS[domain]
    d = uc.simulate(cfg, 8000, np.random.default_rng(SEED + 1000 * r + off(domain)))
    b, okh = logistic_bridge(d)
    n = len(d["y"])
    X = lambda t: np.column_stack([np.ones(n), np.full(n, t) if np.isscalar(t) else t, d["w"], d["age"]])
    h1, h0, ha = uc.sigmoid(X(1.0) @ b), uc.sigmoid(X(0.0) @ b), uc.sigmoid(X(d["a"]) @ b)
    _, q, okq = bridges(d)
    pl = h1 - h0
    psi = (2 * d["a"] - 1) * q * (d["y"] - ha) + pl
    rn = np.sqrt(n)
    return dict(plug=pl.mean(), plug_se=pl.std(ddof=1) / rn, dr=psi.mean(), dr_se=psi.std(ddof=1) / rn,
                ok=bool(okh and okq))


def agg_binbr(res):
    rows = []
    for (domain, _), reps in sorted(res.items()):
        t = uc.true_ate(uc.DOMAINS[domain], 200_000, np.random.default_rng(SEED + 999))
        for e, lab in (("plug", "logistic outcome bridge (plug-in)"), ("dr", "DR, logistic h + q")):
            est = np.array([x[e] for x in reps]); se = np.array([x[e + "_se"] for x in reps])
            rows.append(dict(domain=domain, estimator=lab, true_ate=r4(t), bias=r4(est.mean() - t),
                             mcse=r4(est.std(ddof=1) / np.sqrt(len(est))), emp_sd=r4(est.std(ddof=1)),
                             rmse=r4(np.sqrt(np.mean((est - t) ** 2))), coverage=round(cover(est, se, t), 3),
                             nonconverged=int(sum(not x["ok"] for x in reps)), reps=len(reps)))
    write(rows, "r2_logistic_bridge.csv")


# =========================================================================== part shift
SHIFTS = (("age mean +0.5", 0.5, 1.0), ("age mean +1.0", 1.0, 1.0), ("proxy loadings x0.8", 0.0, 0.8),
          ("proxy loadings x0.9", 0.0, 0.9), ("proxy loadings x1.1", 0.0, 1.1), ("proxy loadings x1.2", 0.0, 1.2),
          ("none", 0.0, 1.0))


def sim_site(cfg, n, rng, age_shift, load):
    c = dict(cfg, sev_to_z=cfg["sev_to_z"] * load, sev_to_w=cfg["sev_to_w"] * load)
    d = uc.simulate(c, n, rng)
    if age_shift:
        sev_noise = d["severity"] - c["age_to_sev"] * d["age"]
        d["age"] = d["age"] + age_shift
        d["severity"] = c["age_to_sev"] * d["age"] + sev_noise
        rngb = np.random.default_rng(rng.integers(1 << 31))
        d["z"] = c["sev_to_z"] * d["severity"] + rngb.normal(0, 1, n)
        d["w"] = c["sev_to_w"] * d["severity"] + rngb.normal(0, 1, n)
    return d


def site_truth(cfg, age_shift):
    rng = np.random.default_rng(SEED + 999)
    age = rng.normal(age_shift, 1, 400_000)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, 400_000)
    base = cfg["sev_to_out"] * sev + cfg["age_to_out"] * age + cfg["out_intercept"]
    return float(np.mean(uc.sigmoid(base + cfg["treat_to_out"]) - uc.sigmoid(base)))


def rep_shift(domain, r, cell):
    cfg = uc.DOMAINS[domain]
    o = SEED + 1000 * r + off(domain)
    dA = uc.simulate(cfg, 8000, np.random.default_rng(o))
    b, q, ok = bridges(dA)
    ate, se = dr_ate(dA, b, q)
    out = {}
    for lab, sh, ld in SHIFTS:
        dB = sim_site(cfg, 8000, np.random.default_rng(o + 11_000_000), sh, ld)
        pi = uc.policy_knowledge(dB, cfg)
        vhat_all = float(np.mean(h_lin(b, dB, 1.0)))
        vhat_kd = float(np.mean(h_lin(b, dB, pi)))
        out[lab] = dict(ate=ate, se=se, v_all_bias=vhat_all - uc.true_policy_value(cfg, dB, np.ones(8000)),
                        v_kd_bias=vhat_kd - uc.true_policy_value(cfg, dB, pi), truth=site_truth(cfg, sh))
    return out


def agg_shift(res):
    rows = []
    for (domain, _), reps in sorted(res.items()):
        for lab, _, _ in SHIFTS:
            R = {k: np.array([x[lab][k] for x in reps]) for k in reps[0][lab]}
            t = R["truth"][0]
            rows.append(dict(domain=domain, shift=lab, true_ate_site_B=r4(t), ate_bias=r4(R["ate"].mean() - t),
                             ate_coverage=round(cover(R["ate"], R["se"], t), 3),
                             value_treat_all_bias=r4(R["v_all_bias"].mean()),
                             value_kd_rule_bias=r4(R["v_kd_bias"].mean()), reps=len(reps)))
    write(rows, "r2_site_shift.csv")


# =========================================================================== part diffnoise
def rep_diffnoise(domain, r, cell):
    cfg = uc.DOMAINS["ards"]
    rng = np.random.default_rng(SEED + 1511 * r + 4_000_000)
    n = 8000
    d = uc.simulate(cfg, n, rng)
    G = rng.uniform(0, 1, n) < 0.5
    d["w"] = cfg["sev_to_w"] * d["severity"] + np.where(G, 2.0, 1.0) * rng.normal(0, 1, n)
    b, q, ok = bridges(d)
    psi = (2 * d["a"] - 1) * q * (d["y"] - h_lin(b, d, d["a"])) + b[1]
    out = dict(ok=ok)
    for g, lab in ((False, "G=0 (noise SD 1)"), (True, "G=1 (noise SD 2)")):
        m = G == g
        out[lab + " pooled bridges"] = (psi[m].mean(), psi[m].std(ddof=1) / np.sqrt(m.sum()))
        ds = {k: v[m] for k, v in d.items()}
        bs, qs, _ = bridges(ds)
        out[lab + " bridges fitted within group"] = dr_ate(ds, bs, qs)
    return out


def agg_diffnoise(res):
    rows = []
    t = uc.true_ate(uc.DOMAINS["ards"], 200_000, np.random.default_rng(SEED + 999))
    for (_, _), reps in res.items():
        for k in [k for k in reps[0] if k != "ok"]:
            est = np.array([x[k][0] for x in reps]); se = np.array([x[k][1] for x in reps])
            rows.append(dict(domain="ards", cell=k, true_ate=r4(t), bias=r4(est.mean() - t),
                             mcse=r4(est.std(ddof=1) / np.sqrt(len(est))), coverage=round(cover(est, se, t), 3),
                             reps=len(reps)))
    write(rows, "r2_differential_proxy_error.csv")


# =========================================================================== part prop
def pop_coefs(cfg, sz=1.0, sw=1.0, gamma=0.0, delta=0.0):
    """Exact population OLS (Y on T,A,Z,W) and proximal 2SLS coefficients on T in the linear
    analogue: U=aA+eU, Z=lz U+sz eZ, T=aT U+eT, W=lw U+delta T+sw eW, Y=bT T+bU U+bA A+gamma Z+eY."""
    aA, lz, lw, aT = cfg["age_to_sev"], cfg["sev_to_z"], cfg["sev_to_w"], cfg["sev_to_treat"]
    bT, bU, bA = cfg["treat_to_out"], cfg["sev_to_out"], cfg["age_to_out"]
    # columns: A, eU, eZ, eT, eW, eY
    A = np.array([1, 0, 0, 0, 0, 0.0]); U = aA * A + np.array([0, 1, 0, 0, 0, 0.0])
    Z = lz * U + sz * np.eye(6)[2]; T = aT * U + np.eye(6)[3]
    W = lw * U + delta * T + sw * np.eye(6)[4]
    Y = bT * T + bU * U + bA * A + gamma * Z + np.eye(6)[5]
    cov = lambda P, Q: np.array([[p @ q for q in Q] for p in P])
    Xo = [T, A, Z, W]
    ols = np.linalg.solve(cov(Xo, Xo), cov(Xo, [Y]))[0, 0]
    Xr, Zi = [T, W, A], [T, Z, A]
    Szz = cov(Zi, Zi); Sxz = cov(Xr, Zi); Szy = cov(Zi, [Y])
    tsls = np.linalg.solve(Sxz @ np.linalg.solve(Szz, Sxz.T), Sxz @ np.linalg.solve(Szz, Szy))[0, 0]
    s2 = 1.0 / (1.0 + lz ** 2 / sz ** 2 + lw ** 2 / sw ** 2)
    formula = bT + bU * aT * s2 / (aT ** 2 * s2 + 1.0)
    return ols, tsls, formula


def rep_prop(domain, r, cell):
    sz, sw = cell
    cfg = uc.DOMAINS[domain]
    rng = np.random.default_rng(SEED + 1601 * r + off(domain))
    n = 8000
    age = rng.normal(0, 1, n)
    U = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    Z = cfg["sev_to_z"] * U + sz * rng.normal(0, 1, n)
    T = cfg["sev_to_treat"] * U + rng.normal(0, 1, n)
    W = cfg["sev_to_w"] * U + sw * rng.normal(0, 1, n)
    Y = cfg["treat_to_out"] * T + cfg["sev_to_out"] * U + cfg["age_to_out"] * age + rng.normal(0, 1, n)
    return dict(ols=float(uc.ols(np.column_stack([T, age, Z, W]), Y)[1]))


def agg_prop(res):
    rows, cr = [], []
    for (domain, (sz, sw)), reps in sorted(res.items(), key=lambda x: str(x[0])):
        cfg = uc.DOMAINS[domain]
        ols, _, formula = pop_coefs(cfg, sz, sw)
        sim = np.array([x["ols"] for x in reps])
        rows.append(dict(domain=domain, sd_eZ=sz, sd_eW=sw, predicted_bias=r4(formula - cfg["treat_to_out"]),
                         population_bias=r4(ols - cfg["treat_to_out"]), simulated_bias=r4(sim.mean() - cfg["treat_to_out"]),
                         mcse=r4(sim.std(ddof=1) / np.sqrt(len(sim))), reps=len(reps)))
    for domain, cfg in uc.DOMAINS.items():
        sdw = float(np.sqrt(cfg["sev_to_w"] ** 2 * (cfg["age_to_sev"] ** 2 + 1) + 1))
        for kind in ("gamma", "delta"):
            def f(v):
                o, t, _ = pop_coefs(cfg, gamma=v if kind == "gamma" else 0, delta=v if kind == "delta" else 0)
                return abs(t - cfg["treat_to_out"]) - abs(o - cfg["treat_to_out"])
            star = brentq(f, 1e-6, 5.0) if f(1e-6) < 0 < f(5.0) else np.nan
            cr.append(dict(domain=domain, restriction=kind, linear_analogue_crossing=round(star, 3),
                           common_scale=round(star / (cfg["sev_to_out"] if kind == "gamma" else sdw), 3),
                           scale="gamma*/beta_U" if kind == "gamma" else "delta*/SD(W)"))
    write(rows, "r2_proposition_check.csv")
    write(cr, "r2_linear_analytic_crossing.csv")


# =========================================================================== part conv
def rep_conv(domain, r, cell):
    kind, v = cell
    cfg = uc.DOMAINS[domain]
    if kind == "weak":
        c = dict(cfg, sev_to_z=cfg["sev_to_z"] * v, sev_to_w=cfg["sev_to_w"] * v)
        d = pr.simulate_violated(c, 8000, np.random.default_rng(SEED + 401 * r + off(domain, 97)), 0.0)
    else:
        d = pr.simulate_nonlinear(cfg, 8000, np.random.default_rng(SEED + 307 * r + off(domain, 97)), v)
    b, q, ok = bridges(d)
    est, se = dr_ate(d, b, q)
    return dict(est=est, se=se, ok=ok)


def agg_conv(res):
    rows = []
    for (domain, (kind, v)), reps in sorted(res.items(), key=lambda x: str(x[0])):
        cfg = uc.DOMAINS[domain]
        t = (uc.true_ate(cfg, 200_000, np.random.default_rng(SEED + 7)) if kind == "weak"
             else pr.true_ate_nonlinear(cfg, 200_000, np.random.default_rng(SEED + 13), v))
        est = np.array([x["est"] for x in reps]); se = np.array([x["se"] for x in reps])
        ok = np.array([x["ok"] for x in reps])
        hit = np.abs(est - t) <= 1.96 * se
        rows.append(dict(domain=domain, sweep=kind, cell=v, n_nonconverged=int((~ok).sum()), reps=len(reps),
                         dr_bias_all=r4(est.mean() - t), dr_bias_converged=r4(est[ok].mean() - t) if ok.any() else "",
                         dr_cov_all=round(float(hit.mean()), 3),
                         dr_cov_converged=round(float(hit[ok].mean()), 3) if ok.any() else ""))
    write(rows, "r2_nonconvergence_per_cell.csv")


# =========================================================================== driver
D3 = tuple(uc.DOMAINS)


def jobs_for(part):
    J = []
    if part == "nlh":
        for dm in D3:
            for r in range(200):
                for q in (True, False):
                    J.append((dm, r, ("nlh", q)))
                    J.append((dm, r, ("armF", q)))
                for hq in ((True, True), (True, False), (False, True), (False, False)):
                    J.append((dm, r, ("orig", hq)))
    elif part == "pvmain":
        J = [(dm, r, g) for dm in D3 for r in range(200) for g in (0.0, 0.3, 0.5)]
    elif part == "pvboot":
        J = [(dm, r, 0) for dm in D3 for r in range(40)]
    elif part == "calib":
        for dm in D3:
            for r in range(200):
                J.append((dm, r, ("calibrated effect", 0.0)))
                J.append((dm, r, ("calibrated effect + proxy AUROC 0.70", 0.0)))
            for g in (0.1, 0.2, 0.3, 0.4, 0.5):
                J += [(dm, r, ("calibrated effect", g)) for r in range(100)]
    elif part == "harm":
        J = [(dm, r, 0) for dm in D3 for r in range(100)]
    elif part == "lhs":
        J = [("lhs", r, c) for c in range(LHS_N) for r in range(LHS_REPS)]
        J += [(dm, r, arm) for dm in D3 for r in range(200)
              for arm in ("second confounder U2 (0.5)", "Z-W residual correlation 0.3", "joint gamma*/2 + delta*/2 (DR)")]
    elif part == "falsify":
        J = [(dm, r, 0) for dm in D3 for r in range(200)]
    elif part == "dml":
        for dm in D3:
            for cols in (("age", "severity"), ("age", "z", "w")):
                ti = tuned_config(dm, list(cols))
                for kind in ("hgb_default", "hgb_tuned", "logistic"):
                    J += [(dm, r, (kind, cols, ti)) for r in range(DML_REPS)]
    elif part == "ovb":
        J = [(dm, r, 0) for dm in D3 for r in range(20)]
    elif part == "cross":
        J = [(dm, r, (k, v)) for dm in D3 for k in ("gamma", "delta") for v in DENSE for r in range(200)]
    elif part == "overid":
        J = [(dm, r, (k, v)) for dm in D3 for k in ("gamma", "delta") for v in OV_GRID for r in range(200)]
    elif part == "binbr":
        J = [(dm, r, 0) for dm in D3 for r in range(200)]
    elif part == "shift":
        J = [(dm, r, 0) for dm in D3 for r in range(200)]
    elif part == "diffnoise":
        J = [("ards", r, 0) for r in range(200)]
    elif part == "prop":
        J = [(dm, r, c) for dm in D3 for r in range(200) for c in ((1.0, 1.0), (1.5, 0.7))]
    elif part == "conv":
        J = [(dm, r, ("weak", s)) for dm in D3 for s in pr.PROXY_SCALES for r in range(100)]
        J += [(dm, r, ("nonlinear", g)) for dm in D3 for g in pr.REGIMES for r in range(100)]
    return J


REP = dict(nlh=rep_nlh, pvmain=rep_pvmain, pvboot=rep_pvboot, calib=rep_calib, harm=rep_harm, lhs=rep_lhs,
           falsify=rep_falsify, dml=rep_dml, ovb=rep_ovb, cross=rep_cross, overid=rep_overid, binbr=rep_binbr,
           shift=rep_shift, diffnoise=rep_diffnoise, prop=rep_prop, conv=rep_conv)
AGG = dict(nlh=agg_nlh, pvmain=agg_pvmain, pvboot=agg_pvboot, calib=agg_calib, harm=agg_harm, lhs=agg_lhs,
           falsify=agg_falsify, dml=agg_dml, ovb=agg_ovb, cross=agg_cross, overid=agg_overid, binbr=agg_binbr,
           shift=agg_shift, diffnoise=agg_diffnoise, prop=agg_prop, conv=agg_conv)


def _run(job):
    part, dm, r, cell = job
    return (dm, cell), r, REP[part](dm, r, cell)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="all")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    parts = list(REP) if a.parts == "all" else a.parts.split(",")
    with ProcessPoolExecutor(a.workers) as ex:
        for part in parts:
            jobs = [(part,) + j for j in jobs_for(part)]
            res = defaultdict(dict)
            for key, r, out in ex.map(_run, jobs, chunksize=4):
                res[key][r] = out
            res = {k: [v[r] for r in sorted(v)] for k, v in res.items()}
            AGG[part](res)
            print(f"part {part} done ({len(jobs)} jobs)", flush=True)


if __name__ == "__main__":
    main()
