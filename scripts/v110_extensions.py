"""v1.1.0 extensions: real-data application, plasmode study, sharp sensitivity bounds,
regularised treatment bridge and numerical checks of the propositions.

Parts (python scripts/v110_extensions.py --parts rhc,plasmode,sharp,qreg,theory --workers 8):
  rhc       SUPPORT right heart catheterization data (Connors et al. 1996), proxies of
            Cui et al. (2024): proxy-adjusted OLS, proximal 2SLS (just- and over-identified,
            Hansen J), proximal IPW/DR with the plain and the regularised q solver, AIPW;
            nonparametric bootstrap CIs; Tan and sharp sensitivity values.
  plasmode  real X, Z, W of the RHC cohort; withheld real APACHE III score (aps1) as the
            unrecorded confounder; simulated treatment and 30-day death with a known effect.
  sharp     sharp marginal-sensitivity bounds (Dorn and Guo 2023) next to the Tan/ZSB bounds
            on the main generators and the held-out set B.
  qreg      Tikhonov-regularised treatment-bridge solver with the penalty chosen by 3-fold
            cross-validation of the held-out moment norm; held-out sets A and B rerun with
            the seeds of decision_rule_v2.rep.
  theory    numerical checks of Propositions 1-3 of the manuscript.
Writes results/v110_*.csv and tables/v110_*.tex.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
TAB = Path(__file__).resolve().parents[1] / "tables"
LAM_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
TOL = 1e-6


def write(name, rows):
    with open(RES / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name, flush=True)


def r4(x):
    return round(float(x), 4)


# =========================================================================== generic estimators
def add1(M):
    return np.column_stack([np.ones(len(M)), M])


def logit_fit(X, t, iters=50, ridge=1e-8):
    """Logistic regression by Newton-Raphson (tiny ridge for separation); returns coefficients."""
    Xd = add1(X)
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = uc.sigmoid(Xd @ b)
        H = (Xd * (p * (1 - p))[:, None]).T @ Xd + ridge * np.eye(len(b))
        step = np.linalg.solve(H, Xd.T @ (t - p) - ridge * b)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def ols_hc0(D, y, k):
    """OLS with intercept prepended to D; coefficient and HC0 SE of column k of D."""
    Xd = add1(D)
    beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
    e = y - Xd @ beta
    bread = np.linalg.inv(Xd.T @ Xd)
    V = bread @ ((Xd * e[:, None] ** 2).T @ Xd) @ bread
    return float(beta[k + 1]), float(np.sqrt(V[k + 1, k + 1]))


def p2sls(a, x, z, w, y):
    """Proximal 2SLS: structural Y = c + bT T + bW'W + bX'X, instruments (1, T, Z, X).
    Returns (bT, HC0 se, full beta). With dim Z = dim W the system is just identified."""
    n = len(y)
    D = np.column_stack([np.ones(n), a, w, x])
    Zi = np.column_stack([np.ones(n), a, z, x])
    P = Zi @ np.linalg.lstsq(Zi, D, rcond=None)[0]
    beta = np.linalg.lstsq(P, y, rcond=None)[0]
    e = y - D @ beta
    bread = np.linalg.inv(P.T @ P)
    V = bread @ ((P * e[:, None] ** 2).T @ P) @ bread
    return float(beta[1]), float(np.sqrt(V[1, 1])), beta


def gmm_overid(a, x, z_ext, w, y):
    """Two-step efficient GMM for the outcome bridge with over-identifying instruments z_ext;
    returns (bT, se, Hansen J, df, p)."""
    n = len(y)
    D = np.column_stack([np.ones(n), a, w, x])
    Zi = np.column_stack([np.ones(n), a, z_ext, x])
    S0 = Zi.T @ Zi / n
    W0 = np.linalg.pinv(S0)
    ZD, Zy = Zi.T @ D / n, Zi.T @ y / n
    b1 = np.linalg.solve(ZD.T @ W0 @ ZD, ZD.T @ W0 @ Zy)
    e = y - D @ b1
    S = (Zi * e[:, None] ** 2).T @ Zi / n
    W1 = np.linalg.pinv(S)
    b2 = np.linalg.solve(ZD.T @ W1 @ ZD, ZD.T @ W1 @ Zy)
    g = Zi.T @ (y - D @ b2) / n
    J = float(n * g @ W1 @ g)
    df = Zi.shape[1] - D.shape[1]
    V = np.linalg.inv(ZD.T @ W1 @ ZD) / n
    return float(b2[1]), float(np.sqrt(V[1, 1])), J, df, float(stats.chi2.sf(J, df))


def partial_f(target, base, extra):
    rss = lambda M: float(np.sum((target - M @ np.linalg.lstsq(M, target, rcond=None)[0]) ** 2))
    full = add1(np.column_stack([base, extra]))
    red = add1(base)
    q = extra.shape[1] if extra.ndim > 1 else 1
    return (rss(red) - rss(full)) / q / (rss(full) / (len(target) - full.shape[1]))


def q_solve(F, G, ind, start, lam=0.0, theta0=None, nfev=2000):
    """Treatment bridge q = 1 + exp(F theta) for one arm from E[(ind q - 1) G] = 0.

    lam > 0 adds the Tikhonov term lam * ||theta - theta0||^2 (theta0 = inverse-propensity
    start), i.e. least squares on [m(theta); sqrt(lam)(theta - theta0)]. Returns
    (theta, max |m|, optimiser status)."""
    n = len(ind)
    th0 = start if theta0 is None else theta0
    sl = np.sqrt(lam)

    def res(t):
        q = 1.0 + np.exp(np.clip(F @ t, -50, 50))
        m = ((ind * q)[:, None] * G - G).mean(axis=0)
        return np.r_[m, sl * (t - th0)] if lam > 0 else m

    def jac(t):
        ex = np.exp(np.clip(F @ t, -50, 50))
        J = (G * (ind * ex)[:, None]).T @ F / n
        return np.vstack([J, sl * np.eye(len(t))]) if lam > 0 else J

    method = "lm" if (lam > 0 or G.shape[1] == F.shape[1]) else "trf"
    sol = least_squares(res, start, jac=jac, method=method, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=nfev)
    q = 1.0 + np.exp(np.clip(F @ sol.x, -50, 50))
    m = ((ind * q)[:, None] * G - G).mean(axis=0)
    return sol.x, float(np.max(np.abs(m))), int(sol.status)


def q_bridge(F, G, a, mode="plain", seed=0, folds=3, nfev=2000):
    """q evaluated at each unit's own arm, max moment per arm, chosen lambda per arm, status.

    plain: unpenalised, inverse-propensity start then zero start (as treatment_bridge in
           unmeasured_confounding.py). reg: the plain solution when it balances the moments
           (max |m| < TOL); otherwise the Tikhonov solution with lambda chosen from LAM_GRID by
           K-fold cross-validation of the held-out squared moment norm, refitted on the full
           sample."""
    eta = logit_fit(F[:, 1:], a)
    q = np.empty(len(a))
    info = []
    for arm, start in ((1, -eta), (0, eta)):
        ind = (a == arm).astype(float)
        best = None
        for t0 in (start, np.zeros(F.shape[1])):
            th, mm, st = q_solve(F, G, ind, t0, nfev=nfev)
            if best is None or mm < best[1]:
                best = (th, mm, st)
            if mm < TOL:
                break
        th, mm, st = best
        lam = 0.0
        if mode == "reg" and mm >= TOL:
            rng = np.random.default_rng(seed + arm)
            fold = rng.permutation(len(a)) % folds
            cv = []
            for lam in LAM_GRID:
                s = 0.0
                for k in range(folds):
                    tr, te = fold != k, fold == k
                    th_k, _, _ = q_solve(F[tr], G[tr], ind[tr], start, lam, start)
                    qk = 1.0 + np.exp(np.clip(F[te] @ th_k, -50, 50))
                    s += float(np.sum(((ind[te] * qk)[:, None] * G[te] - G[te]).mean(axis=0) ** 2))
                cv.append(s / folds)
            lam = LAM_GRID[int(np.argmin(cv))]
            th, mm, st = q_solve(F, G, ind, start, lam, start)
        qa = 1.0 + np.exp(np.clip(F @ th, -50, 50))
        q[ind == 1] = qa[ind == 1]
        info.append((mm, lam, st))
    return q, info


def pdr(a, y, h1, h0, q):
    """Proximal DR (Cui et al. 2024) and proximal IPW with plug-in SEs."""
    h = np.where(a == 1, h1, h0)
    sg = 2 * a - 1
    psi = sg * q * (y - h) + h1 - h0
    ipw = sg * q * y
    rn = np.sqrt(len(y))
    return float(psi.mean()), float(psi.std(ddof=1) / rn), float(ipw.mean()), float(ipw.std(ddof=1) / rn)


def aipw(X, a, y, clip=0.01):
    e = np.clip(uc.sigmoid(add1(X) @ logit_fit(X, a)), clip, 1 - clip)
    if np.all((y == 0) | (y == 1)):
        m1 = uc.sigmoid(add1(X) @ logit_fit(X[a == 1], y[a == 1]))
        m0 = uc.sigmoid(add1(X) @ logit_fit(X[a == 0], y[a == 0]))
    else:
        m1 = add1(X) @ np.linalg.lstsq(add1(X[a == 1]), y[a == 1], rcond=None)[0]
        m0 = add1(X) @ np.linalg.lstsq(add1(X[a == 0]), y[a == 0], rcond=None)[0]
    psi = m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(y)))


# ----------------------------------------------------------- sensitivity bounds (binary Y)
def tan_bounds(X, a, y, lams):
    """Tan / Zhao-Small-Bhattacharya normalised-IPW bounds (as msm_bounds.py)."""
    e = np.clip(uc.sigmoid(add1(X) @ logit_fit(X, a)), 1e-6, 1 - 1e-6)
    out = []
    for lam in lams:
        res = []
        for arm, bw in ((1, 1 / e), (0, 1 / (1 - e))):
            s = a == arm
            yy, b = y[s], bw[s]
            lo_w, hi_w = 1 + (b - 1) / lam, 1 + (b - 1) * lam
            one = yy == 1
            up = hi_w[one].sum() / (hi_w[one].sum() + lo_w[~one].sum())
            dn = lo_w[one].sum() / (lo_w[one].sum() + hi_w[~one].sum())
            res.append((dn, up))
        (l1, u1), (l0, u0) = res
        out.append((l1 - u0, u1 - l0))
    return out


def sharp_bounds(X, a, y, lams):
    """Sharp MSM bounds for a binary outcome (Dorn and Guo 2023).

    Under the MSM the law of Y(t) among units in the other arm, given X, is the observed law
    of Y in arm t tilted by a likelihood ratio whose largest-to-smallest ratio is at most
    Lambda^2; the quantile-balancing constraint of Dorn and Guo forces the tilt to integrate
    to one within each X. For binary Y the extreme tilt puts the large ratio on Y = 1, so the
    sharp bound on the unobserved arm mean is mu L^2 / (mu L^2 + 1 - mu) (upper) and
    mu / (mu + (1 - mu) L^2) (lower), with mu = P(Y = 1 | X, T = t). Estimated by plug-in with
    logistic outcome models; at Lambda = 1 the bounds collapse to the g-formula estimate."""
    m1 = uc.sigmoid(add1(X) @ logit_fit(X[a == 1], y[a == 1]))
    m0 = uc.sigmoid(add1(X) @ logit_fit(X[a == 0], y[a == 0]))
    out = []
    for lam in lams:
        L2 = lam * lam
        up = lambda m: m * L2 / (m * L2 + 1 - m)
        dn = lambda m: m / (m + (1 - m) * L2)
        u1 = np.mean(a * y + (1 - a) * up(m1))
        l1 = np.mean(a * y + (1 - a) * dn(m1))
        u0 = np.mean((1 - a) * y + a * up(m0))
        l0 = np.mean((1 - a) * y + a * dn(m0))
        out.append((l1 - u0, u1 - l0))
    return out


GRID = np.round(np.arange(1.0, 6.0001, 0.01), 2)


def smallest(bounds, target):
    for lam, (lo, hi) in zip(GRID, bounds):
        if lo <= target <= hi:
            return float(lam)
    return float("inf")


# =========================================================================== part rhc
def rhc_estimates(d, mode="plain", seed=0, with_q=True):
    x, z, w, a, y = d["x"], d["z"], d["w"], d["a"], d["y"]
    out = {}
    out["naive"] = float(y[a == 1].mean() - y[a == 0].mean())
    out["x_only"] = ols_hc0(np.column_stack([a, x]), y, 0)[0]
    out["proxy"] = ols_hc0(np.column_stack([a, x, z, w]), y, 0)[0]
    out["aipw"] = aipw(np.column_stack([x, z, w]), a, y)[0]
    bT, _, beta = p2sls(a, x, z, w, y)
    out["p2sls"] = bT
    if with_q:
        F = add1(np.column_stack([z, x]))
        G = add1(np.column_stack([w, x]))
        q, info = q_bridge(F, G, a, mode, seed)
        hw = beta[0] + w @ beta[2:4] + x @ beta[4:]
        dr, _, ipw, _ = pdr(a, y, hw + beta[1], hw, q)
        out["pdr"], out["pipw"] = dr, ipw
        out["q_maxmom"] = max(i[0] for i in info)
    return out


def boot_job(job):
    b, mode, outcome = job
    import rhc_data
    d = rhc_data.load()
    if outcome == "days":
        d["y"] = d["days"]
    rng = np.random.default_rng(91_000 + b)
    i = rng.integers(0, len(d["y"]), len(d["y"]))
    db = {k: d[k][i] for k in ("x", "z", "w", "a", "y")}
    db["days"] = d["days"][i]
    try:
        return rhc_estimates(db, mode, seed=b, with_q=True)
    except Exception:  # noqa: BLE001 - record failed resample
        return None


def part_rhc(workers, B):
    import rhc_data
    d = rhc_data.load()
    rows = []
    n = len(d["y"])
    # proxy relevance and over-identification diagnostics
    base = np.column_stack([d["a"], d["x"]])
    diag = dict(n=n, n_treated=int(d["a"].sum()), deaths30=int(d["y"].sum()), p_x=d["x"].shape[1],
                F_ph1=r4(partial_f(d["w"][:, 0], base, d["z"])),
                F_hema1=r4(partial_f(d["w"][:, 1], base, d["z"])))
    z = d["z"]
    zext = np.column_stack([z, z ** 2, z[:, :1] * z[:, 1:]])
    for oc in ("dth30", "days"):
        y = d["y"] if oc == "dth30" else d["days"]
        dd = dict(d, y=y)
        est = {}
        for mode in ("plain", "reg"):
            e = rhc_estimates(dd, mode, seed=5)
            est.update({k + ("" if mode == "plain" else "_reg"): v for k, v in e.items()
                        if mode == "plain" or k in ("pdr", "pipw", "q_maxmom")})
        bJ, seJ, J, df, pJ = gmm_overid(d["a"], d["x"], zext, d["w"], y)
        est["gmm_overid"] = bJ
        diag.update({f"{oc}_hansenJ": r4(J), f"{oc}_J_df": df, f"{oc}_J_p": r4(pJ)})
        diag[f"{oc}_q_maxmom_plain"] = est["q_maxmom"]
        diag[f"{oc}_q_maxmom_reg"] = est["q_maxmom_reg"]
        boots = {}
        # the regularised q is fitted on the full sample only: with 70 covariates its
        # cross-validated fit takes minutes per resample, so it is not bootstrapped
        for mode in ("plain",):
            with ProcessPoolExecutor(workers) as ex:
                R = list(ex.map(boot_job, [(b, mode, "days" if oc == "days" else "y") for b in range(B)]))
            boots[mode] = [r for r in R if r is not None]
            diag[f"{oc}_boot_ok_{mode}"] = len(boots[mode])
            diag[f"{oc}_boot_q_unsolved_{mode}"] = r4(np.mean([r["q_maxmom"] >= TOL for r in boots[mode]]))
        for k in ("naive", "x_only", "proxy", "aipw", "p2sls", "gmm_overid", "pipw", "pdr", "pipw_reg", "pdr_reg"):
            src = boots["plain"]
            if k.endswith("_reg"):
                rows.append(dict(outcome=oc, estimator=k, estimate=r4(est[k]), se="", lo="", hi="",
                                 ci="point estimate only"))
                continue
            kk = k
            if k == "gmm_overid":
                lo, hi, se = bJ - 1.96 * seJ, bJ + 1.96 * seJ, seJ
                ci = "Wald"
            else:
                v = np.array([r[kk] for r in src])
                lo, hi = np.percentile(v, [2.5, 97.5])
                se = v.std(ddof=1)
                ci = f"bootstrap B={len(v)}"
            rows.append(dict(outcome=oc, estimator=k, estimate=r4(est[k]), se=r4(se),
                             lo=r4(lo), hi=r4(hi), ci=ci))
    # sensitivity values for the proxy-adjusted analysis (binary outcome)
    X = np.column_stack([d["x"], d["z"], d["w"]])
    tb, sb = tan_bounds(X, d["a"], d["y"], GRID), sharp_bounds(X, d["a"], d["y"], GRID)
    diag["lambda0_tan"] = smallest(tb, 0.0)
    diag["lambda0_sharp"] = smallest(sb, 0.0)
    i15 = int(np.where(GRID == 1.5)[0][0])
    diag["width15_tan"], diag["width15_sharp"] = r4(tb[i15][1] - tb[i15][0]), r4(sb[i15][1] - sb[i15][0])
    write("v110_rhc_estimates.csv", rows)
    write("v110_rhc_diagnostics.csv", [diag])


# =========================================================================== part plasmode
PLAS_HIDDEN = ("aps1",)
PLAS_SCEN = {"null": 0.0, "harm": 0.5}
ALPHA_U, BETA_U = 0.8, 0.8
PLAS_NFEV = 60


def plas_setup():
    import rhc_data
    d = rhc_data.load(hidden=PLAS_HIDDEN)
    U = d["hidden"]["aps1"]
    bt = logit_fit(d["x"], d["a"])
    by = logit_fit(d["x"], d["y"])
    return d, U, bt, by


def plas_rep(job):
    scen, r = job
    d, U, bt, by = plas_setup()
    tau = PLAS_SCEN[scen]
    rng = np.random.default_rng(700_000 + 1000 * (scen == "harm") + r)
    lt = add1(d["x"]) @ bt + ALPHA_U * U
    a = (rng.uniform(size=len(U)) < uc.sigmoid(lt)).astype(float)
    eta = add1(d["x"]) @ by + BETA_U * U
    y = (rng.uniform(size=len(U)) < uc.sigmoid(eta + tau * a)).astype(float)
    truth = float(np.mean(uc.sigmoid(eta + tau) - uc.sigmoid(eta)))
    x, z, w = d["x"], d["z"], d["w"]
    o = dict(scen=scen, rep=r, truth=truth)
    o["proxy"], o["proxy_se"] = ols_hc0(np.column_stack([a, x, z, w]), y, 0)
    o["x_only"], o["x_only_se"] = ols_hc0(np.column_stack([a, x]), y, 0)
    o["aipw"], o["aipw_se"] = aipw(np.column_stack([x, z, w]), a, y)
    bT, seT, beta = p2sls(a, x, z, w, y)
    o["tsls"], o["tsls_se"] = bT, seT
    hw = beta[0] + w @ beta[2:4] + x @ beta[4:]
    F = add1(np.column_stack([z, x]))
    G = add1(np.column_stack([w, x]))
    # unpenalised q only (the cross-validated regularised fit is too slow at p = 70)
    # p = 70 covariates: cap the evaluations (unsolved fits otherwise run for minutes)
    q, info = q_bridge(F, G, a, "plain", seed=r, nfev=PLAS_NFEV)
    dr, se, ipw, ipw_se = pdr(a, y, hw + beta[1], hw, q)
    o["dr_plain"], o["dr_plain_se"] = dr, se
    o["ok_plain"] = int(max(i[0] for i in info) < TOL)
    o["q_maxmom"] = max(i[0] for i in info)
    base = np.column_stack([a, x])
    o["F_min"] = min(partial_f(w[:, j], base, z) for j in range(2))
    return o


def part_plasmode(workers, reps):
    jobs = [(s, r) for s in PLAS_SCEN for r in range(reps)]
    with ProcessPoolExecutor(workers) as ex:
        R = list(ex.map(plas_rep, jobs))
    write("v110_plasmode_reps.csv", [{k: (r4(v) if isinstance(v, float) else v) for k, v in x.items()} for x in R])
    # residual Z-W dependence given the withheld confounder and X (proximal condition check)
    d, U, _, _ = plas_setup()
    XU = add1(np.column_stack([d["x"], U]))
    rz = d["z"] - XU @ np.linalg.lstsq(XU, d["z"], rcond=None)[0]
    rw = d["w"] - XU @ np.linalg.lstsq(XU, d["w"], rcond=None)[0]
    pc = np.corrcoef(np.column_stack([rz, rw]).T)[:2, 2:]
    summ = []
    for s in PLAS_SCEN:
        S = [x for x in R if x["scen"] == s]
        t = np.array([x["truth"] for x in S])
        for f in ("x_only", "proxy", "aipw", "tsls", "dr_plain"):
            e = np.array([x[f] for x in S])
            se = np.array([x[f + "_se"] for x in S])
            err = e - t
            cov = np.abs(err) <= 1.96 * se
            summ.append(dict(scen=s, estimator=f, reps=len(S), truth=r4(t.mean()), bias=r4(err.mean()),
                             bias_mcse=r4(err.std(ddof=1) / np.sqrt(len(S))),
                             rmse=r4(np.sqrt(np.mean(err ** 2))), coverage=r4(cov.mean()),
                             coverage_mcse=r4(np.sqrt(cov.mean() * (1 - cov.mean()) / len(S)))))
        ok = np.array([x["ok_plain"] for x in S])
        mmq = np.array([x["q_maxmom"] for x in S])
        Fm = np.array([x["F_min"] for x in S])
        dr = np.array([x["dr_plain"] for x in S])
        ts, drs, tss = (np.array([x[k] for x in S]) for k in ("tsls", "dr_plain_se", "tsls_se"))
        px = np.array([x["proxy"] for x in S])
        better_dr = np.abs(dr - t) < np.abs(px - t)
        disagree = np.abs(dr - ts) > 1.96 * np.sqrt(drs ** 2 + tss ** 2)
        flag = (ok == 0) | (Fm < 16) | disagree
        # v1 rule with the true (zero) exclusion violations: DR unless weak/unsolved -> proxy
        v1 = np.where((ok == 0) | (Fm < 16), "proxy", "dr")
        rules = {"always DR": np.full(len(S), "dr"), "always proxy": np.full(len(S), "proxy"),
                 "v1 rule (elicited gamma=delta=0)": v1,
                 "v2 flags -> proxy when flagged": np.where(flag, "proxy", "dr")}
        for name, ch in rules.items():
            acc = (ch == "dr") == better_dr
            summ.append(dict(scen=s, estimator="selection: " + name, reps=len(S), truth=r4(t.mean()),
                             bias="", bias_mcse="", rmse="", coverage=r4(acc.mean()),
                             coverage_mcse=r4(np.sqrt(acc.mean() * (1 - acc.mean()) / len(S)))))
        summ.append(dict(scen=s, estimator="diag: q unsolved / median max|m| / flagged / median F", reps=len(S),
                         truth="", bias=r4(1 - ok.mean()), bias_mcse=float(f"{np.median(mmq):.3g}"),
                         rmse=r4(flag.mean()), coverage=r4(np.median(Fm)), coverage_mcse=""))
    summ.append(dict(scen="all", estimator="partial corr(Z,W | X, U) max abs", reps="", truth="",
                     bias=r4(np.max(np.abs(pc))), bias_mcse="", rmse="", coverage="", coverage_mcse=""))
    write("v110_plasmode_summary.csv", summ)


# =========================================================================== part sharp
def sharp_main_job(job):
    domain, r, B = job
    cfg = uc.DOMAINS[domain]
    truth = uc.true_ate(cfg, 200_000, np.random.default_rng(42 + 999))
    d = uc.simulate(cfg, 8000, np.random.default_rng(42 + 1000 * r + uc.domain_offset(domain, 997)))
    X = np.column_stack([d["age"], d["z"], d["w"]])
    a, y = d["a"], d["y"]
    tb, sb = tan_bounds(X, a, y, GRID), sharp_bounds(X, a, y, GRID)
    o = dict(domain=domain, rep=r, truth=truth,
             lstar_tan=smallest(tb, truth), lstar_sharp=smallest(sb, truth),
             l0_tan=smallest(tb, 0.0), l0_sharp=smallest(sb, 0.0))
    lams = (1.0, 1.15, 1.5, 2.0)
    idx = [int(np.argmin(np.abs(GRID - l))) for l in lams]
    for nm, bb in (("tan", tb), ("sharp", sb)):
        for l, i in zip(lams, idx):
            o[f"w_{nm}_{l}"] = bb[i][1] - bb[i][0]
    rng = np.random.default_rng(42 + 88_000 + 1000 * r + uc.domain_offset(domain, 997))
    lo = {l: [] for l in lams}
    hi = {l: [] for l in lams}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        for l, (bl, bh) in zip(lams, sharp_bounds(X[i], a[i], y[i], lams)):
            lo[l].append(bl)
            hi[l].append(bh)
    for l in lams:
        L, H = np.percentile(lo[l], 2.5), np.percentile(hi[l], 97.5)
        o[f"cov_sharp_{l}"] = int(L <= truth <= H)
        o[f"excl0_sharp_{l}"] = int(H < 0 or L > 0)
    return o


def sharp_held_job(job):
    import decision_rule_v2 as v2
    import review_round2 as r2
    g, r = job
    sname = "B (seed 2031, fresh)"
    p = v2.design(v2.SETS[sname])[g]
    t = r2.truth_general(p)
    sd = v2.SETS[sname] * 1000 + 5_000_000 + 10_000 * g + r
    d = r2.sim_general(p, v2.SIZE, np.random.default_rng(sd))
    X = np.column_stack([d["age"], d["z"], d["w"]])
    tb, sb = tan_bounds(X, d["a"], d["y"], GRID), sharp_bounds(X, d["a"], d["y"], GRID)
    return dict(gen=g, rep=r, truth=t, lstar_tan=smallest(tb, t), lstar_sharp=smallest(sb, t),
                l0_tan=smallest(tb, 0.0), l0_sharp=smallest(sb, 0.0))


def part_sharp(workers, reps, B, held_reps):
    jobs = [(dm, r, B) for dm in uc.DOMAINS for r in range(reps)]
    with ProcessPoolExecutor(workers) as ex:
        M = list(ex.map(sharp_main_job, jobs))
        H = list(ex.map(sharp_held_job, [(g, r) for g in range(60) for r in range(held_reps)]))
    write("v110_sharp_main.csv", [{k: (r4(v) if isinstance(v, float) else v) for k, v in x.items()} for x in M])
    write("v110_sharp_heldout.csv", [{k: (r4(v) if isinstance(v, float) else v) for k, v in x.items()} for x in H])
    summ = []
    for dm in uc.DOMAINS:
        S = [x for x in M if x["domain"] == dm]
        g = lambda k: np.array([x[k] for x in S], float)
        row = dict(analysis="main", domain=dm, reps=len(S), truth=r4(S[0]["truth"]),
                   lstar_tan_med=r4(np.median(g("lstar_tan"))), lstar_sharp_med=r4(np.median(g("lstar_sharp"))),
                   l0_tan_med=r4(np.median(g("l0_tan"))), l0_sharp_med=r4(np.median(g("l0_sharp"))),
                   width_ratio_1_5=r4(np.mean(g("w_sharp_1.5") / g("w_tan_1.5"))),
                   width_sharp_1_5=r4(np.mean(g("w_sharp_1.5"))), width_tan_1_5=r4(np.mean(g("w_tan_1.5"))),
                   share_sign_lost_before_truth_sharp=r4(np.mean(g("l0_sharp") <= g("lstar_sharp"))))
        for l in (1.0, 1.15, 1.5, 2.0):
            row[f"cov_sharp_{l}"] = r4(np.mean(g(f"cov_sharp_{l}")))
            row[f"excl0_sharp_{l}"] = r4(np.mean(g(f"excl0_sharp_{l}")))
        summ.append(row)
    gm = {}
    for x in H:
        gm.setdefault(x["gen"], []).append(x)
    ls_t = np.array([np.median([y["lstar_tan"] for y in v]) for v in gm.values()])
    ls_s = np.array([np.median([y["lstar_sharp"] for y in v]) for v in gm.values()])
    nonnull = np.array([abs(v[0]["truth"]) >= 0.02 for v in gm.values()])
    lost = np.array([np.mean([y["l0_sharp"] <= y["lstar_sharp"] for y in v]) for v in gm.values()])
    lost_t = np.array([np.mean([y["l0_tan"] <= y["lstar_tan"] for y in v]) for v in gm.values()])
    summ.append(dict(analysis="heldout B", domain="60 generators", reps=held_reps, truth="",
                     lstar_tan_med=r4(np.median(ls_t)), lstar_sharp_med=r4(np.median(ls_s)),
                     l0_tan_med="", l0_sharp_med="", width_ratio_1_5="", width_sharp_1_5="", width_tan_1_5="",
                     share_sign_lost_before_truth_sharp=r4(lost[nonnull].mean()),
                     **{"cov_sharp_1.0": r4(lost_t[nonnull].mean()), "excl0_sharp_1.0": int(nonnull.sum()),
                        "cov_sharp_1.15": r4(np.mean(ls_s <= 1.5)), "excl0_sharp_1.15": r4(np.mean(ls_s <= 2.0)),
                        "cov_sharp_1.5": r4(np.mean(~np.isfinite(ls_s))), "excl0_sharp_1.5": "",
                        "cov_sharp_2.0": "", "excl0_sharp_2.0": ""}))
    write("v110_sharp_summary.csv", summ)


# =========================================================================== part qreg
def qreg_job(job):
    import decision_rule_v2 as v2
    import review_round2 as r2
    sname, g, r = job
    p = v2.design(v2.SETS[sname])[g]
    sd = v2.SETS[sname] * 1000 + 5_000_000 + 10_000 * g + r
    d = r2.sim_general(p, v2.SIZE, np.random.default_rng(sd))
    n = len(d["y"])
    F = np.column_stack([np.ones(n), d["z"], d["age"]])
    G = np.column_stack([np.ones(n), d["w"], d["age"]])
    b = uc.outcome_bridge(d)
    h0 = b[0] + b[2] * d["w"] + b[3] * d["age"]
    o = dict(set=sname, gen=g, rep=r)
    for mode in ("plain", "reg"):
        q, info = q_bridge(F, G, d["a"], mode, seed=sd)
        dr, se, _, _ = pdr(d["a"], d["y"], h0 + b[1], h0, q)
        o[f"dr_{mode}"], o[f"se_{mode}"] = dr, se
        o[f"maxmom_{mode}"] = max(i[0] for i in info)
        o[f"lam_{mode}"] = max(i[1] for i in info)
        o[f"status_{mode}"] = min(i[2] for i in info)
    return o


def part_qreg(workers, reps):
    import decision_rule_v2 as v2
    import review_round2 as r2
    jobs = [(s, g, r) for s in v2.SETS for g in range(v2.N_GEN) for r in range(reps)]
    with ProcessPoolExecutor(workers) as ex:
        R = list(ex.map(qreg_job, jobs, chunksize=8))
    write("v110_qreg_reps.csv", [{k: (r4(v) if isinstance(v, float) and k.startswith(("dr", "se")) else v)
                                  for k, v in x.items()} for x in R])
    old = list(csv.DictReader(open(RES / "v2_rule_reps.csv")))
    summ = []
    for s in v2.SETS:
        S = [x for x in R if x["set"] == s]
        des = v2.design(v2.SETS[s])
        truth = {g: r2.truth_general(des[g]) for g in range(v2.N_GEN)}
        t = np.array([truth[x["gen"]] for x in S])
        row = dict(set=s, reps=len(S),
                   old_unsolved_stored=r4(1 - np.mean([int(x["ok"]) for x in old if x["set"] == s])))
        for mode in ("plain", "reg"):
            e = np.array([x[f"dr_{mode}"] for x in S])
            se = np.array([x[f"se_{mode}"] for x in S])
            mm = np.array([x[f"maxmom_{mode}"] for x in S])
            st = np.array([x[f"status_{mode}"] for x in S])
            cov = np.abs(e - t) <= 1.96 * se
            gb = [np.mean((e - t)[[x["gen"] == g for x in S]]) for g in range(v2.N_GEN)]
            row.update({f"{mode}_unsolved_1e6": r4(np.mean(mm >= TOL)),
                        f"{mode}_imbalance_gt_0.01": r4(np.mean(mm > 0.01)),
                        f"{mode}_optimiser_fail": r4(np.mean(st <= 0)),
                        f"{mode}_abs_bias_mean_gen": r4(np.mean(np.abs(gb))),
                        f"{mode}_rmse": r4(np.sqrt(np.mean((e - t) ** 2))),
                        f"{mode}_coverage": r4(cov.mean()),
                        f"{mode}_coverage_mcse": r4(np.sqrt(cov.mean() * (1 - cov.mean()) / len(S)))})
        lam = np.array([x["lam_reg"] for x in S])
        row["reg_share_lambda_pos"] = r4(np.mean(lam > 0))
        summ.append(row)
    write("v110_qreg_summary.csv", summ)


# =========================================================================== part theory
def prop1_check(rng):
    """Proposition 1 (extended): limit of the proxy-adjusted OLS coefficient in the linear
    Gaussian model with unequal noise variances, expressed through proxy reliabilities."""
    rows = []
    n = 2_000_000
    for (aA, aT, bT, bU, lz, lw, sz, sw, sT) in [(0.6, 1.2, -0.5, 1.1, 1.0, 1.1, 1.0, 1.0, 1.0),
                                                (0.5, 0.8, 0.3, 1.0, 0.5, 0.3, 1.5, 0.7, 0.5),
                                                (0.0, 1.5, 0.0, 0.7, 2.0, 0.2, 0.5, 2.0, 1.2),
                                                (0.7, 0.4, -1.0, 1.4, 0.25, 0.25, 1.0, 1.0, 0.3)]:
        A = rng.normal(size=n)
        U = aA * A + rng.normal(size=n)
        Z = lz * U + sz * rng.normal(size=n)
        W = lw * U + sw * rng.normal(size=n)
        T = aT * U + sT * rng.normal(size=n)
        Y = bT * T + bU * U + 0.3 * A + rng.normal(size=n)
        est = ols_hc0(np.column_stack([T, A, Z, W]), Y, 0)[0]
        kz, kw = lz ** 2 / sz ** 2, lw ** 2 / sw ** 2          # signal-to-noise; rho_j = k_j/(1+k_j)
        s2 = 1.0 / (1.0 + kz + kw)
        R = (kz + kw) / (1 + kz + kw)                          # reliability of E(U | A, Z, W)
        lim = bT + bU * aT * s2 / (aT ** 2 * s2 + sT ** 2)
        rows.append(dict(check="prop1 OLS limit", params=f"lz={lz},lw={lw},sz={sz},sw={sw},aT={aT},sT={sT}",
                         formula=r4(lim), simulated=r4(est), diff=r4(est - lim), reliability=r4(R)))
    return rows


def prop2_check(rng):
    """Proposition 2: an observationally equivalent no-hidden-confounder model M2 for a law
    generated by a proximal model M1 (linear Gaussian). Same observed covariance; the proximal
    estimand equals the M1 effect, the adjusted-regression estimand equals the M2 effect."""
    n = 1_000_000
    aA, aT, bT, bU, lz, lw = 0.6, 1.0, -0.5, 1.0, 1.0, 1.0
    A = rng.normal(size=n)
    U = aA * A + rng.normal(size=n)
    Z = lz * U + rng.normal(size=n)
    W = lw * U + rng.normal(size=n)
    T = aT * U + rng.normal(size=n)
    Y = bT * T + bU * U + 0.3 * A + rng.normal(size=n)
    V1 = np.column_stack([A, Z, W, T, Y])
    S = np.cov(V1.T)
    # M2: recursive Gaussian SCM A -> Z -> W -> T -> Y with every observed parent, no U
    V2 = np.zeros((n, 5))
    coefs = []
    for j in range(5):
        if j == 0:
            V2[:, 0] = rng.normal(size=n) * np.sqrt(S[0, 0])
            continue
        Spp, Spj = S[:j, :j], S[:j, j]
        c = np.linalg.solve(Spp, Spj)
        resid = S[j, j] - Spj @ c
        V2[:, j] = V2[:, :j] @ c + np.sqrt(resid) * rng.normal(size=n)
        coefs.append(c)
    S2 = np.cov(V2.T)
    m2_effect = coefs[-1][3]               # coefficient of T in the M2 outcome equation
    d1 = dict(a=T, z=Z, w=W, x=A[:, None], y=Y)
    d2 = dict(a=V2[:, 3], z=V2[:, 1], w=V2[:, 2], x=V2[:, :1], y=V2[:, 4])
    out = []
    for nm, d in (("M1", d1), ("M2", d2)):
        out.append(dict(check=f"prop2 {nm}", params="linear Gaussian", formula="",
                        simulated=r4(p2sls(d["a"], d["x"], d["z"][:, None], d["w"][:, None], d["y"])[0]),
                        diff=r4(ols_hc0(np.column_stack([d["a"], d["x"], d["z"], d["w"]]), d["y"], 0)[0]),
                        reliability=""))
    out.append(dict(check="prop2 max |cov(M1) - cov(M2)|", params="", formula="",
                    simulated=r4(np.max(np.abs(S - S2))), diff="", reliability=""))
    out.append(dict(check="prop2 true effects M1 / M2", params="", formula=r4(bT), simulated=r4(m2_effect),
                    diff="", reliability=""))
    return out


def prop3_check(rng, reps=200, n=4000):
    """Proposition 3: double robustness of the proximal policy-value estimator for a rule
    reading only A. Continuous linear outcome so the linear h is correct when W is kept."""
    rows = []
    res = {k: [] for k in ("both", "h_wrong", "q_wrong", "both_wrong")}
    truth_draw = None
    for r in range(reps):
        A = rng.normal(size=n)
        U = 0.6 * A + rng.normal(size=n)
        Z = U + rng.normal(size=n)
        W = U + rng.normal(size=n)
        T = (rng.uniform(size=n) < uc.sigmoid(1.0 * U - 0.2)).astype(float)
        Y = -0.5 * T + 1.0 * U + 0.3 * A + rng.normal(size=n)
        pi = (A > 0).astype(float)
        if truth_draw is None:
            Am = rng.normal(size=2_000_000)
            Um = 0.6 * Am + rng.normal(size=2_000_000)
            truth_draw = float(np.mean(-0.5 * (Am > 0) + Um + 0.3 * Am))
        x = A[:, None]
        _, _, beta = p2sls(T, x, Z[:, None], W[:, None], Y)
        hc = lambda t: beta[0] + beta[1] * t + beta[2] * W + beta[3] * A
        bw = np.linalg.lstsq(add1(np.column_stack([T, A])), Y, rcond=None)[0]
        hw = lambda t: bw[0] + bw[1] * t + bw[2] * A          # drops W: misspecified
        Fc, Gc = add1(np.column_stack([Z, A])), add1(np.column_stack([W, A]))
        qc, _ = q_bridge(Fc, Gc, T, "plain")
        Fw = add1(A[:, None])
        qw, _ = q_bridge(Fw, Fw, T, "plain")                  # drops Z: misspecified
        for key, h, q in (("both", hc, qc), ("h_wrong", hw, qc), ("q_wrong", hc, qw), ("both_wrong", hw, qw)):
            v = np.mean(h(pi) + (T == pi) * q * (Y - h(T)))
            res[key].append(v - truth_draw)
    for k, v in res.items():
        v = np.array(v)
        rows.append(dict(check=f"prop3 value bias, {k}", params=f"reps={reps}, n={n}", formula=0.0,
                         simulated=r4(v.mean()), diff=r4(v.std(ddof=1) / np.sqrt(reps)), reliability=""))
    return rows


def part_theory():
    rng = np.random.default_rng(20261022)
    rows = prop1_check(rng) + prop2_check(rng) + prop3_check(rng)
    write("v110_theory_checks.csv", rows)


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="theory,sharp,qreg,rhc,plasmode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--plas-reps", type=int, default=100)
    ap.add_argument("--sharp-reps", type=int, default=100)
    ap.add_argument("--sharp-boot", type=int, default=100)
    ap.add_argument("--qreg-reps", type=int, default=40)
    a = ap.parse_args()
    parts = a.parts.split(",")
    if "theory" in parts:
        part_theory()
    if "sharp" in parts:
        part_sharp(a.workers, a.sharp_reps, a.sharp_boot, 10)
    if "qreg" in parts:
        part_qreg(a.workers, a.qreg_reps)
    if "rhc" in parts:
        part_rhc(a.workers, a.boot)
    if "plasmode" in parts:
        part_plasmode(a.workers, a.plas_reps)


if __name__ == "__main__":
    main()
