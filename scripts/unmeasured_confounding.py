"""Two things the backdoor pipeline cannot do, and the estimators that can.

The main driver (``run_all.py``) adjusts for the confounder it simulated. That is a
correctness check of the implementation, not evidence about evaluation practice: in a real
CDSS audit the confounder that matters is usually the one nobody recorded, and the quantity
a hospital wants is not an average treatment effect but the value of *following the system's
advice*. This script supplies both, on a simulator where the answer is known.

Experiment A -- proximal causal inference under unmeasured confounding
    Severity is hidden. Observed instead are two proxies: a negative-control exposure Z
    (driven by severity, with no direct effect on the outcome) and a negative-control outcome
    W (driven by severity, unaffected by treatment). This is the proximal structure of
    Miao, Geng and Tchetgen Tchetgen (2018) and Tchetgen Tchetgen et al. (2024). Backdoor
    adjustment on the recorded covariates alone is then biased by construction, and proximal
    two-stage least squares recovers the effect. Both are run against the interventional
    ground truth over many seeds, so the claim is a measured bias, not an assertion.

Experiment B -- off-policy evaluation of the CDSS as a decision rule
    A CDSS is a policy pi(x) in {treat, withhold}. Its value V(pi) = E[Y(pi(X))] is estimated
    from observational data by inverse propensity weighting, self-normalised IPW and the
    doubly-robust estimator, and compared with the true value obtained by intervening in the
    SCM with do(A = pi(x)). Reported per policy: bias, RMSE, and the coverage of nominal 95%
    intervals across seeds -- the property that decides whether an estimate can be trusted.

Outputs: results/proximal_estimates.csv, results/policy_value_estimates.csv,
         results/policy_value_coverage.csv

Usage: python scripts/unmeasured_confounding.py [--reps 200] [--n 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import zlib

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"

# Domain parameters mirror run_all.DOMAINS; they are restated here because this simulator
# adds the two proxy variables and hides severity, so the graph is not the same object.
DOMAINS = {
    "sepsis":  dict(age_to_sev=0.60, sev_to_treat=1.20, treat_intercept=-0.30,
                    sev_to_out=1.10, age_to_out=0.35, treat_to_out=-1.00, out_intercept=-0.40,
                    sev_to_z=1.00, sev_to_w=1.10),
    "ards":    dict(age_to_sev=0.50, sev_to_treat=0.90, treat_intercept=-0.20,
                    sev_to_out=0.95, age_to_out=0.30, treat_to_out=-0.65, out_intercept=-0.35,
                    sev_to_z=0.90, sev_to_w=1.00),
    "cardiac": dict(age_to_sev=0.70, sev_to_treat=1.40, treat_intercept=-0.40,
                    sev_to_out=1.25, age_to_out=0.40, treat_to_out=-1.30, out_intercept=-0.45,
                    sev_to_z=1.10, sev_to_w=1.20),
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(cfg: dict, n: int, rng: np.random.Generator) -> dict:
    """Draw one cohort. ``severity`` is returned but treated as unmeasured by the estimators."""
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    # Negative-control exposure: driven by severity, no edge into the outcome.
    z = cfg["sev_to_z"] * sev + rng.normal(0, 1, n)
    # Negative-control outcome: driven by severity, no edge from treatment.
    w = cfg["sev_to_w"] * sev + rng.normal(0, 1, n)
    pt = sigmoid(cfg["sev_to_treat"] * sev + cfg["treat_intercept"])
    a = (rng.uniform(0, 1, n) < pt).astype(float)
    py = sigmoid(cfg["sev_to_out"] * sev + cfg["age_to_out"] * age
                 + cfg["treat_to_out"] * a + cfg["out_intercept"])
    y = (rng.uniform(0, 1, n) < py).astype(float)
    return dict(age=age, severity=sev, z=z, w=w, a=a, y=y, p_treat=pt, p_out=py)


def domain_offset(domain: str, mod: int) -> int:
    """Per-domain seed offset that is identical in every process (hash() is salted)."""
    return zlib.crc32(domain.encode("utf-8")) % mod


def true_ate(cfg: dict, n: int, rng: np.random.Generator) -> float:
    """E[Y(1)] - E[Y(0)] by intervening on the same mechanisms."""
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    base = cfg["sev_to_out"] * sev + cfg["age_to_out"] * age + cfg["out_intercept"]
    return float(np.mean(sigmoid(base + cfg["treat_to_out"]) - sigmoid(base)))


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def naive_ate(d: dict) -> float:
    return float(d["y"][d["a"] == 1].mean() - d["y"][d["a"] == 0].mean())


def ols_robust(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC0 sandwich standard errors (intercept first)."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
    e = y - Xd @ beta
    bread = np.linalg.inv(Xd.T @ Xd)
    meat = (Xd * e[:, None] ** 2).T @ Xd
    return beta, np.sqrt(np.diag(bread @ meat @ bread))


def backdoor_observed_ate(d: dict) -> float:
    """Adjust for age only: the comparator that ignores the recorded proxies."""
    X = np.column_stack([d["a"], d["age"]])
    beta = ols(X, d["y"])
    return float(beta[1])


def backdoor_proxy_se(d: dict) -> tuple[float, float]:
    """Adjust for every recorded variable, proxies included (age, Z, W): the fair comparator.

    Z and W are noisy measurements of severity, so conditioning on them removes part of the
    confounding but not all of it; this is what an auditor holding the same records as the
    proximal estimator would do without invoking the proximal assumptions.
    """
    beta, se = ols_robust(np.column_stack([d["a"], d["age"], d["z"], d["w"]]), d["y"])
    return float(beta[1]), float(se[1])


def backdoor_proxy_ate(d: dict) -> float:
    return backdoor_proxy_se(d)[0]


def proximal_ate(d: dict) -> float:
    """Proximal two-stage least squares with the (Z, W) negative-control pair.

    Stage 1 regresses the negative-control outcome W on (A, Z, age) to form a bridge; stage 2
    regresses Y on (A, What, age). With W and Z both driven by the hidden confounder and
    satisfying the exclusion restrictions above, the coefficient on A in stage 2 identifies
    the treatment effect on the linear-probability scale.
    """
    return proximal_se(d)[0]


def proximal_se(d: dict) -> tuple[float, float]:
    """Proximal 2SLS estimate and its heteroskedasticity-robust (HC0) 2SLS standard error.

    The residual is formed with the observed W, not its first-stage fit, as 2SLS inference
    requires; the bread uses the second-stage design.
    """
    n = len(d["y"])
    X1 = np.column_stack([d["a"], d["z"], d["age"]])
    w_hat = np.column_stack([np.ones(n), X1]) @ ols(X1, d["w"])
    Xh = np.column_stack([np.ones(n), d["a"], w_hat, d["age"]])
    beta = np.linalg.lstsq(Xh, d["y"], rcond=None)[0]
    e = d["y"] - np.column_stack([np.ones(n), d["a"], d["w"], d["age"]]) @ beta
    bread = np.linalg.inv(Xh.T @ Xh)
    meat = (Xh * e[:, None] ** 2).T @ Xh
    return float(beta[1]), float(np.sqrt((bread @ meat @ bread)[1, 1]))


# Non-convergence counter for the treatment-bridge GMM; read and reset by the drivers.
BRIDGE_FAILS = {"count": 0, "calls": 0}


def outcome_bridge(d: dict) -> np.ndarray:
    """Linear outcome bridge h(W, A, X) = b0 + bA*A + bW*W + bX*age, fitted by 2SLS with
    instruments (1, A, Z, age). Returns (b0, bA, bW, bX), the coefficients of proximal_se."""
    n = len(d["y"])
    X1 = np.column_stack([d["a"], d["z"], d["age"]])
    w_hat = np.column_stack([np.ones(n), X1]) @ ols(X1, d["w"])
    Xh = np.column_stack([np.ones(n), d["a"], w_hat, d["age"]])
    return np.linalg.lstsq(Xh, d["y"], rcond=None)[0]


def treatment_bridge(d: dict) -> np.ndarray:
    """Treatment bridge q(Z, A, X) of Cui et al. (2024), evaluated at the observed arm.

    For each arm a, q_a(Z, X) = 1 + exp(t_a'(1, Z, age)) solves the just-identified moment
    equations E[1{A=a} q_a(Z, X) g(W, X) - g(W, X)] = 0 with g = (1, W, age). Starting values
    are the inverse-propensity form implied by a logistic regression of A on (1, Z, age);
    a zero start is tried if that fails. A solve counts as converged when every sample moment
    is below 1e-6 in absolute value.
    """
    from scipy.optimize import least_squares
    n = len(d["a"])
    F = np.column_stack([np.ones(n), d["z"], d["age"]])
    G = np.column_stack([np.ones(n), d["w"], d["age"]])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(F[:, 1:], d["a"])
    eta = np.r_[lr.intercept_, lr.coef_.ravel()]
    q = np.empty(n)
    for arm, start in ((1, -eta), (0, eta)):
        ind = (d["a"] == arm).astype(float)

        def mom(t):
            qa = 1.0 + np.exp(np.clip(F @ t, -50, 50))
            return ((ind * qa)[:, None] * G - G).mean(axis=0)

        BRIDGE_FAILS["calls"] += 1
        best = None
        for t0 in (start, np.zeros(3)):
            sol = least_squares(mom, t0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.max(np.abs(sol.fun)) < np.max(np.abs(best.fun)):
                best = sol
            if np.max(np.abs(sol.fun)) < 1e-6:
                break
        if np.max(np.abs(best.fun)) >= 1e-6:
            BRIDGE_FAILS["count"] += 1
        qa = 1.0 + np.exp(np.clip(F @ best.x, -50, 50))
        q[ind == 1] = qa[ind == 1]
    return q


def proximal_dr_se(d: dict) -> tuple[float, float, float, float]:
    """Semiparametric doubly robust proximal estimator (Cui et al., 2024) and proximal IPW.

    psi_DR  = (2A-1) q(Z,A,X) (Y - h(W,A,X)) + h(W,1,X) - h(W,0,X)
    psi_IPW = (2A-1) q(Z,A,X) Y
    Each SE is sd(psi)/sqrt(n): it treats h and q as known and so ignores the variance
    contributed by estimating them. Returns (dr, dr_se, ipw, ipw_se).
    """
    b = outcome_bridge(d)
    h = b[0] + b[1] * d["a"] + b[2] * d["w"] + b[3] * d["age"]
    q = treatment_bridge(d)
    sgn = 2 * d["a"] - 1
    psi = sgn * q * (d["y"] - h) + b[1]
    psi_ipw = sgn * q * d["y"]
    rn = np.sqrt(len(psi))
    return (float(psi.mean()), float(psi.std(ddof=1) / rn),
            float(psi_ipw.mean()), float(psi_ipw.std(ddof=1) / rn))


def proximal_dr_only(d: dict) -> tuple[float, float]:
    th, se, _, _ = proximal_dr_se(d)
    return th, se


def first_stage_f(d: dict) -> float:
    """Partial F statistic of Z in the first-stage regression of W on (A, Z, age)."""
    n = len(d["w"])
    full = np.column_stack([np.ones(n), d["a"], d["z"], d["age"]])
    red = np.column_stack([np.ones(n), d["a"], d["age"]])
    rss = lambda X: float(np.sum((d["w"] - X @ np.linalg.lstsq(X, d["w"], rcond=None)[0]) ** 2))
    return (rss(red) - rss(full)) / (rss(full) / (n - full.shape[1]))


# --------------------------------------------------------------------------- policies
def policy_treat_all(d):
    return np.ones(len(d["a"]))


def policy_treat_none(d):
    return np.zeros(len(d["a"]))


def policy_cdss(d, threshold=0.5, train=None):
    """A data-driven CDSS: treat when predicted risk from recorded covariates is in the top q.

    It sees age, z and w -- everything except the latent severity. The risk model and the
    cut-off are learned on ``train``, an independent cohort, so the policy is fixed before the
    evaluation data are touched (a policy fitted on the evaluation sample would make the
    doubly robust standard error invalid).
    """
    train = d if train is None else train
    Xt = np.column_stack([train["age"], train["z"], train["w"]])
    beta = ols(Xt, train["y"])
    cut = np.quantile(np.column_stack([np.ones(len(Xt)), Xt]) @ beta, 1 - threshold)
    X = np.column_stack([np.ones(len(d["age"])), d["age"], d["z"], d["w"]])
    return (X @ beta > cut).astype(float)


KB_MIN_REDUCTION = 0.15   # minimal clinically important absolute risk reduction


def posterior_severity(cfg: dict, d: dict) -> np.ndarray:
    """E[severity | age, z, w] under the declared linear-Gaussian structural equations."""
    # severity | age ~ N(age_to_sev*age, 1); z, w | severity ~ N(k*severity, 1)
    prec = 1.0 + cfg["sev_to_z"] ** 2 + cfg["sev_to_w"] ** 2
    return (cfg["age_to_sev"] * d["age"] + cfg["sev_to_z"] * d["z"]
            + cfg["sev_to_w"] * d["w"]) / prec


def policy_knowledge(d, cfg):
    """A knowledge-driven CDSS: recommend treatment when the declared causal model predicts it.

    The recommendation is a function of the declared graph and its structural equations only:
    severity is inferred from the recorded proxies by the declared measurement model, and the
    patient is treated when the model's predicted absolute risk reduction exceeds
    KB_MIN_REDUCTION. No outcome data are used to form the rule.
    """
    sev = posterior_severity(cfg, d)
    base = cfg["sev_to_out"] * sev + cfg["age_to_out"] * d["age"] + cfg["out_intercept"]
    reduction = sigmoid(base) - sigmoid(base + cfg["treat_to_out"])
    return (reduction > KB_MIN_REDUCTION).astype(float)


POLICY_NAMES = ("treat_all", "treat_none", "cdss_top50", "cdss_top25", "knowledge")


def make_policies(d, train, cfg):
    return {"treat_all": policy_treat_all(d), "treat_none": policy_treat_none(d),
            "cdss_top50": policy_cdss(d, 0.5, train), "cdss_top25": policy_cdss(d, 0.25, train),
            "knowledge": policy_knowledge(d, cfg)}


def true_policy_value(cfg: dict, d: dict, pi: np.ndarray) -> float:
    """V(pi) under do(A = pi(x)), using the outcome mechanism the data came from."""
    base = (cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"]
            + cfg["out_intercept"])
    return float(np.mean(sigmoid(base + cfg["treat_to_out"] * pi)))


NUISANCES = ("linear_recorded", "logistic_recorded", "logistic_oracle")


def fit_nuisances(d: dict, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propensity e(x) and arm-specific outcome means mu1(x), mu0(x).

    linear_recorded    linear-probability models on (age, z, w): the first version's choice
    logistic_recorded  logistic models, arm-specific outcome fits, same recorded covariates
    logistic_oracle    as above with severity added -- not available to an audit; it separates
                       bias from the missing confounder from bias due to limited overlap
    """
    cols = ["age", "z", "w"] + (["severity"] if kind == "logistic_oracle" else [])
    X = np.column_stack([d[c] for c in cols])
    if kind == "linear_recorded":
        Xd = np.column_stack([np.ones(len(X)), X])
        e = np.clip(Xd @ ols(X, d["a"]), 0.02, 0.98)
        beta_y = ols(np.column_stack([d["a"], X]), d["y"])
        mu1 = np.column_stack([np.ones(len(X)), np.ones(len(X)), X]) @ beta_y
        mu0 = np.column_stack([np.ones(len(X)), np.zeros(len(X)), X]) @ beta_y
        return e, mu1, mu0
    lr = lambda: LogisticRegression(C=1e6, max_iter=1000)
    e = np.clip(lr().fit(X, d["a"]).predict_proba(X)[:, 1], 0.02, 0.98)
    t = d["a"] == 1
    mu1 = lr().fit(X[t], d["y"][t]).predict_proba(X)[:, 1]
    mu0 = lr().fit(X[~t], d["y"][~t]).predict_proba(X)[:, 1]
    return e, mu1, mu0


def ope_estimates(d: dict, pi: np.ndarray, nuis=None) -> dict:
    """IPW, self-normalised IPW and doubly-robust value estimates for policy ``pi``."""
    e, mu1, mu0 = fit_nuisances(d, "linear_recorded") if nuis is None else nuis
    mu_pi = pi * mu1 + (1 - pi) * mu0

    match = (d["a"] == pi).astype(float)
    prob_pi = pi * e + (1 - pi) * (1 - e)
    w = match / prob_pi
    ipw = float(np.mean(w * d["y"]))
    snipw = float(np.sum(w * d["y"]) / np.sum(w)) if np.sum(w) > 0 else float("nan")
    dr = float(np.mean(mu_pi + w * (d["y"] - (d["a"] * mu1 + (1 - d["a"]) * mu0))))
    se_dr = float(np.std(mu_pi + w * (d["y"] - (d["a"] * mu1 + (1 - d["a"]) * mu0)),
                         ddof=1) / np.sqrt(len(d["y"])))
    # Overlap diagnostics: a policy far from observed practice is evaluated through a few
    # heavily weighted patients, and the interval is then meaningless however small its bias.
    ess = float(w.sum() ** 2 / np.sum(w ** 2)) if np.any(w > 0) else 0.0
    psi_ipw = w * d["y"]
    return {"ipw": ipw, "snipw": snipw, "dr": dr, "dr_se": se_dr,
            "ipw_se": float(np.std(psi_ipw, ddof=1) / np.sqrt(len(psi_ipw))),
            "outcome_only": float(np.mean(mu_pi)),
            "ess_frac": ess / len(w), "agreement": float(match.mean()),
            "max_weight_share": float(w.max() / w.sum()) if w.sum() > 0 else float("nan")}


def write(rows, name):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name)


def main() -> None:
    ap = argparse.ArgumentParser(description="unmeasured confounding and policy value")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prox_rows, pol_rows, cov_rows = [], [], []
    for domain, cfg in DOMAINS.items():
        truth = true_ate(cfg, 200_000, np.random.default_rng(args.seed + 999))
        est = {k: [] for k in ("naive", "backdoor_observed", "backdoor_proxy", "proximal",
                               "proximal_dr", "proximal_ipw")}
        cover = {k: [] for k in ("backdoor_proxy", "proximal", "proximal_dr", "proximal_ipw")}
        keys = ("ipw", "snipw", "dr", "outcome_only", "truth", "dr_cover", "dr_se",
                "ipw_cover", "ess_frac", "agreement", "max_weight_share", "treated")
        pol_acc = {(p, nk): {k: [] for k in keys} for p in POLICY_NAMES for nk in NUISANCES}

        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + 1000 * r + domain_offset(domain, 997))
            d = simulate(cfg, args.n, rng)
            # independent cohort on which the data-driven policies are learned
            train = simulate(cfg, args.n, np.random.default_rng(
                args.seed + 1000 * r + domain_offset(domain, 997) + 500_000))
            est["naive"].append(naive_ate(d))
            est["backdoor_observed"].append(backdoor_observed_ate(d))
            for name, fn in (("backdoor_proxy", backdoor_proxy_se), ("proximal", proximal_se)):
                th, se = fn(d)
                est[name].append(th)
                cover[name].append(float(th - 1.96 * se <= truth <= th + 1.96 * se))
            dr, dr_se, ipw, ipw_se = proximal_dr_se(d)
            for name, th, se in (("proximal_dr", dr, dr_se), ("proximal_ipw", ipw, ipw_se)):
                est[name].append(th)
                cover[name].append(float(th - 1.96 * se <= truth <= th + 1.96 * se))

            pols = make_policies(d, train, cfg)
            for nk in NUISANCES:
                nuis = fit_nuisances(d, nk)
                for pname, pi in pols.items():
                    v_true = true_policy_value(cfg, d, pi)
                    o = ope_estimates(d, pi, nuis)
                    acc = pol_acc[(pname, nk)]
                    for k in ("ipw", "snipw", "dr", "outcome_only", "dr_se",
                              "ess_frac", "agreement", "max_weight_share"):
                        acc[k].append(o[k])
                    acc["truth"].append(v_true)
                    acc["treated"].append(float(pi.mean()))
                    acc["dr_cover"].append(float(abs(o["dr"] - v_true) <= 1.96 * o["dr_se"]))
                    acc["ipw_cover"].append(float(abs(o["ipw"] - v_true) <= 1.96 * o["ipw_se"]))

        for name, vals in est.items():
            v = np.array(vals)
            prox_rows.append({
                "domain": domain, "estimator": name, "true_ate": round(truth, 4),
                "mean_estimate": round(float(v.mean()), 4),
                "bias": round(float(v.mean() - truth), 4),
                "abs_bias": round(abs(float(v.mean() - truth)), 4),
                "mcse_bias": round(float(v.std(ddof=1) / np.sqrt(len(v))), 4),
                "rmse": round(float(np.sqrt(np.mean((v - truth) ** 2))), 4),
                "sd": round(float(v.std(ddof=1)), 4),
                "ci95_coverage": (round(float(np.mean(cover[name])), 3)
                                  if name in cover else ""),
                "reps": args.reps, "n": args.n,
            })
            print(f"  {domain:8} {name:18} true {truth:+.4f}  est {v.mean():+.4f}  "
                  f"bias {v.mean() - truth:+.4f}")

        for (pname, nk), acc in pol_acc.items():
            truth_v = np.array(acc["truth"])
            row = {"domain": domain, "policy": pname, "nuisance": nk,
                   "true_value": round(float(truth_v.mean()), 4), "reps": args.reps}
            for k in ("ipw", "snipw", "dr", "outcome_only"):
                e = np.array(acc[k])
                row[f"{k}_mean"] = round(float(e.mean()), 4)
                row[f"{k}_bias"] = round(float(np.mean(e - truth_v)), 4)
                row[f"{k}_rmse"] = round(float(np.sqrt(np.mean((e - truth_v) ** 2))), 4)
            pol_rows.append(row)
            cv = float(np.mean(acc["dr_cover"]))
            cov_rows.append({"domain": domain, "policy": pname, "nuisance": nk,
                             "dr_ci95_coverage": round(cv, 4),
                             "mcse_coverage": round(float(np.sqrt(cv * (1 - cv) / args.reps)), 4),
                             "ipw_ci95_coverage": round(float(np.mean(acc["ipw_cover"])), 4),
                             "dr_bias": row["dr_bias"],
                             "mean_dr_se": round(float(np.mean(acc["dr_se"])), 5),
                             "nominal": 0.95,
                             "treated_fraction": round(float(np.mean(acc["treated"])), 4),
                             "mean_ess_fraction": round(float(np.mean(acc["ess_frac"])), 4),
                             "mean_policy_agreement": round(float(np.mean(acc["agreement"])), 4),
                             "mean_max_weight_share": round(float(np.mean(acc["max_weight_share"])), 5),
                             "reps": args.reps})

    print(f"treatment-bridge GMM non-convergence: {BRIDGE_FAILS['count']} of "
          f"{BRIDGE_FAILS['calls']} arm-level solves")
    write(prox_rows, "proximal_estimates.csv")
    write(pol_rows, "policy_value_estimates.csv")
    write(cov_rows, "policy_value_coverage.csv")


if __name__ == "__main__":
    main()
