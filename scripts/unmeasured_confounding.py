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

import numpy as np

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


def backdoor_observed_ate(d: dict) -> float:
    """Adjust for what was recorded -- age only, because severity is unmeasured."""
    X = np.column_stack([d["a"], d["age"]])
    beta = ols(X, d["y"])
    return float(beta[1])


def proximal_ate(d: dict) -> float:
    """Proximal two-stage least squares with the (Z, W) negative-control pair.

    Stage 1 regresses the negative-control outcome W on (A, Z, age) to form a bridge; stage 2
    regresses Y on (A, What, age). With W and Z both driven by the hidden confounder and
    satisfying the exclusion restrictions above, the coefficient on A in stage 2 identifies
    the treatment effect on the linear-probability scale.
    """
    X1 = np.column_stack([d["a"], d["z"], d["age"]])
    w_hat = np.column_stack([np.ones(len(X1)), X1]) @ ols(X1, d["w"])
    X2 = np.column_stack([d["a"], w_hat, d["age"]])
    return float(ols(X2, d["y"])[1])


# --------------------------------------------------------------------------- policies
def policy_treat_all(d):
    return np.ones(len(d["a"]))


def policy_treat_none(d):
    return np.zeros(len(d["a"]))


def policy_cdss(d, threshold=0.5):
    """A CDSS that treats when the recorded covariates put predicted risk above a threshold.

    It sees age, z and w -- everything except the latent severity -- exactly as a deployed
    system would see recorded proxies rather than the underlying state.
    """
    X = np.column_stack([d["age"], d["z"], d["w"]])
    beta = ols(X, d["y"])
    risk = np.column_stack([np.ones(len(X)), X]) @ beta
    return (risk > np.quantile(risk, 1 - threshold)).astype(float)


POLICIES = {"treat_all": policy_treat_all, "treat_none": policy_treat_none,
            "cdss_top50": lambda d: policy_cdss(d, 0.5),
            "cdss_top25": lambda d: policy_cdss(d, 0.25)}


def true_policy_value(cfg: dict, d: dict, pi: np.ndarray) -> float:
    """V(pi) under do(A = pi(x)), using the outcome mechanism the data came from."""
    base = (cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"]
            + cfg["out_intercept"])
    return float(np.mean(sigmoid(base + cfg["treat_to_out"] * pi)))


def ope_estimates(d: dict, pi: np.ndarray) -> dict:
    """IPW, self-normalised IPW and doubly-robust value estimates.

    The propensity and outcome models use the recorded covariates only (age, z, w): the
    estimators are given exactly what a real audit would have.
    """
    X = np.column_stack([d["age"], d["z"], d["w"]])
    Xd = np.column_stack([np.ones(len(X)), X])
    e = np.clip(Xd @ ols(X, d["a"]), 0.02, 0.98)           # propensity, linear probability
    Xa1 = np.column_stack([np.ones(len(X)), np.ones(len(X)), X])
    Xa0 = np.column_stack([np.ones(len(X)), np.zeros(len(X)), X])
    beta_y = ols(np.column_stack([d["a"], X]), d["y"])
    mu1, mu0 = Xa1 @ beta_y, Xa0 @ beta_y
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
    return {"ipw": ipw, "snipw": snipw, "dr": dr, "dr_se": se_dr,
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
        est = {k: [] for k in ("naive", "backdoor_observed", "proximal")}
        pol_acc = {p: {k: [] for k in ("ipw", "snipw", "dr", "outcome_only", "truth",
                                       "dr_cover", "ess_frac", "agreement",
                                       "max_weight_share")} for p in POLICIES}

        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + 1000 * r + hash(domain) % 997)
            d = simulate(cfg, args.n, rng)
            est["naive"].append(naive_ate(d))
            est["backdoor_observed"].append(backdoor_observed_ate(d))
            est["proximal"].append(proximal_ate(d))

            for pname, pfun in POLICIES.items():
                pi = pfun(d)
                v_true = true_policy_value(cfg, d, pi)
                o = ope_estimates(d, pi)
                for k in ("ipw", "snipw", "dr", "outcome_only"):
                    pol_acc[pname][k].append(o[k])
                pol_acc[pname]["truth"].append(v_true)
                for k in ("ess_frac", "agreement", "max_weight_share"):
                    pol_acc[pname][k].append(o[k])
                lo, hi = o["dr"] - 1.96 * o["dr_se"], o["dr"] + 1.96 * o["dr_se"]
                pol_acc[pname]["dr_cover"].append(float(lo <= v_true <= hi))

        for name, vals in est.items():
            v = np.array(vals)
            prox_rows.append({
                "domain": domain, "estimator": name, "true_ate": round(truth, 4),
                "mean_estimate": round(float(v.mean()), 4),
                "bias": round(float(v.mean() - truth), 4),
                "abs_bias": round(abs(float(v.mean() - truth)), 4),
                "rmse": round(float(np.sqrt(np.mean((v - truth) ** 2))), 4),
                "sd": round(float(v.std(ddof=1)), 4), "reps": args.reps, "n": args.n,
            })
            print(f"  {domain:8} {name:18} true {truth:+.4f}  est {v.mean():+.4f}  "
                  f"bias {v.mean() - truth:+.4f}")

        for pname, acc in pol_acc.items():
            truth_v = np.array(acc["truth"])
            row = {"domain": domain, "policy": pname,
                   "true_value": round(float(truth_v.mean()), 4), "reps": args.reps}
            for k in ("ipw", "snipw", "dr", "outcome_only"):
                e = np.array(acc[k])
                row[f"{k}_mean"] = round(float(e.mean()), 4)
                row[f"{k}_bias"] = round(float(np.mean(e - truth_v)), 4)
                row[f"{k}_rmse"] = round(float(np.sqrt(np.mean((e - truth_v) ** 2))), 4)
            pol_rows.append(row)
            cov_rows.append({"domain": domain, "policy": pname,
                             "dr_ci95_coverage": round(float(np.mean(acc["dr_cover"])), 4),
                             "nominal": 0.95,
                             "mean_ess_fraction": round(float(np.mean(acc["ess_frac"])), 4),
                             "mean_policy_agreement": round(float(np.mean(acc["agreement"])), 4),
                             "mean_max_weight_share": round(float(np.mean(acc["max_weight_share"])), 5),
                             "reps": args.reps})

    write(prox_rows, "proximal_estimates.csv")
    write(pol_rows, "policy_value_estimates.csv")
    write(cov_rows, "policy_value_coverage.csv")


if __name__ == "__main__":
    main()
