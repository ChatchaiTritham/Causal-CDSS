"""Causal-CDSS reproducibility driver (seed = 42).

Recomputes every quantity reported for the manuscript from a seeded synthetic
ICU cohort, using the vendored ``basics_cdss.causal`` module. Nothing is
hardcoded: each number below is produced at run time from structural causal
models (SCMs) and standard estimators. Whatever this driver prints is the
ground truth for the figures and tables.

Pipeline (per clinical domain: sepsis, ARDS, ACS):
  1. Build a structural causal model whose treatment assignment is confounded
     by disease severity (sicker patients are more likely to be treated), so
     the observational/naive treatment-outcome association is biased.
  2. Sample observational data P(X) and read off the naive (unadjusted)
     association between treatment and mortality.
  3. Obtain the TRUE causal ATE by do-calculus intervention on the SCM.
  4. Recover the causal effect from observational data with three estimators
     (naive difference, regression/backdoor adjustment, doubly-robust) and
     score five causal-evaluation metrics defined in the manuscript.
  5. Estimate conditional average treatment effects (CATE) across age/severity
     subgroups to quantify treatment-effect heterogeneity.
  6. Quantify confounding bias and run an E-value sensitivity analysis.

Outputs:
  results/causal_results.json      full machine-readable results
  results/ate_by_domain.csv        naive vs true vs adjusted ATE per domain
  results/cate_subgroups.csv       CATE across age/severity subgroups
  results/causal_metrics.csv       five causal-evaluation metrics per estimator
  results/confounding_sensitivity.csv  confounding bias + E-value per domain

Run:
    python run_all.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from basics_cdss.causal import (
    CausalGraph,
    StructuralCausalModel,
    backdoor_adjustment,
    compute_ate,
    compute_cate,
    confounding_bias_estimate,
    identify_confounders,
)
from basics_cdss.causal.confounding import sensitivity_analysis_evalue

SEED = 42
N_OBS = 8000          # observational cohort size per domain
N_INT = 8000          # interventional samples per arm for do-calculus
OUT = Path(__file__).parent / "results"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# Domain SCMs.
#
# Each domain shares the same canonical confounded structure required to
# demonstrate the manuscript's point (associational metrics miss causal
# failures): a latent severity drives BOTH treatment assignment and the
# mortality outcome, so the naive association is confounded, while the true
# causal effect is identifiable by backdoor adjustment on {age, severity}.
# Coefficients differ per domain to give distinct effect magnitudes, but the
# numbers reported are whatever the seeded SCM produces -- they are not set to
# match any manuscript constant.
# --------------------------------------------------------------------------- #
DOMAINS = {
    "sepsis": {
        "label": "Sepsis - Early Antibiotics",
        "treatment": "treatment",
        "outcome": "outcome",
        "age_to_sev": 0.60,
        "sev_to_treat": 1.20,
        "treat_intercept": -0.30,
        "sev_to_out": 1.10,
        "age_to_out": 0.50,
        "treat_to_out": -1.00,     # protective on the logit scale
        "out_intercept": -0.30,
    },
    "ards": {
        "label": "ARDS - Low Tidal Volume",
        "treatment": "treatment",
        "outcome": "outcome",
        "age_to_sev": 0.45,
        "sev_to_treat": 0.90,
        "treat_intercept": -0.20,
        "sev_to_out": 0.95,
        "age_to_out": 0.40,
        "treat_to_out": -0.65,
        "out_intercept": -0.20,
    },
    "acs": {
        "label": "ACS - Early Reperfusion",
        "treatment": "treatment",
        "outcome": "outcome",
        "age_to_sev": 0.70,
        "sev_to_treat": 1.40,
        "treat_intercept": -0.40,
        "sev_to_out": 1.25,
        "age_to_out": 0.60,
        "treat_to_out": -1.30,
        "out_intercept": -0.40,
    },
}


def build_scm(cfg, seed):
    """Construct a confounded structural causal model for one domain.

    Graph:  age -> severity ;  severity -> treatment ;
            {severity, age, treatment} -> outcome ;  age -> severity -> outcome.
    """
    g = CausalGraph()
    for node in ("age", "severity", "treatment", "outcome"):
        g.add_node(node)
    g.add_edge("age", "severity")
    g.add_edge("severity", "treatment")
    g.add_edge("severity", "outcome")
    g.add_edge("age", "outcome")
    g.add_edge("treatment", "outcome")

    scm = StructuralCausalModel(g, seed=seed, default_mechanisms=False)

    scm.add_mechanism(
        "age", [],
        function=lambda p, n: n,
        noise_distribution=lambda r: r.normal(0, 1),
    )
    a2s = cfg["age_to_sev"]
    scm.add_mechanism(
        "severity", ["age"],
        function=lambda p, n, a2s=a2s: a2s * p["age"] + n,
        noise_distribution=lambda r: r.normal(0, 1),
    )
    s2t, ti = cfg["sev_to_treat"], cfg["treat_intercept"]
    # Treatment is assigned with probability increasing in severity -> the
    # source of confounding. Bernoulli draw via a uniform noise threshold.
    scm.add_mechanism(
        "treatment", ["severity"],
        function=lambda p, n, s2t=s2t, ti=ti: 1.0
        if _sigmoid(s2t * p["severity"] + ti) > n else 0.0,
        noise_distribution=lambda r: r.uniform(0, 1),
    )
    s2o, a2o, t2o, oi = (
        cfg["sev_to_out"], cfg["age_to_out"], cfg["treat_to_out"], cfg["out_intercept"]
    )
    # Binary mortality outcome; treatment lowers the mortality log-odds.
    scm.add_mechanism(
        "outcome", ["severity", "age", "treatment"],
        function=lambda p, n, s2o=s2o, a2o=a2o, t2o=t2o, oi=oi: 1.0
        if _sigmoid(s2o * p["severity"] + a2o * p["age"] + t2o * p["treatment"] + oi) > n
        else 0.0,
        noise_distribution=lambda r: r.uniform(0, 1),
    )
    return g, scm


# --------------------------------------------------------------------------- #
# Estimators of the ATE from observational data.
# --------------------------------------------------------------------------- #
def naive_ate(data, treatment, outcome):
    """Unadjusted difference in mean outcome between treated and untreated."""
    treated = data[data[treatment] == 1][outcome].mean()
    control = data[data[treatment] == 0][outcome].mean()
    return float(treated - control)


def doubly_robust_ate(data, treatment, outcome, confounders):
    """Augmented inverse-probability-weighted (doubly-robust) ATE estimate.

    Combines a propensity model P(T|Z) with outcome regressions E[Y|T,Z];
    consistent if either model is correct.
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression

    z = data[confounders].values
    t = data[treatment].values.astype(int)
    y = data[outcome].values.astype(float)

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(z, t)
    ps = np.clip(ps_model.predict_proba(z)[:, 1], 1e-3, 1 - 1e-3)

    feat = np.column_stack([data[treatment].values, z])
    out_model = LinearRegression().fit(feat, y)
    mu1 = out_model.predict(np.column_stack([np.ones(len(y)), z]))
    mu0 = out_model.predict(np.column_stack([np.zeros(len(y)), z]))

    dr1 = mu1 + t * (y - mu1) / ps
    dr0 = mu0 + (1 - t) * (y - mu0) / (1 - ps)
    return float(np.mean(dr1 - dr0))


# --------------------------------------------------------------------------- #
# Per-domain analysis.
# --------------------------------------------------------------------------- #
def analyse_domain(name, cfg):
    g, scm = build_scm(cfg, seed=SEED)
    treatment, outcome = cfg["treatment"], cfg["outcome"]

    # Observational cohort (fresh SCM so the seed governs the draw).
    obs = scm.sample(n=N_OBS)

    # Ground-truth causal ATE by do-calculus intervention (independent SCM
    # instance to keep the intervention sampling deterministic and separate).
    _, scm_int = build_scm(cfg, seed=SEED + 1)
    ate_truth = compute_ate(
        scm_int, treatment, outcome, treatment_values=[0, 1], n_samples=N_INT
    )
    true_ate = float(ate_truth["ate"])

    # Confounder identification from the graph (backdoor criterion).
    conf = identify_confounders(g, treatment, outcome)
    adjustment_set = conf["confounders"] or ["severity", "age"]
    # Use the full canonical adjustment set for estimation when available.
    est_confounders = [c for c in ("severity", "age") if c in obs.columns]

    # Three observational estimators.
    est = {
        "naive": naive_ate(obs, treatment, outcome),
        "backdoor_regression": float(
            backdoor_adjustment(obs, treatment, outcome, est_confounders)["ate"]
        ),
        "doubly_robust": doubly_robust_ate(obs, treatment, outcome, est_confounders),
    }

    # Interventional reference data for the confounding-bias metric.
    int1 = scm_int.do_intervention({treatment: 1}, n=N_INT)
    int0 = scm_int.do_intervention({treatment: 0}, n=N_INT)
    int1[treatment] = 1
    int0[treatment] = 0
    interventional = pd.concat([int0, int1], ignore_index=True)
    bias = confounding_bias_estimate(obs, interventional, treatment, outcome)

    # --- Five causal-evaluation metrics, per estimator ---------------------- #
    # 1. CEE  : Causal Effect Estimation Error = |estimate - true ATE|
    # 2. ATE Alignment Score = 1 - |estimate - true| / |true|  (clipped to >=0)
    # 3. Confounding Bias Magnitude = |observational assoc - true causal effect|
    # 4. Counterfactual Prediction Accuracy = sign + magnitude agreement of the
    #    estimator's direction with the true causal direction, scored as the
    #    fraction of subgroup CATE signs matched (computed below).
    # 5. Causal Consistency Index = correlation of estimated vs true CATE across
    #    subgroups (computed below, shared across estimators of the same model).
    metrics_per_estimator = {}
    for est_name, est_val in est.items():
        cee = abs(est_val - true_ate)
        align = max(0.0, 1.0 - cee / abs(true_ate)) if true_ate != 0 else float("nan")
        metrics_per_estimator[est_name] = {
            "ate_estimate": est_val,
            "cee": cee,
            "ate_alignment_score": align,
        }

    confounding_bias_magnitude = abs(bias["bias"])

    # --- CATE heterogeneity across subgroups -------------------------------- #
    # True CATE by intervening within age x severity strata on the SCM.
    age_bins = [(-np.inf, -0.43, "age_low"), (-0.43, 0.43, "age_mid"), (0.43, np.inf, "age_high")]
    sev_bins = [(-np.inf, -0.43, "sev_low"), (-0.43, 0.43, "sev_mid"), (0.43, np.inf, "sev_high")]

    cate_rows = []
    true_cates, est_cates = [], []
    for a_lo, a_hi, a_lab in age_bins:
        for s_lo, s_hi, s_lab in sev_bins:
            # True CATE: difference in mean outcome under do(T=1) vs do(T=0),
            # restricted to the stratum, estimated by stratified intervention.
            mask1 = (int1["age"] > a_lo) & (int1["age"] <= a_hi) & \
                    (int1["severity"] > s_lo) & (int1["severity"] <= s_hi)
            mask0 = (int0["age"] > a_lo) & (int0["age"] <= a_hi) & \
                    (int0["severity"] > s_lo) & (int0["severity"] <= s_hi)
            if mask1.sum() < 20 or mask0.sum() < 20:
                continue
            true_cate = float(int1.loc[mask1, outcome].mean() - int0.loc[mask0, outcome].mean())

            # Estimated CATE from observational data via backdoor regression
            # restricted to the same stratum (DR-adjusted within stratum).
            sub = obs[(obs["age"] > a_lo) & (obs["age"] <= a_hi) &
                      (obs["severity"] > s_lo) & (obs["severity"] <= s_hi)]
            if (sub[treatment] == 1).sum() < 10 or (sub[treatment] == 0).sum() < 10:
                continue
            est_cate = float(
                backdoor_adjustment(sub, treatment, outcome, ["severity", "age"])["ate"]
            )
            cate_rows.append({
                "domain": name, "age_group": a_lab, "severity_group": s_lab,
                "true_cate": true_cate, "estimated_cate": est_cate,
                "n_obs": int(len(sub)),
            })
            true_cates.append(true_cate)
            est_cates.append(est_cate)

    # 5. Causal Consistency Index: Pearson r between true and estimated CATE.
    if len(true_cates) >= 3 and np.std(true_cates) > 0 and np.std(est_cates) > 0:
        cci = float(np.corrcoef(true_cates, est_cates)[0, 1])
    else:
        cci = float("nan")

    # 4. Counterfactual Prediction Accuracy: fraction of subgroup CATEs whose
    # estimated sign matches the true sign (direction of recommendation).
    if true_cates:
        sign_match = np.mean(
            [np.sign(t) == np.sign(e) for t, e in zip(true_cates, est_cates)]
        )
        cpa = float(sign_match)
    else:
        cpa = float("nan")

    cate_range = (min(true_cates), max(true_cates)) if true_cates else (None, None)

    # E-value sensitivity for the adjusted estimate (on a risk-ratio scale).
    p_treat = obs[obs[treatment] == 1][outcome].mean()
    p_ctrl = obs[obs[treatment] == 0][outcome].mean()
    rr = (p_treat / p_ctrl) if p_ctrl > 0 else float("nan")
    rr_for_evalue = rr if (rr and rr >= 1) else (1.0 / rr if rr and rr > 0 else float("nan"))
    evalue = sensitivity_analysis_evalue(rr_for_evalue)["e_value"] if rr_for_evalue == rr_for_evalue else float("nan")

    return {
        "label": cfg["label"],
        "n_obs": int(len(obs)),
        "treatment_prevalence": float(obs[treatment].mean()),
        "mortality_rate": float(obs[outcome].mean()),
        "true_ate": true_ate,
        "adjustment_set_identified": conf["confounders"],
        "estimators": est,
        "causal_metrics": {
            "per_estimator": metrics_per_estimator,
            "confounding_bias_magnitude": confounding_bias_magnitude,
            "causal_consistency_index": cci,
            "counterfactual_prediction_accuracy": cpa,
        },
        "confounding": {
            "observational_effect": float(bias["observational_effect"]),
            "causal_effect": float(bias["causal_effect"]),
            "bias": float(bias["bias"]),
            "relative_bias_pct": float(bias["relative_bias"]),
        },
        "cate_subgroups": cate_rows,
        "cate_true_range": cate_range,
        "sensitivity": {
            "risk_ratio": float(rr) if rr == rr else None,
            "e_value": float(evalue) if evalue == evalue else None,
        },
    }


def main():
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[*] Causal-CDSS reproducibility driver  (seed={SEED})")

    results = {"seed": SEED, "n_obs_per_domain": N_OBS, "n_interventional_per_arm": N_INT,
               "domains": {}}
    for name, cfg in DOMAINS.items():
        print(f"[*] domain: {name}")
        res = analyse_domain(name, cfg)
        results["domains"][name] = res
        cm = res["causal_metrics"]
        print(f"    n={res['n_obs']} treat_prev={res['treatment_prevalence']:.3f} "
              f"mortality={res['mortality_rate']:.3f}")
        print(f"    true ATE (do-calculus)     = {res['true_ate']:+.4f}")
        print(f"    naive (observational) ATE  = {res['estimators']['naive']:+.4f}")
        print(f"    backdoor-adjusted ATE      = {res['estimators']['backdoor_regression']:+.4f}")
        print(f"    doubly-robust ATE          = {res['estimators']['doubly_robust']:+.4f}")
        print(f"    confounding bias magnitude = {cm['confounding_bias_magnitude']:.4f}")
        print(f"    causal consistency index   = {cm['causal_consistency_index']:.4f}")
        print(f"    counterfactual sign acc.   = {cm['counterfactual_prediction_accuracy']:.4f}")

    # --- write JSON --------------------------------------------------------- #
    (OUT / "causal_results.json").write_text(
        json.dumps(results, indent=2, default=lambda o: float(o))
    )
    print("[OK] results/causal_results.json")

    # --- CSV: ATE by domain ------------------------------------------------- #
    with (OUT / "ate_by_domain.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "label", "true_ate", "naive_ate",
                    "backdoor_ate", "doubly_robust_ate",
                    "confounding_bias", "relative_bias_pct"])
        for name, r in results["domains"].items():
            w.writerow([name, r["label"], f"{r['true_ate']:.4f}",
                        f"{r['estimators']['naive']:.4f}",
                        f"{r['estimators']['backdoor_regression']:.4f}",
                        f"{r['estimators']['doubly_robust']:.4f}",
                        f"{r['confounding']['bias']:.4f}",
                        f"{r['confounding']['relative_bias_pct']:.2f}"])
    print("[OK] results/ate_by_domain.csv")

    # --- CSV: causal metrics per estimator ---------------------------------- #
    with (OUT / "causal_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "estimator", "ate_estimate", "cee",
                    "ate_alignment_score", "confounding_bias_magnitude",
                    "causal_consistency_index", "counterfactual_prediction_accuracy"])
        for name, r in results["domains"].items():
            cm = r["causal_metrics"]
            for est_name, m in cm["per_estimator"].items():
                w.writerow([name, est_name, f"{m['ate_estimate']:.4f}",
                            f"{m['cee']:.4f}", f"{m['ate_alignment_score']:.4f}",
                            f"{cm['confounding_bias_magnitude']:.4f}",
                            f"{cm['causal_consistency_index']:.4f}",
                            f"{cm['counterfactual_prediction_accuracy']:.4f}"])
    print("[OK] results/causal_metrics.csv")

    # --- CSV: CATE subgroups ------------------------------------------------ #
    with (OUT / "cate_subgroups.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "age_group", "severity_group",
                    "true_cate", "estimated_cate", "n_obs"])
        for name, r in results["domains"].items():
            for row in r["cate_subgroups"]:
                w.writerow([row["domain"], row["age_group"], row["severity_group"],
                            f"{row['true_cate']:.4f}", f"{row['estimated_cate']:.4f}",
                            row["n_obs"]])
    print("[OK] results/cate_subgroups.csv")

    # --- CSV: confounding + sensitivity ------------------------------------- #
    with (OUT / "confounding_sensitivity.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "observational_effect", "causal_effect",
                    "confounding_bias", "risk_ratio", "e_value"])
        for name, r in results["domains"].items():
            w.writerow([name, f"{r['confounding']['observational_effect']:.4f}",
                        f"{r['confounding']['causal_effect']:.4f}",
                        f"{r['confounding']['bias']:.4f}",
                        ("" if r["sensitivity"]["risk_ratio"] is None
                         else f"{r['sensitivity']['risk_ratio']:.4f}"),
                        ("" if r["sensitivity"]["e_value"] is None
                         else f"{r['sensitivity']['e_value']:.4f}")])
    print("[OK] results/confounding_sensitivity.csv")
    print("[DONE]")


if __name__ == "__main__":
    main()
