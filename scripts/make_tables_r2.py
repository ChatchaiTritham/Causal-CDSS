"""LaTeX table bodies for the second-round supplementary sections, from results/r2_*.csv.

Also writes tables/numbers_r2.tex (macros) and checks the numbers quoted in the manuscript
front matter against the CSVs (see tests/test_manuscript_numbers.py).
Usage: python scripts/make_tables_r2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES, OUT = REPO / "results", REPO / "tables"
DOM = {"sepsis": "Sepsis", "ards": "ARDS", "cardiac": "ACS"}
POL = {"age_rule": "Age rule", "treat_all": "Treat all", "treat_none": "Treat none",
       "cdss_top50": "Risk, top 50\\%", "cdss_top25": "Risk, top 25\\%", "knowledge": "Knowledge-driven",
       "kd_ignores_harm": "Knowledge-driven (ignores harm)", "kd_harm_aware": "Knowledge-driven (harm-aware)"}


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == "":
        return "---"
    try:
        v = float(x)
    except ValueError:
        return str(x)
    return f"${v:+.{d}f}$" if sign else f"{v:.{d}f}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("wrote tables/" + name)


def t_nlh():
    d = pd.read_csv(RES / "r2_bridge_nonlinear.csv")
    L = []
    for _, r in d[~d.cell.str.startswith("F by")].iterrows():
        L.append(f"{DOM[r.domain]} & {r.cell} & {r.estimator}, {r.se_type} & {f(r.bias, 4, True)} & "
                 f"{f(r.emp_sd, 4)} & {f(r.median_se, 4)} & {f(r.coverage, 3)} & "
                 f"{f(r.coverage_worst_case, 3)} & {int(r.nonconverged)} \\\\")
    write("r2_nlh.tex", L)
    F = d[d.cell.str.startswith("F by")]
    write("r2_armf.tex", [f"{DOM[r.domain]} & {r.cell[6:].strip('()')} & {r.bias:.0f} & {r.emp_sd:.0f} & "
                          f"{r.median_se:.0f} \\\\" for _, r in F.iterrows()])


def t_pv():
    d = pd.read_csv(RES / "r2_policy_value_main.csv")
    L = []
    for _, r in d.iterrows():
        L.append(f"{DOM[r.domain]} & {POL[r.policy]} & {f(r.inconclusive_rate, 3)} & {f(r.prox_cov_all, 3)} & "
                 f"{f(r.prox_cov_conclusive, 3)} & {f(r.dr_insample_cov, 3)} & {f(r.cf_rec_cov_clip001, 3)} & "
                 f"{f(r.cf_rec_cov_clip002, 3)} & {f(r.cf_rec_cov_clip005, 3)} & {f(r.cf_oracle_cov_clip001, 3)} & "
                 f"{f(r.withhold_share_indicated, 2)} & {r.q_max_median:.0f} / {r.q_p99_median:.1f} \\\\")
    write("r2_pv.tex", L)
    L = [f"{DOM[r.domain]} & {POL[r.policy]} & {f(r.bias_ipw, 4, True)} & {f(r.bias_snipw, 4, True)} & "
         f"{f(r.bias_dm, 4, True)} & {f(r.bias_dr, 4, True)} & {f(r.rmse_ipw, 4)} & {f(r.rmse_snipw, 4)} & "
         f"{f(r.rmse_dm, 4)} & {f(r.rmse_dr, 4)} \\\\" for _, r in d.iterrows()]
    write("r2_pv_est.tex", L)
    v = pd.read_csv(RES / "r2_policy_value_violation.csv")
    v = v[v.policy.isin(["age_rule", "cdss_top50", "knowledge"])]
    write("r2_pv_viol.tex", [f"{DOM[r.domain]} & {r.gamma:.1f} & {POL[r.policy]} & {f(r.inconclusive_rate, 3)} & "
                             f"{f(r.prox_cov_all, 3)} & {f(r.prox_cov_conclusive, 3)} & "
                             f"{f(r.prox_bias_conclusive, 4, True)} \\\\" for _, r in v.iterrows()])
    b = pd.read_csv(RES / "r2_policy_value_bootstrap.csv")
    write("r2_pvboot.tex", [f"{DOM[r.domain]} & {POL[r.policy]} & {f(r.plugin_cov, 3)} & {f(r.boot_wald_cov, 3)} & "
                            f"{f(r.boot_pct_cov, 3)} & {f(r.se_ratio_boot_over_plugin, 2)} \\\\" for _, r in b.iterrows()])
    g = pd.read_csv(RES / "r2_subgroup_diagnostics.csv")
    write("r2_sg.tex", [f"{DOM[r.domain]} & {r.third} & {r.n_median} & {r.nonconverged} & {r.F_median:.0f} & "
                        f"{r.kd_treated_min} / {r.rr_treated_min} & {f(r.recall_diff_mean, 3, True)} "
                        f"[{f(r.recall_diff_lo, 3, True)}, {f(r.recall_diff_hi, 3, True)}] & "
                        f"{f(r.gain_diff_mean, 3, True)} [{f(r.gain_diff_lo, 3, True)}, {f(r.gain_diff_hi, 3, True)}] & "
                        f"{f(r.kd_contrast_cov, 3)} / {f(r.rr_contrast_cov, 3)} \\\\" for _, r in g.iterrows()])


def t_calib():
    d = pd.read_csv(RES / "r2_calibrated.csv")
    d = d[d.gamma == 0.0]
    L = []
    for _, r in d.iterrows():
        reg = "trial" if "AUROC" not in r.regime else "trial + AUROC 0.70"
        L.append(f"{DOM[r.domain]} & {reg} & {r.baseline_mortality:.3f} / {r.target_arr:.3f} & {r.loading_scale:.2f} & "
                 f"{r.F_median:.0f} & {r.estimator.upper() if r.estimator != 'proxy' else 'Proxy adj.'} & "
                 f"{f(r.true_ate, 4, True)} & {f(r.bias, 4, True)} & {r.bias_pct_of_effect:.0f} & "
                 f"{f(r.coverage, 3)} & {r.nonconverged} \\\\")
    write("r2_calib.tex", L)
    c = pd.read_csv(RES / "r2_calibrated_crossing.csv")
    write("r2_calib_cross.tex", [f"{DOM[r.domain]} & {r.estimator.upper()} & {r.gamma_star} \\\\" for _, r in c.iterrows()])


def t_harm():
    d = pd.read_csv(RES / "r2_harm_decision_curve.csv")
    d = d[d.cost_threshold.isin([0.0, 0.05, 0.10, 0.15, 0.20])]
    L = []
    for _, r in d.iterrows():
        if r.policy == "treat_none":
            continue
        L.append(f"{DOM[r.domain]} & {POL[r.policy]} & {r.cost_threshold:.2f} & {f(r.treated, 2)} & "
                 f"{f(r.harm_recall, 2)} & {f(r.net_benefit_true, 4, True)} & {f(r.net_benefit_proximal_bias, 4, True)} & "
                 f"{f(r.net_benefit_proximal_cov, 2)} \\\\")
    write("r2_harm.tex", L)


def t_lhs():
    d = pd.read_csv(RES / "r2_lhs_design.csv")
    w = 1 - d.share_inconclusive
    acc = float((d.decision_correct.fillna(0) * w).sum() / w.sum())
    add = np.where(d.gamma / 0.28 + d.delta / 0.23 < 0.5, "proximal", "proxy")
    rows = [("Generators (held out, none used to set a threshold)", f"{len(d)}"),
            ("DR proximal has the smaller RMSE", f"{(d.better == 'proximal').sum()} of {len(d)}"),
            ("Median RMSE ratio, DR / proxy adjustment", f"{(d.rmse_dr / d.rmse_proxy).median():.2f}"),
            ("Algorithm~1 inconclusive (mean share of replications)", f"{d.share_inconclusive.mean():.2f}"),
            ("Algorithm~1 decision correct (conclusive replications)", f"{acc:.2f}"),
            ("Additive rule $\\bar\\gamma/\\gamma^\\ast + \\bar\\delta/\\delta^\\ast < 1/2$ correct", f"{(add == d.better).mean():.2f}"),
            ("Always choosing DR proximal correct", f"{(d.better == 'proximal').mean():.2f}"),
            ("Median coverage: DR proximal / proxy adjustment", f"{d.cov_dr.median():.2f} / {d.cov_proxy.median():.2f}")]
    write("r2_lhs.tex", [f"{a} & {b} \\\\" for a, b in rows])
    m = pd.read_csv(RES / "r2_lhs_metamodel.csv")
    write("r2_lhs_meta.tex", [f"\\texttt{{{r.parameter.replace('_', chr(92) + '_')}}} & {r.std_coef:+.3f} \\\\"
                              for _, r in m.iterrows()])
    a = pd.read_csv(RES / "r2_stress_arms.csv")
    write("r2_arms.tex", [f"{DOM[r.domain]} & {r.arm} & {r.estimator.upper() if r.estimator != 'proxy' else 'Proxy adj.'} & "
                          f"{f(r.bias, 4, True)} & {f(r.rmse, 4)} & {f(r.coverage, 3)} & {r.nonconverged} \\\\"
                          for _, r in a.iterrows()])


def t_falsify():
    d = pd.read_csv(RES / "r2_falsification_size_power.csv")
    L = []
    for _, r in d.iterrows():
        imp = r.implication.replace("_||_", "$\\perp\\!\\!\\!\\perp$").replace("severity", "U").replace(",a,", ",T,").replace("| a,", "| T,").replace("fisher_z", "Fisher-$z$").replace("(gcm)", "(GCM)")
        L.append(f"{DOM[r.domain]} & {imp} & {r.quantity} & {f(r.fisher_z_reject_005, 3)} & {f(r.fisher_z_reject_bonf, 3)} & "
                 f"{f(r.gcm_reject_005, 3)} & {f(r.gcm_reject_bonf, 3)} \\\\")
    write("r2_falsify.tex", L)


def t_dml():
    d = pd.read_csv(RES / "r2_dml_repeated.csv")
    L = []
    for _, r in d.iterrows():
        L.append(f"{DOM[r.domain]} & {r.learner.replace('_', ' ')} & {r.adjustment_set.split(' ')[0]} & "
                 f"{r.score.replace('_', ' ')} & {r.target.replace('overlap-weighted PLR estimand', 'PLR estimand')} & "
                 f"{f(r.bias, 4, True)} & {f(r.coverage, 3)} & {f(r.prop_auc, 3)} & {f(r.prop_brier, 3)} & "
                 f"{f(r.outcome_logloss, 3)} & {f(r.share_clipped, 3)} \\\\")
    write("r2_dml.tex", L)
    o = pd.read_csv(RES / "r2_dml_ovb_bounds.csv")
    write("r2_ovb.tex", [f"{DOM[r.domain]} & {r.benchmark.upper() if r.benchmark == 'z' else r.benchmark} & "
                         f"{f(r.true_ate, 3, True)} & {f(r.theta_mean, 3, True)} & {f(r.cf_y, 3)} & {f(r.cf_d, 4)} & "
                         f"[{f(r.bound_lo, 3, True)}, {f(r.bound_hi, 3, True)}] & {f(r.share_bounds_contain_truth, 2)} & "
                         f"{f(r.share_ci_bounds_contain_truth, 2)} & {f(r.robustness_value, 3)} \\\\" for _, r in o.iterrows()])
    c = pd.read_csv(RES / "r2_reference_crosscheck.csv")
    write("r2_check.tex", [f"{DOM[r.domain]} & {f(r.doubleml_irm, 4, True)} & {f(r.own_aipw_same_folds, 4, True)} & "
                           f"{f(r.doubleml_se, 4)} / {f(r.own_aipw_se, 4)} & {f(r.linearmodels_iv2sls, 4, True)} & "
                           f"{f(r.own_proximal_2sls, 4, True)} & {f(r.linearmodels_se, 4)} / {f(r.own_2sls_se, 4)} \\\\"
                           for _, r in c.iterrows()])


SYM = {"gamma": "$"+chr(92)+"gamma^"+chr(92)+"ast$", "delta": "$"+chr(92)+"delta^"+chr(92)+"ast$"}
SYM2 = {"gamma": "$"+chr(92)+"gamma$", "delta": "$"+chr(92)+"delta$"}


def ci(r):
    return "---" if pd.isna(r.ci_lo) or r.ci_lo == "" else "[" + f(r.ci_lo) + ", " + f(r.ci_hi) + "]"


def t_cross():
    d = pd.read_csv(RES / "r2_crossing_dense.csv")
    write("r2_cross.tex", [f"{DOM[r.domain]} & {SYM[r.restriction]} & "
                           f"{r.estimator.upper()} & {r.comparator} & {r.crossing} & "
                           f"{ci(r)} & "
                           f"{f(r.comparator_bias_at_0_10, 4, True)} \\\\" for _, r in d.iterrows()])
    a = pd.read_csv(RES / "r2_linear_analytic_crossing.csv")
    write("r2_analytic.tex", [f"{DOM[r.domain]} & {SYM[r.restriction]} & "
                              f"{f(r.linear_analogue_crossing)} & {f(r.common_scale)} \\\\" for _, r in a.iterrows()])
    o = pd.read_csv(RES / "r2_overid_hansen_j.csv")
    write("r2_overid.tex", [f"{DOM[r.domain]} & {SYM2[r.violation]} & {r.strength:.1f} & "
                            f"{f(r.hansen_J_reject_005, 3)} & {f(r.mean_gmm_bT, 4, True)} \\\\" for _, r in o.iterrows()])


def t_misc():
    b = pd.read_csv(RES / "r2_logistic_bridge.csv")
    write("r2_binbr.tex", [f"{DOM[r.domain]} & {r.estimator} & {f(r.bias, 4, True)} & {f(r.emp_sd, 4)} & "
                           f"{f(r.rmse, 4)} & {f(r.coverage, 3)} & {r.nonconverged} \\\\" for _, r in b.iterrows()])
    s = pd.read_csv(RES / "r2_site_shift.csv")
    write("r2_shift.tex", [f"{DOM[r.domain]} & {r.iloc[1]} & {f(r.true_ate_site_B, 4, True)} & {f(r.ate_bias, 4, True)} & "
                           f"{f(r.ate_coverage, 3)} & {f(r.value_treat_all_bias, 4, True)} & {f(r.value_kd_rule_bias, 4, True)} \\\\"
                           for _, r in s.iterrows()])
    g = pd.read_csv(RES / "r2_differential_proxy_error.csv")
    write("r2_diff.tex", [f"{r.cell} & {f(r.bias, 4, True)} & {f(r.mcse, 4)} & {f(r.coverage, 3)} \\\\" for _, r in g.iterrows()])
    p = pd.read_csv(RES / "r2_proposition_check.csv")
    write("r2_prop.tex", [f"{DOM[r.domain]} & {r.sd_eZ:.1f} / {r.sd_eW:.1f} & {f(r.predicted_bias, 4, True)} & "
                          f"{f(r.population_bias, 4, True)} & {f(r.simulated_bias, 4, True)} & {f(r.mcse, 4)} \\\\"
                          for _, r in p.iterrows()])
    c = pd.read_csv(RES / "r2_nonconvergence_per_cell.csv")
    write("r2_conv.tex", [f"{DOM[r.domain]} & {r.sweep} & {str(r.cell).replace(chr(95), chr(32))} & {r.n_nonconverged} & {f(r.dr_bias_all, 4, True)} & "
                          f"{f(r.dr_bias_converged, 4, True)} & {f(r.dr_cov_all, 3)} & {f(r.dr_cov_converged, 3)} \\\\"
                          for _, r in c.iterrows()])


def main():
    OUT.mkdir(exist_ok=True)
    for fn, need in ((t_nlh, "r2_bridge_nonlinear.csv"), (t_pv, "r2_policy_value_bootstrap.csv"),
                     (t_calib, "r2_calibrated.csv"), (t_harm, "r2_harm_decision_curve.csv"),
                     (t_lhs, "r2_lhs_design.csv"), (t_falsify, "r2_falsification_size_power.csv"),
                     (t_dml, "r2_reference_crosscheck.csv"), (t_cross, "r2_overid_hansen_j.csv"),
                     (t_misc, "r2_nonconvergence_per_cell.csv")):
        if (RES / need).exists():
            fn()


if __name__ == "__main__":
    main()
