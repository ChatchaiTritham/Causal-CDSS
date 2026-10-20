"""Write every LaTeX table of the manuscript and supplement from the released result files.

No number in a table is typed by hand: each table body below is generated from a CSV in
results/ and written to tables/*.tex, which the manuscript includes with \\input.

Usage: python scripts/make_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES, OUT = REPO / "results", REPO / "tables"
OUT.mkdir(exist_ok=True)

DOM = {"sepsis": "Sepsis", "ards": "ARDS", "cardiac": "ACS"}
EST = {"naive": "Naive", "backdoor_observed": "Backdoor, age only",
       "backdoor_proxy": "Backdoor, age $+$ $Z$, $W$", "proximal": "\\textbf{Proximal}",
       "proximal_ipw": "Proximal IPW", "proximal_dr": "\\textbf{Proximal DR}"}
POL = {"treat_all": "Treat all", "treat_none": "Treat none",
       "cdss_top50": "Risk rule, top 50\\%", "cdss_top25": "Risk rule, top 25\\%",
       "knowledge": "Knowledge-driven rule"}


def f(x, d=3, sign=False):
    if pd.isna(x) or x == "":
        return "---"
    s = f"{float(x):+.{d}f}" if sign else f"{float(x):.{d}f}"
    return f"${s}$" if sign else s


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote tables/" + name)


def tab_proximal() -> None:
    d = pd.read_csv(RES / "proximal_estimates.csv")
    lines = []
    for dom, nice in DOM.items():
        g = d[d.domain == dom].set_index("estimator")
        for i, (k, lab) in enumerate(EST.items()):
            r = g.loc[k]
            lines.append(f"{nice if i == 0 else ''} & {lab} & {f(r.mean_estimate, 3, True)} & "
                         f"{f(r.bias, 3, True)} & {f(r.mcse_bias, 4)} & {f(r.rmse)} & "
                         f"{f(r.ci95_coverage, 2) if k not in ('naive', 'backdoor_observed') else '---'} \\\\")
        lines.append("\\midrule" if dom != "cardiac" else "")
    write("tab_proximal.tex", "\n".join(l for l in lines if l))


def tab_ope() -> None:
    c = pd.read_csv(RES / "policy_value_coverage.csv")
    v = pd.read_csv(RES / "policy_value_estimates.csv")
    m = c.merge(v[["domain", "policy", "nuisance", "true_value"]], on=["domain", "policy", "nuisance"])
    g = m.groupby(["policy", "nuisance"]).mean(numeric_only=True)
    lines = []
    for p, lab in POL.items():
        rec, ora = g.loc[(p, "logistic_recorded")], g.loc[(p, "logistic_oracle")]
        cov_rng = m[(m.policy == p) & (m.nuisance == "logistic_recorded")].dr_ci95_coverage
        lines.append(f"{lab} & {f(rec.treated_fraction, 2)} & {f(rec.true_value)} & "
                     f"{f(rec.mean_ess_fraction, 2)} & {f(rec.mean_policy_agreement, 2)} & "
                     f"{f(rec.dr_bias, 3, True)} & {f(rec.dr_ci95_coverage, 2)} "
                     f"({f(cov_rng.min(), 2)}--{f(cov_rng.max(), 2)}) & "
                     f"{f(ora.dr_bias, 3, True)} & {f(ora.dr_ci95_coverage, 2)} \\\\")
    write("tab_ope.tex", "\n".join(lines))


def tab_ope_full() -> None:
    c = pd.read_csv(RES / "policy_value_coverage.csv")
    lines = []
    for dom, nice in DOM.items():
        for i, (p, lab) in enumerate(POL.items()):
            row = [nice if i == 0 else "", lab]
            for nk in ("linear_recorded", "logistic_recorded", "logistic_oracle"):
                r = c[(c.domain == dom) & (c.policy == p) & (c.nuisance == nk)].iloc[0]
                row += [f(r.dr_bias, 3, True), f"{f(r.dr_ci95_coverage, 2)}"]
            r = c[(c.domain == dom) & (c.policy == p) & (c.nuisance == "logistic_recorded")].iloc[0]
            row += [f(r.mean_ess_fraction, 2)]
            lines.append(" & ".join(row) + " \\\\")
        if dom != "cardiac":
            lines.append("\\midrule")
    write("tab_ope_full.tex", "\n".join(lines))


def tab_dml() -> None:
    d = pd.read_csv(RES / "dml_estimates.csv")
    lines = []
    for dom, nice in DOM.items():
        for i, (adj, lab) in enumerate((("oracle_severity_recorded", "Age, severity (oracle)"),
                                        ("recorded_only", "Age, $Z$, $W$ (recorded)"))):
            a = d[(d.domain == dom) & (d.score == "aipw") & (d.adjustment_set == adj)].iloc[0]
            p = d[(d.domain == dom) & (d.score == "plr") & (d.adjustment_set == adj)].iloc[0]
            lines.append(f"{nice if i == 0 else ''} & {lab} & {f(a.bias, 3, True)} & "
                         f"{f(a.ci95_coverage, 2)} & {f(p.bias, 3, True)} & {f(p.ci95_coverage, 2)} \\\\")
        if dom != "cardiac":
            lines.append("\\midrule")
    write("tab_dml.tex", "\n".join(lines))


def tab_dag() -> None:
    d = pd.read_csv(RES / "dag_implication_tests.csv")
    r = pd.read_csv(RES / "dag_implication_tests_recorded.csv")
    names = {"true": "True graph", "missing_edge_sev_to_y": "Missing severity $\\to$ outcome",
             "extra_edge_age_to_a": "Extra age $\\to$ treatment"}
    lines = []
    for g, lab in names.items():
        cells = []
        for src in (d, r):
            for dom in DOM:
                s = src[(src.domain == dom) & (src.graph == g)]
                cells.append("---" if s.empty else f"{int(s.holds_bonferroni.sum())}/{len(s)}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_dag.tex", "\n".join(lines))


def tab_sweep(fname: str, key: str, out: str) -> None:
    d = pd.read_csv(RES / fname)
    grid = sorted(d[key].unique())
    lines = []
    for dom, nice in DOM.items():
        for i, (e, lab) in enumerate((("proximal", "Proximal"), ("backdoor_proxy", "Backdoor, age $+$ $Z$, $W$"),
                                      ("backdoor_observed", "Backdoor, age only"),
                                      ("proximal_dr", "Proximal DR"))):
            g = d[(d.domain == dom) & (d.estimator == e)].set_index(key)
            lines.append(f"{nice if i == 0 else ''} & {lab} & "
                         + " & ".join(f(g.loc[x, "abs_bias"]) for x in grid) + " \\\\")
        g = d[(d.domain == dom) & (d.estimator == "proximal")].set_index(key)
        lines.append(" & \\emph{Proximal 95\\% coverage} & "
                     + " & ".join(f(g.loc[x, "ci95_coverage"], 2) for x in grid) + " \\\\")
        g = d[(d.domain == dom) & (d.estimator == "proximal_dr")].set_index(key)
        lines.append(" & \\emph{Proximal DR 95\\% coverage} & "
                     + " & ".join(f(g.loc[x, "ci95_coverage"], 2) for x in grid) + " \\\\")
        if dom != "cardiac":
            lines.append("\\midrule")
    write(out, "\n".join(lines))


def tab_crossing() -> None:
    d = pd.read_csv(RES / "proximal_crossing.csv")
    write("tab_crossing.tex", "\n".join(
        f"{DOM[r.domain]} & {f(r.gamma_star, 2)} & {f(r.delta_star, 2)} & "
        f"{f(r.gamma_star_dr, 2)} & {f(r.delta_star_dr, 2)} \\\\" for r in d.itertuples()))


def tab_nonlinear() -> None:
    d = pd.read_csv(RES / "proximal_nonlinear.csv")
    reg = {"linear": "Linear (reference)", "quadratic": "Quadratic severity",
           "threshold": "Threshold severity", "proxy_nl": "Dichotomised proxies"}
    lines = []
    for dom, nice in DOM.items():
        for i, (k, lab) in enumerate(reg.items()):
            g = d[(d.domain == dom) & (d.regime == k)].set_index("estimator")
            lines.append(f"{nice if i == 0 else ''} & {lab} & {f(g.loc['proximal', 'true_ate'], 3, True)} & "
                         f"{f(g.loc['proximal', 'abs_bias'], 4)} & {f(g.loc['proximal', 'mcse_bias'], 4)} & "
                         f"{f(g.loc['proximal', 'ci95_coverage'], 2)} & "
                         f"{f(g.loc['backdoor_proxy', 'abs_bias'], 3)} & {f(g.loc['backdoor_observed', 'abs_bias'], 3)} & "
                         f"{f(g.loc['proximal_dr', 'abs_bias'], 4)} & "
                         f"{f(g.loc['proximal_dr', 'ci95_coverage'], 2)} \\\\")
        if dom != "cardiac":
            lines.append("\\midrule")
    write("tab_nonlinear.tex", "\n".join(lines))


def tab_weak() -> None:
    d = pd.read_csv(RES / "proximal_weak_proxies.csv")
    lines = []
    for dom, nice in DOM.items():
        for i, s in enumerate(sorted(d.proxy_scale.unique(), reverse=True)):
            g = d[(d.domain == dom) & (d.proxy_scale == s)].set_index("estimator")
            p = g.loc["proximal"]
            lines.append(f"{nice if i == 0 else ''} & {s:g} & {p.first_stage_f:.1f} & {f(p.bias, 3, True)} & "
                         f"{f(p.sd)} & {f(p.ci95_coverage, 2)} & {f(g.loc['backdoor_proxy', 'bias'], 3, True)} & "
                         f"{f(g.loc['proximal_dr', 'bias'], 3, True)} & {f(g.loc['proximal_dr', 'sd'])} & "
                         f"{f(g.loc['proximal_dr', 'ci95_coverage'], 2)} \\\\")
        if dom != "cardiac":
            lines.append("\\midrule")
    write("tab_weak.tex", "\n".join(lines))


def tab_models() -> None:
    d = pd.read_csv(RES / "model_comparison_seeds.csv")
    lines = []
    for r in d.itertuples():
        if r.model == "Mean":
            lines.append("\\midrule")
            lines.append(f"Mean & {f(r.auroc_mean)} & {r.assoc_acc_mean * 100:.1f}\\% & "
                         f"{r.adj_causal_acc_mean * 100:.1f}\\% & {f(r.cee_causal_mean)} & {f(r.ate_align_causal_mean)} \\\\")
            continue
        lines.append(f"{r.model} & ${r.auroc_mean:.3f} \\pm {r.auroc_std:.3f}$ & "
                     f"${r.assoc_acc_mean * 100:.1f} \\pm {r.assoc_acc_std * 100:.1f}$\\% & "
                     f"${r.adj_causal_acc_mean * 100:.1f} \\pm {r.adj_causal_acc_std * 100:.1f}$\\% & "
                     f"${r.cee_causal_mean:.3f} \\pm {r.cee_causal_std:.3f}$ & "
                     f"${r.ate_align_causal_mean:.3f} \\pm {r.ate_align_causal_std:.3f}$ \\\\")
    write("tab_models.tex", "\n".join(lines))


def tab_bootstrap() -> None:
    d = pd.read_csv(RES / "proximal_bootstrap_summary.csv")
    name = {"proximal_2sls": "Proximal 2SLS", "proximal_dr": "Proximal DR"}
    write("tab_bootstrap.tex", "\n".join(
        f"{DOM[r.domain]} & {name[r.estimator]} & {f(r.emp_sd, 4)} & {f(r.mean_plugin_se, 4)} & "
        f"{f(r.mean_boot_se, 4)} & {f(r.cov_plugin, 2)} & {f(r.cov_boot_wald, 2)} & "
        f"{f(r.cov_boot_pct, 2)} \\\\" for r in d.itertuples()))


def tab_sample_size() -> None:
    d = pd.read_csv(RES / "sample_size.csv")
    name = {"backdoor_proxy": "Backdoor, age $+$ $Z$, $W$", "proximal": "Proximal 2SLS",
            "proximal_dr": "Proximal DR"}
    lines = []
    for dm in DOM:
        for e in name:
            g = d[(d.domain == dm) & (d.estimator == e)].sort_values("n")
            lines.append(f"{DOM[dm]} & {name[e]} & " + " & ".join(
                f"{f(r.rmse, 3)} / {f(r.ci95_coverage, 2)}" for r in g.itertuples()) + " \\\\")
    write("tab_sample_size.tex", "\n".join(lines))


def tab_audit_card() -> None:
    import json
    cards = json.loads((RES / "audit_card.json").read_text(encoding="utf-8"))
    rows = []
    for c in cards:
        t = c["detail"]
        ci = lambda v: f"${v[0]:+.3f}$ [${v[1]:+.3f}$, ${v[2]:+.3f}$]"
        rows.append(f"{DOM[c['domain']]} & {t['first_stage_F']:.0f} & {ci(t['ate_proximal_dr'])} & "
                    f"{ci(t['ate_proxy_adjusted'])} & {t['gamma_star']:.2f} / {t['delta_star']:.2f} & "
                    f"{t['policy_value_dr']:.3f} & {t['policy_ess_fraction']:.2f} \\\\")
    write("tab_audit_card.tex", "\n".join(rows))
    write("card_sepsis.json", json.dumps(cards[0], indent=1))


def tab_proximal_ope() -> None:
    d = pd.read_csv(RES / "proximal_policy_value.csv")
    pol = dict(POL, age_rule="Age rule (age $>0$)")
    lines = []
    for p in ["age_rule", "treat_all", "treat_none", "cdss_top50", "cdss_top25", "knowledge"]:
        g = d[d.policy == p]
        rng = lambda c: f"{f(g[c].min(), 2)}--{f(g[c].max(), 2)}"
        lines.append(f"{pol[p]} & {'yes' if g.identified.iloc[0] else 'no'} & "
                     f"{f(g.dr_bias.abs().max(), 3)} & {rng('dr_coverage')} & "
                     f"{f(g.prox_bias.abs().max(), 3)} & {rng('prox_coverage')} \\\\")
    write("tab_proximal_ope.tex", "\n".join(lines))


def tab_subgroup() -> None:
    d = pd.read_csv(RES / "subgroup_audit.csv")
    grp = {"young": "Youngest", "middle": "Middle", "old": "Oldest"}
    lines = []
    for dm in DOM:
        for i, (k, lab) in enumerate(grp.items()):
            g = d[(d.domain == dm) & (d.age_group == k)].set_index("item")
            bp, dr = g.loc["backdoor_proxy"], g.loc["proximal_dr"]
            kn, rr = g.loc["knowledge"], g.loc["cdss_top50"]
            lines.append(f"{DOM[dm] if i == 0 else ''} & {lab} & {f(bp.truth, 3, True)} & "
                         f"{f(bp.coverage, 2)} & {f(dr.coverage, 2)} & {f(kn.benefit_share, 2)} & "
                         f"{f(kn.benefit_recall, 2)} & {f(rr.benefit_recall, 2)} & "
                         f"{f(kn.risk_reduction, 3)} & {f(rr.risk_reduction, 3)} \\\\")
        if dm != "cardiac":
            lines.append(r"\midrule")
    write("tab_subgroup.tex", "\n".join(lines))



def tab_policy_violation() -> None:
    d = pd.read_csv(RES / "proximal_policy_violation.csv")
    pol = {"age_rule": "Age rule (age $>0$)", "cdss_top50": "Risk rule, top 50\%",
           "knowledge": "Knowledge-driven rule"}
    sym = {"gamma": r"$\gamma$ ($Z \to Y$)", "delta": r"$\delta$ ($T \to W$)"}
    lines = []
    for k in ("gamma", "delta"):
        for i, (p, lab) in enumerate(pol.items()):
            cells = []
            for lv in sorted(d.level.unique()):
                g = d[(d.violation == k) & (d.policy == p) & (d.level == lv)]
                cells.append(f"{f(g.prox_bias.abs().max(), 3)} / {f(g.prox_coverage.min(), 2)}")
            lines.append(f"{sym[k] if i == 0 else ''} & {lab} & " + " & ".join(cells) + " \\\\")
        if k == "gamma":
            lines.append(r"\midrule")
    write("tab_policy_violation.tex", "\n".join(lines))

if __name__ == "__main__":
    tab_proximal()
    tab_ope()
    tab_ope_full()
    tab_sweep("proximal_robustness.csv", "gamma_z_to_y", "tab_sweep_gamma.tex")
    tab_sweep("proximal_robustness_w.csv", "delta_a_to_w", "tab_sweep_delta.tex")
    tab_crossing()
    tab_nonlinear()
    tab_weak()
    tab_models()
    if (RES / "dml_estimates.csv").exists():
        tab_dml()
    if "holds_bonferroni" in pd.read_csv(RES / "dag_implication_tests.csv").columns:
        tab_dag()
    if (RES / "proximal_bootstrap_summary.csv").exists():
        tab_bootstrap()
    if (RES / "sample_size.csv").exists():
        tab_sample_size()
    if (RES / "audit_card.json").exists():
        tab_audit_card()
    if (RES / "proximal_policy_value.csv").exists():
        tab_proximal_ope()
        tab_subgroup()
    if (RES / "proximal_policy_violation.csv").exists():
        tab_policy_violation()
