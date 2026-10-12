"""Figures for scripts/unmeasured_confounding.py.

fig_proximal_bias.pdf  : estimator bias against the interventional ground truth, per domain,
                         when the confounder is unmeasured.
fig_policy_overlap.pdf : doubly-robust interval coverage against the overlap the policy
                         leaves, across policies and domains.
fig_proximal_dag.pdf   : the causal graph of the unmeasured-confounding design, with the two
                         negative controls and the exclusion restrictions the sweeps violate.
fig_proximal_sweeps.pdf: absolute bias of the proximal and backdoor estimators as each
                         exclusion restriction is violated (Z -> Y and T -> W).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES, OUT = REPO / "results", REPO / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

NICE_DOMAIN = {"sepsis": "Sepsis", "ards": "ARDS", "cardiac": "ACS"}
EST_STYLE = {"naive": ("#CC79A7", "Naive (unadjusted)"),
             "backdoor_observed": ("#E69F00", "Backdoor on age only"),
             "backdoor_proxy": ("#56B4E9", "Backdoor on age, $Z$, $W$"),
             "proximal": ("#009E73", "Proximal (negative controls)")}
POL_STYLE = {"treat_all": ("#D55E00", "o"), "treat_none": ("#CC79A7", "s"),
             "cdss_top50": ("#0072B2", "^"), "cdss_top25": ("#009E73", "D"),
             "knowledge": ("#000000", "*")}
NICE_POLICY = {"treat_all": "Treat all", "treat_none": "Treat none",
               "cdss_top50": "Risk rule, top 50%", "cdss_top25": "Risk rule, top 25%",
               "knowledge": "Knowledge-driven rule"}


def fig_bias() -> None:
    d = pd.read_csv(RES / "proximal_estimates.csv")
    domains = list(NICE_DOMAIN)
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.7))
    width = 0.2
    for j, (est, (col, lab)) in enumerate(EST_STYLE.items()):
        g = d[d.estimator == est].set_index("domain").loc[domains]
        xs = [i + (j - 1.5) * width for i in range(len(domains))]
        a.bar(xs, g.mean_estimate, width=width, color=col, edgecolor="black",
              linewidth=0.4, label=lab, yerr=1.96 * g.sd,
              error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#333333"})
        b.bar(xs, g.abs_bias, width=width, color=col, edgecolor="black", linewidth=0.4)
    truth = d[d.estimator == "proximal"].set_index("domain").loc[domains, "true_ate"]
    a.plot(range(len(domains)), truth, "k_", markersize=26, markeredgewidth=1.4,
           label="True ATE (interventional)")
    a.axhline(0, color="#999999", linewidth=0.6)
    a.set_xticks(range(len(domains)), [NICE_DOMAIN[x] for x in domains])
    b.set_xticks(range(len(domains)), [NICE_DOMAIN[x] for x in domains])
    a.set_ylabel("Estimated ATE on mortality")
    b.set_ylabel("Absolute bias")
    a.text(0.0, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    b.text(0.0, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(OUT / "fig_proximal_bias.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_proximal_bias.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_proximal_bias.pdf")


def fig_overlap() -> None:
    """Interval coverage against overlap, with recorded and with oracle nuisance models.

    Hollow markers use the recorded covariates (logistic nuisances); filled markers add the
    unrecorded severity. If overlap were the cause of undercoverage, adding severity would not
    restore it at an unchanged effective sample size.
    """
    c = pd.read_csv(RES / "policy_value_coverage.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.7))
    for pol, (col, mk) in POL_STYLE.items():
        for nk, face in (("logistic_recorded", "none"), ("logistic_oracle", col)):
            g = c[(c.policy == pol) & (c.nuisance == nk)]
            a.scatter(g.mean_ess_fraction, g.dr_ci95_coverage, marker=mk, s=34,
                      facecolors=face, edgecolors=col, linewidths=0.9,
                      label=NICE_POLICY[pol] if nk == "logistic_oracle" else None)
            b.scatter(g.mean_ess_fraction, g.dr_bias.abs(), marker=mk, s=34,
                      facecolors=face, edgecolors=col, linewidths=0.9)
    a.axhline(0.95, color="#4D4D4D", linestyle=(0, (1, 2)), linewidth=1.0)
    a.annotate("nominal 0.95", (0.1, 0.95), xytext=(0, 4), textcoords="offset points",
               fontsize=7, color="#4D4D4D")
    a.set_xlabel("Effective sample size (fraction)")
    a.set_ylabel("Coverage of 95% DR interval")
    a.set_ylim(-0.03, 1.05)
    b.set_xlabel("Effective sample size (fraction)")
    b.set_ylabel("|Bias| of DR value estimate")
    a.text(0.0, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    b.text(0.0, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    from matplotlib.lines import Line2D
    h, lab = a.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", label="recorded covariates"),
          Line2D([], [], marker="o", ls="", mfc="0.3", mec="0.3", label="+ severity (oracle)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(OUT / "fig_policy_overlap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_policy_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_policy_overlap.pdf")


def fig_dag() -> None:
    """Graph of the unmeasured-confounding design; dashed red edges are the two restrictions."""
    from matplotlib.patches import FancyArrowPatch
    nodes = {"Age": (0.50, 0.92), "U": (0.50, 0.55), "Z": (0.09, 0.55),
             "T": (0.22, 0.08), "Y": (0.78, 0.08), "W": (0.91, 0.55)}
    label = {"Age": "Age", "U": "Severity $U$\n(unrecorded)",
             "Z": "$Z$\nneg.-control\nexposure",
             "T": "Treatment $T$", "Y": "Mortality $Y$",
             "W": "$W$\nneg.-control\noutcome"}
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.set_xlim(0, 1); ax.set_ylim(-0.04, 1.02); ax.axis("off")
    for k, (x, y) in nodes.items():
        dashed = k == "U"
        ax.text(x, y, label[k], ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round,pad=0.35", "fc": "#F2F2F2" if dashed else "white",
                      "ec": "#333333", "lw": 0.7, "ls": "--" if dashed else "-"})

    def edge(u, v, color="#333333", ls="-", rad=0.0, lw=0.9):
        ax.add_patch(FancyArrowPatch(nodes[u], nodes[v], arrowstyle="-|>", mutation_scale=9,
                                     color=color, lw=lw, ls=ls, shrinkA=22, shrinkB=22,
                                     connectionstyle=f"arc3,rad={rad}"))
    for u, v in [("Age", "U"), ("U", "T"), ("U", "Y"), ("U", "Z"), ("U", "W"),
                 ]:
        edge(u, v)
    edge("Age", "Y", rad=-0.25)
    edge("T", "Y", color="#0072B2", lw=1.6)
    edge("Z", "Y", color="#D55E00", ls="--", rad=0.12)
    edge("T", "W", color="#D55E00", ls="--", rad=0.12)
    ax.text(0.27, 0.36, r"$\gamma$", color="#D55E00", fontsize=9)
    ax.text(0.71, 0.36, r"$\delta$", color="#D55E00", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_proximal_dag.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_proximal_dag.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_proximal_dag.pdf")


def fig_sweeps() -> None:
    z = pd.read_csv(RES / "proximal_robustness.csv")
    w = pd.read_csv(RES / "proximal_robustness_w.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), sharey=True)
    dom_col = {"sepsis": "#0072B2", "ards": "#009E73", "cardiac": "#D55E00"}
    for ax, d, key, xl, tag in ((axes[0], z, "gamma_z_to_y", r"Direct effect $\gamma$ of $Z$ on $Y$", "(a)"),
                                (axes[1], w, "delta_a_to_w", r"Direct effect $\delta$ of $T$ on $W$", "(b)")):
        for dom, col in dom_col.items():
            for est, ls, mk in (("proximal", "-", "o"), ("backdoor_proxy", ":", "^"),
                                ("backdoor_observed", "--", "s"), ("proximal_dr", "-.", "D")):
                g = d[(d.domain == dom) & (d.estimator == est)].sort_values(key)
                ax.plot(g[key], g.abs_bias, ls=ls, marker=mk, ms=3, lw=1.0, color=col)
        ax.set_xlabel(xl)
        ax.text(0.0, 1.04, tag, transform=ax.transAxes, fontweight="bold")
    axes[0].set_ylabel("Absolute bias (mortality)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=1.2, label=NICE_DOMAIN[k]) for k, c in dom_col.items()]
    h += [Line2D([], [], color="black", ls="-", marker="o", ms=3, label="Proximal"),
          Line2D([], [], color="black", ls=":", marker="^", ms=3, label="Backdoor on age, Z, W"),
          Line2D([], [], color="black", ls="--", marker="s", ms=3, label="Backdoor on age only"),
          Line2D([], [], color="black", ls="-.", marker="D", ms=3, label="Proximal DR")]
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=6.5,
               handlelength=2.2, columnspacing=1.0, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "fig_proximal_sweeps.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_proximal_sweeps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_proximal_sweeps.pdf")


if __name__ == "__main__":
    fig_bias()
    fig_overlap()
    fig_dag()
    fig_sweeps()
