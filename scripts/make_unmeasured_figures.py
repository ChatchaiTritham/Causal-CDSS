"""Figures for scripts/unmeasured_confounding.py.

fig_proximal_bias.pdf  : estimator bias against the interventional ground truth, per domain,
                         when the confounder is unmeasured.
fig_policy_overlap.pdf : doubly-robust interval coverage against the overlap the policy
                         leaves, across policies and domains.
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
             "backdoor_observed": ("#E69F00", "Backdoor on recorded covariates"),
             "proximal": ("#009E73", "Proximal (negative controls)")}
POL_STYLE = {"treat_all": ("#D55E00", "o"), "treat_none": ("#CC79A7", "s"),
             "cdss_top50": ("#0072B2", "^"), "cdss_top25": ("#009E73", "D")}


def fig_bias() -> None:
    d = pd.read_csv(RES / "proximal_estimates.csv")
    domains = list(NICE_DOMAIN)
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.7))
    width = 0.26
    for j, (est, (col, lab)) in enumerate(EST_STYLE.items()):
        g = d[d.estimator == est].set_index("domain").loc[domains]
        xs = [i + (j - 1) * width for i in range(len(domains))]
        a.bar(xs, g.mean_estimate, width=width, color=col, edgecolor="black",
              linewidth=0.4, label=lab)
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
    c = pd.read_csv(RES / "policy_value_coverage.csv")
    p = pd.read_csv(RES / "policy_value_estimates.csv")
    m = c.merge(p[["domain", "policy", "dr_rmse"]], on=["domain", "policy"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.7))
    for pol, (col, mk) in POL_STYLE.items():
        g = m[m.policy == pol]
        a.scatter(g.mean_ess_fraction, g.dr_ci95_coverage, color=col, marker=mk, s=34,
                  edgecolors="black", linewidths=0.4, label=pol.replace("_", " "))
        b.scatter(g.mean_ess_fraction, g.dr_rmse, color=col, marker=mk, s=34,
                  edgecolors="black", linewidths=0.4)
    a.axhline(0.95, color="#4D4D4D", linestyle=(0, (1, 2)), linewidth=1.0)
    a.annotate("nominal 0.95", (0.12, 0.95), xytext=(0, 4), textcoords="offset points",
               fontsize=7, color="#4D4D4D")
    a.set_xlabel("Effective sample size (fraction)")
    a.set_ylabel("Coverage of 95\\% DR interval" if plt.rcParams["text.usetex"]
                 else "Coverage of 95% DR interval")
    a.set_ylim(-0.03, 1.05)
    b.set_xlabel("Effective sample size (fraction)")
    b.set_ylabel("RMSE of DR value estimate")
    a.text(0.0, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    b.text(0.0, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "fig_policy_overlap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_policy_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_policy_overlap.pdf")


if __name__ == "__main__":
    fig_bias()
    fig_overlap()
