"""Overview figure of the evaluation protocol (fig_protocol_overview.pdf).

Pure diagram: no numbers are drawn from results, so it needs no driver to be run first.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42,
})

# Okabe-Ito colours, matching the other figures
C_IN, C_ID, C_CHK, C_OUT = "#56B4E9", "#009E73", "#E69F00", "#CC79A7"


def box(ax, x, y, w, h, title, body, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=colour + "22", ec=colour, lw=0.9))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", weight="bold", fontsize=8)
    ax.text(x + w / 2, y + h - 0.075, body, ha="center", va="top", fontsize=7, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.8, color="0.25"))


fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.set_xlim(-0.015, 1.005); ax.set_ylim(0, 1); ax.axis("off")

# Stage 1: knowledge -> SCM
box(ax, 0.00, 0.30, 0.19, 0.48, "1  Domain knowledge",
    "Hand-specified DAGs\n(sepsis, ARDS, ACS)\nfrom trial and\nguideline evidence\n\nStructural causal\nmodels, seeded", C_IN)
# Stage 2: two cohorts
box(ax, 0.25, 0.56, 0.20, 0.36, "2a  Severity recorded",
    "Verification cohort\n5 CDSS models\nnaive / backdoor / DR", C_IN)
box(ax, 0.25, 0.08, 0.20, 0.40, "2b  Severity unrecorded",
    "Audit cohort\nproxies $Z$, $W$ only\n(the realistic case)", C_IN)
# Stage 3: identification and checks
box(ax, 0.51, 0.56, 0.22, 0.36, "3a  Identification",
    "Proximal 2SLS\nvs. backdoor on\nrecorded covariates\n+ exclusion sweeps $\\gamma,\\delta$", C_ID)
box(ax, 0.51, 0.08, 0.22, 0.40, "3b  Checks",
    "Cross-fitted DML\n(oracle vs. recorded)\n\nDAG implication tests\n(Bonferroni)", C_CHK)
# Stage 4: outputs
box(ax, 0.79, 0.30, 0.21, 0.48, "4  Reported",
    "Bias vs. interventional\nground truth\n\nPolicy value $V(\\pi)$\n+ ESS and agreement\n(overlap diagnostics)", C_OUT)

arrow(ax, 0.19, 0.62, 0.25, 0.74)
arrow(ax, 0.19, 0.46, 0.25, 0.28)
arrow(ax, 0.45, 0.74, 0.51, 0.74)
arrow(ax, 0.45, 0.28, 0.51, 0.28)
arrow(ax, 0.45, 0.36, 0.51, 0.66)
arrow(ax, 0.73, 0.74, 0.79, 0.62)
arrow(ax, 0.73, 0.28, 0.79, 0.46)

ax.text(0.5, 0.985, r"Ground truth at every stage: $\mathbb{E}[Y\mid\mathrm{do}(T=t)]$ computed by intervening on the same mechanisms",
        ha="center", va="top", fontsize=7.5, style="italic", color="0.3")

fig.savefig(OUT / "fig_protocol_overview.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_protocol_overview.png", dpi=600, bbox_inches="tight")
print("wrote", OUT / "fig_protocol_overview.pdf")
