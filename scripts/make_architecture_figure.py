"""Where the audit sits in a knowledge-based CDSS (fig_architecture.pdf), single column.

Pure diagram: no numbers are drawn from results.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6.3, "pdf.fonttype": 42})
C_KB, C_DATA, C_ALG, C_OUT = "#56B4E9", "#E69F00", "#009E73", "#CC79A7"


def box(ax, x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc=c + "22", ec=c, lw=0.8))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", weight="bold", fontsize=6.6)
    ax.text(x + w / 2, y + h - 0.12, body, ha="center", va="top", fontsize=5.9, linespacing=1.25)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.25"))


fig, ax = plt.subplots(figsize=(3.45, 2.05))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.00, 0.53, 0.47, 0.45, "Knowledge base",
    "declared DAG $\\to$ FHIR Library / CQL\nproxies $Z,W$ $\\to$ ValueSet (LOINC,\nSNOMED CT); restriction rationale\n$\\to$ Evidence; recommender $\\pi$\n$\\to$ PlanDefinition", C_KB)
box(ax, 0.53, 0.53, 0.47, 0.45, "Records (FHIR extraction)",
    "age, $Z$, $W$, treatment $T$,\noutcome $Y$; severity $U$ absent\n\nindication marker for the\nsafety veto", C_DATA)
box(ax, 0.00, 0.02, 0.47, 0.45, "Algorithm 1 (audit)",
    "graph tests; first-stage $F$;\nbridges $h,q$; elicited $\\bar\\gamma,\\bar\\delta$\nvs reference $\\gamma^\\ast,\\delta^\\ast$;\nverdict: proximal / proxy /\ninconclusive; $V(\\pi)$, subgroups", C_ALG)
box(ax, 0.53, 0.02, 0.47, 0.45, "Audit card",
    "CDS Hooks 2.0 response,\nnumbers in card.extension;\nmaps to HTI-1 source attributes,\nFDA/IMDRF/EU AI Act evidence;\nre-audit triggers", C_OUT)
arrow(ax, (0.23, 0.53), (0.23, 0.47))
arrow(ax, (0.765, 0.53), (0.40, 0.47))
arrow(ax, (0.47, 0.245), (0.53, 0.245))
fig.savefig(OUT / "fig_architecture.pdf", bbox_inches="tight")
print("wrote", OUT / "fig_architecture.pdf")
