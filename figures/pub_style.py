#!/usr/bin/env python3
"""
pub_style.py
============
Canonical Top-Tier matplotlib style shared by every figure script in the
ChatchaiTritham PhD repositories. Mirrors _management/FIGURE_STYLE.md so all
repos render with one publication-grade look:

  * Okabe-Ito colour-blind-safe palette (consistent series order)
  * Times serif fonts, STIX math
  * spines off, light grid, constrained layout
  * 300 dpi PNG + vector PDF, tight bbox

Call apply_pub_style() ONCE before plotting. Use save_fig() to emit the
matched .pdf + .png pair.

Data must always be loaded from results/ at run time (never hardcode numbers).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Color-blind-safe (Okabe-Ito) — use in this order
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#000000"]

# Semantic aliases drawn from the same palette (keeps DAGs/bars consistent)
C_TREATMENT  = PALETTE[0]   # blue
C_OUTCOME    = PALETTE[1]   # vermillion
C_CONFOUNDER = PALETTE[4]   # orange
C_MEDIATOR   = PALETTE[2]   # bluish green
C_NEUTRAL    = "#555555"    # grey for naive / reference


def apply_pub_style():
    """Apply the canonical publication rcParams (idempotent)."""
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def results_dir():
    """Absolute path to the repo's results/ directory."""
    return (Path(__file__).resolve().parent.parent / "results")


def save_fig(fig, out_dir, basename):
    """Save matched vector .pdf + 300-dpi .png with a tight bbox."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{basename}.{ext}",
                    bbox_inches="tight", facecolor="white")
    print(f"saved: {basename}.pdf / .png  ->  {out_dir}")
