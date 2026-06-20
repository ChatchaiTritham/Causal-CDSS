#!/usr/bin/env python3
"""
Figure 3: CATE Heterogeneity — Heatmap + Scatter Plot
KAIS Causal Models for CDSS — Springer submission

Left panel (A): heatmap of true CATE across severity x age subgroups (sepsis).
Right panel (B): estimated vs true CATE scatter for the 9 subgroups, with the
                 empirical Pearson r and OLS slope computed from the data.

Data source: results/cate_subgroups.csv (written by run_all.py, seed 42).
CATE stored on probability scale, shown in percentage points (x100).
NOTHING is hardcoded — r, slope and every cell come from the CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pub_style import apply_pub_style, save_fig, results_dir, PALETTE

apply_pub_style()

DOMAIN = "sepsis"          # figure illustrates the sepsis CATE surface
AGE_ORDER = ["age_low", "age_mid", "age_high"]
SEV_ORDER = ["sev_low", "sev_mid", "sev_high"]
AGE_LABELS = ["Age < 50", "Age 50–70", "Age > 70"]
SEV_LABELS = ["qSOFA 0–1\n(Low risk)", "qSOFA 2\n(Moderate risk)", "qSOFA 3\n(High risk)"]

# ---------- Load data ----------
df = pd.read_csv(results_dir() / "cate_subgroups.csv")
d = df[df["domain"] == DOMAIN].copy()

# build 3x3 grids (rows=severity, cols=age) in percentage points
true_grid = np.full((3, 3), np.nan)
est_grid = np.full((3, 3), np.nan)
for _, r in d.iterrows():
    i = SEV_ORDER.index(r["severity_group"])
    j = AGE_ORDER.index(r["age_group"])
    true_grid[i, j] = r["true_cate"] * 100.0
    est_grid[i, j] = r["estimated_cate"] * 100.0

true_flat = true_grid.flatten()
est_flat = est_grid.flatten()

# empirical fit (computed, not asserted)
r_val = float(np.corrcoef(true_flat, est_flat)[0, 1])
slope, intercept = np.polyfit(true_flat, est_flat, 1)

# ---------- Figure ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.05]})

# ===== Panel A: heatmap =====
ax1.grid(False)
vmin, vmax = np.nanmin(true_grid), np.nanmax(true_grid)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "cate", ["#DEEBF7", "#9ECAE1", PALETTE[0], "#08306B"], N=256)
im = ax1.imshow(true_grid, cmap=cmap, aspect="auto",
                vmin=vmin, vmax=vmax)

mid = (vmin + vmax) / 2
for i in range(3):
    for j in range(3):
        val = true_grid[i, j]
        tc = "white" if val < mid else "#08306B"
        ax1.text(j, i, f"{val:.1f}%", ha="center", va="center",
                 fontsize=11, fontweight="bold", color=tc)

ax1.set_xticks(range(3)); ax1.set_xticklabels(AGE_LABELS)
ax1.set_yticks(range(3)); ax1.set_yticklabels(SEV_LABELS)
ax1.tick_params(length=0)
for s in ax1.spines.values():
    s.set_visible(False)
ax1.set_title("(A)  True CATE by Severity × Age\n"
              "(Sepsis: Early Antibiotics <3 h vs. Delayed)",
              fontsize=10.5, fontweight="bold")
cbar = fig.colorbar(im, ax=ax1, shrink=0.82, pad=0.03)
cbar.set_label("CATE: change in mortality probability (pp)")
cbar.ax.tick_params(labelsize=8)

# ===== Panel B: scatter =====
ax2.set_facecolor("#FCFCFD")
sev_colors = [PALETTE[5], PALETTE[0], "#08306B"]   # low / mid / high severity
for i in range(3):          # severity row
    for j in range(3):      # age col
        ax2.scatter(true_grid[i, j], est_grid[i, j], color=sev_colors[i],
                    s=95, edgecolors="white", linewidths=0.8, zorder=4)

lo = float(min(true_flat.min(), est_flat.min())) - 2
hi = float(max(true_flat.max(), est_flat.max())) + 2
diag = np.array([lo, hi])
ax2.plot(diag, diag, "--", color="#555555", lw=1.2, alpha=0.7,
         label="Perfect agreement")
xs = np.linspace(lo, hi, 100)
ax2.plot(xs, slope * xs + intercept, color=PALETTE[1], lw=2.0,
         label=f"OLS fit (slope = {slope:.2f})")

ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
ax2.set_aspect("equal")
ax2.set_xlabel("True CATE (pp)")
ax2.set_ylabel("Estimated CATE (pp)")
ax2.set_title("(B)  Estimated vs. True CATE\n(9 Subgroups: 3 Severity × 3 Age Groups)",
              fontsize=10.5, fontweight="bold")

# severity legend handles
sev_handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markeredgecolor="white",
                          markersize=9, label=lbl)
               for c, lbl in zip(sev_colors,
                                 ["qSOFA 0–1 (Low)", "qSOFA 2 (Moderate)", "qSOFA 3 (High)"])]
fit_handles = [plt.Line2D([0], [0], ls="--", color="#555555", lw=1.2,
                          label="Perfect agreement"),
               plt.Line2D([0], [0], color=PALETTE[1], lw=2.0, label="OLS fit")]
ax2.legend(handles=sev_handles + fit_handles, fontsize=8, loc="upper left")

ax2.text(0.97, 0.05, f"r = {r_val:.2f},  slope = {slope:.2f}",
         transform=ax2.transAxes, fontsize=9, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor=PALETTE[0], alpha=0.95))

fig.suptitle("Conditional Average Treatment Effect (CATE) Heterogeneity — "
             "Sepsis Antibiotic Timing", fontsize=12, fontweight="bold")

OUT = __import__("pathlib").Path(__file__).resolve().parent
save_fig(fig, OUT, "fig03_cate_heterogeneity")
plt.close(fig)
print(f"  (empirical r={r_val:.3f}, slope={slope:.3f})")
