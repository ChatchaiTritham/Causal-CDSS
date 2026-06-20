#!/usr/bin/env python3
"""
Figure 2: Average Treatment Effect (ATE) Forest Plot
KAIS Causal Models for CDSS — Springer submission

Per-domain forest plot of the three estimators (naive, backdoor regression,
doubly-robust) against the do-calculus ground truth. The story is the sign
flip: the naive point sits on the positive (apparently harmful) side while
the adjusted estimators recover the protective true ATE.

Data source: results/ate_by_domain.csv (written by run_all.py, seed 42).
Values are stored on the mortality-probability scale and shown here in
percentage points (x100). NOTHING is hardcoded.
"""

import pandas as pd
import matplotlib.pyplot as plt

from pub_style import apply_pub_style, save_fig, results_dir, PALETTE

apply_pub_style()

# ---------- Load data from results/ ----------
df = pd.read_csv(results_dir() / "ate_by_domain.csv")

# Estimators to plot (column -> display name, marker, palette colour)
EST = [
    ("naive_ate",        "Naive (unadjusted)",        "v", PALETTE[1]),
    ("backdoor_ate",     "Backdoor regression",       "s", PALETTE[2]),
    ("doubly_robust_ate","Doubly-robust",             "D", PALETTE[0]),
]
TRUE_COL = "true_ate"

domains = df.to_dict("records")
n_est = len(EST)

fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)

for ax_idx, (row, ax) in enumerate(zip(domains, axes)):
    true_pp = row[TRUE_COL] * 100.0

    # plot each estimator (point estimates from results; no synthetic CIs)
    for i, (col, name, marker, color) in enumerate(EST):
        y = n_est - 1 - i
        est_pp = row[col] * 100.0
        ax.plot(est_pp, y, marker=marker, color=color, ms=9,
                markeredgecolor='white', markeredgewidth=0.6, zorder=4,
                linestyle='none')
        # connector to truth for readability
        ax.plot([est_pp, true_pp], [y, y], color=color, lw=1.0,
                alpha=0.35, zorder=2)
        ax.annotate(f"{est_pp:+.1f}", (est_pp, y), textcoords="offset points",
                    xytext=(0, 9), ha='center', fontsize=8, color=color)

    # ground-truth reference + zero line
    ax.axvline(true_pp, color=PALETTE[6], lw=1.8, zorder=3,
               label='True ATE (do-calculus)')
    ax.axvline(0, color='#888888', lw=0.8, ls=':', zorder=1)
    ax.text(true_pp - 0.4, -0.55, f"True: {true_pp:.1f} pp",
            fontsize=8.5, color=PALETTE[6], fontweight='bold',
            ha='right', va='bottom')

    # domain label (top-left, kept clear of the truth line)
    ax.text(0.015, 0.95, f"({chr(65 + ax_idx)})  {row['label']}",
            transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='top')

    ax.set_yticks(range(n_est))
    ax.set_yticklabels([name for _, name, _, _ in reversed(EST)])
    ax.set_ylim(-0.95, n_est + 0.1)
    ax.grid(axis='y', visible=False)

axes[-1].set_xlabel('Estimated change in mortality probability (percentage points)')

# shared legend (estimators + truth)
handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                      markeredgecolor='white', markersize=9, label=n)
           for _, n, m, c in EST]
handles.append(plt.Line2D([0], [0], color=PALETTE[6], lw=1.8,
                          label='True ATE (do-calculus)'))
axes[0].legend(handles=handles, loc='lower right', fontsize=8, ncol=2)

fig.suptitle('Average Treatment Effect (ATE) Estimates vs. Causal Ground Truth\n'
             'Naive estimate carries the wrong sign; adjustment recovers the protective effect',
             fontsize=12, fontweight='bold')

OUT = __import__('pathlib').Path(__file__).resolve().parent
save_fig(fig, OUT, 'fig02_intervention_effects')
plt.close(fig)
