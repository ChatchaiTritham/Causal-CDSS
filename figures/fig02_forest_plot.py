#!/usr/bin/env python3
"""
Figure 2: Average Treatment Effect (ATE) Forest Plot
KAIS Causal Models for CDSS — Springer submission

Per-domain forest plot of the three estimators (naive, backdoor regression,
doubly-robust) against the do-calculus ground truth. Each point estimate now
carries a 95% bootstrap confidence interval (seed 42), so the figure reads as a
proper forest plot. The story is the sign flip: the naive interval sits wholly
on the positive (apparently harmful) side while the adjusted intervals fall
wholly on the protective side and bracket the true ATE.

Data source: results/ate_by_domain.csv (written by run_all.py, seed 42).
Point estimates and the *_ci_low / *_ci_high bounds are read straight from the
CSV; values are stored on the mortality-probability scale and shown here in
percentage points (x100). NOTHING is hardcoded.
"""

import pandas as pd
import matplotlib.pyplot as plt

from pubviz import apply_pub_style, save_fig, results_dir, PALETTE

apply_pub_style()

# ---------- Load data from results/ ----------
df = pd.read_csv(results_dir() / "ate_by_domain.csv")

# Estimator -> (point col, ci-low col, ci-high col, display, marker, colour)
EST = [
    ("naive_ate", "naive_ci_low", "naive_ci_high",
     "Naive (unadjusted)", "v", PALETTE[1]),
    ("backdoor_ate", "backdoor_ci_low", "backdoor_ci_high",
     "Backdoor regression", "s", PALETTE[2]),
    ("doubly_robust_ate", "doubly_robust_ci_low", "doubly_robust_ci_high",
     "Doubly-robust", "D", PALETTE[0]),
]
TRUE_COL = "true_ate"

domains = df.to_dict("records")
n_est = len(EST)

fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)

for ax_idx, (row, ax) in enumerate(zip(domains, axes)):
    true_pp = row[TRUE_COL] * 100.0

    # bold zero line (drawn first so estimates sit on top)
    ax.axvline(0, color=PALETTE[6], lw=1.8, ls='-', zorder=1.5,
               label='No effect (zero line)')

    # plot each estimator: point + 95% bootstrap CI (all from results/)
    for i, (col, lo_col, hi_col, name, marker, color) in enumerate(EST):
        y = n_est - 1 - i
        est_pp = row[col] * 100.0
        lo_pp = row[lo_col] * 100.0
        hi_pp = row[hi_col] * 100.0
        # CI whisker
        ax.errorbar(est_pp, y, xerr=[[est_pp - lo_pp], [hi_pp - est_pp]],
                    fmt='none', ecolor=color, elinewidth=1.6,
                    capsize=4, capthick=1.6, zorder=3)
        # point estimate
        ax.plot(est_pp, y, marker=marker, color=color, ms=9,
                markeredgecolor='white', markeredgewidth=0.6, zorder=4,
                linestyle='none')
        ax.annotate(f"{est_pp:+.1f}  [{lo_pp:+.1f}, {hi_pp:+.1f}]",
                    (hi_pp, y), textcoords="offset points",
                    xytext=(8, 0), ha='left', va='center',
                    fontsize=7.5, color=color)

    # ground-truth reference
    ax.axvline(true_pp, color=PALETTE[0], lw=1.4, ls='--', zorder=2,
               label='True ATE (do-calculus)')
    ax.text(true_pp - 0.4, -0.55, f"True: {true_pp:.1f} pp",
            fontsize=8.5, color=PALETTE[0], fontweight='bold',
            ha='right', va='bottom')

    # domain label (top-left, kept clear of the truth line)
    ax.text(0.015, 0.95, f"({chr(65 + ax_idx)})  {row['label']}",
            transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='top')

    ax.set_yticks(range(n_est))
    ax.set_yticklabels([name for _, _, _, name, _, _ in reversed(EST)])
    ax.set_ylim(-0.95, n_est + 0.1)
    ax.grid(axis='y', visible=False)

axes[-1].set_xlabel('Estimated change in mortality probability '
                    '(percentage points; markers = point estimate, '
                    'whiskers = 95% bootstrap CI)')

# single consolidated legend (estimators + zero line + truth)
handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                      markeredgecolor='white', markersize=9, label=n)
           for _, _, _, n, m, c in EST]
handles.append(plt.Line2D([0], [0], color=PALETTE[6], lw=1.8,
                          label='No effect (zero line)'))
handles.append(plt.Line2D([0], [0], color=PALETTE[0], lw=1.4, ls='--',
                          label='True ATE (do-calculus)'))
axes[0].legend(handles=handles, loc='lower right', fontsize=8, ncol=2)

fig.suptitle('Average Treatment Effect (ATE) Estimates vs. Causal Ground Truth\n'
             'Naive interval carries the wrong sign; adjustment recovers the protective effect',
             fontsize=12, fontweight='bold')

OUT = __import__('pathlib').Path(__file__).resolve().parent
save_fig(fig, 'fig02_intervention_effects', OUT)
plt.close(fig)
