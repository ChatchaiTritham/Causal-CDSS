#!/usr/bin/env python3
"""
Figure 4: Confounding Bias Quantification — Grouped Bar Chart
KAIS Causal Models for CDSS — Springer submission

Compares naive, LSTM, and causally adjusted (DR) ATE estimates
across three clinical domains.
Data source: Tables 2 and 4 of the manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- Data from Tables 2 & 4 ----------
domains = ['Sepsis\n(Early Antibiotics)', 'ARDS\n(Low Tidal Volume)', 'ACS\n(Early Reperfusion)']

# Values in percentage points (negative = mortality reduction)
naive_ate   = [-3.5, -5.2, -9.2]
lstm_ate    = [-14.1, -8.9, -17.5]
causal_ate  = [-15.2, -9.8, -18.6]

# 95% bootstrap CI (half-widths)
naive_err   = [1.8, 1.2, 2.5]
lstm_err    = [2.2, 1.5, 2.8]
causal_err  = [1.5, 1.0, 2.0]

# Bias annotations (naive vs causal)
bias_pp = [11.7, 4.6, 9.4]
bias_colors = ['#2471A3', '#E67E22', '#C0392B']

# ---------- Build figure ----------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

x = np.arange(len(domains))
width = 0.24

# Bar styles
bar_naive = ax.bar(x - width, naive_ate, width, yerr=naive_err,
                   color='#BDC3C7', edgecolor='#7F8C8D', linewidth=0.8,
                   capsize=4, error_kw={'linewidth': 1.2},
                   label='Naive (no adjustment)', zorder=3)
bar_lstm = ax.bar(x, lstm_ate, width, yerr=lstm_err,
                  color='#5DADE2', edgecolor='#2E86C1', linewidth=0.8,
                  hatch='///', capsize=4, error_kw={'linewidth': 1.2},
                  label='LSTM CDSS estimate', zorder=3)
bar_causal = ax.bar(x + width, causal_ate, width, yerr=causal_err,
                    color='#1A5276', edgecolor='#0E3550', linewidth=0.8,
                    capsize=4, error_kw={'linewidth': 1.2},
                    label='Causal (DR-adjusted)', zorder=3)

# Annotate naive bar tops with values
for i, (val, bar) in enumerate(zip(naive_ate, bar_naive)):
    ax.text(bar.get_x() + bar.get_width() / 2, val - naive_err[i] - 0.6,
            f'{val}%', ha='center', va='top', fontsize=8, color='#555555')

# Annotate LSTM bar tops
for i, (val, bar) in enumerate(zip(lstm_ate, bar_lstm)):
    ax.text(bar.get_x() + bar.get_width() / 2, val - lstm_err[i] - 0.6,
            f'{val}%', ha='center', va='top', fontsize=8, color='#1A5276',
            fontweight='bold')

# Annotate causal bar tops
for i, (val, bar) in enumerate(zip(causal_ate, bar_causal)):
    ax.text(bar.get_x() + bar.get_width() / 2, val - causal_err[i] - 0.6,
            f'{val}%', ha='center', va='top', fontsize=8, color='#0E3550',
            fontweight='bold')

# Bias arrows (between naive and causal)
for i in range(len(domains)):
    arrow_x = x[i] + width + 0.42
    y_start = naive_ate[i]
    y_end = causal_ate[i]
    ax.annotate('', xy=(arrow_x, y_end), xytext=(arrow_x, y_start),
                arrowprops=dict(arrowstyle='<->', color=bias_colors[i],
                                lw=1.8, shrinkA=2, shrinkB=2))
    mid_y = (y_start + y_end) / 2
    ax.text(arrow_x + 0.12, mid_y,
            f'+{bias_pp[i]} pp\nbias',
            fontsize=7.5, color=bias_colors[i], fontweight='bold',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=bias_colors[i], alpha=0.85, linewidth=0.8))

# Reference line at 0
ax.axhline(0, color='#333333', linewidth=0.8, zorder=1)

# Title and labels
ax.set_title('Confounding Bias Magnitude: Naive vs. CDSS vs. Causal Estimates\n'
             'Across Three Critical Care Domains',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylabel('Estimated Mortality Reduction (%)', fontsize=10.5)
ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=10)
ax.set_ylim(-23, 3)
ax.tick_params(axis='y', labelsize=9.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Bottom annotation
ax.text(0.01, -0.11,
        'Naive estimates underestimate true treatment benefit by 47–77%  |  '
        'Confounding bias: +4.6 to +11.7 percentage points',
        transform=ax.transAxes, fontsize=8, color='#555555', style='italic')

# Legend
ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='#CCCCCC')

plt.tight_layout()
plt.savefig('fig04_confounding_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('fig04_confounding_analysis.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: fig04_confounding_analysis.png / .pdf")
plt.close()
