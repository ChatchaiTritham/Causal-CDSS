#!/usr/bin/env python3
"""
Figure 2: Average Treatment Effect (ATE) Forest Plot
KAIS Causal Models for CDSS — Springer submission

Generates a forest plot comparing ATE estimates from five CDSS models
against ground-truth causal effects across three clinical domains.
Data source: Table 2 of the manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- Data from Table 2 ----------

domains = {
    'Sepsis — Early Antibiotics (<3 h)': {
        'true_ate': -15.2,
        'naive_ate': -3.5,
        'bg_color': '#D6EAF8',
        'label_bg': '#2980B9',
        'models': {
            'LSTM':        {'est': -14.1, 'ci_lo': -19.5, 'ci_hi': -8.7},
            'TCN':         {'est': -13.0, 'ci_lo': -19.0, 'ci_hi': -7.0},
            'XGBoost':     {'est': -11.5, 'ci_lo': -17.5, 'ci_hi': -5.5},
            'Random Forest': {'est': -10.2, 'ci_lo': -16.8, 'ci_hi': -3.6},
            'Log. Regression': {'est': -8.5, 'ci_lo': -15.5, 'ci_hi': -1.5},
        }
    },
    'ARDS — Low Tidal Volume (6 mL/kg)': {
        'true_ate': -9.8,
        'naive_ate': -5.2,
        'bg_color': '#FDEBD0',
        'label_bg': '#E67E22',
        'models': {
            'LSTM':        {'est': -8.9, 'ci_lo': -13.0, 'ci_hi': -4.8},
            'TCN':         {'est': -8.2, 'ci_lo': -12.5, 'ci_hi': -3.9},
            'XGBoost':     {'est': -7.5, 'ci_lo': -12.0, 'ci_hi': -3.0},
            'Random Forest': {'est': -6.8, 'ci_lo': -11.5, 'ci_hi': -2.1},
            'Log. Regression': {'est': -5.8, 'ci_lo': -10.8, 'ci_hi': -0.8},
        }
    },
    'ACS — Early Reperfusion (<90 min)': {
        'true_ate': -18.6,
        'naive_ate': -9.2,
        'bg_color': '#D5F5E3',
        'label_bg': '#27AE60',
        'models': {
            'LSTM':        {'est': -17.5, 'ci_lo': -24.0, 'ci_hi': -11.0},
            'TCN':         {'est': -16.2, 'ci_lo': -23.0, 'ci_hi': -9.4},
            'XGBoost':     {'est': -14.5, 'ci_lo': -21.5, 'ci_hi': -7.5},
            'Random Forest': {'est': -12.8, 'ci_lo': -20.0, 'ci_hi': -5.6},
            'Log. Regression': {'est': -11.0, 'ci_lo': -18.5, 'ci_hi': -3.5},
        }
    }
}

# ---------- Marker styles per model ----------
markers = {
    'LSTM':            {'marker': 'o', 'color': '#2C3E88', 'size': 9},
    'TCN':             {'marker': 's', 'color': '#3C78D8', 'size': 9},
    'XGBoost':         {'marker': 'D', 'color': '#7B3FA0', 'size': 8},
    'Random Forest':   {'marker': '^', 'color': '#0E8A6E', 'size': 9},
    'Log. Regression': {'marker': 'v', 'color': '#C0392B', 'size': 9},
}

# ---------- Build figure ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), dpi=300)
fig.suptitle(
    'Average Treatment Effect (ATE) Estimates: Five CDSS Models vs. Causal Ground Truth',
    fontsize=13, fontweight='bold', y=0.97
)

model_names = list(markers.keys())
n_models = len(model_names)

for ax_idx, (domain_name, d) in enumerate(domains.items()):
    ax = axes[ax_idx]

    # Background shading
    ax.axvspan(-30, 5, color=d['bg_color'], alpha=0.35, zorder=0)

    # Reference lines
    ax.axvline(d['true_ate'], color='#1A3C6E', linewidth=2.0, linestyle='-',
               zorder=2, label='True ATE (DR-adjusted)')
    ax.axvline(d['naive_ate'], color='#888888', linewidth=1.5, linestyle='--',
               zorder=2, label='Naive (unadjusted)')

    # Annotate reference lines
    ax.text(d['true_ate'] + 0.3, n_models + 0.25,
            f"True: {d['true_ate']}%",
            fontsize=8.5, color='#1A3C6E', fontweight='bold', va='bottom')
    ax.text(d['naive_ate'] + 0.3, n_models + 0.25,
            f"Naive: {d['naive_ate']}%",
            fontsize=8.5, color='#888888', va='bottom')

    # Plot each model
    for i, model in enumerate(model_names):
        md = d['models'][model]
        ms = markers[model]
        y_pos = n_models - 1 - i

        # Confidence interval line
        ax.plot([md['ci_lo'], md['ci_hi']], [y_pos, y_pos],
                color=ms['color'], linewidth=1.8, zorder=3, solid_capstyle='round')
        # CI caps
        ax.plot([md['ci_lo'], md['ci_lo']], [y_pos - 0.12, y_pos + 0.12],
                color=ms['color'], linewidth=1.5, zorder=3)
        ax.plot([md['ci_hi'], md['ci_hi']], [y_pos - 0.12, y_pos + 0.12],
                color=ms['color'], linewidth=1.5, zorder=3)
        # Point estimate
        ax.scatter(md['est'], y_pos, marker=ms['marker'], s=ms['size']**2,
                   color=ms['color'], edgecolors='white', linewidths=0.5, zorder=4)

    # Domain label
    ax.text(0.02, 0.92, f'({chr(65 + ax_idx)})',
            transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor=d['label_bg'],
                      edgecolor='none', alpha=0.15)
    ax.text(0.05, 0.92, domain_name,
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', bbox=bbox_props)

    # Axes
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(list(reversed(model_names)), fontsize=9.5)
    ax.set_xlim(-30, 5)
    ax.set_ylim(-0.6, n_models + 0.5)
    ax.set_xlabel('Estimated Mortality Reduction (percentage points)', fontsize=9.5)
    ax.tick_params(axis='x', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend only on first panel
    if ax_idx == 0:
        line_handles = [
            plt.Line2D([0], [0], color='#1A3C6E', lw=2, ls='-', label='True ATE (DR-adjusted)'),
            plt.Line2D([0], [0], color='#888888', lw=1.5, ls='--', label='Naive (unadjusted)'),
        ]
        marker_handles = [
            plt.Line2D([0], [0], marker=markers[m]['marker'], color='w',
                       markerfacecolor=markers[m]['color'],
                       markersize=markers[m]['size'], label=m)
            for m in model_names
        ]
        ax.legend(handles=line_handles + marker_handles,
                  loc='center right', fontsize=7.5, framealpha=0.9,
                  edgecolor='#CCCCCC', ncol=1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig02_intervention_effects.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('fig02_intervention_effects.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: fig02_intervention_effects.png / .pdf")
plt.close()
