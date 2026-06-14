#!/usr/bin/env python3
"""
Figure 1: Causal Directed Acyclic Graphs (DAGs) for Three Critical Care Domains
KAIS Causal Models for CDSS — Springer submission

Three-panel figure showing domain-specific SCMs for:
  (A) Sepsis — Antibiotic Timing
  (B) ARDS — Ventilation Strategy
  (C) ACS — Reperfusion Timing

Node shapes: rectangle=treatment, double-circle=outcome, circle=confounder/mediator
Arrow styles: solid blue=causal, dashed red=confounding (backdoor)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import numpy as np

# ─── Colour Palette ───
COL_TREAT   = '#1A5276'   # dark blue — treatment
COL_TREAT_F = '#2980B9'   # fill
COL_OUTCOME = '#C0392B'   # dark red — outcome
COL_OUT_F   = '#E74C3C'
COL_CONF    = '#7D6608'   # dark gold — confounder
COL_CONF_F  = '#F1C40F'
COL_MED     = '#1E8449'   # dark green — mediator
COL_MED_F   = '#27AE60'
COL_CAUSAL  = '#2471A3'   # blue arrows
COL_CONFND  = '#E74C3C'   # red dashed arrows
COL_BG      = '#FAFCFE'

def draw_rect(ax, cx, cy, w, h, text, fill, edge, fontsize=8.5):
    """Treatment node (rounded rectangle)."""
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.06", facecolor=fill,
                          edgecolor=edge, linewidth=1.6, zorder=4)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=5)

def draw_circle(ax, cx, cy, r, text, fill, edge, fontsize=8, double=False):
    """Confounder/mediator (circle) or outcome (double circle)."""
    if double:
        outer = Circle((cx, cy), r * 1.15, facecolor=fill, edgecolor=edge,
                        linewidth=1.6, zorder=4)
        ax.add_patch(outer)
        inner = Circle((cx, cy), r * 0.92, facecolor=fill, edgecolor='white',
                        linewidth=1.8, zorder=4.5)
        ax.add_patch(inner)
    else:
        c = Circle((cx, cy), r, facecolor=fill, edgecolor=edge,
                    linewidth=1.4, zorder=4)
        ax.add_patch(c)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white' if fill not in [COL_CONF_F] else '#333',
            zorder=5, linespacing=1.15)

def draw_arrow(ax, x1, y1, x2, y2, color, dashed=False, lw=1.8):
    """Directed edge."""
    style = 'Simple,tail_width=0.6,head_width=6,head_length=5'
    ls = '--' if dashed else '-'
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', mutation_scale=14,
                             color=color, linewidth=lw, linestyle=ls,
                             connectionstyle='arc3,rad=0.08',
                             zorder=3)
    if dashed:
        arrow.set_linestyle((0, (6, 3)))
    ax.add_patch(arrow)

# ─── Figure Layout ───
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
fig.patch.set_facecolor('white')

titles = [
    '(A)  Sepsis: Antibiotic Timing',
    '(B)  ARDS: Ventilation Strategy',
    '(C)  ACS: Reperfusion Timing',
]
annotations = [
    'T = Early Antibiotics (<3 h)   |   Y = 28-day Mortality',
    'T = Low Tidal Volume (6 mL/kg)   |   Y = Ventilator-Free Days',
    'T = Door-to-Balloon (<90 min)   |   Y = 30-day MACE',
]

# ─── DAG Definitions ───
# Each: treatment, mediator, outcome, confounders
# Positions normalised to [0,10] x [0,10]

dags = [
    {   # SEPSIS
        'T':  {'pos': (1.2, 4.5), 'label': 'Early\nAntibiotics', 'w': 1.8, 'h': 0.9},
        'M':  {'pos': (4.5, 4.5), 'label': 'Pathogen\nClearance', 'r': 0.72},
        'Y':  {'pos': (8.0, 4.5), 'label': '28-day\nMortality', 'r': 0.78},
        'Z': [
            {'pos': (2.0, 8.0), 'label': 'SOFA\nScore',     'r': 0.65},
            {'pos': (4.8, 8.0), 'label': 'Age',              'r': 0.52},
            {'pos': (1.5, 1.2), 'label': 'Comor-\nbidity',   'r': 0.62},
            {'pos': (4.5, 1.2), 'label': 'Infection\nSource', 'r': 0.62},
        ],
    },
    {   # ARDS
        'T':  {'pos': (1.2, 4.5), 'label': 'Low Tidal\nVolume', 'w': 1.8, 'h': 0.9},
        'M':  {'pos': (4.5, 4.5), 'label': 'Alveolar\nRecruitment', 'r': 0.72},
        'Y':  {'pos': (8.0, 4.5), 'label': 'Ventilator-\nFree Days', 'r': 0.78},
        'Z': [
            {'pos': (2.2, 8.0), 'label': 'P/F\nRatio',        'r': 0.58},
            {'pos': (5.0, 8.0), 'label': 'Baseline\nSeverity', 'r': 0.62},
            {'pos': (2.8, 1.2), 'label': 'ARDS\nAetiology',    'r': 0.62},
        ],
    },
    {   # ACS
        'T':  {'pos': (1.2, 4.5), 'label': 'Time-to-\nReperfusion', 'w': 1.8, 'h': 0.9},
        'M':  {'pos': (4.5, 4.5), 'label': 'Infarct\nSize', 'r': 0.68},
        'Y':  {'pos': (8.0, 4.5), 'label': '30-day\nMortality', 'r': 0.78},
        'Z': [
            {'pos': (2.0, 8.0), 'label': 'Symptom\nDuration', 'r': 0.62},
            {'pos': (5.2, 8.0), 'label': 'Infarct\nTerritory', 'r': 0.62},
            {'pos': (2.8, 1.2), 'label': 'Killip\nClass',      'r': 0.58},
        ],
    },
]

for ax_i, (ax, dag, title, ann) in enumerate(zip(axes, dags, titles, annotations)):
    ax.set_xlim(-0.5, 10.2)
    ax.set_ylim(-0.8, 10.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4.8, 9.6, title, ha='center', va='top', fontsize=11.5,
            fontweight='bold', fontfamily='sans-serif')

    # Draw nodes
    T = dag['T']
    draw_rect(ax, *T['pos'], T['w'], T['h'], T['label'],
              COL_TREAT_F, COL_TREAT, fontsize=8.5)

    M = dag['M']
    draw_circle(ax, *M['pos'], M['r'], M['label'], COL_MED_F, COL_MED, fontsize=8)

    Y = dag['Y']
    draw_circle(ax, *Y['pos'], Y['r'], Y['label'], COL_OUT_F, COL_OUTCOME,
                fontsize=8, double=True)

    for z in dag['Z']:
        draw_circle(ax, *z['pos'], z['r'], z['label'], COL_CONF_F, COL_CONF, fontsize=7.5)

    # Causal arrows: T → M → Y (blue solid)
    tx, ty = T['pos']
    mx, my = M['pos']
    yx, yy = Y['pos']
    draw_arrow(ax, tx + T['w']/2 + 0.1, ty, mx - M['r'] - 0.08, my, COL_CAUSAL)
    draw_arrow(ax, mx + M['r'] + 0.08, my, yx - Y['r']*1.15 - 0.08, yy, COL_CAUSAL)

    # Confounding arrows: each Z → T and Z → Y (red dashed)
    for z in dag['Z']:
        zx, zy = z['pos']
        zr = z['r']
        # Z → T
        if zy > ty:
            draw_arrow(ax, zx, zy - zr - 0.05, tx + T['w']/4, ty + T['h']/2 + 0.05,
                       COL_CONFND, dashed=True, lw=1.3)
        else:
            draw_arrow(ax, zx, zy + zr + 0.05, tx + T['w']/4, ty - T['h']/2 - 0.05,
                       COL_CONFND, dashed=True, lw=1.3)
        # Z → Y
        if zy > yy:
            draw_arrow(ax, zx + zr * 0.5, zy - zr - 0.05,
                       yx - Y['r']*0.5, yy + Y['r']*1.15 + 0.05,
                       COL_CONFND, dashed=True, lw=1.3)
        else:
            draw_arrow(ax, zx + zr * 0.5, zy + zr + 0.05,
                       yx - Y['r']*0.5, yy - Y['r']*1.15 - 0.05,
                       COL_CONFND, dashed=True, lw=1.3)

    # Annotation
    ax.text(4.8, -0.5, ann, ha='center', va='top', fontsize=7.5,
            fontstyle='italic', color='#666666')

# ─── Legend (below all panels) ───
legend_y = -0.10
legend_items = [
    (FancyBboxPatch((0, 0), 0.18, 0.10, boxstyle="round,pad=0.02",
                     facecolor=COL_TREAT_F, edgecolor=COL_TREAT, lw=1.2),
     'Treatment (T) — rectangle'),
    (Circle((0, 0), 0.06, facecolor=COL_OUT_F, edgecolor=COL_OUTCOME, lw=1.2),
     'Outcome (Y) — double circle'),
    (Circle((0, 0), 0.06, facecolor=COL_CONF_F, edgecolor=COL_CONF, lw=1.2),
     'Confounder (Z) — circle'),
    (Circle((0, 0), 0.06, facecolor=COL_MED_F, edgecolor=COL_MED, lw=1.2),
     'Mediator (M) — circle'),
]

from matplotlib.lines import Line2D
arrow_handles = [
    Line2D([0], [0], color=COL_CAUSAL, lw=2.0, ls='-', label='Causal path'),
    Line2D([0], [0], color=COL_CONFND, lw=1.5, ls='--', label='Backdoor (confounding) path'),
]

handles = [item[0] for item in legend_items] + arrow_handles
labels  = [item[1] for item in legend_items] + [h.get_label() for h in arrow_handles]

fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8.5,
           frameon=True, edgecolor='#CCCCCC', fancybox=True,
           bbox_to_anchor=(0.5, -0.02), handlelength=1.8, columnspacing=1.5)

fig.suptitle('Causal Directed Acyclic Graphs (DAGs) for Three Critical Care Treatment Pathways',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig('fig01_causal_dag.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig('fig01_causal_dag.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
print("Saved: fig01_causal_dag.png / .pdf")
plt.close()
