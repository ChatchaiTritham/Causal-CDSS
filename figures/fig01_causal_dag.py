#!/usr/bin/env python3
"""
Figure 1: Causal Directed Acyclic Graphs (DAGs) for Three Critical Care Domains
KAIS Causal Models for CDSS — Springer submission

Three-panel SCM schematic for:
  (A) Sepsis — Antibiotic Timing
  (B) ARDS — Ventilation Strategy
  (C) ACS — Reperfusion Timing

Node shapes: rectangle=treatment, double-circle=outcome, circle=confounder/mediator
Arrow styles: solid blue=causal, dashed orange=confounding (backdoor)

This is a structural schematic (no numeric data); it shares the canonical
publication palette/fonts from figures/pub_style.py so it matches every other
figure in the portfolio.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from pub_style import (apply_pub_style, save_fig,
                       C_TREATMENT, C_OUTCOME, C_CONFOUNDER, C_MEDIATOR)

apply_pub_style()

# ─── Palette (semantic, drawn from the Okabe-Ito canon) ───
COL_TREAT   = C_TREATMENT      # treatment edge/fill (blue)
COL_OUTCOME = C_OUTCOME        # outcome (vermillion)
COL_CONF    = C_CONFOUNDER     # confounder (orange)
COL_MED     = C_MEDIATOR       # mediator (bluish green)
COL_CAUSAL  = C_TREATMENT      # causal arrows (blue)
COL_CONFND  = C_CONFOUNDER     # confounding arrows (orange, dashed)


def _lighten(hex_color, f=0.55):
    """Blend a hex colour toward white for soft node fills."""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02X}{g:02X}{b:02X}"


def draw_rect(ax, cx, cy, w, h, text, edge, fontsize=8.5):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.06", facecolor=_lighten(edge),
                         edgecolor=edge, linewidth=1.6, zorder=4)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#1F2937', zorder=5)


def draw_circle(ax, cx, cy, r, text, edge, fontsize=8, double=False):
    fill = _lighten(edge)
    if double:
        ax.add_patch(Circle((cx, cy), r * 1.15, facecolor=fill, edgecolor=edge,
                            linewidth=1.6, zorder=4))
        ax.add_patch(Circle((cx, cy), r * 0.92, facecolor=fill, edgecolor='white',
                            linewidth=1.8, zorder=4.5))
    else:
        ax.add_patch(Circle((cx, cy), r, facecolor=fill, edgecolor=edge,
                            linewidth=1.4, zorder=4))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#1F2937', zorder=5, linespacing=1.15)


def draw_arrow(ax, x1, y1, x2, y2, color, dashed=False, lw=1.8):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                            mutation_scale=14, color=color, linewidth=lw,
                            connectionstyle='arc3,rad=0.08', zorder=3)
    if dashed:
        arrow.set_linestyle((0, (6, 3)))
    ax.add_patch(arrow)


# ─── Figure layout (constrained_layout from rcParams) ───
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
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

for ax, dag, title, ann in zip(axes, dags, titles, annotations):
    ax.set_xlim(-0.5, 10.2)
    ax.set_ylim(-0.8, 10.0)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(4.8, 9.6, title, ha='center', va='top', fontsize=11.5,
            fontweight='bold')

    T = dag['T']
    draw_rect(ax, *T['pos'], T['w'], T['h'], T['label'], COL_TREAT, fontsize=8.5)
    M = dag['M']
    draw_circle(ax, *M['pos'], M['r'], M['label'], COL_MED, fontsize=8)
    Y = dag['Y']
    draw_circle(ax, *Y['pos'], Y['r'], Y['label'], COL_OUTCOME, fontsize=8, double=True)
    for z in dag['Z']:
        draw_circle(ax, *z['pos'], z['r'], z['label'], COL_CONF, fontsize=7.5)

    tx, ty = T['pos']; mx, my = M['pos']; yx, yy = Y['pos']
    draw_arrow(ax, tx + T['w'] / 2 + 0.1, ty, mx - M['r'] - 0.08, my, COL_CAUSAL)
    draw_arrow(ax, mx + M['r'] + 0.08, my, yx - Y['r'] * 1.15 - 0.08, yy, COL_CAUSAL)

    for z in dag['Z']:
        zx, zy = z['pos']; zr = z['r']
        if zy > ty:
            draw_arrow(ax, zx, zy - zr - 0.05, tx + T['w'] / 4, ty + T['h'] / 2 + 0.05,
                       COL_CONFND, dashed=True, lw=1.3)
        else:
            draw_arrow(ax, zx, zy + zr + 0.05, tx + T['w'] / 4, ty - T['h'] / 2 - 0.05,
                       COL_CONFND, dashed=True, lw=1.3)
        if zy > yy:
            draw_arrow(ax, zx + zr * 0.5, zy - zr - 0.05, yx - Y['r'] * 0.5,
                       yy + Y['r'] * 1.15 + 0.05, COL_CONFND, dashed=True, lw=1.3)
        else:
            draw_arrow(ax, zx + zr * 0.5, zy + zr + 0.05, yx - Y['r'] * 0.5,
                       yy - Y['r'] * 1.15 - 0.05, COL_CONFND, dashed=True, lw=1.3)

    ax.text(4.8, -0.5, ann, ha='center', va='top', fontsize=7.5,
            fontstyle='italic', color='#6B7280')

# ─── Legend (below all panels) ───
legend_handles = [
    FancyBboxPatch((0, 0), 0.18, 0.10, boxstyle="round,pad=0.02",
                   facecolor=_lighten(COL_TREAT), edgecolor=COL_TREAT, lw=1.2),
    Circle((0, 0), 0.06, facecolor=_lighten(COL_OUTCOME), edgecolor=COL_OUTCOME, lw=1.2),
    Circle((0, 0), 0.06, facecolor=_lighten(COL_CONF), edgecolor=COL_CONF, lw=1.2),
    Circle((0, 0), 0.06, facecolor=_lighten(COL_MED), edgecolor=COL_MED, lw=1.2),
    Line2D([0], [0], color=COL_CAUSAL, lw=2.0, ls='-'),
    Line2D([0], [0], color=COL_CONFND, lw=1.5, ls='--'),
]
legend_labels = [
    'Treatment (T) — rectangle',
    'Outcome (Y) — double circle',
    'Confounder (Z) — circle',
    'Mediator (M) — circle',
    'Causal path',
    'Backdoor (confounding) path',
]
fig.legend(legend_handles, legend_labels, loc='lower center', ncol=6, fontsize=8.5,
           frameon=False, handlelength=1.8, columnspacing=1.5,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Causal Directed Acyclic Graphs (DAGs) for Three Critical Care Treatment Pathways',
             fontsize=13, fontweight='bold')

OUT = __import__('pathlib').Path(__file__).resolve().parent
save_fig(fig, OUT, 'fig01_causal_dag')
plt.close(fig)
