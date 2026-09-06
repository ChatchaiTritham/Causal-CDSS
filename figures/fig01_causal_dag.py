#!/usr/bin/env python3
"""Figure 1: the 4-node causal DAG actually declared in run_all.py build_scm().

Nodes: age, severity, treatment, outcome.
Edges: age->severity, age->outcome, severity->treatment, severity->outcome,
       treatment->outcome.
Minimal backdoor adjustment set: {severity}.
Panels differ only in the clinical reading of T (and the coefficients, Table S2).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from pubviz import apply_pub_style, C_TREATMENT, C_OUTCOME, C_CONFOUNDER, C_NEUTRAL

apply_pub_style()
COL_TREAT, COL_OUT, COL_CONF = C_TREATMENT, C_OUTCOME, C_CONFOUNDER
COL_CAUSAL, COL_CONFND = C_TREATMENT, C_CONFOUNDER


def _lighten(h, f=0.55):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02X}{g:02X}{b:02X}"


def draw_rect(ax, cx, cy, w, h, text, edge, fs=8.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.06", facecolor=_lighten(edge),
                                edgecolor=edge, linewidth=1.6, zorder=4))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#1F2937', zorder=5, linespacing=1.15)


def draw_circle(ax, cx, cy, r, text, edge, fs=8, double=False):
    fill = _lighten(edge)
    if double:
        ax.add_patch(Circle((cx, cy), r * 1.15, facecolor=fill, edgecolor=edge,
                            linewidth=1.6, zorder=4))
        ax.add_patch(Circle((cx, cy), r * 0.92, facecolor=fill, edgecolor='white',
                            linewidth=1.8, zorder=4.5))
    else:
        ax.add_patch(Circle((cx, cy), r, facecolor=fill, edgecolor=edge,
                            linewidth=1.4, zorder=4))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#1F2937', zorder=5, linespacing=1.15)


def draw_arrow(ax, x1, y1, x2, y2, color, dashed=False, lw=1.8, rad=0.08):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=14,
                        color=color, linewidth=lw,
                        connectionstyle=f'arc3,rad={rad}', zorder=3)
    if dashed:
        a.set_linestyle((0, (6, 3)))
    ax.add_patch(a)


fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
fig.patch.set_facecolor('white')

panels = [
    ('(A)  Sepsis: Early Antibiotics',
     'Early\nAntibiotics',
     'T = early antibiotic administration   |   Y = in-hospital mortality (binary)'),
    ('(B)  ARDS: Low Tidal Volume',
     'Low Tidal\nVolume',
     'T = low tidal volume ventilation   |   Y = in-hospital mortality (binary)'),
    ('(C)  ACS: Early Reperfusion',
     'Early\nReperfusion',
     'T = early reperfusion   |   Y = in-hospital mortality (binary)'),
]

# node geometry (shared by all three panels)
AGE = (1.6, 8.0, 0.62)
SEV = (4.6, 6.6, 0.80)
T_POS = (2.2, 3.0, 1.9, 0.95)
Y_POS = (7.6, 3.0, 0.85)

for ax, (title, tlabel, ann) in zip(axes, panels):
    ax.set_xlim(-0.3, 9.6)
    ax.set_ylim(0.2, 10.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(4.7, 9.8, title, ha='center', va='top', fontsize=11.5, fontweight='bold')

    draw_circle(ax, AGE[0], AGE[1], AGE[2], 'Age', COL_CONF, fs=8)
    draw_circle(ax, SEV[0], SEV[1], SEV[2], 'Severity', COL_CONF, fs=8)
    draw_rect(ax, T_POS[0], T_POS[1], T_POS[2], T_POS[3], tlabel, COL_TREAT)
    draw_circle(ax, Y_POS[0], Y_POS[1], Y_POS[2], 'Mortality', COL_OUT, fs=8, double=True)

    ax_, ay_, ar_ = AGE
    sx, sy, sr = SEV
    tx, ty, tw, th = T_POS
    yx, yy, yr = Y_POS

    # age -> severity (causal, within the confounding structure)
    draw_arrow(ax, ax_ + ar_ * 0.8, ay_ - ar_ * 0.5, sx - sr - 0.05, sy + sr * 0.4,
               COL_CONFND, dashed=True, lw=1.4)
    # severity -> treatment (backdoor)
    draw_arrow(ax, sx - sr * 0.7, sy - sr - 0.05, tx + tw * 0.25, ty + th / 2 + 0.05,
               COL_CONFND, dashed=True, lw=1.4)
    # severity -> outcome (backdoor)
    draw_arrow(ax, sx + sr * 0.7, sy - sr - 0.05, yx - yr * 0.6, yy + yr * 1.15 + 0.05,
               COL_CONFND, dashed=True, lw=1.4)
    # age -> outcome
    draw_arrow(ax, ax_ + ar_ * 0.5, ay_ + ar_ * 0.6, yx + yr * 0.4, yy + yr * 1.35,
               COL_CONFND, dashed=True, lw=1.4, rad=-0.32)
    # treatment -> outcome (the causal effect of interest)
    draw_arrow(ax, tx + tw / 2 + 0.08, ty, yx - yr * 1.15 - 0.08, yy, COL_CAUSAL, lw=2.2)

    ax.text(4.9, 1.35, ann, ha='center', va='top', fontsize=7.5,
            fontstyle='italic', color='#6B7280')
    ax.text(4.9, 0.75, r'Minimal backdoor adjustment set  Z = {Severity}',
            ha='center', va='top', fontsize=7.5, color=C_NEUTRAL)

legend_handles = [
    FancyBboxPatch((0, 0), 0.18, 0.10, boxstyle="round,pad=0.02",
                   facecolor=_lighten(COL_TREAT), edgecolor=COL_TREAT, lw=1.2),
    Circle((0, 0), 0.06, facecolor=_lighten(COL_OUT), edgecolor=COL_OUT, lw=1.2),
    Circle((0, 0), 0.06, facecolor=_lighten(COL_CONF), edgecolor=COL_CONF, lw=1.2),
    Line2D([0], [0], color=COL_CAUSAL, lw=2.2, ls='-'),
    Line2D([0], [0], color=COL_CONFND, lw=1.5, ls='--'),
]
legend_labels = [
    'Treatment (T) — rectangle',
    'Outcome (Y) — double circle',
    'Baseline covariate — circle',
    'Treatment effect path',
    'Backdoor (confounding) path',
]
fig.legend(legend_handles, legend_labels, loc='lower center', ncol=5, fontsize=8.5,
           frameon=False, handlelength=1.8, columnspacing=1.5,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Causal DAG for the Three Critical Care Treatment Domains '
             '(identical structure, domain-specific coefficients)',
             fontsize=13, fontweight='bold')

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig01_causal_dag.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("ok")
