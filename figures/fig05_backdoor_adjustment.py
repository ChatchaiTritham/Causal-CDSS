#!/usr/bin/env python3
"""
Figure 5: Backdoor Criterion for Confounding Control (Sepsis Domain)
KAIS Causal Models for CDSS — Springer submission

Three panels:
  (A) Original DAG with active confounding paths (dashed orange).
  (B) DAG after conditioning on Z = {SOFA, Age, Comorbidity, Inf. Source}.
  (C) Numerical demonstration: naive vs. severity-stratified vs. adjusted ATE.

Panels A/B are structural schematics restyled to the canonical palette.
Panel C numbers are loaded from results/ (ate_by_domain.csv +
cate_subgroups.csv); NOTHING is hardcoded.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from pubviz import (apply_pub_style, save_fig, results_dir, PALETTE,
                    C_TREATMENT, C_OUTCOME, C_CONFOUNDER, C_MEDIATOR)

apply_pub_style()

DOMAIN = "sepsis"
SEV_ORDER = ["sev_low", "sev_mid", "sev_high"]
SEV_LABELS = {"sev_low": "qSOFA 0–1", "sev_mid": "qSOFA 2", "sev_high": "qSOFA 3"}

# ---------- Load data ----------
ate = pd.read_csv(results_dir() / "ate_by_domain.csv")
row = ate[ate["domain"] == DOMAIN].iloc[0]
naive_pp = row["naive_ate"] * 100.0
adj_pp = row["doubly_robust_ate"] * 100.0       # adjusted = doubly-robust
true_pp = row["true_ate"] * 100.0
bias_pp = row["confounding_bias"] * 100.0

cate = pd.read_csv(results_dir() / "cate_subgroups.csv")
cs = cate[cate["domain"] == DOMAIN]
# severity-stratified true CATE, averaged over age (n-weighted)
strat = {}
for sev in SEV_ORDER:
    sub = cs[cs["severity_group"] == sev]
    strat[sev] = np.average(sub["true_cate"], weights=sub["n_obs"]) * 100.0

# ─── Palette aliases ───
COL_TREAT  = C_TREATMENT
COL_OUT    = C_OUTCOME
COL_CONF   = C_CONFOUNDER
COL_MED    = C_MEDIATOR
COL_CAUSAL = C_TREATMENT
COL_CONFND = C_CONFOUNDER
COL_BLOCK  = "#BBBBBB"
COL_COND   = "#CFCFCF"


def _lighten(hex_color, f=0.55):
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02X}{g:02X}{b:02X}"


def draw_rect(ax, cx, cy, w, h, text, edge, fontsize=8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.06", facecolor=_lighten(edge),
                                edgecolor=edge, linewidth=1.6, zorder=4))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#1F2937', zorder=5)


def draw_circle(ax, cx, cy, r, text, edge, fontsize=7.5, double=False, fill=None):
    fill = _lighten(edge) if fill is None else fill
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


def draw_arrow(ax, x1, y1, x2, y2, color, dashed=False, lw=1.6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=13,
                        color=color, linewidth=lw, connectionstyle='arc3,rad=0.10',
                        zorder=3)
    if dashed:
        a.set_linestyle((0, (5, 3)))
    ax.add_patch(a)


# ─── Figure ───
fig = plt.figure(figsize=(16, 5.0))
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(1, 5, wspace=0.35)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4])

dag_nodes = {
    'T':  (1.2, 4.5, 1.7, 0.85, 'Early\nAntibiotics'),
    'M':  (4.3, 4.5, 0.68, 'Pathogen\nClearance'),
    'Y':  (7.5, 4.5, 0.72, '28-day\nMortality'),
    'Z1': (2.0, 8.0, 0.60, 'SOFA\nScore'),
    'Z2': (4.8, 8.0, 0.48, 'Age'),
    'Z3': (1.5, 1.2, 0.58, 'Comor-\nbidity'),
    'Z4': (4.3, 1.2, 0.58, 'Infection\nSource'),
}


def draw_dag_panel(ax, title, conditioned=False):
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 9.8)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')
    n = dag_nodes
    draw_rect(ax, n['T'][0], n['T'][1], n['T'][2], n['T'][3], n['T'][4], COL_TREAT)
    draw_circle(ax, n['M'][0], n['M'][1], n['M'][2], n['M'][3], COL_MED)
    draw_circle(ax, n['Y'][0], n['Y'][1], n['Y'][2], n['Y'][3], COL_OUT, double=True)
    cfill = COL_COND if conditioned else _lighten(COL_CONF)
    cedge = "#999999" if conditioned else COL_CONF
    for key in ['Z1', 'Z2', 'Z3', 'Z4']:
        nd = n[key]
        draw_circle(ax, nd[0], nd[1], nd[2], nd[3], cedge, fontsize=7, fill=cfill)

    tx, ty, tw, th = n['T'][0], n['T'][1], n['T'][2], n['T'][3]
    mx, my, mr = n['M'][0], n['M'][1], n['M'][2]
    yx, yy, yr = n['Y'][0], n['Y'][1], n['Y'][2]
    draw_arrow(ax, tx + tw / 2 + 0.08, ty, mx - mr - 0.08, my, COL_CAUSAL)
    draw_arrow(ax, mx + mr + 0.08, my, yx - yr * 1.15 - 0.08, yy, COL_CAUSAL)

    arr_col = COL_BLOCK if conditioned else COL_CONFND
    arr_lw = 1.0 if conditioned else 1.4
    for key in ['Z1', 'Z2', 'Z3', 'Z4']:
        zx, zy, zr = n[key][0], n[key][1], n[key][2]
        if zy > ty:
            draw_arrow(ax, zx, zy - zr - 0.05, tx + tw * 0.2, ty + th / 2 + 0.05,
                       arr_col, dashed=True, lw=arr_lw)
        else:
            draw_arrow(ax, zx, zy + zr + 0.05, tx + tw * 0.2, ty - th / 2 - 0.05,
                       arr_col, dashed=True, lw=arr_lw)
        if zy > yy:
            draw_arrow(ax, zx + zr * 0.4, zy - zr - 0.05, yx - yr * 0.4,
                       yy + yr * 1.15 + 0.05, arr_col, dashed=True, lw=arr_lw)
        else:
            draw_arrow(ax, zx + zr * 0.4, zy + zr + 0.05, yx - yr * 0.4,
                       yy - yr * 1.15 - 0.05, arr_col, dashed=True, lw=arr_lw)


# ── Panel A ──
draw_dag_panel(ax_a, '(A)  Original DAG: Confounding Paths', conditioned=False)
ax_a.text(4.5, -0.2, 'Dashed orange arrows = backdoor (confounding) paths',
          ha='center', fontsize=7, fontstyle='italic', color=COL_CONFND)

# ── Panel B ──
draw_dag_panel(ax_b, '(B)  After Conditioning on\nZ = {SOFA, Age, Comorbidity, Inf. Source}',
               conditioned=True)
ax_b.text(4.5, -0.2, 'Z blocks all backdoor paths (grey = conditioned)',
          ha='center', fontsize=7, fontstyle='italic', color=C_MEDIATOR)

# ── Panel C: data-driven numerical demonstration ──
ax_c.grid(axis='y', visible=False)
labels = ['Naive\n(unadjusted)']
values = [naive_pp]
colors = [PALETTE[1]]
hatches = ['']
for sev in SEV_ORDER:
    labels.append(f"{SEV_LABELS[sev]}\n(stratified)")
    values.append(strat[sev])
    colors.append(_lighten(PALETTE[0], 0.4) if sev != "sev_high" else PALETTE[0])
    hatches.append('//')
labels.append('Adjusted\n(doubly-robust)')
values.append(adj_pp)
colors.append("#08306B")
hatches.append('')

y = np.arange(len(labels))[::-1]
bars = ax_c.barh(y, values, color=colors, edgecolor='#1F2937', linewidth=0.6,
                 height=0.6, zorder=3)
for b, h in zip(bars, hatches):
    b.set_hatch(h)
for yi, v in zip(y, values):
    ax_c.text(v + (0.4 if v >= 0 else -0.4), yi, f"{v:.1f}",
              va='center', ha='left' if v >= 0 else 'right',
              fontsize=8, fontweight='bold', color='#1F2937', zorder=4)

ax_c.axvline(0, color='#888888', lw=0.8, ls=':')
ax_c.axvline(true_pp, color=PALETTE[6], lw=1.4, ls='-.', alpha=0.8,
             label=f"True ATE ({true_pp:.1f} pp)")

# bias arrow naive -> adjusted (value from results)
ytop = y.max() + 0.55
ax_c.annotate('', xy=(adj_pp, ytop), xytext=(naive_pp, ytop),
              arrowprops=dict(arrowstyle='<->', color=PALETTE[1], lw=1.5))
ax_c.text((naive_pp + adj_pp) / 2, ytop + 0.12, f"bias {bias_pp:.1f} pp",
          ha='center', va='bottom', fontsize=8, fontweight='bold', color=PALETTE[1])

ax_c.set_yticks(y); ax_c.set_yticklabels(labels, fontsize=7.5)
ax_c.set_ylim(-0.6, y.max() + 1.1)
ax_c.set_xlabel('Change in mortality\nprobability (pp)', fontsize=8.5)
ax_c.legend(loc='lower left', fontsize=7)
ax_c.set_title('(C)  Numerical Demonstration\n'
               f'Naive ({naive_pp:.1f}) vs. Adjusted ({adj_pp:.1f}) pp',
               fontsize=9.5, fontweight='bold')

fig.suptitle('Backdoor Criterion: Identifying and Blocking Confounding Paths (Sepsis Domain)',
             fontsize=12, fontweight='bold')

OUT = __import__("pathlib").Path(__file__).resolve().parent
save_fig(fig, 'fig05_backdoor_adjustment', OUT)
plt.close(fig)
