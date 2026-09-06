#!/usr/bin/env python3
"""
Figure 5: Backdoor Criterion for Confounding Control (Sepsis Domain)
KAIS Causal Models for CDSS — Springer submission

Three panels:
  (A) Original DAG with active confounding paths (dashed orange).
  (B) DAG after conditioning on the minimal backdoor set Z = {Severity}.
  (C) Numerical demonstration: naive vs. severity-stratified vs. adjusted ATE.

Panels A/B are structural schematics restyled to the canonical palette.
Panel C numbers are loaded from results/ (ate_by_domain.csv +
cate_subgroups.csv); NOTHING is hardcoded.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from pubviz import (apply_pub_style, PALETTE,
                    C_TREATMENT, C_OUTCOME, C_CONFOUNDER, C_MEDIATOR)

apply_pub_style()

DOMAIN = "sepsis"
SEV_ORDER = ["sev_low", "sev_mid", "sev_high"]
SEV_LABELS = {"sev_low": "Severity: low", "sev_mid": "Severity: mid", "sev_high": "Severity: high"}

# ---------- Load data ----------
ate = pd.read_csv(REPO / "results" / "ate_by_domain.csv")
row = ate[ate["domain"] == DOMAIN].iloc[0]
naive_pp = row["naive_ate"] * 100.0
adj_pp = row["doubly_robust_ate"] * 100.0       # adjusted = doubly-robust
true_pp = row["true_ate"] * 100.0
bias_pp = row["confounding_bias"] * 100.0

cate = pd.read_csv(REPO / "results" / "cate_subgroups.csv")
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
    'T':   (1.7, 3.0, 1.8, 0.90, 'Early' + chr(10) + 'Antibiotics'),
    'Y':   (7.3, 3.0, 0.78, 'Mortality'),
    'AGE': (1.5, 8.2, 0.58, 'Age'),
    'SEV': (4.4, 6.6, 0.78, 'Severity'),
}


def draw_dag_panel(ax, title, conditioned=False):
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(0.5, 9.8)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')
    n = dag_nodes
    tx, ty, tw, th = n['T'][0], n['T'][1], n['T'][2], n['T'][3]
    yx, yy, yr = n['Y'][0], n['Y'][1], n['Y'][2]
    ax_, ay_, ar_ = n['AGE'][0], n['AGE'][1], n['AGE'][2]
    sx, sy, sr = n['SEV'][0], n['SEV'][1], n['SEV'][2]

    draw_rect(ax, tx, ty, tw, th, n['T'][4], COL_TREAT)
    draw_circle(ax, yx, yy, yr, n['Y'][3], COL_OUT, double=True)
    draw_circle(ax, ax_, ay_, ar_, n['AGE'][3], COL_CONF, fontsize=7.5)

    sev_edge = "#999999" if conditioned else COL_CONF
    sev_fill = COL_COND if conditioned else _lighten(COL_CONF)
    draw_circle(ax, sx, sy, sr, n['SEV'][3], sev_edge, fontsize=7.5, fill=sev_fill)

    blocked = COL_BLOCK if conditioned else COL_CONFND
    blw = 1.0 if conditioned else 1.4

    draw_arrow(ax, ax_ + ar_ * 0.8, ay_ - ar_ * 0.5, sx - sr - 0.05, sy + sr * 0.4,
               COL_CONFND, dashed=True, lw=1.4)
    draw_arrow(ax, ax_ + ar_ * 0.5, ay_ + ar_ * 0.6, yx + yr * 0.4, yy + yr * 1.35,
               COL_CONFND, dashed=True, lw=1.4)
    draw_arrow(ax, sx - sr * 0.7, sy - sr - 0.05, tx + tw * 0.25, ty + th / 2 + 0.05,
               blocked, dashed=True, lw=blw)
    draw_arrow(ax, sx + sr * 0.7, sy - sr - 0.05, yx - yr * 0.6, yy + yr * 1.15 + 0.05,
               blocked, dashed=True, lw=blw)
    draw_arrow(ax, tx + tw / 2 + 0.08, ty, yx - yr * 1.15 - 0.08, yy, COL_CAUSAL, lw=2.0)


# ── Panel A ──
draw_dag_panel(ax_a, '(A)  Original DAG: Confounding Paths', conditioned=False)
ax_a.text(4.5, 0.9, 'Dashed orange arrows = backdoor (confounding) paths',
          ha='center', fontsize=7, fontstyle='italic', color=COL_CONFND)

# ── Panel B ──
draw_dag_panel(ax_b, '(B)  After Conditioning on\nZ = {Severity}',
               conditioned=True)
ax_b.text(4.5, 0.9, 'Conditioning on severity blocks the only backdoor path (grey)',
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

for _ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig05_backdoor_adjustment.{_ext}", bbox_inches="tight",
                facecolor="white")
plt.close(fig)
