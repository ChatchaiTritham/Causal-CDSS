#!/usr/bin/env python3
"""
Figure 5: Backdoor Criterion for Confounding Control (Sepsis Domain)
KAIS Causal Models for CDSS — Springer submission

Three-panel figure:
  (A) Original DAG with active confounding paths (red dashed)
  (B) DAG after conditioning on Z = {SOFA, Age, Comorbidity, Infection Source}
  (C) Numerical demonstration: naive −3.5% vs adjusted −15.2%
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import numpy as np

# ─── Colours ───
COL_TREAT_F = '#2980B9'
COL_TREAT   = '#1A5276'
COL_OUT_F   = '#E74C3C'
COL_OUTCOME = '#C0392B'
COL_CONF_F  = '#F1C40F'
COL_CONF    = '#7D6608'
COL_MED_F   = '#27AE60'
COL_MED     = '#1E8449'
COL_CAUSAL  = '#2471A3'
COL_CONFND  = '#E74C3C'
COL_BLOCKED = '#BDC3C7'
COL_COND_F  = '#D5D8DC'
COL_COND    = '#95A5A6'

def draw_rect(ax, cx, cy, w, h, text, fill, edge, fontsize=8):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.06", facecolor=fill,
                          edgecolor=edge, linewidth=1.6, zorder=4)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=5)

def draw_circle(ax, cx, cy, r, text, fill, edge, fontsize=7.5, double=False,
                text_color=None):
    if text_color is None:
        text_color = 'white' if fill not in [COL_CONF_F, COL_COND_F] else '#333'
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
            fontweight='bold', color=text_color, zorder=5, linespacing=1.15)

def draw_arrow(ax, x1, y1, x2, y2, color, dashed=False, lw=1.6):
    ls = '-'
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', mutation_scale=13,
                             color=color, linewidth=lw, linestyle=ls,
                             connectionstyle='arc3,rad=0.10',
                             zorder=3)
    if dashed:
        arrow.set_linestyle((0, (5, 3)))
    ax.add_patch(arrow)

# ─── Figure ───
fig = plt.figure(figsize=(16, 5.0), dpi=300)
fig.patch.set_facecolor('white')

# Grid: panels A, B share left 60%, panel C takes right 40%
gs = fig.add_gridspec(1, 5, wspace=0.35)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4])

# ═══════ Shared DAG layout (normalised to [0,10]×[0,10]) ═══════
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
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    n = dag_nodes
    # Treatment
    draw_rect(ax, n['T'][0], n['T'][1], n['T'][2], n['T'][3], n['T'][4],
              COL_TREAT_F, COL_TREAT, fontsize=8)
    # Mediator
    draw_circle(ax, n['M'][0], n['M'][1], n['M'][2], n['M'][3],
                COL_MED_F, COL_MED, fontsize=7.5)
    # Outcome
    draw_circle(ax, n['Y'][0], n['Y'][1], n['Y'][2], n['Y'][3],
                COL_OUT_F, COL_OUTCOME, fontsize=7.5, double=True)

    # Confounders
    conf_fill = COL_COND_F if conditioned else COL_CONF_F
    conf_edge = COL_COND if conditioned else COL_CONF
    for key in ['Z1', 'Z2', 'Z3', 'Z4']:
        nd = n[key]
        draw_circle(ax, nd[0], nd[1], nd[2], nd[3], conf_fill, conf_edge,
                    fontsize=7, text_color='#555' if conditioned else '#333')

    # Causal arrows (always blue)
    tx, ty = n['T'][0], n['T'][1]
    tw, th = n['T'][2], n['T'][3]
    mx, my, mr = n['M'][0], n['M'][1], n['M'][2]
    yx, yy, yr = n['Y'][0], n['Y'][1], n['Y'][2]

    draw_arrow(ax, tx + tw/2 + 0.08, ty, mx - mr - 0.08, my, COL_CAUSAL)
    draw_arrow(ax, mx + mr + 0.08, my, yx - yr*1.15 - 0.08, yy, COL_CAUSAL)

    # Confounding arrows
    arr_col = COL_BLOCKED if conditioned else COL_CONFND
    arr_lw = 1.0 if conditioned else 1.4

    for key in ['Z1', 'Z2', 'Z3', 'Z4']:
        zx, zy, zr = n[key][0], n[key][1], n[key][2]
        # Z → T
        if zy > ty:
            draw_arrow(ax, zx, zy - zr - 0.05,
                       tx + tw*0.2, ty + th/2 + 0.05,
                       arr_col, dashed=True, lw=arr_lw)
        else:
            draw_arrow(ax, zx, zy + zr + 0.05,
                       tx + tw*0.2, ty - th/2 - 0.05,
                       arr_col, dashed=True, lw=arr_lw)
        # Z → Y
        if zy > yy:
            draw_arrow(ax, zx + zr*0.4, zy - zr - 0.05,
                       yx - yr*0.4, yy + yr*1.15 + 0.05,
                       arr_col, dashed=True, lw=arr_lw)
        else:
            draw_arrow(ax, zx + zr*0.4, zy + zr + 0.05,
                       yx - yr*0.4, yy - yr*1.15 - 0.05,
                       arr_col, dashed=True, lw=arr_lw)

# ═══════ Panel A ═══════
draw_dag_panel(ax_a, '(A)  Original DAG: Confounding Paths', conditioned=False)
ax_a.text(4.5, -0.2,
          'Red dashed arrows = backdoor (confounding) paths',
          ha='center', fontsize=7, fontstyle='italic', color='#E74C3C')

# ═══════ Panel B ═══════
draw_dag_panel(ax_b,
               '(B)  After Conditioning on\nZ = {SOFA, Age, Comorbidity, Inf. Source}',
               conditioned=True)
ax_b.text(4.5, -0.2,
          'Z blocks all backdoor paths (grey = conditioned)',
          ha='center', fontsize=7, fontstyle='italic', color='#27AE60')

# ═══════ Panel C: Numerical Demonstration ═══════
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 10)
ax_c.axis('off')
ax_c.set_title('(C)  Numerical Demonstration\nNaive (−3.5%) vs. Adjusted (−15.2%)',
               fontsize=9.5, fontweight='bold', pad=8)

# Horizontal bars — values as bar widths (scaled: 1 pp ≈ 0.38 units)
scale = 0.38
bar_h = 0.55
bar_x0 = 1.0

# Bias annotation
ax_c.annotate('', xy=(bar_x0 + 15.2*scale, 8.8),
              xytext=(bar_x0 + 3.5*scale, 8.8),
              arrowprops=dict(arrowstyle='<->', color='#C0392B', lw=1.8))
ax_c.text(bar_x0 + 9.3*scale, 9.15, '+11.7 pp bias',
          ha='center', fontsize=8.5, fontweight='bold', color='#C0392B')

# Adjusted ATE bar (dark blue)
adj_w = 15.2 * scale
ax_c.barh(8.0, adj_w, height=bar_h, left=bar_x0,
          color='#1A5276', edgecolor='#0E3550', linewidth=0.8, zorder=3)
ax_c.text(bar_x0 + adj_w/2, 8.0, '−15.2%', ha='center', va='center',
          fontsize=9, fontweight='bold', color='white', zorder=4)
ax_c.text(bar_x0 + adj_w + 0.2, 8.0, 'Adjusted\n(Weighted avg.)',
          ha='left', va='center', fontsize=7, color='#1A5276')

# Stratified bars
strat_data = [
    (6.3, 'qSOFA 3', '-19.8%', '#2471A3'),
    (5.0, 'qSOFA 2', '-12.2%', '#5DADE2'),
    (3.7, 'qSOFA 0-1', '-6.2%', '#85C1E9'),
]
for y_pos, label, val_str, col in strat_data:
    w = abs(float(val_str.replace('−', '-').replace('%', ''))) * scale
    ax_c.barh(y_pos, w, height=bar_h * 0.85, left=bar_x0,
              color=col, edgecolor='white', linewidth=0.6, zorder=3,
              hatch='///' if col == '#2471A3' else '')
    # Label
    if w > 2.5:
        ax_c.text(bar_x0 + w/2, y_pos, val_str, ha='center', va='center',
                  fontsize=8, fontweight='bold', color='white', zorder=4)
    else:
        ax_c.text(bar_x0 + w + 0.15, y_pos, val_str, ha='left', va='center',
                  fontsize=8, fontweight='bold', color='#333', zorder=4)
    ax_c.text(bar_x0 - 0.15, y_pos, label, ha='right', va='center',
              fontsize=7.5, color='#555')
    # (Stratified) annotation
    if label == 'qSOFA 3':
        ax_c.text(bar_x0 + w + 0.2, y_pos, '(Stratified)',
                  ha='left', va='center', fontsize=6.5, fontstyle='italic',
                  color='#888')

# Naive estimate bar (grey)
naive_w = 3.5 * scale
ax_c.barh(2.2, naive_w, height=bar_h, left=bar_x0,
          color='#BDC3C7', edgecolor='#95A5A6', linewidth=0.8, zorder=3)
ax_c.text(bar_x0 + naive_w + 0.15, 2.2, '−3.5%', ha='left', va='center',
          fontsize=9, fontweight='bold', color='#666', zorder=4)
ax_c.text(bar_x0 - 0.15, 2.2, 'Naive\n(Unadjusted)', ha='right', va='center',
          fontsize=7.5, color='#888')

# X-axis label
ax_c.text(5.0, 1.0, 'Mortality Reduction (pp)', ha='center', fontsize=8,
          color='#666')

# Scale ticks
for tick_val in [0, 5, 10, 15, 20]:
    tick_x = bar_x0 + tick_val * scale
    ax_c.plot([tick_x, tick_x], [1.4, 1.6], color='#999', lw=0.6)
    ax_c.text(tick_x, 1.25, f'-{tick_val}', ha='center', fontsize=6.5, color='#999')

# Legend
leg_items = [
    Line2D([0], [0], color='#1A5276', lw=6, label='Adjusted ATE (−15.2%)'),
    Line2D([0], [0], color='#BDC3C7', lw=6, label='Naive estimate (−3.5%)'),
]
ax_c.legend(handles=leg_items, loc='lower left', fontsize=7,
            framealpha=0.9, edgecolor='#DDD', bbox_to_anchor=(0.0, -0.02))

plt.savefig('fig05_backdoor_adjustment.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.12)
plt.savefig('fig05_backdoor_adjustment.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.12)
print("Saved: fig05_backdoor_adjustment.png / .pdf")
plt.close()
