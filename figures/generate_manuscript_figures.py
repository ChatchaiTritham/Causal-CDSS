"""Draw the five manuscript figures at the journal text-block width.

The per-figure scripts fig01..fig05 were authored 9.5-16 in wide for a poster-
sized canvas; KAIS gives a figure 372 pt (5.17 in), so they were shrunk to about
a third and their labels landed at 2.2-3.8 pt. Rescaling alone does not rescue
them -- at that width the old layouts collide -- so the two diagram figures are
redrawn and the three data figures are relaid out vertically.

Every number still comes from results/*.csv written by run_all.py at seed 42;
nothing here is hardcoded. The one content change is in fig01: the previous
version drew the same DAG three times (its own caption said "identical
structure, domain-specific coefficients"), so it now draws the structure once
and names the three treatments beneath it.

    python figures/generate_manuscript_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pubviz import PALETTE, results_dir, save_fig  # noqa: E402

TEXT_PT = 372.0
W_IN = TEXT_PT / 72.0
BODY_PT = 8.0
LABEL_PT = 9.0

BLUE, VERM, GREEN, PINK, AMBER, SKY, BLACK = PALETTE
GREY = "#4D4D4D"
INK = "#1A1A1A"

DOMAIN_ORDER = ["sepsis", "ards", "acs"]
DOMAIN_SHORT = {"sepsis": "Sepsis", "ards": "ARDS", "acs": "ACS"}
DOMAIN_TREATMENT = {"sepsis": "early antibiotics",
                    "ards": "low tidal volume",
                    "acs": "early reperfusion"}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BODY_PT,
        "axes.labelsize": LABEL_PT,
        "axes.titlesize": LABEL_PT,
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "text.color": INK,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })



def save_exact(fig, stem):
    """Save at the text block exactly.

    bbox_inches="tight" trims to the artists, so the result is rarely the
    canvas width. Measure the first save and, if it missed, rescale the canvas
    by the observed ratio and save again. One correction is always enough
    because the trim is proportional.
    """
    import pypdf

    for _ in range(2):
        save_fig(fig, stem, out_dir=str(HERE))
        page = pypdf.PdfReader(str(HERE / f"{stem}.pdf")).pages[0]
        got = float(page.mediabox.width)
        if abs(got - TEXT_PT) <= 1.0:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * TEXT_PT / got, h * TEXT_PT / got)


def ate_table():
    df = pd.read_csv(results_dir() / "ate_by_domain.csv")
    return df.set_index("domain").loc[DOMAIN_ORDER].reset_index()


# --------------------------------------------------------------------------
def fig01_causal_dag():
    """One DAG: the structure is common to all three domains."""
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.60))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def node(x, y, label, shape, colour, face):
        if shape == "rect":
            ax.add_patch(FancyBboxPatch((x - 11, y - 4.6), 22, 9.2,
                                        boxstyle="round,pad=0,rounding_size=1.2",
                                        facecolor=face, edgecolor=colour, linewidth=1.0, zorder=4))
        else:
            ax.add_patch(Circle((x, y), 7.2, facecolor=face, edgecolor=colour,
                                linewidth=1.0, zorder=4))
            if shape == "double":
                ax.add_patch(Circle((x, y), 5.9, facecolor="none", edgecolor="white",
                                    linewidth=1.0, zorder=5))
        ax.text(x, y, label, ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold", color=INK, zorder=6)

    def edge(p0, p1, colour, dashed):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.1, color=colour, zorder=3,
                                     linestyle=(0, (4, 2)) if dashed else "solid",
                                     connectionstyle="arc3,rad=0.14" if dashed else "arc3,rad=0"))

    # Age -> Severity -> {T, Y}; Age -> Y ; T -> Y
    node(14, 47, "Age", "circle", AMBER, "#FBEBD2")
    node(40, 38, "Severity", "circle", AMBER, "#FBEBD2")
    node(14, 20, "T", "rect", BLUE, "#DCE9F5")
    node(78, 20, "Y", "double", VERM, "#F7DCCD")

    edge((20, 44), (34, 41), AMBER, True)
    edge((36, 32), (22, 24), AMBER, True)
    edge((46, 33), (72, 24), AMBER, True)
    edge((20, 50), (72, 27), AMBER, True)
    edge((25, 20), (70, 20), BLUE, False)

    ax.text(50, 10.5, "T = treatment    ·    Y = in-hospital mortality (binary)"
                      "    ·    minimal backdoor set Z = {Severity}",
            ha="center", va="center", fontsize=BODY_PT, color=GREY)
    treat = "        ".join("%s: %s" % (DOMAIN_SHORT[d], DOMAIN_TREATMENT[d]) for d in DOMAIN_ORDER)
    ax.text(50, 5.0, treat, ha="center", va="center", fontsize=BODY_PT, color=INK)

    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=6,
               markerfacecolor="#DCE9F5", markeredgecolor=BLUE, label="Treatment (T)"),
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor="#F7DCCD", markeredgecolor=VERM, label="Outcome (Y)"),
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor="#FBEBD2", markeredgecolor=AMBER, label="Baseline covariate"),
        Line2D([], [], color=BLUE, label="Treatment effect path"),
        Line2D([], [], color=AMBER, linestyle=(0, (4, 2)), label="Backdoor path"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=1,
              handletextpad=0.6, borderpad=0.2, labelspacing=0.35,
              bbox_to_anchor=(1.01, 1.02))
    save_exact(fig, "fig01_causal_dag")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig02_intervention_effects():
    df = ate_table()
    est = [("naive", "Naive (unadjusted)", "v", VERM),
           ("backdoor", "Backdoor regression", "s", GREEN),
           ("doubly_robust", "Doubly-robust", "D", BLUE)]

    fig, axes = plt.subplots(3, 1, figsize=(W_IN, W_IN * 1.02), sharex=True)
    for ax, (_, row) in zip(axes, df.iterrows()):
        for k, (key, _lab, marker, colour) in enumerate(est):
            y = len(est) - 1 - k
            pt = row[f"{key}_ate"] * 100
            lo = row[f"{key}_ci_low"] * 100
            hi = row[f"{key}_ci_high"] * 100
            ax.plot([lo, hi], [y, y], color=colour, linewidth=1.3, solid_capstyle="butt")
            ax.plot([pt], [y], marker=marker, color=colour, markersize=5.2, zorder=4)
            ax.text(hi + 0.7, y, "%+.1f  [%+.1f, %+.1f]" % (pt, lo, hi),
                    va="center", ha="left", fontsize=BODY_PT, color=colour)
        true_ate = row["true_ate"] * 100
        ax.axvline(true_ate, color=BLUE, linestyle=(0, (4, 2)), linewidth=1.0)
        ax.axvline(0, color=BLACK, linewidth=0.9)
        ax.set_yticks(range(len(est)))
        ax.set_yticklabels([lab for _, lab, _, _ in est][::-1])
        ax.set_ylim(-0.7, len(est) - 0.3)
        ax.set_xlim(-34, 30)
        ax.text(0.015, 0.94, "%s — %s" % (DOMAIN_SHORT[row["domain"]],
                                          DOMAIN_TREATMENT[row["domain"]].capitalize()),
                transform=ax.transAxes, fontsize=BODY_PT, fontweight="bold", va="top")
        ax.text(true_ate - 0.7, -0.55, "true %+.1f" % true_ate, ha="right", va="center",
                fontsize=BODY_PT, color=BLUE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Change in mortality probability (percentage points)\n"
                        "markers = point estimate, bars = 95% bootstrap CI")
    handles = [Line2D([], [], color=c, marker=m, markersize=5.2, label=l)
               for _, l, m, c in est]
    handles.append(Line2D([], [], color=BLUE, linestyle=(0, (4, 2)),
                          label="True ATE (do-calculus)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=0.9)
    save_exact(fig, "fig02_intervention_effects")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig03_cate_heterogeneity():
    df = pd.read_csv(results_dir() / "cate_subgroups.csv")
    sep = df[df.domain == "sepsis"]
    ages = ["age_low", "age_mid", "age_high"]
    sevs = ["sev_high", "sev_mid", "sev_low"]
    age_lab = ["Age < 50", "Age 50–70", "Age > 70"]
    sev_lab = ["qSOFA 3\n(high)", "qSOFA 2\n(moderate)", "qSOFA 0–1\n(low)"]

    grid = np.array([[float(sep[(sep.age_group == a) & (sep.severity_group == s)]
                             ["true_cate"].iloc[0]) * 100 for a in ages] for s in sevs])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_IN, W_IN * 1.05))

    im = ax1.imshow(grid, cmap="Blues_r", aspect="auto")
    ax1.set_xticks(range(3), age_lab)
    ax1.set_yticks(range(3), sev_lab)
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            ax1.text(j, i, "%.1f" % v, ha="center", va="center", fontsize=BODY_PT,
                     color="white" if v < grid.mean() else INK, fontweight="bold")
    ax1.set_title("True CATE by severity and age (percentage points)", pad=5)
    cb = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=BODY_PT)
    for s in ("top", "right", "bottom", "left"):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(length=0)

    colours = {"sev_low": SKY, "sev_mid": BLUE, "sev_high": "#08306B"}
    for s, lab in (("sev_low", "qSOFA 0–1"), ("sev_mid", "qSOFA 2"), ("sev_high", "qSOFA 3")):
        sub = sep[sep.severity_group == s]
        ax2.scatter(sub.true_cate * 100, sub.estimated_cate * 100, s=26,
                    color=colours[s], edgecolor="white", linewidth=0.5, label=lab, zorder=4)
    lo, hi = -34, -3
    ax2.plot([lo, hi], [lo, hi], color=GREY, linestyle=(0, (4, 2)), linewidth=0.9,
             label="Perfect agreement", zorder=2)
    m, b = np.polyfit(sep.true_cate * 100, sep.estimated_cate * 100, 1)
    ax2.plot([lo, hi], [m * lo + b, m * hi + b], color=VERM, linewidth=1.2,
             label="OLS fit (slope %.2f)" % m, zorder=3)
    r = np.corrcoef(sep.true_cate, sep.estimated_cate)[0, 1]
    ax2.text(0.03, 0.95, "r = %.2f,  slope = %.2f" % (r, m), transform=ax2.transAxes,
             fontsize=BODY_PT, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREY, linewidth=0.6))
    ax2.set_xlabel("True CATE (percentage points)")
    ax2.set_ylabel("Estimated CATE (pp)")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(linewidth=0.4, alpha=0.3)
    ax2.set_axisbelow(True)

    fig.tight_layout(h_pad=1.2)
    save_exact(fig, "fig03_cate_heterogeneity")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig04_confounding_analysis():
    df = ate_table()
    x = np.arange(len(df))
    w = 0.2
    series = [("naive_ate", "Naive (no adjustment)", VERM),
              ("backdoor_ate", "Backdoor regression", AMBER),
              ("doubly_robust_ate", "Doubly-robust", BLUE),
              ("true_ate", "True ATE (do-calculus)", BLACK)]

    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.78))
    for k, (col, lab, colour) in enumerate(series):
        vals = df[col] * 100
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab, color=colour,
               edgecolor=INK, linewidth=0.4)
        for xi, v in zip(x + (k - 1.5) * w, vals):
            ax.text(xi, v + (0.9 if v >= 0 else -0.9), "%.1f" % v, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=BODY_PT)
    for xi, (_, row) in zip(x, df.iterrows()):
        ax.text(xi, 16.5, "bias %.1f pp" % (row["confounding_bias"] * 100), ha="center",
                va="center", fontsize=BODY_PT, color=VERM,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=VERM, linewidth=0.6))
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, ["%s\n(%s)" % (DOMAIN_SHORT[d], DOMAIN_TREATMENT[d]) for d in df.domain])
    ax.set_ylabel("Change in mortality\nprobability (pp)")
    ax.set_ylim(-28, 22)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.30))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.34)
    save_exact(fig, "fig04_confounding_analysis")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig05_backdoor_adjustment():
    """Two DAG panels stacked over the sepsis numbers."""
    df = ate_table()
    sep = df[df.domain == "sepsis"].iloc[0]
    cate = pd.read_csv(results_dir() / "cate_subgroups.csv")
    strata = cate[cate.domain == "sepsis"].groupby("severity_group").apply(
        lambda g: np.average(g.estimated_cate, weights=g.n_obs), include_groups=False)

    fig = plt.figure(figsize=(W_IN, W_IN * 1.02))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.18)

    def dag(ax, conditioned, title):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 62)
        ax.axis("off")
        ax.set_title(title, fontsize=BODY_PT, fontweight="bold", pad=3)
        face = "#D7D7D7" if conditioned else "#FBEBD2"
        edge = GREY if conditioned else AMBER
        ax.add_patch(Circle((50, 46), 11, facecolor=face, edgecolor=edge,
                            linewidth=1.0, zorder=4))
        ax.text(50, 46, "Severity", ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold", zorder=5)
        if conditioned:
            ax.add_patch(Rectangle((36, 32), 28, 28, facecolor="none", edgecolor=GREEN,
                                   linewidth=1.1, linestyle=(0, (3, 2)), zorder=6))
        ax.add_patch(FancyBboxPatch((4, 10), 30, 13, boxstyle="round,pad=0,rounding_size=1.5",
                                    facecolor="#DCE9F5", edgecolor=BLUE, linewidth=1.0, zorder=4))
        ax.text(19, 16.5, "Early\nantibiotics", ha="center", va="center",
                fontsize=BODY_PT, fontweight="bold", zorder=5)
        ax.add_patch(Circle((80, 16.5), 11, facecolor="#F7DCCD", edgecolor=VERM,
                            linewidth=1.0, zorder=4))
        ax.text(80, 16.5, "Mortality", ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold", zorder=5)
        bd = GREY if conditioned else AMBER
        ax.add_patch(FancyArrowPatch((43, 38), (27, 24), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=1.0, color=bd, linestyle=(0, (4, 2)), zorder=3))
        ax.add_patch(FancyArrowPatch((58, 38), (74, 26), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=1.0, color=bd, linestyle=(0, (4, 2)), zorder=3))
        ax.add_patch(FancyArrowPatch((35, 16.5), (68, 16.5), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=1.2, color=BLUE, zorder=3))
        ax.text(50, 2.5, "backdoor path blocked" if conditioned else "backdoor path open",
                ha="center", va="center", fontsize=BODY_PT,
                color=GREEN if conditioned else AMBER)

    dag(fig.add_subplot(gs[0, 0]), False, "(A)  Original DAG")
    dag(fig.add_subplot(gs[0, 1]), True, "(B)  Conditioning on Z = {Severity}")

    ax = fig.add_subplot(gs[1, :])
    labels = ["Naive\n(unadjusted)", "Low", "Moderate", "High", "Adjusted\n(doubly-robust)"]
    vals = [sep["naive_ate"] * 100] + \
           [strata["sev_low"] * 100, strata["sev_mid"] * 100, strata["sev_high"] * 100] + \
           [sep["doubly_robust_ate"] * 100]
    colours = [VERM] + [SKY, BLUE, "#08306B"] + [BLUE]
    bars = ax.bar(range(len(vals)), vals, 0.58, color=colours, edgecolor=INK, linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.9 if v >= 0 else -0.9), "%.1f" % v,
                ha="center", va="bottom" if v >= 0 else "top", fontsize=BODY_PT)
    ax.axhline(sep["true_ate"] * 100, color=BLACK, linestyle=(0, (4, 2)), linewidth=1.0)
    ax.text(0.5, sep["true_ate"] * 100 + 0.9, "true ATE %.1f" % (sep["true_ate"] * 100),
            va="bottom", ha="center", fontsize=BODY_PT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(vals)), labels)
    ax.annotate("stratified by severity", xy=(2, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -26), textcoords="offset points", ha="center", va="top",
                fontsize=BODY_PT, color=GREY)
    ax.set_ylabel("Change in mortality\nprobability (pp)")
    ax.set_title("(C)  Naive %.1f pp versus adjusted %.1f pp (sepsis)"
                 % (sep["naive_ate"] * 100, sep["doubly_robust_ate"] * 100),
                 fontsize=BODY_PT, fontweight="bold", pad=4)
    ax.set_xlim(-0.6, len(vals) - 0.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_exact(fig, "fig05_backdoor_adjustment")
    plt.close(fig)


def main():
    style()
    fig01_causal_dag()
    fig02_intervention_effects()
    fig03_cate_heterogeneity()
    fig04_confounding_analysis()
    fig05_backdoor_adjustment()
    print("five figures at %.0f pt with an %.1f pt floor" % (TEXT_PT, BODY_PT))


if __name__ == "__main__":
    main()
