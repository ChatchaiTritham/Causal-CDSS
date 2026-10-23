"""Write the three held-out tables used by the Statistics in Medicine manuscript.

tables/v2_rule_sim.tex  <- results/v2_rule_summary.csv    (Supplementary Table S37)
tables/v3_loss.tex      <- results/default_loss_summary.csv (main Table 3)
tables/v3_strata.tex    <- results/default_loss_strata.csv  (Supplementary Table S51)
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES, TAB = REPO / "results", REPO / "tables"
SETS = [("A (seed 2027, used in v1 test)", "A (dev.)"), ("B (seed 2031, fresh)", "B (fresh)")]


def ci(s):
    lo, hi = str(s).split(" to ")
    return f"${lo}, {hi}$"


def rule_table():
    d = pd.read_csv(RES / "v2_rule_summary.csv")
    keep = ["always proxy", "always DR", "v1 oracle bounds", "v1 elicited bounds"]
    out = []
    for k, (key, lab) in enumerate(SETS):
        if k:
            out.append("\midrule")
        for s in keep:
            r = d[(d.set == key) & (d.strategy == s)].iloc[0]
            out.append(f"{lab} & {s} & {r.correct_all_reps:.2f} & {r.correct_all_ci.replace('-', '--')} & "
                       f"{r.correct_conclusive:.2f} & {r.inconclusive:.2f} & {r.mean_rmse:.3f} & "
                       f"{r.mean_regret:.3f} & {ci(r.regret_minus_dr_ci)} & {r.coverage:.2f} \\\\")
    return out


def loss_table():
    d = pd.read_csv(RES / "default_loss_summary.csv")
    names = {"proxy": "proxy adjustment", "tsls": "proximal 2SLS", "dr": "DR proximal"}
    out = []
    for k, (key, lab) in enumerate(SETS):
        if k:
            out.append("\midrule")
        for fam, name in names.items():
            i = d[(d.set == key) & (d.loss == "is") & (d.family == fam)].iloc[0]
            m = d[(d.set == key) & (d.loss == "rmse") & (d.family == fam)].iloc[0]
            di = "---" if fam == "dr" else ci(i.diff_vs_dr_ci)
            dm = "---" if fam == "dr" else ci(m.diff_vs_dr_ci)
            out.append(f"{lab} & {name} & {i.mean_loss:.2f} & {di} & {i.best_share:.2f} & {m.mean_loss:.3f} & "
                       f"{dm} & {m.best_share:.2f} & {i.coverage:.2f} & {i.mean_width:.3f} \\\\")
    return out


def strata_table():
    d = pd.read_csv(RES / "default_loss_strata.csv")
    lab_s = {"u2<=median": r"$U_2 \le$ median", "u2>median": r"$U_2 >$ median",
             "rho<=median": r"$\rho_{ZW} \le$ median", "rho>median": r"$\rho_{ZW} >$ median", "all": "all"}
    out = []
    for key, lab in SETS:
        for s, ls in lab_s.items():
            r = d[(d.set == key) & (d.stratum == s)].iloc[0]
            out.append(f"{lab} & {ls} & {int(r.n_gen)} & {r.dr_cov:.2f} & {r.sd_over_se:.2f} & "
                       f"{r.absbias_over_sd:.2f} & {r.dr_width:.3f} & {r.wide_cov:.2f} & {r.wide_width:.3f} \\\\")
    return out


if __name__ == "__main__":
    for name, rows in [("v2_rule_sim.tex", rule_table()), ("v3_loss.tex", loss_table()),
                       ("v3_strata.tex", strata_table())]:
        (TAB / name).write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        print("wrote", TAB / name)
