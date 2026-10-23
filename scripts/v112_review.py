"""v1.1.2 review additions (Statistics in Medicine review round 2026-10-23).

Parts (python scripts/v112_review.py --parts roles,fig --boot 300 --workers 8):
  roles  SUPPORT right heart catheterization data: proximal 2SLS with every single-proxy
         pair (one Z from (pafi1, paco21), one W from (ph1, hema1)) and with the roles of
         the two proxy pairs swapped, for 30-day death and days survived up to day 30;
         first-stage partial F of Z for W and nonparametric percentile bootstrap intervals.
         Writes results/v112_rhc_proxy_roles.csv and tables/v112_roles.tex.
  fig    Figure of the held-out generators from results/default_loss_generators.csv:
         (a) generator-level DR coverage against the share of replications flagged;
         (b) generator-level RMSE of proximal 2SLS and proxy adjustment against DR.
         Writes figures/fig_heldout.pdf.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v110_extensions import ols_hc0, p2sls, partial_f  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES, TAB, FIG = REPO / "results", REPO / "tables", REPO / "figures"
Z_NAMES, W_NAMES = ("pafi1", "paco21"), ("ph1", "hema1")
LABEL = {"pafi1": "PaO$_2$/FiO$_2$", "paco21": "PaCO$_2$", "ph1": "pH", "hema1": "haematocrit"}


def variants():
    """(name, z column indices into Z, w column indices into W, swapped)."""
    out = [("both pairs (as published)", (0, 1), (0, 1), False)]
    for i, zn in enumerate(Z_NAMES):
        for j, wn in enumerate(W_NAMES):
            out.append((f"Z={zn}, W={wn}", (i,), (j,), False))
    out.append(("roles swapped", (0, 1), (0, 1), True))
    return out


def fit(d, zi, wi, swap, y):
    z, w = d["z"][:, list(zi)], d["w"][:, list(wi)]
    if swap:
        z, w = w, z
    return p2sls(d["a"], d["x"], z, w, y)[0]


_D = None


def boot_job(job):
    b, outcome = job
    global _D
    if _D is None:
        import rhc_data
        _D = rhc_data.load()
    d = _D
    rng = np.random.default_rng(112_000 + b)
    i = rng.integers(0, len(d["y"]), len(d["y"]))
    db = {k: d[k][i] for k in ("x", "z", "w", "a")}
    y = (d["y"] if outcome == "dth30" else d["days"])[i]
    out = []
    for _, zi, wi, sw in variants():
        try:
            out.append(fit(db, zi, wi, sw, y))
        except Exception:  # noqa: BLE001 - singular resample recorded as missing
            out.append(np.nan)
    return out


def part_roles(workers, B):
    import rhc_data
    d = rhc_data.load()
    base = np.column_stack([d["a"], d["x"]])
    rows = []
    for oc in ("dth30", "days"):
        y = d["y"] if oc == "dth30" else d["days"]
        with ProcessPoolExecutor(workers) as ex:
            R = np.array(list(ex.map(boot_job, [(b, oc) for b in range(B)])))
        for k, (name, zi, wi, sw) in enumerate(variants()):
            z, w = d["z"][:, list(zi)], d["w"][:, list(wi)]
            if sw:
                z, w = w, z
            fs = [partial_f(w[:, j], base, z) for j in range(w.shape[1])]
            lo, hi = np.nanpercentile(R[:, k], [2.5, 97.5])
            rows.append(dict(outcome=oc, variant=name, estimate=round(fit(d, zi, wi, sw, y), 4),
                             lo=round(float(lo), 4), hi=round(float(hi), 4), boot=B,
                             min_first_stage_F=round(float(min(fs)), 1)))
        rows.append(dict(outcome=oc, variant="proxy adjustment (reference)",
                         estimate=round(ols_hc0(np.column_stack([d["a"], d["x"], d["z"], d["w"]]), y, 0)[0], 4),
                         lo="", hi="", boot="", min_first_stage_F=""))
    with open(RES / "v112_rhc_proxy_roles.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    write_roles_table(rows)
    print("wrote results/v112_rhc_proxy_roles.csv, tables/v112_roles.tex")


def write_roles_table(rows=None):
    """tables/v112_roles.tex from results/v112_rhc_proxy_roles.csv."""
    if rows is None:
        with open(RES / "v112_rhc_proxy_roles.csv", encoding="utf-8") as fh:
            rows = [dict(r, **{k: float(r[k]) for k in ("estimate", "lo", "hi", "min_first_stage_F") if r[k]})
                    for r in csv.DictReader(fh)]
    lines = []
    for name, *_ in variants():
        a = next(r for r in rows if r["outcome"] == "dth30" and r["variant"] == name)
        b = next(r for r in rows if r["outcome"] == "days" and r["variant"] == name)
        lab = name
        if lab.startswith("Z="):
            zn, wn = lab[2:].split(", W=")
            lab = f"$Z$ = {LABEL[zn]}, $W$ = {LABEL[wn]}"
        lines.append(f"{lab} & {a['min_first_stage_F']:.1f} & ${a['estimate']:.3f}$ (${a['lo']:.3f}$, ${a['hi']:.3f}$) & "
                     f"${b['estimate']:.2f}$ (${b['lo']:.2f}$, ${b['hi']:.2f}$) " + chr(92) * 2)
    (TAB / "v112_roles.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def part_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    matplotlib.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    matplotlib.rcParams["svg.hashsalt"] = "v112"
    g = pd.read_csv(RES / "default_loss_generators.csv")
    pd.DataFrame([dict(set=k, generators=len(s),
                       share_tsls_rmse_below_dr=round(float((s.tsls_rmse < s.dr_rmse).mean()), 2),
                       share_proxy_rmse_below_dr=round(float((s.proxy_rmse < s.dr_rmse).mean()), 2),
                       share_never_flagged=round(float((s.flag_rate == 0).mean()), 2))
                  for k, s in g.groupby("set")]).to_csv(RES / "v112_heldout_pairwise.csv", index=False)
    sets = [("A (seed 2027, used in v1 test)", "set A (development)", "o", "#4C72B0"),
            ("B (seed 2031, fresh)", "set B (fresh)", "^", "#DD8452")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for key, lab, mk, col in sets:
        s = g[g.set == key]
        ax[0].scatter(s.flag_rate, s.dr_cov, marker=mk, s=18, color=col, alpha=0.8, label=lab)
        ax[1].scatter(s.dr_rmse, s.tsls_rmse, marker=mk, s=18, color=col, alpha=0.8, label=f"2SLS, {lab}")
        ax[1].scatter(s.dr_rmse, s.proxy_rmse, marker=mk, s=18, facecolors="none", edgecolors=col,
                      alpha=0.8, label=f"proxy adj., {lab}")
    ax[0].axhline(0.95, ls=":", color="k", lw=0.8)
    ax[0].axhline(0.80, ls="--", color="grey", lw=0.8)
    ax[0].set_xlabel("share of replications flagged")
    ax[0].set_ylabel("DR proximal coverage (generator)")
    ax[0].set_title("(a) flags against DR coverage", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left")
    m = float(max(g.dr_rmse.max(), g.tsls_rmse.max(), g.proxy_rmse.max())) * 1.05
    ax[1].plot([0, m], [0, m], color="k", lw=0.8)
    ax[1].set_xlim(0, m)
    ax[1].set_ylim(0, m)
    ax[1].set_xlabel("RMSE of DR proximal (generator)")
    ax[1].set_ylabel("RMSE of comparator (generator)")
    ax[1].set_title("(b) comparators against DR", fontsize=9)
    ax[1].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_heldout.pdf", metadata={"CreationDate": None, "ModDate": None})
    print("wrote figures/fig_heldout.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="roles,fig")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    if "roles" in a.parts:
        part_roles(a.workers, a.boot)
    if "table" in a.parts:
        write_roles_table()
    if "fig" in a.parts:
        part_fig()


if __name__ == "__main__":
    main()
