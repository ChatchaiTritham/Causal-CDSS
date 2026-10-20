"""Monte-Carlo uncertainty and a common scale for the crossing points gamma*, delta* (v1.0.3).

The crossing points of results/proximal_crossing.csv are interpolated from Monte-Carlo biases
(100 replications per grid point). Here each grid bias is redrawn from N(bias, mcse_bias)
(parametric bootstrap over replications, 5,000 draws), the crossing is recomputed by the same
linear interpolation, and the 2.5/97.5 percentiles are reported. The grid spacing (0.1 near
the crossings) adds an interpolation error that the interval does not include.

Common scale: gamma* is divided by the severity->outcome coefficient beta_U (both on the logit
scale of Y), and delta* by the standard deviation of W, so each reads as a fraction of the
confounding path it competes with.

Output: results/proximal_crossing_ci.csv
Usage:  python scripts/crossing_uncertainty.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402


def load(name):
    return list(csv.DictReader(open(uc.RESULTS / name)))


def cross(grid, dprox, dprx):
    diff = dprox - dprx
    for i in range(len(grid) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            return grid[i] + (grid[i + 1] - grid[i]) * (-diff[i]) / (diff[i + 1] - diff[i])
    return np.nan


def ci(rows, key, domain, est, rng, draws=5000):
    rs = [r for r in rows if r["domain"] == domain]
    grid = np.array(sorted({float(r[key]) for r in rs}))
    get = lambda e, f: np.array([float(next(r[f] for r in rs if r["estimator"] == e
                                            and float(r[key]) == g)) for g in grid])
    bp, mp = get(est, "bias"), get(est, "mcse_bias")
    bb, mb = get("backdoor_proxy", "bias"), get("backdoor_proxy", "mcse_bias")
    point = cross(grid, np.abs(bp), np.abs(bb))
    sims = [cross(grid, np.abs(rng.normal(bp, mp)), np.abs(rng.normal(bb, mb)))
            for _ in range(draws)]
    lo, hi = np.nanpercentile(sims, [2.5, 97.5])
    return point, lo, hi


def main() -> None:
    rng = np.random.default_rng(2026)
    g_rows = load("proximal_robustness.csv")
    d_rows = load("proximal_robustness_w.csv")
    out = []
    for domain, cfg in uc.DOMAINS.items():
        var_u = cfg["age_to_sev"] ** 2 + 1.0
        sd_w = float(np.sqrt(cfg["sev_to_w"] ** 2 * var_u + 1.0))
        for est, lab in (("proximal", "2sls"), ("proximal_dr", "dr")):
            gp, gl, gh = ci(g_rows, "gamma_z_to_y", domain, est, rng)
            dp, dl, dh = ci(d_rows, "delta_a_to_w", domain, est, rng)
            out.append(dict(domain=domain, estimator=lab,
                            gamma_star=round(gp, 3), gamma_lo=round(gl, 3), gamma_hi=round(gh, 3),
                            gamma_over_betaU=round(gp / cfg["sev_to_out"], 3),
                            delta_star=round(dp, 3), delta_lo=round(dl, 3), delta_hi=round(dh, 3),
                            delta_over_sdW=round(dp / sd_w, 3), sd_w=round(sd_w, 3)))
    with open(uc.RESULTS / "proximal_crossing_ci.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
