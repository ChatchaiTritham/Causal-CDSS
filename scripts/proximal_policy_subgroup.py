"""Proximal policy-value estimation and an age-subgroup audit on the audit cohort.

Part 1 -- proximal off-policy evaluation. For a rule pi(.) the doubly robust proximal value
estimator uses the same bridges as the proximal ATE (outcome bridge h, treatment bridge q):

    psi_pi = h(W, pi, X) + 1{A = pi} q(Z, A, X) (Y - h(W, A, X))

V(pi) is identified by this formula when pi depends on X (age) only. When pi also reads Z or W,
as both recommenders in the paper do, identification needs further conditions (Qi, Miao & Zhang,
JASA 2024); the estimator is still computed for them and its coverage reported, so the reader
sees what happens rather than what is assumed. An age-only rule (treat if age > 0) is added as
the identified reference case. The standard DR estimator (logistic nuisances on recorded covariates) is computed
on the same cohorts for comparison.

Part 2 -- subgroup audit by age tertile (cut points of N(0,1): +/-0.4307). Within each tertile:
proxy-adjusted and DR proximal ATE against the tertile's interventional truth, and, for the
two recommenders, the treated fraction, the share of patients with true absolute risk reduction
above KB_MIN_REDUCTION who are treated (benefit recall), and the true absolute reduction in
adverse-outcome risk relative to treating no one.

Seeds follow scripts/unmeasured_confounding.py.
Outputs: results/proximal_policy_value.csv, results/subgroup_audit.csv
Usage:   python scripts/proximal_policy_subgroup.py [--reps 200] [--n 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402

CUT = 0.4307  # tertiles of the standard normal age score
GROUPS = ("young", "middle", "old")
POLICIES = ("age_rule", "treat_all", "treat_none", "cdss_top50", "cdss_top25", "knowledge")


def group_of(age: np.ndarray) -> np.ndarray:
    return np.where(age < -CUT, 0, np.where(age > CUT, 2, 1))


def subset(d: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in d.items()}


def true_reduction(cfg, d):
    base = cfg["sev_to_out"] * d["severity"] + cfg["age_to_out"] * d["age"] + cfg["out_intercept"]
    return uc.sigmoid(base) - uc.sigmoid(base + cfg["treat_to_out"])


def subgroup_truth(cfg, n, rng):
    age = rng.normal(0, 1, n)
    sev = cfg["age_to_sev"] * age + rng.normal(0, 1, n)
    base = cfg["sev_to_out"] * sev + cfg["age_to_out"] * age + cfg["out_intercept"]
    eff = uc.sigmoid(base + cfg["treat_to_out"]) - uc.sigmoid(base)
    g = group_of(age)
    return [float(eff[g == k].mean()) for k in range(3)]


def proximal_value(d, pi, b, q):
    h = lambda a: b[0] + b[1] * a + b[2] * d["w"] + b[3] * d["age"]
    psi = h(pi) + (d["a"] == pi) * q * (d["y"] - h(d["a"]))
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


def run(job):
    domain, n, reps, seed = job
    cfg = uc.DOMAINS[domain]
    sub_truth = subgroup_truth(cfg, 200_000, np.random.default_rng(seed + 999))
    pv = {(p, k): [] for p in POLICIES for k in ("truth", "prox", "prox_cov", "dr", "dr_cov")}
    sg = {}
    for r in range(reps):
        off = seed + 1000 * r + uc.domain_offset(domain, 997)
        d = uc.simulate(cfg, n, np.random.default_rng(off))
        train = uc.simulate(cfg, n, np.random.default_rng(off + 500_000))
        pols = uc.make_policies(d, train, cfg)
        pols["age_rule"] = (d["age"] > 0).astype(float)
        b = uc.outcome_bridge(d)
        q = uc.treatment_bridge(d)
        nuis = uc.fit_nuisances(d, "logistic_recorded")
        for p in POLICIES:
            pi = pols[p]
            v = uc.true_policy_value(cfg, d, pi)
            th, se = proximal_value(d, pi, b, q)
            o = uc.ope_estimates(d, pi, nuis)
            pv[(p, "truth")].append(v)
            pv[(p, "prox")].append(th)
            pv[(p, "prox_cov")].append(float(abs(th - v) <= 1.96 * se))
            pv[(p, "dr")].append(o["dr"])
            pv[(p, "dr_cov")].append(float(abs(o["dr"] - v) <= 1.96 * o["dr_se"]))
        # subgroup audit
        g = group_of(d["age"])
        red = true_reduction(cfg, d)
        for k in range(3):
            m = g == k
            ds = subset(d, m)
            for name, fn in (("backdoor_proxy", uc.backdoor_proxy_se),
                             ("proximal_dr", uc.proximal_dr_only)):
                th, se = fn(ds)
                acc = sg.setdefault((k, "ate", name), {"est": [], "cov": []})
                acc["est"].append(th)
                acc["cov"].append(float(abs(th - sub_truth[k]) <= 1.96 * se))
            benef = red[m] > uc.KB_MIN_REDUCTION
            for p in ("knowledge", "cdss_top50"):
                pi = pols[p][m]
                gain = (uc.true_policy_value(cfg, ds, np.zeros(m.sum()))
                        - uc.true_policy_value(cfg, ds, pi))
                acc = sg.setdefault((k, "policy", p), {"treated": [], "recall": [], "gain": [],
                                                       "benef": []})
                acc["treated"].append(float(pi.mean()))
                acc["recall"].append(float(pi[benef].mean()) if benef.any() else np.nan)
                acc["gain"].append(gain)
                acc["benef"].append(float(benef.mean()))

    rows_pv = []
    for p in POLICIES:
        t = np.array(pv[(p, "truth")])
        row = {"domain": domain, "policy": p, "identified": int(p in ("age_rule", "treat_all", "treat_none")),
               "true_value": round(float(t.mean()), 4)}
        for k in ("prox", "dr"):
            e = np.array(pv[(p, k)])
            c = float(np.mean(pv[(p, k + "_cov")]))
            row[f"{k}_bias"] = round(float(np.mean(e - t)), 4)
            row[f"{k}_rmse"] = round(float(np.sqrt(np.mean((e - t) ** 2))), 4)
            row[f"{k}_coverage"] = round(c, 3)
            row[f"{k}_mcse_coverage"] = round(float(np.sqrt(c * (1 - c) / reps)), 3)
        row["reps"] = reps
        rows_pv.append(row)

    rows_sg = []
    for k in range(3):
        for name in ("backdoor_proxy", "proximal_dr"):
            a = sg[(k, "ate", name)]
            e = np.array(a["est"])
            ok = np.isfinite(e)
            rows_sg.append({"domain": domain, "age_group": GROUPS[k], "kind": "ate",
                            "item": name, "truth": round(sub_truth[k], 4),
                            "bias": round(float(e[ok].mean() - sub_truth[k]), 4),
                            "coverage": round(float(np.mean(np.array(a["cov"])[ok])), 3),
                            "treated": "", "benefit_share": "", "benefit_recall": "",
                            "risk_reduction": "", "reps": int(ok.sum())})
        for p in ("knowledge", "cdss_top50"):
            a = sg[(k, "policy", p)]
            rows_sg.append({"domain": domain, "age_group": GROUPS[k], "kind": "policy",
                            "item": p, "truth": "", "bias": "", "coverage": "",
                            "treated": round(float(np.mean(a["treated"])), 3),
                            "benefit_share": round(float(np.mean(a["benef"])), 3),
                            "benefit_recall": round(float(np.nanmean(a["recall"])), 3),
                            "risk_reduction": round(float(np.mean(a["gain"])), 4), "reps": reps})
    print(f"  {domain} done; bridge non-convergence {uc.BRIDGE_FAILS['count']}/"
          f"{uc.BRIDGE_FAILS['calls']}", flush=True)
    return rows_pv, rows_sg, uc.BRIDGE_FAILS["count"], uc.BRIDGE_FAILS["calls"]


def write(rows, name):
    with open(uc.RESULTS / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    jobs = [(dm, a.n, a.reps, a.seed) for dm in uc.DOMAINS]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(run, jobs))
    write([r for o in out for r in o[0]], "proximal_policy_value.csv")
    write([r for o in out for r in o[1]], "subgroup_audit.csv")
    fails, calls = sum(o[2] for o in out), sum(o[3] for o in out)
    with open(uc.RESULTS / "bridge_convergence_policy_subgroup.txt", "w", encoding="utf-8") as fh:
        fh.write(f"treatment-bridge non-convergence: {fails} of {calls} arm-level solves\n")


if __name__ == "__main__":
    main()
