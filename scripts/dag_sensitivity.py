"""DAG sensitivity analysis via single-edge perturbations (seed = 42).

For each of the three clinical-domain SCMs defined in ``run_all.py``
(sepsis, ARDS, ACS -- the only domains actually modeled), this script
perturbs the causal graph one edge at a time:

  * SINGLE-EDGE DELETION -- zero out the structural coefficient for one of
    the 5 edges present in the canonical 4-node DAG
    (age->severity, severity->treatment, severity->outcome, age->outcome,
    treatment->outcome), rebuild the SCM, and recompute the true causal ATE
    by do-calculus.
  * SINGLE-EDGE ADDITION -- add the one edge NOT already present in the
    topological closure of {age, severity, treatment, outcome}
    (age->treatment) with a moderate coefficient, rebuild the SCM, and
    recompute the ATE.

For each perturbation we record the resulting ATE and its deviation from
the unperturbed (baseline) true ATE for that domain. This directly
supports the manuscript's DAG-sensitivity table with numbers that are
actually produced by the vendored SCM/CausalGraph classes -- nothing here
is hardcoded.

Total perturbations = 3 domains x 6 (5 deletions + 1 addition) = 18.
(Not "20" -- that count in an earlier manuscript draft did not correspond
to any experiment; this script reports however many perturbations are
actually well defined for the modeled DAGs.)

Run:
    PYTHONPATH=src python scripts/dag_sensitivity.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basics_cdss.causal import CausalGraph, StructuralCausalModel, compute_ate  # noqa: E402
from run_all import DOMAINS, _sigmoid  # noqa: E402

SEED = 42
N_INT = 8000
OUT = ROOT / "results" / "dag_sensitivity.csv"

# Canonical edges of the 4-node confounded DAG used by every domain in
# run_all.py. The one edge NOT present (age->treatment) is the sole
# addition perturbation available while preserving the topological order
# age -> severity -> treatment -> outcome.
EDGES_PRESENT = [
    ("age", "severity"),
    ("severity", "treatment"),
    ("severity", "outcome"),
    ("age", "outcome"),
    ("treatment", "outcome"),
]
EDGE_ADDABLE = ("age", "treatment")
ADDED_EDGE_COEF = 0.50  # moderate magnitude, comparable to sev_to_treat (0.9-1.4)


def build_scm_perturbed(cfg, seed, delete_edge=None, add_edge_coef=None):
    """Rebuild the domain SCM with one edge deleted or the extra edge added.

    Mirrors run_all.build_scm exactly, except the structural coefficient of
    ``delete_edge`` is zeroed (if given) or an age->treatment term with
    coefficient ``add_edge_coef`` is added to the treatment mechanism (if
    given). Only one of the two may be set.
    """
    a2s = 0.0 if delete_edge == ("age", "severity") else cfg["age_to_sev"]
    s2t = 0.0 if delete_edge == ("severity", "treatment") else cfg["sev_to_treat"]
    s2o = 0.0 if delete_edge == ("severity", "outcome") else cfg["sev_to_out"]
    a2o = 0.0 if delete_edge == ("age", "outcome") else cfg["age_to_out"]
    t2o = 0.0 if delete_edge == ("treatment", "outcome") else cfg["treat_to_out"]
    ti = cfg["treat_intercept"]
    oi = cfg["out_intercept"]
    a2t = add_edge_coef if add_edge_coef is not None else 0.0

    g = CausalGraph()
    for node in ("age", "severity", "treatment", "outcome"):
        g.add_node(node)
    if a2s != 0.0 or delete_edge != ("age", "severity"):
        g.add_edge("age", "severity")
    if s2t != 0.0 or delete_edge != ("severity", "treatment"):
        g.add_edge("severity", "treatment")
    if s2o != 0.0 or delete_edge != ("severity", "outcome"):
        g.add_edge("severity", "outcome")
    if a2o != 0.0 or delete_edge != ("age", "outcome"):
        g.add_edge("age", "outcome")
    if t2o != 0.0 or delete_edge != ("treatment", "outcome"):
        g.add_edge("treatment", "outcome")
    if a2t != 0.0:
        g.add_edge("age", "treatment")

    sev_parents = [] if delete_edge == ("age", "severity") else ["age"]
    treat_parents = [x for x in ("severity", "age")
                     if not (x == "severity" and delete_edge == ("severity", "treatment"))
                     and not (x == "age" and a2t == 0.0)]
    out_parents = [x for x in ("severity", "age", "treatment")
                   if not (x == "severity" and delete_edge == ("severity", "outcome"))
                   and not (x == "age" and delete_edge == ("age", "outcome"))
                   and not (x == "treatment" and delete_edge == ("treatment", "outcome"))]

    scm = StructuralCausalModel(g, seed=seed, default_mechanisms=False)
    scm.add_mechanism(
        "age", [],
        function=lambda p, n: n,
        noise_distribution=lambda r: r.normal(0, 1),
    )
    scm.add_mechanism(
        "severity", sev_parents,
        function=lambda p, n, a2s=a2s: a2s * p.get("age", 0.0) + n,
        noise_distribution=lambda r: r.normal(0, 1),
    )
    scm.add_mechanism(
        "treatment", treat_parents,
        function=lambda p, n, s2t=s2t, ti=ti, a2t=a2t: 1.0
        if _sigmoid(s2t * p.get("severity", 0.0) + a2t * p.get("age", 0.0) + ti) > n else 0.0,
        noise_distribution=lambda r: r.uniform(0, 1),
    )
    scm.add_mechanism(
        "outcome", out_parents,
        function=lambda p, n, s2o=s2o, a2o=a2o, t2o=t2o, oi=oi: 1.0
        if _sigmoid(s2o * p.get("severity", 0.0) + a2o * p.get("age", 0.0)
                    + t2o * p.get("treatment", 0.0) + oi) > n
        else 0.0,
        noise_distribution=lambda r: r.uniform(0, 1),
    )
    return scm


def ate_for(scm):
    result = compute_ate(scm, "treatment", "outcome", treatment_values=[0, 1], n_samples=N_INT)
    return float(result["ate"])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[*] DAG sensitivity driver (seed={SEED})")
    for name, cfg in DOMAINS.items():
        baseline_scm = build_scm_perturbed(cfg, seed=SEED + 1)
        baseline_ate = ate_for(baseline_scm)
        print(f"[*] domain: {name}  baseline true ATE = {baseline_ate:+.4f}")

        for (u, v) in EDGES_PRESENT:
            pert_scm = build_scm_perturbed(cfg, seed=SEED + 1, delete_edge=(u, v))
            pert_ate = ate_for(pert_scm)
            dev = pert_ate - baseline_ate
            rows.append({
                "domain": name, "perturbation": "delete_edge", "edge": f"{u}->{v}",
                "baseline_ate": baseline_ate, "perturbed_ate": pert_ate,
                "ate_deviation": dev, "abs_ate_deviation": abs(dev),
            })
            print(f"    delete {u}->{v:<10s} ATE={pert_ate:+.4f}  dev={dev:+.4f}")

        u, v = EDGE_ADDABLE
        pert_scm = build_scm_perturbed(cfg, seed=SEED + 1, add_edge_coef=ADDED_EDGE_COEF)
        pert_ate = ate_for(pert_scm)
        dev = pert_ate - baseline_ate
        rows.append({
            "domain": name, "perturbation": "add_edge", "edge": f"{u}->{v}",
            "baseline_ate": baseline_ate, "perturbed_ate": pert_ate,
            "ate_deviation": dev, "abs_ate_deviation": abs(dev),
        })
        print(f"    add    {u}->{v:<10s} ATE={pert_ate:+.4f}  dev={dev:+.4f}")

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "perturbation", "edge", "baseline_ate", "perturbed_ate",
                    "ate_deviation", "abs_ate_deviation"])
        for r in rows:
            w.writerow([r["domain"], r["perturbation"], r["edge"],
                        f"{r['baseline_ate']:.4f}", f"{r['perturbed_ate']:.4f}",
                        f"{r['ate_deviation']:+.4f}", f"{r['abs_ate_deviation']:.4f}"])
    print(f"[OK] {OUT}  ({len(rows)} perturbations)")


if __name__ == "__main__":
    main()
