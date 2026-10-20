"""One audit run per domain, emitted as the payload a CDS Hooks card could carry.

Runs Algorithm 1 of the paper once, on replication 0 of the main study (same seed as
scripts/unmeasured_confounding.py), using only recorded variables. The crossing points are read
from results/proximal_crossing.csv; the decision rule follows Table "decision" of the paper.
The JSON is an illustration of the knowledge artefact, not a validated CDS Hooks service.
Each card follows the CDS Hooks 2.0 Card schema (uuid, summary <= 140 chars, markdown
``detail``, indicator, source with label/url/topic); the structured numbers travel in a
separate ``audit`` object beside the card (as a FHIR Evidence/GuidanceResponse would), and the
interventional truth is never written to the card (simulation-only).

Output: results/audit_card.json
Usage:  python scripts/audit_card.py [--n 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402

F_MIN = 16.0  # lowest first-stage F at which the proximal interval held (weak-proxy sweep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cross = {r["domain"]: r for r in csv.DictReader(open(uc.RESULTS / "proximal_crossing.csv"))}
    cards = []
    for domain, cfg in uc.DOMAINS.items():
        r = 0
        rng = np.random.default_rng(a.seed + 1000 * r + uc.domain_offset(domain, 997))
        d = uc.simulate(cfg, a.n, rng)
        train = uc.simulate(cfg, a.n, np.random.default_rng(
            a.seed + 1000 * r + uc.domain_offset(domain, 997) + 500_000))
        rec = {k: d[k] for k in ("age", "z", "w", "a", "y")}  # severity withheld
        f_stat = uc.first_stage_f(rec)
        bd, bd_se = uc.backdoor_proxy_se(rec)
        dr, dr_se = uc.proximal_dr_only(rec)
        pi = uc.policy_knowledge(rec, cfg)
        o = uc.ope_estimates(rec, pi, uc.fit_nuisances(rec, "logistic_recorded"))
        strategy = "proximal (DR form)" if f_stat >= F_MIN else "inconclusive: weak proxies"
        gs, ds = float(cross[domain]["gamma_star_dr"]), float(cross[domain]["delta_star_dr"])
        detail = (f"**Strategy:** {strategy}.  \n"
                  f"First-stage F = {f_stat:.0f} (threshold {F_MIN:.0f}).  \n"
                  f"ATE, proximal DR: {dr:.3f} [{dr - 1.96 * dr_se:.3f}, {dr + 1.96 * dr_se:.3f}]; "
                  f"proxy-adjusted: {bd:.3f} [{bd - 1.96 * bd_se:.3f}, {bd + 1.96 * bd_se:.3f}].  \n"
                  f"Design-stage crossing points for this graph: gamma* = {gs:.2f}, "
                  f"delta* = {ds:.2f} (not estimated from these records).  \n"
                  f"Knowledge-driven rule value (standard DR) {o['dr']:.3f}: "
                  "unverified, confounder not recorded.")
        cards.append({
            "hook": "order-select",
            "card": {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"causal-cdss/{domain}/{a.seed}")),
                "summary": f"Causal audit ({domain}): {strategy}"[:140],
                "detail": detail,
                "indicator": "info" if f_stat >= F_MIN else "warning",
                "source": {"label": "Causal-CDSS audit protocol (Algorithm 1)",
                           "url": "https://github.com/ChatchaiTritham/Causal-CDSS",
                           "topic": {"system": "https://github.com/ChatchaiTritham/Causal-CDSS",
                                     "code": "causal-audit", "display": "Causal audit verdict"}},
            },
            "audit": {
                "domain": domain,
                "intended_population": "synthetic audit cohort; method comparison only",
                "first_stage_F": round(f_stat, 1),
                "ate_proximal_dr": [round(dr, 3), round(dr - 1.96 * dr_se, 3),
                                    round(dr + 1.96 * dr_se, 3)],
                "ate_proxy_adjusted": [round(bd, 3), round(bd - 1.96 * bd_se, 3),
                                       round(bd + 1.96 * bd_se, 3)],
                "gamma_star_design_stage": gs,
                "delta_star_design_stage": ds,
                "policy": "knowledge-driven rule",
                "policy_value_dr": round(o["dr"], 3),
                "policy_ess_fraction": round(o["ess_frac"], 3),
                "policy_agreement": round(o["agreement"], 3),
                "policy_value_status": "unverified: confounder not recorded",
                "subgroup_table": "results/subgroup_audit.csv",
                "reaudit_triggers": ["declared DAG or proxy definition changed",
                                     f"first-stage F below {F_MIN:.0f}",
                                     "treatment-assignment practice or guideline changed",
                                     "case-mix shift between audit and deployment site"],
                "fhir_mapping": {"declared_model": "Library (attached graph)",
                                 "proxies": "Observation", "verdict": "GuidanceResponse",
                                 "estimate": "Evidence"},
            },
        })
    out = uc.RESULTS / "audit_card.json"
    out.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    print(json.dumps(cards[0], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
