"""One audit run per domain, emitted as the payload a CDS Hooks card could carry.

Runs Algorithm 1 of the paper once, on replication 0 of the main study (same seed as
scripts/unmeasured_confounding.py), using only recorded variables. The crossing points are read
from results/proximal_crossing.csv; the decision rule follows Table "decision" of the paper.
The JSON is an illustration of the knowledge artefact, not a validated CDS Hooks service.

Output: results/audit_card.json
Usage:  python scripts/audit_card.py [--n 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402

F_MIN = 10.0  # "first-stage F in the tens or above" (decision table)


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
        strategy = "proximal (doubly robust)" if f_stat >= F_MIN else "inconclusive: weak proxies"
        cards.append({
            "domain": domain,
            "summary": f"Causal audit: {strategy}",
            "indicator": "info" if f_stat >= F_MIN else "warning",
            "detail": {
                "first_stage_F": round(f_stat, 1),
                "ate_proximal_dr": [round(dr, 3), round(dr - 1.96 * dr_se, 3),
                                    round(dr + 1.96 * dr_se, 3)],
                "ate_proxy_adjusted": [round(bd, 3), round(bd - 1.96 * bd_se, 3),
                                       round(bd + 1.96 * bd_se, 3)],
                "gamma_star": float(cross[domain]["gamma_star_dr"]),
                "delta_star": float(cross[domain]["delta_star_dr"]),
                "policy": "knowledge-driven rule",
                "policy_value_dr": round(o["dr"], 3),
                "policy_ess_fraction": round(o["ess_frac"], 3),
                "policy_agreement": round(o["agreement"], 3),
                "policy_value_status": "unverified: confounder not recorded",
            },
            "source": {"label": "Causal-CDSS audit protocol (Algorithm 1)",
                       "url": "https://github.com/ChatchaiTritham/Causal-CDSS"},
        })
    out = uc.RESULTS / "audit_card.json"
    out.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    print(json.dumps(cards[0], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
