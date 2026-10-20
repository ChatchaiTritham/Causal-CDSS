"""One audit run per domain, emitted as a CDS Hooks 2.0 service response and a FHIR R5 Bundle.

Runs Algorithm 1 of the paper once, on replication 0 of the main study (same seed as
scripts/unmeasured_confounding.py), using only recorded variables.

What the code does and does not do (v1.0.4):
* gamma* and delta* are DESIGN-STAGE REFERENCE VALUES: the doubly robust (DR) crossing points
  of this paper's generator, read from results/proximal_crossing.csv. They are not computed
  from the audited records; an auditor with a different design must rerun the sweeps.
* Each domain's output is a CDS Hooks 2.0 response {"cards": [Card]}. Structured audit numbers
  travel in card.extension["org.causal-cdss.audit"], the specification's extension mechanism.
  The hook (order-select) and the prefetch template are in a separate discovery document.
  tests/test_audit_card.py validates every response against a JSON schema of the 2.0 Card.
* A safety veto: the card's indicator is "critical" when the rule withholds treatment from
  more than WITHHOLD_TOL of patients flagged by the recorded indication marker (W in the top
  30%), whatever the proximal verdict.
* The FHIR R5 Bundle (Library with the declared graph, Observation with an example LOINC code,
  Evidence, GuidanceResponse) is an illustration of the mapping. It has a structural check in
  the tests; it was not validated with the HL7 FHIR validator.
* The interventional truth is never written to any output (simulation-only).

Outputs: results/audit_card.json (domain -> CDS Hooks response),
         results/cds_services_discovery.json, results/fhir_bundle_<domain>.json
Usage:   python scripts/audit_card.py [--n 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import uuid
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unmeasured_confounding as uc  # noqa: E402

F_MIN = 16.0          # Stock-Yogo 10% maximal-size critical value for one instrument is 16.38;
                      # the paper uses 16 as a design-specific threshold (see Algorithm 1)
WITHHOLD_TOL = 0.10   # declared tolerance for withholding treatment from indicated patients
CROSSING_SOURCE = ("design-stage DR crossing of this paper's generator "
                   "(results/proximal_crossing.csv); not computed from the audited records")
DAG_DOT = ("digraph audit { age -> severity; age -> mortality; severity -> treatment; "
           "severity -> mortality; severity -> Z; severity -> W; treatment -> mortality; "
           "severity [latent=true]; }")


def discovery() -> dict:
    return {"services": [{
        "hook": "order-select",
        "id": "causal-cdss-audit",
        "title": "Causal audit of a treatment recommender",
        "description": "Returns the audit verdict (Algorithm 1) for the recommender serving this order.",
        "prefetch": {"patient": "Patient/{{context.patientId}}",
                     "lactate": "Observation?patient={{context.patientId}}&code=http://loinc.org|2524-7"},
    }]}


def fhir_bundle(domain: str, dr, ci, verdict: str) -> dict:
    lib_id, ev_id = f"declared-graph-{domain}", f"ate-proximal-dr-{domain}"
    return {"resourceType": "Bundle", "type": "collection", "entry": [
        {"fullUrl": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, lib_id)}", "resource": {
            "resourceType": "Library", "id": lib_id, "status": "draft",
            "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/library-type",
                                 "code": "model-definition"}]},
            "content": [{"contentType": "text/vnd.graphviz",
                         "data": base64.b64encode(DAG_DOT.encode()).decode()}]}},
        {"fullUrl": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'obs-' + domain)}", "resource": {
            "resourceType": "Observation", "id": f"proxy-example-{domain}", "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "2524-7",
                                 "display": "Lactate [Moles/volume] in Serum or Plasma"}]},
            "valueQuantity": {"value": 3.1, "unit": "mmol/L", "system": "http://unitsofmeasure.org",
                              "code": "mmol/L"}}},
        {"fullUrl": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, ev_id)}", "resource": {
            "resourceType": "Evidence", "id": ev_id, "status": "draft",
            "variableDefinition": [{"variableRole": "outcome",
                                    "description": "mortality, risk difference treat vs withhold"}],
            "statistic": [{"statisticType": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/statistic-type",
                                                         "code": "0000424", "display": "Risk Difference"}]},
                           "quantity": {"value": round(dr, 3)},
                           "attributeEstimate": [{"type": {"text": "95% confidence interval"},
                                                  "level": 0.95,
                                                  "range": {"low": {"value": round(ci[0], 3)},
                                                            "high": {"value": round(ci[1], 3)}}}]}]}},
        {"fullUrl": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'gr-' + domain)}", "resource": {
            "resourceType": "GuidanceResponse", "id": f"audit-verdict-{domain}",
            "moduleUri": "https://github.com/ChatchaiTritham/Causal-CDSS",
            "status": "success", "note": [{"text": verdict}]}},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cross = {r["domain"]: r for r in csv.DictReader(open(uc.RESULTS / "proximal_crossing.csv"))}
    responses = {}
    for domain, cfg in uc.DOMAINS.items():
        r = 0
        rng = np.random.default_rng(a.seed + 1000 * r + uc.domain_offset(domain, 997))
        d = uc.simulate(cfg, a.n, rng)
        rec = {k: d[k] for k in ("age", "z", "w", "a", "y")}  # severity withheld
        f_stat = uc.first_stage_f(rec)
        bd, bd_se = uc.backdoor_proxy_se(rec)
        before = uc.BRIDGE_FAILS["count"]
        dr, dr_se = uc.proximal_dr_only(rec)
        q_ok = uc.BRIDGE_FAILS["count"] == before
        pi = uc.policy_knowledge(rec, cfg)
        o = uc.ope_estimates(rec, pi, uc.fit_nuisances(rec, "logistic_recorded"))
        indicated = rec["w"] > np.quantile(rec["w"], 0.7)
        withhold = float(np.mean(pi[indicated] == 0))
        if f_stat < F_MIN or not q_ok:
            strategy = "inconclusive (weak proxies or unsolved treatment bridge)"
        else:
            strategy = "proximal (DR form), if elicited violations are below half the reference crossing"
        veto = withhold > WITHHOLD_TOL
        gs, ds = float(cross[domain]["gamma_star_dr"]), float(cross[domain]["delta_star_dr"])
        ci = (dr - 1.96 * dr_se, dr + 1.96 * dr_se)
        detail = (f"**Strategy:** {strategy}.  \n"
                  f"First-stage F = {f_stat:.0f} (design-specific threshold {F_MIN:.0f}; "
                  "use weak-identification-robust intervals below F = 104.7).  \n"
                  f"ATE, proximal DR: {dr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]; "
                  f"proxy-adjusted: {bd:.3f} [{bd - 1.96 * bd_se:.3f}, {bd + 1.96 * bd_se:.3f}].  \n"
                  f"Reference crossing points (design stage, DR): gamma* = {gs:.2f}, delta* = {ds:.2f}.  \n"
                  f"Rule withholds treatment from {withhold:.0%} of indication-flagged patients "
                  f"(tolerance {WITHHOLD_TOL:.0%}).  \n"
                  f"Knowledge-driven rule value (standard DR) {o['dr']:.3f}: unverified, confounder not recorded.")
        card = {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"causal-cdss/{domain}/{a.seed}")),
            "summary": (f"Causal audit ({domain}): "
                        + ("SAFETY VETO, rule withholds indicated therapy" if veto else strategy))[:140],
            "detail": detail,
            "indicator": "critical" if veto else ("info" if f_stat >= F_MIN and q_ok else "warning"),
            "source": {"label": "Causal-CDSS audit protocol (Algorithm 1)",
                       "url": "https://github.com/ChatchaiTritham/Causal-CDSS",
                       "topic": {"system": "https://github.com/ChatchaiTritham/Causal-CDSS",
                                 "code": "causal-audit", "display": "Causal audit verdict"}},
            "extension": {"org.causal-cdss.audit": {
                "domain": domain,
                "intended_population": "synthetic audit cohort; method comparison only",
                "first_stage_F": round(f_stat, 1),
                "treatment_bridge_converged": bool(q_ok),
                "ate_proximal_dr": [round(dr, 3), round(ci[0], 3), round(ci[1], 3)],
                "ate_proxy_adjusted": [round(bd, 3), round(bd - 1.96 * bd_se, 3), round(bd + 1.96 * bd_se, 3)],
                "gamma_star_reference": gs, "delta_star_reference": ds,
                "crossing_source": CROSSING_SOURCE,
                "policy": "knowledge-driven rule",
                "policy_value_dr": round(o["dr"], 3),
                "policy_ess_fraction": round(o["ess_frac"], 3),
                "policy_agreement": round(o["agreement"], 3),
                "policy_value_status": "unverified: confounder not recorded",
                "withhold_share_indicated": round(withhold, 3),
                "safety_veto": bool(veto),
                "subgroup_table": "results/subgroup_audit.csv",
                "reaudit_triggers": ["declared DAG or proxy definition changed",
                                     f"first-stage F below {F_MIN:.0f}",
                                     "treatment-assignment practice or guideline changed",
                                     "proxy loading or age case-mix shift at the deployment site "
                                     "(see results/r2_site_shift.csv)"],
                "fhir_bundle": f"results/fhir_bundle_{domain}.json",
            }},
        }
        responses[domain] = {"cards": [card]}
        (uc.RESULTS / f"fhir_bundle_{domain}.json").write_text(
            json.dumps(fhir_bundle(domain, dr, ci, strategy), indent=2), encoding="utf-8")
    (uc.RESULTS / "audit_card.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")
    (uc.RESULTS / "cds_services_discovery.json").write_text(json.dumps(discovery(), indent=2),
                                                            encoding="utf-8")
    print(json.dumps(responses["sepsis"], indent=2))
    print("wrote results/audit_card.json, results/cds_services_discovery.json, results/fhir_bundle_*.json")


if __name__ == "__main__":
    main()
