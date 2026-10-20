"""The audit card is a CDS Hooks 2.0 response, its crossing source is declared, and no
simulation-only truth leaks into it (v1.0.4)."""
import json
import os

import jsonschema
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def _load(name):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def test_every_response_validates_against_cds_hooks_2_schema():
    schema = json.load(open(os.path.join(HERE, "schemas", "cds_hooks_2_0_response.schema.json"),
                            encoding="utf-8"))
    for dom, resp in _load("audit_card.json").items():
        jsonschema.validate(resp, schema)
        assert len(resp["cards"]) == 1


def test_hook_lives_in_discovery_not_in_response():
    disc = _load("cds_services_discovery.json")
    assert disc["services"][0]["hook"] == "order-select"
    assert "prefetch" in disc["services"][0]
    for resp in _load("audit_card.json").values():
        assert "hook" not in resp


def test_crossing_points_are_declared_design_stage_dr_values():
    import csv
    cross = {r["domain"]: r for r in csv.DictReader(open(os.path.join(RES, "proximal_crossing.csv")))}
    for dom, resp in _load("audit_card.json").items():
        a = resp["cards"][0]["extension"]["org.causal-cdss.audit"]
        assert "design-stage DR" in a["crossing_source"]
        assert a["gamma_star_reference"] == pytest.approx(float(cross[dom]["gamma_star_dr"]))
        assert a["delta_star_reference"] == pytest.approx(float(cross[dom]["delta_star_dr"]))


def test_no_interventional_truth_on_card():
    text = json.dumps(_load("audit_card.json")).lower()
    assert "true_ate" not in text and "truth" not in text


def test_safety_veto_sets_critical_indicator():
    for resp in _load("audit_card.json").values():
        c = resp["cards"][0]
        a = c["extension"]["org.causal-cdss.audit"]
        assert (c["indicator"] == "critical") == a["safety_veto"]


def test_fhir_bundles_are_structurally_complete():
    for dom in ("sepsis", "ards", "cardiac"):
        b = _load(f"fhir_bundle_{dom}.json")
        types = [e["resource"]["resourceType"] for e in b["entry"]]
        assert b["resourceType"] == "Bundle" and b["type"] == "collection"
        assert sorted(types) == ["Evidence", "GuidanceResponse", "Library", "Observation"]
        obs = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Observation")
        assert obs["code"]["coding"][0] == {"system": "http://loinc.org", "code": "2524-7",
                                            "display": "Lactate [Moles/volume] in Serum or Plasma"}
