"""Reproducibility unit tests for Causal-CDSS (seed = 42).

These tests LOCK the verified key results of the manuscript so that any
regression in the deterministic pipeline is caught. Expected values are read
from / mirror the committed artifacts under ``results/``:

  * ``results/model_comparison.csv``  -- the five-model associational-vs-causal
    table (LR, RF, XGBoost, LSTM, TCN).
  * ``results/model_comparison.json`` -- same numbers, full precision.
  * ``results/ate_by_domain.csv``     -- per-domain true / naive / adjusted ATE.

The locked facts encode the paper's central thesis:

  1. Every model is a STRONG associational predictor (accuracy ~= 0.745-0.749).
  2. The NAIVE causal-effect accuracy COLLAPSES to 0.0 for ALL five models
     (the confounded, sign-flipped estimate is worse than useless).
  3. The ADJUSTED (backdoor / do-intervention) causal accuracy RECOVERS
     (mean ~= 0.846).
  4. The naive ATE is POSITIVE (sign-flipped) while the true ATE is NEGATIVE
     (protective, ~= -0.14) -- confounding inverts the apparent treatment sign.

All tolerances use ``pytest.approx`` with values sized to the magnitude of the
quantity being checked.

Run:
    pytest tests/test_reproducibility.py
    pytest -m slow tests/test_reproducibility.py   # include the live re-run
"""

import csv
import importlib
import json
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
CSV_PATH = RESULTS / "model_comparison.csv"
JSON_PATH = RESULTS / "model_comparison.json"
ATE_BY_DOMAIN = RESULTS / "ate_by_domain.csv"

MODELS = ["LR", "RF", "XGBoost", "LSTM", "TCN"]

# --------------------------------------------------------------------------- #
# Committed reference values (read from results/ on 2026-06-20). These are the
# verified, deterministic seed-42 numbers; tests assert the artifacts still
# match them and that a live re-run reproduces them.
# --------------------------------------------------------------------------- #
EXPECTED_ASSOC_ACC = {
    "LR": 0.7486,
    "RF": 0.7457,
    "XGBoost": 0.7429,
    "LSTM": 0.7492,
    "TCN": 0.7474,
}
EXPECTED_ADJ_ACC = {
    "LR": 0.9409,
    "RF": 0.4717,
    "XGBoost": 0.8901,
    "LSTM": 0.9643,
    "TCN": 0.9625,
}
EXPECTED_NAIVE_ATE = {  # all POSITIVE -> sign-flipped vs the protective truth
    "LR": 0.1580,
    "RF": 0.1986,
    "XGBoost": 0.1622,
    "LSTM": 0.1570,
    "TCN": 0.1506,
}
TRUE_ATE = -0.1425          # negative (protective); same for every model row
MEAN_ASSOC_ACC = 0.7468
MEAN_ADJ_ACC = 0.8459

# per-domain (results/ate_by_domain.csv): true ATE < 0, naive ATE > 0
EXPECTED_DOMAIN_ATE = {
    #          true_ate, naive_ate
    "sepsis": (-0.1489, 0.0985),
    "ards":   (-0.1131, 0.0614),
    "acs":    (-0.1720, 0.1263),
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_csv_rows(path):
    """Return {model: row_dict} for the per-model rows (excludes the Mean row)."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["model"]: r for r in rows}


def _load_mean_row(path):
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == "Mean":
                return r
    raise AssertionError("no Mean row in %s" % path)


@pytest.fixture(scope="module")
def csv_rows():
    assert CSV_PATH.exists(), f"missing committed artifact: {CSV_PATH}"
    return _load_csv_rows(CSV_PATH)


@pytest.fixture(scope="module")
def json_data():
    assert JSON_PATH.exists(), f"missing committed artifact: {JSON_PATH}"
    return json.loads(JSON_PATH.read_text())


# --------------------------------------------------------------------------- #
# 1. The five-model table: associational accuracy band
# --------------------------------------------------------------------------- #
def test_associational_accuracy_band(csv_rows):
    """Every model is a strong associational predictor (~0.74-0.75)."""
    for model in MODELS:
        acc = float(csv_rows[model]["assoc_acc"])
        assert 0.742 <= acc <= 0.750, f"{model} assoc_acc={acc} out of band"
        assert acc == pytest.approx(EXPECTED_ASSOC_ACC[model], abs=1e-3)


def test_mean_associational_accuracy(csv_rows):
    mean_acc = float(_load_mean_row(CSV_PATH)["assoc_acc"])
    assert mean_acc == pytest.approx(MEAN_ASSOC_ACC, abs=1e-3)


# --------------------------------------------------------------------------- #
# 2. Naive causal accuracy collapses to 0.0 for ALL models (sign-flip failure)
# --------------------------------------------------------------------------- #
def test_naive_causal_accuracy_is_zero(csv_rows):
    """The naive (unadjusted) causal-effect accuracy is exactly 0.0 for all 5."""
    for model in MODELS:
        naive = float(csv_rows[model]["naive_causal_acc"])
        assert naive == pytest.approx(0.0, abs=1e-9), (
            f"{model} naive_causal_acc={naive}, expected collapse to 0.0"
        )


def test_naive_causal_accuracy_is_zero_json(json_data):
    for model in MODELS:
        assert json_data["models"][model]["naive_causal_acc"] == pytest.approx(
            0.0, abs=1e-9
        )
    assert json_data["mean"]["naive_causal_acc"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. Adjusted causal accuracy recovers (mean ~0.846)
# --------------------------------------------------------------------------- #
def test_adjusted_causal_accuracy_recovers(csv_rows):
    """Backdoor/do-intervention restores causal accuracy per model."""
    for model in MODELS:
        adj = float(csv_rows[model]["adj_causal_acc"])
        assert adj == pytest.approx(EXPECTED_ADJ_ACC[model], abs=2e-3)
        # Recovery: adjusted strictly beats the collapsed naive estimate.
        assert adj > float(csv_rows[model]["naive_causal_acc"])


def test_mean_adjusted_causal_accuracy(csv_rows):
    mean_adj = float(_load_mean_row(CSV_PATH)["adj_causal_acc"])
    assert mean_adj == pytest.approx(MEAN_ADJ_ACC, abs=2e-3)
    assert mean_adj > 0.80, "adjusted causal accuracy should recover above 0.80"


# --------------------------------------------------------------------------- #
# 4. The paper's thesis: naive ATE POSITIVE (sign-flipped) vs true ATE NEGATIVE
# --------------------------------------------------------------------------- #
def test_true_ate_is_negative_protective(csv_rows):
    for model in MODELS:
        true_ate = float(csv_rows[model]["true_ate"])
        assert true_ate < 0, f"{model} true_ate={true_ate}, expected protective(<0)"
        assert true_ate == pytest.approx(TRUE_ATE, abs=1e-3)


def test_naive_ate_sign_flipped_vs_true(csv_rows):
    """Naive ATE is POSITIVE while the true ATE is NEGATIVE -- the confounding
    inverts the apparent treatment direction (central manuscript claim)."""
    for model in MODELS:
        naive_ate = float(csv_rows[model]["naive_ate"])
        true_ate = float(csv_rows[model]["true_ate"])
        assert naive_ate > 0, f"{model} naive_ate={naive_ate}, expected sign-flip(>0)"
        assert true_ate < 0
        # explicit opposite-sign relationship
        assert naive_ate * true_ate < 0
        assert naive_ate == pytest.approx(EXPECTED_NAIVE_ATE[model], abs=2e-3)


def test_adjusted_ate_recovers_sign(csv_rows):
    """Adjusted ATE returns to the correct (negative) sign."""
    for model in MODELS:
        adj_ate = float(csv_rows[model]["adj_ate"])
        assert adj_ate < 0, f"{model} adj_ate={adj_ate}, expected negative after adjust"


def test_ate_by_domain_sign_relationship():
    """Per-domain artifact: true ATE < 0 (protective), naive ATE > 0 (flipped)."""
    assert ATE_BY_DOMAIN.exists(), f"missing artifact: {ATE_BY_DOMAIN}"
    with ATE_BY_DOMAIN.open(newline="") as f:
        rows = {r["domain"]: r for r in csv.DictReader(f)}
    for domain, (exp_true, exp_naive) in EXPECTED_DOMAIN_ATE.items():
        true_ate = float(rows[domain]["true_ate"])
        naive_ate = float(rows[domain]["naive_ate"])
        backdoor_ate = float(rows[domain]["backdoor_ate"])
        assert true_ate < 0 and naive_ate > 0, f"{domain}: sign relationship broken"
        assert true_ate * naive_ate < 0
        assert backdoor_ate < 0, f"{domain}: backdoor estimate should be negative"
        assert true_ate == pytest.approx(exp_true, abs=1e-3)
        assert naive_ate == pytest.approx(exp_naive, abs=1e-3)


# --------------------------------------------------------------------------- #
# 5. Cross-artifact consistency: committed CSV matches committed JSON
# --------------------------------------------------------------------------- #
def test_csv_matches_json(csv_rows, json_data):
    for model in MODELS:
        row = csv_rows[model]
        jm = json_data["models"][model]
        assert float(row["assoc_acc"]) == pytest.approx(jm["accuracy"], abs=1e-3)
        assert float(row["naive_causal_acc"]) == pytest.approx(
            jm["naive_causal_acc"], abs=1e-3
        )
        assert float(row["adj_causal_acc"]) == pytest.approx(
            jm["adj_causal_acc"], abs=1e-3
        )
        assert float(row["naive_ate"]) == pytest.approx(jm["tau_hat_naive"], abs=1e-3)
        assert float(row["true_ate"]) == pytest.approx(jm["true_ate"], abs=1e-3)


def test_seed_is_42(json_data):
    assert json_data["seed"] == 42


# --------------------------------------------------------------------------- #
# 6. Determinism: a live re-run regenerates the committed numbers.
#    Marked `slow` because it trains all five models (incl. PyTorch LSTM/TCN).
#    Default `pytest` run asserts against the committed CSV (fast, above);
#    `pytest -m slow` executes the pipeline and compares to the artifact.
# --------------------------------------------------------------------------- #
def _import_model_comparison():
    """Import scripts/model_comparison.py as a module (script has no package)."""
    import sys

    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("model_comparison")


def test_model_comparison_importable():
    """The entry point imports cleanly and exposes the expected API."""
    mc = _import_model_comparison()
    assert hasattr(mc, "main")
    assert mc.SEED == 42
    assert mc.MODELS == MODELS


@pytest.mark.slow
def test_live_rerun_matches_committed_csv(csv_rows):
    """Run the pipeline end-to-end and assert the regenerated per-model metrics
    match the committed results/model_comparison.csv to tolerance.

    NOTE: main() rewrites results/model_comparison.csv|json in place. We capture
    the returned dict (not the rewritten file) for the comparison so the test is
    a true reproduction check against the *previously committed* reference held
    in the `csv_rows` fixture.
    """
    mc = _import_model_comparison()
    if not getattr(mc, "HAVE_XGB", False):
        pytest.skip("xgboost unavailable; XGBoost row cannot be reproduced")
    js = mc.main()

    for model in MODELS:
        produced = js["models"][model]
        committed = csv_rows[model]
        assert produced["accuracy"] == pytest.approx(
            float(committed["assoc_acc"]), abs=5e-3
        ), f"{model} associational accuracy drifted"
        assert produced["naive_causal_acc"] == pytest.approx(0.0, abs=1e-6), (
            f"{model} naive causal accuracy no longer collapses to 0.0"
        )
        assert produced["adj_causal_acc"] == pytest.approx(
            float(committed["adj_causal_acc"]), abs=1e-2
        ), f"{model} adjusted causal accuracy drifted"
        # Thesis invariants must survive a live run.
        assert produced["tau_hat_naive"] > 0
        assert produced["true_ate"] < 0
