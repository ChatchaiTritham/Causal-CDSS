# Changelog

## v1.0.7 (2026-10-21)
- scripts/msm_bounds.py: marginal-sensitivity-model (Tan 2006) bounds for the ATE with the
  Zhao-Small-Bhattacharya percentile bootstrap; smallest Lambda covering the truth and smallest
  Lambda covering zero on the main generators (100 replications, 100 resamples) and on held-out
  set B (60 generators, 10 replications). Results results/msm_bounds_*.csv, table tables/msm_bounds.tex.
- tests/test_msm_bounds.py. Manuscript retargeted to Statistics in Medicine.
- Regenerate: `python scripts/msm_bounds.py`.

## v1.0.6 (2026-10-21)
- scripts/default_loss_audit.py: pre-stated interval-score loss over all three estimator families,
  DR coverage strata, bias/SE decomposition, flag AUC, disagreement-widened interval, AR boundedness.
- tests/test_default_loss_audit.py.

## v1.0.5 (2026-10-21)

- `scripts/decision_rule_v2.py`: revised audit decision rule (DR proximal by default, inconclusive
  flags on unsolved q, F < 16 or DR/2SLS disagreement) scored with the v1 threshold rule, always-proxy,
  always-2SLS and always-DR on the 60 held-out generators (set A, seed 2027) and 60 fresh generators
  (set B, seed 2031); cluster-bootstrap CIs; noisy elicitation. Anderson-Rubin projection interval for
  the proximal 2SLS step with a weak-proxy coverage experiment (`--part weak`).
- Results `results/v2_*.csv`, tables `tables/v2_rule.tex`, `tables/v2_weak.tex`; tests
  `tests/test_decision_rule_v2.py`.
- Regenerate: `python scripts/decision_rule_v2.py && python scripts/decision_rule_v2.py --part weak && python scripts/make_tables_r2.py`.

## v1.0.4 (2026-10-21)
Second review round: `scripts/review_round2.py` (results/r2_*.csv), CDS Hooks 2.0 response
format with schema test, FHIR R5 example bundles, split SHA256 manifests, `scripts/run_paper.py`,
pinned lock file. Version of record for the KBS submission.

## v1.0.3 (2026-10-20), commit 8123cb0
Bridge and knowledge-base misspecification, sandwich SE, crossing-point CIs, audit card update.
Zenodo 10.5281/zenodo.23505334 (superseded by the v1.0.4 record).

## v1.0.2
Lightweight tag on the v1.0.1 commit 121e8a1; no code change. Zenodo 10.5281/zenodo.23504148.

## v1.0.1
Proximal policy-value estimator, age-subgroup audit and violation sweep.
