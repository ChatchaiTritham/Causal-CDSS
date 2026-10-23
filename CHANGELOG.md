# Changelog

## v1.1.3 (2026-10-23)

- `scripts/make_tables_sim.py` now also writes `tables/v110_rhc.tex` (the RHC application table)
  from `results/v110_rhc_estimates.csv`; the table was previously typeset by hand from that file.
  `tests/test_sim_tables.py` checks byte-identical regeneration and spot values.
- `MANUSCRIPT_MAP.md`: every table and figure of the Biometrical Journal manuscript and its
  Supporting Information mapped to the script and result file that produce it, with run order,
  seeds and software versions.
- CITATION.cff and setup.py version 1.1.3.

## v1.1.2 (2026-10-23)
- scripts/make_tables_sim.py: writes the held-out tables of the Statistics in Medicine
  manuscript (tables/v2_rule_sim.tex, v3_loss.tex, v3_strata.tex) from results/v2_rule_summary.csv,
  default_loss_summary.csv and default_loss_strata.csv; previously these three tables had no
  generating script. tests/test_sim_tables.py checks byte-identical regeneration.
- scripts/v112_review.py: proximal 2SLS in the RHC data with every single-proxy pair and with the
  proxy roles swapped (results/v112_rhc_proxy_roles.csv, tables/v112_roles.tex); figure of the
  held-out generators (figures/fig_heldout.pdf) and pairwise RMSE shares
  (results/v112_heldout_pairwise.csv).
- CITATION.cff and setup.py version 1.1.2.

## v1.1.1 (2026-10-23)
- v110_extensions.py part plasdiag: correlations and partial correlations (given X) of the real
  proxies with the withheld APACHE III score; results/v110_plasmode_proxy_strength.csv.
- CITATION.cff and setup.py version 1.1.1.

## v1.1.0 (2026-10-23)
- scripts/rhc_data.py: download (with SHA-256 check) and preparation of the public SUPPORT right
  heart catheterization data (Connors et al. 1996) from https://hbiostat.org/data/repo/rhc.csv.
  The file is not redistributed (data/rhc/ is git-ignored); proxies as in Cui et al. (2024).
- scripts/v110_extensions.py: real-data application (proxy-adjusted OLS, proximal 2SLS, over-
  identified GMM with Hansen J, proximal IPW/DR, AIPW, bootstrap CIs, Tan and sharp sensitivity
  values); plasmode study on the RHC cohort with the APACHE III score withheld as the confounder;
  sharp marginal-sensitivity bounds (Dorn and Guo 2023) on the main and held-out generators;
  Tikhonov-regularised treatment-bridge solver rerun on held-out sets A and B; numerical checks of
  Propositions 1-3. Results results/v110_*.csv.
- tests/test_v110_extensions.py.
- Regenerate: `python scripts/rhc_data.py && python scripts/v110_extensions.py` (set
  OMP_NUM_THREADS=1 when using several workers).

## v1.0.8 (2026-10-21)
- scripts/sim_review_checks.py (no new simulation): Spearman correlation of flag rate with DR
  coverage and flag AUC at three thresholds; 90% interval score of the three families; share of
  held-out replications with an unsolved treatment bridge. Output results/sim_review_checks.csv.
- msm_bounds.py: third domain labelled ACS in tables/msm_bounds.tex, as in the other tables.

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
