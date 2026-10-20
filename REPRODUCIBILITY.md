# Reproducibility

This repository is deterministic at a fixed seed (`42`). Every number in the
manuscript that this code is responsible for is recomputed at run time from
structural causal models and standard estimators; nothing is hardcoded in the
figure or table scripts.

## Environment

- Python 3.10+
- The complete numerical environment is pinned in `requirements.txt`; PyTorch
  is mandatory for the manuscript model-comparison driver.
- Install in editable mode: `pip install -e .`

## Two experiments, two drivers

The repository regenerates two distinct experiments. Each writes machine-readable
outputs that the manuscript tables and figures are drawn from.

### 1. Causal-mechanism driver — `run_all.py`

```bash
python run_all.py            # seed 42; under a minute on a standard workstation
```

- Cohort size: **8,000 observational patients per domain** (`N_OBS = 8000`),
  8,000 interventional samples per arm for do-calculus.
- Writes `results/causal_results.json` plus `ate_by_domain.csv`,
  `cate_subgroups.csv`, `causal_metrics.csv`, `confounding_sensitivity.csv`.

Headline values produced (seed 42), as they appear in `results/`:

| Domain | True ATE | Naive | Backdoor | Doubly-robust | Bias (pp) | E-value |
|---|---|---|---|---|---|---|
| Sepsis | -0.149 | +0.098 | -0.170 | -0.162 | 26.9 | 1.92 |
| ARDS   | -0.113 | +0.061 | -0.128 | -0.127 | 18.9 | 1.59 |
| ACS    | -0.172 | +0.126 | -0.207 | -0.196 | 33.4 | 2.21 |

(Bias = |naive - backdoor-adjusted| x 100; counterfactual sign accuracy = 1.00
in all three domains.)

### 2. Associational-vs-causal model comparison — `scripts/model_comparison.py`

```bash
python scripts/model_comparison.py    # seed 42
```

The driver fails closed if PyTorch is unavailable. It never writes manuscript
evidence from the labelled NumPy proxy implementation.

- Cohort size: **10,000 trajectories per domain** (`N_PATIENTS = 10000`),
  6 timesteps each — this is the "10,000 trajectories" experiment reported in the
  manuscript abstract and Table "Associational vs. causal performance".
- Five architectures (LR, RF, XGBoost, LSTM, TCN). Global determinism is set via
  `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, and `torch.manual_seed`.
- Writes `results/model_comparison.csv` and `results/model_comparison.json`.

Headline values produced (seed 42): mean associational accuracy 74.7%
(AUROC ~0.81), naive causal accuracy 0% for every architecture, adjusted causal
accuracy mean 84.9% (TCN 98.4%, LSTM 96.4%, LR 94.1%, XGBoost 88.4%, RF 47.2%).

## Determinism notes

- All draws (observational cohorts, interventional samples, bootstrap CIs,
  train/test splits) are seeded. A re-run on the same machine/library versions
  reproduces the stdout and the metric values.
- The torch-based sequence models (LSTM/TCN) are seeded for reproducibility;
  exact bit-for-bit identity across different hardware/BLAS/torch builds is not
  guaranteed, as is standard for deep-learning components.

## Figures

```bash
python figures/generate_figures.py    # renders all five PNGs from results/
```

## Honest scope

All data are synthetic, sampled from structural causal models with known
ground-truth effects. This is an in-distribution demonstration of the evaluation
methodology, not a clinical validation. No patient data and no IRB review apply.

## Data and code availability

Source code is in this repository under the MIT License (see `LICENSE`). A
permanent, citable archival snapshot of the released code will be provided upon
acceptance.

## Identification without the simulated confounder (added 2026-10-04)

`run_all.py` adjusts for the confounder it simulates, which verifies the implementation but
not the evaluation practice. `scripts/unmeasured_confounding.py` removes that convenience:

- **Proximal identification.** Severity is hidden and replaced by a negative-control exposure
  and a negative-control outcome (Miao et al., 2018; Tchetgen Tchetgen et al., 2024). Backdoor
  adjustment on the recorded covariates then carries a bias of 0.134-0.184 in absolute
  mortality, of the wrong sign in sepsis and ARDS, while proximal two-stage least squares stays
  within 0.016 of the interventional truth (200 replications, n = 8,000).
- **Off-policy evaluation.** The CDSS is evaluated as a policy: IPW, self-normalised IPW and
  doubly-robust value estimates against the true value under `do(A = pi(x))`, with effective
  sample size and policy-agreement diagnostics. Interval coverage is 0.925 for a policy that
  agrees with recorded practice two thirds of the time and below 0.03 for treat-all or
  treat-none.

Outputs: `results/proximal_estimates.csv`, `results/policy_value_estimates.csv`,
`results/policy_value_coverage.csv`; figures from `scripts/make_unmeasured_figures.py`.

Reproduce with: `python scripts/unmeasured_confounding.py --reps 200 --n 8000`

## v1.0.3 (2026-10-20): audit extensions, traceability and validation note

New drivers (manuscript Supplementary Sections S16-S20):

- `scripts/proximal_audit_extensions.py` -- 2x2 bridge misspecification with a stacked
  M-estimation sandwich SE (`results/bridge_misspecification.csv`), misspecified knowledge base
  for the knowledge-driven rule, population-vs-sample value estimand, convergence exclusion and
  the auditor-side youngest-third contrast (`results/kb_misspecification.csv`,
  `results/estimand_checks.csv`).
- `scripts/crossing_uncertainty.py` -- Monte-Carlo intervals and a common scale for the
  crossing points (`results/proximal_crossing_ci.csv`).
- `scripts/audit_card.py` -- card now follows the CDS Hooks 2.0 Card schema (uuid, markdown
  detail, source.topic, order-select hook); structured numbers in a separate `audit` object;
  first-stage F threshold 16; interventional truth never written to the card.
- `tests/test_audit_extensions.py` -- partial-F symmetry, DR = proximal IPW identity, DR
  consistency with a correct outcome bridge, CDS Hooks required fields.

Note: tag v1.0.2 on the remote points to the same commit as v1.0.1 (121e8a1); it adds no code.
v1.0.3 is the first release after v1.0.1 with new code.

### Traceability: manuscript element -> script -> result file

| Manuscript element | Script | Result file |
|---|---|---|
| Table 1, Fig. S5 (ATE) | scripts/unmeasured_confounding.py | results/proximal_estimates.csv |
| Table 2, Fig. S7, Table S10 (standard policy value) | scripts/unmeasured_confounding.py | results/policy_value_*.csv |
| Table 3, Fig. 3, Table S27 (proximal value, subgroups) | scripts/proximal_policy_subgroup.py | results/proximal_policy_value.csv, results/subgroup_audit.csv |
| Fig. 2, Tables S4-S6, S9, S17-S19 (sweeps, nonlinear, weak, DML, DAG tests) | scripts/proximal_robustness.py | results/proximal_robustness*.csv, proximal_nonlinear.csv, proximal_weak_proxies.csv, proximal_crossing.csv, dml_estimates.csv, dag_implication_tests*.csv |
| Table S12, Fig. S6 (cohort size) | scripts/sample_size.py | results/sample_size.csv |
| Table S13 (audit card) | scripts/audit_card.py | results/audit_card.json |
| Table S14 | scripts/proximal_policy_violation.py | results/proximal_policy_violation.csv |
| Table S16 (bootstrap) | scripts/proximal_bootstrap.py | results/proximal_bootstrap*.csv |
| Tables S21, S23, S24 | scripts/proximal_audit_extensions.py | results/bridge_misspecification.csv, kb_misspecification.csv, estimand_checks.csv |
| Table S22 | scripts/crossing_uncertainty.py | results/proximal_crossing_ci.csv |
| All LaTeX table bodies | scripts/make_tables.py | tables/*.tex |

Files from the separate verification cohort (`meta_learners*`, `model_comparison*`,
`causal_metrics.csv`, `cate_subgroups.csv`, `confounding_sensitivity.csv`, `ate_by_domain.csv`)
support Supplementary Sections S1-S5 and S8 only; they play no part in the audit argument.
`results/SHA256SUMS` lists the hash of every result file in this release.

### Validation note (intended use)

Research software for method comparison on synthetic data. It is not a validated
medical-device tool (IEC 62304 / ISO 13485 4.1.6 tool validation has not been performed).
Credibility evidence is limited to code verification: seeded reproduction of every table and
the unit tests in `tests/`. Environment: `requirements.txt` (pinned).
