# Reproducibility

This repository is deterministic at a fixed seed (`42`). Every number in the
manuscript that this code is responsible for is recomputed at run time from
structural causal models and standard estimators; nothing is hardcoded in the
figure or table scripts.

## Environment

- Python 3.10+
- Dependencies pinned in `requirements.txt`
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

- Cohort size: **10,000 trajectories per domain** (`N_PATIENTS = 10000`),
  6 timesteps each — this is the "10,000 trajectories" experiment reported in the
  manuscript abstract and Table "Associational vs. causal performance".
- Five architectures (LR, RF, XGBoost, LSTM, TCN). Global determinism is set via
  `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, and `torch.manual_seed`.
- Writes `results/model_comparison.csv` and `results/model_comparison.json`.

Headline values produced (seed 42): mean associational accuracy 74.7%
(AUROC ~0.81), naive causal accuracy 0% for every architecture, adjusted causal
accuracy mean 84.6% (LSTM 96.4%, TCN 96.3%, LR 94.1%, XGBoost 89.0%, RF 47.2%).

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

Source code is in this repository under the MIT License (see `LICENSE`). A tagged
GitHub release can be archived to Zenodo for a citable DOI; see
`ZENODO_RELEASE.md`.
