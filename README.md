# Structural Causal Models for CDSS Evaluation (Causal-CDSS)

> Run one seeded script and watch a confounded ICU cohort flip the sign of a treatment effect, then watch backdoor and doubly-robust adjustment put it back.

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Reproducible](https://img.shields.io/badge/reproducible-seed--42-success)

## Overview

Clinical decision support is usually graded the way a forecaster is graded: AUROC, accuracy, calibration. Those scores reward a model for matching what was observed, but a system that *recommends* a treatment is making an interventional claim, and an interventional claim cannot be checked by a correlational yardstick. When sicker patients are the ones who get treated, the raw treated-minus-untreated comparison can even point the wrong way.

This repository turns that argument into something you can execute. The driver builds a small structural causal model for each of three critical-care settings, draws a synthetic observational cohort in which disease severity drives both who is treated and who survives, and then estimates the treatment effect three ways: the naive difference, backdoor regression adjustment, and a doubly-robust estimator. Against each estimate it places the ground truth recovered by do-calculus on the same model, so the gap between association and causation is measured rather than asserted.

The aim is narrow and honest. Everything here runs on simulated data with a known answer, which is what lets us score an estimator against truth at all; it is an in-distribution demonstration of the evaluation idea, not a clinical validation. The manuscript (*Auditing Knowledge-Driven Clinical Decision Support under Unmeasured Confounding: A Structural Causal Model Protocol*, submitted to *Knowledge-Based Systems*) builds on this verification cohort and adds an audit cohort in which severity is unrecorded; its analyses are listed under **Audit-cohort analyses** below.

## Key results

All figures below are written by `run_all.py` at seed 42, from an 8,000-patient observational cohort per domain (treatment prevalence 0.44–0.46, mortality 0.35–0.41).

- The naive association carries the **wrong sign** in every domain: treatment looks harmful (sepsis +0.099, ARDS +0.061, ACS +0.126 on the mortality-probability scale) because severity confounds the comparison.
- Do-calculus on the model recovers a genuine mortality reduction — true ATE −0.149 (sepsis), −0.113 (ARDS), −0.172 (ACS) — and both adjusted estimators land close to it (e.g. ACS doubly-robust −0.196 vs true −0.172).
- Confounding bias, the distance between the observational and the causal effect, runs from 0.205 (ARDS) to 0.321 (ACS).
- Subgroup CATE varies several-fold: for sepsis the true effect ranges from −0.053 (young, low-severity) to −0.307 (young, high-severity), and the estimated CATE tracks it with a Causal Consistency Index (true-vs-estimated correlation) of 0.74–0.95 across domains.
- Counterfactual sign accuracy is 1.00 in all three domains, and the E-value sensitivity sits between 1.59 (ARDS) and 2.21 (ACS).

**Relationship to the manuscript.** The results above are the *verification cohort*
(Supplementary Section S8 of the paper). The repository contains every experiment
reported in the paper. `run_all.py` regenerates the domain-level SCM,
intervention, subgroup, and sensitivity analyses. The separate
`scripts/model_comparison.py` driver trains LR, RF, XGBoost, LSTM, and TCN on
10,000 trajectories per domain and regenerates the associational-versus-causal
table of Supplementary Section S8; `scripts/seed_replication.py` repeats it over seeds
42–51 and adds the Friedman and Holm–Wilcoxon tests. The numerical environment is fully
pinned in `requirements.txt`; the model-comparison driver fails closed rather
than writing submission evidence from its labelled proxy if PyTorch is absent.

## Repository structure

```
Causal-CDSS/
├── run_all.py                 # seeded driver (seed 42) — recomputes everything below
├── setup.py, requirements.txt # install + pinned dependencies
├── src/basics_cdss/           # vendored package; src/basics_cdss/causal/ holds the SCM engine
│   └── causal/                # CausalGraph, StructuralCausalModel, ATE/CATE, backdoor, E-value
├── results/                   # CSV + JSON written by run_all.py
│   ├── causal_results.json    # full machine-readable results, all domains
│   ├── ate_by_domain.csv      # naive / true(do) / backdoor / doubly-robust ATE
│   ├── cate_subgroups.csv     # CATE over age × severity strata
│   ├── causal_metrics.csv     # the five causal-evaluation metrics per estimator
│   └── confounding_sensitivity.csv  # confounding bias, risk ratio, E-value
├── scripts/
│   ├── unmeasured_confounding.py   # audit cohort: ATE estimators, policy value (recorded vs oracle)
│   ├── proximal_robustness.py      # exclusion sweeps, crossing points, weak proxies,
│   │                               #   nonlinear severity, DML, graph tests (Bonferroni)
│   ├── model_comparison.py         # verification cohort: five architectures (CAUSAL_SEED env)
│   ├── seed_replication.py         # ten-seed replication + Friedman/Holm tests
│   ├── make_unmeasured_figures.py  # figures of the paper
│   ├── make_protocol_figure.py     # protocol overview diagram
│   └── make_tables.py              # writes every LaTeX table of the paper to tables/
├── tables/                    # generated LaTeX table bodies (no hand-typed numbers)
└── figures/                   # rendered figures (PDF/PNG) and the original plotting scripts
```

## Installation

```bash
git clone https://github.com/ChatchaiTritham/Causal-CDSS.git
cd Causal-CDSS
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # installs requirements + the vendored basics_cdss package
```

## Reproducing the results

```bash
python run_all.py          # fixed seed 42; under a minute on a standard workstation
```

The driver prints a per-domain summary and writes `results/causal_results.json` plus the four CSV files listed above. Because the seed governs every draw — the observational cohort, the interventional samples used for do-calculus, and the bootstrap-free metric computations are all deterministic — a re-run on the same machine reproduces the stdout and the metric values exactly. The doubly-robust estimator and the propensity model come from scikit-learn with fixed inputs, so they too return identical numbers run to run. To redraw the diagrams afterwards, run `python figures/generate_figures.py`.

## Audit-cohort analyses (the paper's main results)

Run in this order; times are for a standard laptop.

```bash
python scripts/unmeasured_confounding.py   # ~6 min  -> proximal_estimates.csv, policy_value_*.csv
python scripts/proximal_robustness.py      # ~90 min -> proximal_robustness*.csv, proximal_crossing.csv,
                                           #            proximal_weak_proxies.csv, proximal_nonlinear.csv,
                                           #            dml_estimates.csv, dag_implication_tests*.csv
python scripts/seed_replication.py         # ~30 min first run (ten seeds), seconds afterwards
python scripts/proximal_bootstrap.py      # bootstrap check of proximal SEs -> proximal_bootstrap*.csv
python scripts/sample_size.py             # ~5 min  -> sample_size.csv (n = 1,000 ... 16,000)
python scripts/proximal_policy_subgroup.py # ~1 min -> proximal_policy_value.csv, subgroup_audit.csv
python scripts/proximal_policy_violation.py # ~2 min -> proximal_policy_violation.csv
python scripts/audit_card.py              # one audit run per domain -> audit_card.json
python scripts/make_unmeasured_figures.py  # figures/fig_proximal_*.pdf, fig_policy_overlap.pdf
python scripts/make_tables.py              # tables/*.tex, included by the manuscript
```

Seeds are derived per analysis and replication from base seed 42 plus a per-domain CRC32
offset, so results do not depend on process order or hash salting. The backdoor comparator
is reported twice: on age only (`backdoor_observed`) and on age plus both proxies
(`backdoor_proxy`), the fair competitor for the proximal estimator.

## Verification-cohort figures

The descriptions below pair each original plotting script with the numbers in `results/` that it visualises. Run `python figures/generate_figures.py` to produce them.

- `figures/fig01_causal_dag.py` — draws the shared causal graph (age → severity → treatment, with severity, age and treatment all feeding the outcome). Look at the severity node: it is the single backdoor path that makes the naive estimate biased, and it is the one variable `adjustment_set_identified` flags in `causal_results.json`.
- `figures/fig02_forest_plot.py` — forest plot of the three ATE estimates against the do-calculus truth, drawn from `results/ate_by_domain.csv`. The take-away is the sign flip: the naive point sits on the positive (harmful) side while true, backdoor, and doubly-robust all sit together on the protective side.
- `figures/fig03_cate_heterogeneity.py` — renders the age × severity CATE grid from `results/cate_subgroups.csv`. Read it column to column: the effect deepens with severity (sepsis −0.05 → −0.31), and the estimated cells stay close to the true cells, which is what the 0.74–0.95 Consistency Index reports numerically.
- `figures/fig04_confounding_analysis.py` — bar comparison of observational vs causal effect per domain, i.e. the `confounding_bias` column of `results/confounding_sensitivity.csv`. The bars are largest for ACS (0.321) and smallest for ARDS (0.205).
- `figures/fig05_backdoor_adjustment.py` — illustrates how conditioning on the adjustment set closes the backdoor path; pair it with the `backdoor_regression` column in `results/causal_metrics.csv`, where the alignment score jumps from 0.00 (naive) to 0.79–0.87 once severity and age are adjusted for.
- `figures/generate_figures.py` — batch driver that renders all five scripts.

If you prefer the raw evidence, `results/causal_metrics.csv` is the most compact summary: one row per (domain, estimator) with the ATE estimate, the causal-effect-estimation error, the alignment score, the confounding bias magnitude, the Consistency Index, and the counterfactual sign accuracy.

## Data

No patient data is used. Each domain is a synthetic cohort sampled from a structural causal model whose mechanisms (severity from age, treatment from severity, mortality from severity, age and treatment) are written out in `run_all.py`. Because the data are simulated and contain no human subjects, no ethics approval or IRB review applies.

## Citation

```bibtex
@article{tritham_causal_cdss,
  title  = {Auditing Knowledge-Driven Clinical Decision Support under Unmeasured
            Confounding: A Structural Causal Model Protocol},
  author = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  year   = {2026},
  note   = {Manuscript under review; to appear}
}
```

## License

Released under the MIT License (see `LICENSE`).

## Contact

**Chatchai Tritham** — Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand. Email: chatchai.t@mail.rmutk.ac.th · ORCID: 0000-0001-7899-228X
**Chakkrit Snae Namahoot** — same affiliation. Email: chakkrits@nu.ac.th · ORCID: 0000-0003-4660-4590

## Portfolio relationship

| Repository | Role |
|---|---|
| BASICS-CDSS | Beyond-accuracy evaluation methodology |
| TRI-X | Framework-level package |
| ORASR | Routing and safety-action component |
| DRAS-5 | Dynamic risk-state component |
| SAFE-Gate | Safety-gated ensemble framework |
| SynDX | Synthetic validation and explainability evidence |
| SURgul | SRGL/governance reproducibility component |
| TRI-X-CDSS | Integration and implementation package |
| Selective-CDSS | Risk-controlled selective-prediction (abstention) component |
| Causal-CDSS | Causal-inference evaluation component |
| Beyond-Accuracy | Simulation-based safety/calibration evaluation framework |

## Relationship to our other repositories

The package under `src/basics_cdss/` is shared infrastructure: the same modules appear in
several of our repositories covering complementary aspects of clinical decision support
evaluation. Sharing it keeps one implementation of the common metric and simulation code
rather than four divergent copies.

What belongs to **this** study is the driver, the computed results and the figures:

- `run_all.py` — causal-mechanism driver
- `scripts/model_comparison.py` — architecture-comparison driver
- `results/` — the computed metric tables
- `figures/` — the figures and their generators

Anything under `src/basics_cdss/` that this study does not exercise is inherited from the
shared package and is not part of the results reported here.
