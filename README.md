# Causal-CDSS — Structural Causal Models for Evaluating Clinical Decision Support Systems

Companion code for the manuscript:

> **Structural Causal Models for Evaluating Clinical Decision Support Systems:
> A Causal Inference Framework**
> (target: *Knowledge and Information Systems* (KAIS), Springer)

The paper proposes a causal-inference framework with domain-specific causal DAGs
for three critical-care settings and five metrics that expose treatment-effect
reasoning failures invisible to standard accuracy measures.

## Scope of this repository

This repository contains a **seeded reproducibility driver** (`run_all.py`,
seed 42) together with the figure-generation and DAG-definition scripts. The
driver *recomputes* every reported causal quantity from a synthetic ICU cohort
using the vendored `basics_cdss.causal` module — it builds structural causal
models (SCMs), samples observational and interventional data via do-calculus,
estimates treatment effects with three estimators, scores the five
causal-evaluation metrics, and writes machine-readable results to
`results/`. **No manuscript constant is hardcoded in the driver**: the numbers
in the table below are produced at run time from the seeded SCMs.

The figure scripts under `figures/` render the manuscript diagrams; the
quantitative narrative they illustrate is reproduced independently by
`run_all.py`.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # installs deps + vendored basics_cdss
python run_all.py           # writes results/causal_results.json + results/*.csv
```

Outputs (regenerated deterministically on every run):

| File | Contents |
|------|----------|
| `results/causal_results.json` | Full machine-readable results for all domains |
| `results/ate_by_domain.csv` | Naive vs. true (do-calculus) vs. adjusted ATE |
| `results/cate_subgroups.csv` | CATE across age × severity subgroups |
| `results/causal_metrics.csv` | Five causal-evaluation metrics per estimator |
| `results/confounding_sensitivity.csv` | Confounding bias + E-value sensitivity |

## Headline results (seed 42, 8,000 observational patients per domain)

The treatment-assignment mechanism is confounded by disease severity, so the
**naive (observational) association has the wrong sign** (treatment looks
harmful because sicker patients are more likely to be treated). The **true
causal effect** — recovered by do-calculus on the SCM and by backdoor /
doubly-robust adjustment — shows a substantial mortality reduction.

| Domain | Naive ATE | True ATE (do) | Backdoor-adj. | Doubly-robust | Confounding bias |
|--------|----------:|--------------:|--------------:|--------------:|-----------------:|
| Sepsis — Early Antibiotics | +0.099 | −0.149 | −0.170 | −0.162 | 0.279 |
| ARDS — Low Tidal Volume    | +0.061 | −0.113 | −0.128 | −0.127 | 0.205 |
| ACS — Early Reperfusion     | +0.126 | −0.172 | −0.207 | −0.196 | 0.321 |

Causal-evaluation metrics (best observational estimator per domain): ATE
alignment score 0.80–0.91; Causal Consistency Index (true vs. estimated CATE
correlation) 0.74–0.95; counterfactual sign accuracy 1.00 across all
subgroups. CATE varies markedly across strata (e.g. sepsis: −0.05 to −0.31 by
age × severity). E-value sensitivity 1.59–2.21.

ATE here is on the mortality-probability scale (negative = mortality
reduction); the manuscript reports the same quantities as percentage points.

## Figures

| Path | Contents |
|------|----------|
| `figures/fig01_causal_dag.py` | Causal DAG for the CDSS evaluation setting |
| `figures/fig02_forest_plot.py` | Treatment-effect forest plot |
| `figures/fig03_cate_heterogeneity.py` | Conditional average treatment-effect heterogeneity |
| `figures/fig04_confounding_analysis.py` | Confounding analysis |
| `figures/fig05_backdoor_adjustment.py` | Back-door adjustment illustration |
| `figures/generate_figures.py` | Batch driver for the figure scripts |

```bash
python figures/generate_figures.py
```

## Citation

```bibtex
@article{tritham_causal_cdss,
  title  = {Structural Causal Models for Evaluating Clinical Decision Support
            Systems: A Causal Inference Framework},
  author = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Licensed under the MIT License (see `LICENSE`).

## Portfolio Relationship

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

## Contact

**Chatchai Tritham**  
Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand  
Email: chatchait66@nu.ac.th  
ORCID: 0000-0001-7899-228X

**Chakkrit Snae Namahoot**  
Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand  
Email: chakkrits@nu.ac.th  
ORCID: 0000-0003-4660-4590
