# Causal-CDSS — Structural Causal Models for Evaluating Clinical Decision Support Systems

Companion code for the manuscript:

> **Structural Causal Models for Evaluating Clinical Decision Support Systems:
> A Causal Inference Framework**
> (target: *Knowledge and Information Systems* (KAIS), Springer)

The paper proposes a causal-inference framework with domain-specific causal DAGs
for three critical-care settings and five metrics that expose treatment-effect
reasoning failures invisible to standard accuracy measures.

## Scope of this repository (please read)

This repository currently contains the **figure-generation and DAG-definition
scripts** used to produce the diagrams in the manuscript. It is a *methods /
figure* repository, **not yet a full reproducibility package**: there is no
single seeded driver that regenerates the reported empirical tables from raw
data. The quantitative values rendered by these scripts are taken from the
study's analysis and are reproduced here for figure transparency, not
recomputed at plot time.

A `run_all.py` driver (seed 42) that recomputes every reported metric into
committed `results/*.csv` is planned before journal submission.

## Contents

| Path | Contents |
|------|----------|
| `figures/fig01_causal_dag.py` | Causal DAG for the CDSS evaluation setting |
| `figures/fig02_forest_plot.py` | Treatment-effect forest plot |
| `figures/fig03_cate_heterogeneity.py` | Conditional average treatment-effect heterogeneity |
| `figures/fig04_confounding_analysis.py` | Confounding analysis |
| `figures/fig05_backdoor_adjustment.py` | Back-door adjustment illustration |
| `figures/generate_figures.py` | Batch driver for the figure scripts |

## Usage

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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
