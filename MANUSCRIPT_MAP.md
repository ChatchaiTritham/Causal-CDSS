# Manuscript map (Biometrical Journal submission, release v1.1.3)

Every table and figure of the manuscript "Proximal versus proxy-adjusted estimation under
unmeasured confounding" and of its Supporting Information is listed with the script that writes it
and the result file it is drawn from. No table cell is typed by hand: each `tables/*.tex` file is
written by the script named, and the manuscript includes it with `\input`.

## Environment used for the released results and the v1.1.3 checks

- Python 3.11.15; numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.0, statsmodels 0.15.0,
  matplotlib 3.11.1, xgboost 3.2.0, DoubleML 0.11.4.
- `requirements.txt` pins the core packages (pandas corrected to 2.3.3 in v1.1.3).
  `requirements-lock.txt` is a full dump of the development environment and contains packages
  unrelated to this repository; it is kept for provenance only.
- `pip install -e .` then `pytest -q` (all tests must pass before any table is regenerated).

## Seeds

- Simulation studies: base seed 42 plus a per-domain CRC32 offset, so results do not depend on
  process order (`src/` and each driver).
- Held-out generators: Latin-hypercube set A seed 2027 (used during development), set B seed 2031
  (drawn once, after the selection rule had failed on set A).
- Propositions, numerical checks: seed 20261022 (`scripts/v110_extensions.py`, part `theory`).

## Real data

The SUPPORT right heart catheterisation data are not redistributed. `python scripts/rhc_data.py`
downloads `rhc.csv` from https://hbiostat.org/data/repo/rhc.csv into `data/rhc/` and verifies the
SHA-256 checksum `9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8` before use.

## Run order

```bash
python scripts/unmeasured_confounding.py      # main study  -> results/proximal_estimates.csv, policy_value_*.csv
python scripts/proximal_robustness.py         # sweeps, weak proxies, nonlinear, DML, falsification
python scripts/seed_replication.py
python scripts/proximal_bootstrap.py
python scripts/sample_size.py
python scripts/proximal_policy_subgroup.py    # -> results/proximal_policy_value.csv
python scripts/proximal_policy_violation.py
python scripts/audit_card.py
python scripts/review_round2.py               # -> results/r2_*.csv
python scripts/decision_rule_v2.py && python scripts/decision_rule_v2.py --part weak
python scripts/default_loss_audit.py          # -> results/default_loss_*.csv
python scripts/msm_bounds.py                  # -> results/msm_bounds_*.csv
python scripts/rhc_data.py
python scripts/v110_extensions.py --parts theory,sharp,qreg,plasmode,rhc --workers 7
python scripts/v112_review.py                 # proxy-role table, held-out figure
python scripts/make_unmeasured_figures.py     # figures/fig_proximal_*.pdf, fig_policy_overlap.pdf, ...
python scripts/make_tables.py
python scripts/make_tables_r2.py
python scripts/make_tables_sim.py
pytest -q
```

`python scripts/run_paper.py` reruns the first block and checks the result files against
`results/SHA256SUMS_paper`.

## Main article

| Item | File | Written by | Result file |
|---|---|---|---|
| Figure 1 (causal graph) | `figures/fig_proximal_dag.pdf` | `scripts/make_unmeasured_figures.py` | (diagram) |
| Table 1 (ADEMP design) | typeset in the manuscript (design text, no results) | - | - |
| Table 2 (main generators) | `tables/tab_proximal.tex` | `scripts/make_tables.py` | `results/proximal_estimates.csv` |
| Figure 2 (restriction sweeps) | `figures/fig_proximal_sweeps.pdf` | `scripts/make_unmeasured_figures.py` | `results/proximal_robustness.csv`, `results/proximal_robustness_w.csv` |
| Table 3 (held-out families) | `tables/v3_loss.tex` | `scripts/make_tables_sim.py` | `results/default_loss_summary.csv` |
| Figure 3 (held-out generators) | `figures/fig_heldout.pdf` | `scripts/v112_review.py` | `results/default_loss_generators.csv`, `results/v112_heldout_pairwise.csv` |
| Table 4 (marginal-sensitivity bounds) | `tables/msm_bounds.tex` | `scripts/msm_bounds.py` | `results/msm_bounds_summary.csv` |
| Table 5 (value of treatment rules) | `tables/tab_proximal_ope.tex` | `scripts/make_tables.py` | `results/proximal_policy_value.csv` |
| Table 6 (RHC application) | `tables/v110_rhc.tex` | `scripts/make_tables_sim.py` | `results/v110_rhc_estimates.csv` |
| Appendix numerical checks | - | `scripts/v110_extensions.py` (part `theory`) | `results/v110_theory_checks.csv` |

## Supporting Information

| Tables | Written by |
|---|---|
| `tab_audit_card`, `tab_bootstrap`, `tab_crossing`, `tab_crossing_ci`, `tab_dag`, `tab_dml`, `tab_estimand`, `tab_kb`, `tab_misspec`, `tab_nonlinear`, `tab_ope_full`, `tab_policy_violation`, `tab_sample_size`, `tab_subgroup`, `tab_sweep_delta`, `tab_sweep_gamma`, `tab_weak`, `card_sepsis.json` | `scripts/make_tables.py` |
| `r2_analytic`, `r2_armf`, `r2_arms`, `r2_binbr`, `r2_calib`, `r2_calib_cross`, `r2_check`, `r2_conv`, `r2_cross`, `r2_diff`, `r2_dml`, `r2_falsify`, `r2_harm`, `r2_lhs`, `r2_lhs_meta`, `r2_nlh`, `r2_ovb`, `r2_overid`, `r2_prop`, `r2_pv`, `r2_pv_est`, `r2_pv_viol`, `r2_pvboot`, `r2_sg`, `r2_shift`, `v2_weak` | `scripts/make_tables_r2.py` (from `results/r2_*.csv` and `results/v2_*.csv`) |
| `v2_rule_sim`, `v3_strata` | `scripts/make_tables_sim.py` |
| `v112_roles` | `scripts/v112_review.py` |
| Marginal-sensitivity, sharp-bound, regularised-bridge, plasmode and RHC sections | `scripts/msm_bounds.py`, `scripts/v110_extensions.py` (CSV files `results/msm_bounds_*.csv`, `results/v110_*.csv`) |

| Figures | Written by |
|---|---|
| `fig_policy_overlap.pdf`, `fig_proximal_bias.pdf`, `fig_proximal_ope.pdf`, `fig_sample_size.pdf`, `fig_weak_proxies.pdf` | `scripts/make_unmeasured_figures.py` |

## Checks

- `tests/test_sim_tables.py`: the held-out and RHC tables regenerate byte-for-byte.
- `tests/test_manuscript_numbers.py`: numbers quoted in the text agree with the result files.
- `tests/test_v110_extensions.py`: closed-form checks of the propositions and estimators.
