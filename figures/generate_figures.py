#!/usr/bin/env python3
"""
generate_figures.py
===================
Batch driver: render all five publication-quality figures for

    "Structural Causal Models for Evaluating CDSS"  (target: KAIS, Springer)

Each figure is produced by its own standalone script (fig01..fig05). Every
script:
  * applies the canonical Top-Tier style from figures/pub_style.py
    (Okabe-Ito colour-blind-safe palette, Times serif, 300 dpi, spines off),
  * loads ALL numeric data from results/ at run time (NOTHING is hardcoded;
    the data-bearing panels read the CSVs written by run_all.py at seed 42),
  * saves a matched vector .pdf + 300-dpi .png with a tight bbox into figures/.

Usage
-----
    python figures/generate_figures.py        # or: cd figures && python generate_figures.py

Figures
-------
  fig01_causal_dag           Three-panel causal DAG schematic (Sepsis|ARDS|ACS)
  fig02_intervention_effects Per-domain ATE forest plot (naive/backdoor/DR vs truth)
  fig03_cate_heterogeneity   CATE heatmap + estimated-vs-true scatter (sepsis)
  fig04_confounding_analysis Grouped bars: naive/backdoor/DR/true + bias
  fig05_backdoor_adjustment  Backdoor demonstration (2 DAG panels + numeric panel)
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "fig01_causal_dag.py",
    "fig02_forest_plot.py",
    "fig03_cate_heterogeneity.py",
    "fig04_confounding_analysis.py",
    "fig05_backdoor_adjustment.py",
]


def main():
    # ensure the standalone scripts can import pub_style and run from anywhere
    sys.path.insert(0, str(HERE))
    cwd0 = Path.cwd()
    import os
    os.chdir(HERE)
    try:
        for s in SCRIPTS:
            print(f"\n=== {s} ===")
            runpy.run_path(str(HERE / s), run_name="__main__")
    finally:
        os.chdir(cwd0)
    print(f"\nAll 5 figures saved to: {HERE}")


if __name__ == "__main__":
    main()
