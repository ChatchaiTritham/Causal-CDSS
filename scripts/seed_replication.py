"""Seed replication of the associational-vs-causal comparison (Table 1 uncertainty).

Re-runs scripts/model_comparison.py under seeds 42..51 (each a fresh cohort, split and model
initialisation) and writes the mean and SD per model to results/model_comparison_seeds.csv.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEEDS = range(42, 52)
frames = []
for s in SEEDS:
    f = REPO / "results" / ("model_comparison.csv" if s == 42 else f"model_comparison_seed{s}.csv")
    if s != 42 and not f.exists():
        subprocess.run([sys.executable, str(REPO / "scripts" / "model_comparison.py")],
                       env={**os.environ, "CAUSAL_SEED": str(s)}, check=True, cwd=REPO)
    d = pd.read_csv(f)
    d["seed"] = s
    frames.append(d)
df = pd.concat(frames)
cols = ["auroc", "assoc_acc", "naive_causal_acc", "adj_causal_acc", "cee_causal", "ate_align_causal"]
out = df.groupby("model", sort=False)[cols].agg(["mean", "std"])
out.columns = [f"{a}_{b}" for a, b in out.columns]
out.insert(0, "n_seeds", df.groupby("model", sort=False).size())
out.round(4).to_csv(REPO / "results" / "model_comparison_seeds.csv")
print(out.round(4).to_string())

# Do the architectures differ in adjusted causal accuracy across seeds? Friedman test over the
# ten seeds (blocks), then pairwise Wilcoxon signed-rank tests with Holm correction.
from itertools import combinations  # noqa: E402

from scipy import stats  # noqa: E402

per = df[df["model"] != "Mean"].pivot(index="seed", columns="model", values="adj_causal_acc")
fr = stats.friedmanchisquare(*[per[c] for c in per.columns])
pairs = []
for a, b in combinations(per.columns, 2):
    pairs.append([a, b, float(stats.wilcoxon(per[a], per[b]).pvalue)])
pairs.sort(key=lambda r: r[2])
m = len(pairs)
for i, r in enumerate(pairs):
    r.append(min(1.0, max(r[2] * (m - i), pairs[i - 1][3] if i else 0.0)))
tests = pd.DataFrame(pairs, columns=["model_a", "model_b", "p_wilcoxon", "p_holm"])
tests.insert(0, "friedman_p", round(fr.pvalue, 6))
tests.to_csv(REPO / "results" / "model_comparison_tests.csv", index=False)
print(f"Friedman chi2={fr.statistic:.2f} p={fr.pvalue:.2e}")
print(tests.round(4).to_string(index=False))
