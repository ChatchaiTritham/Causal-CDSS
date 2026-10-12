"""Write tables/tab_meta.tex (body rows) from results/meta_learners.csv."""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
d = pd.read_csv(REPO / "results" / "meta_learners.csv")
lines = []
for _, r in d.iterrows():
    if r["learner"].startswith("Oracle"):
        lines.append(r"\midrule")
    ca = f"${r.ca_mean*100:.1f} \pm {r.ca_std*100:.1f}$\%"
    name = r"Oracle $E[\tau \mid X]$" if r.learner.startswith("Oracle") else r.learner
    lines.append(f"{name} & {r.base} & {ca} & ${r.cee_mean:.3f} \pm {r.cee_std:.3f}$ & "
                 f"${r.pehe_mean:.3f} \pm {r.pehe_std:.3f}$ & "
                 f"${r.ate_bias_mean:+.3f} \pm {r.ate_bias_std:.3f}$ \\\\")
(REPO / "tables").mkdir(exist_ok=True)
(REPO / "tables" / "tab_meta.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
