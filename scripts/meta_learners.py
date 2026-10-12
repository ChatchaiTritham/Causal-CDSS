"""Meta-learner comparison for the individual treatment effect (verification cohort).

Uses the trajectory SCM, domains and split of scripts/model_comparison.py (imported, not
modified). Inputs = flattened observed vitals over all time steps + age; the latent severity
and the treatment channel are never features. Learners: S-, T-, X- (Kuenzel et al. 2019) and
cross-fitted DR-learner (Kennedy 2023), each with a logistic/linear base and a histogram
gradient-boosting base. Scored against the generator's true tau on the 30% test split, seeds
42..51 (cohort RandomState(seed), split RandomState(seed+7), as in model_comparison).
Oracle floor: HistGB regression of the TRUE tau on the same inputs (train split) -> CEE on test.

Run: python scripts/meta_learners.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_comparison import DOMAINS, generate_domain, split

OUT = Path(__file__).resolve().parent.parent / "results"
SEEDS = range(42, 52)
EPS = 0.01

try:
    import econml  # noqa: F401
    HAVE_ECONML = True
except Exception:
    HAVE_ECONML = False


def features(d):
    vit = d["seq"][:, :, :3].reshape(len(d["seq"]), -1)       # vitals x time steps
    return np.column_stack([vit, d["age"]]).astype(float)


def clf(base, seed):
    if base == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return HistGradientBoostingClassifier(random_state=seed)


def reg(base, seed):
    if base == "LR":
        return make_pipeline(StandardScaler(), LinearRegression())
    return HistGradientBoostingRegressor(random_state=seed)


def p1(m, x):
    return m.predict_proba(x)[:, 1]


def s_learner(X, T, Y, Xt, base, seed):
    m = clf(base, seed).fit(np.column_stack([X, T]), Y)
    n = len(Xt)
    return p1(m, np.column_stack([Xt, np.ones(n)])) - p1(m, np.column_stack([Xt, np.zeros(n)]))


def t_learner(X, T, Y, Xt, base, seed):
    m1 = clf(base, seed).fit(X[T == 1], Y[T == 1])
    m0 = clf(base, seed).fit(X[T == 0], Y[T == 0])
    return p1(m1, Xt) - p1(m0, Xt)


def x_learner(X, T, Y, Xt, base, seed):
    m1 = clf(base, seed).fit(X[T == 1], Y[T == 1])
    m0 = clf(base, seed).fit(X[T == 0], Y[T == 0])
    d1 = Y[T == 1] - p1(m0, X[T == 1])
    d0 = p1(m1, X[T == 0]) - Y[T == 0]
    g1 = reg(base, seed).fit(X[T == 1], d1)
    g0 = reg(base, seed).fit(X[T == 0], d0)
    e = np.clip(p1(clf(base, seed).fit(X, T), Xt), EPS, 1 - EPS)
    return e * g0.predict(Xt) + (1 - e) * g1.predict(Xt)


def dr_learner(X, T, Y, Xt, base, seed, k=2):
    folds = np.random.RandomState(seed).permutation(len(X)) % k
    psi = np.empty(len(X))
    for f in range(k):
        a, b = folds != f, folds == f
        Xa, Ta, Ya = X[a], T[a], Y[a]
        m1 = clf(base, seed).fit(Xa[Ta == 1], Ya[Ta == 1])
        m0 = clf(base, seed).fit(Xa[Ta == 0], Ya[Ta == 0])
        e = np.clip(p1(clf(base, seed).fit(Xa, Ta), X[b]), EPS, 1 - EPS)
        mu1, mu0 = p1(m1, X[b]), p1(m0, X[b])
        Tb, Yb = T[b], Y[b]
        psi[b] = mu1 - mu0 + Tb * (Yb - mu1) / e - (1 - Tb) * (Yb - mu0) / (1 - e)
    return reg(base, seed).fit(X, psi).predict(Xt)


LEARNERS = {"S-learner": s_learner, "T-learner": t_learner,
            "X-learner": x_learner, "DR-learner": dr_learner}


def metrics(th, tau):
    ate, ate_hat = tau.mean(), th.mean()
    return dict(ca=max(0.0, 1 - abs(ate_hat - ate) / abs(ate)),
                cee=float(np.mean(np.abs(th - tau))),
                pehe=float(np.sqrt(np.mean((th - tau) ** 2))),
                ate_bias=float(ate_hat - ate), true_ate=float(ate))


def cohort(seed):
    g = np.random.RandomState(seed)
    doms = {n: generate_domain(c, g) for n, c in DOMAINS.items()}
    srng = np.random.RandomState(seed + 7)
    tr, te = zip(*(split(doms[n], srng) for n in DOMAINS))
    cat = lambda ps, k: np.concatenate([p[k] for p in ps])
    return [(features_cat(ps), cat(ps, "treat"), cat(ps, "outcome"), cat(ps, "tau"))
            for ps in (tr, te)]


def features_cat(parts):
    return np.concatenate([features(p) for p in parts])


def main():
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        (X, T, Y, tau_tr), (Xt, _, _, tau) = cohort(seed)
        for name, fn in LEARNERS.items():
            for base in ("LR", "HistGB"):
                th = fn(X, T, Y, Xt, base, seed)
                rows.append(dict(seed=seed, learner=name, base=base, **metrics(th, tau)))
        if HAVE_ECONML:
            from econml.dml import CausalForestDML
            cf = CausalForestDML(discrete_treatment=True, random_state=seed).fit(Y, T, X=X)
            rows.append(dict(seed=seed, learner="Causal forest", base="-",
                             **metrics(cf.effect(Xt).ravel(), tau)))
        orc = HistGradientBoostingRegressor(random_state=seed).fit(X, tau_tr).predict(Xt)
        rows.append(dict(seed=seed, learner="Oracle E[tau|X]", base="HistGB", **metrics(orc, tau)))
        print(f"[seed {seed}] done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.round(6).to_csv(OUT / "meta_learners_seeds.csv", index=False)

    agg = df.groupby(["learner", "base"], sort=False)[["ca", "cee", "pehe", "ate_bias"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "n_seeds", df.groupby(["learner", "base"], sort=False).size())
    agg.round(4).to_csv(OUT / "meta_learners.csv")

    est = df[~df["learner"].str.startswith("Oracle")]
    piv = est.pivot_table(index="seed", columns=["learner", "base"], values="cee")
    stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1).mean().sort_values()
    with open(OUT / "meta_learners_friedman.txt", "w") as f:
        f.write(f"Friedman test on CEE across {piv.shape[1]} learner x base combos, "
                f"{piv.shape[0]} seeds: chi2={stat:.3f}, p={p:.3g}\nMean ranks (1=lowest CEE):\n")
        for k, v in ranks.items():
            f.write(f"  {k[0]} / {k[1]}: {v:.2f}\n")
    print(agg.round(4).to_string())
    print(open(OUT / "meta_learners_friedman.txt").read())
    print(f"econml available: {HAVE_ECONML}; runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
