"""Download and prepare the SUPPORT right heart catheterization (RHC) data (Connors et al., 1996).

The file is public at https://hbiostat.org/data/repo/rhc.csv (Vanderbilt Department of
Biostatistics data repository). It is NOT redistributed with this repository: run this script
to fetch it into data/rhc/rhc.csv; the SHA-256 checksum below is verified before use.

Variable roles follow Tchetgen Tchetgen et al. (2020) and Cui et al. (2024, JASA):
treatment-inducing proxies Z = (pafi1, paco21), outcome-inducing proxies W = (ph1, hema1),
the remaining baseline covariates in X. Outcomes: dth30 (death within 30 days, binary) and
t3d30 (days survived up to day 30, the outcome of Cui et al.).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://hbiostat.org/data/repo/rhc.csv"
SHA256 = "9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8"
PATH = Path(__file__).resolve().parents[1] / "data" / "rhc" / "rhc.csv"

Z_COLS = ["pafi1", "paco21"]
W_COLS = ["ph1", "hema1"]
NUM = ["age", "edu", "das2d3pc", "surv2md1", "aps1", "scoma1", "meanbp1", "wblc1", "hrt1",
       "resp1", "temp1", "alb1", "bili1", "crea1", "sod1", "pot1", "wtkilo1",
       "cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx", "liverhx",
       "gibledhx", "malighx", "immunhx", "transhx", "amihx"]
YESNO = ["dnr1", "resp", "card", "neuro", "gastr", "renal", "meta", "hema", "seps",
         "trauma", "ortho"]
CAT = ["sex", "race", "income", "ninsclas", "cat1", "cat2", "ca"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> Path:
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, PATH)
    got = sha256(PATH)
    if got != SHA256:
        raise RuntimeError(f"checksum mismatch for {PATH}: {got}")
    return PATH


def load(hidden: tuple[str, ...] = ()) -> dict:
    """Arrays for the analysis. Columns listed in ``hidden`` are removed from X and returned
    separately (used by the plasmode study to build an unrecorded confounder)."""
    df = pd.read_csv(fetch())
    df["cat2"] = df["cat2"].fillna("None")
    X = df[[c for c in NUM if c not in hidden]].astype(float)
    for c in YESNO:
        if c not in hidden:
            X[c] = (df[c] == "Yes").astype(float)
    X = pd.concat([X, pd.get_dummies(df[CAT], drop_first=True).astype(float)], axis=1)
    keep = X.std() > 0
    X = X.loc[:, keep]
    Xs = ((X - X.mean()) / X.std()).to_numpy()
    std = lambda c: ((df[c] - df[c].mean()) / df[c].std()).to_numpy(float)
    return dict(
        x=Xs, x_names=list(X.columns),
        z=np.column_stack([std(c) for c in Z_COLS]),
        w=np.column_stack([std(c) for c in W_COLS]),
        a=(df["swang1"] == "RHC").to_numpy(float),
        y=(df["dth30"] == "Yes").to_numpy(float),
        days=df["t3d30"].to_numpy(float),
        hidden={c: std(c) for c in hidden},
    )


if __name__ == "__main__":
    p = fetch()
    d = load()
    print(p, d["x"].shape, int(d["a"].sum()), int(d["y"].sum()), file=sys.stdout)
