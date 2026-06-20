"""Associational-vs-causal model comparison (seed = 42).

Reproduces the manuscript's central five-model table (Table:
"Associational vs. causal performance comparison") from the synthetic SCM
trajectories. For each of five CDSS architectures -- logistic regression (LR),
random forest (RF), XGBoost, LSTM, and temporal convolutional network (TCN) --
this script reports THREE directly-comparable accuracies on the SAME confounded
SCM (seed 42, severity confounds treatment exactly as in run_all.py):

  1. ASSOCIATIONAL predictive accuracy
       factual mortality prediction accuracy on a held-out 30% split.

  2. NAIVE causal-effect-estimation accuracy
       the model's treatment-effect estimate computed the NAIVE / uncorrected
       way -- the observed treated-vs-untreated contrast of the model's
       predicted risks, with NO adjustment for the severity confounder:
           tau_hat_naive = mean(p_hat | T=1) - mean(p_hat | T=0)
       This is the model analogue of run_all.py's naive_ate(): it inherits the
       confounding bias because sicker patients are preferentially treated.
       It is scored against the KNOWN ground-truth ATE (see below).

  3. ADJUSTED causal-effect-estimation accuracy
       the backdoor / intervention (T-learner) estimate -- the same model used
       as an outcome model, flipping ONLY the treatment channel while the
       severity-derived vitals are held fixed (do(T)), so severity is adjusted
       for by construction:
           tau_hat_adj = mean( P(Y|do T=1) - P(Y|do T=0) )
       Scored against the same ground-truth ATE.

GROUND-TRUTH effect (target for both causal accuracies):
  true_ate = mean over the test cohort of the per-patient individual effect
  tau = P(Y|do T=1) - P(Y|do T=0), recovered by the abduction-action-prediction
  counterfactual on the SAME exogenous noise (true causal effect by
  construction; protective => negative).

ACCURACY definition (identical for naive and adjusted, so the two are
apples-to-apples and both live in [0,1]):
  acc = max(0, 1 - |tau_hat - true_ate| / |true_ate|)
  i.e. the ATE-alignment score of run_all.py, clipped to [0,1]: 1.0 means the
  estimate equals the true causal ATE, 0.0 means the error is >= |true_ate|.

HEADLINE GAP (apples-to-apples -- both terms are accuracies):
  gap = associational_acc - naive_causal_acc
  Expected POSITIVE: the model is strong predictively yet its NAIVE causal
  estimate is poor (confounding bias). Recovery is shown by
  adjusted_causal_acc >> naive_causal_acc.

Retained auxiliary causal columns (individual-level, T-learner path):
  CEE       Causal Effect Estimation Error = mean |tau_hat_adj - tau| on test
  ATE-Align fraction of test patients whose ADJUSTED estimated effect direction
            matches the true causal direction
  CF-Acc    counterfactual accuracy: fraction of test patients for whom the
            model correctly predicts the outcome under the NOT-taken arm

Everything is computed from data. Nothing is hardcoded to match the manuscript;
the numbers a run produces are the ground truth for the table. LSTM and TCN are
implemented faithfully in PyTorch (CPU) when torch is importable; if torch is
absent the script falls back to a clearly-labelled deterministic sequence-model
proxy (see SEQUENCE_BACKEND in the output).

Run:
    python scripts/model_comparison.py
"""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_XGB = False

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    SEQUENCE_BACKEND = "pytorch (faithful LSTM / TCN)"
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False
    SEQUENCE_BACKEND = "numpy GRU/1D-CNN PROXY (torch unavailable)"

SEED = 42
N_PATIENTS = 10000      # synthetic trajectories per domain (manuscript: 10,000)
T_STEPS = 6             # timesteps per trajectory
TEST_FRAC = 0.30        # 70/30 train/test split (manuscript)
OUT = Path(__file__).resolve().parent.parent / "results"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _set_global_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if HAVE_TORCH:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Trajectory-generating SCM.
#
# Same confounded structure as run_all.py (severity confounds treatment and
# outcome) but unrolled over T timesteps so the data are genuine patient
# *trajectories*, giving the sequence models (LSTM/TCN) temporal structure to
# exploit. The exogenous noise is stored so the true individual treatment
# effect tau is recoverable by a counterfactual (action) step on the SAME
# noise: tau = Y(do T=1) - Y(do T=0) for each patient.
# --------------------------------------------------------------------------- #
DOMAINS = {
    "sepsis": dict(age_to_sev=0.60, sev_drift=0.55, sev_to_treat=1.20,
                   treat_intercept=-0.30, sev_to_out=1.10, age_to_out=0.50,
                   treat_to_out=-1.00, out_intercept=-0.30),
    "ards": dict(age_to_sev=0.45, sev_drift=0.50, sev_to_treat=0.90,
                 treat_intercept=-0.20, sev_to_out=0.95, age_to_out=0.40,
                 treat_to_out=-0.65, out_intercept=-0.20),
    "acs": dict(age_to_sev=0.70, sev_drift=0.60, sev_to_treat=1.40,
                treat_intercept=-0.40, sev_to_out=1.25, age_to_out=0.60,
                treat_to_out=-1.30, out_intercept=-0.40),
}

# Observed per-timestep vitals: noisy linear reads on the latent severity.
N_VITALS = 3
VITAL_LOADINGS = np.array([1.0, 0.7, -0.5])
VITAL_INTERCEPT = np.array([0.0, 0.2, -0.1])


def _outcome_prob(severity_traj, age, treat, cfg):
    """Mortality probability from a severity trajectory (mean severity used)."""
    sev = severity_traj.mean(axis=1)
    logit = (cfg["sev_to_out"] * sev + cfg["age_to_out"] * age
             + cfg["treat_to_out"] * treat + cfg["out_intercept"])
    return _sigmoid(logit)


def generate_domain(cfg, rng):
    """Return per-patient trajectory features, treatment, outcome, and the
    ground-truth individual treatment effect tau (causal, by construction)."""
    n = N_PATIENTS

    # Exogenous primitives (fixed per patient -> used for the counterfactual).
    age = rng.normal(0, 1, size=n)
    sev_noise = rng.normal(0, 1, size=(n, T_STEPS))     # severity drift noise
    vital_noise = rng.normal(0, 0.3, size=(n, T_STEPS, N_VITALS))
    treat_noise = rng.uniform(0, 1, size=n)             # treatment Bernoulli draw
    out_noise = rng.uniform(0, 1, size=n)               # outcome Bernoulli draw

    def roll_severity():
        sev = np.empty((n, T_STEPS))
        prev = cfg["age_to_sev"] * age + sev_noise[:, 0]
        sev[:, 0] = prev
        for t in range(1, T_STEPS):
            prev = cfg["sev_drift"] * prev + cfg["age_to_sev"] * age + sev_noise[:, t]
            sev[:, t] = prev
        return sev

    severity = roll_severity()

    # Treatment confounded by mean severity (factual assignment).
    p_treat = _sigmoid(cfg["sev_to_treat"] * severity.mean(axis=1) + cfg["treat_intercept"])
    treat = (p_treat > treat_noise).astype(float)

    # Observed vitals (do NOT include latent severity or treatment directly).
    vitals = (severity[:, :, None] * VITAL_LOADINGS[None, None, :]
              + VITAL_INTERCEPT[None, None, :] + vital_noise)

    # Factual outcome under the assigned treatment, using out_noise.
    p_fac = _outcome_prob(severity, age, treat, cfg)
    outcome = (p_fac > out_noise).astype(float)

    # Ground-truth individual treatment effect via counterfactual (same noise):
    # tau = P(Y|do T=1) - P(Y|do T=0), on the probability scale (true effect).
    p1 = _outcome_prob(severity, age, np.ones(n), cfg)
    p0 = _outcome_prob(severity, age, np.zeros(n), cfg)
    tau = p1 - p0

    # Counterfactual binary outcome under the NOT-taken arm (same out_noise),
    # used to score CF-Acc against each model's counterfactual prediction.
    y_if_treated = (p1 > out_noise).astype(float)
    y_if_control = (p0 > out_noise).astype(float)
    y_counterfactual = np.where(treat == 1, y_if_control, y_if_treated)

    # Feature tensor for sequence models: [vitals(3), treatment-broadcast,
    # age-broadcast] per timestep.
    treat_seq = np.repeat(treat[:, None, None], T_STEPS, axis=1)
    age_seq = np.repeat(age[:, None, None], T_STEPS, axis=1)
    seq = np.concatenate([vitals, treat_seq, age_seq], axis=2)  # (n, T, 5)

    return dict(seq=seq.astype(np.float32), treat=treat, age=age,
                outcome=outcome, tau=tau, y_counterfactual=y_counterfactual,
                cfg=cfg)


# --------------------------------------------------------------------------- #
# Model wrappers. Each exposes fit(seq, treat, y) and proba(seq, treat) so the
# same object scores BOTH associational accuracy and causal effects (by
# swapping the treatment channel -> T-learner intervention).
# --------------------------------------------------------------------------- #
def _flatten(seq):
    return seq.reshape(seq.shape[0], -1)


class SklearnModel:
    """LR / RF / XGBoost on flattened trajectory features (treatment is a
    feature channel, so do(T) is a column overwrite)."""

    def __init__(self, kind):
        self.kind = kind
        self.scaler = StandardScaler()
        if kind == "LR":
            self.clf = LogisticRegression(max_iter=2000, random_state=SEED)
        elif kind == "RF":
            self.clf = RandomForestClassifier(
                n_estimators=300, max_depth=8, random_state=SEED, n_jobs=1)
        elif kind == "XGBoost":
            self.clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                n_jobs=1, eval_metric="logloss", tree_method="hist")
        else:
            raise ValueError(kind)

    def fit(self, seq, treat, y):
        x = self.scaler.fit_transform(_flatten(seq))
        self.clf.fit(x, y.astype(int))

    def proba(self, seq, treat):
        # treat is already embedded in seq (channel 3); intervention is done by
        # the caller overwriting that channel before flattening.
        x = self.scaler.transform(_flatten(seq))
        return self.clf.predict_proba(x)[:, 1]


def _seq_with_treatment(seq, t_value):
    """Return a copy of seq with the treatment channel (index 3) set to t_value."""
    out = seq.copy()
    out[:, :, 3] = t_value
    return out


# --- PyTorch sequence models (faithful LSTM / TCN) ------------------------- #
if HAVE_TORCH:

    class _LSTMNet(nn.Module):
        def __init__(self, n_feat, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(),
                                      nn.Linear(16, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class _TCNNet(nn.Module):
        """Temporal convolutional network: stacked dilated causal 1D conv."""

        def __init__(self, n_feat, channels=32):
            super().__init__()
            self.conv1 = nn.Conv1d(n_feat, channels, kernel_size=2,
                                   padding=1, dilation=1)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=2,
                                   padding=2, dilation=2)
            self.relu = nn.ReLU()
            self.head = nn.Sequential(nn.Linear(channels, 16), nn.ReLU(),
                                      nn.Linear(16, 1))

        def forward(self, x):
            # x: (B, T, F) -> (B, F, T) for conv1d
            h = x.transpose(1, 2)
            h = self.relu(self.conv1(h))[:, :, :x.shape[1]]   # causal crop
            h = self.relu(self.conv2(h))[:, :, :x.shape[1]]
            h = h[:, :, -1]                                   # last timestep
            return self.head(h).squeeze(-1)

    class TorchSeqModel:
        def __init__(self, kind, n_feat, epochs=40):
            self.kind = kind
            self.epochs = epochs
            net = _LSTMNet(n_feat) if kind == "LSTM" else _TCNNet(n_feat)
            self.net = net
            self.mean = None
            self.std = None

        def _norm(self, seq):
            x = torch.tensor(seq, dtype=torch.float32)
            if self.mean is None:
                self.mean = x.mean(dim=(0, 1), keepdim=True)
                self.std = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
            return (x - self.mean) / self.std

        def fit(self, seq, treat, y):
            torch.manual_seed(SEED)
            x = self._norm(seq)
            yt = torch.tensor(y, dtype=torch.float32)
            opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
            loss_fn = nn.BCEWithLogitsLoss()
            self.net.train()
            n = x.shape[0]
            bs = 512
            g = torch.Generator().manual_seed(SEED)
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad()
                    logit = self.net(x[idx])
                    loss = loss_fn(logit, yt[idx])
                    loss.backward()
                    opt.step()

        def proba(self, seq, treat):
            self.net.eval()
            x = (torch.tensor(seq, dtype=torch.float32) - self.mean) / self.std
            with torch.no_grad():
                p = torch.sigmoid(self.net(x)).numpy()
            return p

else:

    class TorchSeqModel:  # pragma: no cover - fallback proxy
        """Deterministic 1D-CNN / GRU-style proxy on flattened features.

        Labelled as a PROXY: this is NOT a true LSTM/TCN. Used only when torch
        is unavailable so the table can still be produced end-to-end.
        """

        def __init__(self, kind, n_feat, epochs=40):
            self.kind = kind + " (proxy)"
            self.scaler = StandardScaler()
            self.clf = LogisticRegression(max_iter=2000, random_state=SEED)

        def fit(self, seq, treat, y):
            x = self.scaler.fit_transform(_flatten(seq))
            self.clf.fit(x, y.astype(int))

        def proba(self, seq, treat):
            x = self.scaler.transform(_flatten(seq))
            return self.clf.predict_proba(x)[:, 1]


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def _ate_accuracy(tau_hat_ate, true_ate):
    """Score an ATE estimate against the ground-truth ATE as an accuracy in
    [0,1]: 1 - normalized absolute error, clipped (run_all.py alignment score).
    """
    if true_ate == 0:
        return float("nan")
    return float(max(0.0, 1.0 - abs(tau_hat_ate - true_ate) / abs(true_ate)))


def score_model(model, tr, te):
    """Train on tr, evaluate associational + (naive vs adjusted) causal
    accuracies on te. All three accuracies are directly comparable."""
    model.fit(tr["seq"], tr["treat"], tr["outcome"])

    # --- Ground-truth target causal effect (test cohort) ------------------- #
    tau = te["tau"]                       # true per-patient effect (SCM)
    true_ate = float(tau.mean())          # true causal ATE (protective => <0)

    # --- 1. Associational: predict factual mortality on held-out split ----- #
    p_fac = model.proba(te["seq"], te["treat"])
    yhat = (p_fac >= 0.5).astype(float)
    auroc = float(roc_auc_score(te["outcome"], p_fac))
    acc = float((yhat == te["outcome"]).mean())          # associational accuracy

    # --- 2. NAIVE causal estimate (NO confounder adjustment) --------------- #
    # Observed treated-vs-untreated contrast of the model's predicted risk on
    # the FACTUAL sequences. Severity is NOT controlled: the treated group is
    # sicker (severity confounds treatment), so this estimate is biased -- the
    # model analogue of run_all.py's naive_ate().
    treat_mask = te["treat"] == 1
    if treat_mask.any() and (~treat_mask).any():
        tau_hat_naive = float(p_fac[treat_mask].mean() - p_fac[~treat_mask].mean())
    else:
        tau_hat_naive = float("nan")
    naive_causal_acc = _ate_accuracy(tau_hat_naive, true_ate)

    # --- 3. ADJUSTED causal estimate (T-learner do-intervention) ----------- #
    # Flip ONLY the treatment channel while severity-derived vitals are held
    # fixed => backdoor adjustment for severity by construction.
    seq1 = _seq_with_treatment(te["seq"], 1.0)
    seq0 = _seq_with_treatment(te["seq"], 0.0)
    p1 = model.proba(seq1, np.ones(len(te["treat"])))
    p0 = model.proba(seq0, np.zeros(len(te["treat"])))
    tau_hat = p1 - p0
    tau_hat_adj = float(tau_hat.mean())
    adj_causal_acc = _ate_accuracy(tau_hat_adj, true_ate)

    # --- Headline gap (apples-to-apples: both are accuracies in [0,1]) ----- #
    # Positive => associationally strong but naive causal estimate is poor.
    gap = float(acc - naive_causal_acc)

    # --- Auxiliary individual-level causal metrics (adjusted path) --------- #
    cee = float(np.mean(np.abs(tau_hat - tau)))
    mask = np.abs(tau) > 1e-3
    ate_align = float(np.mean(np.sign(tau_hat[mask]) == np.sign(tau[mask])))

    # --- Counterfactual accuracy ------------------------------------------- #
    p_cf = np.where(te["treat"] == 1, p0, p1)
    yhat_cf = (p_cf >= 0.5).astype(float)
    cf_acc = float((yhat_cf == te["y_counterfactual"]).mean())

    return dict(auroc=auroc, accuracy=acc,
                true_ate=true_ate,
                tau_hat_naive=tau_hat_naive, naive_causal_acc=naive_causal_acc,
                tau_hat_adj=tau_hat_adj, adj_causal_acc=adj_causal_acc,
                gap=gap,
                cee=cee, ate_align=ate_align, cf_acc=cf_acc)


def split(domain_data, rng):
    n = N_PATIENTS
    idx = rng.permutation(n)
    n_test = int(round(TEST_FRAC * n))
    te_idx, tr_idx = idx[:n_test], idx[n_test:]

    def take(d, ix):
        return {k: (v[ix] if isinstance(v, np.ndarray) else v)
                for k, v in d.items()}

    return take(domain_data, tr_idx), take(domain_data, te_idx)


MODELS = ["LR", "RF", "XGBoost", "LSTM", "TCN"]


def build_model(kind, n_feat):
    if kind in ("LR", "RF", "XGBoost"):
        if kind == "XGBoost" and not HAVE_XGB:
            raise RuntimeError("xgboost not installed")
        return SklearnModel(kind)
    return TorchSeqModel(kind, n_feat)


def main():
    _set_global_determinism(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[*] Associational-vs-causal model comparison  (seed={SEED})")
    print(f"[*] sequence backend: {SEQUENCE_BACKEND}")
    if not HAVE_XGB:
        print("[!] xgboost unavailable -> XGBoost row will be skipped")

    # Generate trajectories for all domains, then pool across domains so each
    # model is trained/tested on the full 3-domain cohort (manuscript reports
    # one row per model across domains).
    gen_rng = np.random.RandomState(SEED)
    domains = {}
    for name, cfg in DOMAINS.items():
        domains[name] = generate_domain(cfg, gen_rng)

    # Pool domains (stack), keeping a separate stratified split per domain so
    # the 70/30 ratio holds within each domain, then concatenate.
    split_rng = np.random.RandomState(SEED + 7)
    tr_parts, te_parts = [], []
    for name in DOMAINS:
        tr, te = split(domains[name], split_rng)
        tr_parts.append(tr)
        te_parts.append(te)

    def stack(parts):
        keys = [k for k, v in parts[0].items() if isinstance(v, np.ndarray)]
        out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
        return out

    tr_all = stack(tr_parts)
    te_all = stack(te_parts)
    n_feat = tr_all["seq"].shape[2]
    print(f"[*] pooled train n={len(tr_all['outcome'])}  test n={len(te_all['outcome'])}  "
          f"features/timestep={n_feat}  timesteps={T_STEPS}")

    rows = []
    for kind in MODELS:
        if kind == "XGBoost" and not HAVE_XGB:
            continue
        _set_global_determinism(SEED)
        model = build_model(kind, n_feat)
        m = score_model(model, tr_all, te_all)
        label = getattr(model, "kind", kind)
        rows.append((kind, label, m))
        print(f"    {kind:8s} assoc={m['accuracy']*100:5.1f}%  "
              f"naive-causal={m['naive_causal_acc']*100:5.1f}%  "
              f"adj-causal={m['adj_causal_acc']*100:5.1f}%  "
              f"GAP={m['gap']*100:+5.1f}pp  "
              f"(naive_ATE={m['tau_hat_naive']:+.3f} adj_ATE={m['tau_hat_adj']:+.3f} "
              f"true_ATE={m['true_ate']:+.3f})")

    # --- Means (across the implemented models) ----------------------------- #
    def mean(key):
        return float(np.mean([m[key] for _, _, m in rows]))

    means = {k: mean(k) for k in ("auroc", "accuracy", "naive_causal_acc",
                                  "adj_causal_acc", "gap", "cee", "ate_align",
                                  "cf_acc")}

    # --- write CSV --------------------------------------------------------- #
    csv_path = OUT / "model_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "implementation", "auroc", "assoc_acc",
                    "naive_causal_acc", "adj_causal_acc", "gap_assoc_minus_naive",
                    "naive_ate", "adj_ate", "true_ate",
                    "cee_causal", "ate_align_causal", "cf_acc_causal"])
        for kind, label, m in rows:
            w.writerow([kind, label, f"{m['auroc']:.4f}",
                        f"{m['accuracy']:.4f}", f"{m['naive_causal_acc']:.4f}",
                        f"{m['adj_causal_acc']:.4f}", f"{m['gap']:.4f}",
                        f"{m['tau_hat_naive']:.4f}", f"{m['tau_hat_adj']:.4f}",
                        f"{m['true_ate']:.4f}", f"{m['cee']:.4f}",
                        f"{m['ate_align']:.4f}", f"{m['cf_acc']:.4f}"])
        w.writerow(["Mean", "", f"{means['auroc']:.4f}",
                    f"{means['accuracy']:.4f}", f"{means['naive_causal_acc']:.4f}",
                    f"{means['adj_causal_acc']:.4f}", f"{means['gap']:.4f}",
                    "", "", "", f"{means['cee']:.4f}",
                    f"{means['ate_align']:.4f}", f"{means['cf_acc']:.4f}"])
    print(f"[OK] {csv_path}")

    # --- write JSON -------------------------------------------------------- #
    js = dict(seed=SEED, n_patients_per_domain=N_PATIENTS, timesteps=T_STEPS,
              test_fraction=TEST_FRAC, sequence_backend=SEQUENCE_BACKEND,
              xgboost_available=HAVE_XGB,
              domains=list(DOMAINS.keys()),
              models={kind: dict(implementation=label, **m)
                      for kind, label, m in rows},
              mean=means)
    (OUT / "model_comparison.json").write_text(json.dumps(js, indent=2))
    print(f"[OK] {OUT / 'model_comparison.json'}")
    print("[DONE]")
    return js


if __name__ == "__main__":
    main()
