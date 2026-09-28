"""Dataset-shift diagnostics for echo videos.

* domain-classifier AUROC and proxy A-distance on descriptor / embedding matrices
* kernel MMD with permutation p-value
* per-descriptor "shift card": standardised mean difference and KS statistic
* effective sample size of importance weights (how much calibration data survives reweighting)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def domain_classifier_auc(Xa: np.ndarray, Xb: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    """Cross-validated AUROC of a gradient-boosting classifier separating A (0) from B (1)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.vstack([Xa, Xb]).astype(float)
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def proxy_a_distance(auc: float) -> float:
    """2 * (2 * AUC - 1), folded so that AUC < 0.5 counts as separable too."""
    return float(2.0 * (2.0 * max(auc, 1 - auc) - 1.0))


def mmd_rbf(Xa: np.ndarray, Xb: np.ndarray, gamma: float | None = None, n_perm: int = 200, seed: int = 0) -> tuple[float, float]:
    """Unbiased MMD^2 with an RBF kernel (median heuristic) and permutation p-value."""
    rng = np.random.default_rng(seed)
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    Z = np.vstack([Xa, Xb])
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Z = (Z - mu) / sd
    if gamma is None:
        sub = Z[rng.choice(len(Z), min(500, len(Z)), replace=False)]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        gamma = 1.0 / (np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0)

    def k(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d2, 0))

    def stat(A: np.ndarray, B: np.ndarray) -> float:
        m, n = len(A), len(B)
        Kaa, Kbb, Kab = k(A, A), k(B, B), k(A, B)
        return float((Kaa.sum() - np.trace(Kaa)) / (m * (m - 1)) + (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1)) - 2 * Kab.mean())

    m = len(Xa)
    obs = stat(Z[:m], Z[m:])
    null = np.array([stat(Z[p[:m]], Z[p[m:]]) for p in (rng.permutation(len(Z)) for _ in range(n_perm))])
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1))


def shift_card(D_source: np.ndarray, D_target: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-descriptor standardised mean difference (Cohen's d) and KS statistic between two datasets."""
    rows = []
    for j, name in enumerate(names):
        a, b = D_source[:, j], D_target[:, j]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 1.0
        d = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
        ks = stats.ks_2samp(a, b).statistic if len(a) and len(b) else np.nan
        rows.append({"descriptor": name, "source_mean": a.mean(), "target_mean": b.mean(), "cohens_d": float(d), "ks": float(ks)})
    return pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)


def effective_sample_size(w: np.ndarray) -> float:
    """Kish ESS = (sum w)^2 / sum w^2."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if w.size else 0.0


def shift_summary(D_source: np.ndarray, D_target: np.ndarray, names: list[str], w_cal: np.ndarray | None = None) -> dict[str, object]:
    """One-call summary for a source -> target pair (used for the README 'shift cards')."""
    auc = domain_classifier_auc(D_source, D_target, n_splits=min(5, max(2, min(len(D_source), len(D_target)) // 5)))
    mmd, p = mmd_rbf(D_source, D_target, n_perm=100)
    out: dict[str, object] = {"domain_auc": auc, "proxy_a_distance": proxy_a_distance(auc), "mmd2": mmd, "mmd_p": p,
                              "card": shift_card(D_source, D_target, names)}
    if w_cal is not None:
        out["ess_cal"] = effective_sample_size(w_cal)
        out["ess_fraction"] = effective_sample_size(w_cal) / len(w_cal)
    return out
