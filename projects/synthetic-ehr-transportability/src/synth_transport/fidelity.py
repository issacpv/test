"""Fidelity and privacy-proxy metrics between a real and a synthetic table."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .generators import detect_binary_columns

__all__ = ["marginal_fidelity", "correlation_distance", "pmse", "mmd_rbf", "dcr", "missingness_agreement", "fidelity_report"]


def marginal_fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    """Per-column KS statistic (numeric) or total-variation distance (binary); lower is better."""
    binary = set(detect_binary_columns(real))
    rows = []
    for c in real.columns:
        r = real[c].dropna().to_numpy(dtype=float)
        s = syn[c].dropna().to_numpy(dtype=float)
        if c in binary:
            stat = abs(r.mean() - s.mean())
            kind = "tv"
        else:
            stat = float(stats.ks_2samp(r, s).statistic) if r.size and s.size else np.nan
            kind = "ks"
        rows.append({"feature": c, "metric": kind, "distance": float(stat)})
    return pd.DataFrame(rows)


def correlation_distance(real: pd.DataFrame, syn: pd.DataFrame) -> float:
    """Mean absolute difference of the off-diagonal Pearson correlation matrices."""
    cr = np.nan_to_num(real.corr().to_numpy())
    cs = np.nan_to_num(syn[real.columns].corr().to_numpy())
    iu = np.triu_indices_from(cr, k=1)
    return float(np.mean(np.abs(cr[iu] - cs[iu])))


def pmse(real: pd.DataFrame, syn: pd.DataFrame, n_splits: int = 5, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Propensity-score mean-squared error (Snoke et al., 2018) with its null-expected ratio.

    A logistic regression is trained (cross-fitted) to distinguish real (0) from synthetic (1);
    ``pmse = mean((p_hat - c)^2)`` with ``c = n_syn / (n_real + n_syn)``. Under the null of identical
    distributions the expectation is ``(k - 1) (1 - c)^2 c / N`` where ``k`` is the number of model
    parameters, so ``ratio ~ 1`` indicates indistinguishable tables and larger values worse fidelity.
    """
    rng = rng or np.random.default_rng(0)
    X = pd.concat([real, syn[real.columns]], axis=0).to_numpy(dtype=float)
    y = np.r_[np.zeros(len(real)), np.ones(len(syn))]
    X = np.nan_to_num(X)
    N = len(y)
    c = len(syn) / N
    p_hat = np.zeros(N)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(2**31 - 1)))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        p_hat[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    val = float(np.mean((p_hat - c) ** 2))
    k = X.shape[1] + 1
    null = (k - 1) * (1 - c) ** 2 * c / N
    return {"pmse": val, "null_expected": float(null), "ratio": float(val / null) if null > 0 else np.nan}


def mmd_rbf(real: pd.DataFrame, syn: pd.DataFrame, gamma: Optional[float] = None, max_n: int = 2000, rng: Optional[np.random.Generator] = None) -> float:
    """Unbiased RBF maximum mean discrepancy on standardised features (subsampled to ``max_n``)."""
    rng = rng or np.random.default_rng(0)
    A = np.nan_to_num(real.to_numpy(dtype=float))
    B = np.nan_to_num(syn[real.columns].to_numpy(dtype=float))
    sc = StandardScaler().fit(A)
    A, B = sc.transform(A), sc.transform(B)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    if gamma is None:
        gamma = 1.0 / A.shape[1]

    def k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d2 = np.sum(x**2, 1)[:, None] + np.sum(y**2, 1)[None, :] - 2 * x @ y.T
        return np.exp(-gamma * np.maximum(d2, 0))

    kaa, kbb, kab = k(A, A), k(B, B), k(A, B)
    m, n = len(A), len(B)
    np.fill_diagonal(kaa, 0)
    np.fill_diagonal(kbb, 0)
    return float(kaa.sum() / (m * (m - 1)) + kbb.sum() / (n * (n - 1)) - 2 * kab.mean())


def dcr(real_train: pd.DataFrame, syn: pd.DataFrame, real_holdout: Optional[pd.DataFrame] = None, max_n: int = 5000, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Distance to closest (training) record: 5th percentile and median, standardised Euclidean.

    If ``real_holdout`` is given, the same statistics are computed for holdout-vs-train as the
    privacy baseline: synthetic DCR clearly *below* the holdout DCR suggests memorisation.
    """
    rng = rng or np.random.default_rng(0)
    cols = list(real_train.columns)
    sc = StandardScaler().fit(np.nan_to_num(real_train.to_numpy(dtype=float)))

    def prep(df: pd.DataFrame) -> np.ndarray:
        X = sc.transform(np.nan_to_num(df[cols].to_numpy(dtype=float)))
        return X[rng.choice(len(X), max_n, replace=False)] if len(X) > max_n else X

    T = prep(real_train)

    def nn_dist(Q: np.ndarray) -> np.ndarray:
        out = np.empty(len(Q))
        for i in range(0, len(Q), 500):
            q = Q[i : i + 500]
            d2 = np.sum(q**2, 1)[:, None] + np.sum(T**2, 1)[None, :] - 2 * q @ T.T
            out[i : i + 500] = np.sqrt(np.maximum(d2.min(axis=1), 0))
        return out

    ds = nn_dist(prep(syn))
    res = {"syn_p5": float(np.percentile(ds, 5)), "syn_median": float(np.median(ds))}
    if real_holdout is not None:
        dh = nn_dist(prep(real_holdout))
        res.update({"holdout_p5": float(np.percentile(dh, 5)), "holdout_median": float(np.median(dh))})
    return res


def missingness_agreement(real: pd.DataFrame, syn: pd.DataFrame, prefix: str = "miss_") -> float:
    """Mean absolute difference in missingness-indicator prevalence (observation-process fidelity)."""
    cols = [c for c in real.columns if c.startswith(prefix)]
    if not cols:
        return float("nan")
    return float(np.mean([abs(real[c].mean() - syn[c].mean()) for c in cols]))


def fidelity_report(real: pd.DataFrame, syn: pd.DataFrame, real_holdout: Optional[pd.DataFrame] = None, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """One-row summary of all fidelity/privacy metrics."""
    rng = rng or np.random.default_rng(0)
    marg = marginal_fidelity(real, syn)
    out = {
        "marginal_mean": float(marg["distance"].mean()),
        "marginal_max": float(marg["distance"].max()),
        "corr_distance": correlation_distance(real, syn),
        "mmd": mmd_rbf(real, syn, rng=rng),
        "missingness_gap": missingness_agreement(real, syn),
    }
    out.update({f"pmse_{k}": v for k, v in pmse(real, syn, rng=rng).items()})
    out.update({f"dcr_{k}": v for k, v in dcr(real, syn, real_holdout, rng=rng).items()})
    return out
