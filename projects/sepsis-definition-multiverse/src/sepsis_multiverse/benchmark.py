"""Early-warning benchmark under multiple sepsis definitions.

Leakage-safe construction: for each stay and prediction hour t, features use only data at
hours <= t and the label is "onset within (t, t + horizon]". Prediction hours after onset are
dropped (cases contribute only pre-onset hours). The same patient-level split is reused for
every specification so that metric differences come from the labels.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def hourly_features(hourly: pd.DataFrame, lookback_h: int = 24, concepts: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Trailing-window summary features (last, min, max, count) per (stay_id, hour), causal only."""
    cols = list(concepts) if concepts else [c for c in hourly.columns]
    df = hourly.sort_index()
    feats = {}
    for c in cols:
        g = df[c].groupby(level=0, group_keys=False)
        feats[f"{c}_last"] = g.apply(lambda s: s.ffill(limit=lookback_h))
        feats[f"{c}_min"] = g.apply(lambda s: s.rolling(lookback_h, min_periods=1).min())
        feats[f"{c}_max"] = g.apply(lambda s: s.rolling(lookback_h, min_periods=1).max())
        feats[f"{c}_n"] = g.apply(lambda s: s.notna().rolling(lookback_h, min_periods=1).sum())
    return pd.DataFrame(feats, index=df.index)


def early_warning_labels(index: pd.MultiIndex, onset_h: pd.Series, horizon_h: int = 12,
                         min_hour: int = 0) -> Tuple[pd.Series, pd.Series]:
    """Binary label per (stay_id, hour): onset within (hour, hour + horizon]. Returns (label, keep_mask)."""
    stay = index.get_level_values(0)
    hour = index.get_level_values(1).to_numpy(float)
    on = onset_h.reindex(stay).to_numpy(float)
    label = (np.isfinite(on)) & (on > hour) & (on <= hour + horizon_h)
    keep = (hour >= min_hour) & ((~np.isfinite(on)) | (hour < on))
    return pd.Series(label.astype(int), index=index), pd.Series(keep, index=index)


def patient_split(stays: Sequence, seed: int = 0, frac_train: float = 0.7, frac_val: float = 0.15) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = np.array(sorted(set(stays)))
    rng.shuffle(s)
    n = len(s)
    a, b = int(frac_train * n), int((frac_train + frac_val) * n)
    return {"train": s[:a], "val": s[a:b], "test": s[b:]}


def default_model_factory() -> Callable[[], object]:
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def evaluate(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {"n": int(len(y)), "prevalence": float(y.mean()) if len(y) else np.nan}
    if len(np.unique(y)) < 2:
        out.update({"auroc": np.nan, "auprc": np.nan, "brier": np.nan})
        return out
    out["auroc"] = float(roc_auc_score(y, p))
    out["auprc"] = float(average_precision_score(y, p))
    out["brier"] = float(np.mean((p - y) ** 2))
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
    out["cal_slope"] = float(lr.coef_[0, 0])
    out["cal_intercept"] = float(lr.intercept_[0] + (lr.coef_[0, 0] - 1) * 0)  # intercept given fitted slope
    return out


def definition_transfer_matrix(
    X: pd.DataFrame, labels_by_spec: Dict[str, pd.Series], keep_by_spec: Dict[str, pd.Series],
    split: Dict[str, np.ndarray], model_factory: Optional[Callable[[], object]] = None, metric: str = "auroc",
) -> pd.DataFrame:
    """Train under definition A (rows), evaluate under definition B (columns) on the shared test stays."""
    mf = model_factory or default_model_factory()
    stay = X.index.get_level_values(0)
    tr_mask, te_mask = np.isin(stay, split["train"]), np.isin(stay, split["test"])
    names = list(labels_by_spec)
    M = pd.DataFrame(np.nan, index=names, columns=names)
    Xv = X.fillna(X.median()).to_numpy(float)
    for a in names:
        ka = keep_by_spec[a].to_numpy(bool) & tr_mask
        ya = labels_by_spec[a].to_numpy(int)[ka]
        if len(np.unique(ya)) < 2:
            continue
        m = mf().fit(Xv[ka], ya)
        for b in names:
            kb = keep_by_spec[b].to_numpy(bool) & te_mask
            yb = labels_by_spec[b].to_numpy(int)[kb]
            if len(np.unique(yb)) < 2:
                continue
            M.loc[a, b] = evaluate(yb, m.predict_proba(Xv[kb])[:, 1])[metric]
    return M


def ranking_stability(metric_table: pd.DataFrame) -> Dict[str, float]:
    """Kendall's W of model rankings across specifications (rows = specs, cols = models)."""
    R = metric_table.rank(axis=1)
    n, k = R.shape
    if n < 2 or k < 2:
        return {"kendall_w": np.nan, "p_friedman": np.nan}
    S = np.sum((R.sum(axis=0) - n * (k + 1) / 2) ** 2)
    W = 12 * S / (n ** 2 * (k ** 3 - k))
    p = stats.friedmanchisquare(*[metric_table[c].to_numpy() for c in metric_table.columns]).pvalue if k >= 3 else np.nan
    return {"kendall_w": float(W), "p_friedman": float(p)}


def range_across_vs_within(metric_table: pd.DataFrame) -> Dict[str, float]:
    """Compare the metric range across specifications (per model) with the range across models (per spec)."""
    across_specs = (metric_table.max(axis=0) - metric_table.min(axis=0)).mean()
    across_models = (metric_table.max(axis=1) - metric_table.min(axis=1)).mean()
    return {"mean_range_across_specs": float(across_specs), "mean_range_across_models": float(across_models),
            "ratio": float(across_specs / across_models) if across_models > 0 else np.nan}
