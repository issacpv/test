"""Evaluation for depth-of-anaesthesia indicators and agent / dataset transfer.

* :func:`prediction_probability` - Pk (Smith, Dutton & Smith, 1996): probability that
  the indicator correctly orders a random pair of observations with different depth,
  with ties in the indicator counted as half.  O(n^2) pairs are subsampled above ``max_pairs``.
* :func:`pk_jackknife` - per-case jackknife standard error.
* :func:`lin_ccc`, :func:`bland_altman` - agreement with BIS.
* :func:`agent_transfer_table` - train-on-A / test-on-B grid for any model factory.
* :func:`age_stratified` - metric by age decade.
* :func:`label_permutation_null` - case-level permutation null for Pk.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def prediction_probability(indicator: np.ndarray, depth: np.ndarray, max_pairs: int = 200_000, seed: int = 0) -> float:
    """Pk between an indicator and a reference depth (higher indicator should mean higher depth)."""
    x, y = np.asarray(indicator, float), np.asarray(depth, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_all = n * (n - 1) // 2
    if n_all <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    dy = y[i] - y[j]
    dx = x[i] - x[j]
    informative = dy != 0
    if informative.sum() == 0:
        return float("nan")
    dy, dx = dy[informative], dx[informative]
    concordant = (np.sign(dx) == np.sign(dy)).sum()
    tied_x = (dx == 0).sum()
    return float((concordant + 0.5 * tied_x) / dy.size)


def pk_jackknife(indicator_by_case: Sequence[np.ndarray], depth_by_case: Sequence[np.ndarray]) -> Dict[str, float]:
    """Pooled Pk with a leave-one-case-out jackknife standard error."""
    xs = [np.asarray(a, float) for a in indicator_by_case]
    ys = [np.asarray(b, float) for b in depth_by_case]
    full = prediction_probability(np.concatenate(xs), np.concatenate(ys))
    n = len(xs)
    if n < 2:
        return {"pk": full, "se": float("nan"), "n_cases": n}
    loo = np.array([prediction_probability(np.concatenate(xs[:k] + xs[k + 1:]), np.concatenate(ys[:k] + ys[k + 1:]))
                    for k in range(n)])
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return {"pk": full, "se": se, "n_cases": n}


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (vx + vy + (mx - my) ** 2 + 1e-12))


def bland_altman(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Bias and 95% limits of agreement of x - y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return {"bias": float("nan"), "loa_lo": float("nan"), "loa_hi": float("nan")}
    return {"bias": float(d.mean()), "loa_lo": float(d.mean() - 1.96 * d.std(ddof=1)),
            "loa_hi": float(d.mean() + 1.96 * d.std(ddof=1))}


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")


ModelFactory = Callable[[], object]


def default_ridge() -> object:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def agent_transfer_table(arms: Dict[str, Dict[str, np.ndarray]], model_factory: ModelFactory = default_ridge,
                         n_splits: int = 5) -> pd.DataFrame:
    """Train on each arm, test on every arm (grouped CV on the diagonal).

    ``arms[name]`` is a dict with keys ``X`` (n, p), ``y`` (n,), ``case`` (n,).  Returns a
    long table with Pk, RMSE and CCC per (train, test).
    """
    from sklearn.model_selection import GroupKFold

    rows = []
    for a, da in arms.items():
        Xa, ya, ga = np.asarray(da["X"], float), np.asarray(da["y"], float), np.asarray(da["case"])
        for b, db in arms.items():
            Xb, yb, gb = np.asarray(db["X"], float), np.asarray(db["y"], float), np.asarray(db["case"])
            if a == b:
                pred = np.full(ya.shape, np.nan)
                k = min(n_splits, len(np.unique(ga)))
                for tr, te in GroupKFold(n_splits=k).split(Xa, ya, ga):
                    m = model_factory()
                    m.fit(Xa[tr], ya[tr])
                    pred[te] = m.predict(Xa[te])
                yt = ya
            else:
                m = model_factory()
                m.fit(Xa, ya)
                pred, yt = m.predict(Xb), yb
            rows.append(dict(train=a, test=b, pk=prediction_probability(pred, yt), rmse=rmse(pred, yt),
                             ccc=lin_ccc(pred, yt), n=int(yt.size)))
    return pd.DataFrame(rows)


def transfer_drop(table: pd.DataFrame, metric: str = "pk") -> pd.DataFrame:
    """Within-arm minus cross-arm metric for each ordered pair (train, test)."""
    diag = {r.train: getattr(r, metric) for r in table.itertuples() if r.train == r.test}
    out = table[table["train"] != table["test"]].copy()
    out["drop"] = [diag[t] - v for t, v in zip(out["train"], out[metric])]
    return out[["train", "test", metric, "drop"]]


def age_stratified(metric_fn: Callable[[np.ndarray, np.ndarray], float], pred: np.ndarray, target: np.ndarray,
                   ages: np.ndarray, edges: Sequence[float] = (18, 40, 50, 60, 70, 80, 120)) -> pd.DataFrame:
    """Metric by age band."""
    pred, target, ages = np.asarray(pred, float), np.asarray(target, float), np.asarray(ages, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ages >= lo) & (ages < hi)
        rows.append(dict(age_band=f"{lo}-{hi}", n=int(m.sum()), value=metric_fn(pred[m], target[m]) if m.sum() > 1 else np.nan))
    return pd.DataFrame(rows)


def label_permutation_null(pred: np.ndarray, target: np.ndarray, cases: np.ndarray, n_perm: int = 200, seed: int = 0) -> Dict[str, float]:
    """Permute the target across cases (keeping within-case series intact) to get a Pk null."""
    rng = np.random.default_rng(seed)
    pred, target, cases = np.asarray(pred, float), np.asarray(target, float), np.asarray(cases)
    obs = prediction_probability(pred, target)
    uniq = np.unique(cases)
    blocks = {c: target[cases == c] for c in uniq}
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(uniq)
        t_perm = np.empty_like(target)
        for c, pc in zip(uniq, perm):
            src = blocks[pc]
            m = cases == c
            t_perm[m] = np.resize(src, m.sum())
        null[i] = prediction_probability(pred, t_perm)
    return {"pk": obs, "null_mean": float(np.nanmean(null)), "p": float((np.sum(null >= obs) + 1) / (n_perm + 1))}


def calibration_slope(pred: np.ndarray, target: np.ndarray) -> float:
    """Slope of target ~ pred (1 = calibrated scale)."""
    p, t = np.asarray(pred, float), np.asarray(target, float)
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 3 or p[ok].std() == 0:
        return float("nan")
    return float(np.polyfit(p[ok], t[ok], 1)[0])
