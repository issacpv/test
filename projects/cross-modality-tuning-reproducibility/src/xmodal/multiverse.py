"""Distribution-shift statistics, propensity reweighting and sequential decomposition of the modality gap."""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression


def weighted_quantile(x: np.ndarray, q: float, w: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, x))


def distribution_shift(a: np.ndarray, b: np.ndarray, w_a: Optional[np.ndarray] = None,
                       w_b: Optional[np.ndarray] = None) -> Dict[str, float]:
    """KS statistic, Wasserstein distance, median difference and quartile differences (b - a), optionally
    weighted (weights are used for the Wasserstein distance and quantiles; KS is unweighted)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    ks = float(ks_2samp(a, b).statistic) if len(a) and len(b) else np.nan
    wd = float(wasserstein_distance(a, b, u_weights=w_a, v_weights=w_b)) if len(a) and len(b) else np.nan
    out = {"ks": ks, "wasserstein": wd}
    for q, name in ((0.25, "q25"), (0.5, "median"), (0.75, "q75")):
        out[f"{name}_diff"] = weighted_quantile(b, q, w_b) - weighted_quantile(a, q, w_a)
    out["standardised_median_diff"] = out["median_diff"] / (np.sqrt(0.5 * (a.var() + b.var())) + 1e-12)
    return out


def propensity_weights(cov_ref: pd.DataFrame, cov_target: pd.DataFrame, clip: float = 10.0,
                       C: float = 1.0) -> np.ndarray:
    """Weights for the *target* sample that make its covariate distribution resemble the *reference* sample.

    Logistic regression of membership (ref = 1) on covariates; w = p / (1 - p) for target rows, normalised to
    mean 1 and clipped. Effective sample size is stored in ``weights.attrs`` style via a returned attribute-less
    array; use ``effective_sample_size``."""
    X = pd.concat([cov_ref, cov_target], axis=0).to_numpy(dtype=float)
    y = np.concatenate([np.ones(len(cov_ref)), np.zeros(len(cov_target))])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    model = LogisticRegression(C=C, max_iter=1000).fit((X - mu) / sd, y)
    p = model.predict_proba((cov_target.to_numpy(dtype=float) - mu) / sd)[:, 1]
    w = p / np.clip(1.0 - p, 1e-6, None)
    w = np.clip(w / w.mean(), 0.0, clip)
    return w / w.mean()


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def covariate_balance(cov_ref: pd.DataFrame, cov_target: pd.DataFrame, w_target: Optional[np.ndarray] = None) -> pd.Series:
    """Standardised mean differences per covariate (target - ref), optionally weighted."""
    out = {}
    w = np.ones(len(cov_target)) if w_target is None else np.asarray(w_target, dtype=float)
    for c in cov_ref.columns:
        a = cov_ref[c].to_numpy(dtype=float)
        b = cov_target[c].to_numpy(dtype=float)
        mb = np.sum(w * b) / w.sum()
        sd = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
        out[c] = (mb - a.mean()) / sd
    return pd.Series(out, name="smd")


def sequential_gap_decomposition(gap_fn: Callable[[Dict[str, bool]], float], terms: Sequence[str]) -> pd.DataFrame:
    """Shapley-style attribution of the gap closed by each correction term.

    ``gap_fn(flags)`` returns the remaining gap when the corrections whose flag is True have been applied.
    For every ordering of terms the marginal contribution of each term is computed; the average over orderings
    is the attribution. Returns a table with mean and SD over orderings and the share of the raw gap."""
    raw = gap_fn({t: False for t in terms})
    contrib: Dict[str, List[float]] = {t: [] for t in terms}
    for order in itertools.permutations(terms):
        flags = {t: False for t in terms}
        prev = raw
        for t in order:
            flags[t] = True
            cur = gap_fn(dict(flags))
            contrib[t].append(prev - cur)
            prev = cur
    rows = [{"term": t, "closed_mean": float(np.mean(v)), "closed_sd": float(np.std(v)),
             "share_of_raw": float(np.mean(v) / raw) if raw != 0 else np.nan} for t, v in contrib.items()]
    df = pd.DataFrame(rows)
    df.attrs["raw_gap"] = float(raw)
    df.attrs["residual_gap"] = float(gap_fn({t: True for t in terms}))
    return df


def specification_grid(**factors: Sequence[object]) -> pd.DataFrame:
    names = list(factors)
    rows = [dict(zip(names, combo)) for combo in itertools.product(*[list(v) for v in factors.values()])]
    df = pd.DataFrame(rows)
    df.insert(0, "spec_id", np.arange(len(df)))
    return df


def invariance_check(metric_ephys: np.ndarray, metric_ophys: np.ndarray, threshold_sd: float = 0.2) -> Dict[str, object]:
    """Pre-registered invariance criterion: |standardised median difference| < threshold."""
    shift = distribution_shift(metric_ephys, metric_ophys)
    return {"standardised_median_diff": shift["standardised_median_diff"],
            "invariant": bool(abs(shift["standardised_median_diff"]) < threshold_sd), "wasserstein": shift["wasserstein"]}
