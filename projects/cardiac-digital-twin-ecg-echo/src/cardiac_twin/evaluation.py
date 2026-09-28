"""Evaluation utilities: regression/classification metrics, patient-level bootstrap, calibration, discordance."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    err = yhat - y
    r = stats.pearsonr(y, yhat)[0] if len(y) > 2 else float("nan")
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r": float(r),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def classification_metrics(y_bin: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_bin = np.asarray(y_bin, int)
    score = np.asarray(score, float)
    if len(np.unique(y_bin)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "prevalence": float(y_bin.mean())}
    return {
        "auroc": float(roc_auc_score(y_bin, score)),
        "auprc": float(average_precision_score(y_bin, score)),
        "prevalence": float(y_bin.mean()),
    }


def cluster_bootstrap(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    yhat: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI by resampling patients (clusters) with replacement."""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        vals.append(metric_fn(y[idx], yhat[idx]))
    vals = np.asarray(vals, float)
    return float(metric_fn(y, yhat)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1) -> float:
    """Gneiting-Raftery interval score (lower is better) for central (1-alpha) intervals."""
    y = np.asarray(y, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    width = hi - lo
    pen = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(width + pen))


def discordance_auroc(abs_error: np.ndarray, uncertainty: np.ndarray, threshold: float = 15.0) -> float:
    """AUROC of the model's uncertainty for flagging discordant predictions (|error| > threshold)."""
    flag = (np.asarray(abs_error, float) > threshold).astype(int)
    if flag.sum() in (0, len(flag)):
        return float("nan")
    return float(roc_auc_score(flag, np.asarray(uncertainty, float)))


def counterfactual_consistency(delta_features: np.ndarray, expected_signs: np.ndarray) -> float:
    """Fraction of (sample, feature) pairs whose change has the physiologically expected sign.

    ``expected_signs`` holds +1 / -1 / 0 per feature (0 = no expectation, ignored).
    """
    d = np.asarray(delta_features, float)
    s = np.asarray(expected_signs, float)
    mask = s != 0
    if d.ndim == 1:
        d = d[None, :]
    agree = np.sign(d[:, mask]) == s[mask]
    return float(agree.mean())


def permutation_null_consistency(delta_features: np.ndarray, expected_signs: np.ndarray, n_perm: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Observed consistency and permutation p-value where feature signs are randomly flipped."""
    rng = np.random.default_rng(seed)
    obs = counterfactual_consistency(delta_features, expected_signs)
    d = np.asarray(delta_features, float)
    null = np.empty(n_perm)
    for k in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=d.shape)
        null[k] = counterfactual_consistency(d * flips, expected_signs)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, float(p)
