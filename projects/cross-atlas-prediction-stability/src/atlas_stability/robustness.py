"""Atlas-robustness metrics.

All functions take dictionaries keyed by atlas name so that any subset of atlases can be
compared. The atlas-robustness index (ARI) combines three components in [0, 1]:

* ``1 - dispersion``: accuracies across atlases are similar (dispersion = range of r divided by
  the mean r, clipped to [0, 1]);
* prediction concordance: subjects receive similar predictions across atlases (mean pairwise
  Spearman correlation of out-of-fold predictions);
* finding concordance: predictive patterns projected into a common vertex space are similar
  (mean pairwise Pearson correlation, optionally z-scored against a parcel-permutation null and
  mapped back to [0, 1]).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .parcellate import node_to_space


def accuracy_dispersion(accuracies: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([v for v in accuracies.values() if np.isfinite(v)])
    if len(vals) < 2:
        return {"range": np.nan, "sd": np.nan, "cv": np.nan, "mean": float(vals.mean()) if len(vals) else np.nan, "normalised_range": np.nan}
    mean = vals.mean()
    rng_ = vals.max() - vals.min()
    return {
        "range": float(rng_),
        "sd": float(vals.std(ddof=1)),
        "cv": float(vals.std(ddof=1) / mean) if mean > 0 else np.nan,
        "mean": float(mean),
        "normalised_range": float(np.clip(rng_ / mean, 0, 1)) if mean > 0 else 1.0,
    }


def _pairwise(d: Dict[str, np.ndarray], fn) -> Tuple[float, Dict[Tuple[str, str], float]]:
    pairs = {}
    for a, b in combinations(sorted(d), 2):
        x, y = np.asarray(d[a], float), np.asarray(d[b], float)
        ok = np.isfinite(x) & np.isfinite(y)
        pairs[(a, b)] = float(fn(x[ok], y[ok])) if ok.sum() > 2 else np.nan
    vals = np.array(list(pairs.values()))
    return float(np.nanmean(vals)) if len(vals) else np.nan, pairs


def prediction_concordance(predictions: Dict[str, np.ndarray]) -> Tuple[float, Dict[Tuple[str, str], float]]:
    """Mean pairwise Spearman correlation of out-of-fold predictions across atlases."""
    return _pairwise(predictions, lambda x, y: stats.spearmanr(x, y).statistic)


def finding_concordance(
    node_maps: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Similarity of node-level findings across atlases after projection to the common space.

    ``node_maps[a]`` is a per-node quantity (e.g. Haufe node strength) for atlas ``a`` and
    ``labels[a]`` its vertex labels. Returns the mean pairwise Pearson r of the projected maps,
    and, if ``n_perm > 0``, a parcel-permutation null (node values shuffled within each atlas
    before projection) with the corresponding z-score and a [0, 1] concordance score
    ``Phi(z) `` for use in the ARI.
    """
    rng = np.random.default_rng() if rng is None else rng
    proj = {a: node_to_space(node_maps[a], labels[a]) for a in node_maps}
    observed, pairs = _pairwise(proj, lambda x, y: np.corrcoef(x, y)[0, 1])
    out: Dict[str, object] = {"mean_r": observed, "pairs": pairs, "null_mean": np.nan, "null_sd": np.nan, "z": np.nan, "score": np.nan}
    if n_perm > 0:
        null = np.empty(n_perm)
        for i in range(n_perm):
            shuffled = {a: node_to_space(rng.permutation(node_maps[a]), labels[a]) for a in node_maps}
            null[i], _ = _pairwise(shuffled, lambda x, y: np.corrcoef(x, y)[0, 1])
        sd = null.std(ddof=1) if n_perm > 1 else np.nan
        z = (observed - null.mean()) / sd if sd and sd > 0 else np.nan
        out.update(null_mean=float(null.mean()), null_sd=float(sd), z=float(z), score=float(stats.norm.cdf(z)) if np.isfinite(z) else np.nan)
    else:
        out["score"] = float(np.clip(observed, 0, 1))
    return out


def atlas_robustness_index(
    accuracies: Dict[str, float],
    predictions: Dict[str, np.ndarray],
    finding_score: float,
    weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Combine the three components into ARI in [0, 1] (weighted mean)."""
    disp = accuracy_dispersion(accuracies)
    pc, _ = prediction_concordance(predictions)
    comps = np.array([1 - disp["normalised_range"], np.clip(pc, 0, 1), np.clip(finding_score, 0, 1)])
    w = np.asarray(weights, float)
    ok = np.isfinite(comps)
    ari = float((comps[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    return {"ari": ari, "accuracy_component": float(comps[0]), "prediction_concordance": float(pc), "finding_component": float(finding_score), "mean_r": disp["mean"], "range_r": disp["range"]}


def posthoc_selection_inflation(acc: np.ndarray, ensemble_acc: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Inflation of accuracy from choosing the best atlas post hoc.

    ``acc`` is a ``(n_folds_or_repeats, n_atlases)`` matrix of out-of-fold accuracies. "Best-of"
    picks the best atlas on the same folds; "nested" picks, for each fold, the atlas that was
    best on the *other* folds. ``ensemble_acc`` optionally gives the accuracy of averaging
    predictions across atlases per fold.
    """
    A = np.asarray(acc, float)
    best_of = float(np.nanmean(np.nanmax(A, axis=1)))
    nested = []
    for f in range(A.shape[0]):
        others = np.delete(A, f, axis=0)
        j = int(np.nanargmax(np.nanmean(others, axis=0)))
        nested.append(A[f, j])
    nested_acc = float(np.nanmean(nested))
    out = {"best_of": best_of, "nested": nested_acc, "inflation": best_of - nested_acc, "mean_atlas": float(np.nanmean(A))}
    if ensemble_acc is not None:
        out["ensemble"] = float(np.nanmean(ensemble_acc))
        out["ensemble_minus_nested"] = out["ensemble"] - nested_acc
    return out
