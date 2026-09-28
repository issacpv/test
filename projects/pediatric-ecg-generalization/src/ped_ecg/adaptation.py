"""Shift decomposition and label-efficiency of adult->pediatric adaptation.

- ``decompose_gap``: Shapley-averaged attribution of the infant-vs-adolescent
  AUROC gap to covariate renormalisation, prior reweighting and recalibration.
- ``label_efficiency_curve``: AUROC of an adaptation method as a function of the
  number of pediatric labels used, against a pediatric-only ceiling.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def decompose_gap(operations: dict[str, "callable"], base_auroc_fn, order_average: bool = True) -> dict:
    """Shapley-average the AUROC change contributed by each harmonisation operation.

    Parameters
    ----------
    operations : dict name -> op
        Each op transforms an evaluation state; ``base_auroc_fn(active_set)``
        returns the AUROC when the named operations in ``active_set`` are applied.
    """
    names = list(operations)
    k = len(names)
    contrib = {n: [] for n in names}
    orders = permutations(names) if order_average else [tuple(names)]
    for order in orders:
        active: set[str] = set()
        prev = base_auroc_fn(frozenset(active))
        for n in order:
            active.add(n)
            cur = base_auroc_fn(frozenset(active))
            contrib[n].append(cur - prev)
            prev = cur
    return {n: float(np.mean(v)) for n, v in contrib.items()}


def age_conditioned_standardise(Z: np.ndarray, age: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """Standardise features within age band (a simple age-conditioned normalisation)."""
    Z = np.asarray(Z, float).copy()
    for b in np.unique(bands):
        m = bands == b
        if m.sum() > 1:
            mu = Z[m].mean(0)
            sd = Z[m].std(0) + 1e-8
            Z[m] = (Z[m] - mu) / sd
    return Z


def label_efficiency_curve(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray, y_test: np.ndarray,
    frozen_adult_probe=None,
    n_labels_grid=(25, 50, 100, 250, 500, 1000),
    seed: int = 0,
) -> dict:
    """AUROC vs number of pediatric labels for a linear probe on (adult) features.

    ``frozen_adult_probe`` (optional) provides the zero-shot AUROC reference; the
    ceiling is a probe trained on all available pediatric labels.
    """
    rng = np.random.default_rng(seed)
    Z_train, y_train = np.asarray(Z_train, float), np.asarray(y_train)
    Z_test, y_test = np.asarray(Z_test, float), np.asarray(y_test)
    n = len(y_train)
    sc = StandardScaler().fit(Z_train)
    Ztr, Zte = sc.transform(Z_train), sc.transform(Z_test)

    ceiling_clf = LogisticRegression(max_iter=1000).fit(Ztr, y_train)
    ceiling = _auroc(y_test, ceiling_clf.predict_proba(Zte)[:, 1])

    curve = {}
    for k in n_labels_grid:
        if k > n:
            break
        aucs = []
        for _ in range(5):
            idx = rng.choice(n, k, replace=False)
            if len(np.unique(y_train[idx])) < 2:
                continue
            clf = LogisticRegression(max_iter=1000).fit(Ztr[idx], y_train[idx])
            aucs.append(_auroc(y_test, clf.predict_proba(Zte)[:, 1]))
        curve[k] = float(np.nanmean(aucs)) if aucs else float("nan")

    zero_shot = None
    if frozen_adult_probe is not None:
        zero_shot = _auroc(y_test, frozen_adult_probe(Z_test))

    return {"curve": curve, "ceiling": float(ceiling), "zero_shot": zero_shot}


def labels_to_reach(curve: dict[int, float], ceiling: float, tol: float = 0.02) -> int | None:
    """Smallest label count whose AUROC is within ``tol`` of the ceiling."""
    for k in sorted(curve):
        if np.isfinite(curve[k]) and curve[k] >= ceiling - tol:
            return k
    return None
