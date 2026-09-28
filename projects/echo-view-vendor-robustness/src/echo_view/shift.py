"""Cross-vendor shift diagnostics for view classification.

- ``vendor_discriminability``: AUROC of a classifier telling one vendor's frames
  from another's (on features), a proxy for how large the acquisition shift is.
- ``per_view_drop``: in-source minus target recall per canonical view.
- ``decompose_drop``: Shapley-average the target macro-F1 gain contributed by
  each acquisition-matching operation applied before inference.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def vendor_discriminability(Xa: np.ndarray, Xb: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUROC separating vendor A (0) from vendor B (1) on features."""
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def per_view_drop(y_true_src, y_pred_src, y_true_tgt, y_pred_tgt, views) -> dict:
    """Per-view recall drop (source recall minus target recall)."""
    out = {}
    for v in views:
        rs = recall_score(np.asarray(y_true_src) == v, np.asarray(y_pred_src) == v, zero_division=0)
        rt = recall_score(np.asarray(y_true_tgt) == v, np.asarray(y_pred_tgt) == v, zero_division=0)
        out[v] = float(rs - rt)
    return out


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def decompose_drop(eval_fn, op_names, order_average: bool = True) -> dict:
    """Shapley-average each acquisition-matching op's contribution to target macro-F1.

    ``eval_fn(active_frozenset)`` returns target macro-F1 when the named ops are
    applied to the target frames before inference.
    """
    contrib = {n: [] for n in op_names}
    orders = permutations(op_names) if order_average else [tuple(op_names)]
    for order in orders:
        active: set[str] = set()
        prev = eval_fn(frozenset(active))
        for n in order:
            active.add(n)
            cur = eval_fn(frozenset(active))
            contrib[n].append(cur - prev)
            prev = cur
    return {n: float(np.mean(v)) for n, v in contrib.items()}
