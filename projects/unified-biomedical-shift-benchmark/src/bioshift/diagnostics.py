"""Shift diagnostics and the covariate / label / concept gap decomposition.

Diagnostics are computed on whatever ``X`` the adapter exposes (handcrafted
features, or a model's penultimate embedding), which makes them comparable
across modalities:

* ``domain_classifier_auc`` - cross-validated AUROC of a classifier separating
  source from target (0.5 = indistinguishable); ``proxy_a_distance`` maps it to
  the Ben-David et al. proxy A-distance.
* ``mmd_rbf`` - kernel MMD^2 with a median-heuristic bandwidth and a permutation p.
* ``bbse_label_shift`` - Black-Box Shift Estimation (Lipton, Wang & Smola, 2018):
  target/source class-prior ratio from the source confusion matrix and the
  target predicted-label distribution.
* ``decompose_gap`` - sequential re-weighting of the *source* evaluation to the
  target covariate distribution (importance weights from the domain
  classifier) and to the target label prior (BBSE), Shapley-averaged over the two
  orderings; the remainder is concept shift.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .metrics import auroc


def domain_classifier_auc(Xs: np.ndarray, Xt: np.ndarray, n_splits: int = 5, seed: int = 0, max_n: int = 5000) -> float:
    """Cross-validated AUROC of a logistic domain classifier (source = 0, target = 1)."""
    rng = np.random.default_rng(seed)
    Xs, Xt = np.asarray(Xs, float), np.asarray(Xt, float)
    if len(Xs) > max_n:
        Xs = Xs[rng.choice(len(Xs), max_n, replace=False)]
    if len(Xt) > max_n:
        Xt = Xt[rng.choice(len(Xt), max_n, replace=False)]
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0)
    cv = StratifiedKFold(n_splits=min(n_splits, int(min(len(Xs), len(Xt)))), shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(auroc(y, p))


def proxy_a_distance(auc: float) -> float:
    return float(2.0 * (2.0 * max(auc, 1 - auc) - 1.0))


def _rbf(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def mmd_rbf(Xa: np.ndarray, Xb: np.ndarray, gamma: Optional[float] = None, n_perm: int = 100, seed: int = 0, max_n: int = 1000) -> Tuple[float, float]:
    """Unbiased MMD^2 (RBF kernel, median heuristic) and permutation p-value."""
    rng = np.random.default_rng(seed)
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    if len(Xa) > max_n:
        Xa = Xa[rng.choice(len(Xa), max_n, replace=False)]
    if len(Xb) > max_n:
        Xb = Xb[rng.choice(len(Xb), max_n, replace=False)]
    Z = np.vstack([Xa, Xb])
    if gamma is None:
        d2 = (Z**2).sum(1)[:, None] + (Z**2).sum(1)[None, :] - 2 * Z @ Z.T
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / med

    def stat(A: np.ndarray, B: np.ndarray) -> float:
        m, n = len(A), len(B)
        Kaa, Kbb, Kab = _rbf(A, A, gamma), _rbf(B, B, gamma), _rbf(A, B, gamma)
        t1 = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1))
        t2 = (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1))
        return float(t1 + t2 - 2 * Kab.mean())

    obs = stat(Xa, Xb)
    m = len(Xa)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(Z))
        null[i] = stat(Z[perm[:m]], Z[perm[m:]])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


def importance_weights(Xs: np.ndarray, Xt: np.ndarray, clip: float = 20.0, seed: int = 0) -> np.ndarray:
    """w(x) ~ p_target(x) / p_source(x) for source rows, from a logistic domain classifier; mean-normalised."""
    Xs, Xt = np.asarray(Xs, float), np.asarray(Xt, float)
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=500, random_state=seed).fit(sc.transform(X), y)
    p = lr.predict_proba(sc.transform(Xs))[:, 1]
    w = (p / np.clip(1 - p, 1e-6, None)) * (len(Xs) / len(Xt))
    w = np.clip(w, 1.0 / clip, clip)
    return w / w.mean()


def bbse_label_shift(y_source: np.ndarray, yhat_source: np.ndarray, yhat_target: np.ndarray, n_classes: int = 2, clip: float = 20.0) -> np.ndarray:
    """Black-Box Shift Estimation of ``q(y)/p(y)`` (Lipton et al., 2018).

    ``yhat_*`` are hard predicted labels from the *same* source model
    (``yhat_source`` on held-out source data).  Solves ``C w = mu`` where
    ``C[i, j] = P_source(yhat = i, y = j)`` and ``mu[i] = P_target(yhat = i)``.
    """
    C = np.zeros((n_classes, n_classes))
    for yh, yt in zip(yhat_source, y_source):
        C[int(yh), int(yt)] += 1
    C /= max(len(y_source), 1)
    mu = np.bincount(np.asarray(yhat_target).astype(int), minlength=n_classes) / max(len(yhat_target), 1)
    try:
        w = np.linalg.solve(C + 1e-6 * np.eye(n_classes), mu)
    except np.linalg.LinAlgError:
        w = np.ones(n_classes)
    return np.clip(w, 1.0 / clip, clip)


def label_weights_from_ratio(y: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    w = ratio[np.asarray(y).astype(int)]
    return w / w.mean()


def decompose_gap(
    y_s: np.ndarray,
    s_s: np.ndarray,
    X_s: np.ndarray,
    y_t: np.ndarray,
    s_t: np.ndarray,
    X_t: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float] = auroc,
    threshold: float = 0.5,
    seed: int = 0,
) -> Dict[str, float]:
    """Decompose ``metric(target) - metric(source)`` into covariate, label and concept parts.

    Parameters are held-out *source* labels/scores/features and *target*
    labels/scores/features for the same model.  Returns the four numbers and the
    two orderings so the order-dependence can be inspected.
    """
    y_s, s_s, y_t, s_t = (np.asarray(a) for a in (y_s, s_s, y_t, s_t))
    m_s = metric(y_s, s_s, None)
    m_t = metric(y_t, s_t, None)
    w_x = importance_weights(X_s, X_t, seed=seed)
    ratio = bbse_label_shift(y_s, (s_s >= threshold).astype(int), (s_t >= threshold).astype(int))
    w_y = label_weights_from_ratio(y_s, ratio)
    w_xy = w_x * w_y
    w_xy = w_xy / w_xy.mean()
    m_x = metric(y_s, s_s, w_x)
    m_y = metric(y_s, s_s, w_y)
    m_xy = metric(y_s, s_s, w_xy)
    # ordering A: covariate first; ordering B: label first
    cov_a, lab_a = m_x - m_s, m_xy - m_x
    lab_b, cov_b = m_y - m_s, m_xy - m_y
    concept = m_t - m_xy
    return {
        "metric_source": float(m_s),
        "metric_target": float(m_t),
        "total_gap": float(m_t - m_s),
        "covariate": float((cov_a + cov_b) / 2),
        "label": float((lab_a + lab_b) / 2),
        "concept": float(concept),
        "covariate_orderA": float(cov_a),
        "covariate_orderB": float(cov_b),
        "label_orderA": float(lab_a),
        "label_orderB": float(lab_b),
        "label_ratio_pos": float(ratio[1]) if len(ratio) > 1 else float("nan"),
        "ess_covariate": float(w_x.sum() ** 2 / np.sum(w_x**2)),
    }
