"""Decomposing a cross-site performance gap into covariate, label and concept shift.

Given a model trained at the source S, evaluated at S and at a target T:

* covariate-shift component: metric on S re-weighted by w(x) = P(T|x)/P(S|x) * n_S/n_T
  (estimated by a domain classifier) minus metric on S.
* label-shift component: metric on S re-weighted by w(y) = q(y)/p(y) (Black-Box Shift
  Estimation; Lipton, Wang & Smola, 2018) minus metric on S.
* concept-shift component: what remains of (metric_T - metric_S).

Because sequential attribution depends on order, the two orderings are averaged
(a 2-player Shapley value).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]


def domain_classifier_weights(X_source: np.ndarray, X_target: np.ndarray, clip: tuple[float, float] = (0.05, 20.0), seed: int = 0) -> np.ndarray:
    """Importance weights for source rows under the target covariate distribution.

    Out-of-fold predictions of a gradient-boosting domain classifier avoid
    over-confident in-sample weights.  Weights are normalised to mean 1 and clipped.
    """
    X = np.vstack([X_source, X_target])
    d = np.r_[np.zeros(len(X_source)), np.ones(len(X_target))]
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed)
    p = cross_val_predict(clf, X, d, cv=5, method="predict_proba")[:, 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    w = p[: len(X_source)] / (1 - p[: len(X_source)]) * (len(X_source) / len(X_target))
    w = np.clip(w, *clip)
    return w / w.mean()


def bbse_label_shift(y_source: np.ndarray, yhat_source: np.ndarray, yhat_target: np.ndarray) -> tuple[np.ndarray, float]:
    """Black-Box Shift Estimation for binary labels.

    Returns (w, q1) where w[k] = q(y=k)/p(y=k) for k in {0,1} and q1 is the
    estimated target positive prevalence.  ``yhat_*`` are hard predictions
    (0/1) from the same source model on held-out source data and on target data.
    """
    y_source, yhat_source, yhat_target = (np.asarray(a).astype(int) for a in (y_source, yhat_source, yhat_target))
    C = np.zeros((2, 2))
    for k in (0, 1):
        for j in (0, 1):
            C[j, k] = np.mean((yhat_source == j) & (y_source == k))  # joint P(yhat=j, y=k)
    mu_t = np.array([np.mean(yhat_target == 0), np.mean(yhat_target == 1)])
    w, *_ = np.linalg.lstsq(C, mu_t, rcond=None)
    w = np.clip(w, 0.0, None)
    p = np.array([np.mean(y_source == 0), np.mean(y_source == 1)])
    q = w * p
    q = q / q.sum() if q.sum() > 0 else p
    w = q / p
    return w, float(q[1])


def _metric(metric: Metric, y: np.ndarray, p: np.ndarray, w: np.ndarray | None) -> float:
    return float(metric(np.asarray(y), np.asarray(p), None if w is None else np.asarray(w)))


def decompose_gap(
    y_s: np.ndarray,
    p_s: np.ndarray,
    y_t: np.ndarray,
    p_t: np.ndarray,
    w_cov: np.ndarray,
    w_lab: np.ndarray,
    metric: Metric,
) -> dict[str, float]:
    """Attribute metric_T - metric_S to covariate, label and concept shift (2-ordering Shapley).

    ``w_cov`` are per-source-row covariate weights, ``w_lab`` the 2-vector from BBSE
    (applied per source row through its label).  ``metric(y, p, sample_weight)``.
    """
    y_s, p_s = np.asarray(y_s), np.asarray(p_s)
    w_c = np.asarray(w_cov)
    w_l = np.asarray(w_lab)[y_s.astype(int)]
    m_s = _metric(metric, y_s, p_s, None)
    m_t = _metric(metric, y_t, p_t, None)
    m_c = _metric(metric, y_s, p_s, w_c)
    m_l = _metric(metric, y_s, p_s, w_l)
    m_cl = _metric(metric, y_s, p_s, w_c * w_l)
    # ordering 1: covariate then label; ordering 2: label then covariate
    cov1, lab1 = m_c - m_s, m_cl - m_c
    lab2, cov2 = m_l - m_s, m_cl - m_l
    cov, lab = 0.5 * (cov1 + cov2), 0.5 * (lab1 + lab2)
    concept = (m_t - m_s) - cov - lab
    return {"metric_source": m_s, "metric_target": m_t, "gap": m_t - m_s, "covariate": cov, "label": lab, "concept": concept}


def bootstrap_decomposition(y_s, p_s, y_t, p_t, w_cov, w_lab, metric: Metric, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Row-level bootstrap (source and target independently) of the decomposition."""
    rng = np.random.default_rng(seed)
    y_s, p_s, y_t, p_t, w_cov = (np.asarray(a) for a in (y_s, p_s, y_t, p_t, w_cov))
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_s), len(y_s))
        j = rng.integers(0, len(y_t), len(y_t))
        rows.append(decompose_gap(y_s[i], p_s[i], y_t[j], p_t[j], w_cov[i], w_lab, metric))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Recalibration
# ----------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def recalibrate(p_fit: np.ndarray, y_fit: np.ndarray, method: str = "intercept") -> Callable[[np.ndarray], np.ndarray]:
    """Return a function mapping raw probabilities to recalibrated ones.

    method: 'intercept' (shift on logit scale), 'platt' (slope + intercept),
    'temperature' (slope only), 'isotonic'.
    """
    z, y = _logit(p_fit), np.asarray(y_fit).astype(int)
    if method == "intercept":
        # intercept-only recalibration: Newton solve for a in y ~ expit(z + a)
        a = 0.0
        for _ in range(50):
            q = _expit(z + a)
            h = -np.sum(q * (1 - q))
            step = np.sum(y - q) / h if h != 0 else 0.0
            a -= step
            if abs(step) < 1e-8:
                break
        return lambda p, a=a: _expit(_logit(p) + a)
    if method == "platt":
        lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return lambda p: _expit(a * _logit(p) + b)
    if method == "temperature":
        lr = LogisticRegression(C=1e6, fit_intercept=False).fit(z.reshape(-1, 1), y)
        a = float(lr.coef_[0, 0])
        return lambda p: _expit(a * _logit(p))
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(p_fit, float), y)
        return lambda p: iso.predict(np.asarray(p, float))
    raise ValueError(method)


def few_shot_curve(p_t: np.ndarray, y_t: np.ndarray, sizes: tuple[int, ...] = (50, 100, 200, 500, 1000, 2000), methods: tuple[str, ...] = ("intercept", "platt", "temperature"), n_rep: int = 50, seed: int = 0, evaluator: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None) -> pd.DataFrame:
    """Recalibration quality vs number of labelled target encounters; fit and eval samples are disjoint."""
    from .metrics import calibration_summary

    evaluator = evaluator or calibration_summary
    rng = np.random.default_rng(seed)
    p_t, y_t = np.asarray(p_t, float), np.asarray(y_t).astype(int)
    rows = []
    for n in sizes:
        if n >= len(y_t) // 2:
            continue
        for r in range(n_rep):
            idx = rng.permutation(len(y_t))
            fit, ev = idx[:n], idx[n:]
            if y_t[fit].sum() == 0 or y_t[fit].sum() == n:
                continue
            for m in methods:
                f = recalibrate(p_t[fit], y_t[fit], m)
                res = evaluator(y_t[ev], f(p_t[ev]))
                rows.append({"n": n, "rep": r, "method": m, **res})
    return pd.DataFrame(rows)
