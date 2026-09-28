"""Distribution-shift estimation and performance-gap decomposition.

Three shift components between a source site S and a target site T:

* **covariate shift**  P_S(x) != P_T(x): estimated with a domain classifier; the density ratio
  w(x) = P_T(x)/P_S(x) = [p(T|x)/(1-p(T|x))] * n_S/n_T (Sugiyama et al., 2007; Bickel et al., 2009).
* **label shift**      P_S(y) != P_T(y): estimated with Black-Box Shift Estimation, BBSE
  (Lipton, Wang & Smola, 2018 ICML) from the source confusion matrix and the target predicted-label
  distribution; w(y) = q(y)/p(y).
* **concept shift**    P_S(y|x) != P_T(y|x): whatever gap remains after re-weighting the source
  evaluation by w(x) and w(y).

:func:`decompose_gap` reports the sequential attribution in both orders and the Shapley average,
with a stay-level bootstrap. Metrics are computed with weighted versions of AUROC, Brier score,
log-loss and calibration-in-the-large so that "the metric under re-weighting" is well defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

Metric = Literal["auroc", "brier", "logloss", "citl", "prevalence"]

# ----------------------------------------------------------------------------------------------
# weighted metrics
# ----------------------------------------------------------------------------------------------


def weighted_auroc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """AUROC with per-sample weights (probability that a random positive outranks a random
    negative, each drawn with probability proportional to its weight). Ties count 1/2."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    p_s, y_s, w_s = p[order], y[order], w[order]
    # cumulative weight of negatives below each score, handling ties by averaging
    neg_w = np.where(y_s == 0, w_s, 0.0)
    cum_neg = np.cumsum(neg_w) - neg_w  # negatives strictly before position (in sorted order)
    # tie handling: for equal scores, give half credit for tied negatives
    _, first_idx, counts = np.unique(p_s, return_index=True, return_counts=True)
    group_id = np.repeat(np.arange(len(first_idx)), counts)
    neg_in_group = np.bincount(group_id, weights=neg_w)
    neg_before_group = np.cumsum(neg_in_group) - neg_in_group
    below = neg_before_group[group_id] + 0.5 * neg_in_group[group_id]
    num = np.sum(np.where(y_s == 1, w_s * below, 0.0))
    den = w_s[y_s == 1].sum() * w_s[y_s == 0].sum()
    return float(num / den)


def weighted_metric(y: np.ndarray, p: np.ndarray, w: np.ndarray | None, metric: Metric) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
    w = w / w.sum()
    if metric == "auroc":
        return weighted_auroc(y, p, w)
    if metric == "brier":
        return float(np.sum(w * (p - y) ** 2))
    if metric == "logloss":
        return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))))
    if metric == "citl":  # calibration-in-the-large on the logit scale: logit(mean y) - logit(mean p)
        my, mp = np.sum(w * y), np.sum(w * p)
        return float(np.log(my / (1 - my)) - np.log(mp / (1 - mp)))
    if metric == "prevalence":
        return float(np.sum(w * y))
    raise ValueError(metric)


# ----------------------------------------------------------------------------------------------
# covariate shift: domain classifier
# ----------------------------------------------------------------------------------------------


@dataclass
class CovariateShift:
    domain_auroc: float          # 0.5 = indistinguishable sites, 1.0 = perfectly separable
    proxy_a_distance: float      # 2 * (1 - 2 * err) in [0, 2] (Ben-David et al., 2007)
    weights_source: np.ndarray   # w(x) = P_T(x) / P_S(x) for the source rows, normalised to mean 1
    effective_sample_size: float  # (sum w)^2 / sum w^2 for the source rows


def covariate_shift(X_source: np.ndarray, X_target: np.ndarray, clf=None, n_splits: int = 5,
                    clip: tuple[float, float] = (0.02, 50.0), random_state: int = 0) -> CovariateShift:
    """Estimate P_T(x)/P_S(x) with a cross-fitted domain classifier.

    Cross-fitting (out-of-fold probabilities) avoids the optimistic bias of scoring rows the
    classifier was trained on. Weights are clipped to ``clip`` and normalised to mean 1.
    """
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=random_state) if clf is None else clf
    X = np.vstack([X_source, X_target])
    d = np.r_[np.zeros(len(X_source)), np.ones(len(X_target))]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prob = cross_val_predict(clone(clf), X, d, cv=cv, method="predict_proba")[:, 1]
    auc = weighted_auroc(d, prob)
    err = np.mean((prob > 0.5) != d)
    n_s, n_t = len(X_source), len(X_target)
    ps = np.clip(prob[:n_s], 1e-4, 1 - 1e-4)
    w = (ps / (1 - ps)) * (n_s / n_t)
    w = np.clip(w, *clip)
    w = w / w.mean()
    ess = w.sum() ** 2 / np.sum(w ** 2)
    return CovariateShift(float(auc), float(2 * (1 - 2 * err)), w, float(ess))


# ----------------------------------------------------------------------------------------------
# label shift: BBSE
# ----------------------------------------------------------------------------------------------


@dataclass
class LabelShift:
    source_prior: np.ndarray     # p(y) on the source hold-out
    target_prior: np.ndarray     # estimated q(y) on the target
    weights_by_class: np.ndarray  # w(y) = q(y)/p(y)
    confusion: np.ndarray         # C[i, j] = P_S(yhat = i | y = j)


def bbse_label_shift(y_source: np.ndarray, yhat_source: np.ndarray, yhat_target: np.ndarray,
                     n_classes: int = 2, ridge: float = 1e-6) -> LabelShift:
    """Black-Box Shift Estimation (Lipton et al., 2018).

    Solves  mu_T(yhat) = C w_p  for w = q(y)/p(y) where C is the source confusion matrix
    (columns = true class, rows = predicted class) and mu_T is the predicted-label distribution
    on the target. Uses hard labels; pass ``yhat = (p >= threshold)`` for a probabilistic model
    (the threshold only needs to give a non-singular confusion matrix). Weights are clipped >= 0.
    """
    y_source = np.asarray(y_source).astype(int)
    yhat_source = np.asarray(yhat_source).astype(int)
    yhat_target = np.asarray(yhat_target).astype(int)
    p = np.bincount(y_source, minlength=n_classes) / len(y_source)
    C = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        m = y_source == j
        if m.any():
            C[:, j] = np.bincount(yhat_source[m], minlength=n_classes) / m.sum()
    mu_t = np.bincount(yhat_target, minlength=n_classes) / len(yhat_target)
    A = C * p[None, :]  # A_ij = P(yhat=i | y=j) p(y=j); mu_t = A w
    w = np.linalg.solve(A.T @ A + ridge * np.eye(n_classes), A.T @ mu_t)
    w = np.clip(w, 0.0, None)
    q = w * p
    q = q / q.sum() if q.sum() > 0 else p
    w = np.where(p > 0, q / p, 0.0)
    return LabelShift(p, q, w, C)


# ----------------------------------------------------------------------------------------------
# decomposition
# ----------------------------------------------------------------------------------------------


@dataclass
class GapDecomposition:
    metric: str
    source_value: float
    target_value: float
    total_gap: float
    covariate: float
    label: float
    concept: float
    covariate_first: tuple[float, float, float]  # (cov, label, concept) with cov applied first
    label_first: tuple[float, float, float]      # (cov, label, concept) with label applied first
    ci: dict[str, tuple[float, float]] | None = None

    def as_dict(self) -> dict[str, float]:
        return {"metric": self.metric, "source": self.source_value, "target": self.target_value,
                "gap": self.total_gap, "covariate": self.covariate, "label": self.label, "concept": self.concept}


def _components(y_s, p_s, y_t, p_t, w_x, w_y, metric: Metric) -> tuple[float, float, float, float, float]:
    """Return (M_S, M_S|cov, M_S|label, M_S|cov+label, M_T)."""
    wy = w_y[np.asarray(y_s).astype(int)]
    m_s = weighted_metric(y_s, p_s, None, metric)
    m_cov = weighted_metric(y_s, p_s, w_x, metric)
    m_lab = weighted_metric(y_s, p_s, wy, metric)
    m_both = weighted_metric(y_s, p_s, w_x * wy, metric)
    m_t = weighted_metric(y_t, p_t, None, metric)
    return m_s, m_cov, m_lab, m_both, m_t


def decompose_gap(y_source: np.ndarray, p_source: np.ndarray, y_target: np.ndarray, p_target: np.ndarray,
                  w_x: np.ndarray, w_y: np.ndarray, metric: Metric = "auroc", n_boot: int = 0,
                  random_state: int = 0) -> GapDecomposition:
    """Decompose M_T - M_S into covariate, label and concept components.

    Sequential attribution with covariate shift first:
        cov = M_S[w_x] - M_S ;  label = M_S[w_x w_y] - M_S[w_x] ;  concept = M_T - M_S[w_x w_y]
    and with label shift first:
        label = M_S[w_y] - M_S ;  cov = M_S[w_x w_y] - M_S[w_y] ;  concept = M_T - M_S[w_x w_y]
    The reported components are the average of the two orders (the two-player Shapley value);
    concept shift is identical in both. ``n_boot`` > 0 adds percentile bootstrap CIs by resampling
    source and target rows independently (the weights are kept fixed, i.e. CIs do not include
    weight-estimation uncertainty; re-estimate weights inside the bootstrap for the full CI).
    """
    y_s, p_s = np.asarray(y_source), np.asarray(p_source)
    y_t, p_t = np.asarray(y_target), np.asarray(p_target)
    w_x = np.asarray(w_x, dtype=float)
    w_y = np.asarray(w_y, dtype=float)

    def one(ys, ps, yt, pt, wx):
        m_s, m_cov, m_lab, m_both, m_t = _components(ys, ps, yt, pt, wx, w_y, metric)
        cf = (m_cov - m_s, m_both - m_cov, m_t - m_both)
        lf = (m_both - m_lab, m_lab - m_s, m_t - m_both)
        return m_s, m_t, cf, lf

    m_s, m_t, cf, lf = one(y_s, p_s, y_t, p_t, w_x)
    cov, lab, con = (cf[0] + lf[0]) / 2, (cf[1] + lf[1]) / 2, cf[2]
    ci = None
    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y_s), len(y_s))
            j = rng.integers(0, len(y_t), len(y_t))
            _, _, cfb, lfb = one(y_s[i], p_s[i], y_t[j], p_t[j], w_x[i])
            boots.append(((cfb[0] + lfb[0]) / 2, (cfb[1] + lfb[1]) / 2, cfb[2]))
        b = np.array(boots)
        ci = {k: (float(np.nanpercentile(b[:, n], 2.5)), float(np.nanpercentile(b[:, n], 97.5)))
              for n, k in enumerate(["covariate", "label", "concept"])}
    return GapDecomposition(metric, float(m_s), float(m_t), float(m_t - m_s), float(cov), float(lab), float(con),
                            tuple(map(float, cf)), tuple(map(float, lf)), ci)


def full_decomposition(model_predict: Callable[[np.ndarray], np.ndarray], X_source: np.ndarray, y_source: np.ndarray,
                       X_target: np.ndarray, y_target: np.ndarray, metrics: tuple[Metric, ...] = ("auroc", "brier", "citl"),
                       threshold: float | None = None, n_boot: int = 0, random_state: int = 0,
                       domain_clf=None) -> dict[str, GapDecomposition]:
    """One-call pipeline: domain-classifier weights + BBSE weights + decomposition per metric.

    ``model_predict`` maps X to P(y=1); ``X_source`` / ``y_source`` must be a hold-out set the
    model was not trained on (otherwise the confusion matrix and M_S are optimistic).
    ``threshold`` for hard labels in BBSE defaults to the source prevalence quantile so that the
    predicted-positive rate on the source equals the source prevalence.
    """
    p_s = np.asarray(model_predict(X_source), dtype=float)
    p_t = np.asarray(model_predict(X_target), dtype=float)
    cs = covariate_shift(X_source, X_target, clf=domain_clf, random_state=random_state)
    if threshold is None:
        threshold = float(np.quantile(p_s, 1 - np.mean(y_source)))
    ls = bbse_label_shift(y_source, p_s >= threshold, p_t >= threshold)
    return {m: decompose_gap(y_source, p_s, y_target, p_t, cs.weights_source, ls.weights_by_class, m, n_boot, random_state)
            for m in metrics}


def concept_shift_test(y_target: np.ndarray, p_target: np.ndarray, w_y: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Diagnostic for residual P(y|x) change on the target after label-shift correction.

    Adjusts target predictions for the estimated prior change (Saerens et al., 2002 prior
    correction), then reports calibration-in-the-large and slope of the adjusted predictions on
    the target. If only covariate + label shift were present, both should be ~0 and ~1.
    """
    from .calibration import calibration_slope_intercept

    p = np.clip(np.asarray(p_target, dtype=float), 1e-7, 1 - 1e-7)
    r = w_y[1] / w_y[0] if w_y[0] > 0 else 1.0
    odds = p / (1 - p) * r
    p_adj = odds / (1 + odds)
    res = calibration_slope_intercept(y_target, p_adj)
    return {"citl_adjusted": res["intercept"], "slope_adjusted": res["slope"], "prior_ratio": float(r)}
