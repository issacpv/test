"""Cross-cohort transportability: site harmonization, leave-one-cohort-out evaluation
and paired comparison of feature sets (e.g. coupling vs spindle-density features)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# harmonization
# --------------------------------------------------------------------------- #
def _design(covariates: Optional[np.ndarray], n: int) -> np.ndarray:
    if covariates is None:
        return np.ones((n, 1))
    C = np.atleast_2d(np.asarray(covariates, float))
    C = C.T if C.shape[0] != n else C
    return np.column_stack([np.ones(n), C])


class LocationScaleHarmonizer:
    """ComBat-style location/scale adjustment of features per site, preserving covariate effects.

    Fit: regress each feature on covariates (pooled), compute per-site mean and SD of the
    residuals. Transform: standardize residuals per site to the pooled residual scale and
    add the covariate fit back. Sites unseen at fit time are passed through unchanged
    (their residuals are standardized with their *own* statistics when ``allow_new_sites``).
    """

    def __init__(self, allow_new_sites: bool = True):
        self.allow_new_sites = allow_new_sites
        self.beta_: Optional[np.ndarray] = None
        self.site_stats_: Dict[str, tuple] = {}
        self.pooled_sd_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, site: Sequence[str], covariates: Optional[np.ndarray] = None) -> "LocationScaleHarmonizer":
        X = np.asarray(X, float); site = np.asarray(site, str)
        D = _design(covariates, len(X))
        self.beta_, *_ = np.linalg.lstsq(D, X, rcond=None)
        R = X - D @ self.beta_
        self.pooled_sd_ = R.std(axis=0) + 1e-12
        for s in np.unique(site):
            r = R[site == s]
            self.site_stats_[s] = (r.mean(axis=0), r.std(axis=0) + 1e-12)
        return self

    def transform(self, X: np.ndarray, site: Sequence[str], covariates: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, float); site = np.asarray(site, str)
        D = _design(covariates, len(X))
        fit = D @ self.beta_
        R = X - fit
        out = np.empty_like(R)
        for s in np.unique(site):
            m = site == s
            if s in self.site_stats_:
                mu, sd = self.site_stats_[s]
            elif self.allow_new_sites:
                mu, sd = R[m].mean(axis=0), R[m].std(axis=0) + 1e-12
            else:
                out[m] = X[m]
                continue
            out[m] = (R[m] - mu) / sd * self.pooled_sd_ + fit[m]
        return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _default_model():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    if y_pred.std() == 0:
        return np.nan
    return float(np.polyfit(y_pred, y_true, 1)[0])


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return {"mae": float(np.mean(np.abs(y - p))), "r2": float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else np.nan,
            "calibration_slope": calibration_slope(y, p), "n": int(len(y))}


def within_cohort_cv(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
                     model_factory: Callable = _default_model, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions within one cohort (GroupKFold if ``groups`` given, else KFold)."""
    from sklearn.model_selection import GroupKFold, KFold

    n = len(y)
    pred = np.full(n, np.nan)
    if groups is None:
        splitter = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=seed).split(X)
    else:
        g = np.asarray(groups)
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(g)))).split(X, y, g)
    for tr, te in splitter:
        pred[te] = model_factory().fit(X[tr], y[tr]).predict(X[te])
    return pred


def leave_one_cohort_out(X: np.ndarray, y: np.ndarray, cohort: Sequence[str],
                         model_factory: Callable = _default_model, harmonize: bool = False,
                         covariates: Optional[np.ndarray] = None, subject: Optional[Sequence] = None,
                         n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """LOCO evaluation with a within-cohort CV reference for each held-out cohort.

    Returns one row per cohort with ``mae_loco, r2_loco, calib_loco, mae_within, r2_within,
    transport_gap`` (= mae_loco - mae_within). When ``harmonize`` is True, the harmonizer is
    fitted on the training cohorts only (covariate-preserving) and applied to the held-out
    cohort using that cohort's own residual statistics (no labels used).
    """
    X = np.asarray(X, float); y = np.asarray(y, float); cohort = np.asarray(cohort, str)
    rows = []
    for c in np.unique(cohort):
        te = cohort == c; tr = ~te
        Xtr, Xte = X[tr], X[te]
        if harmonize:
            cov_tr = None if covariates is None else np.asarray(covariates)[tr]
            cov_te = None if covariates is None else np.asarray(covariates)[te]
            h = LocationScaleHarmonizer().fit(Xtr, cohort[tr], cov_tr)
            Xtr = h.transform(Xtr, cohort[tr], cov_tr)
            Xte = h.transform(Xte, cohort[te], cov_te)
        pred = model_factory().fit(Xtr, y[tr]).predict(Xte)
        m_loco = _metrics(y[te], pred)
        g = None if subject is None else np.asarray(subject)[te]
        pw = within_cohort_cv(Xte, y[te], g, model_factory, n_splits, seed)
        m_within = _metrics(y[te], pw)
        rows.append({"cohort": c, "n": m_loco["n"], "mae_loco": m_loco["mae"], "r2_loco": m_loco["r2"],
                     "calib_loco": m_loco["calibration_slope"], "mae_within": m_within["mae"],
                     "r2_within": m_within["r2"], "transport_gap": m_loco["mae"] - m_within["mae"]})
    return pd.DataFrame(rows)


def loco_predictions(X: np.ndarray, y: np.ndarray, cohort: Sequence[str], model_factory: Callable = _default_model,
                     harmonize: bool = False, covariates: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-subject LOCO predictions (for paired comparisons across feature sets)."""
    X = np.asarray(X, float); y = np.asarray(y, float); cohort = np.asarray(cohort, str)
    pred = np.full(len(y), np.nan)
    for c in np.unique(cohort):
        te = cohort == c; tr = ~te
        Xtr, Xte = X[tr], X[te]
        if harmonize:
            cov_tr = None if covariates is None else np.asarray(covariates)[tr]
            cov_te = None if covariates is None else np.asarray(covariates)[te]
            h = LocationScaleHarmonizer().fit(Xtr, cohort[tr], cov_tr)
            Xtr, Xte = h.transform(Xtr, cohort[tr], cov_tr), h.transform(Xte, cohort[te], cov_te)
        pred[te] = model_factory().fit(Xtr, y[tr]).predict(Xte)
    return pred


def paired_bootstrap_difference(err_a: np.ndarray, err_b: np.ndarray, groups: Optional[Sequence] = None,
                                n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean difference ``err_a - err_b`` (e.g. absolute errors) with a subject/cluster bootstrap CI."""
    rng = np.random.default_rng(seed)
    a = np.asarray(err_a, float); b = np.asarray(err_b, float)
    d = a - b
    if groups is None:
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    else:
        g = np.asarray(groups); ug = np.unique(g)
        members = {u: np.where(g == u)[0] for u in ug}
        boots = np.array([d[np.concatenate([members[u] for u in rng.choice(ug, len(ug))])].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_boot": min(p, 1.0)}


def compare_feature_sets(feature_sets: Dict[str, np.ndarray], y: np.ndarray, cohort: Sequence[str],
                         model_factory: Callable = _default_model, harmonize: bool = False,
                         covariates: Optional[np.ndarray] = None, reference: Optional[str] = None,
                         subject: Optional[Sequence] = None) -> pd.DataFrame:
    """LOCO MAE per feature set plus paired bootstrap of |error| differences vs ``reference``."""
    y = np.asarray(y, float)
    preds = {k: loco_predictions(v, y, cohort, model_factory, harmonize, covariates) for k, v in feature_sets.items()}
    reference = reference or next(iter(feature_sets))
    rows: List[dict] = []
    for k, p in preds.items():
        err = np.abs(y - p)
        row = {"feature_set": k, "mae_loco": float(err.mean()), "r2_loco": _metrics(y, p)["r2"]}
        if k != reference:
            row.update({f"vs_{reference}_{kk}": vv for kk, vv in
                        paired_bootstrap_difference(err, np.abs(y - preds[reference]), subject).items()})
        rows.append(row)
    return pd.DataFrame(rows)
