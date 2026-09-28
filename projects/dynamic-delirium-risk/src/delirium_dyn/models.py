"""Landmark classifiers, transition-intensity regression, assessment weights and the sedation-leakage ablation.

Everything uses grouped cross-validation by ``stay_id`` (or ``subject_id`` when supplied) so that windows of
the same patient never straddle a fold.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str = "logistic") -> Pipeline:
    """Imputation + scaling + classifier. ``kind`` in {"logistic", "gbm"}."""
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if kind == "gbm":
        return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15))
    raise ValueError("kind must be 'logistic' or 'gbm'")


def cross_validated_predictions(
    X: pd.DataFrame, y: Sequence, groups: Sequence, kind: str = "logistic", n_splits: int = 5, sample_weight: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Out-of-fold predicted probabilities (binary) or class-probability matrix (multiclass)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y)
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros((len(y), len(classes))) if len(classes) > 2 else np.zeros(len(y))
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    for tr, te in gkf.split(X, y, groups):
        m = make_model(kind)
        fit_kw = {}
        if sw is not None:
            last = m.steps[-1][0]
            fit_kw[f"{last}__sample_weight"] = sw[tr]
        m.fit(X.iloc[tr], y[tr], **fit_kw)
        p = m.predict_proba(X.iloc[te])
        if len(classes) > 2:
            # align to the global class order
            full = np.zeros((len(te), len(classes)))
            for j, c in enumerate(m.classes_):
                full[:, np.where(classes == c)[0][0]] = p[:, j]
            oof[te] = full
        else:
            oof[te] = p[:, 1]
    return oof


def binary_metrics(y: Sequence[int], p: Sequence[float]) -> Dict[str, float]:
    """AUROC, Brier, calibration intercept and slope (logistic recalibration of the logit)."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan, "brier": float(brier_score_loss(y, p))}
    logit = np.log(p / (1 - p))
    # calibration-in-the-large: log-odds of observed vs mean predicted prevalence (slope fixed at 1)
    ybar = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    out["cal_intercept_at_slope1"] = float(np.log(ybar / (1 - ybar)) - np.log(p.mean() / (1 - p.mean())))
    try:
        fit = sm.Logit(y, sm.add_constant(logit)).fit(disp=0)
        out["cal_slope"] = float(fit.params[1])
    except Exception:  # separation / degenerate
        out["cal_slope"] = np.nan
    return out


def sedation_leakage_ablation(
    lm: pd.DataFrame,
    feature_cols: Sequence[str],
    sedation_cols: Sequence[str],
    assessment_cols: Sequence[str] = (),
    admission_cols: Sequence[str] = (),
    kind: str = "logistic",
    n_splits: int = 5,
) -> pd.DataFrame:
    """AUROC of nested feature sets: admission-only, physiology-only, +assessment, full.

    ``lm`` needs binary ``label`` and ``stay_id``. The "leakage share" is
    ``(AUROC_full - AUROC_physiology) / (AUROC_full - AUROC_admission)`` when the denominator is positive.
    """
    y = lm["label"].to_numpy(dtype=int)
    g = lm["stay_id"].to_numpy()
    phys = [c for c in feature_cols if c not in set(sedation_cols) and c not in set(assessment_cols)]
    sets = {
        "admission_only": list(admission_cols) or phys[: max(1, len(phys) // 4)],
        "physiology_only": phys,
        "physiology_plus_assessment": phys + list(assessment_cols),
        "full": list(feature_cols),
    }
    rows = []
    for name, cols in sets.items():
        if len(cols) == 0:  # prevalence-only baseline (no features): constant prediction
            p = np.full(len(y), y.mean())
        else:
            p = cross_validated_predictions(lm[cols], y, g, kind=kind, n_splits=n_splits)
        m = binary_metrics(y, p)
        if len(cols) == 0:
            m["auroc"] = 0.5
        rows.append({"feature_set": name, "n_features": len(cols), **m})
    res = pd.DataFrame(rows).set_index("feature_set")
    denom = res.loc["full", "auroc"] - res.loc["admission_only", "auroc"]
    share = (res.loc["full", "auroc"] - res.loc["physiology_only", "auroc"]) / denom if denom > 0 else np.nan
    res.attrs["sedation_leakage_share"] = float(share)
    return res


def transition_intensities(
    windows: pd.DataFrame,
    exposures: pd.DataFrame,
    exposure_cols: Sequence[str],
    from_state: str,
    to_state: str,
) -> pd.DataFrame:
    """Poisson regression of the count of ``from_state -> to_state`` transitions on exposures.

    Each window in ``from_state`` (non-terminal next window present) is one unit of exposure time; the outcome
    is 1 when the next window is ``to_state``. ``exposures`` is merged on ``stay_id, window_idx`` (the exposure
    measured in the *from* window). Returns rate ratios with 95% CIs (cluster-robust by stay).
    """
    w = windows.sort_values(["stay_id", "window_idx"]).reset_index(drop=True)
    w["next_state"] = w.groupby("stay_id")["state"].shift(-1)
    at_risk = w[(w["state"] == from_state) & w["next_state"].notna()].copy()
    at_risk["event"] = (at_risk["next_state"] == to_state).astype(int)
    d = at_risk.merge(exposures, on=["stay_id", "window_idx"], how="left")
    X = sm.add_constant(d[list(exposure_cols)].fillna(0.0).astype(float))
    model = sm.GLM(d["event"], X, family=sm.families.Poisson())
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": d["stay_id"].astype("category").cat.codes.to_numpy()})
    ci = fit.conf_int()
    out = pd.DataFrame({"rate_ratio": np.exp(fit.params), "ci_lo": np.exp(ci[0]), "ci_hi": np.exp(ci[1]), "p": fit.pvalues})
    out.attrs["n_at_risk"] = int(len(d))
    out.attrs["n_events"] = int(d["event"].sum())
    return out


def ipaw_weights(windows: pd.DataFrame, features: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Stabilised inverse-probability-of-assessment weights.

    A window is "assessed" when its state is not ``unscreened`` (a CAM-ICU or a coma-level RASS was charted).
    ``P(assessed | features)`` is estimated by logistic regression on the assessment-process and sedation
    features; weights are ``P(assessed) / P(assessed | features)`` for assessed windows and 0 otherwise.
    """
    d = windows.merge(features, on=["stay_id", "window_idx"], how="left")
    d = d[~d["state"].isin(["discharged", "dead"])].copy()
    d["assessed"] = (d["state"] != "unscreened").astype(int)
    X = d[list(feature_cols)].astype(float)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(X, d["assessed"])
    p = np.clip(m.predict_proba(X)[:, 1], 0.02, 0.999)
    marg = d["assessed"].mean()
    d["p_assessed"] = p
    d["ipaw"] = np.where(d["assessed"] == 1, marg / p, 0.0)
    return d[["stay_id", "window_idx", "assessed", "p_assessed", "ipaw"]]


def auroc_by_landmark(lm: pd.DataFrame, p: Sequence[float], min_n: int = 30) -> pd.DataFrame:
    """AUROC per landmark index (binary labels), with row counts; NaN where a class is missing."""
    d = lm[["landmark_idx", "label"]].copy()
    d["p"] = np.asarray(p, dtype=float)
    rows = []
    for t, g in d.groupby("landmark_idx"):
        if len(g) < min_n or g["label"].nunique() < 2:
            rows.append({"landmark_idx": t, "n": len(g), "auroc": np.nan})
        else:
            rows.append({"landmark_idx": t, "n": len(g), "auroc": float(roc_auc_score(g["label"], g["p"]))})
    return pd.DataFrame(rows)
