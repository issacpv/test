"""Train-on-X / test-on-Y matrices and the synthetic-gap vs site-gap decomposition.

Notation for the 2x2 design (one task, one model class, one generator):

    real_src -> src   model trained on real source data, tested on the source holdout
    syn_src  -> src   trained on synthetic source data, tested on the source holdout
    real_src -> tgt   trained on real source data, tested on the external target
    syn_src  -> tgt   trained on synthetic source data, tested on the external target

    synthetic_gap_src = m(real->src) - m(syn->src)          (what TSTR measures)
    site_gap_real     = m(real->src) - m(real->tgt)         (ordinary transport loss)
    site_gap_syn      = m(syn->src)  - m(syn->tgt)
    amplification     = site_gap_syn - site_gap_real
                      = [m(syn->src) - m(syn->tgt)] - [m(real->src) - m(real->tgt)]

For "higher is better" metrics (AUROC) a positive amplification means the synthetic-trained model
loses *more* at the target than the real-trained one. For "lower is better" metrics (Brier, ECE,
|calibration intercept|) the sign convention is flipped internally by ``gap_decomposition``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "fit_model",
    "evaluate",
    "tstr_matrix",
    "gap_decomposition",
    "bootstrap_decomposition",
    "subgroup_metrics",
    "LOWER_IS_BETTER",
]

LOWER_IS_BETTER = {"brier", "ece", "abs_cal_intercept", "abs_log_slope"}


def fit_model(X: pd.DataFrame, y: np.ndarray, kind: str = "logreg", seed: int = 0):
    """Fit a classifier: ``logreg`` (standardised L2 logistic regression) or ``gbm``."""
    if kind == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    elif kind == "gbm":
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    else:
        raise ValueError("kind must be logreg|gbm")
    return model.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=int))


def _calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    # intercept given slope 1 (calibration-in-the-large) and slope via logistic recalibration
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
    slope = float(lr.coef_[0, 0])
    intercept_given_slope1 = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean()))) if 0 < y.mean() < 1 else np.nan
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {"cal_intercept": intercept_given_slope1, "cal_slope": slope, "ece": float(ece), "abs_cal_intercept": abs(intercept_given_slope1), "abs_log_slope": abs(np.log(max(slope, 1e-6)))}


def evaluate(model, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    """AUROC, AUPRC, Brier, calibration intercept/slope, ECE for a fitted model."""
    y = np.asarray(y, dtype=int)
    p = model.predict_proba(X.to_numpy(dtype=float))[:, 1]
    out = {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "prevalence": float(y.mean()),
        "n": int(len(y)),
    }
    out.update(_calibration(y, p))
    return out


def tstr_matrix(
    train_sets: Mapping[str, Tuple[pd.DataFrame, np.ndarray]],
    test_sets: Mapping[str, Tuple[pd.DataFrame, np.ndarray]],
    kind: str = "logreg",
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate every training set on every test set; one row per (train, test) pair."""
    rows = []
    for tr_name, (Xtr, ytr) in train_sets.items():
        model = fit_model(Xtr, ytr, kind=kind, seed=seed)
        for te_name, (Xte, yte) in test_sets.items():
            m = evaluate(model, Xte[Xtr.columns], yte)
            rows.append({"train": tr_name, "test": te_name, "model": kind, **m})
    return pd.DataFrame(rows)


def gap_decomposition(matrix: pd.DataFrame, metric: str = "auroc", real: str = "real_src", syn: str = "syn_src", src: str = "src", tgt: str = "tgt") -> Dict[str, float]:
    """Synthetic gap, site gaps and amplification from a ``tstr_matrix`` output.

    All gaps are expressed as *losses* (positive = worse) regardless of the metric's direction.
    """
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0

    def m(tr: str, te: str) -> float:
        sel = matrix[(matrix["train"] == tr) & (matrix["test"] == te)]
        if sel.empty:
            raise KeyError(f"missing cell train={tr} test={te}")
        return sign * float(sel[metric].iloc[0])

    rs, ss, rt, st = m(real, src), m(syn, src), m(real, tgt), m(syn, tgt)
    out = {
        "synthetic_gap_src": rs - ss,
        "synthetic_gap_tgt": rt - st,
        "site_gap_real": rs - rt,
        "site_gap_syn": ss - st,
    }
    out["amplification"] = out["site_gap_syn"] - out["site_gap_real"]
    return out


def bootstrap_decomposition(
    models: Mapping[str, object],
    test_sets: Mapping[str, Tuple[pd.DataFrame, np.ndarray]],
    metric: str = "auroc",
    n_boot: int = 200,
    rng: Optional[np.random.Generator] = None,
    real: str = "real_src",
    syn: str = "syn_src",
    src: str = "src",
    tgt: str = "tgt",
) -> pd.DataFrame:
    """Bootstrap (over test encounters) the decomposition for fitted models.

    ``models`` maps training-set names (``real_src``, ``syn_src``) to fitted classifiers;
    ``test_sets`` maps ``src`` / ``tgt`` to ``(X, y)``. Returns the point estimate and percentile
    95% CIs for each component.
    """
    rng = rng or np.random.default_rng(0)
    # cache predictions once
    preds: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for tr, model in models.items():
        for te, (X, y) in test_sets.items():
            preds[(tr, te)] = (model.predict_proba(X.to_numpy(dtype=float))[:, 1], np.asarray(y, dtype=int))

    def metric_from(p: np.ndarray, y: np.ndarray) -> float:
        if metric == "auroc":
            return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        if metric == "auprc":
            return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        if metric == "brier":
            return float(brier_score_loss(y, p))
        return float(_calibration(y, p)[metric])

    def decompose(idx: Dict[str, np.ndarray]) -> Dict[str, float]:
        rows = []
        for (tr, te), (p, y) in preds.items():
            i = idx[te]
            rows.append({"train": tr, "test": te, metric: metric_from(p[i], y[i])})
        return gap_decomposition(pd.DataFrame(rows), metric=metric, real=real, syn=syn, src=src, tgt=tgt)

    full = {te: np.arange(len(y)) for te, (_, y) in test_sets.items()}
    point = decompose(full)
    boots = []
    for _ in range(n_boot):
        idx = {te: rng.integers(0, len(y), len(y)) for te, (_, y) in test_sets.items()}
        boots.append(decompose(idx))
    bdf = pd.DataFrame(boots)
    rows = []
    for comp, val in point.items():
        rows.append({"component": comp, "estimate": val, "ci_low": float(np.nanpercentile(bdf[comp], 2.5)), "ci_high": float(np.nanpercentile(bdf[comp], 97.5))})
    return pd.DataFrame(rows)


def subgroup_metrics(model, X: pd.DataFrame, y: np.ndarray, groups: Sequence, min_n: int = 50) -> pd.DataFrame:
    """Per-subgroup evaluation (AUROC, Brier, calibration) plus max-min gaps."""
    groups = np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() < min_n or len(np.unique(np.asarray(y)[m])) < 2:
            continue
        rows.append({"group": g, **evaluate(model, X[m], np.asarray(y)[m])})
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ("auroc", "brier", "ece", "cal_intercept"):
            df[f"gap_{col}"] = df[col].max() - df[col].min()
    return df
