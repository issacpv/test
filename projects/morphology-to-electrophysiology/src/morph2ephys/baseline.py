"""Ridge baseline on hand-crafted morphometrics, grouped cross-validation, permutation nulls,
mouse -> human transfer metrics and within-type partial R².

All functions take numpy arrays / pandas objects so they can be reused for GNN predictions
(e.g. ``transfer_gap`` and ``within_type_partial_r2`` are model-agnostic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_ALPHAS = np.logspace(-2, 3, 12)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _make_model(alphas: Sequence[float] = DEFAULT_ALPHAS):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray(alphas)))


@dataclass
class CVResult:
    target: str
    r2: float
    spearman: float
    rmse: float
    oof_pred: np.ndarray
    n: int


def ridge_grouped_cv(X: np.ndarray, y: np.ndarray, groups: Sequence, n_splits: int = 5,
                     alphas: Sequence[float] = DEFAULT_ALPHAS, target: str = "y") -> CVResult:
    """Out-of-fold ridge predictions with GroupKFold (alpha chosen by inner RidgeCV on train folds only)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, groups = X[ok], y[ok], groups[ok]
    n_splits = min(n_splits, len(np.unique(groups)))
    oof = np.full(y.shape, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = _make_model(alphas).fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    rho = stats.spearmanr(y, oof).correlation if y.size > 2 else float("nan")
    return CVResult(target=target, r2=r2_score(y, oof), spearman=float(rho),
                    rmse=float(np.sqrt(np.mean((y - oof) ** 2))), oof_pred=oof, n=int(y.size))


def multi_target_baseline(X: pd.DataFrame, Y: pd.DataFrame, groups: Sequence, n_splits: int = 5) -> pd.DataFrame:
    """Ridge CV for every column of ``Y``; returns a metrics table."""
    rows = []
    for col in Y.columns:
        res = ridge_grouped_cv(X.to_numpy(), Y[col].to_numpy(), groups, n_splits=n_splits, target=col)
        rows.append({"target": col, "r2": res.r2, "spearman": res.spearman, "rmse": res.rmse, "n": res.n})
    return pd.DataFrame(rows)


def permutation_null_r2(X: np.ndarray, y: np.ndarray, groups: Sequence, n_perm: int = 200, seed: int = 0,
                        n_splits: int = 5) -> Tuple[float, np.ndarray, float]:
    """Observed OOF R², null R² distribution from label permutation, and the permutation p-value."""
    rng = np.random.default_rng(seed)
    obs = ridge_grouped_cv(X, y, groups, n_splits=n_splits).r2
    null = np.array([ridge_grouped_cv(X, rng.permutation(y), groups, n_splits=n_splits).r2 for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, null, p


# ---------------------------------------------------------------------- transfer
@dataclass
class TransferResult:
    r2_within_source: float
    r2_within_target: float
    r2_transfer: float
    spearman_transfer: float
    transfer_gap: float
    n_source: int
    n_target: int


def transfer_evaluation(X_src: np.ndarray, y_src: np.ndarray, groups_src: Sequence, X_tgt: np.ndarray,
                        y_tgt: np.ndarray, groups_tgt: Sequence, alphas: Sequence[float] = DEFAULT_ALPHAS,
                        standardize_targets_per_species: bool = True) -> TransferResult:
    """Train on the source species, test zero-shot on the target species; compare with within-target CV.

    Targets are z-scored within species when ``standardize_targets_per_species`` so that a pure
    offset between species (e.g. larger human R_in) does not count against transfer of the *mapping*.
    ``transfer_gap = r2_within_target - r2_transfer`` (0 = perfect transfer).
    """
    X_src, X_tgt = np.asarray(X_src, float), np.asarray(X_tgt, float)
    y_src, y_tgt = np.asarray(y_src, float), np.asarray(y_tgt, float)
    ok_s = np.isfinite(y_src) & np.all(np.isfinite(X_src), axis=1)
    ok_t = np.isfinite(y_tgt) & np.all(np.isfinite(X_tgt), axis=1)
    X_src, y_src, groups_src = X_src[ok_s], y_src[ok_s], np.asarray(groups_src)[ok_s]
    X_tgt, y_tgt, groups_tgt = X_tgt[ok_t], y_tgt[ok_t], np.asarray(groups_tgt)[ok_t]
    if standardize_targets_per_species:
        y_src = (y_src - y_src.mean()) / (y_src.std() + 1e-12)
        y_tgt = (y_tgt - y_tgt.mean()) / (y_tgt.std() + 1e-12)
    within_src = ridge_grouped_cv(X_src, y_src, groups_src, alphas=alphas).r2
    within_tgt = ridge_grouped_cv(X_tgt, y_tgt, groups_tgt, alphas=alphas).r2
    model = _make_model(alphas).fit(X_src, y_src)
    pred = model.predict(X_tgt)
    r2_tr = r2_score(y_tgt, pred)
    rho = float(stats.spearmanr(y_tgt, pred).correlation)
    return TransferResult(r2_within_source=within_src, r2_within_target=within_tgt, r2_transfer=r2_tr,
                          spearman_transfer=rho, transfer_gap=within_tgt - r2_tr, n_source=int(y_src.size),
                          n_target=int(y_tgt.size))


def few_shot_curve(X_src, y_src, X_tgt, y_tgt, groups_tgt, shots: Sequence[int] = (10, 25, 50),
                   n_repeats: int = 20, seed: int = 0, alphas: Sequence[float] = DEFAULT_ALPHAS) -> pd.DataFrame:
    """Fine-tune-free few-shot analogue for ridge: fit on source + k target cells, test on the rest."""
    rng = np.random.default_rng(seed)
    X_src, X_tgt = np.asarray(X_src, float), np.asarray(X_tgt, float)
    y_src, y_tgt = np.asarray(y_src, float), np.asarray(y_tgt, float)
    groups_tgt = np.asarray(groups_tgt)
    rows = []
    for k in shots:
        for rep in range(n_repeats):
            uniq = np.unique(groups_tgt)
            rng.shuffle(uniq)
            train_mask = np.zeros(y_tgt.size, dtype=bool)
            for g in uniq:  # add whole donors until k cells reached (no donor leakage)
                train_mask |= groups_tgt == g
                if train_mask.sum() >= k:
                    break
            test_mask = ~train_mask
            if test_mask.sum() < 5:
                continue
            X_tr = np.vstack([X_src, X_tgt[train_mask]])
            y_tr = np.concatenate([y_src, y_tgt[train_mask]])
            w = np.concatenate([np.ones(y_src.size), np.full(int(train_mask.sum()), 3.0)])  # upweight target
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X_tr, y_tr, ridge__sample_weight=w)
            rows.append({"shots": k, "repeat": rep, "r2": r2_score(y_tgt[test_mask], model.predict(X_tgt[test_mask]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- within-type information
def within_type_partial_r2(X: np.ndarray, y: np.ndarray, types: Sequence, groups: Sequence,
                           n_splits: int = 5, alphas: Sequence[float] = DEFAULT_ALPHAS) -> Dict[str, float]:
    """Does morphology explain ephys *within* cell types?

    Residualises ``y`` on type means (the "type-only" null model) and regresses the residual on
    morphometrics with grouped CV. Also returns the type-only R² and the raw morphology R².
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    types = pd.Series(np.asarray(types))
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, types, groups = X[ok], y[ok], types[ok].reset_index(drop=True), np.asarray(groups)[ok]
    type_mean = types.map(pd.Series(y).groupby(types).mean()).to_numpy()
    r2_type_only = r2_score(y, type_mean)
    resid = y - type_mean
    r2_within = ridge_grouped_cv(X, resid, groups, n_splits=n_splits, alphas=alphas).r2
    r2_raw = ridge_grouped_cv(X, y, groups, n_splits=n_splits, alphas=alphas).r2
    return {"r2_type_only": r2_type_only, "r2_morphology_raw": r2_raw, "partial_r2_within_type": r2_within,
            "n_types": int(types.nunique()), "n": int(y.size)}


def residual_type_anova(residuals: np.ndarray, types: Sequence) -> Dict[str, float]:
    """One-way ANOVA of model residuals across transcriptomic/coarse types (are residuals type-structured?)."""
    df = pd.DataFrame({"r": np.asarray(residuals, float), "t": np.asarray(types)}).dropna()
    groups = [g["r"].to_numpy() for _, g in df.groupby("t") if len(g) >= 3]
    if len(groups) < 2:
        return {"F": float("nan"), "p": float("nan"), "eta2": float("nan")}
    F, p = stats.f_oneway(*groups)
    grand = df["r"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(np.sum((df["r"] - grand) ** 2))
    return {"F": float(F), "p": float(p), "eta2": float(ss_between / ss_total) if ss_total > 0 else float("nan")}


__all__ = ["r2_score", "CVResult", "ridge_grouped_cv", "multi_target_baseline", "permutation_null_r2",
           "TransferResult", "transfer_evaluation", "few_shot_curve", "within_type_partial_r2", "residual_type_anova"]
