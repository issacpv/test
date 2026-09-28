"""Control-referenced z-scores with age / sex / ICV regression.

The reference model is fitted on amyloid-negative cognitively normal (A-CN) *baseline* sessions only
and then applied to every session (including follow-ups), so that longitudinal evaluation never
touches the fitted parameters. Signs are flipped so that a higher z means more abnormal for every
feature (volumes and thickness decrease with disease, ventricles and WMH increase).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

POSITIVE_DIRECTION_KEYWORDS: tuple[str, ...] = ("ventricle", "vent", "hypointens", "wmh", "lesion", "csf")


def abnormal_is_positive(feature: str) -> bool:
    """True for features that *increase* with disease (ventricles, WMH), False otherwise."""
    f = feature.lower()
    return any(k in f for k in POSITIVE_DIRECTION_KEYWORDS)


class ControlZScorer:
    """Fit ``feature ~ 1 + covariates`` on controls, z-score residuals, orient so that abnormal > 0.

    Parameters
    ----------
    features : names of the columns to z-score.
    covariates : columns regressed out (default age, sex_male, icv).
    directions : optional ``{feature: +1/-1}``; +1 means the raw value increases with disease.
        Defaults from :func:`abnormal_is_positive`.
    """

    def __init__(self, features: Sequence[str], covariates: Sequence[str] = ("age", "sex_male", "icv"),
                 directions: Optional[dict[str, int]] = None):
        self.features = list(features)
        self.covariates = list(covariates)
        self.directions = {f: (directions or {}).get(f, 1 if abnormal_is_positive(f) else -1) for f in self.features}

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cols = [pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in self.covariates]
        return np.column_stack([np.ones(len(df))] + cols)

    def fit(self, controls: pd.DataFrame) -> "ControlZScorer":
        X = self._design(controls)
        Y = controls[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.beta_ = np.full((X.shape[1], len(self.features)), np.nan)
        self.sd_ = np.full(len(self.features), np.nan)
        self.n_controls_ = np.zeros(len(self.features), int)
        for j in range(len(self.features)):
            ok = np.isfinite(X).all(1) & np.isfinite(Y[:, j])
            if ok.sum() <= X.shape[1] + 2:
                continue
            b, *_ = np.linalg.lstsq(X[ok], Y[ok, j], rcond=None)
            resid = Y[ok, j] - X[ok] @ b
            self.beta_[:, j] = b
            self.sd_[j] = resid.std(ddof=X.shape[1])
            self.n_controls_[j] = ok.sum()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._design(df)
        Y = df[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        Z = (Y - X @ self.beta_) / self.sd_
        signs = np.array([self.directions[f] for f in self.features], float)
        return pd.DataFrame(Z * signs, columns=[f"z_{f}" for f in self.features], index=df.index)

    def fit_transform(self, controls: pd.DataFrame, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.fit(controls).transform(controls if df is None else df)


def average_hemispheres(df: pd.DataFrame, pairs: Iterable[tuple[str, str, str]]) -> pd.DataFrame:
    """Average left/right columns: ``pairs`` = [(left_col, right_col, new_name), ...]."""
    out = df.copy()
    for l, r, name in pairs:
        out[name] = (pd.to_numeric(out[l], errors="coerce") + pd.to_numeric(out[r], errors="coerce")) / 2
    return out


def control_calibration_check(z_controls: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """Fraction of held-out controls with z > threshold per feature (should be ≈ 2.3% for z > 2)."""
    return (z_controls > threshold).mean()
