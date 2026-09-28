"""Brain-age baselines: ridge regression and gradient boosting on FreeSurfer
features, with leakage-safe cross-validation.

Design choices that matter for a transportability audit
-------------------------------------------------------
* Feature scaling is fitted inside each CV fold (``Pipeline``), never on the
  full sample.
* ``cross_validated_predictions`` groups folds by *subject* so that repeat
  OASIS-3 visits of one person never straddle train and test.
* ``predict_external`` deliberately takes raw features so that harmonisation
  (:mod:`brainage_transport.harmonize`) can be applied *before* or *after* the
  model is fitted, letting you compare pipelines A (harmonise then train),
  B (train on reference, ComBat-transform target) and C (no harmonisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelKind = Literal["ridge", "gbm"]


def make_estimator(kind: ModelKind = "ridge", random_state: int = 0) -> Pipeline:
    """Return a scikit-learn pipeline for the requested baseline."""
    if kind == "ridge":
        est = RidgeCV(alphas=np.logspace(-2, 4, 25))
        return Pipeline([("scale", StandardScaler()), ("model", est)])
    if kind == "gbm":
        est = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=random_state,
        )
        return Pipeline([("scale", StandardScaler()), ("model", est)])
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass
class BrainAgeModel(BaseEstimator, RegressorMixin):
    """Thin wrapper holding a fitted baseline plus training-set metadata.

    Attributes
    ----------
    kind
        ``"ridge"`` or ``"gbm"``.
    train_age_range_
        (min, max) of the training ages: predictions outside this range are
        *extrapolations* and are flagged by :meth:`predict_external`.
    """

    kind: ModelKind = "ridge"
    random_state: int = 0
    pipeline_: Optional[Pipeline] = field(default=None, repr=False)
    feature_names_: Optional[list] = field(default=None, repr=False)
    train_age_range_: Optional[tuple] = field(default=None, repr=False)

    def fit(self, X, y) -> "BrainAgeModel":
        X = pd.DataFrame(X)
        self.feature_names_ = list(X.columns)
        self.pipeline_ = make_estimator(self.kind, self.random_state)
        self.pipeline_.fit(X.values, np.asarray(y, dtype=float))
        self.train_age_range_ = (float(np.min(y)), float(np.max(y)))
        return self

    def predict(self, X) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("model not fitted")
        X = pd.DataFrame(X)
        if self.feature_names_ is not None and list(X.columns) != self.feature_names_:
            X = X.reindex(columns=self.feature_names_)
        return self.pipeline_.predict(X.values)

    def predict_external(self, X, age: Optional[Sequence[float]] = None) -> pd.DataFrame:
        """Predict on an external cohort and flag extrapolation.

        Returns a DataFrame with ``pred``, and if ``age`` is given ``age``,
        ``delta`` (pred - age) and ``extrapolated`` (age outside training range).
        """
        pred = self.predict(X)
        out = pd.DataFrame({"pred": pred})
        if age is not None:
            age = np.asarray(age, dtype=float)
            out["age"] = age
            out["delta"] = pred - age
            lo, hi = self.train_age_range_ or (-np.inf, np.inf)
            out["extrapolated"] = (age < lo) | (age > hi)
        return out


def cross_validated_predictions(
    X,
    age: Sequence[float],
    groups: Optional[Sequence] = None,
    kind: ModelKind = "ridge",
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Out-of-fold brain-age predictions with subject-grouped folds.

    Parameters
    ----------
    X : (n, p) features
    age : chronological age per row
    groups : subject id per row (rows sharing a subject stay in one fold).
        If None, plain shuffled KFold is used.
    kind : baseline kind

    Returns
    -------
    DataFrame with columns ``age``, ``pred``, ``delta``, ``fold``.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    age = np.asarray(age, dtype=float)
    pred = np.full(age.shape, np.nan)
    fold_id = np.full(age.shape, -1)
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, age, groups=np.asarray(groups))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X, age)
    for k, (tr, te) in enumerate(splits):
        model = BrainAgeModel(kind=kind, random_state=random_state).fit(X.iloc[tr], age[tr])
        pred[te] = model.predict(X.iloc[te])
        fold_id[te] = k
    return pd.DataFrame({"age": age, "pred": pred, "delta": pred - age, "fold": fold_id})


def performance_summary(age: Sequence[float], pred: Sequence[float]) -> Dict[str, float]:
    """MAE, RMSE, Pearson r, R^2 and the raw age-delta correlation.

    The last quantity (``r_age_delta``) is the *age bias* that motivates the
    correction procedures in :mod:`brainage_transport.bias_correction`.
    """
    age = np.asarray(age, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(age) | np.isnan(pred))
    age, pred = age[ok], pred[ok]
    delta = pred - age
    ss_res = np.sum(delta ** 2)
    ss_tot = np.sum((age - age.mean()) ** 2)
    return {
        "n": int(age.size),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "r": float(np.corrcoef(age, pred)[0, 1]) if age.size > 2 else np.nan,
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "r_age_delta": float(np.corrcoef(age, delta)[0, 1]) if age.size > 2 else np.nan,
        "mean_delta": float(delta.mean()),
    }


def simulate_features(
    n: int = 400,
    n_features: int = 60,
    age_range: tuple = (20.0, 90.0),
    noise: float = 1.0,
    scanner_effects: Optional[Dict[str, float]] = None,
    scanner_labels: Optional[Sequence] = None,
    seed: int = 0,
) -> tuple:
    """Simulate FreeSurfer-like features with linear age effects.

    Half the features shrink with age (thickness / volume), half are noise.
    Optional additive scanner offsets mimic protocol differences.
    Returns ``(X, age, scanner)`` where scanner is None if not requested.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(*age_range, size=n)
    slopes = np.concatenate([rng.uniform(-0.05, -0.01, n_features // 2), np.zeros(n_features - n_features // 2)])
    X = age[:, None] * slopes[None, :] + rng.normal(0, noise, size=(n, n_features))
    scanner = None
    if scanner_effects is not None:
        labels = list(scanner_effects)
        scanner = np.array(labels)[rng.integers(0, len(labels), n)] if scanner_labels is None else np.asarray(scanner_labels)
        for lab, eff in scanner_effects.items():
            X[scanner == lab] += eff
    return X, age, scanner
