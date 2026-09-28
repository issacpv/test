"""Glucose-forecasting baselines with optional exogenous inputs.

- ``PersistenceModel``: predict the last observed glucose (the floor).
- ``ARRidgeModel``: ridge regression on lagged glucose (+ optional IOB/carbs),
  a strong, transparent baseline that most deep models must beat.

A small LSTM/temporal-CNN lives behind an optional torch import in
``_deep_models`` for the full study; these two run CPU-only and anchor the
benchmark.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class PersistenceModel:
    """Predict glucose(t+h) = glucose(t)."""

    def fit(self, X: np.ndarray, y: np.ndarray, X_exo=None) -> "PersistenceModel":
        return self

    def predict(self, X: np.ndarray, X_exo=None) -> np.ndarray:
        return np.asarray(X, float)[:, -1]


class ARRidgeModel:
    """Ridge regression on lagged glucose plus summarised exogenous inputs."""

    def __init__(self, alpha: float = 10.0, use_exo: bool = True) -> None:
        self.alpha = alpha
        self.use_exo = use_exo
        self._model = Ridge(alpha=alpha)

    def _features(self, X: np.ndarray, X_exo) -> np.ndarray:
        X = np.asarray(X, float)
        feats = [X]  # lagged glucose
        # trend and volatility features
        feats.append(np.diff(X, axis=1)[:, -3:])
        feats.append(X.std(axis=1, keepdims=True))
        if self.use_exo and X_exo is not None:
            exo = np.asarray(X_exo, float)
            # last IOB and total carbs over the window
            feats.append(exo[:, -1, :])
            feats.append(exo[:, :, 1].sum(axis=1, keepdims=True))
        return np.hstack(feats)

    def fit(self, X: np.ndarray, y: np.ndarray, X_exo=None) -> "ARRidgeModel":
        self._model.fit(self._features(X, X_exo), np.asarray(y, float))
        return self

    def predict(self, X: np.ndarray, X_exo=None) -> np.ndarray:
        return self._model.predict(self._features(X, X_exo))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def lag_corrected_rmse(y_true: np.ndarray, y_pred: np.ndarray, last_value: np.ndarray) -> float:
    """RMSE improvement over persistence (positive = better than last-value)."""
    return rmse(y_true, last_value) - rmse(y_true, y_pred)
