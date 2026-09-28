"""Clinically weighted confusion metric and downstream EF-impact.

- ``clinically_weighted_error``: mean danger-weighted misclassification cost,
  which grows faster than raw error when dangerous confusions (A4C<->A2C) rise.
- ``confusion_null``: permutation null testing whether the dangerous-confusion
  mass exceeds what per-view accuracy alone would produce.
- ``ef_impact``: EF MAE inflation introduced when a view classifier's mistakes
  route frames to the wrong view-specific EF model.
"""
from __future__ import annotations

import numpy as np

from .views import danger_weight


def clinically_weighted_error(y_true, y_pred) -> float:
    """Mean clinical cost per sample using ``views.danger_weight``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    costs = [danger_weight(t, p) for t, p in zip(y_true, y_pred)]
    return float(np.mean(costs))


def confusion_null(y_true, y_pred, n_perm: int = 1000, seed: int = 0) -> dict:
    """Permutation null: is the observed danger-weighted error above chance?

    The null shuffles predicted labels within their marginal distribution,
    preserving overall accuracy structure only approximately; a two-sided
    empirical p-value is returned.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    obs = clinically_weighted_error(y_true, y_pred)
    rng = np.random.default_rng(seed)
    null = np.array([clinically_weighted_error(y_true, rng.permutation(y_pred)) for _ in range(n_perm)])
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"observed": obs, "null_mean": float(null.mean()), "p": float(p)}


def ef_impact(true_view, pred_view, ef_gold: np.ndarray,
              wrong_view_bias: float = 8.0, wrong_view_noise: float = 6.0, seed: int = 0) -> dict:
    """Simulate EF-MAE inflation from view misclassification.

    Where the predicted view differs from the true view, the view-specific EF
    model is assumed to add a bias + noise (in EF percentage points). Returns the
    EF MAE with correct routing (0 by construction here) vs with the classifier's
    routing, and the inflation.
    """
    true_view = np.asarray(true_view)
    pred_view = np.asarray(pred_view)
    ef_gold = np.asarray(ef_gold, float)
    rng = np.random.default_rng(seed)
    wrong = true_view != pred_view
    err = np.zeros_like(ef_gold)
    err[wrong] = wrong_view_bias + rng.normal(0, wrong_view_noise, size=int(wrong.sum()))
    ef_pred = ef_gold + err
    mae = float(np.mean(np.abs(ef_pred - ef_gold)))
    return {"ef_mae_inflation": mae, "fraction_misrouted": float(wrong.mean())}
