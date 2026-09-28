"""Heterogeneity of treatment effect with internal-external cross-validation.

Following the PATH statement (Kent et al., 2020), the primary HTE analysis is
a *risk/severity-based* linear interaction model rather than a black-box
learner.  Transportability of the predicted benefit is assessed by
internal-external cross-validation across trials (Steyerberg & Harrell,
2016): each trial is held out in turn, the interaction model is fitted on the
others, and in the held-out trial we regress the outcome on treatment x
predicted benefit.  A slope of 1 means the predicted benefit is calibrated in
the new trial; 0 means no transportable heterogeneity.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .engagement import _ols


def _interaction_design(df: pd.DataFrame, z: str, moderators: Sequence[str], baseline: str) -> np.ndarray:
    zc = df[z].to_numpy(float)
    cols = [np.ones(len(df)), df[baseline].to_numpy(float)]
    cols += [df[m].to_numpy(float) for m in moderators if m != baseline]
    cols.append(zc)
    cols += [zc * df[m].to_numpy(float) for m in moderators]
    return np.column_stack(cols)


def fit_interaction_model(df: pd.DataFrame, y: str = "y", z: str = "z", moderators: Sequence[str] = ("baseline",), baseline: str = "baseline") -> Dict[str, object]:
    """OLS ``y ~ baseline + moderators + z + z:moderators`` on complete cases; centred moderators."""
    d = df.dropna(subset=[y]).copy()
    centre = {m: float(d[m].mean()) for m in moderators}
    for m in moderators:
        d[m] = d[m] - centre[m]
    X = _interaction_design(d, z, moderators, baseline)
    beta, se = _ols(d[y].to_numpy(float), X)
    k_main = 2 + len([m for m in moderators if m != baseline])
    names = ["intercept", baseline] + [m for m in moderators if m != baseline] + ["z"] + [f"z:{m}" for m in moderators]
    return {"coef": dict(zip(names, beta.tolist())), "se": dict(zip(names, se.tolist())), "centre": centre, "moderators": list(moderators), "z": z, "baseline": baseline, "k_main": k_main, "n": int(len(d))}


def predict_benefit(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    """Predicted individual benefit ``z_coef + sum_m z:m * (m - centre_m)``."""
    coef = model["coef"]
    out = np.full(len(df), coef["z"], float)
    for m in model["moderators"]:
        out += coef[f"z:{m}"] * (df[m].to_numpy(float) - model["centre"][m])
    return out


def internal_external_cv(df: pd.DataFrame, trial: str = "trial", y: str = "y", z: str = "z", moderators: Sequence[str] = ("baseline",), baseline: str = "baseline") -> pd.DataFrame:
    """Hold out each trial, fit on the rest, evaluate calibration of predicted benefit in the held-out trial.

    Per held-out trial: ``cal_slope`` (coefficient on z x predicted benefit; 1 =
    calibrated), ``cal_intercept`` (z coefficient at mean predicted benefit),
    ``itt_observed``, ``benefit_mean/sd`` and the Spearman correlation between
    predicted-benefit quartile and the quartile-specific observed ITT.
    """
    rows: List[Dict[str, float]] = []
    for t in sorted(df[trial].unique()):
        train, test = df[df[trial] != t], df[df[trial] == t].dropna(subset=[y])
        model = fit_interaction_model(train, y, z, moderators, baseline)
        pb = predict_benefit(model, test)
        pbc = pb - pb.mean()
        X = np.column_stack([np.ones(len(test)), test[baseline].to_numpy(float), pbc, test[z].to_numpy(float), test[z].to_numpy(float) * pbc])
        beta, se = _ols(test[y].to_numpy(float), X)
        # quartile check
        q = pd.qcut(pb, 4, labels=False, duplicates="drop") if len(np.unique(pb)) > 4 else np.zeros(len(pb), int)
        obs_q, pred_q = [], []
        for k in np.unique(q):
            sub = test[q == k]
            if sub[z].nunique() == 2:
                b, _ = _ols(sub[y].to_numpy(float), np.column_stack([np.ones(len(sub)), sub[z].to_numpy(float), sub[baseline].to_numpy(float)]))
                obs_q.append(b[1])
                pred_q.append(pb[q == k].mean())
        rho = float(stats.spearmanr(pred_q, obs_q)[0]) if len(obs_q) >= 3 else np.nan
        rows.append(
            {
                "held_out": t,
                "n_test": int(len(test)),
                "cal_slope": float(beta[4]),
                "cal_slope_se": float(se[4]),
                "cal_intercept": float(beta[3]),
                "itt_observed": float(_ols(test[y].to_numpy(float), np.column_stack([np.ones(len(test)), test[z].to_numpy(float), test[baseline].to_numpy(float)]))[0][1]),
                "benefit_mean": float(pb.mean()),
                "benefit_sd": float(pb.std()),
                "quartile_spearman": rho,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["pooled_cal_slope"] = float(np.average(out["cal_slope"], weights=1 / np.clip(out["cal_slope_se"] ** 2, 1e-9, None)))
    return out
