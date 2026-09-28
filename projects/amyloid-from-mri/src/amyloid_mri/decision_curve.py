"""Decision-curve analysis and a PET-scans-avoided cost model for trial pre-screening.

Net benefit follows Vickers & Elkin (2006, Medical Decision Making):
``NB(t) = TP/n − FP/n · t/(1−t)``. "Treat all" means sending everybody to PET; "treat none" is
NB = 0. For pre-screening the relevant policy is: send to PET only those with predicted
probability ≥ threshold; the threshold is chosen on training data to reach a target sensitivity
(so that few true amyloid-positives are lost) and then held fixed.

Literature comparators (e.g. plasma p-tau217 at published sensitivity/specificity) are evaluated
analytically at the cohort prevalence so that all arms share the same population.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Net benefit of the model, treat-all and treat-none across probability thresholds."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    n = len(y)
    prev = y.mean()
    thresholds = np.arange(0.05, 0.80, 0.01) if thresholds is None else np.asarray(thresholds, float)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)) / n
        fp = np.sum(pred & (y == 0)) / n
        w = t / (1 - t)
        rows.append({"threshold": t, "nb_model": tp - fp * w, "nb_all": prev - (1 - prev) * w,
                     "nb_none": 0.0, "referred_fraction": pred.mean(),
                     "sensitivity": tp / prev if prev > 0 else np.nan,
                     "specificity": 1 - fp / (1 - prev) if prev < 1 else np.nan})
    return pd.DataFrame(rows)


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, sensitivity: float = 0.9) -> float:
    """Largest threshold whose sensitivity is at least ``sensitivity`` (chosen on training/OOF data)."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        raise ValueError("no positives")
    k = int(np.floor((1 - sensitivity) * len(pos)))  # number of positives allowed below threshold
    return float(pos[min(k, len(pos) - 1)])


def screening_policy(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Operating characteristics of "PET only if p ≥ threshold", per 1,000 screened."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = p >= threshold
    n = len(y)
    tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
    fn, tn = np.sum(~pred & (y == 1)), np.sum(~pred & (y == 0))
    n_pos = tp + fn
    return _policy_dict(n=n, tp=tp / n, fp=fp / n, fn=fn / n, tn=tn / n, prevalence=n_pos / n,
                        threshold=threshold)


def policy_from_sens_spec(prevalence: float, sensitivity: float, specificity: float) -> dict[str, float]:
    """Analytical operating characteristics for a comparator with known sensitivity/specificity."""
    tp = prevalence * sensitivity
    fn = prevalence * (1 - sensitivity)
    tn = (1 - prevalence) * specificity
    fp = (1 - prevalence) * (1 - specificity)
    return _policy_dict(n=np.nan, tp=tp, fp=fp, fn=fn, tn=tn, prevalence=prevalence, threshold=np.nan)


def _policy_dict(n: float, tp: float, fp: float, fn: float, tn: float, prevalence: float,
                 threshold: float) -> dict[str, float]:
    referred = tp + fp
    return {
        "n": n, "threshold": threshold, "prevalence": prevalence,
        "sensitivity": tp / prevalence if prevalence > 0 else np.nan,
        "specificity": tn / (1 - prevalence) if prevalence < 1 else np.nan,
        "ppv": tp / referred if referred > 0 else np.nan,
        "pet_per_1000": 1000 * referred,
        "pet_avoided_fraction": 1 - referred,
        "positives_enrolled_per_1000": 1000 * tp,
        "positives_missed_per_1000": 1000 * fn,
    }


def screening_cost(policy: Mapping[str, float], cost_pet: float, cost_prescreen: float = 0.0,
                   cost_missed_positive: float = 0.0) -> dict[str, float]:
    """Cost per 1,000 screened and per enrolled amyloid-positive participant.

    Parameters
    ----------
    cost_pet
        Cost of one amyloid PET scan (USD; ~3,000-5,000 in US trials).
    cost_prescreen
        Marginal cost of the pre-screen per person (0 if the MRI is acquired anyway; assay cost for a
        blood test; full MRI cost if it has to be acquired for screening).
    cost_missed_positive
        Optional penalty for each true positive not sent to PET (opportunity cost of a lost eligible
        participant); 0 by default so results stay transparent.
    """
    pet_cost = policy["pet_per_1000"] * cost_pet
    pre_cost = 1000 * cost_prescreen
    missed_cost = policy["positives_missed_per_1000"] * cost_missed_positive
    total = pet_cost + pre_cost + missed_cost
    enrolled = policy["positives_enrolled_per_1000"]
    return {"cost_per_1000": total, "pet_cost_per_1000": pet_cost, "prescreen_cost_per_1000": pre_cost,
            "cost_per_enrolled_positive": total / enrolled if enrolled > 0 else np.nan,
            "positives_enrolled_per_1000": enrolled}


def compare_policies(y: np.ndarray, predictions: Mapping[str, np.ndarray], sensitivity_target: float,
                     cost_pet: float, prescreen_costs: Mapping[str, float],
                     comparators: Optional[Mapping[str, tuple[float, float, float]]] = None,
                     thresholds: Optional[Mapping[str, float]] = None) -> pd.DataFrame:
    """Tabulate no-prescreen, model-based and literature-comparator policies at a common prevalence.

    Parameters
    ----------
    predictions
        ``{name: probabilities}`` for each model (out-of-fold or external predictions).
    prescreen_costs
        ``{name: marginal cost per person}`` for every model and comparator; missing names cost 0.
    comparators
        ``{name: (sensitivity, specificity, cost)}`` from the literature, e.g. plasma p-tau217.
    thresholds
        Optional pre-fixed thresholds per model (e.g. chosen on OASIS-3 and applied externally);
        otherwise thresholds are chosen on ``y``/``p`` for the target sensitivity.
    """
    y = np.asarray(y, int)
    prev = y.mean()
    rows = []
    base = policy_from_sens_spec(prev, 1.0, 0.0)
    rows.append({"policy": "no_prescreen", **base, **screening_cost(base, cost_pet, 0.0)})
    for name, p in predictions.items():
        t = thresholds[name] if thresholds and name in thresholds else threshold_for_sensitivity(y, p, sensitivity_target)
        pol = screening_policy(y, p, t)
        rows.append({"policy": name, **pol, **screening_cost(pol, cost_pet, prescreen_costs.get(name, 0.0))})
    for name, (sens, spec, cost) in (comparators or {}).items():
        pol = policy_from_sens_spec(prev, sens, spec)
        rows.append({"policy": name, **pol, **screening_cost(pol, cost_pet, cost)})
    df = pd.DataFrame(rows)
    df["pet_cost_saved_vs_no_prescreen"] = df["pet_cost_per_1000"].iloc[0] - df["pet_cost_per_1000"]
    df["net_saving_per_1000"] = df["cost_per_1000"].iloc[0] - df["cost_per_1000"]
    return df
