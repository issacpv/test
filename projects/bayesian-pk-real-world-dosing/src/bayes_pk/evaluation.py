"""Forecast-error metrics, target attainment, timing-perturbation experiment and renal-function equations.

Metrics follow the external-evaluation conventions used for MIPD models: relative bias (rBias, %), relative
root-mean-square error (rRMSE, %) and mean prediction error (MPE, mg/L), with cluster-bootstrap confidence
intervals over patients (a patient may contribute several courses/levels).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .bayes_forecast import map_estimate
from .pk_models import ModelSpec, as_dose_frame, auc_interval


# ------------------------------------------------------------------------------------------- error metrics
def prediction_errors(observed: Sequence[float], predicted: Sequence[float]) -> Dict[str, float]:
    """rBias (%), rRMSE (%), MPE (mg/L), RMSE (mg/L) and n."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape or obs.size == 0:
        raise ValueError("observed and predicted must be non-empty and of equal length")
    rel = (pred - obs) / obs
    return {
        "n": int(obs.size),
        "rbias_pct": float(100.0 * rel.mean()),
        "rrmse_pct": float(100.0 * np.sqrt(np.mean(rel**2))),
        "mpe": float(np.mean(pred - obs)),
        "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
    }


def cluster_bootstrap(
    observed: Sequence[float],
    predicted: Sequence[float],
    cluster: Sequence,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Percentile bootstrap CIs of :func:`prediction_errors`, resampling whole clusters (patients)."""
    rng = np.random.default_rng() if rng is None else rng
    df = pd.DataFrame({"obs": observed, "pred": predicted, "c": cluster})
    groups = {c: g for c, g in df.groupby("c")}
    keys = list(groups)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        boot = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        stats.append(prediction_errors(boot["obs"], boot["pred"]))
    s = pd.DataFrame(stats)
    point = prediction_errors(df["obs"], df["pred"])
    out = pd.DataFrame(
        {
            "estimate": pd.Series(point),
            "ci_lo": s.quantile(alpha / 2),
            "ci_hi": s.quantile(1 - alpha / 2),
        }
    )
    return out.drop(index="n", errors="ignore")


def target_attainment(auc24: Sequence[float], lo: float = 400.0, hi: float = 600.0) -> Dict[str, float]:
    """Fraction of courses with AUC24 in ``[lo, hi]`` mg*h/L, below and above."""
    a = np.asarray(auc24, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "in_range": float("nan"), "below": float("nan"), "above": float("nan")}
    return {"n": int(a.size), "in_range": float(np.mean((a >= lo) & (a <= hi))), "below": float(np.mean(a < lo)), "above": float(np.mean(a > hi))}


# ----------------------------------------------------------------------------- timing perturbation experiment
def timing_perturbation_experiment(
    spec: ModelSpec,
    covariates: Mapping[str, float],
    doses: pd.DataFrame,
    levels: pd.DataFrame,
    sd_minutes: float = 30.0,
    n_rep: int = 100,
    n_levels_used: Optional[int] = None,
    auc_window_h: float = 24.0,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Re-estimate the individual after jittering every recorded dose start by ``N(0, sd_minutes)``.

    Durations are preserved (start and end shift together). The AUC over the last ``auc_window_h`` hours of the
    recorded course is compared with the unperturbed estimate. Returns one row per replicate with ``auc24``,
    ``rel_change`` and the fitted clearance; the first row (``rep = -1``) is the unperturbed reference.
    """
    rng = np.random.default_rng() if rng is None else rng
    d0 = as_dose_frame(doses)
    lv = levels.sort_values("time_h").reset_index(drop=True)
    k = len(lv) if n_levels_used is None else min(n_levels_used, len(lv))
    lt, lval = lv["time_h"].to_numpy()[:k], lv["value"].to_numpy()[:k]
    t1 = float(d0["end"].max())
    t0 = max(0.0, t1 - auc_window_h)

    def fit_and_auc(d: pd.DataFrame) -> tuple[float, float]:
        r = map_estimate(spec, covariates, d, lt, lval)
        return auc_interval(spec, r.params, d, t0, t1, n_grid=1201), float(r.params["CL"])

    auc_ref, cl_ref = fit_and_auc(d0)
    rows = [{"rep": -1, "auc24": auc_ref, "rel_change": 0.0, "CL": cl_ref}]
    for rep in range(n_rep):
        shift = rng.normal(0.0, sd_minutes / 60.0, size=len(d0))
        d = d0.copy()
        d["start"] = np.clip(d["start"] + shift, 0.0, None)
        d["end"] = d["end"] + (d["start"] - d0["start"])
        auc, cl = fit_and_auc(d)
        rows.append({"rep": rep, "auc24": auc, "rel_change": (auc - auc_ref) / auc_ref, "CL": cl})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ renal function
def ckd_epi_2021(scr_mg_dl: float, age_years: float, female: bool) -> float:
    """CKD-EPI 2021 race-free creatinine eGFR (mL/min/1.73 m^2), Inker et al., 2021, N Engl J Med.

    eGFR = 142 * min(Scr/k, 1)^a * max(Scr/k, 1)^-1.200 * 0.9938^age * 1.012 [if female],
    with k = 0.7 (female) / 0.9 (male) and a = -0.241 (female) / -0.302 (male).
    """
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    r = scr_mg_dl / k
    egfr = 142.0 * min(r, 1.0) ** a * max(r, 1.0) ** (-1.200) * 0.9938**age_years
    return egfr * (1.012 if female else 1.0)


def cockcroft_gault(scr_mg_dl: float, age_years: float, female: bool, weight_kg: float) -> float:
    """Cockcroft-Gault creatinine clearance (mL/min): (140 - age) * weight / (72 * Scr), x0.85 if female."""
    crcl = (140.0 - age_years) * weight_kg / (72.0 * scr_mg_dl)
    return crcl * (0.85 if female else 1.0)


def kdigo_stage_from_creatinine(baseline_scr: float, scr: float, rise_within_48h: float = 0.0) -> int:
    """KDIGO AKI stage from creatinine only (urine-output criteria handled separately in the cohort code).

    Stage 3: Scr >= 3x baseline or Scr >= 4.0 mg/dL; stage 2: >= 2x; stage 1: >= 1.5x or a rise >= 0.3 mg/dL
    within 48 h (pass the 48-h rise as ``rise_within_48h``). Returns 0 when no criterion is met.
    """
    ratio = scr / baseline_scr if baseline_scr > 0 else float("nan")
    if ratio >= 3.0 or scr >= 4.0:
        return 3
    if ratio >= 2.0:
        return 2
    if ratio >= 1.5 or rise_within_48h >= 0.3:
        return 1
    return 0
