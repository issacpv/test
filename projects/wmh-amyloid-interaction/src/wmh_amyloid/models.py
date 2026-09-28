"""Interaction models.

* :func:`fit_interaction_mixed_model` — linear mixed model with random intercept and slope,
  ``outcome ~ time * amyloid * wmh + covariates``; the three-way term is the multiplicative
  interaction on the *rate* of change.
* :func:`person_period` and :func:`discrete_time_hazard` — logistic discrete-time hazard for
  CDR progression with time-varying covariates.
* :func:`additive_interaction` — RERI, attributable proportion and synergy index for two binary
  exposures from a log-linear (logistic/hazard) model's coefficients (Rothman; Knol and
  VanderWeele, 2012); :func:`bootstrap_additive_interaction` gives subject-bootstrap CIs.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_interaction_mixed_model(
    df: pd.DataFrame,
    outcome: str,
    time: str = "time",
    amyloid: str = "amyloid_pos",
    wmh: str = "wmh_log",
    covariates: Sequence[str] = (),
    subject: str = "subject",
    random_slope: bool = True,
):
    """Fit ``outcome ~ time * amyloid * wmh + covariates`` with subject random effects.

    Returns ``(result, table)`` where ``table`` holds the estimates of the interaction terms
    (``time:amyloid``, ``time:wmh``, ``time:amyloid:wmh``) with SE, z and p.
    """
    cov = " + ".join(covariates)
    formula = f"{outcome} ~ {time} * {amyloid} * {wmh}" + (f" + {cov}" if cov else "")
    d = df.dropna(subset=[outcome, time, amyloid, wmh, *covariates, subject])
    model = smf.mixedlm(formula, d, groups=d[subject], re_formula=f"~{time}" if random_slope else "1")
    result = model.fit(reml=True, method=["lbfgs"])
    keys = [f"{time}:{amyloid}", f"{time}:{wmh}", f"{time}:{amyloid}:{wmh}"]
    rows = []
    for k in keys:
        if k in result.params.index:
            rows.append({"term": k, "estimate": result.params[k], "se": result.bse[k], "z": result.tvalues[k], "p": result.pvalues[k]})
    return result, pd.DataFrame(rows)


def person_period(
    df: pd.DataFrame,
    subject: str = "subject",
    time: str = "time",
    event_time: str = "event_time",
    event: str = "event",
    period_length: float = 1.0,
) -> pd.DataFrame:
    """Expand a longitudinal table to person-periods for a discrete-time hazard model.

    Each subject contributes one row per period until the event/censoring period; time-varying
    covariates are carried forward from the most recent observation at or before the period
    start. Requires per-subject ``event_time`` (years) and ``event`` (0/1) columns (constant
    within subject).
    """
    rows = []
    for s, g in df.sort_values(time).groupby(subject):
        t_end = float(g[event_time].iloc[0])
        ev = int(g[event].iloc[0])
        n_periods = int(np.ceil(max(t_end, 1e-9) / period_length))
        for k in range(n_periods):
            start = k * period_length
            obs = g[g[time] <= start + 1e-9]
            src = obs.iloc[-1] if len(obs) else g.iloc[0]
            row = src.to_dict()
            row.update({"period": k, "period_start": start, "y": int(ev == 1 and k == n_periods - 1)})
            rows.append(row)
    return pd.DataFrame(rows)


def discrete_time_hazard(pp: pd.DataFrame, formula_rhs: str, cluster: str = "subject"):
    """Logistic discrete-time hazard ``y ~ C(period) + <formula_rhs>`` with cluster-robust SEs."""
    model = smf.logit(f"y ~ C(period) + {formula_rhs}", pp)
    return model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": pp[cluster]}, maxiter=200)


def additive_interaction(b_a: float, b_w: float, b_aw: float) -> Dict[str, float]:
    """RERI, attributable proportion and synergy index from log-scale coefficients.

    ``b_a`` and ``b_w`` are the main effects of the two binary exposures and ``b_aw`` their
    product term in a logistic / log-linear model, so that ``RR11 = exp(b_a + b_w + b_aw)``.
    """
    rr10, rr01, rr11 = np.exp(b_a), np.exp(b_w), np.exp(b_a + b_w + b_aw)
    reri = rr11 - rr10 - rr01 + 1
    ap = reri / rr11
    denom = (rr10 - 1) + (rr01 - 1)
    s = (rr11 - 1) / denom if denom != 0 else np.nan
    return {"RR10": float(rr10), "RR01": float(rr01), "RR11": float(rr11), "RERI": float(reri), "AP": float(ap), "S": float(s), "multiplicative_ratio": float(np.exp(b_aw))}


def bootstrap_additive_interaction(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], Tuple[float, float, float]],
    subject: str = "subject",
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Subject-level bootstrap of :func:`additive_interaction`.

    ``fit_fn`` takes a (resampled) data frame and returns ``(b_a, b_w, b_aw)``; resampling is by
    subject with replacement (subjects get new ids so that person-periods stay grouped).
    """
    rng = np.random.default_rng(seed)
    subjects = df[subject].unique()
    groups = {s: g for s, g in df.groupby(subject)}
    draws = []
    for b in range(n_boot):
        pick = rng.choice(subjects, len(subjects), replace=True)
        parts = []
        for i, s in enumerate(pick):
            g = groups[s].copy()
            g[subject] = f"{s}_b{i}"
            parts.append(g)
        try:
            ba, bw, baw = fit_fn(pd.concat(parts, ignore_index=True))
            draws.append(additive_interaction(ba, bw, baw))
        except Exception:  # noqa: BLE001 - non-convergence in a resample
            continue
    out = pd.DataFrame(draws)
    summary = out.quantile([0.025, 0.5, 0.975]).T
    summary.columns = ["ci_low", "median", "ci_high"]
    summary["n_success"] = len(out)
    return summary
