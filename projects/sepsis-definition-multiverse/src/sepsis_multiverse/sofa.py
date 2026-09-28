"""Hourly SOFA scoring from a long or wide hourly concept table.

Input convention: a wide DataFrame indexed by (stay_id, hour) with columns among
``pao2, fio2, vent, platelets, bilirubin, map, norepinephrine, epinephrine, dopamine,
dobutamine, gcs, creatinine, urine_output``. Vasopressor doses in ug/kg/min; urine output
in mL per hour; FiO2 as a fraction (0.21-1.0); ``vent`` as 0/1.

Missing-data handling is part of the multiverse: ``carry_forward_h`` per concept and
``assume_normal`` (score 0 when a component was never measured) vs ``nan`` (stay excluded
from scoring at that hour).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

COMPONENTS = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]

DEFAULT_CARRY_FORWARD_H: Dict[str, int] = {
    "pao2": 24, "fio2": 24, "vent": 24, "platelets": 24, "bilirubin": 24, "map": 1,
    "norepinephrine": 1, "epinephrine": 1, "dopamine": 1, "dobutamine": 1, "gcs": 24,
    "creatinine": 24, "urine_output": 0,
}


@dataclass
class SofaConfig:
    carry_forward_h: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CARRY_FORWARD_H))
    assume_normal: bool = True          # never-measured component scores 0 (else NaN)
    urine_window_h: int = 24            # window for the urine-output criterion
    urine_extrapolate: bool = False     # allow shorter windows scaled to 24 h
    gcs_sedated_value: Optional[float] = None  # if set, GCS while sedated is replaced by this value


def _carry_forward(s: pd.Series, hours: int) -> pd.Series:
    """Forward-fill within a stay for at most ``hours`` rows (assumes an hourly grid)."""
    if hours <= 0:
        return s
    return s.ffill(limit=hours)


def respiration_score(pao2: pd.Series, fio2: pd.Series, vent: pd.Series) -> pd.Series:
    ratio = pao2 / fio2.clip(lower=0.21)
    v = vent.fillna(0).astype(bool)
    score = pd.Series(np.nan, index=ratio.index)
    score[ratio.notna()] = 0
    score[ratio < 400] = 1
    score[ratio < 300] = 2
    score[(ratio < 200) & v] = 3
    score[(ratio < 100) & v] = 4
    return score


def coagulation_score(platelets: pd.Series) -> pd.Series:
    return pd.cut(platelets, [-np.inf, 20, 50, 100, 150, np.inf], labels=[4, 3, 2, 1, 0], right=False).astype(float)


def liver_score(bilirubin: pd.Series) -> pd.Series:
    return pd.cut(bilirubin, [-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf], labels=[0, 1, 2, 3, 4], right=False).astype(float)


def cardiovascular_score(map_: pd.Series, norepi: pd.Series, epi: pd.Series, dopa: pd.Series, dobu: pd.Series) -> pd.Series:
    ne, ep, dp, db = (x.fillna(0) for x in (norepi, epi, dopa, dobu))
    score = pd.Series(np.nan, index=map_.index)
    score[map_.notna()] = 0
    score[map_ < 70] = 1
    score[(dp > 0) | (db > 0)] = 2
    score[(dp > 5) | ((ne > 0) & (ne <= 0.1)) | ((ep > 0) & (ep <= 0.1))] = 3
    score[(dp > 15) | (ne > 0.1) | (ep > 0.1)] = 4
    return score


def cns_score(gcs: pd.Series) -> pd.Series:
    return pd.cut(gcs, [-np.inf, 6, 10, 13, 15, np.inf], labels=[4, 3, 2, 1, 0], right=False).astype(float)


def renal_score(creatinine: pd.Series, urine_24h: pd.Series) -> pd.Series:
    cr = pd.cut(creatinine, [-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf], labels=[0, 1, 2, 3, 4], right=False).astype(float)
    uo = pd.Series(np.nan, index=urine_24h.index)
    uo[urine_24h.notna()] = 0
    uo[urine_24h < 500] = 3
    uo[urine_24h < 200] = 4
    return pd.concat([cr, uo], axis=1).max(axis=1, skipna=True)


def hourly_sofa(wide: pd.DataFrame, config: Optional[SofaConfig] = None) -> pd.DataFrame:
    """Compute per-hour component and total SOFA for a (stay_id, hour)-indexed wide table.

    Returns a DataFrame with the six component columns and ``sofa`` (sum, treating NaN
    components as 0 when ``assume_normal``; else NaN if any component is NaN).
    """
    cfg = config or SofaConfig()
    df = wide.sort_index().copy()
    for col in DEFAULT_CARRY_FORWARD_H:
        if col not in df:
            df[col] = np.nan
    g = df.groupby(level=0, group_keys=False)
    filled = pd.DataFrame(index=df.index)
    for col, h in cfg.carry_forward_h.items():
        filled[col] = g[col].apply(lambda s, h=h: _carry_forward(s, h)) if h > 0 else df[col]
    if cfg.gcs_sedated_value is not None and "sedated" in df:
        filled.loc[df["sedated"].fillna(0).astype(bool), "gcs"] = cfg.gcs_sedated_value
    # urine output over a trailing window (sum of hourly values)
    uo = df["urine_output"]
    win = cfg.urine_window_h
    roll = uo.groupby(level=0, group_keys=False).apply(lambda s: s.rolling(win, min_periods=1).sum())
    n_obs = uo.notna().groupby(level=0, group_keys=False).apply(lambda s: s.rolling(win, min_periods=1).sum())
    if cfg.urine_extrapolate:
        urine24 = roll / n_obs.clip(lower=1) * win
    else:
        urine24 = roll.where(n_obs >= win)
    out = pd.DataFrame(index=df.index)
    out["respiration"] = respiration_score(filled["pao2"], filled["fio2"], filled["vent"])
    out["coagulation"] = coagulation_score(filled["platelets"])
    out["liver"] = liver_score(filled["bilirubin"])
    out["cardiovascular"] = cardiovascular_score(filled["map"], filled["norepinephrine"], filled["epinephrine"],
                                                filled["dopamine"], filled["dobutamine"])
    out["cns"] = cns_score(filled["gcs"])
    out["renal"] = renal_score(filled["creatinine"], urine24)
    if cfg.assume_normal:
        out["sofa"] = out[COMPONENTS].fillna(0).sum(axis=1)
    else:
        out["sofa"] = out[COMPONENTS].sum(axis=1, min_count=len(COMPONENTS))
    return out


def sofa_baseline(sofa: pd.Series, t_si: int, method: str, lookback_h: int = 48, icu_first_hour: int = 0) -> float:
    """Baseline SOFA for one stay relative to the suspicion hour ``t_si``.

    method: ``zero`` | ``min_lookback`` (minimum in [t_si - lookback_h, t_si)) | ``first_icu``
    (value at the first ICU hour) | ``min_prior`` (minimum of all hours before t_si).
    Falls back to 0 when the requested window has no values.
    """
    s = sofa.dropna()
    if method == "zero":
        return 0.0
    if method == "first_icu":
        v = s[s.index >= icu_first_hour]
        return float(v.iloc[0]) if len(v) else 0.0
    if method == "min_lookback":
        v = s[(s.index >= t_si - lookback_h) & (s.index < t_si)]
        return float(v.min()) if len(v) else 0.0
    if method == "min_prior":
        v = s[s.index < t_si]
        return float(v.min()) if len(v) else 0.0
    raise ValueError(method)
