"""Event tables, documentation delays, windowed labels and the phase-randomisation null."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .circular import clock_hour

MIMIC_EVENT_SQL = """
-- One row per (stay, event_type) with the defining timestamp. Run against a DuckDB built by
-- scripts/download_data.py --build-duckdb (tables hosp_*, icu_*). Extend with sepsis3 / kdigo
-- concept tables from mimic-code once they are materialised.
WITH icu AS (
  SELECT stay_id, hadm_id, subject_id, intime, outtime FROM icu_icustays
), adm AS (
  SELECT hadm_id, admittime, dischtime, deathtime, admission_location FROM hosp_admissions
)
SELECT icu.stay_id, 'icu_admit'     AS event_type, icu.intime   AS event_time FROM icu
UNION ALL SELECT icu.stay_id, 'icu_discharge', icu.outtime FROM icu
UNION ALL SELECT icu.stay_id, 'death', adm.deathtime FROM icu JOIN adm USING (hadm_id) WHERE adm.deathtime IS NOT NULL
UNION ALL SELECT icu.stay_id, 'hospital_discharge', adm.dischtime FROM icu JOIN adm USING (hadm_id)
UNION ALL SELECT icu.stay_id, 'first_antibiotic', MIN(p.starttime) FROM icu JOIN hosp_prescriptions p USING (hadm_id)
          WHERE lower(p.drug) SIMILAR TO '%(cillin|cef|penem|mycin|floxacin|cycline|vancomycin|zolid|sulfa|metronidazole)%'
          GROUP BY icu.stay_id
UNION ALL SELECT icu.stay_id, 'first_culture', MIN(m.charttime) FROM icu JOIN hosp_microbiologyevents m USING (hadm_id)
          WHERE m.charttime IS NOT NULL GROUP BY icu.stay_id
"""


def event_table_from_frames(frames: Dict[str, pd.DataFrame], time_col: str = "event_time", id_col: str = "stay_id") -> pd.DataFrame:
    """Stack {event_type: DataFrame(stay_id, event_time[, storetime])} into one event table with clock features."""
    parts = []
    for et, df in frames.items():
        d = df[[id_col, time_col] + (["storetime"] if "storetime" in df else [])].copy()
        d["event_type"] = et
        parts.append(d)
    ev = pd.concat(parts, ignore_index=True)
    ev[time_col] = pd.to_datetime(ev[time_col])
    ev["clock_hour"] = clock_hour(ev[time_col])
    ev["weekday"] = ev[time_col].dt.dayofweek
    ev["is_weekend"] = ev["weekday"] >= 5
    if "storetime" in ev:
        ev["store_delay_min"] = (pd.to_datetime(ev["storetime"]) - ev[time_col]).dt.total_seconds() / 60.0
    return ev


def documentation_delay_by_hour(df: pd.DataFrame, chart_col: str = "charttime", store_col: str = "storetime",
                                abnormal_col: Optional[str] = None, n_bins: int = 24) -> pd.DataFrame:
    """Median and IQR of (storetime - charttime) in minutes by chart hour (and abnormality flag if given)."""
    d = df[[chart_col, store_col] + ([abnormal_col] if abnormal_col else [])].dropna(subset=[chart_col, store_col]).copy()
    d["delay_min"] = (pd.to_datetime(d[store_col]) - pd.to_datetime(d[chart_col])).dt.total_seconds() / 60.0
    d = d[d["delay_min"] >= 0]
    d["hour_bin"] = (clock_hour(d[chart_col]) // (24 / n_bins)).astype(int)
    keys = ["hour_bin"] + ([abnormal_col] if abnormal_col else [])
    g = d.groupby(keys)["delay_min"]
    out = g.agg(n="size", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
    return out


def windowed_label(event_time: pd.Series, prediction_time: pd.Series, horizon_h: float) -> pd.Series:
    """1 if the event happens within (prediction_time, prediction_time + horizon]; 0 otherwise (NaT event = 0)."""
    et = pd.to_datetime(event_time)
    pt = pd.to_datetime(prediction_time)
    delta_h = (et - pt).dt.total_seconds() / 3600.0
    return ((delta_h > 0) & (delta_h <= horizon_h)).astype(int)


def prediction_times(admit_time: pd.Series, offsets_h: Sequence[float]) -> pd.DataFrame:
    """Long table of (stay index, prediction_time) at fixed offsets after admission."""
    rows = []
    for off in offsets_h:
        rows.append(pd.DataFrame({"stay_idx": admit_time.index, "offset_h": off,
                                  "prediction_time": pd.to_datetime(admit_time) + pd.to_timedelta(off, unit="h")}))
    return pd.concat(rows, ignore_index=True)


def phase_randomize(event_time: pd.Series, rng: np.random.Generator, mode: str = "uniform_24h",
                    group: Optional[pd.Series] = None) -> pd.Series:
    """Destroy the clock phase of event times while keeping their day-scale structure.

    mode = "uniform_24h": add an offset U(-12, +12) h per event (or per group if ``group`` is
    given, so all events of a stay move together and their ordering is preserved).
    mode = "same_day": replace the clock time by a uniform draw within the same calendar day.
    """
    et = pd.to_datetime(event_time)
    if mode == "uniform_24h":
        if group is not None:
            offs = pd.Series(rng.uniform(-12, 12, group.nunique()), index=group.unique())
            off = group.map(offs).to_numpy(float)
        else:
            off = rng.uniform(-12, 12, len(et))
        return et + pd.to_timedelta(off, unit="h")
    if mode == "same_day":
        day = et.dt.normalize()
        return day + pd.to_timedelta(rng.uniform(0, 24, len(et)), unit="h")
    raise ValueError(mode)


def label_flip_rate(event_time: pd.Series, prediction_time: pd.Series, horizon_h: float, n_rep: int = 200,
                    seed: int = 0, group: Optional[pd.Series] = None, mode: str = "uniform_24h") -> Dict[str, float]:
    """Fraction of windowed labels that change under phase randomisation of the event clock."""
    rng = np.random.default_rng(seed)
    y0 = windowed_label(event_time, prediction_time, horizon_h).to_numpy()
    flips, prev = [], []
    for _ in range(n_rep):
        y1 = windowed_label(phase_randomize(event_time, rng, mode=mode, group=group), prediction_time, horizon_h).to_numpy()
        flips.append(np.mean(y0 != y1))
        prev.append(y1.mean())
    return {"flip_rate": float(np.mean(flips)), "flip_rate_sd": float(np.std(flips)), "prevalence_obs": float(y0.mean()),
            "prevalence_null_mean": float(np.mean(prev)), "n": int(len(y0))}


def hazard_by_hour(event_time: pd.Series, at_risk_hours: pd.Series, n_bins: int = 24) -> pd.DataFrame:
    """Crude event rate per clock hour: events in bin / person-hours at risk in bin.

    ``at_risk_hours`` is a Series of per-stay arrays of the clock hours during which the stay
    was at risk (e.g. every hour from admission to discharge); it is expanded to bin counts.
    """
    edges = np.linspace(0, 24, n_bins + 1)
    ev_counts, _ = np.histogram(clock_hour(pd.Series(pd.to_datetime(event_time))).dropna() % 24, bins=edges)
    exposure = np.zeros(n_bins)
    for arr in at_risk_hours:
        c, _ = np.histogram(np.asarray(arr, float) % 24, bins=edges)
        exposure += c
    rate = np.divide(ev_counts, exposure, out=np.full(n_bins, np.nan), where=exposure > 0)
    return pd.DataFrame({"hour": edges[:-1], "events": ev_counts, "person_hours": exposure, "rate_per_hour": rate})


def richardson_lucy_circular(observed: np.ndarray, kernel: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Deconvolve a circular hourly histogram with a delay kernel (Richardson-Lucy, periodic)."""
    obs = np.clip(np.asarray(observed, float), 1e-12, None)
    k = np.asarray(kernel, float)
    k = k / k.sum()
    K = np.fft.fft(k, len(obs))
    est = np.full(len(obs), obs.mean())
    for _ in range(n_iter):
        conv = np.real(np.fft.ifft(np.fft.fft(est) * K))
        ratio = obs / np.clip(conv, 1e-12, None)
        est = est * np.real(np.fft.ifft(np.fft.fft(ratio) * np.conj(K)))
        est = np.clip(est, 0, None)
    return est
