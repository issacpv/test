"""Harmonise CGM / insulin / carb streams to a common 5-minute grid and window it.

Each dataset has its own file format (OhioT1DM XML, AZT1D CSV, DiaTrend
spreadsheets, Nightscout JSON for OpenAPS). The reader-specific parsers live in
``read_*`` functions and all produce a canonical long DataFrame with columns
[time, glucose, basal, bolus, carbs, device_mode]. A synthetic generator is
provided for tests and for exercising the pipeline without real data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID_MIN = 5  # CGM cadence in minutes


@dataclass
class Windowed:
    X: np.ndarray          # (n_windows, history_len) glucose history
    X_exo: np.ndarray      # (n_windows, history_len, n_exo) exogenous inputs
    y: np.ndarray          # (n_windows,) target glucose at the horizon
    valid: np.ndarray      # (n_windows,) bool: target not imputed
    subject: np.ndarray    # (n_windows,) subject id per window


def to_grid(df: pd.DataFrame, freq_min: int = GRID_MIN) -> pd.DataFrame:
    """Resample a canonical long stream to a regular grid; flag imputed glucose.

    Glucose is linearly interpolated across short gaps; an ``imputed`` boolean
    marks interpolated samples so evaluation can exclude them. Insulin/carbs are
    summed into their bin (event-like); device_mode is forward-filled.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")
    rule = f"{freq_min}min"
    g = df["glucose"].resample(rule).mean()
    imputed = g.isna()
    g = g.interpolate("linear", limit=6)  # up to 30 min gap
    out = pd.DataFrame({"glucose": g, "imputed": imputed})
    for col in ("basal", "bolus", "carbs"):
        if col in df.columns:
            out[col] = df[col].resample(rule).sum().reindex(out.index).fillna(0.0)
        else:
            out[col] = 0.0
    if "device_mode" in df.columns:
        out["device_mode"] = df["device_mode"].resample(rule).ffill().reindex(out.index).ffill()
    else:
        out["device_mode"] = "unknown"
    return out.reset_index()


def insulin_on_board(bolus: np.ndarray, tau_min: float = 55.0, freq_min: int = GRID_MIN) -> np.ndarray:
    """Simple exponential insulin-on-board (IOB) from past boluses (causal)."""
    bolus = np.asarray(bolus, float)
    decay = np.exp(-freq_min / tau_min)
    iob = np.zeros_like(bolus)
    acc = 0.0
    for i, b in enumerate(bolus):
        acc = acc * decay + b
        iob[i] = acc
    return iob


def make_windows(grid: pd.DataFrame, history_len: int = 12, horizon_min: int = 30,
                 subject: int = 0, freq_min: int = GRID_MIN) -> Windowed:
    """Slice a gridded stream into supervised windows for h-min-ahead forecasting.

    History = last ``history_len`` glucose samples (default 60 min); exogenous
    inputs are IOB and carbs over the same window; target is glucose
    ``horizon_min`` ahead. Windows whose history contains long gaps are skipped.
    """
    g = grid["glucose"].to_numpy(float)
    imp = grid["imputed"].to_numpy(bool)
    iob = insulin_on_board(grid["bolus"].to_numpy(float), freq_min=freq_min)
    carbs = grid["carbs"].to_numpy(float)
    h = horizon_min // freq_min
    X, X_exo, y, valid, subj = [], [], [], [], []
    for t in range(history_len, len(g) - h):
        hist = g[t - history_len:t]
        if np.any(np.isnan(hist)) or np.isnan(g[t + h]):
            continue
        # skip windows dominated by imputed history
        if imp[t - history_len:t].mean() > 0.5:
            continue
        X.append(hist)
        X_exo.append(np.stack([iob[t - history_len:t], carbs[t - history_len:t]], axis=-1))
        y.append(g[t + h])
        valid.append(not imp[t + h])
        subj.append(subject)
    return Windowed(np.array(X), np.array(X_exo), np.array(y),
                    np.array(valid, bool), np.array(subj))


def synthetic_stream(n_hours: int = 48, regimen: str = "open-loop", seed: int = 0,
                     start: str = "2024-01-01") -> pd.DataFrame:
    """Generate a canonical CGM stream. AID ('closed-loop') has lower variance
    and stronger mean-reversion (controller feedback), matching the shift axis."""
    rng = np.random.default_rng(seed)
    n = n_hours * 60 // GRID_MIN
    times = pd.date_range(start, periods=n, freq=f"{GRID_MIN}min")
    g = np.zeros(n)
    g[0] = 140.0
    reversion = 0.15 if regimen in ("closed-loop", "DIY-loop", "hybrid-closed-loop") else 0.04
    noise = 6.0 if regimen != "open-loop" else 14.0
    bolus = np.zeros(n)
    carbs = np.zeros(n)
    for t in range(1, n):
        meal = 0.0
        if rng.random() < 0.02:  # occasional meal
            carbs[t] = rng.uniform(20, 70)
            bolus[t] = carbs[t] / 10.0
            meal = carbs[t] * 0.8
        g[t] = g[t - 1] + reversion * (120 - g[t - 1]) + meal - 5 * bolus[t - 1] + rng.normal(0, noise)
        g[t] = np.clip(g[t], 40, 400)
    mode = "sleep" if regimen != "open-loop" else "unknown"
    return pd.DataFrame({"time": times, "glucose": g, "basal": 0.8, "bolus": bolus,
                         "carbs": carbs, "device_mode": mode})
