"""Knee-level OAI cohort construction and competing-risk event coding.

Inputs are long tables in the spirit of the OAI central x-ray readings
(``ID, SIDE, VISIT, KL[, READER]``), the outcomes file (TKR dates as months
from baseline) and a follow-up table (last completed imaging visit, death).
The functions are agnostic to the exact OAI column names; map them when
loading (e.g. ``V00XRKL`` -> ``KL`` for visit ``V00``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IMAGING_VISITS: dict[str, int] = {"V00": 0, "V01": 12, "V03": 24, "V05": 36, "V06": 48, "V08": 72, "V10": 96}
EVENT_CENSORED, EVENT_PROGRESSION, EVENT_TKR, EVENT_DEATH = 0, 1, 2, 3


def knee_id(df: pd.DataFrame) -> pd.Series:
    """Unique knee identifier ``<ID>_<SIDE>`` (SIDE: 1 = right, 2 = left as in OAI)."""
    return df["ID"].astype(str) + "_" + df["SIDE"].astype(int).astype(str)


def knee_long_table(readings: pd.DataFrame) -> pd.DataFrame:
    """Sort readings by knee and visit and attach ``months`` from ``IMAGING_VISITS``.

    When several readings exist per knee-visit (multiple readers), the
    *adjudicated* or median grade is used for the trajectory; reader-level
    rows are kept for :func:`reader_disagreement`.
    """
    df = readings.copy()
    df["knee"] = knee_id(df)
    df["months"] = df["VISIT"].map(IMAGING_VISITS)
    if df["months"].isna().any():
        bad = sorted(df.loc[df["months"].isna(), "VISIT"].unique())
        raise ValueError(f"unknown imaging visit codes: {bad}")
    agg = df.groupby(["knee", "ID", "SIDE", "VISIT", "months"], as_index=False)["KL"].median()
    return agg.sort_values(["knee", "months"]).reset_index(drop=True)


def progression_time(long: pd.DataFrame, min_increase: int = 1) -> pd.DataFrame:
    """Per knee: baseline KL and months of the first visit with KL increase >= ``min_increase``.

    Returns columns ``knee, ID, SIDE, kl0, prog_months (NaN if none), last_visit_months``.
    """
    rows = []
    for knee, g in long.groupby("knee", sort=False):
        g = g.dropna(subset=["KL"]).sort_values("months")
        if g.empty or g["months"].iloc[0] != 0:
            continue
        kl0 = float(g["KL"].iloc[0])
        inc = g[g["KL"] >= kl0 + min_increase]
        rows.append({
            "knee": knee, "ID": g["ID"].iloc[0], "SIDE": int(g["SIDE"].iloc[0]), "kl0": kl0,
            "prog_months": float(inc["months"].iloc[0]) if len(inc) else np.nan,
            "last_visit_months": float(g["months"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def competing_risk_table(
    prog: pd.DataFrame,
    tkr: pd.DataFrame | None = None,
    death: pd.DataFrame | None = None,
    exclude_kl4: bool = True,
) -> pd.DataFrame:
    """Assign one event type and time per knee.

    Parameters
    ----------
    prog : output of :func:`progression_time`.
    tkr : optional ``ID, SIDE, tkr_months`` (knee-specific replacement).
    death : optional ``ID, death_months``.

    Rules: the earliest of progression, TKR and death defines the event;
    ties favour TKR over progression (a replaced knee is no longer graded);
    otherwise censor at the last imaging visit. Knees with baseline KL 4 are
    excluded by default (cannot progress on the KL scale).
    """
    df = prog.copy()
    if exclude_kl4:
        df = df[df["kl0"] < 4]
    df["tkr_months"] = np.nan
    if tkr is not None and len(tkr):
        t = tkr[["ID", "SIDE", "tkr_months"]].copy()
        t["knee"] = knee_id(t)
        df = df.drop(columns="tkr_months").merge(t[["knee", "tkr_months"]], on="knee", how="left")
    df["death_months"] = np.nan
    if death is not None and len(death):
        df = df.drop(columns="death_months").merge(death[["ID", "death_months"]], on="ID", how="left")
    times = np.column_stack([df["prog_months"], df["tkr_months"], df["death_months"]]).astype(float)
    codes = np.array([EVENT_PROGRESSION, EVENT_TKR, EVENT_DEATH])
    event = np.full(len(df), EVENT_CENSORED)
    time = df["last_visit_months"].to_numpy(float).copy()
    for i in range(len(df)):
        row = times[i]
        if np.all(np.isnan(row)):
            continue
        order = np.argsort(np.where(np.isnan(row), np.inf, row), kind="stable")
        best = order[0]
        # tie: TKR beats progression at the same month
        if not np.isnan(row[1]) and row[1] == row[best]:
            best = 1
        event[i] = codes[best]
        time[i] = row[best]
    df["event"] = event
    df["time_months"] = time
    return df.reset_index(drop=True)


def reader_disagreement(readings: pd.DataFrame) -> pd.DataFrame:
    """Per knee-visit: number of readers, whether any pair disagreed, and max absolute KL difference."""
    if "READER" not in readings.columns:
        raise KeyError("readings must include a READER column")
    df = readings.copy()
    df["knee"] = knee_id(df)
    g = df.groupby(["knee", "VISIT"])["KL"]
    out = g.agg(n_readers="count", kl_min="min", kl_max="max").reset_index()
    out["max_abs_diff"] = out["kl_max"] - out["kl_min"]
    out["disagree"] = (out["n_readers"] >= 2) & (out["max_abs_diff"] > 0)
    return out


def participant_grouped_folds(ids: np.ndarray | pd.Series, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per row such that all rows (both knees, all visits) of a participant share a fold."""
    ids = np.asarray(ids)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[perm[i]]: i % n_folds for i in range(len(uniq))}
    return np.array([fold_of[i] for i in ids])


def simulate_oai_like(n_participants: int = 500, seed: int = 0, p_second_reader: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic OAI-like readings, TKR and follow-up tables for offline development.

    Progression hazard increases with baseline KL and a latent risk score;
    TKR occurs mostly in KL 3 knees; dropout is random. A second reader is
    simulated for a fraction of knee-visits with grade noise.
    """
    rng = np.random.default_rng(seed)
    visits = list(IMAGING_VISITS.items())
    readings, tkr_rows, fu_rows = [], [], []
    for pid in range(1, n_participants + 1):
        last_visit = rng.choice([48, 72, 96, 96, 96])
        died = rng.random() < 0.05
        death_m = float(rng.choice([36, 60, 84])) if died else np.nan
        fu_rows.append({"ID": pid, "last_visit_months": float(last_visit), "death_months": death_m})
        for side in (1, 2):
            kl = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.25, 0.25, 0.15]))
            risk = rng.normal()
            prog_hazard = 0.004 * (1 + kl) * np.exp(0.5 * risk)
            prog_m = rng.exponential(1 / prog_hazard)
            tkr_m = rng.exponential(1 / (0.002 * (kl >= 2) * (1 + kl) + 1e-6)) if kl >= 2 else np.inf
            if tkr_m < last_visit and (np.isnan(death_m) or tkr_m < death_m):
                tkr_rows.append({"ID": pid, "SIDE": side, "tkr_months": float(12 * np.ceil(tkr_m / 12))})
            cur = kl
            for vcode, m in visits:
                if m > last_visit or (not np.isnan(death_m) and m > death_m) or (tkr_m < m):
                    break
                if m > 0 and m >= prog_m and cur < 4:
                    cur = min(4, cur + 1)
                    prog_m = m + rng.exponential(1 / prog_hazard)
                readings.append({"ID": pid, "SIDE": side, "VISIT": vcode, "READER": 1, "KL": cur})
                if rng.random() < p_second_reader:
                    noise = rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])
                    readings.append({"ID": pid, "SIDE": side, "VISIT": vcode, "READER": 2, "KL": int(np.clip(cur + noise, 0, 4))})
    return pd.DataFrame(readings), pd.DataFrame(tkr_rows, columns=["ID", "SIDE", "tkr_months"]), pd.DataFrame(fu_rows)
