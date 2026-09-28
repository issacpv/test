"""Landmark dataset construction and landmark supermodel for dynamic HAPI prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAJECTORY_FEATURES: tuple[str, ...] = ("braden_last", "braden_min", "braden_mean48", "braden_slope48", "n_assess_24h", "hours_since_last")
OBSERVATION_FEATURES: tuple[str, ...] = ("n_assess_24h", "hours_since_last")


def _slope(hours: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.ptp(hours) == 0:
        return 0.0
    return float(np.polyfit(hours, values, 1)[0])


def build_landmark_dataset(
    braden: pd.DataFrame,
    stays: pd.DataFrame,
    events: pd.DataFrame,
    prevention: pd.DataFrame | None = None,
    landmark_h: float = 24.0,
    horizon_h: float = 72.0,
    max_landmark_days: int = 14,
) -> pd.DataFrame:
    """One row per (stay, landmark) with strictly-past features and a future outcome.

    ``braden``: stay_id, charttime, braden_total (+ optional subscale columns).
    ``stays``: stay_id, intime, outtime (+ static covariates carried through).
    ``events``: stay_id, first_pi_time (NaT/absent = no event).
    ``prevention``: optional stay_id, charttime of pressure-relief / repositioning documentation.

    Risk set at landmark L: stay ongoing (L < outtime) and no event at or before L.
    Outcome = event in (L, L + horizon].  ``censored`` = discharged before L + horizon
    without event (outcome 0 by administrative censoring).  Features use charttime <= L only.
    """
    b = braden.dropna(subset=["braden_total"]).copy()
    b["charttime"] = pd.to_datetime(b["charttime"])
    b_by = {s: g.sort_values("charttime") for s, g in b.groupby("stay_id")}
    ev = events.set_index("stay_id")["first_pi_time"] if len(events) else pd.Series(dtype="datetime64[ns]")
    prev_by = {}
    if prevention is not None and len(prevention):
        p = prevention.copy()
        p["charttime"] = pd.to_datetime(p["charttime"])
        prev_by = {s: g["charttime"].to_numpy() for s, g in p.groupby("stay_id")}
    static_cols = [c for c in stays.columns if c not in ("stay_id", "intime", "outtime")]
    rows = []
    for _, st in stays.iterrows():
        sid = st["stay_id"]
        intime, outtime = pd.Timestamp(st["intime"]), pd.Timestamp(st["outtime"])
        e_time = ev.get(sid, pd.NaT) if len(ev) else pd.NaT
        g = b_by.get(sid)
        for k in range(1, max_landmark_days + 1):
            L = intime + pd.Timedelta(hours=k * landmark_h)
            if L >= outtime or (pd.notna(e_time) and e_time <= L):
                break
            row = {"stay_id": sid, "landmark_day": k, "landmark_time": L}
            row.update({c: st[c] for c in static_cols})
            if g is not None:
                past = g[g["charttime"] <= L]
            else:
                past = None
            if past is None or past.empty:
                row.update({f: np.nan for f in TRAJECTORY_FEATURES})
                row["n_assess_24h"] = 0
                row["braden_first"] = np.nan
            else:
                hrs = (past["charttime"] - L).dt.total_seconds().to_numpy() / 3600.0  # <= 0
                vals = past["braden_total"].to_numpy(float)
                w48 = hrs >= -48.0
                row["braden_last"] = float(vals[-1])
                row["braden_min"] = float(vals.min())
                row["braden_mean48"] = float(vals[w48].mean()) if w48.any() else float(vals[-1])
                row["braden_slope48"] = _slope(hrs[w48], vals[w48]) if w48.sum() >= 2 else 0.0
                row["n_assess_24h"] = int((hrs >= -24.0).sum())
                row["hours_since_last"] = float(-hrs[-1])
                row["braden_first"] = float(vals[0]) if hrs[0] <= -(k * landmark_h - 24.0) else float(vals[0])
            if prev_by:
                pt = prev_by.get(sid, np.array([], dtype="datetime64[ns]"))
                row["prevention_24h"] = int(np.any((pt <= np.datetime64(L)) & (pt > np.datetime64(L - pd.Timedelta(hours=24)))))
            horizon_end = L + pd.Timedelta(hours=horizon_h)
            row["outcome"] = int(pd.notna(e_time) and (L < e_time <= horizon_end))
            row["censored"] = int(row["outcome"] == 0 and outtime < horizon_end)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def fit_landmark_supermodel(df: pd.DataFrame, features: tuple[str, ...] = TRAJECTORY_FEATURES, interactions: bool = True, C: float = 1.0):
    """Pooled logistic landmark supermodel: outcome ~ features * landmark_day (+ landmark_day^2).

    Missing features are median-imputed with an indicator.  Returns a fitted sklearn
    pipeline and the design-column list, so ``predict_landmark`` can rebuild the design.
    """
    X, cols = _design(df, features, interactions)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(X, df["outcome"].to_numpy(int))
    model.design_columns_ = cols  # type: ignore[attr-defined]
    model.design_features_ = (features, interactions)  # type: ignore[attr-defined]
    model.impute_medians_ = df[list(features)].median(numeric_only=True).to_dict()  # type: ignore[attr-defined]
    return model


def _design(df: pd.DataFrame, features: tuple[str, ...], interactions: bool, medians: dict | None = None) -> tuple[np.ndarray, list[str]]:
    d = df.copy()
    cols: list[str] = []
    meds = medians or d[list(features)].median(numeric_only=True).to_dict()
    for f in features:
        miss = d[f].isna()
        d[f"{f}_missing"] = miss.astype(int)
        d[f] = d[f].fillna(meds.get(f, 0.0))
        cols += [f, f"{f}_missing"]
    d["lm"] = d["landmark_day"].astype(float)
    d["lm2"] = d["lm"] ** 2
    cols += ["lm", "lm2"]
    if interactions:
        for f in features:
            d[f"{f}_x_lm"] = d[f] * d["lm"]
            cols.append(f"{f}_x_lm")
    return d[cols].to_numpy(float), cols


def predict_landmark(model, df: pd.DataFrame) -> np.ndarray:
    features, interactions = model.design_features_
    X, _ = _design(df, features, interactions, medians=model.impute_medians_)
    return model.predict_proba(X)[:, 1]


def fit_static_baseline(df: pd.DataFrame, feature: str = "braden_first", C: float = 1.0):
    """Static baseline: admission Braden (first total) with landmark-day terms only."""
    return fit_landmark_supermodel(df, features=(feature,), interactions=False, C=C)


def permute_observation_features(df: pd.DataFrame, seed: int = 0, features: tuple[str, ...] = OBSERVATION_FEATURES) -> pd.DataFrame:
    """Within-stay permutation of observation-process features (null for the informative-observation test)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for f in features:
        out[f] = out.groupby("stay_id")[f].transform(lambda s: s.to_numpy()[rng.permutation(len(s))])
    return out
