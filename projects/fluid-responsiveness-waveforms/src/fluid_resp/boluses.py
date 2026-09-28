"""Fluid-bolus identification, pre/post response windows, sham windows and validity prerequisites.

Inputs are MIMIC-style tables (``inputevents`` with ``starttime, endtime, amount, amountuom, itemid, rate``;
1-min numerics with ``time, map, hr, sbp, pp``).  Item ids are resolved by regex on the item dictionary.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

FLUID_PATTERNS: Dict[str, str] = {
    "crystalloid": r"^(?:nacl 0\.9%|0\.9% sodium chloride|lr$|lactated ringers?|plasma-?lyte|normal saline)",
    "colloid": r"albumin|hetastarch|hespan|voluven|dextran",
    "blood": r"packed red blood|prbc|fresh frozen plasma|ffp|platelets|cryoprecipitate",
}
VASOPRESSOR_PATTERN = r"norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine|dobutamine"
ML_PER_UNIT = {"ml": 1.0, "l": 1000.0, "uL": 0.001}
CONTROLLED_MODES = ("CMV", "AC", "PCV", "PRVC", "VC", "PC", "SIMV")  # SIMV counted controlled only without triggering


def resolve_itemids(d_items: pd.DataFrame, pattern: str, label_col: str = "label", id_col: str = "itemid") -> pd.DataFrame:
    """Rows of the item dictionary whose label matches ``pattern`` (case-insensitive regex)."""
    m = d_items[label_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    return d_items.loc[m, [id_col, label_col]].reset_index(drop=True)


def identify_boluses(
    inputevents: pd.DataFrame,
    fluid_itemids: Iterable[int],
    min_ml: float = 250.0,
    max_minutes: float = 30.0,
    isolation_pre_min: float = 60.0,
    isolation_post_min: float = 30.0,
) -> pd.DataFrame:
    """Rows of ``inputevents`` that qualify as a bolus, with an ``isolated`` flag.

    A bolus is >= ``min_ml`` delivered in <= ``max_minutes``. ``isolated`` is True when no other qualifying
    bolus of the same stay starts within ``[start - isolation_pre_min, end + isolation_post_min]``.
    """
    ids = set(int(i) for i in fluid_itemids)
    d = inputevents[inputevents["itemid"].isin(ids)].copy()
    d["start"] = pd.to_datetime(d["starttime"])
    d["end"] = pd.to_datetime(d["endtime"])
    unit = d["amountuom"].astype(str).str.strip().str.lower()
    d["amount_ml"] = pd.to_numeric(d["amount"], errors="coerce") * unit.map({"ml": 1.0, "l": 1000.0})
    d["minutes"] = (d["end"] - d["start"]).dt.total_seconds() / 60.0
    d = d[(d["amount_ml"] >= min_ml) & (d["minutes"] <= max_minutes) & (d["minutes"] >= 0)].copy()
    d = d.sort_values(["stay_id", "start"]).reset_index(drop=True)
    isolated = np.ones(len(d), dtype=bool)
    for sid, g in d.groupby("stay_id"):
        idx = g.index.to_numpy()
        st, en = g["start"].to_numpy(), g["end"].to_numpy()
        for k in range(len(idx)):
            lo = st[k] - np.timedelta64(int(isolation_pre_min * 60), "s")
            hi = en[k] + np.timedelta64(int(isolation_post_min * 60), "s")
            others = np.delete(st, k)
            if ((others >= lo) & (others <= hi)).any():
                isolated[idx[k]] = False
    d["isolated"] = isolated
    d["bolus_id"] = np.arange(len(d))
    return d[["bolus_id", "stay_id", "itemid", "start", "end", "amount_ml", "minutes", "isolated"]]


def _window_stats(num: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp, cols: Sequence[str]) -> Dict[str, float]:
    s = num[(num["time"] >= lo) & (num["time"] < hi)]
    out: Dict[str, float] = {"n": int(len(s))}
    for c in cols:
        v = pd.to_numeric(s[c], errors="coerce").dropna() if c in s else pd.Series(dtype=float)
        out[f"{c}_mean"] = float(v.mean()) if len(v) else np.nan
        out[f"{c}_max"] = float(v.max()) if len(v) else np.nan
    return out


def response_windows(
    numerics: pd.DataFrame,
    boluses: pd.DataFrame,
    pre_min: float = 15.0,
    post_min: float = 30.0,
    cols: Sequence[str] = ("map", "pp", "hr"),
    min_samples: int = 5,
) -> pd.DataFrame:
    """Pre-bolus ``[start - pre_min, start)`` and post-bolus ``[end, end + post_min)`` statistics per bolus.

    ``numerics`` needs ``stay_id, time`` and the value columns. Rows with fewer than ``min_samples`` in either
    window get ``valid = False``.
    """
    rows = []
    for _, b in boluses.iterrows():
        num = numerics[numerics["stay_id"] == b["stay_id"]]
        pre = _window_stats(num, b["start"] - pd.Timedelta(minutes=pre_min), b["start"], cols)
        post = _window_stats(num, b["end"], b["end"] + pd.Timedelta(minutes=post_min), cols)
        rec = {"bolus_id": b["bolus_id"], "stay_id": b["stay_id"], "start": b["start"], "end": b["end"], "amount_ml": b.get("amount_ml", np.nan)}
        rec.update({f"pre_{k}": v for k, v in pre.items()})
        rec.update({f"post_{k}": v for k, v in post.items()})
        rec["valid"] = pre["n"] >= min_samples and post["n"] >= min_samples
        rows.append(rec)
    return pd.DataFrame(rows)


def classify_response(df: pd.DataFrame, variable: str = "map", threshold_pct: float = 10.0, use_peak: bool = False) -> pd.Series:
    """Responder label: relative change of ``variable`` from pre-mean to post-mean (or post-max) >= threshold."""
    post = df[f"post_{variable}_max" if use_peak else f"post_{variable}_mean"]
    pre = df[f"pre_{variable}_mean"]
    delta = 100.0 * (post - pre) / pre
    return (delta >= threshold_pct) & df["valid"].astype(bool)


def sham_windows(
    numerics: pd.DataFrame,
    boluses: pd.DataFrame,
    responses: pd.DataFrame,
    n_per_bolus: int = 1,
    match_tol_mmhg: float = 5.0,
    min_gap_min: float = 120.0,
    pre_min: float = 15.0,
    post_min: float = 30.0,
    n_candidates: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Sham "boluses": times in the same stay, >= ``min_gap_min`` from any real bolus, whose pre-window MAP
    matches the real bolus's pre-window MAP within ``match_tol_mmhg``.

    The response computed on sham windows estimates what regression to the mean and background drift produce
    without treatment. Returns the same columns as :func:`response_windows` plus ``matched_bolus_id``.
    """
    rng = np.random.default_rng() if rng is None else rng
    bol_by_stay = {s: g for s, g in boluses.groupby("stay_id")}
    out = []
    for _, r in responses.iterrows():
        if not r["valid"] or not np.isfinite(r["pre_map_mean"]):
            continue
        num = numerics[numerics["stay_id"] == r["stay_id"]].sort_values("time")
        if len(num) < 50:
            continue
        real = bol_by_stay[r["stay_id"]]
        dur = r["end"] - r["start"]
        t_lo, t_hi = num["time"].min() + pd.Timedelta(minutes=pre_min), num["time"].max() - dur - pd.Timedelta(minutes=post_min)
        if t_hi <= t_lo:
            continue
        found = 0
        for _ in range(n_candidates):
            t = t_lo + (t_hi - t_lo) * rng.random()
            t = pd.Timestamp(t).floor("min")
            gaps = np.abs((real["start"] - t).dt.total_seconds() / 60.0)
            if (gaps < min_gap_min).any():
                continue
            fake = pd.DataFrame([{"bolus_id": -1, "stay_id": r["stay_id"], "start": t, "end": t + dur, "amount_ml": 0.0}])
            rw = response_windows(num, fake, pre_min=pre_min, post_min=post_min).iloc[0]
            if not rw["valid"] or abs(rw["pre_map_mean"] - r["pre_map_mean"]) > match_tol_mmhg:
                continue
            rec = rw.to_dict()
            rec["matched_bolus_id"] = r["bolus_id"]
            out.append(rec)
            found += 1
            if found >= n_per_bolus:
                break
    return pd.DataFrame(out)


def vasopressor_change_flag(inputevents: pd.DataFrame, vaso_itemids: Iterable[int], boluses: pd.DataFrame, pre_min: float = 15.0, post_min: float = 30.0) -> pd.Series:
    """True when a vasopressor infusion starts, stops or changes rate inside the bolus analysis window."""
    ids = set(int(i) for i in vaso_itemids)
    v = inputevents[inputevents["itemid"].isin(ids)].copy()
    v["starttime"] = pd.to_datetime(v["starttime"])
    v["endtime"] = pd.to_datetime(v["endtime"])
    flags = []
    for _, b in boluses.iterrows():
        lo, hi = b["start"] - pd.Timedelta(minutes=pre_min), b["end"] + pd.Timedelta(minutes=post_min)
        s = v[v["stay_id"] == b["stay_id"]]
        touches = ((s["starttime"] >= lo) & (s["starttime"] <= hi)) | ((s["endtime"] >= lo) & (s["endtime"] <= hi))
        flags.append(bool(touches.any()))
    return pd.Series(flags, index=boluses.index, name="vaso_change")


def predicted_body_weight(height_cm: float, male: bool) -> float:
    """ARDSNet predicted body weight (kg): 50 + 0.91 (height - 152.4) for men, 45.5 + 0.91 (height - 152.4) for women."""
    base = 50.0 if male else 45.5
    return base + 0.91 * (height_cm - 152.4)


def prerequisite_flags(
    tidal_volume_ml: float,
    pbw_kg: float,
    vent_mode: Optional[str],
    rr_set: Optional[float],
    rr_total: Optional[float],
    rr_irregularity: float,
    vt_threshold_ml_kg: float = 8.0,
    trigger_tolerance: float = 2.0,
    irregularity_max: float = 0.1,
) -> Dict[str, bool]:
    """Validity prerequisites for PPV: adequate tidal volume, controlled ventilation without triggering, regular rhythm."""
    vt_ok = bool(np.isfinite(tidal_volume_ml) and pbw_kg > 0 and tidal_volume_ml / pbw_kg >= vt_threshold_ml_kg)
    mode_ok = bool(vent_mode is not None and any(m.lower() in str(vent_mode).lower() for m in CONTROLLED_MODES))
    no_trigger = bool(rr_set is not None and rr_total is not None and np.isfinite(rr_set) and np.isfinite(rr_total) and rr_total <= rr_set + trigger_tolerance)
    regular = bool(np.isfinite(rr_irregularity) and rr_irregularity <= irregularity_max)
    return {"vt_ok": vt_ok, "controlled_mode": mode_ok, "no_triggering": no_trigger, "regular_rhythm": regular, "all_ok": vt_ok and mode_ok and no_trigger and regular}
