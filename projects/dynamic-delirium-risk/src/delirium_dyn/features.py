"""Time-varying features for landmark models: sedative exposure, RASS trajectory, assessment process.

Three blocks are kept separable so that the sedation-leakage ablation can drop a block at once:

* ``sed_*``   sedation-policy block (drug exposures, RASS statistics);
* ``asm_*``   assessment-process block (hours since last CAM-ICU, previous window state, screening fraction);
* everything else is physiology/demographics (supplied by the caller and merged on ``stay_id, window_idx``).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

SEDATIVE_PATTERNS: Dict[str, str] = {
    "propofol": r"propofol",
    "benzo": r"midazolam|lorazepam|diazepam",
    "dex": r"dexmedetomidine|precedex",
    "opioid": r"fentanyl|hydromorphone|morphine",
    "ketamine": r"ketamine",
}

SEDATION_COLS_PREFIX = "sed_"
ASSESSMENT_COLS_PREFIX = "asm_"


def resolve_sedative_itemids(d_items: pd.DataFrame) -> Dict[str, list]:
    """Map drug class -> list of ``inputevents`` item ids whose label matches the class regex."""
    out: Dict[str, list] = {}
    for cls, pat in SEDATIVE_PATTERNS.items():
        m = d_items["label"].astype(str).str.contains(pat, case=False, regex=True, na=False)
        out[cls] = d_items.loc[m, "itemid"].astype(int).tolist()
    return out


def sedative_exposure_from_inputevents(
    inputevents: pd.DataFrame,
    d_items: pd.DataFrame,
    windows: pd.DataFrame,
    lookback_hours: float = 12.0,
) -> pd.DataFrame:
    """Cumulative amount per drug class administered in ``[w_start - lookback, w_start)`` for each window.

    Infusion rows are apportioned to the lookback interval proportionally to the overlap of ``[starttime,
    endtime]`` with it (constant-rate assumption). Amount units are as charted (mg or mcg); the caller divides
    by weight when needed. Output columns: ``sed_<class>_<lookback>h`` merged on ``stay_id, window_idx``.
    """
    ids = resolve_sedative_itemids(d_items)
    ie = inputevents.copy()
    ie["starttime"] = pd.to_datetime(ie["starttime"])
    ie["endtime"] = pd.to_datetime(ie["endtime"])
    ie["amount"] = pd.to_numeric(ie["amount"], errors="coerce").fillna(0.0)
    lb = pd.Timedelta(hours=lookback_hours)
    rows = []
    for _, w in windows.iterrows():
        if pd.isna(w["w_start"]):
            continue
        lo, hi = pd.to_datetime(w["w_start"]) - lb, pd.to_datetime(w["w_start"])
        s = ie[(ie["stay_id"] == w["stay_id"]) & (ie["endtime"] > lo) & (ie["starttime"] < hi)]
        rec = {"stay_id": w["stay_id"], "window_idx": w["window_idx"]}
        for cls, cls_ids in ids.items():
            sc = s[s["itemid"].isin(cls_ids)]
            total = 0.0
            for st, en, amt in zip(sc["starttime"], sc["endtime"], sc["amount"]):
                dur = max((en - st).total_seconds(), 60.0)
                overlap = max((min(en, hi) - max(st, lo)).total_seconds(), 0.0)
                total += amt * overlap / dur
            rec[f"{SEDATION_COLS_PREFIX}{cls}_{int(lookback_hours)}h"] = total
        rows.append(rec)
    return pd.DataFrame(rows)


def rass_trajectory_features(windows: pd.DataFrame, n_prev: int = 3) -> pd.DataFrame:
    """RASS statistics over the previous ``n_prev`` windows (strictly before the current one).

    Produces ``sed_rass_mean_prev``, ``sed_rass_min_prev``, ``sed_rass_slope_prev`` (per window) and
    ``sed_rass_last``; NaN when no RASS was charted.
    """
    w = windows.sort_values(["stay_id", "window_idx"]).reset_index(drop=True)
    out = []
    for sid, g in w.groupby("stay_id", sort=False):
        means = g["rass_mean"].to_numpy(dtype=float)
        mins = g["rass_min"].to_numpy(dtype=float)
        for i, idx in enumerate(g["window_idx"]):
            prev_m = means[max(0, i - n_prev):i]
            prev_min = mins[max(0, i - n_prev):i]
            valid = prev_m[np.isfinite(prev_m)]
            slope = np.nan
            if valid.size >= 2:
                x = np.arange(valid.size)
                slope = float(np.polyfit(x, valid, 1)[0])
            out.append(
                {
                    "stay_id": sid, "window_idx": idx,
                    "sed_rass_mean_prev": float(valid.mean()) if valid.size else np.nan,
                    "sed_rass_min_prev": float(np.nanmin(prev_min)) if np.isfinite(prev_min).any() else np.nan,
                    "sed_rass_slope_prev": slope,
                    "sed_rass_last": float(valid[-1]) if valid.size else np.nan,
                }
            )
    return pd.DataFrame(out)


def assessment_process_features(windows: pd.DataFrame, window_hours: int = 12) -> pd.DataFrame:
    """Screening-process features computed from windows strictly before the current one.

    ``asm_hours_since_cam``, ``asm_prev_state`` (one-hot for delirium/coma/unscreened; normal is the reference),
    ``asm_frac_screened`` (fraction of previous windows with >= 1 CAM-ICU), ``asm_n_prev_delirium``.
    """
    w = windows.sort_values(["stay_id", "window_idx"]).reset_index(drop=True)
    out = []
    for sid, g in w.groupby("stay_id", sort=False):
        states = g["state"].tolist()
        n_cam = g["n_cam"].to_numpy()
        last_cam_idx = None
        for i, idx in enumerate(g["window_idx"]):
            prev_states = states[:i]
            rec = {
                "stay_id": sid, "window_idx": idx,
                "asm_hours_since_cam": np.nan if last_cam_idx is None else float((i - last_cam_idx) * window_hours),
                "asm_prev_delirium": int(bool(prev_states) and prev_states[-1] == "delirium"),
                "asm_prev_coma": int(bool(prev_states) and prev_states[-1] == "coma"),
                "asm_prev_unscreened": int(bool(prev_states) and prev_states[-1] == "unscreened"),
                "asm_frac_screened": float(np.mean(n_cam[:i] > 0)) if i > 0 else np.nan,
                "asm_n_prev_delirium": int(sum(s == "delirium" for s in prev_states)),
            }
            out.append(rec)
            if n_cam[i] > 0:
                last_cam_idx = i
    return pd.DataFrame(out)


def assemble_features(
    windows: pd.DataFrame,
    sedation: Optional[pd.DataFrame] = None,
    physiology: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge the feature blocks on ``stay_id, window_idx``. ``sedation`` columns not prefixed ``sed_`` are prefixed."""
    f = rass_trajectory_features(windows).merge(assessment_process_features(windows), on=["stay_id", "window_idx"], how="outer")
    if sedation is not None:
        s = sedation.copy()
        s.columns = [c if c in ("stay_id", "window_idx") or c.startswith(SEDATION_COLS_PREFIX) else f"{SEDATION_COLS_PREFIX}{c}" for c in s.columns]
        f = f.merge(s, on=["stay_id", "window_idx"], how="left")
    if physiology is not None:
        f = f.merge(physiology, on=["stay_id", "window_idx"], how="left")
    return f


def feature_blocks(columns: Iterable[str]) -> Dict[str, list]:
    """Split feature names into ``sedation``, ``assessment`` and ``physiology`` blocks by prefix."""
    cols = list(columns)
    sed = [c for c in cols if c.startswith(SEDATION_COLS_PREFIX)]
    asm = [c for c in cols if c.startswith(ASSESSMENT_COLS_PREFIX)]
    phys = [c for c in cols if c not in sed and c not in asm and c not in ("stay_id", "window_idx")]
    return {"sedation": sed, "assessment": asm, "physiology": phys}
