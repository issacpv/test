"""Surgical encounter definitions and outcome labels.

The canonical encounter table has one row per surgery with columns
``op_id, subject_id, hadm_id, op_start, op_end, admit_time, discharge_time,
death_time, site``.  Times are pandas Timestamps (MIMIC, MOVER) or minutes as
floats (INSPIRE); every function below only ever subtracts times, so both
work as long as ``timedelta_hours`` is used consistently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SURGICAL_SERVICES: frozenset[str] = frozenset({"CSURG", "NSURG", "ORTHO", "SURG", "TSURG", "VSURG", "GU", "GYN", "ENT", "PSURG"})
PLANNED_ICU_SERVICES: frozenset[str] = frozenset({"CSURG", "NSURG"})


def timedelta_hours(later, earlier) -> pd.Series:
    """Hours between two time columns; works for Timestamps and for numeric minutes."""
    later, earlier = pd.Series(later), pd.Series(earlier)
    if pd.api.types.is_datetime64_any_dtype(later) or pd.api.types.is_datetime64_any_dtype(earlier):
        return (pd.to_datetime(later) - pd.to_datetime(earlier)).dt.total_seconds() / 3600.0
    return (later.astype(float) - earlier.astype(float)) / 60.0


def is_surgical_icd9_procedure(code: str) -> bool:
    """ICD-9-CM procedure chapters 01-86 are operative; 87-99 are diagnostic/therapeutic non-operative."""
    c = str(code).replace(".", "")
    return c[:2].isdigit() and 1 <= int(c[:2]) <= 86


def is_surgical_icd10pcs(code: str) -> bool:
    """ICD-10-PCS section 0 (Medical and Surgical) excluding inspection (J), drainage (9) and mapping (K) root operations."""
    c = str(code).replace(".", "").upper()
    return len(c) == 7 and c[0] == "0" and c[2] not in {"J", "9", "K"}


def mimic_surgical_definitions(transfers: pd.DataFrame, procedures_icd: pd.DataFrame, services: pd.DataFrame) -> pd.DataFrame:
    """Per-hadm_id boolean columns for the MIMIC-IV surgical-cohort multiverse.

    D1 pacu: any transfer with careunit == 'PACU'
    D2 proc: any operative ICD-9 / ICD-10-PCS procedure code
    D3 svc : any surgical service
    D4 = D1 & D2, D5 = D1 | D2, D6 = D2 & D3.
    Also returns ``surgery_time`` = first PACU intime (D1) else first procedure chartdate.
    """
    pacu = transfers[transfers["careunit"].astype(str).str.upper().eq("PACU")]
    pacu_first = pacu.groupby("hadm_id")["intime"].min().rename("pacu_time")
    p = procedures_icd.copy()
    p["surgical"] = np.where(
        p["icd_version"].astype(int) == 9,
        p["icd_code"].map(is_surgical_icd9_procedure),
        p["icd_code"].map(is_surgical_icd10pcs),
    )
    proc_any = p.groupby("hadm_id")["surgical"].any().rename("proc")
    proc_time = p[p["surgical"]].groupby("hadm_id")["chartdate"].min().rename("proc_time")
    svc_any = services.assign(s=services["curr_service"].isin(SURGICAL_SERVICES)).groupby("hadm_id")["s"].any().rename("svc")
    ids = pd.Index(sorted(set(transfers["hadm_id"]) | set(procedures_icd["hadm_id"]) | set(services["hadm_id"])), name="hadm_id")
    out = pd.DataFrame(index=ids)
    out["pacu"] = out.index.isin(pacu_first.index)
    out = out.join(proc_any).join(svc_any).join(pacu_first).join(proc_time)
    out[["proc", "svc"]] = out[["proc", "svc"]].fillna(False).astype(bool)
    out["pacu_and_proc"] = out["pacu"] & out["proc"]
    out["pacu_or_proc"] = out["pacu"] | out["proc"]
    out["proc_and_svc"] = out["proc"] & out["svc"]
    out["surgery_time"] = out["pacu_time"].where(out["pacu_time"].notna(), out["proc_time"])
    return out.reset_index()


DEFINITIONS: tuple[str, ...] = ("pacu", "proc", "svc", "pacu_and_proc", "pacu_or_proc", "proc_and_svc")


def definition_multiverse(defs: pd.DataFrame, outcome: pd.Series | None = None) -> pd.DataFrame:
    """Cohort size, outcome prevalence and pairwise Jaccard overlap for each definition."""
    rows = []
    for d in DEFINITIONS:
        m = defs[d].to_numpy(bool)
        row = {"definition": d, "n": int(m.sum())}
        if outcome is not None:
            y = outcome.reindex(defs["hadm_id"]).to_numpy(float)
            row["prevalence"] = float(np.nanmean(y[m])) if m.sum() else np.nan
        for e in DEFINITIONS:
            me = defs[e].to_numpy(bool)
            union = (m | me).sum()
            row[f"jaccard_{e}"] = float((m & me).sum() / union) if union else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def label_mortality(enc: pd.DataFrame, window_days: float = 30.0) -> pd.Series:
    """In-hospital death within ``window_days`` of surgery end (death_time NaN = alive)."""
    h = timedelta_hours(enc["death_time"], enc["op_end"])
    return ((h >= 0) & (h <= 24.0 * window_days)).fillna(False).astype(int).rename("mortality")


def label_unplanned_icu(enc: pd.DataFrame, icu: pd.DataFrame, window_h: float = 24.0, planned_services: frozenset[str] = PLANNED_ICU_SERVICES) -> pd.Series:
    """ICU admission within ``window_h`` after surgery end, excluding planned-ICU services and patients already in ICU at surgery start.

    ``icu`` has columns hadm_id, intime, outtime.  ``enc`` may have a ``service`` column.
    """
    m = enc[["op_id", "hadm_id", "op_start", "op_end"]].merge(icu[["hadm_id", "intime", "outtime"]], on="hadm_id", how="left")
    h_in = timedelta_hours(m["intime"], m["op_end"])
    in_icu_before = (timedelta_hours(m["op_start"], m["intime"]) >= 0) & (timedelta_hours(m["outtime"], m["op_start"]) > 0)
    post = (h_in >= 0) & (h_in <= window_h)
    m["post_icu"] = post.fillna(False)
    m["pre_icu"] = in_icu_before.fillna(False)
    g = m.groupby("op_id")[["post_icu", "pre_icu"]].any()
    lab = (g["post_icu"] & ~g["pre_icu"]).astype(int)
    if "service" in enc.columns:
        planned = enc.set_index("op_id")["service"].isin(planned_services)
        lab[planned.reindex(lab.index).fillna(False).to_numpy(bool)] = 0
    return lab.reindex(enc["op_id"]).fillna(0).astype(int).rename("unplanned_icu")


def kdigo_stage(cr: float, baseline: float, min48: float | None) -> int:
    """KDIGO creatinine stage: 1 if rise >= 0.3 mg/dL in 48 h or 1.5-1.9x baseline; 2 if 2.0-2.9x; 3 if >= 3x or >= 4.0 mg/dL."""
    if not np.isfinite(cr) or not np.isfinite(baseline) or baseline <= 0:
        return 0
    ratio = cr / baseline
    if ratio >= 3.0 or cr >= 4.0:
        return 3
    if ratio >= 2.0:
        return 2
    if ratio >= 1.5 or (min48 is not None and np.isfinite(min48) and cr - min48 >= 0.3):
        return 1
    return 0


def label_aki_kdigo(enc: pd.DataFrame, creatinine: pd.DataFrame, pre_days: float = 7.0, pre_fallback_days: float = 30.0, post_days: float = 7.0) -> pd.DataFrame:
    """Postoperative AKI by KDIGO creatinine criteria.

    ``creatinine``: columns subject_id, charttime, value (mg/dL).  Baseline = lowest
    value in the ``pre_days`` before surgery start, else the last value within
    ``pre_fallback_days``.  Returns per op_id: baseline, max stage in the
    ``post_days`` window, n_post (postoperative measurements) and ``aki`` (stage >= 1).
    Encounters with no postoperative creatinine get stage NaN (analysed as missing, not 0).
    """
    rows = []
    cr_by_subj = {s: g.sort_values("charttime") for s, g in creatinine.groupby("subject_id")}
    for _, e in enc.iterrows():
        g = cr_by_subj.get(e["subject_id"])
        if g is None:
            rows.append({"op_id": e["op_id"], "baseline": np.nan, "stage": np.nan, "n_post": 0})
            continue
        h = timedelta_hours(g["charttime"], pd.Series([e["op_start"]] * len(g), index=g.index)).to_numpy()
        vals = g["value"].to_numpy(float)
        pre = (h < 0) & (h >= -24 * pre_days)
        if pre.any():
            base = float(np.nanmin(vals[pre]))
        else:
            fb = (h < 0) & (h >= -24 * pre_fallback_days)
            base = float(vals[fb][-1]) if fb.any() else np.nan
        post = (h >= 0) & (h <= 24 * post_days)
        n_post = int(post.sum())
        if n_post == 0 or not np.isfinite(base):
            rows.append({"op_id": e["op_id"], "baseline": base, "stage": np.nan, "n_post": n_post})
            continue
        stage = 0
        idx_post = np.where(post)[0]
        for i in idx_post:
            win48 = (h >= h[i] - 48) & (h < h[i])
            min48 = float(np.nanmin(vals[win48])) if win48.any() else None
            stage = max(stage, kdigo_stage(vals[i], base, min48))
        rows.append({"op_id": e["op_id"], "baseline": base, "stage": stage, "n_post": n_post})
    out = pd.DataFrame(rows)
    out["aki"] = out["stage"].ge(1).where(out["stage"].notna(), np.nan)
    return out
