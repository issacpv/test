"""Device keys, recall linkage, left-truncated time-to-event construction and lag decomposition."""

from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

LEGAL_SUFFIXES = r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|gmbh|ag|sa|sas|srl|bv|nv|plc|pty|kk|div|division)\b\.?"
STOP = {"the", "of", "and", "medical", "systems", "system", "usa", "us", "international", "worldwide", "group", "holdings", "healthcare", "health"}


def normalize_firm(name: object, aliases: Optional[Dict[str, str]] = None) -> str:
    """Lower-case, strip punctuation / legal suffixes / generic tokens; apply an optional alias table."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    t = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    t = re.sub(LEGAL_SUFFIXES, " ", t)
    t = " ".join(w for w in t.split() if w not in STOP)
    if aliases:
        t = aliases.get(t, t)
    return t


def device_key(product_code: object, firm: object, aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """``"<PRODUCT_CODE>|<firm_norm>"`` or None when either part is missing."""
    code = str(product_code or "").strip().upper()
    f = normalize_firm(firm, aliases)
    if not code or code == "NAN" or not f:
        return None
    return f"{code}|{f}"


def add_keys(df: pd.DataFrame, code_col: str, firm_col: str, aliases: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    out = df.copy()
    out["key"] = [device_key(c, f, aliases) for c, f in zip(out[code_col], out[firm_col])]
    return out


def recall_table(recalls: pd.DataFrame, enforcement: Optional[pd.DataFrame] = None, aliases: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per recall event with key, initiation/posting/classification dates and class.

    ``recalls`` are flattened ``device/recall`` rows; ``enforcement`` rows (optional) add ``classification`` and
    ``center_classification`` joined on ``res_event_number`` (first match).
    """
    r = add_keys(recalls, "product_code", "recalling_firm", aliases)
    if enforcement is not None and len(enforcement):
        e = enforcement.drop_duplicates("res_event_number")[["res_event_number", "classification", "center_classification", "initiation"]]
        r = r.merge(e, on="res_event_number", how="left", suffixes=("", "_enf"))
        r["initiated"] = r["initiated"].fillna(r["initiation"])
    r["software_root_cause"] = r["root_cause"].astype(str).str.contains(r"software|firmware", case=False, regex=True, na=False)
    return r


def build_time_to_event(
    first_events: pd.DataFrame,
    recalls: pd.DataFrame,
    data_end: pd.Timestamp,
    origin_col: str = "first_serious",
    reliable_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Left-truncated, right-censored time-to-recall table per key.

    * origin = ``first_events[origin_col]`` (first serious report by default);
    * event = first recall initiated **after** the origin; time = years from origin to initiation;
    * censored at ``data_end`` otherwise;
    * ``entry_years`` = years from origin to ``reliable_start`` when the origin precedes the start of reliable
      recall records (left truncation; 0 otherwise). Keys whose entry >= time are dropped as unobservable;
    * keys whose only recall precedes the origin are returned with ``status == "recall_before_origin"`` and
      excluded from the analysis set (``analysis == False``).
    """
    rc = recalls.dropna(subset=["key", "initiated"]).sort_values("initiated")
    rows = []
    for _, f in first_events.dropna(subset=[origin_col]).iterrows():
        origin = pd.Timestamp(f[origin_col])
        mine = rc[rc["key"] == f["key"]]
        after = mine[mine["initiated"] > origin]
        before = mine[mine["initiated"] <= origin]
        if len(after):
            t_end, event, status = after["initiated"].iloc[0], 1, "recalled"
        elif len(before):
            t_end, event, status = pd.Timestamp(data_end), 0, "recall_before_origin"
        else:
            t_end, event, status = pd.Timestamp(data_end), 0, "censored"
        time_years = (t_end - origin).days / 365.25
        entry = 0.0
        if reliable_start is not None and origin < pd.Timestamp(reliable_start):
            entry = (pd.Timestamp(reliable_start) - origin).days / 365.25
        rows.append({"key": f["key"], "origin": origin, "end": t_end, "time_years": time_years, "event": event, "entry_years": entry, "status": status,
                     "analysis": status != "recall_before_origin" and time_years > entry and time_years > 0})
    return pd.DataFrame(rows)


def lag_components(reports: pd.DataFrame, recall_row: pd.Series, signal_month: Optional[pd.Period] = None) -> Dict[str, float]:
    """Median lag components (days) for one recalled key.

    * ``event_to_receipt``: median ``date_received - date_of_event`` over reports received before initiation;
    * ``first_serious_to_initiation``; ``signal_to_initiation`` (if a CUSUM signal month is given);
    * ``initiation_to_classification``, ``initiation_to_posting`` from the recall row.
    """
    init = pd.Timestamp(recall_row["initiated"])
    pre = reports[pd.to_datetime(reports["date_received"]) < init]
    dr = (pd.to_datetime(pre["date_received"]) - pd.to_datetime(pre["date_of_event"])).dt.days
    dr = dr[dr >= 0]
    ser = pre[pre["serious"].astype(bool)]
    out = {
        "n_reports_pre": int(len(pre)),
        "event_to_receipt_median_days": float(dr.median()) if len(dr) else np.nan,
        "first_serious_to_initiation_days": float((init - pd.to_datetime(ser["date_received"]).min()).days) if len(ser) else np.nan,
        "signal_to_initiation_days": float((init - signal_month.to_timestamp()).days) if signal_month is not None else np.nan,
        "initiation_to_classification_days": float((pd.Timestamp(recall_row["center_classification"]) - init).days) if pd.notna(recall_row.get("center_classification", pd.NaT)) else np.nan,
        "initiation_to_posting_days": float((pd.Timestamp(recall_row["posted"]) - init).days) if pd.notna(recall_row.get("posted", pd.NaT)) else np.nan,
    }
    return out
