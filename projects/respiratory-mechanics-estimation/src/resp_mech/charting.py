"""Charting-label mapping, pivoting, unit harmonisation, flow proxy and ventilation episodes.

Item identifiers differ between MIMIC-IV (``d_items.label``), eICU
(``respchartvaluelabel``) and HiRID (``hirid_variable_reference``). Rather
than hard-coding ids, labels are matched by regular expressions to canonical
variables; the resulting map must be audited by hand per database.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .single_compartment import inspiratory_flow_l_s, ti_from_rr_ie

LABEL_PATTERNS: dict[str, list[str]] = {
    "ppeak": [r"peak\s*insp", r"peak\s*(airway\s*)?pressure", r"\bppeak\b", r"\bpip\b", r"peak\s*pressure"],
    "pplat": [r"plateau"],
    "peep": [r"\bpeep\b(?!.*auto)", r"peep\s*set", r"total\s*peep"],
    "vt_obs": [r"tidal\s*volume\s*\(observed\)", r"exhaled\s*tv", r"tidal\s*volume\s*exp", r"\bvt\b.*exp", r"expiratory\s*tidal"],
    "vt_set": [r"tidal\s*volume\s*\(set\)", r"set\s*tidal", r"\bvt\s*set"],
    "rr": [r"respiratory\s*rate\s*\(total\)", r"total\s*rr", r"\bvent\s*rate", r"resp(iratory)?\s*rate.*(total|vent)", r"\brr\s*total"],
    "rr_set": [r"respiratory\s*rate\s*\(set\)", r"\brr\s*set", r"set\s*rate"],
    "fio2": [r"fio2", r"inspired\s*o2", r"o2\s*fraction"],
    "ti": [r"inspiratory\s*time", r"\bti\b", r"insp\.?\s*time"],
    "ie": [r"i\s*:\s*e", r"\bie\s*ratio", r"i/e"],
    "flow": [r"peak\s*insp.*flow", r"inspiratory\s*flow", r"\bflow\s*rate"],
    "mode": [r"ventilator\s*mode", r"vent\s*mode", r"\bmode\b"],
    "compliance_charted": [r"compliance"],
    "resistance_charted": [r"resistance"],
    "mean_airway": [r"mean\s*airway"],
}
_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in LABEL_PATTERNS.items()}

CONTROLLED_MODE_RE = re.compile(r"\b(cmv|a/?c|assist|vc|pc(v)?|prvc|simv|apv|bipap\s*asb|volume\s*control|pressure\s*control)\b", re.IGNORECASE)
SPONTANEOUS_MODE_RE = re.compile(r"\b(psv?|pressure\s*support|cpap|spont|asb|nava|aprv)\b", re.IGNORECASE)


def map_labels(labels: Iterable[str]) -> dict[str, str]:
    """Map free-text charting labels to canonical variable names (first matching pattern wins).

    Ordering of ``LABEL_PATTERNS`` matters: e.g. 'Plateau Pressure' matches
    ``pplat`` before the generic 'pressure' pattern of ``ppeak`` because the
    ``ppeak`` patterns require 'peak'.
    """
    out: dict[str, str] = {}
    for lab in labels:
        if not isinstance(lab, str):
            continue
        for canon, pats in _COMPILED.items():
            if any(p.search(lab) for p in pats):
                out[lab] = canon
                break
    return out


def pivot_charting(rows: pd.DataFrame, label_map: dict[str, str], stay_col: str = "stay_id", time_col: str = "time", label_col: str = "label", value_col: str = "value") -> pd.DataFrame:
    """Long charting rows -> wide table (one row per stay/time, canonical columns; duplicates -> median).

    Non-numeric values (e.g. mode strings) are kept in a separate ``mode`` column.
    """
    df = rows.copy()
    df["canon"] = df[label_col].map(label_map)
    df = df[df["canon"].notna()]
    modes = df[df["canon"] == "mode"][[stay_col, time_col, value_col]].rename(columns={value_col: "mode"})
    num = df[df["canon"] != "mode"].copy()
    num[value_col] = pd.to_numeric(num[value_col], errors="coerce")
    num = num.dropna(subset=[value_col])
    wide = num.pivot_table(index=[stay_col, time_col], columns="canon", values=value_col, aggfunc="median")
    wide.columns.name = None
    wide = wide.reset_index()
    if len(modes):
        modes = modes.drop_duplicates([stay_col, time_col], keep="last")
        wide = wide.merge(modes, on=[stay_col, time_col], how="outer")
    return wide.sort_values([stay_col, time_col]).reset_index(drop=True)


def harmonize_units(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert tidal volumes charted in litres to mL and flows in L/min to L/s (heuristic by magnitude)."""
    out = wide.copy()
    for col in ("vt_obs", "vt_set"):
        if col in out and out[col].notna().any() and np.nanmedian(out[col]) < 3.0:
            out[col] = out[col] * 1000.0
    if "flow" in out and out["flow"].notna().any() and np.nanmedian(out["flow"]) > 5.0:
        out["flow"] = out["flow"] / 60.0
    if "fio2" in out and out["fio2"].notna().any() and np.nanmedian(out["fio2"]) > 1.0:
        out["fio2"] = out["fio2"] / 100.0
    return out


def derive_flow_proxy(wide: pd.DataFrame) -> pd.DataFrame:
    """Add ``vt_ml`` and ``flow_l_s`` columns using charted flow, else VT/Ti, else VT and RR + I:E."""
    out = wide.copy()
    vt = out["vt_obs"] if "vt_obs" in out else pd.Series(np.nan, index=out.index)
    if "vt_set" in out:
        vt = vt.fillna(out["vt_set"])
    out["vt_ml"] = vt
    flow = out["flow"].copy() if "flow" in out else pd.Series(np.nan, index=out.index)
    ti = out["ti"].copy() if "ti" in out else pd.Series(np.nan, index=out.index)
    if "rr" in out and "ie" in out:
        ti_ie = [ti_from_rr_ie(r, e) if np.isfinite(r) and np.isfinite(e) else np.nan for r, e in zip(out["rr"], out["ie"])]
        ti = ti.fillna(pd.Series(ti_ie, index=out.index))
    proxy = [inspiratory_flow_l_s(v, t) if np.isfinite(v) and np.isfinite(t) else np.nan for v, t in zip(out["vt_ml"], ti)]
    out["flow_l_s"] = flow.fillna(pd.Series(proxy, index=out.index))
    return out


def classify_mode(mode: str | float) -> str:
    """'controlled', 'spontaneous' or 'unknown' from a free-text ventilator mode."""
    if not isinstance(mode, str):
        return "unknown"
    if SPONTANEOUS_MODE_RE.search(mode) and not CONTROLLED_MODE_RE.search(mode):
        return "spontaneous"
    if CONTROLLED_MODE_RE.search(mode):
        return "controlled"
    return "unknown"


def ventilation_episodes(times_h: np.ndarray, max_gap_h: float = 4.0, min_duration_h: float = 6.0) -> list[tuple[float, float]]:
    """Group charting times (hours) into ventilation episodes separated by gaps > ``max_gap_h``."""
    t = np.sort(np.asarray(times_h, float))
    if len(t) == 0:
        return []
    episodes = []
    start = t[0]
    prev = t[0]
    for x in t[1:]:
        if x - prev > max_gap_h:
            if prev - start >= min_duration_h:
                episodes.append((float(start), float(prev)))
            start = x
        prev = x
    if prev - start >= min_duration_h:
        episodes.append((float(start), float(prev)))
    return episodes
