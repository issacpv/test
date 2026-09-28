"""Suspected-infection rules (Sepsis-3 operationalisations of "infection").

All times are hours relative to ICU admission (floats; negative = before ICU). Inputs are
long tables: ``antibiotics`` with columns (stay_id, time_h, drug) and ``cultures`` with
(stay_id, time_h, specimen). The rule returns the first suspicion time per stay (or NaN).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

RULES = ("culture_abx", "abx_first_only", "culture_first_only", "abx_only", "abx_2plus")


@dataclass(frozen=True)
class SIRule:
    rule: str = "culture_abx"        # one of RULES
    abx_first_window_h: float = 24.0  # antibiotic first: culture within this many hours after
    culture_first_window_h: float = 72.0  # culture first: antibiotic within this many hours after
    suspicion_time: str = "earlier"  # "earlier" (min of pair) | "antibiotic" | "culture"
    repeat_window_h: float = 96.0     # for abx_2plus: second dose of same drug within this window
    min_time_h: Optional[float] = None  # ignore events before this (e.g. -48 h) if set
    max_time_h: Optional[float] = None  # ignore events after this (e.g. ICU discharge) if set

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule}")


def _filter_time(df: pd.DataFrame, rule: SIRule) -> pd.DataFrame:
    if rule.min_time_h is not None:
        df = df[df["time_h"] >= rule.min_time_h]
    if rule.max_time_h is not None:
        df = df[df["time_h"] <= rule.max_time_h]
    return df


def _pair_times(abx: pd.DataFrame, cx: pd.DataFrame, rule: SIRule) -> pd.Series:
    """First suspicion time per stay from antibiotic-culture pairs under asymmetric windows."""
    out: Dict = {}
    abx_g = {k: v["time_h"].to_numpy(float) for k, v in abx.groupby("stay_id")}
    cx_g = {k: v["time_h"].to_numpy(float) for k, v in cx.groupby("stay_id")}
    for sid in set(abx_g) & set(cx_g):
        a = np.sort(abx_g[sid])
        c = np.sort(cx_g[sid])
        diff = c[None, :] - a[:, None]  # culture time - antibiotic time
        ok_abx_first = (diff >= 0) & (diff <= rule.abx_first_window_h)
        ok_cx_first = (diff < 0) & (-diff <= rule.culture_first_window_h)
        if rule.rule == "abx_first_only":
            ok = ok_abx_first
        elif rule.rule == "culture_first_only":
            ok = ok_cx_first
        else:
            ok = ok_abx_first | ok_cx_first
        ia, ic = np.nonzero(ok)
        if len(ia) == 0:
            continue
        if rule.suspicion_time == "antibiotic":
            t = a[ia]
        elif rule.suspicion_time == "culture":
            t = c[ic]
        else:
            t = np.minimum(a[ia], c[ic])
        out[sid] = float(t.min())
    return pd.Series(out, dtype=float)


def suspected_infection_times(antibiotics: pd.DataFrame, cultures: Optional[pd.DataFrame], rule: SIRule) -> pd.Series:
    """First suspicion-of-infection time (hours from ICU admission) per stay under ``rule``.

    - culture_abx / abx_first_only / culture_first_only: Seymour-style pairing.
    - abx_only: first qualifying antibiotic administration (eICU fallback when culture timing is sparse).
    - abx_2plus: first antibiotic with a repeat administration of the same drug within ``repeat_window_h``.
    """
    abx = _filter_time(antibiotics, rule)
    if rule.rule == "abx_only":
        return abx.groupby("stay_id")["time_h"].min().astype(float)
    if rule.rule == "abx_2plus":
        out = {}
        for sid, g in abx.groupby("stay_id"):
            best = np.inf
            for _, d in g.groupby("drug"):
                t = np.sort(d["time_h"].to_numpy(float))
                if len(t) < 2:
                    continue
                gaps = t[1:] - t[:-1]
                idx = np.nonzero(gaps <= rule.repeat_window_h)[0]
                if len(idx):
                    best = min(best, t[idx[0]])
            if np.isfinite(best):
                out[sid] = best
        return pd.Series(out, dtype=float)
    if cultures is None:
        raise ValueError(f"rule {rule.rule} needs a cultures table")
    return _pair_times(abx, _filter_time(cultures, rule), rule)


def filter_antibiotics(antibiotics: pd.DataFrame, allowed: Optional[set], exclude_prophylaxis: bool = True,
                       prophylaxis_routes: tuple = ("topical", "ophthalmic", "otic", "inhalation")) -> pd.DataFrame:
    """Restrict antibiotic administrations to an allowed drug list (lower-cased) and systemic routes."""
    df = antibiotics.copy()
    df["drug_l"] = df["drug"].astype(str).str.lower()
    if allowed is not None:
        allowed_l = {a.lower() for a in allowed}
        df = df[df["drug_l"].apply(lambda d: any(a in d for a in allowed_l))]
    if exclude_prophylaxis and "route" in df:
        df = df[~df["route"].astype(str).str.lower().isin(prophylaxis_routes)]
    return df.drop(columns=["drug_l"])
