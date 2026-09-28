"""Resolve chartevents item ids from d_items labels and reconstruct Braden totals."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ITEM_CONCEPTS: dict[str, str] = {
    "braden_sensory": r"^braden\s*sensory",
    "braden_moisture": r"^braden\s*moisture",
    "braden_activity": r"^braden\s*activity",
    "braden_mobility": r"^braden\s*mobility",
    "braden_nutrition": r"^braden\s*nutrition",
    "braden_friction": r"^braden\s*friction",
    "skin_pi_stage": r"(pressure\s*(ulcer|injury)|impaired\s*skin).*stage",
    "skin_pi_site": r"(pressure\s*(ulcer|injury)|impaired\s*skin).*(site|location)",
    "pressure_relief": r"pressure\s*(reducing|relie[fv])|specialty\s*bed|air\s*mattress|support\s*surface",
    "reposition": r"\bturn(ed|ing)?\b|reposition",
    "rass": r"richmond|^rass\b",
    "fall_risk": r"\bfall",
}
BRADEN_CONCEPTS: tuple[str, ...] = ("braden_sensory", "braden_moisture", "braden_activity", "braden_mobility", "braden_nutrition", "braden_friction")
# Expected MetaVision ids in MIMIC-IV (verify on each release; the resolver refuses mismatches).
EXPECTED_BRADEN_ITEMIDS: dict[str, int] = {
    "braden_sensory": 224054, "braden_moisture": 224055, "braden_activity": 224056,
    "braden_mobility": 224057, "braden_nutrition": 224058, "braden_friction": 224059,
}
BRADEN_RANGES: dict[str, tuple[int, int]] = {c: (1, 4) for c in BRADEN_CONCEPTS}
BRADEN_RANGES["braden_friction"] = (1, 3)


def resolve_items(d_items: pd.DataFrame) -> pd.DataFrame:
    """Match d_items labels against ITEM_CONCEPTS. Returns concept, itemid, label (one row per match)."""
    rows = []
    lab = d_items["label"].fillna("").astype(str)
    for concept, pattern in ITEM_CONCEPTS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        for itemid, label in zip(d_items["itemid"], lab):
            if rx.search(label.strip()):
                rows.append({"concept": concept, "itemid": int(itemid), "label": label})
    return pd.DataFrame(rows, columns=["concept", "itemid", "label"])


def verify_expected(resolved: pd.DataFrame, expected: dict[str, int]) -> list[str]:
    """Check that each expected concept resolved to exactly the expected id. Returns a list of problems."""
    problems = []
    for concept, itemid in expected.items():
        ids = set(resolved.loc[resolved["concept"] == concept, "itemid"])
        if not ids:
            problems.append(f"{concept}: no item resolved (expected {itemid})")
        elif itemid not in ids:
            problems.append(f"{concept}: resolved {sorted(ids)}, expected {itemid}")
        elif len(ids) > 1:
            problems.append(f"{concept}: multiple items resolved {sorted(ids)}")
    return problems


def braden_table(chartevents: pd.DataFrame, resolved: pd.DataFrame, carry_h: float = 1.0) -> pd.DataFrame:
    """Wide Braden table per (stay_id, charttime) with subscales and total.

    ``chartevents`` columns: stay_id, charttime, itemid, valuenum.  Subscale values are
    carried forward within ``carry_h`` hours (components are usually charted together
    but occasionally minutes apart).  ``braden_total`` is set only when all six
    subscales are available; out-of-range values are dropped.
    """
    m = resolved[resolved["concept"].isin(BRADEN_CONCEPTS)][["itemid", "concept"]]
    ce = chartevents.merge(m, on="itemid", how="inner")[["stay_id", "charttime", "concept", "valuenum"]].copy()
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    for c, (lo, hi) in BRADEN_RANGES.items():
        bad = (ce["concept"] == c) & ~ce["valuenum"].between(lo, hi)
        ce = ce[~bad]
    wide = ce.pivot_table(index=["stay_id", "charttime"], columns="concept", values="valuenum", aggfunc="last").reset_index()
    wide = wide.sort_values(["stay_id", "charttime"])
    for c in BRADEN_CONCEPTS:
        if c not in wide.columns:
            wide[c] = np.nan
        obs_t = wide["charttime"].where(wide[c].notna())
        last_t = obs_t.groupby(wide["stay_id"]).ffill()
        val = wide[c].groupby(wide["stay_id"]).ffill()
        gap_h = (wide["charttime"] - last_t).dt.total_seconds() / 3600.0
        wide[c] = val.where(gap_h <= carry_h)
    wide["braden_total"] = wide[list(BRADEN_CONCEPTS)].sum(axis=1, min_count=len(BRADEN_CONCEPTS))
    return wide.reset_index(drop=True)


def assessment_frequency(braden: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """Per-stay number of complete Braden assessments and assessments per 24 h of ICU stay."""
    st = stays[["stay_id", "intime", "outtime"]].copy()
    st["los_h"] = (pd.to_datetime(st["outtime"]) - pd.to_datetime(st["intime"])).dt.total_seconds() / 3600.0
    n = braden.dropna(subset=["braden_total"]).groupby("stay_id").size().rename("n_assessments")
    out = st.merge(n, on="stay_id", how="left").fillna({"n_assessments": 0})
    out["assessments_per_24h"] = out["n_assessments"] / (out["los_h"] / 24.0).clip(lower=1 / 24)
    return out
