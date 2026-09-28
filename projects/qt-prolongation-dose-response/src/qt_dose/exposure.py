"""Normalise eMAR administration events into canonical drug/dose exposures.

The MIMIC-IV ``emar`` / ``emar_detail`` tables record barcode medication
administrations with free-text drug strings, doses and units. This module maps
those to a small pre-registered set of QT-relevant ingredients (plus negative
controls), canonicalises dose to milligrams, and returns a tidy administration
table keyed by (subject_id, ingredient, charttime, dose_mg, route).

The ingredient dictionary here is intentionally small and explicit so it can be
audited; a production run would map against the full RxNorm ingredient set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Pre-registered QT-drug list (CredibleMeds "known risk") + negative controls.
# Keys are canonical ingredient names; values are lowercase match substrings.
INGREDIENT_SYNONYMS: dict[str, list[str]] = {
    "sotalol": ["sotalol"],
    "dofetilide": ["dofetilide", "tikosyn"],
    "amiodarone": ["amiodarone", "cordarone", "pacerone"],
    "haloperidol": ["haloperidol", "haldol"],
    "methadone": ["methadone"],
    "azithromycin": ["azithromycin", "zithromax"],
    "ondansetron": ["ondansetron", "zofran"],
    "citalopram": ["citalopram", "celexa"],
    "quetiapine": ["quetiapine", "seroquel"],
    "ciprofloxacin": ["ciprofloxacin", "cipro"],
    # negative controls (no known QT risk)
    "aspirin": ["aspirin", "acetylsalicylic"],
    "metoprolol": ["metoprolol", "lopressor", "toprol"],
    "acetaminophen": ["acetaminophen", "paracetamol", "tylenol"],
}

QT_DRUGS = [
    "sotalol", "dofetilide", "amiodarone", "haloperidol", "methadone",
    "azithromycin", "ondansetron", "citalopram", "quetiapine", "ciprofloxacin",
]
NEGATIVE_CONTROL_DRUGS = ["aspirin", "metoprolol", "acetaminophen"]

# Unit conversion to milligrams.
_UNIT_TO_MG = {"mg": 1.0, "g": 1000.0, "mcg": 1e-3, "ug": 1e-3, "µg": 1e-3}


@dataclass(frozen=True)
class Administration:
    subject_id: int
    ingredient: str
    charttime: pd.Timestamp
    dose_mg: float
    route: str


def map_ingredient(drug_text: str) -> str | None:
    """Map a free-text drug string to a canonical ingredient, or None."""
    if not isinstance(drug_text, str):
        return None
    t = drug_text.lower()
    for ingredient, subs in INGREDIENT_SYNONYMS.items():
        if any(s in t for s in subs):
            return ingredient
    return None


def parse_dose_mg(dose_val, dose_unit: str) -> float:
    """Convert a (value, unit) dose to milligrams; NaN if not parseable."""
    try:
        v = float(dose_val)
    except (TypeError, ValueError):
        # sometimes dose is embedded in text like "40 mg"
        m = re.search(r"([-+]?\d*\.?\d+)", str(dose_val))
        if not m:
            return float("nan")
        v = float(m.group(1))
    unit = str(dose_unit).strip().lower() if dose_unit is not None else ""
    factor = _UNIT_TO_MG.get(unit)
    if factor is None:
        return float("nan")
    return v * factor


def build_exposures(emar: pd.DataFrame, emar_detail: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a tidy administration table with canonical ingredient and dose_mg.

    Parameters
    ----------
    emar : DataFrame
        Columns at least: subject_id, charttime, medication, event_txt, emar_id.
    emar_detail : DataFrame, optional
        Columns at least: emar_id, dose_given, dose_given_unit, route.

    Only administered events (``event_txt`` containing 'Administered') are kept.
    """
    df = emar.copy()
    if "event_txt" in df.columns:
        df = df[df["event_txt"].astype(str).str.contains("Administered", case=False, na=False)]
    df["ingredient"] = df["medication"].map(map_ingredient)
    df = df[df["ingredient"].notna()].copy()

    if emar_detail is not None and "emar_id" in df.columns:
        det = emar_detail[["emar_id", "dose_given", "dose_given_unit"]].copy()
        det = det.rename(columns={"dose_given": "_dose", "dose_given_unit": "_unit"})
        route = emar_detail[["emar_id", "route"]] if "route" in emar_detail.columns else None
        df = df.merge(det, on="emar_id", how="left")
        if route is not None:
            df = df.merge(route, on="emar_id", how="left")
        df["dose_mg"] = [parse_dose_mg(d, u) for d, u in zip(df["_dose"], df["_unit"])]
    else:
        df["dose_mg"] = np.nan
        df["route"] = df.get("route", "")

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    out = df[["subject_id", "ingredient", "charttime", "dose_mg"]].copy()
    out["route"] = df.get("route", pd.Series([""] * len(df), index=df.index)).fillna("")
    out = out.dropna(subset=["charttime"]).reset_index(drop=True)
    return out.sort_values(["subject_id", "ingredient", "charttime"]).reset_index(drop=True)
