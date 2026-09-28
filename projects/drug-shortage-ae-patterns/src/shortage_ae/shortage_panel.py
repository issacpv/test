"""Exposure episodes, ingredient-name normalisation, outcome groups and the drug x month panel.

The unit of analysis is the (ingredient, calendar month) cell. Exposure is a 0/1 "in shortage"
indicator derived from the FDA (or ASHP) shortage list; outcomes are FAERS report counts in
MedDRA-defined groups. All functions are pure pandas and unit-testable offline.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- outcome groups
#: MedDRA PTs from the HLGT "Medication errors and other product use errors and issues" that
#: plausibly capture substitution / handling errors. Freeze at the analysis date (MedDRA v27).
MEDICATION_ERROR_PTS: Tuple[str, ...] = (
    "Wrong drug administered",
    "Wrong product administered",
    "Product substitution issue",
    "Drug administration error",
    "Incorrect dose administered",
    "Incorrect route of product administration",
    "Wrong technique in product usage process",
    "Product dispensing error",
    "Product prescribing error",
    "Product preparation error",
    "Drug dispensed to wrong patient",
    "Wrong dose",
    "Wrong strength",
    "Wrong dosage form",
    "Wrong dosage formulation",
    "Incorrect product formulation administered",
    "Medication error",
    "Product label issue",
    "Product use issue",
)

#: Dose-magnitude errors, used for substitute spillover (H2).
DOSING_ERROR_PTS: Tuple[str, ...] = (
    "Incorrect dose administered",
    "Accidental overdose",
    "Overdose",
    "Underdose",
    "Wrong dose",
    "Wrong strength",
    "Extra dose administered",
)

#: PTs that should track the shortage list itself (exposure validation, H5b).
SUPPLY_ISSUE_PTS: Tuple[str, ...] = (
    "Product availability issue",
    "Product supply issue",
    "Product distribution issue",
)

#: Handling-unrelated PTs as negative-control outcomes (H5a).
NEGATIVE_CONTROL_PTS: Tuple[str, ...] = (
    "Alopecia",
    "Rash",
    "Nausea",
    "Headache",
    "Fatigue",
)

OUTCOME_GROUPS: Dict[str, Optional[Tuple[str, ...]]] = {
    "all": None,
    "error": MEDICATION_ERROR_PTS,
    "dosing": DOSING_ERROR_PTS,
    "supply": SUPPLY_ISSUE_PTS,
    "negctrl": NEGATIVE_CONTROL_PTS,
}

# --------------------------------------------------------------------------- names
_SALTS = (
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "sulfate", "sulphate",
    "acetate", "citrate", "tartrate", "maleate", "mesylate", "besylate", "succinate", "fumarate",
    "phosphate", "bromide", "chloride", "monohydrate", "dihydrate", "trihydrate", "anhydrous",
    "disodium", "dipotassium", "bitartrate", "hydrobromide", "lactate", "gluconate", "tosylate",
    "pamoate", "valerate", "propionate", "decanoate", "enanthate", "cypionate", "benzoate",
)
_FORMS = (
    "injection", "injectable", "for injection", "tablets", "tablet", "capsules", "capsule",
    "oral solution", "oral suspension", "solution", "suspension", "cream", "ointment", "gel",
    "patch", "inhalation", "aerosol", "spray", "powder", "concentrate", "emulsion", "lyophilized",
    "extended release", "delayed release", "er", "xr", "sr", "usp", "in dextrose", "in sodium chloride",
    "in water", "premixed", "prefilled syringe", "vial", "vials", "ampule", "ampules", "bag", "bags",
)
_STRENGTH_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|meq|units?|iu|%)(\s*/\s*\d*(\.\d+)?\s*(ml|l|kg|dose|hour|hr))?\b")


def normalize_name(name: Optional[str]) -> str:
    """Reduce a product/generic name to a lower-case ingredient key.

    Strips strengths, dosage forms and common salts, keeps the first ingredient of a combination
    (``"A and B"``/``"A/B"`` -> ``"a"``) unless ``keep_combinations`` is needed downstream.

    >>> normalize_name("Heparin Sodium Injection, USP 5,000 units/mL")
    'heparin'
    >>> normalize_name("Semaglutide (Ozempic) 0.25 mg/0.5 mL")
    'semaglutide'
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", " ", s)          # parenthetical brand names
    s = s.replace(",", " ")
    s = _STRENGTH_RE.sub(" ", s)
    s = re.sub(r"\b\d[\d.,]*\b", " ", s)       # leftover numbers
    s = re.split(r"\s+(?:and|with|/|&)\s+|/", s)[0]
    tokens = [t for t in re.split(r"[\s\-]+", s) if t]
    tokens = [t for t in tokens if t not in _SALTS and t not in _FORMS]
    # drop multi-word dosage forms that survived tokenisation
    joined = " ".join(tokens)
    for f in sorted(_FORMS, key=len, reverse=True):
        joined = re.sub(rf"\b{re.escape(f)}\b", " ", joined)
    return re.sub(r"\s+", " ", joined).strip()


# --------------------------------------------------------------------------- episodes
def _to_month(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.to_period("M").dt.to_timestamp()


def episodes_from_openfda(df: pd.DataFrame, resolved_statuses: Sequence[str] = ("Resolved",)) -> pd.DataFrame:
    """Collapse flattened ``drug/shortages`` records to one episode per ingredient.

    Onset = earliest ``initial_posting_date`` across presentations; resolution = latest
    ``update_date`` among records whose status is in ``resolved_statuses`` if *all* records of the
    ingredient are resolved, otherwise NaT (still current). Adds ``injectable``, ``reason`` and
    ``category`` modifiers.

    Returns columns: ``drug, onset, resolution, n_presentations, injectable, reason, category``.
    """
    d = df.copy()
    d["drug"] = d["generic_name"].map(normalize_name)
    d = d[d["drug"] != ""]
    d["onset"] = _to_month(d["initial_posting_date"])
    d["update"] = _to_month(d["update_date"])
    d["is_resolved"] = d["status"].astype(str).str.strip().isin(resolved_statuses)
    pres = d.get("presentation", pd.Series("", index=d.index)).astype(str).str.lower()
    d["injectable"] = pres.str.contains("inject|vial|syringe|infusion|iv |intravenous|bag", regex=True)

    def _agg(g: pd.DataFrame) -> pd.Series:
        all_resolved = bool(g["is_resolved"].all())
        return pd.Series({
            "onset": g["onset"].min(),
            "resolution": g.loc[g["is_resolved"], "update"].max() if all_resolved else pd.NaT,
            "n_presentations": len(g),
            "injectable": bool(g["injectable"].any()),
            "reason": g["shortage_reason"].dropna().astype(str).mode().iloc[0] if g["shortage_reason"].notna().any() else "",
            "category": g["therapeutic_category"].dropna().astype(str).mode().iloc[0] if g["therapeutic_category"].notna().any() else "",
        })

    ep = d.groupby("drug").apply(_agg, include_groups=False).reset_index()
    return ep.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)


# --------------------------------------------------------------------------- counts
def daily_counts_to_monthly(daily: pd.DataFrame, months: pd.DatetimeIndex) -> pd.Series:
    """Aggregate an openFDA daily ``date,count`` table to a monthly Series aligned on ``months`` (0-filled)."""
    if daily is None or len(daily) == 0:
        return pd.Series(0, index=months, dtype=int)
    d = daily.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    s = d.groupby("month")["count"].sum()
    return s.reindex(months, fill_value=0).astype(int)


def build_panel(episodes: pd.DataFrame, counts: Mapping[str, Mapping[str, pd.Series]],
                months: pd.DatetimeIndex, modifiers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Assemble the long drug x month panel.

    Parameters
    ----------
    episodes : output of :func:`episodes_from_openfda` (drugs absent from it are never-treated).
    counts : ``{drug: {group: monthly Series}}``; groups must include ``"all"`` and ``"error"``.
    months : calendar months of the panel.
    modifiers : optional per-drug covariates indexed by drug.

    Returns
    -------
    DataFrame with columns ``drug, month, t, n_<group>..., treated, cohort, rel_time,
    resolution, rel_time_res, ever_treated`` where ``t`` is the integer month index, ``cohort`` the
    onset month index (NaN if never treated) and ``rel_time = t - cohort``.
    """
    ep = episodes.set_index("drug") if "drug" in episodes.columns else episodes
    t_index = {m: i for i, m in enumerate(months)}
    rows: List[pd.DataFrame] = []
    for drug, groups in counts.items():
        df = pd.DataFrame({"drug": drug, "month": months, "t": np.arange(len(months))})
        for g, s in groups.items():
            df[f"n_{g}"] = pd.Series(s).reindex(months, fill_value=0).values
        if drug in ep.index and pd.notna(ep.loc[drug, "onset"]):
            onset = pd.Timestamp(ep.loc[drug, "onset"])
            cohort = t_index.get(onset, np.nan)
            if np.isnan(cohort):  # onset outside the panel window
                cohort = np.nan if onset > months[-1] else 0
            res = ep.loc[drug, "resolution"] if "resolution" in ep.columns else pd.NaT
            res_t = t_index.get(pd.Timestamp(res), np.nan) if pd.notna(res) else np.nan
        else:
            cohort, res_t = np.nan, np.nan
        df["cohort"] = cohort
        df["rel_time"] = df["t"] - cohort if not np.isnan(cohort) else np.nan
        df["resolution_t"] = res_t
        df["rel_time_res"] = df["t"] - res_t if not np.isnan(res_t) else np.nan
        if np.isnan(cohort):
            df["treated"] = 0
        else:
            end = res_t if not np.isnan(res_t) else np.inf
            df["treated"] = ((df["t"] >= cohort) & (df["t"] < end)).astype(int)
        df["ever_treated"] = int(not np.isnan(cohort))
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    if modifiers is not None:
        panel = panel.merge(modifiers, left_on="drug", right_index=True, how="left")
    return panel


# --------------------------------------------------------------------------- substitutes
def find_substitutes(drug: str, products: pd.DataFrame, same_route: bool = True,
                     exclude: Iterable[str] = ()) -> List[str]:
    """Ingredients sharing an Established Pharmacologic Class (and route) with ``drug``.

    ``products`` needs columns ``drug`` (normalised ingredient), ``epc`` (one class per row) and
    ``route``. Returns ingredients other than ``drug`` and those in ``exclude``, sorted.
    """
    p = products.dropna(subset=["epc"])
    mine = p[p["drug"] == drug]
    if mine.empty:
        return []
    classes = set(mine["epc"])
    cand = p[p["epc"].isin(classes)]
    if same_route:
        routes = set(mine["route"].dropna())
        cand = cand[cand["route"].isin(routes)]
    out = sorted(set(cand["drug"]) - {drug} - set(exclude))
    return out
