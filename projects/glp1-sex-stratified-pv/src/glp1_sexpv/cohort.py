"""GLP-1 receptor agonist cohort definition from FAERS reports.

Covers the incretin class as marketed in the US (verify approval details
against the current labels before publication):

* semaglutide - Ozempic (T2D, 2017), Rybelsus (oral, T2D, 2019), Wegovy (chronic
  weight management, 2021)
* tirzepatide (dual GIP/GLP-1) - Mounjaro (T2D, 2022), Zepbound (weight, 2023)
* liraglutide - Victoza (T2D, 2010), Saxenda (weight, 2014)
* dulaglutide - Trulicity (T2D, 2014)
* exenatide - Byetta (2005), Bydureon (2012)
* lixisenatide - Adlyxin (2016); albiglutide - Tanzeum (2014, withdrawn 2018)

Indication is the key confounder for sex comparisons: the weight-management
brands are prescribed predominantly to women, the diabetes brands less so.
:func:`classify_indication` uses the free-text ``drugindication`` first and
falls back to the brand's labelled indication.

"Compounded" semaglutide/tirzepatide (503A/503B pharmacies, widely used
during the 2023-2025 shortages) appears in FAERS as verbatim
``medicinalproduct`` strings without ``openfda`` mapping;
:func:`is_compounded` flags them by string patterns.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .openfda import SEX_LABELS, age_in_years

GLP1_AGENTS: Dict[str, Dict[str, Any]] = {
    "SEMAGLUTIDE": {"brands": {"OZEMPIC": "t2d", "RYBELSUS": "t2d", "WEGOVY": "obesity"}, "class": "GLP-1 RA"},
    "TIRZEPATIDE": {"brands": {"MOUNJARO": "t2d", "ZEPBOUND": "obesity"}, "class": "GIP/GLP-1 RA"},
    "LIRAGLUTIDE": {"brands": {"VICTOZA": "t2d", "SAXENDA": "obesity"}, "class": "GLP-1 RA"},
    "DULAGLUTIDE": {"brands": {"TRULICITY": "t2d"}, "class": "GLP-1 RA"},
    "EXENATIDE": {"brands": {"BYETTA": "t2d", "BYDUREON": "t2d"}, "class": "GLP-1 RA"},
    "LIXISENATIDE": {"brands": {"ADLYXIN": "t2d"}, "class": "GLP-1 RA"},
    "ALBIGLUTIDE": {"brands": {"TANZEUM": "t2d"}, "class": "GLP-1 RA"},
}

#: Active comparators with overlapping indications but different mechanisms.
COMPARATORS: Dict[str, Sequence[str]] = {
    "sglt2": ("EMPAGLIFLOZIN", "DAPAGLIFLOZIN", "CANAGLIFLOZIN", "ERTUGLIFLOZIN"),
    "dpp4": ("SITAGLIPTIN", "LINAGLIPTIN", "SAXAGLIPTIN", "ALOGLIPTIN"),
    "insulin": ("INSULIN GLARGINE", "INSULIN DEGLUDEC", "INSULIN DETEMIR", "INSULIN LISPRO", "INSULIN ASPART"),
    "obesity_other": ("PHENTERMINE", "PHENTERMINE AND TOPIRAMATE", "NALTREXONE AND BUPROPION", "ORLISTAT"),
}

_OBESITY_RE = re.compile(r"\b(obes|overweight|weight|bmi|adipos)", re.I)
_T2D_RE = re.compile(r"\b(diabet|glyc|hba1c|insulin resist|type 2|type ii)", re.I)
_COMPOUND_RE = re.compile(r"\b(compound|503a|503b|semaglutide\s*(sodium|acetate)|tirzepatide\s*(sodium|acetate))", re.I)


def classify_indication(text: Optional[str], brand: Optional[str] = None) -> str:
    """Return 'obesity', 't2d', 'other' or 'unknown' for a drug entry."""
    t = str(text or "")
    if _OBESITY_RE.search(t):
        return "obesity"
    if _T2D_RE.search(t):
        return "t2d"
    if t.strip() and not re.search(r"product used for unknown indication", t, re.I):
        return "other"
    if brand:
        for agent in GLP1_AGENTS.values():
            if brand.upper() in agent["brands"]:
                return agent["brands"][brand.upper()]
    return "unknown"


def is_compounded(medicinalproduct: Optional[str], has_openfda: bool) -> bool:
    """Heuristic flag for compounded GLP-1 products.

    True when the verbatim name contains compounding markers, or when a
    semaglutide/tirzepatide entry lacks openFDA mapping and lacks a brand
    token (bare 'SEMAGLUTIDE' with no product mapping is characteristic of
    compounded products and of poorly coded reports; report both variants).
    """
    s = str(medicinalproduct or "")
    if _COMPOUND_RE.search(s):
        return True
    if not has_openfda and re.fullmatch(r"\s*(SEMAGLUTIDE|TIRZEPATIDE)\s*", s, re.I):
        return True
    return False


def _names(entry: Dict[str, Any]) -> List[str]:
    ofda = entry.get("openfda", {}) or {}
    out = [str(x).upper() for k in ("generic_name", "brand_name", "substance_name") for x in (ofda.get(k) or [])]
    if entry.get("medicinalproduct"):
        out.append(str(entry["medicinalproduct"]).upper())
    return out


def match_agent(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Identify the GLP-1 agent (and brand if any) named by one drug entry."""
    names = _names(entry)
    blob = " | ".join(names)
    for agent, meta in GLP1_AGENTS.items():
        brand_hit = next((b for b in meta["brands"] if re.search(rf"\b{b}\b", blob)), None)
        if brand_hit or re.search(rf"\b{agent}\b", blob):
            has_ofda = bool(entry.get("openfda"))
            return {
                "agent": agent,
                "brand": brand_hit,
                "class": meta["class"],
                "indication": classify_indication(entry.get("drugindication"), brand_hit),
                "compounded": is_compounded(entry.get("medicinalproduct"), has_ofda),
                "role": str(entry.get("drugcharacterization") or ""),
            }
    return None


def match_comparator(entry: Dict[str, Any]) -> Optional[str]:
    blob = " | ".join(_names(entry))
    for cls, drugs in COMPARATORS.items():
        if any(re.search(rf"\b{re.escape(d)}\b", blob) for d in drugs):
            return cls
    return None


def flatten_report(rec: Dict[str, Any]) -> Dict[str, Any]:
    """One row per report with GLP-1 exposure details and comparator classes."""
    patient = rec.get("patient", {}) or {}
    src = rec.get("primarysource", {}) or {}
    agents: List[Dict[str, Any]] = []
    comparators: set = set()
    for d in patient.get("drug", []) or []:
        a = match_agent(d)
        if a:
            agents.append(a)
        c = match_comparator(d)
        if c:
            comparators.add(c)
    reactions = sorted({str(r.get("reactionmeddrapt", "")).upper() for r in (patient.get("reaction", []) or []) if r.get("reactionmeddrapt")})
    indications = [a["indication"] for a in agents]
    indication = "obesity" if "obesity" in indications else ("t2d" if "t2d" in indications else (indications[0] if indications else "none"))
    return {
        "safetyreportid": rec.get("safetyreportid"),
        "receivedate": rec.get("receivedate"),
        "serious": rec.get("serious"),
        "occurcountry": rec.get("occurcountry"),
        "qualification": str(src.get("qualification")) if src.get("qualification") is not None else None,
        "sex": SEX_LABELS.get(str(patient.get("patientsex")), "unknown"),
        "age_years": age_in_years(patient.get("patientonsetage"), patient.get("patientonsetageunit")),
        "glp1": bool(agents),
        "glp1_suspect": any(a["role"] == "1" for a in agents),
        "agents": sorted({a["agent"] for a in agents}),
        "brands": sorted({a["brand"] for a in agents if a["brand"]}),
        "indication": indication,
        "compounded": any(a["compounded"] for a in agents),
        "comparators": sorted(comparators),
        "reactions": reactions,
    }


def cohort_summary(reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts by sex x indication x compounded for the exposed reports."""
    from collections import Counter

    c: Counter = Counter()
    n = 0
    for r in reports:
        if not r.get("glp1"):
            continue
        n += 1
        c[(r.get("sex"), r.get("indication"), bool(r.get("compounded")))] += 1
    return {"n_exposed": n, "cells": {f"{s}|{i}|{'compounded' if k else 'branded'}": v for (s, i, k), v in sorted(c.items(), key=lambda kv: -kv[1])}}
