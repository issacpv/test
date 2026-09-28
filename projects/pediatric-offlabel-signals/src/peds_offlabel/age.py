"""Paediatric age handling for FAERS reports.

FAERS gives age in two places: ``patient.patientonsetage`` with a unit code
(800 decade, 801 year, 802 month, 803 week, 804 day, 805 hour) and the coarse
``patient.patientagegroup`` (1 neonate, 2 infant, 3 child, 4 adolescent,
5 adult, 6 elderly). Roughly a third of reports lack a numeric age, so the
age group is used as a fallback with band-level (not year-level) precision.

Bands follow the ICH E11 / FDA convention used in most paediatric-labelling
text: neonate < 28 days, infant 28 days to < 2 years, child 2 to < 12 years,
adolescent 12 to < 18 years.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .openfda import SEX_LABELS, age_in_years

AGE_GROUP_CODES: Dict[str, str] = {"1": "neonate", "2": "infant", "3": "child", "4": "adolescent", "5": "adult", "6": "elderly"}
BANDS = ("neonate", "infant", "child", "adolescent")
BAND_EDGES_YEARS = {"neonate": (0.0, 28 / 365.25), "infant": (28 / 365.25, 2.0), "child": (2.0, 12.0), "adolescent": (12.0, 18.0)}


def band_from_years(age: Optional[float]) -> Optional[str]:
    if age is None:
        return None
    for band, (lo, hi) in BAND_EDGES_YEARS.items():
        if lo <= age < hi:
            return band
    return "adult" if age >= 18 else None


def pediatric_band(age_years: Optional[float], agegroup_code: Any) -> Optional[str]:
    """Band from numeric age, else from ``patientagegroup``; None if unknown."""
    b = band_from_years(age_years)
    if b is not None:
        return b
    return AGE_GROUP_CODES.get(str(agegroup_code)) if agegroup_code is not None else None


def is_pediatric(band: Optional[str]) -> bool:
    return band in BANDS


def flatten_report(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a FAERS report with age band, per-drug names/roles/indications."""
    patient = rec.get("patient", {}) or {}
    src = rec.get("primarysource", {}) or {}
    age = age_in_years(patient.get("patientonsetage"), patient.get("patientonsetageunit"))
    band = pediatric_band(age, patient.get("patientagegroup"))
    drugs: List[Dict[str, Any]] = []
    for d in patient.get("drug", []) or []:
        names = [str(n).upper() for n in ((d.get("openfda", {}) or {}).get("generic_name") or [])]
        if not names and d.get("medicinalproduct"):
            names = [str(d["medicinalproduct"]).upper().strip()]
        for n in names:
            drugs.append({"name": n, "role": str(d.get("drugcharacterization") or ""), "indication": (d.get("drugindication") or None), "mapped": bool(d.get("openfda"))})
    reactions = sorted({str(r.get("reactionmeddrapt", "")).upper() for r in (patient.get("reaction", []) or []) if r.get("reactionmeddrapt")})
    return {
        "safetyreportid": rec.get("safetyreportid"),
        "receivedate": rec.get("receivedate"),
        "serious": str(rec.get("serious")) == "1",
        "death": str(rec.get("seriousnessdeath")) == "1",
        "hospitalization": str(rec.get("seriousnesshospitalization")) == "1",
        "occurcountry": rec.get("occurcountry"),
        "qualification": str(src.get("qualification")) if src.get("qualification") is not None else None,
        "sex": SEX_LABELS.get(str(patient.get("patientsex")), "unknown"),
        "age_years": age,
        "age_band": band,
        "pediatric": is_pediatric(band),
        "drugs": drugs,
        "suspect": sorted({d["name"] for d in drugs if d["role"] == "1"}),
        "reactions": reactions,
    }
