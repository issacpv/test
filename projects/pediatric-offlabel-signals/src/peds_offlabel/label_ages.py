"""Extract labelled paediatric age floors from SPL text (openFDA ``drug/label``).

The ``pediatric_use`` section (SPL LOINC 34081-0) and ``indications_and_usage``
express the labelled population in a small number of phrasings, e.g.

* "pediatric patients 6 years of age and older"
* "adults and pediatric patients aged 12 years and older"
* "pediatric patients 1 month to less than 2 years of age"
* "Safety and effectiveness in pediatric patients below the age of 2 years
  have not been established."
* "Safety and effectiveness in pediatric patients have not been established."
  (no paediatric approval at all)

:func:`extract_age_floor` returns the minimum labelled age in years (the
"floor") together with a status and the sentence used as evidence, so the
extraction can be audited and compared with FDA's Pediatric Labeling Changes
table. Weight-based thresholds ("patients weighing at least 40 kg") are
flagged but not converted.

This is deliberately a transparent regex extractor, not an LLM; the
project's validation step measures its accuracy on a hand-labelled sample.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r")"
_UNIT = r"(years?|months?|weeks?|days?)"
UNIT_TO_YEARS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52.18, "day": 1 / 365.25}

# "X years of age and older", "X years and older", "aged X years and above", ">= X years"
_ESTABLISHED = [
    re.compile(rf"\b(?:aged?\s+)?{_NUM}\s*{_UNIT}\s+(?:of\s+age\s+)?(?:and|or)\s+(?:older|above|greater)", re.I),
    re.compile(rf"(?:≥|>=|at\s+least)\s*{_NUM}\s*{_UNIT}\s*(?:of\s+age)?", re.I),
    re.compile(rf"\b{_NUM}\s*{_UNIT}\s+(?:of\s+age\s+)?(?:to|through|-)\s+(?:less\s+than\s+)?{_NUM}\s*{_UNIT}", re.I),
    re.compile(rf"\b(?:from\s+)?birth\b", re.I),
    re.compile(rf"\b(?:neonates?|newborns?)\b", re.I),
]
# "below the age of X years", "younger than X years", "less than X years of age", "under X years"
_BELOW = re.compile(rf"(?:below\s+(?:the\s+age\s+of\s+)?|younger\s+than\s+|less\s+than\s+|under\s+(?:the\s+age\s+of\s+)?)\s*{_NUM}\s*{_UNIT}", re.I)
_NOT_ESTABLISHED = re.compile(r"(safety\s+and\s+(?:effectiveness|efficacy)|effectiveness\s+and\s+safety)[^.]{0,120}?(?:have|has)\s+not\s+been\s+established", re.I)
_PEDIATRIC_CONTEXT = re.compile(r"\b(pediatric|paediatric|children|child|adolescent|infant|neonat|newborn|juvenile)", re.I)
_WEIGHT = re.compile(r"\b(?:weighing|body\s+weight)\s+(?:at\s+least|≥|>=|of|greater\s+than)?\s*\d+(?:\.\d+)?\s*kg", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+")


def _to_years(num: str, unit: str) -> float:
    n = float(NUMBER_WORDS.get(num.lower(), num)) if not num.replace(".", "", 1).isdigit() else float(num)
    return n * UNIT_TO_YEARS[unit.lower().rstrip("s")]


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or ""))
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def extract_age_floor(text: str, require_pediatric_context: bool = True) -> Dict[str, Any]:
    """Return ``{"min_age_years", "status", "evidence", "weight_based", "candidates"}``.

    ``status`` is one of ``"approved"`` (an established paediatric age range
    was found), ``"not_established"`` (only a general "not established"
    statement), ``"unknown"`` (no paediatric statement found).
    The floor is the smallest established age; a "below X ... not
    established" statement is used as a consistent floor when no established
    range is stated explicitly.
    """
    candidates: List[Tuple[float, str]] = []
    below: List[Tuple[float, str]] = []
    not_est: List[str] = []
    weight_based = False
    for s in _sentences(text):
        if require_pediatric_context and not _PEDIATRIC_CONTEXT.search(s):
            continue
        if _WEIGHT.search(s):
            weight_based = True
        neg = bool(_NOT_ESTABLISHED.search(s))
        m_below = _BELOW.search(s)
        if neg:
            if m_below:
                below.append((_to_years(m_below.group(1), m_below.group(2)), s))
            else:
                not_est.append(s)
            continue  # a negative sentence never contributes an established range
        for pat in _ESTABLISHED:
            m = pat.search(s)
            if not m:
                continue
            if m.re.pattern.startswith(r"\b(?:from\s+)?birth") or m.re.pattern.startswith(r"\b(?:neonates?"):
                candidates.append((0.0, s))
            else:
                candidates.append((_to_years(m.group(1), m.group(2)), s))
    if candidates:
        floor, ev = min(candidates, key=lambda t: t[0])
        return {"min_age_years": floor, "status": "approved", "evidence": ev, "weight_based": weight_based, "candidates": sorted({c for c, _ in candidates})}
    if below:
        floor, ev = min(below, key=lambda t: t[0])
        return {"min_age_years": floor, "status": "approved", "evidence": ev, "weight_based": weight_based, "candidates": [floor]}
    if not_est:
        return {"min_age_years": None, "status": "not_established", "evidence": not_est[0], "weight_based": weight_based, "candidates": []}
    return {"min_age_years": None, "status": "unknown", "evidence": "", "weight_based": weight_based, "candidates": []}


def floor_from_label(label: Dict[str, Any], sections: Sequence[str] = ("pediatric_use", "indications_and_usage")) -> Dict[str, Any]:
    """Combine sections: ``pediatric_use`` takes precedence over indications."""
    results = {}
    for s in sections:
        txt = label.get(s)
        if isinstance(txt, list):
            txt = " ".join(str(t) for t in txt)
        results[s] = extract_age_floor(txt or "", require_pediatric_context=(s != "pediatric_use"))
    for s in sections:
        if results[s]["status"] == "approved":
            out = dict(results[s])
            out["section"] = s
            return out
    for s in sections:
        if results[s]["status"] == "not_established":
            out = dict(results[s])
            out["section"] = s
            return out
    out = dict(results[sections[0]])
    out["section"] = sections[0]
    return out


def classify_age(age_years: Optional[float], floor: Dict[str, Any]) -> str:
    """Classify a paediatric report's age against a drug's labelled floor.

    Returns ``"on_label_age"``, ``"below_floor"``, ``"no_pediatric_labeling"``
    or ``"unknown"``. Adults (>= 18) always return ``"adult"``.
    """
    if age_years is None:
        return "unknown"
    if age_years >= 18:
        return "adult"
    if floor.get("status") == "approved" and floor.get("min_age_years") is not None:
        return "on_label_age" if age_years >= float(floor["min_age_years"]) else "below_floor"
    if floor.get("status") == "not_established":
        return "no_pediatric_labeling"
    return "unknown"
