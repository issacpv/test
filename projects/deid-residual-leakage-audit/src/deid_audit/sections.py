"""Section splitting and placeholder statistics for MIMIC-style clinical notes.

MIMIC-IV-Note discharge summaries follow a fixed template whose headers are
listed in ``DISCHARGE_HEADERS``; radiology reports use ``RADIOLOGY_HEADERS``.
Knowing the section lets the audit report *where* residual-identifier
candidates concentrate (e.g. signature blocks, facility lines, follow-up
instructions) without ever storing the text itself.

Placeholders: MIMIC-IV-Note replaces PHI with ``___``; MIMIC-III used tagged
surrogates such as ``[**First Name (STitle) 123**]`` or ``[**2150-3-12**]``.
Their density per section is a covariate for the de-identification model's
activity and a denominator for "residuals per replaced identifier".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

DISCHARGE_HEADERS: tuple[str, ...] = (
    "Name", "Unit No", "Admission Date", "Discharge Date", "Date of Birth", "Sex", "Service",
    "Allergies", "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure",
    "History of Present Illness", "Past Medical History", "Social History", "Family History",
    "Physical Exam", "Pertinent Results", "Brief Hospital Course", "Medications on Admission",
    "Discharge Medications", "Discharge Disposition", "Facility", "Discharge Diagnosis",
    "Discharge Condition", "Discharge Instructions", "Followup Instructions",
)
RADIOLOGY_HEADERS: tuple[str, ...] = (
    "EXAMINATION", "INDICATION", "TECHNIQUE", "COMPARISON", "FINDINGS", "IMPRESSION", "NOTIFICATION",
)

MIMIC4_PLACEHOLDER = re.compile(r"_{3,}")
MIMIC3_PLACEHOLDER = re.compile(r"\[\*\*(.*?)\*\*\]", re.DOTALL)


@dataclass(frozen=True)
class Section:
    """A template section: ``[start, end)`` in character offsets; ``header_end`` closes the header text."""

    name: str
    start: int
    end: int
    header_end: int = 0


def _header_regex(headers: Sequence[str]) -> re.Pattern:
    alts = "|".join(re.escape(h) for h in sorted(headers, key=len, reverse=True))
    return re.compile(rf"^[ \t]*(?P<h>{alts})[ \t]*:", re.MULTILINE | re.IGNORECASE)


def split_sections(text: str, headers: Sequence[str] = DISCHARGE_HEADERS) -> list[Section]:
    """Split ``text`` into sections at template headers; text before the first header is ``PREAMBLE``."""
    rx = _header_regex(headers)
    matches = list(rx.finditer(text))
    if not matches:
        return [Section("PREAMBLE", 0, len(text))]
    out: list[Section] = []
    if matches[0].start() > 0:
        out.append(Section("PREAMBLE", 0, matches[0].start()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        canonical = next(h for h in headers if h.lower() == m.group("h").lower())
        out.append(Section(canonical, m.start(), end, m.end()))
    return out


def section_of(offset: int, sections: Iterable[Section]) -> str:
    for s in sections:
        if s.start <= offset < s.end:
            return s.name
    return "PREAMBLE"


def in_header(offset: int, sections: Iterable[Section]) -> bool:
    """True when ``offset`` falls inside a template header (headers are never identifiers)."""
    return any(s.start <= offset < s.header_end for s in sections)


def placeholder_stats(text: str, sections: Sequence[Section] | None = None) -> dict[str, int]:
    """Counts of MIMIC-IV ``___`` and MIMIC-III ``[** **]`` placeholders, overall and per section.

    Keys: ``ph4_total``, ``ph3_total``, ``ph4__<section>``, ``ph3__<section>``, ``n_chars``, ``n_tokens``.
    """
    sections = list(sections) if sections is not None else split_sections(text)
    out: dict[str, int] = {"n_chars": len(text), "n_tokens": len(text.split())}
    p4 = [m.start() for m in MIMIC4_PLACEHOLDER.finditer(text)]
    p3 = [m.start() for m in MIMIC3_PLACEHOLDER.finditer(text)]
    out["ph4_total"], out["ph3_total"] = len(p4), len(p3)
    for s in sections:
        out[f"ph4__{s.name}"] = sum(s.start <= p < s.end for p in p4)
        out[f"ph3__{s.name}"] = sum(s.start <= p < s.end for p in p3)
    return out


def mimic3_placeholder_types(text: str) -> dict[str, int]:
    """Histogram of MIMIC-III surrogate *types* (the label inside ``[** **]`` with digits stripped)."""
    hist: dict[str, int] = {}
    for m in MIMIC3_PLACEHOLDER.finditer(text):
        label = re.sub(r"[\d\-/ ]+$", "", m.group(1)).strip() or "DATE"
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", m.group(1).strip()):
            label = "DATE"
        hist[label] = hist.get(label, 0) + 1
    return hist
