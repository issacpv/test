"""Local regex detectors for *candidate* residual identifiers (HIPAA Safe Harbor classes).

Design rules (these are the whole point of the module):

* A detector returns ``Finding`` objects that carry the *category*, the *section*
  and the *length* of the match - never the matched text. Offsets are kept in
  memory only so that overlapping matches can be de-duplicated and assigned to
  sections; ``count_by_category`` drops them.
* Everything is a *candidate*: "a four-digit 19xx/20xx year" is a residual only
  if the corpus shifts dates (MIMIC-IV shifts to 2100-2200); "Dr. Smith" may be
  a fictional surrogate in some corpora. Calibration against planted synthetic
  identifiers (``deid_audit.estimate``) turns candidate counts into rate estimates.
* No network, no model downloads: pure ``re``.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .sections import Section, section_of, split_sections

MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
STREET = r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)"
TITLE = r"(?:Dr|Mr|Mrs|Ms|Mx|Prof)"

# category -> (HIPAA identifier class, compiled regex)
PATTERNS: dict[str, tuple[str, re.Pattern]] = {
    "phone": ("telephone/fax numbers", re.compile(r"(?<![\d-])(?:\(\d{3}\)\s?|\b\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\d-])")),
    "email": ("email addresses", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    "url": ("URLs", re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)),
    "ssn_like": ("social security numbers", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    "mrn_like": ("medical record numbers", re.compile(r"\b(?:MRN|Medical Record (?:Number|No)|Unit No|Record\s*#)\s*[:#]?\s*\d{5,10}\b", re.IGNORECASE)),
    "zip": ("geographic subdivisions smaller than state (ZIP)", re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")),
    "street_address": ("street addresses", re.compile(rf"\b\d{{1,5}}\s+(?:[A-Z][a-z]+\s){{1,3}}{STREET}\.?\b")),
    "date_unshifted_numeric": ("dates (unshifted, numeric)", re.compile(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}(?!\d)")),
    "date_unshifted_text": ("dates (unshifted, textual)", re.compile(rf"\b{MONTHS}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b")),
    "year_unshifted_context": ("dates (unshifted year in temporal context)", re.compile(r"\b(?:in|since|from|until|of|during)\s+(?:19[2-9]\d|20[0-2]\d)\b", re.IGNORECASE)),
    "date_shifted": ("dates (shifted, expected)", re.compile(r"(?<!\d)2[12]\d{2}-\d{1,2}-\d{1,2}(?!\d)")),
    "age_over_89": ("ages over 89", re.compile(r"\b(?:9\d|1[0-4]\d)[\s-]*(?:year|yo\b|y/o|y\.o\.|years?[\s-]old)", re.IGNORECASE)),
    "name_after_title": ("names (title + capitalised token)", re.compile(rf"\b{TITLE}\.?\s+[A-Z][a-z]{{2,}}(?:\s+[A-Z][a-z]{{2,}})?\b")),
    "name_in_signature": ("names (signature / dictation line)", re.compile(r"\b(?:Dictated by|Signed by|Attending|Provider|Electronically signed by)\s*:?\s+[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")),
    "institution_name": ("institution names (candidate)", re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\s+(?:General\s+|Memorial\s+|Community\s+|University\s+)?(?:Hospital|Medical Center|Health Center|Clinic)\b")),
    "provider_id": ("license / certificate / NPI numbers", re.compile(r"\b(?:NPI|DEA|License|Lic)\s*#?\s*:?\s*\d{6,10}\b", re.IGNORECASE)),
    "device_serial": ("device identifiers / serial numbers", re.compile(r"\b(?:serial|S/N|SN)\s*(?:number|no\.?)?\s*[:#]?\s*[A-Z0-9]{6,}\b", re.IGNORECASE)),
    "ip_address": ("IP addresses", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
}
RESIDUAL_CATEGORIES: tuple[str, ...] = tuple(k for k in PATTERNS if k != "date_shifted")


@dataclass(frozen=True)
class Finding:
    """A candidate residual identifier: what kind, where, how long. Never the text."""

    category: str
    section: str
    start: int
    length: int


def scan(text: str, sections: Sequence[Section] | None = None,
         categories: Iterable[str] | None = None) -> list[Finding]:
    """Run all (or the given) detectors and return de-duplicated findings.

    Overlapping matches of different categories are all kept (a signature line can be both
    ``name_in_signature`` and ``name_after_title``); identical spans within a category are not.
    """
    secs = list(sections) if sections is not None else split_sections(text)
    cats = list(categories) if categories is not None else list(PATTERNS)
    out: list[Finding] = []
    for cat in cats:
        _, rx = PATTERNS[cat]
        seen: set[tuple[int, int]] = set()
        for m in rx.finditer(text):
            span = (m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)
            out.append(Finding(cat, section_of(m.start(), secs), m.start(), m.end() - m.start()))
    return out


def count_by_category(findings: Iterable[Finding], by_section: bool = False) -> Counter:
    """Aggregate findings to counts; keys are ``category`` or ``(category, section)``."""
    if by_section:
        return Counter((f.category, f.section) for f in findings)
    return Counter(f.category for f in findings)


def note_flags(findings: Iterable[Finding], categories: Iterable[str] = RESIDUAL_CATEGORIES) -> dict[str, int]:
    """Per-note presence flags (1/0) per category: the unit for note-level rates and capture-recapture."""
    present = {f.category for f in findings}
    return {c: int(c in present) for c in categories}


def describe_categories() -> dict[str, str]:
    """Category -> HIPAA identifier class description (for tables in the report)."""
    return {k: v[0] for k, v in PATTERNS.items()}
