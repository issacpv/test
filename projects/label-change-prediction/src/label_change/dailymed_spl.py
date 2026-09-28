"""DailyMed SPL version history, LOINC-coded section extraction and SrLC export parsing.

SPL documents are HL7 v3 XML (namespace ``urn:hl7-org:v3``). Each labeling section is a
``<section>`` whose ``<code code="..." codeSystem="2.16.840.1.113883.6.1"/>`` is a LOINC code:

======== ==============================
34066-1  BOXED WARNING SECTION
43685-7  WARNINGS AND PRECAUTIONS SECTION
34071-1  WARNINGS SECTION (old format)
34070-3  CONTRAINDICATIONS SECTION
34084-4  ADVERSE REACTIONS SECTION
======== ==============================

The client's HTTP session is injectable so that parsing and diffing are testable offline.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

LOG = logging.getLogger(__name__)

SECTION_LOINC: Dict[str, str] = {
    "boxed_warning": "34066-1",
    "warnings_and_precautions": "43685-7",
    "warnings": "34071-1",
    "contraindications": "34070-3",
    "adverse_reactions": "34084-4",
}
_LOINC_TO_NAME = {v: k for k, v in SECTION_LOINC.items()}


# --------------------------------------------------------------------------- client
class DailyMedClient:
    """Thin client for DailyMed web services v2.

    Endpoints (https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm):
    ``spls.json?drug_name=..``, ``spls/{setid}/history.json``, ``spls/{setid}.xml``.
    ``VERSION_URL`` holds the pattern for a specific version; NLM documents it on the same page.
    """

    BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    VERSION_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}/version/{version}.xml"

    def __init__(self, session: Any = None, timeout: float = 60.0, min_interval: float = 0.34,
                 max_retries: int = 4) -> None:
        if session is None:
            if requests is None:  # pragma: no cover
                raise ImportError("requests is required for live DailyMed calls")
            session = requests.Session()
            session.headers.update({"User-Agent": "label_change/0.1 (research)"})
        self.session = session
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries:
                    raise
                LOG.warning("DailyMed call failed (%s); retrying", exc)
                time.sleep(1.5 ** attempt)
        raise RuntimeError("unreachable")

    def search_spls(self, drug_name: str, max_pages: int = 20) -> List[Dict[str, Any]]:
        """All SPL entries (setid, title, published_date, version) matching a drug name."""
        out: List[Dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            data = self._get(f"{self.BASE}/spls.json", {"drug_name": drug_name, "pagesize": 100, "page": page}).json()
            rows = data.get("data", [])
            out.extend(rows)
            meta = data.get("metadata", {})
            if not rows or page >= int(meta.get("total_pages", page)):
                break
        return out

    def history(self, setid: str) -> List[Dict[str, Any]]:
        """Version history: list of ``{spl_version, published_date}`` dicts (newest first)."""
        data = self._get(f"{self.BASE}/spls/{setid}/history.json").json()
        hist = data.get("data", {})
        if isinstance(hist, dict):
            hist = hist.get("history", [])
        return list(hist)

    def spl_xml(self, setid: str, version: Optional[int] = None) -> str:
        url = f"{self.BASE}/spls/{setid}.xml" if version is None else self.VERSION_URL.format(setid=setid, version=version)
        return self._get(url).text


# --------------------------------------------------------------------------- parsing
def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _text_of(elem: ET.Element) -> str:
    txt = " ".join(t.strip() for t in elem.itertext() if t and t.strip())
    return re.sub(r"\s+", " ", txt).strip()


def extract_sections(xml_text: str, codes: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Return ``{section_name: text}`` for LOINC-coded sections found in an SPL document.

    Nested sub-sections are merged into their parent's text (itertext over the whole section).
    Unknown codes are keyed by the LOINC code itself when explicitly requested via ``codes``.
    """
    wanted = set(codes) if codes is not None else set(SECTION_LOINC.values())
    root = ET.fromstring(xml_text)
    out: Dict[str, str] = {}
    for sec in root.iter():
        if _local(sec.tag) != "section":
            continue
        code_el = next((c for c in sec if _local(c.tag) == "code"), None)
        if code_el is None:
            continue
        code = code_el.get("code", "")
        if code not in wanted:
            continue
        name = _LOINC_TO_NAME.get(code, code)
        text = _text_of(sec)
        out[name] = (out[name] + " " + text).strip() if name in out else text
    return out


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_terms(text: str, vocabulary: Iterable[str]) -> Set[str]:
    """Whole-phrase, case-insensitive matches of vocabulary terms (e.g. MedDRA PTs) in ``text``."""
    norm = " " + normalize_text(text) + " "
    found: Set[str] = set()
    for term in vocabulary:
        t = normalize_text(term)
        if t and f" {t} " in norm:
            found.add(term)
    return found


def diff_sections(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, Dict[str, Set[str]]]:
    """Sentence-level diff per section: ``{section: {"added": {...}, "removed": {...}}}``."""
    out: Dict[str, Dict[str, Set[str]]] = {}
    for name in set(old) | set(new):
        o = set(s.strip() for s in re.split(r"(?<=[.;])\s+", old.get(name, "")) if s.strip())
        n = set(s.strip() for s in re.split(r"(?<=[.;])\s+", new.get(name, "")) if s.strip())
        out[name] = {"added": n - o, "removed": o - n}
    return out


def new_terms(old: Dict[str, str], new: Dict[str, str], vocabulary: Iterable[str],
              sections: Sequence[str] = ("boxed_warning", "warnings_and_precautions", "warnings")) -> Dict[str, Set[str]]:
    """Vocabulary terms present in ``new`` but absent from ``old`` for the given sections."""
    vocab = list(vocabulary)
    out: Dict[str, Set[str]] = {}
    for name in sections:
        before = extract_terms(old.get(name, ""), vocab)
        after = extract_terms(new.get(name, ""), vocab)
        added = after - before
        if added:
            out[name] = added
    return out


# --------------------------------------------------------------------------- SrLC export
_SRLC_COLS = {
    "drug": ("drug name", "drug", "product", "proprietary name"),
    "application": ("application", "nda", "application number", "appl"),
    "section": ("section", "labeling section", "sections"),
    "date": ("date", "approval date", "supplement approval date", "effective"),
    "summary": ("summary", "description", "labeling change", "change"),
}


def parse_srlc_export(path_or_buffer: Any, vocabulary: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read an SrLC export (CSV) with tolerant column matching.

    Returns columns ``drug, application, section, date, summary, quarter`` (+ ``terms`` when a
    vocabulary is given: set of PTs matched in the summary). Sections are normalised to
    ``boxed_warning``, ``warnings_and_precautions``, ``contraindications``, ``adverse_reactions`` or ``other``.
    """
    raw = pd.read_csv(path_or_buffer)
    cols = {c: c.strip().lower() for c in raw.columns}
    mapping: Dict[str, str] = {}
    for key, options in _SRLC_COLS.items():
        for c, lc in cols.items():
            if any(lc == o or lc.startswith(o) for o in options):
                mapping[key] = c
                break
    missing = [k for k in ("drug", "section", "date") if k not in mapping]
    if missing:
        raise ValueError(f"SrLC export lacks columns for {missing}; found {list(raw.columns)}")
    df = pd.DataFrame({k: raw[c] for k, c in mapping.items()})
    for k in ("application", "summary"):
        if k not in df:
            df[k] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    sec = df["section"].astype(str).str.lower()
    df["section"] = pd.Series("other", index=df.index)
    df.loc[sec.str.contains("boxed"), "section"] = "boxed_warning"
    df.loc[sec.str.contains("precaution") | (sec.str.strip() == "warnings"), "section"] = "warnings_and_precautions"
    df.loc[sec.str.contains("contraindication"), "section"] = "contraindications"
    df.loc[sec.str.contains("adverse reaction"), "section"] = "adverse_reactions"
    if vocabulary is not None:
        vocab = list(vocabulary)
        df["terms"] = df["summary"].fillna("").map(lambda s: extract_terms(s, vocab))
    return df
