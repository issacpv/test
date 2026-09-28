"""SPL label access and section parsing.

Sources
-------
* openFDA ``drug/label`` (https://open.fda.gov/apis/drug/label/): the *current*
  version of every SPL, one record per ``set_id`` with ``id`` (SPL version
  GUID), ``version``, ``effective_time`` (YYYYMMDD) and plain-text sections
  (``boxed_warning``, ``warnings_and_cautions``, ``warnings``, ``precautions``,
  ``adverse_reactions``, ``adverse_reactions_table`` ...). ``openfda`` carries
  ``generic_name``, ``brand_name``, ``rxcui``, ``unii``, ``application_number``,
  ``product_type``.
* DailyMed web services (https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm)
  for *version history* of a set id and archived SPL XML, which openFDA does not
  serve. ``fetch_dailymed_history`` uses ``/services/v2/spls/<setid>/history.json``;
  verify the exact URL against the DailyMed documentation if it changes.
* SPL XML (HL7 v3) sections are identified by LOINC codes; see
  ``LOINC_SECTIONS`` below.
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENFDA_BASE = "https://api.fda.gov"
DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

#: openFDA plain-text section fields that carry safety information.
SAFETY_SECTIONS = (
    "boxed_warning",
    "warnings_and_cautions",
    "warnings",
    "precautions",
    "adverse_reactions",
    "adverse_reactions_table",
)

#: LOINC codes of SPL sections (HL7 SPL implementation guide).
LOINC_SECTIONS = {
    "34066-1": "boxed_warning",
    "43685-7": "warnings_and_cautions",
    "34071-1": "warnings",
    "42232-9": "precautions",
    "34084-4": "adverse_reactions",
    "34067-9": "indications_and_usage",
}

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
_SECTION_NUM_RE = re.compile(r"(?m)^\s*\d{1,2}(?:\.\d{1,2})?\s+(?=[A-Z])")
_WS_RE = re.compile(r"\s+")
_SUBSECTION_START_RE = re.compile(r"(?:(?<=\s)|^)(\d{1,2}\.\d{1,2})\s+(?=[A-Z])")
_TOKEN_RE = re.compile(r"\S+")
_TITLE_CONNECTORS = {"and", "or", "of", "in", "the", "to", "with", "for", "a", "an", "on", "by", "&", "/"}
_POSTMARKETING_RE = re.compile(r"post[- ]?marketing experience", re.IGNORECASE)


@dataclass
class LabelDoc:
    """Parsed SPL label (one version of one set id)."""

    set_id: str
    spl_id: Optional[str]
    version: Optional[int]
    effective_time: Optional[pd.Timestamp]
    generic_names: List[str]
    brand_names: List[str]
    rxcui: List[str]
    application_numbers: List[str]
    product_type: Optional[str]
    sections: Dict[str, str] = field(default_factory=dict)

    @property
    def generic_name(self) -> str:
        return self.generic_names[0].upper() if self.generic_names else ""

    @property
    def postmarketing_text(self) -> str:
        """Text of the '6.2 Postmarketing Experience' subsection if present."""
        subs = split_subsections(self.sections.get("adverse_reactions", ""))
        for title, body in subs.items():
            if _POSTMARKETING_RE.search(title):
                return body
        return ""

    def safety_text(self) -> str:
        return "\n".join(self.sections.get(s, "") for s in SAFETY_SECTIONS if self.sections.get(s))


def clean_section_text(text: Any) -> str:
    """Join list-valued openFDA sections, drop leading section numbers, squeeze whitespace."""
    if text is None:
        return ""
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    text = str(text)
    text = _SECTION_NUM_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def split_subsections(text: str) -> Dict[str, str]:
    """Split a section into numbered subsections ('6.1 Clinical Trials Experience', ...).

    Returns ``{title: body}``; text before the first subsection is under ``""``.
    """
    if not text:
        return {}
    matches = list(_SUBSECTION_START_RE.finditer(text))
    if not matches:
        return {"": text}
    out: Dict[str, str] = {}
    if matches[0].start() > 0:
        out[""] = text[: matches[0].start()].strip()
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[m.end() : end]
        title_words, body_start = _title_span(chunk)
        title = f"{m.group(1)} {' '.join(title_words)}".strip()
        out[title] = chunk[body_start:].strip()
    return out


def _title_span(chunk: str, max_words: int = 12):
    """Heuristic Title-Case run after a subsection number.

    Consumes capitalised tokens (plus connector words). If the run ends because
    the next token is lower-case, the sentence-initial word of the body is
    returned to the body: "6.2 Postmarketing Experience The following ..."
    -> title "Postmarketing Experience"; "5.1 Myopathy and Rhabdomyolysis Cases
    of myopathy ..." -> title "Myopathy and Rhabdomyolysis".
    """
    toks = list(_TOKEN_RE.finditer(chunk))
    run = []  # (word, end_offset)
    ended_by_lower = False
    for t in toks:
        w = t.group(0)
        core = w.strip(",;:()")
        if not core:
            break
        if core[0].isupper() or core.lower() in _TITLE_CONNECTORS:
            run.append((w, t.end()))
            if len(run) >= max_words:
                break
        else:
            ended_by_lower = True
            break

    def _is_lower_connector(item) -> bool:
        c = item[0].strip(",;:()")
        return c.lower() in _TITLE_CONNECTORS and not c[0].isupper()

    if ended_by_lower and len(run) > 1:
        if _is_lower_connector(run[-1]):
            # "... Cases of | myopathy": drop the connector, then the sentence-initial word
            run.pop()
            if len(run) > 1:
                run.pop()
        else:
            # "... Experience The | following": the capitalised word starts the body
            run.pop()
    while run and _is_lower_connector(run[-1]):
        run.pop()
    if not run:
        return [], 0
    words = [w.rstrip(",;:") for w, _ in run]
    return words, run[-1][1]


def parse_label(rec: Dict[str, Any]) -> LabelDoc:
    """openFDA ``drug/label`` record -> :class:`LabelDoc`."""
    ofda = rec.get("openfda") or {}
    et = rec.get("effective_time")
    eff = pd.to_datetime(str(et), format="%Y%m%d", errors="coerce") if et else None
    sections = {}
    for s in SAFETY_SECTIONS + ("indications_and_usage", "drug_interactions", "contraindications"):
        if rec.get(s):
            sections[s] = clean_section_text(rec.get(s))
    return LabelDoc(
        set_id=rec.get("set_id"),
        spl_id=rec.get("id"),
        version=int(rec["version"]) if rec.get("version") not in (None, "") else None,
        effective_time=None if eff is None or pd.isna(eff) else eff,
        generic_names=list(ofda.get("generic_name") or []),
        brand_names=list(ofda.get("brand_name") or []),
        rxcui=list(ofda.get("rxcui") or []),
        application_numbers=list(ofda.get("application_number") or []),
        product_type=(ofda.get("product_type") or [None])[0],
        sections=sections,
    )


def parse_spl_xml_sections(xml_text: str) -> Dict[str, str]:
    """Extract safety sections from raw SPL XML (HL7 v3) by LOINC section code.

    Works on archived DailyMed versions. Nested subsections are concatenated
    into their parent section.
    """
    ns = {"v3": "urn:hl7-org:v3"}
    root = ET.fromstring(xml_text)
    out: Dict[str, str] = {}
    for sec in root.iter("{urn:hl7-org:v3}section"):
        code_el = sec.find("v3:code", ns)
        if code_el is None:
            continue
        name = LOINC_SECTIONS.get(code_el.get("code", ""))
        if not name:
            continue
        text = " ".join(t.strip() for t in sec.itertext() if t and t.strip())
        out[name] = clean_section_text((out.get(name, "") + " " + text).strip())
    return out


@dataclass
class LabelClient:
    """openFDA ``drug/label`` + DailyMed history client (rate limited, retried)."""

    api_key: Optional[str] = None
    max_per_minute: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 5
    session: Optional[requests.Session] = None
    _stamps: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENFDA_API_KEY") or None
        if self.max_per_minute is None:
            self.max_per_minute = 240 if self.api_key else 40
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": "unlabeled_ae/0.1 (research)"})

    def _throttle(self) -> None:
        now = time.monotonic()
        while self._stamps and now - self._stamps[0] > 60.0:
            self._stamps.popleft()
        if len(self._stamps) >= int(self.max_per_minute):
            time.sleep(max(60.0 - (now - self._stamps[0]) + 0.05, 0.0))
        self._stamps.append(time.monotonic())

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None, use_key: bool = True) -> requests.Response:
        params = dict(params or {})
        if use_key and self.api_key and "api_key" not in params and "api_key=" not in url:
            params["api_key"] = self.api_key
        backoff = 1.0
        last: Optional[BaseException] = None
        for _ in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
            else:
                if r.status_code == 404 or r.status_code < 400:
                    return r
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        raise RuntimeError(f"giving up: {last}")

    def iter_records(self, endpoint: str, search: Optional[str], limit: int = 100, max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        url = f"{OPENFDA_BASE}/{endpoint.strip('/')}.json"
        params: Dict[str, Any] = {"limit": int(min(limit, 1000))}
        if search:
            params["search"] = search
        n, skip, next_url, followed = 0, 0, None, False
        while True:
            r = self._get(next_url, None) if next_url else self._get(url, {**params, "skip": skip})
            if r.status_code == 404:
                return
            payload = r.json()
            results = payload.get("results", [])
            if not results:
                return
            for rec in results:
                yield rec
                n += 1
                if max_records is not None and n >= max_records:
                    return
            link = r.headers.get("Link") or r.headers.get("link")
            m = _LINK_NEXT_RE.search(link) if link else None
            if m:
                next_url, followed = m.group(1), True
                continue
            if followed:
                return
            skip += len(results)
            total = payload.get("meta", {}).get("results", {}).get("total")
            if skip >= 25000 or (total is not None and skip >= int(total)) or len(results) < params["limit"]:
                return
            next_url = None

    def count(self, endpoint: str, search: Optional[str], count_field: str, limit: int = 1000) -> pd.DataFrame:
        params: Dict[str, Any] = {"count": count_field, "limit": int(limit)}
        if search:
            params["search"] = search
        r = self._get(f"{OPENFDA_BASE}/{endpoint.strip('/')}.json", params)
        if r.status_code == 404:
            return pd.DataFrame(columns=["term", "count"])
        df = pd.DataFrame(r.json().get("results", []))
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d", errors="coerce")
        return df

    # ------------------------------------------------------------ labels
    def labels_for_generic(self, generic_name: str, rx_only: bool = True, max_records: int = 200) -> List[LabelDoc]:
        """All current SPLs whose ``openfda.generic_name`` matches (one per set id)."""
        q = f'openfda.generic_name:"{generic_name.upper()}"'
        if rx_only:
            q += ' AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
        docs = [parse_label(r) for r in self.iter_records("drug/label", q, limit=100, max_records=max_records)]
        # de-duplicate by set id keeping highest version
        best: Dict[str, LabelDoc] = {}
        for d in docs:
            if d.set_id and (d.set_id not in best or (d.version or 0) > (best[d.set_id].version or 0)):
                best[d.set_id] = d
        return list(best.values())

    def label_by_set_id(self, set_id: str) -> Optional[LabelDoc]:
        recs = list(self.iter_records("drug/label", f"set_id:{set_id}", limit=1, max_records=1))
        return parse_label(recs[0]) if recs else None

    # ---------------------------------------------------------- DailyMed
    def fetch_dailymed_history(self, set_id: str) -> pd.DataFrame:
        """Version history of an SPL set id from DailyMed (``spl_version, published_date``)."""
        r = self._get(f"{DAILYMED_BASE}/spls/{set_id}/history.json", use_key=False)
        if r.status_code == 404:
            return pd.DataFrame(columns=["set_id", "spl_version", "published_date"])
        data = r.json().get("data", {})
        hist = data.get("history", []) if isinstance(data, dict) else []
        df = pd.DataFrame(hist)
        if df.empty:
            return pd.DataFrame(columns=["set_id", "spl_version", "published_date"])
        df["set_id"] = set_id
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
        return df[["set_id", "spl_version", "published_date"]].sort_values("spl_version").reset_index(drop=True)

    def fetch_dailymed_spl_xml(self, set_id: str, version: Optional[int] = None) -> str:
        """Raw SPL XML for a set id (``version`` for an archived version, if supported)."""
        params = {"version": int(version)} if version is not None else None
        r = self._get(f"{DAILYMED_BASE}/spls/{set_id}.xml", params=params, use_key=False)
        if r.status_code == 404:
            return ""
        return r.text


def labels_to_frame(docs: Iterable[LabelDoc]) -> pd.DataFrame:
    rows = []
    for d in docs:
        rows.append(
            {
                "set_id": d.set_id,
                "spl_id": d.spl_id,
                "version": d.version,
                "effective_time": d.effective_time,
                "generic_name": d.generic_name,
                "brand_name": d.brand_names[0] if d.brand_names else None,
                "application_number": d.application_numbers[0] if d.application_numbers else None,
                "product_type": d.product_type,
                "n_safety_chars": len(d.safety_text()),
                "has_postmarketing": bool(d.postmarketing_text),
            }
        )
    return pd.DataFrame(rows)
