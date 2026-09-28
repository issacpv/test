"""Europe PMC REST client (open, no API key) and full-text XML to plain text.

Endpoints
---------
``GET https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=...&format=json&pageSize=100&cursorMark=*``
``GET https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML``
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterator, Optional

import requests

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMC:
    def __init__(self, timeout: float = 60.0, retries: int = 4, session: Optional[requests.Session] = None) -> None:
        self.s = session or requests.Session()
        self.s.headers.update({"Accept": "application/json", "User-Agent": "spine_mining/0.1"})
        self.timeout, self.retries = timeout, retries

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None, json_out: bool = True):
        for attempt in range(self.retries + 1):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r.json() if json_out else r.text
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == self.retries:
                    raise
                time.sleep(2.0 ** attempt)
        raise RuntimeError("unreachable")

    def search(self, query: str, page_size: int = 100, max_results: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over search hits using ``cursorMark`` pagination (stable for deep paging)."""
        cursor = "*"
        n = 0
        while True:
            data = self._get(f"{BASE}/search", {"query": query, "format": "json", "pageSize": page_size,
                                                "cursorMark": cursor, "resultType": "lite"})
            hits = data.get("resultList", {}).get("result", [])
            if not hits:
                return
            for h in hits:
                yield {"id": h.get("id"), "source": h.get("source"), "pmid": h.get("pmid"), "pmcid": h.get("pmcid"),
                       "doi": h.get("doi"), "title": h.get("title"), "year": h.get("pubYear"), "journal": h.get("journalTitle"),
                       "is_oa": h.get("isOpenAccess"), "has_ft": h.get("hasTextMinedTerms") or h.get("inEPMC")}
                n += 1
                if max_results is not None and n >= max_results:
                    return
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                return
            cursor = next_cursor

    def full_text_xml(self, pmcid: str) -> str:
        return self._get(f"{BASE}/{pmcid}/fullTextXML", json_out=False)


def _text_of(el: ET.Element) -> str:
    return " ".join(t.strip() for t in el.itertext() if t and t.strip())


def xml_to_text(xml: str) -> str:
    """Body paragraphs, section titles, figure captions and table cells of a JATS full-text XML, as plain text.

    Paragraphs are separated by newlines so that sentence boundaries survive; table cells are
    joined with ' | ' within a row so numeric cells keep their column context.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    chunks = []
    body = root.find(".//body")
    scope = body if body is not None else root
    for el in scope.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("p", "title"):
            if any(a.tag.split("}")[-1] == "table" for a in el.iter()):
                continue
            t = _text_of(el)
            if t:
                chunks.append(t)
        elif tag == "tr":
            cells = [_text_of(c) for c in el if c.tag.split("}")[-1] in ("td", "th")]
            if cells:
                chunks.append(" | ".join(cells))
    for cap in root.iter():
        if cap.tag.split("}")[-1] == "caption" and (body is None or cap not in scope.iter()):
            t = _text_of(cap)
            if t:
                chunks.append(t)
    return "\n".join(chunks)


__all__ = ["EuropePMC", "xml_to_text", "BASE"]
