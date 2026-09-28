"""Minimal, retrying client for openFDA (https://open.fda.gov/apis/).

Design choices
--------------
* Everything the panel needs is obtained with ``count`` queries (one HTTP call per
  drug x outcome-group), which sidesteps openFDA's 25,000-record ``skip`` ceiling.
* Record paging is still provided (``iter_records``), split by date windows so that no
  single window exceeds the ceiling.
* The HTTP session is injectable so tests can run offline with a fake session.

Endpoints used: ``drug/event`` (FAERS), ``drug/shortages`` (FDA Drug Shortage Database),
``drug/ndc`` (product classes). An optional API key is read from ``OPENFDA_API_KEY``.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

try:  # requests is only needed for live calls
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

LOG = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov"
MAX_SKIP = 25_000          # openFDA hard limit on skip
MAX_LIMIT = 1_000          # max records (or count buckets) per call
DATE_FMT = "%Y%m%d"


def _quote_term(term: str) -> str:
    """Quote a search value containing spaces or special characters."""
    term = term.replace('"', "")
    return f'"{term}"' if any(ch in term for ch in " -/,()+") else term


def build_faers_search(drug: Optional[str] = None, reactions: Optional[Sequence[str]] = None,
                       suspect_only: bool = True, serious: Optional[bool] = None,
                       extra: Optional[str] = None, drug_field: str = "patient.drug.openfda.generic_name") -> str:
    """Compose an openFDA ``search`` string for ``drug/event``.

    Parameters
    ----------
    drug : ingredient (matched against ``patient.drug.openfda.generic_name`` by default).
    reactions : MedDRA PTs ORed together (``reactionmeddrapt``).
    suspect_only : restrict to ``drugcharacterization:1`` (primary suspect).
    serious : ``True`` -> ``serious:1``; ``False`` -> ``serious:2``.
    extra : any additional raw clause ANDed in.

    Returns
    -------
    The search string, e.g.
    ``patient.drug.openfda.generic_name:"HEPARIN SODIUM"+AND+patient.reaction.reactionmeddrapt:("Wrong drug administered")``
    """
    clauses: List[str] = []
    if drug:
        clauses.append(f"{drug_field}:{_quote_term(drug.upper())}")
    if suspect_only:
        clauses.append("patient.drug.drugcharacterization:1")
    if reactions:
        inner = "+".join(_quote_term(r) for r in reactions)
        clauses.append(f"patient.reaction.reactionmeddrapt:({inner})")
    if serious is not None:
        clauses.append(f"serious:{1 if serious else 2}")
    if extra:
        clauses.append(extra)
    return "+AND+".join(clauses)


class OpenFDAClient:
    """Retrying openFDA client with polite pacing.

    Parameters
    ----------
    api_key : openFDA key; defaults to ``OPENFDA_API_KEY``.
    session : object with ``get(url, params=..., timeout=...)`` returning a response with
        ``status_code``, ``json()`` and ``raise_for_status()``; injectable for tests.
    min_interval : minimum seconds between calls (240/min with key -> 0.25 s).
    """

    def __init__(self, api_key: Optional[str] = None, session: Any = None, timeout: float = 60.0,
                 max_retries: int = 5, backoff: float = 1.7, min_interval: float = 0.26,
                 base_url: str = BASE_URL) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENFDA_API_KEY")
        if session is None:
            if requests is None:  # pragma: no cover
                raise ImportError("requests is required for live openFDA calls")
            session = requests.Session()
            session.headers.update({"User-Agent": "shortage_ae/0.1 (research)"})
        self.session = session
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.min_interval = min_interval
        self.base_url = base_url.rstrip("/")
        self._last_call = 0.0

    # ------------------------------------------------------------------ low level
    def _pace(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """GET ``<base>/<endpoint>.json`` with retries on 429/5xx. 404 (no matches) -> empty results."""
        url = f"{self.base_url}/{endpoint.strip('/')}.json"
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        # openFDA wants '+' and quotes literally in `search`; requests would percent-encode '+'
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full = f"{url}?{query}"
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                resp = self.session.get(full, timeout=self.timeout)
                if resp.status_code == 404:
                    return {"meta": {"results": {"total": 0}}, "results": []}
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                last_exc = exc
                wait = self.backoff ** attempt
                LOG.warning("openFDA call failed (%s); retry %d/%d in %.1fs", exc, attempt + 1,
                            self.max_retries, wait)
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------ counts
    def count(self, endpoint: str, search: str, field: str, limit: int = MAX_LIMIT) -> pd.DataFrame:
        """Run a ``count`` query. Returns a DataFrame with ``term``/``count`` or ``time``/``count``."""
        data = self.get(endpoint, {"search": search, "count": field, "limit": min(limit, MAX_LIMIT)})
        rows = data.get("results", [])
        if not rows:
            return pd.DataFrame(columns=["term", "count"])
        return pd.DataFrame(rows)

    def count_by_date(self, endpoint: str, search: str, date_field: str = "receivedate",
                      start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
        """Daily counts of ``date_field``; splits the range into <=1,000-day windows (bucket cap)."""
        start = start or date(2004, 1, 1)
        end = end or date.today()
        frames: List[pd.DataFrame] = []
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=999), end)
            window = f"{date_field}:[{cur.strftime(DATE_FMT)}+TO+{nxt.strftime(DATE_FMT)}]"
            s = f"{search}+AND+{window}" if search else window
            df = self.count(endpoint, s, date_field)
            if len(df):
                frames.append(df)
            cur = nxt + timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=["date", "count"])
        out = pd.concat(frames, ignore_index=True)
        out = out.rename(columns={"time": "date"})
        out["date"] = pd.to_datetime(out["date"], format=DATE_FMT)
        return out.groupby("date", as_index=False)["count"].sum().sort_values("date")

    # ------------------------------------------------------------------ records
    def iter_records(self, endpoint: str, search: str, limit: int = 100,
                     max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Page through records with ``skip``; stops at openFDA's 25,000 ceiling with a warning."""
        skip, seen = 0, 0
        while True:
            data = self.get(endpoint, {"search": search, "limit": min(limit, MAX_LIMIT), "skip": skip})
            rows = data.get("results", [])
            if not rows:
                return
            for r in rows:
                yield r
                seen += 1
                if max_records is not None and seen >= max_records:
                    return
            skip += len(rows)
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            if skip >= total:
                return
            if skip >= MAX_SKIP:
                LOG.warning("skip ceiling reached for search=%s (total=%s); narrow the date window", search, total)
                return

    def iter_records_by_date(self, endpoint: str, search: str, date_field: str, start: date, end: date,
                             step_days: int = 30, limit: int = 100,
                             max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Page through records in successive date windows so each stays under the skip ceiling."""
        n = 0
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=step_days - 1), end)
            window = f"{date_field}:[{cur.strftime(DATE_FMT)}+TO+{nxt.strftime(DATE_FMT)}]"
            s = f"{search}+AND+{window}" if search else window
            for rec in self.iter_records(endpoint, s, limit=limit):
                yield rec
                n += 1
                if max_records is not None and n >= max_records:
                    return
            cur = nxt + timedelta(days=1)

    # ------------------------------------------------------------------ shortages
    def shortage_records(self, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
        """All records of ``drug/shortages`` (a few hundred to a few thousand)."""
        return list(self.iter_records("drug/shortages", search="", limit=MAX_LIMIT, max_records=max_records))


def flatten_shortage_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the scalar fields the panel needs from one ``drug/shortages`` record."""
    keep = ("generic_name", "proprietary_name", "company_name", "presentation", "status",
            "initial_posting_date", "update_date", "update_type", "shortage_reason",
            "therapeutic_category", "availability", "resolved_note")
    out = {k: rec.get(k) for k in keep}
    of = rec.get("openfda", {}) or {}
    out["route"] = ";".join(of.get("route", []) or [])
    out["pharm_class_epc"] = ";".join(of.get("pharm_class_epc", []) or [])
    return out


def records_to_frame(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([flatten_shortage_record(r) for r in records])
