"""Quarterly drug x PT report counts from openFDA ``drug/event`` using ``count`` queries only.

For a 2x2 table per (drug, PT, quarter) we need four count series by ``receivedate``:
``a``: drug & PT, ``drug_total``: drug (any PT), ``pt_total``: PT (any drug), ``N``: all reports.
Then ``b = drug_total - a``, ``c = pt_total - a``, ``d = N - a - b - c``.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

LOG = logging.getLogger(__name__)
BASE_URL = "https://api.fda.gov/drug/event.json"
DATE_FMT = "%Y%m%d"


def _q(term: str) -> str:
    term = term.replace('"', "")
    return f'"{term}"' if any(ch in term for ch in " -/,()+'") else term


def search_clause(drug: Optional[str] = None, pt: Optional[str] = None, suspect_only: bool = True,
                  serious: Optional[bool] = None, hcp_only: bool = False) -> str:
    parts: List[str] = []
    if drug:
        parts.append(f"patient.drug.openfda.generic_name:{_q(drug.upper())}")
        if suspect_only:
            parts.append("patient.drug.drugcharacterization:1")
    if pt:
        parts.append(f"patient.reaction.reactionmeddrapt:{_q(pt)}")
    if serious is not None:
        parts.append(f"serious:{1 if serious else 2}")
    if hcp_only:
        parts.append("primarysource.qualification:(1+2+3)")
    return "+AND+".join(parts)


class OpenFDACounts:
    """Daily/quarterly count retrieval with retries, pacing and an injectable session."""

    def __init__(self, api_key: Optional[str] = None, session: Any = None, timeout: float = 60.0,
                 min_interval: float = 0.26, max_retries: int = 5) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENFDA_API_KEY")
        if session is None:
            if requests is None:  # pragma: no cover
                raise ImportError("requests is required for live openFDA calls")
            session = requests.Session()
            session.headers.update({"User-Agent": "label_change/0.1 (research)"})
        self.session = session
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.api_key:
            params = {**params, "api_key": self.api_key}
        url = BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    return {"results": []}
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries:
                    raise
                LOG.warning("openFDA call failed (%s); retrying", exc)
                time.sleep(1.7 ** attempt)
        raise RuntimeError("unreachable")

    def daily_counts(self, search: str, start: date, end: date) -> pd.DataFrame:
        """Daily report counts (``date,count``) for a search, in <=1,000-day windows."""
        frames: List[pd.DataFrame] = []
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=999), end)
            window = f"receivedate:[{cur.strftime(DATE_FMT)}+TO+{nxt.strftime(DATE_FMT)}]"
            s = f"{search}+AND+{window}" if search else window
            rows = self._get({"search": s, "count": "receivedate", "limit": 1000}).get("results", [])
            if rows:
                frames.append(pd.DataFrame(rows))
            cur = nxt + timedelta(days=1)
        if not frames:
            return pd.DataFrame({"date": pd.to_datetime([]), "count": pd.Series([], dtype=int)})
        df = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"], format=DATE_FMT)
        return df.groupby("date", as_index=False)["count"].sum()

    def top_pts(self, drug: str, limit: int = 300) -> pd.DataFrame:
        """Most frequently reported PTs for a drug (``term,count``)."""
        rows = self._get({"search": search_clause(drug=drug), "count": "patient.reaction.reactionmeddrapt.exact",
                          "limit": min(limit, 1000)}).get("results", [])
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["term", "count"])


def daily_to_quarterly(daily: pd.DataFrame) -> pd.Series:
    """Sum a ``date,count`` table to a Series indexed by quarter period (e.g. ``2019Q3``)."""
    if daily is None or len(daily) == 0:
        return pd.Series(dtype=int)
    d = daily.copy()
    d["quarter"] = pd.to_datetime(d["date"]).dt.to_period("Q")
    return d.groupby("quarter")["count"].sum().astype(int)
