"""Minimal, rate-limited openFDA client (FAERS ``drug/event``, ``drug/label``,
CAERS ``food/event``) with cursor pagination.

openFDA facts this module relies on (https://open.fda.gov/apis/):

* Base URL ``https://api.fda.gov/<endpoint>.json`` with query params ``search``,
  ``count``, ``limit``, ``skip``, ``sort`` and ``api_key``.
* Record queries return at most ``limit=1000`` records per page. ``skip`` is
  capped at 25 000, so deep pagination must follow the
  ``Link: <url>; rel="next"`` response header (a ``search_after`` cursor).
  :meth:`OpenFDAClient.iter_records` follows that header automatically and
  falls back to ``skip`` only when the header is absent.
* ``count=<field>`` returns up to 1000 ``{"term", "count"}`` buckets (or
  ``{"time", "count"}`` for date fields).
* A search that matches nothing returns HTTP 404 with ``{"error": {"code":
  "NOT_FOUND"}}``; this is treated as an empty result, not an error.
* Rate limits: 40 requests/min and 1000/day without a key; 240/min and
  120 000/day with a free key passed through the ``OPENFDA_API_KEY``
  environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

OPENFDA_BASE_URL = "https://api.fda.gov"
_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')

SEX_LABELS: Dict[str, str] = {"0": "unknown", "1": "male", "2": "female"}
#: FAERS ``patientonsetageunit`` codes -> multiplier to years.
AGE_UNIT_TO_YEARS: Dict[str, float] = {
    "800": 10.0,
    "801": 1.0,
    "802": 1.0 / 12.0,
    "803": 1.0 / 52.18,
    "804": 1.0 / 365.25,
    "805": 1.0 / (365.25 * 24.0),
}


class OpenFDAError(RuntimeError):
    """Raised when openFDA returns a non-retryable error."""


@dataclass
class OpenFDAClient:
    """Synchronous openFDA client with a sliding-window rate limiter and retries.

    Parameters
    ----------
    api_key:
        openFDA API key; defaults to ``os.environ["OPENFDA_API_KEY"]``.
    max_per_minute:
        Requests per rolling 60 s window (240 with a key, 40 without).
    timeout, max_retries:
        Per-request timeout and retries on 429/5xx/connection errors.
    session:
        Optional ``requests.Session``-like object (injected in tests). It only
        needs a ``get(url, params=, timeout=)`` method returning an object with
        ``status_code``, ``headers``, ``text`` and ``json()``.
    """

    api_key: Optional[str] = None
    max_per_minute: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 5
    base_url: str = OPENFDA_BASE_URL
    session: Any = None
    user_agent: str = "peds_offlabel/0.1 (research)"
    _timestamps: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENFDA_API_KEY") or None
        if self.max_per_minute is None:
            self.max_per_minute = 240 if self.api_key else 40
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": self.user_agent})

    # ------------------------------------------------------------------ core
    def _throttle(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > 60.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= int(self.max_per_minute):
            sleep_for = 60.0 - (now - self._timestamps[0]) + 0.05
            logger.debug("rate limit reached; sleeping %.1fs", sleep_for)
            time.sleep(max(sleep_for, 0.0))
        self._timestamps.append(time.monotonic())

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(params or {})
        if self.api_key and "api_key" not in params and "api_key=" not in url:
            params["api_key"] = self.api_key
        backoff = 1.0
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("request error (%s); retry %d", exc, attempt + 1)
            else:
                if resp.status_code == 404 or resp.status_code < 400:
                    return resp
                if resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning("HTTP %s from openFDA; retry %d", resp.status_code, attempt + 1)
                else:
                    raise OpenFDAError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        raise OpenFDAError(f"giving up after {self.max_retries} retries: {last_exc}")

    def endpoint_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.strip('/')}.json"

    # ----------------------------------------------------------------- counts
    def count(self, endpoint: str, search: Optional[str], count_field: str, limit: int = 1000, exact: bool = True) -> List[Dict[str, Any]]:
        """Run a ``count`` query; returns ``[{"term"|"time": ..., "count": ...}]``.

        ``.exact`` is appended to non-date string fields so multi-word terms
        are not tokenised (openFDA convention).
        """
        fld = count_field
        is_date = "date" in fld.split(".")[-1].lower()
        if exact and not is_date and not fld.endswith(".exact"):
            fld += ".exact"
        params: Dict[str, Any] = {"count": fld, "limit": int(limit)}
        if search:
            params["search"] = search
        resp = self._get(self.endpoint_url(endpoint), params)
        if resp.status_code == 404:
            return []
        return list(resp.json().get("results", []))

    def total(self, endpoint: str, search: Optional[str]) -> int:
        """Number of records matching ``search`` (``meta.results.total``)."""
        params: Dict[str, Any] = {"limit": 1}
        if search:
            params["search"] = search
        resp = self._get(self.endpoint_url(endpoint), params)
        if resp.status_code == 404:
            return 0
        return int(resp.json().get("meta", {}).get("results", {}).get("total", 0))

    # ---------------------------------------------------------------- records
    def iter_records(
        self,
        endpoint: str,
        search: Optional[str] = None,
        limit: int = 100,
        max_records: Optional[int] = None,
        sort: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate over records, following the ``Link`` header for deep pages."""
        url = self.endpoint_url(endpoint)
        params: Dict[str, Any] = {"limit": int(min(limit, 1000))}
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        n = 0
        skip = 0
        next_url: Optional[str] = None
        while True:
            resp = self._get(next_url, None) if next_url else self._get(url, {**params, "skip": skip})
            if resp.status_code == 404:
                return
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                return
            for rec in results:
                yield rec
                n += 1
                if max_records is not None and n >= max_records:
                    return
            link = resp.headers.get("Link") or resp.headers.get("link")
            m = _LINK_NEXT_RE.search(link) if link else None
            if m:
                next_url = m.group(1)
                continue
            if next_url is not None:
                # we were following a search_after cursor chain and it ended
                return
            skip += len(results)
            total = payload.get("meta", {}).get("results", {}).get("total")
            if len(results) < params["limit"] or skip >= 25000 or (total is not None and skip >= int(total)):
                return

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def quote(term: str) -> str:
        """Quote a multi-word value for an openFDA ``search`` clause."""
        return '"' + term.replace('"', "") + '"'

    @staticmethod
    def date_range(start: str, end: str, field_name: str = "receivedate") -> str:
        return f"{field_name}:[{start.replace('-', '')} TO {end.replace('-', '')}]"


def age_in_years(age: Any, unit: Any) -> Optional[float]:
    """Convert FAERS ``patientonsetage`` + unit code to years (None if unusable)."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    factor = AGE_UNIT_TO_YEARS.get(str(unit))
    if factor is None:
        return None
    yrs = a * factor
    return yrs if 0 <= yrs <= 120 else None


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    """Write an iterable of dicts as JSONL; returns the number of lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
