"""Minimal openFDA client for the device endpoints, plus MAUDE record flattening.

openFDA facts used here (https://open.fda.gov/apis/):

* ``GET https://api.fda.gov/<endpoint>.json`` with ``search``, ``count``, ``limit`` (<= 1000), ``skip`` (<= 25000),
  ``sort`` and ``api_key``.
* ``count=<field>`` returns ``results: [{"term": ..., "count": ...}]``.
* Deep pagination uses the ``Link: <url>; rel="next"`` response header (``search_after`` cursor).
* A query with no matches returns HTTP 404 with ``error.code == "NOT_FOUND"``; this is not an error here.
* Rate limits: 40/min and 1,000/day without a key; 240/min and 120,000/day with ``OPENFDA_API_KEY``.
* ``GET https://api.fda.gov/download.json`` lists bulk partitions per endpoint.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENFDA_BASE_URL = "https://api.fda.gov"
_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


class OpenFDAError(RuntimeError):
    """Non-retryable openFDA error."""


@dataclass
class OpenFDAClient:
    """Synchronous client with a sliding-window rate limiter, retries and cursor pagination.

    ``session`` can be injected (tests use a fake with a ``get`` method).
    """

    api_key: Optional[str] = None
    max_per_minute: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 5
    base_url: str = OPENFDA_BASE_URL
    session: Optional[Any] = None
    _times: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENFDA_API_KEY") or None
        if self.max_per_minute is None:
            self.max_per_minute = 240 if self.api_key else 40
        if self.session is None:
            self.session = requests.Session()

    # ------------------------------------------------------------------ low level
    def _throttle(self) -> None:
        now = time.monotonic()
        while self._times and now - self._times[0] > 60.0:
            self._times.popleft()
        if len(self._times) >= self.max_per_minute:
            time.sleep(max(0.0, 60.0 - (now - self._times[0])) + 0.05)
        self._times.append(time.monotonic())

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise OpenFDAError(str(exc)) from exc
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            return r
        raise OpenFDAError("unreachable")

    def _endpoint_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.strip('/')}.json"

    # ------------------------------------------------------------------ queries
    def count(self, endpoint: str, search: Optional[str], field_name: str, limit: int = 1000) -> pd.DataFrame:
        """``count`` query -> DataFrame(term, count); empty frame when nothing matches."""
        params: Dict[str, Any] = {"count": field_name, "limit": limit}
        if search:
            params["search"] = search
        r = self._get(self._endpoint_url(endpoint), params)
        if r.status_code == 404:
            return pd.DataFrame(columns=["term", "count"])
        if r.status_code != 200:
            raise OpenFDAError(f"{r.status_code}: {r.text[:200]}")
        res = r.json().get("results", [])
        return pd.DataFrame(res).rename(columns={"time": "term"}) if res else pd.DataFrame(columns=["term", "count"])

    def iter_records(self, endpoint: str, search: Optional[str], limit: int = 1000, max_records: Optional[int] = None, sort: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over all matching records, following the ``Link`` header and falling back to ``skip``."""
        url = self._endpoint_url(endpoint)
        params: Dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        n = 0
        skip = 0
        next_url: Optional[str] = None
        while True:
            r = self._get(next_url, None) if next_url else self._get(url, params)
            if r.status_code == 404:
                return
            if r.status_code != 200:
                raise OpenFDAError(f"{r.status_code}: {r.text[:200]}")
            results = r.json().get("results", [])
            for rec in results:
                yield rec
                n += 1
                if max_records is not None and n >= max_records:
                    return
            if not results:
                return
            m = _LINK_NEXT_RE.search(r.headers.get("Link", "") or "")
            if m:
                next_url = m.group(1)
                continue
            skip += len(results)
            if skip > 25000 or len(results) < limit:
                return
            params["skip"] = skip
            next_url = None

    def fetch_records(self, endpoint: str, search: Optional[str], max_records: Optional[int] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        return list(self.iter_records(endpoint, search, limit=min(limit, max_records or limit), max_records=max_records))

    def bulk_partitions(self, category: str = "device", endpoint: str = "event") -> List[Dict[str, Any]]:
        """Partition descriptors (``file``, ``size_mb``, ``records``) from the openFDA download manifest."""
        r = self._get(f"{self.base_url}/download.json")
        if r.status_code != 200:
            raise OpenFDAError(f"manifest: {r.status_code}")
        return r.json()["results"][category][endpoint]["partitions"]


# ---------------------------------------------------------------------------------------------- flattening
def flatten_device_event(rec: Dict[str, Any]) -> Dict[str, Any]:
    """One flat row per MDR with the fields the project uses; multi-device reports take the first device."""
    dev = (rec.get("device") or [{}])[0] or {}
    ofda = dev.get("openfda") or {}
    texts: Dict[str, List[str]] = {}
    for t in rec.get("mdr_text") or []:
        texts.setdefault(str(t.get("text_type_code", "")).strip(), []).append(str(t.get("text", "")))
    return {
        "report_number": rec.get("report_number") or rec.get("mdr_report_key"),
        "mdr_report_key": rec.get("mdr_report_key"),
        "event_type": rec.get("event_type"),
        "date_received": rec.get("date_received"),
        "date_of_event": rec.get("date_of_event"),
        "date_report": rec.get("date_report"),
        "report_source_code": rec.get("report_source_code"),
        "source_type": ";".join(rec.get("source_type") or []) if isinstance(rec.get("source_type"), list) else rec.get("source_type"),
        "brand_name": dev.get("brand_name"),
        "generic_name": dev.get("generic_name"),
        "product_code": dev.get("device_report_product_code"),
        "manufacturer_d_name": dev.get("manufacturer_d_name"),
        "model_number": dev.get("model_number"),
        "udi_di": dev.get("udi_di"),
        "device_class": ofda.get("device_class"),
        "medical_specialty": ofda.get("medical_specialty_description"),
        "product_problems": ";".join(rec.get("product_problems") or []),
        "patient_problems": ";".join((rec.get("patient") or [{}])[0].get("patient_problems") or []) if rec.get("patient") else "",
        "text_event": " ".join(texts.get("Description of Event or Problem", [])),
        "text_manufacturer": " ".join(texts.get("Additional Manufacturer Narrative", [])),
        "n_devices": rec.get("number_devices_in_event"),
    }


def flatten_events(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """DataFrame of flattened MDRs."""
    return pd.DataFrame([flatten_device_event(r) for r in records])
