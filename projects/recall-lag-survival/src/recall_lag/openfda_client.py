"""openFDA client for device endpoints and minimal record flattening for MAUDE, recall and enforcement.

openFDA facts (https://open.fda.gov/apis/): endpoints ``https://api.fda.gov/<endpoint>.json`` with ``search``,
``count``, ``limit`` (<= 1000), ``skip`` (<= 25000), ``sort``, ``api_key``; deep pagination through the
``Link: <url>; rel="next"`` header; HTTP 404 with ``NOT_FOUND`` for empty result sets; bulk partitions listed
by ``https://api.fda.gov/download.json``; 40 requests/min without a key, 240/min with ``OPENFDA_API_KEY``.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import requests

OPENFDA_BASE_URL = "https://api.fda.gov"
_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
SERIOUS_EVENT_TYPES = ("Death", "Injury")


class OpenFDAError(RuntimeError):
    """Non-retryable openFDA error."""


@dataclass
class OpenFDAClient:
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

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.strip('/')}.json"

    def count(self, endpoint: str, search: Optional[str], field_name: str, limit: int = 1000) -> pd.DataFrame:
        params: Dict[str, Any] = {"count": field_name, "limit": limit}
        if search:
            params["search"] = search
        r = self._get(self._url(endpoint), params)
        if r.status_code == 404:
            return pd.DataFrame(columns=["term", "count"])
        if r.status_code != 200:
            raise OpenFDAError(f"{r.status_code}: {r.text[:200]}")
        res = r.json().get("results", [])
        return pd.DataFrame(res).rename(columns={"time": "term"}) if res else pd.DataFrame(columns=["term", "count"])

    def iter_records(self, endpoint: str, search: Optional[str], limit: int = 1000, max_records: Optional[int] = None, sort: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        url = self._url(endpoint)
        params: Dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        n, skip, next_url = 0, 0, None
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

    def fetch_records(self, endpoint: str, search: Optional[str], max_records: Optional[int] = None, limit: int = 1000, sort: Optional[str] = None) -> List[Dict[str, Any]]:
        return list(self.iter_records(endpoint, search, limit=min(limit, max_records or limit), max_records=max_records, sort=sort))

    def bulk_partitions(self, category: str = "device", endpoint: str = "event") -> List[Dict[str, Any]]:
        r = self._get(f"{self.base_url}/download.json")
        if r.status_code != 200:
            raise OpenFDAError(f"manifest: {r.status_code}")
        return r.json()["results"][category][endpoint]["partitions"]


# ---------------------------------------------------------------------------------------------- flattening
def _date(s: Any) -> Optional[pd.Timestamp]:
    if s is None or s == "":
        return None
    t = pd.to_datetime(str(s), format="%Y%m%d", errors="coerce")
    if pd.isna(t):
        t = pd.to_datetime(str(s), errors="coerce")
    return None if pd.isna(t) else t


def flatten_event_min(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal MAUDE row: dates, event type, product code, manufacturer, brand (first device)."""
    dev = (rec.get("device") or [{}])[0] or {}
    return {
        "report_number": rec.get("report_number") or rec.get("mdr_report_key"),
        "date_received": _date(rec.get("date_received")),
        "date_of_event": _date(rec.get("date_of_event")),
        "date_report": _date(rec.get("date_report")),
        "event_type": rec.get("event_type"),
        "serious": rec.get("event_type") in SERIOUS_EVENT_TYPES,
        "product_code": (dev.get("device_report_product_code") or "").upper() or None,
        "manufacturer": dev.get("manufacturer_d_name"),
        "brand_name": dev.get("brand_name"),
        "report_source_code": rec.get("report_source_code"),
    }


def flatten_recall(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal ``device/recall`` row."""
    ofda = rec.get("openfda") or {}
    return {
        "res_event_number": rec.get("res_event_number"),
        "product_res_number": rec.get("product_res_number"),
        "initiated": _date(rec.get("event_date_initiated")),
        "posted": _date(rec.get("event_date_posted")),
        "terminated": _date(rec.get("event_date_terminated")),
        "product_code": (rec.get("product_code") or "").upper() or None,
        "recalling_firm": rec.get("recalling_firm"),
        "root_cause": rec.get("root_cause_description"),
        "k_numbers": ";".join(rec.get("k_numbers") or []),
        "pma_numbers": ";".join(rec.get("pma_numbers") or []),
        "device_class": ofda.get("device_class"),
        "recall_status": rec.get("recall_status"),
    }


def flatten_enforcement(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal ``device/enforcement`` row (classification and FDA dates)."""
    return {
        "res_event_number": rec.get("event_id"),
        "product_res_number": rec.get("recall_number"),
        "classification": rec.get("classification"),
        "initiation": _date(rec.get("recall_initiation_date")),
        "center_classification": _date(rec.get("center_classification_date")),
        "report_date": _date(rec.get("report_date")),
        "recalling_firm": rec.get("recalling_firm"),
        "product_description": rec.get("product_description"),
    }
