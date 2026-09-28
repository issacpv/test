"""openFDA device endpoints: client, record flatteners and the recall-class join.

Endpoints (https://open.fda.gov/apis/device/):

==================  =====================================================
``device/event``    MAUDE medical device reports (narratives in ``mdr_text``)
``device/recall``   Recall records (``res_event_number``, ``k_numbers``, ``pma_numbers``)
``device/enforcement``  Enforcement reports carrying the recall ``classification``
                    (Class I/II/III); join key ``event_id`` == recall ``res_event_number``
``device/510k``     Premarket notifications (``k_number``, ``decision_date``, ``product_code``)
``device/pma``      Premarket approvals and supplements
``device/classification``  Product-code dictionary (class, panel, implant flags)
``device/udi``      GUDID device identifiers with ``premarket_submissions``
==================  =====================================================

Pagination follows the ``Link: <...>; rel="next"`` header (search_after) so
that queries larger than the 25 000 ``skip`` cap can be exhausted; bulk JSON
partitions from https://api.fda.gov/download.json are still preferred for
full-database work (see ``scripts/download_data.py``).
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov"
MAUDE = "device/event"
RECALL = "device/recall"
ENFORCEMENT = "device/enforcement"
K510 = "device/510k"
PMA = "device/pma"
CLASSIFICATION = "device/classification"
UDI = "device/udi"

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
_DATE_FIELDS = ("date", "time")


class OpenFDAError(RuntimeError):
    """Non-retryable openFDA error."""


@dataclass
class DeviceClient:
    """Synchronous openFDA client with rolling-window rate limiting and retries."""

    api_key: Optional[str] = None
    max_per_minute: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 5
    base_url: str = BASE_URL
    session: Optional[requests.Session] = None
    _stamps: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENFDA_API_KEY") or None
        if self.max_per_minute is None:
            self.max_per_minute = 240 if self.api_key else 40
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": "device_recall/0.1 (research)"})

    def _throttle(self) -> None:
        now = time.monotonic()
        while self._stamps and now - self._stamps[0] > 60.0:
            self._stamps.popleft()
        if len(self._stamps) >= int(self.max_per_minute):
            time.sleep(max(60.0 - (now - self._stamps[0]) + 0.05, 0.0))
        self._stamps.append(time.monotonic())

    def _get(self, url: str, params: Optional[Dict[str, Any]]) -> requests.Response:
        params = dict(params or {})
        if self.api_key and "api_key" not in params and "api_key=" not in url:
            params["api_key"] = self.api_key
        backoff = 1.0
        last: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
            else:
                if resp.status_code == 404 or resp.status_code < 400:
                    return resp
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise OpenFDAError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        raise OpenFDAError(f"giving up: {last}")

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.strip('/')}.json"

    def count(self, endpoint: str, search: Optional[str], count_field: str, limit: int = 1000, exact: bool = True) -> pd.DataFrame:
        """``count`` query -> DataFrame(term|time, count)."""
        f = count_field
        is_date = any(f.split(".")[-1].lower().startswith(x) or f.split(".")[-1].lower().endswith(x) for x in _DATE_FIELDS)
        if exact and not is_date and not f.endswith(".exact"):
            f += ".exact"
        params: Dict[str, Any] = {"count": f, "limit": int(limit)}
        if search:
            params["search"] = search
        resp = self._get(self._url(endpoint), params)
        if resp.status_code == 404:
            return pd.DataFrame(columns=["time" if is_date else "term", "count"])
        df = pd.DataFrame(resp.json().get("results", []))
        if df.empty:
            return pd.DataFrame(columns=["time" if is_date else "term", "count"])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d", errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
        return df

    def iter_records(
        self,
        endpoint: str,
        search: Optional[str] = None,
        limit: int = 100,
        max_records: Optional[int] = None,
        sort: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield records, following ``Link`` (search_after) for deep pagination."""
        url = self._url(endpoint)
        params: Dict[str, Any] = {"limit": int(min(limit, 1000))}
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        n, skip, next_url, followed = 0, 0, None, False
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
                next_url, followed = m.group(1), True
                continue
            if followed:
                return
            skip += len(results)
            total = payload.get("meta", {}).get("results", {}).get("total")
            if skip >= 25000 or (total is not None and skip >= int(total)) or len(results) < params["limit"]:
                return
            next_url = None

    def fetch(self, endpoint: str, search: Optional[str] = None, **kw: Any) -> List[Dict[str, Any]]:
        return list(self.iter_records(endpoint, search=search, **kw))

    @staticmethod
    def quote(v: str) -> str:
        return '"' + str(v).replace('"', "") + '"'

    @staticmethod
    def date_range(field_name: str, start: str, end: str) -> str:
        return f"{field_name}:[{start.replace('-', '')} TO {end.replace('-', '')}]"

    def maude_monthly_counts(self, search: str, start: str, end: str, date_field: str = "date_received") -> pd.DataFrame:
        """Monthly MAUDE report counts for a search clause (one call per year)."""
        frames = []
        for y in range(int(start[:4]), int(end[:4]) + 1):
            ys, ye = max(f"{y}-01-01", start), min(f"{y}-12-31", end)
            clause = f"{search} AND {self.date_range(date_field, ys, ye)}" if search else self.date_range(date_field, ys, ye)
            df = self.count(MAUDE, clause, date_field, exact=False)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["month", "count"])
        d = pd.concat(frames)
        d["month"] = d["time"].dt.to_period("M").dt.to_timestamp()
        m = d.groupby("month", as_index=False)["count"].sum()
        idx = pd.date_range(m["month"].min(), m["month"].max(), freq="MS")
        return m.set_index("month").reindex(idx, fill_value=0).rename_axis("month").reset_index()


# ------------------------------------------------------------- flatteners
def _first(x: Any) -> Any:
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _date(s: Any) -> Optional[pd.Timestamp]:
    if not s:
        return None
    t = pd.to_datetime(str(s), format="%Y%m%d", errors="coerce")
    return None if pd.isna(t) else t


def flatten_maude(rec: Dict[str, Any]) -> Dict[str, Any]:
    """One MAUDE MDR -> flat row (first device only; narratives concatenated)."""
    dev = (rec.get("device") or [{}])[0] or {}
    ofda = dev.get("openfda") or {}
    texts = rec.get("mdr_text") or []
    event_texts = [t.get("text", "") for t in texts if str(t.get("text_type_code", "")).lower().startswith("description")]
    mfr_texts = [t.get("text", "") for t in texts if "manufacturer" in str(t.get("text_type_code", "")).lower()]
    outcomes: List[str] = []
    for p in rec.get("patient") or []:
        outcomes.extend(p.get("sequence_number_outcome") or [])
    return {
        "mdr_report_key": rec.get("mdr_report_key"),
        "report_number": rec.get("report_number"),
        "date_received": _date(rec.get("date_received")),
        "date_of_event": _date(rec.get("date_of_event")),
        "event_type": rec.get("event_type"),
        "report_source_code": rec.get("report_source_code"),
        "source_type": "|".join(rec.get("source_type") or []),
        "adverse_event_flag": rec.get("adverse_event_flag"),
        "product_problem_flag": rec.get("product_problem_flag"),
        "product_problems": rec.get("product_problems") or [],
        "brand_name": dev.get("brand_name"),
        "generic_name": dev.get("generic_name"),
        "product_code": dev.get("device_report_product_code"),
        "manufacturer_d_name": dev.get("manufacturer_d_name"),
        "model_number": dev.get("model_number"),
        "udi_di": dev.get("udi_di"),
        "device_class": _first(ofda.get("device_class")),
        "regulation_number": _first(ofda.get("regulation_number")),
        "medical_specialty": _first(ofda.get("medical_specialty_description")),
        "n_devices": rec.get("number_devices_in_event"),
        "patient_outcomes": outcomes,
        "narrative": " ".join(event_texts).strip(),
        "manufacturer_narrative": " ".join(mfr_texts).strip(),
    }


def flatten_510k(rec: Dict[str, Any]) -> Dict[str, Any]:
    ofda = rec.get("openfda") or {}
    return {
        "k_number": rec.get("k_number"),
        "applicant": rec.get("applicant"),
        "device_name": rec.get("device_name"),
        "product_code": rec.get("product_code"),
        "decision_code": rec.get("decision_code"),
        "decision_date": _date(rec.get("decision_date")),
        "date_received": _date(rec.get("date_received")),
        "clearance_type": rec.get("clearance_type"),
        "statement_or_summary": rec.get("statement_or_summary"),
        "third_party_flag": rec.get("third_party_flag"),
        "expedited_review_flag": rec.get("expedited_review_flag"),
        "review_advisory_committee": rec.get("review_advisory_committee"),
        "device_class": _first(ofda.get("device_class")),
        "regulation_number": _first(ofda.get("regulation_number")),
        "medical_specialty": _first(ofda.get("medical_specialty_description")),
        "country_code": rec.get("country_code"),
    }


def flatten_pma(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pma_number": rec.get("pma_number"),
        "supplement_number": rec.get("supplement_number"),
        "supplement_type": rec.get("supplement_type"),
        "applicant": rec.get("applicant"),
        "trade_name": rec.get("trade_name"),
        "generic_name": rec.get("generic_name"),
        "product_code": rec.get("product_code"),
        "decision_code": rec.get("decision_code"),
        "decision_date": _date(rec.get("decision_date")),
        "advisory_committee": rec.get("advisory_committee"),
    }


def flatten_recall(rec: Dict[str, Any]) -> Dict[str, Any]:
    ofda = rec.get("openfda") or {}
    return {
        "res_event_number": rec.get("res_event_number"),
        "product_res_number": rec.get("product_res_number"),
        "product_code": rec.get("product_code"),
        "k_numbers": rec.get("k_numbers") or [],
        "pma_numbers": rec.get("pma_numbers") or [],
        "recalling_firm": rec.get("recalling_firm"),
        "root_cause_description": rec.get("root_cause_description"),
        "product_description": rec.get("product_description"),
        "reason_for_recall": rec.get("reason_for_recall"),
        "event_date_initiated": _date(rec.get("event_date_initiated")),
        "event_date_posted": _date(rec.get("event_date_posted")),
        "recall_status": rec.get("recall_status"),
        "device_class": _first(ofda.get("device_class")),
        "device_name": _first(ofda.get("device_name")),
    }


def flatten_enforcement(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "recall_number": rec.get("recall_number"),
        "event_id": rec.get("event_id"),
        "classification": rec.get("classification"),
        "status": rec.get("status"),
        "recalling_firm": rec.get("recalling_firm"),
        "product_description": rec.get("product_description"),
        "reason_for_recall": rec.get("reason_for_recall"),
        "recall_initiation_date": _date(rec.get("recall_initiation_date")),
        "center_classification_date": _date(rec.get("center_classification_date")),
        "voluntary_mandated": rec.get("voluntary_mandated"),
    }


def flatten_classification(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "product_code": rec.get("product_code"),
        "device_name": rec.get("device_name"),
        "device_class": rec.get("device_class"),
        "regulation_number": rec.get("regulation_number"),
        "medical_specialty": rec.get("medical_specialty"),
        "medical_specialty_description": rec.get("medical_specialty_description"),
        "review_panel": rec.get("review_panel"),
        "submission_type_id": rec.get("submission_type_id"),
        "implant_flag": rec.get("implant_flag"),
        "life_sustain_support_flag": rec.get("life_sustain_support_flag"),
        "gmp_exempt_flag": rec.get("gmp_exempt_flag"),
        "definition": rec.get("definition"),
    }


def flatten_udi(rec: Dict[str, Any]) -> Dict[str, Any]:
    ids = rec.get("identifiers") or []
    primary = next((i for i in ids if str(i.get("type", "")).lower() == "primary"), ids[0] if ids else {})
    subs = rec.get("premarket_submissions") or []
    codes = rec.get("product_codes") or []
    return {
        "public_device_record_key": rec.get("public_device_record_key"),
        "primary_di": primary.get("id"),
        "brand_name": rec.get("brand_name"),
        "version_or_model_number": rec.get("version_or_model_number"),
        "company_name": rec.get("company_name"),
        "device_description": rec.get("device_description"),
        "product_codes": [c.get("code") for c in codes],
        "submission_numbers": [s.get("submission_number") for s in subs],
        "is_single_use": rec.get("is_single_use"),
        "publish_date": _date(rec.get("publish_date")),
    }


def frame(records: Iterable[Dict[str, Any]], flattener) -> pd.DataFrame:
    return pd.DataFrame([flattener(r) for r in records])


def join_recall_class(recalls: pd.DataFrame, enforcement: pd.DataFrame) -> pd.DataFrame:
    """Attach the enforcement ``classification`` (Class I/II/III) to recall rows.

    Join key: ``recall.res_event_number`` == ``enforcement.event_id``. One recall
    event can have several enforcement lines (one per product); we keep the most
    severe class per event and the earliest initiation date.
    """
    sev = {"Class I": 1, "Class II": 2, "Class III": 3}
    e = enforcement.copy()
    e["class_rank"] = e["classification"].map(sev).fillna(9)
    e = e.sort_values(["event_id", "class_rank", "recall_initiation_date"]).drop_duplicates("event_id")
    e = e.rename(columns={"event_id": "res_event_number", "classification": "recall_class"})
    keep = ["res_event_number", "recall_class", "recall_initiation_date", "center_classification_date"]
    out = recalls.merge(e[keep], on="res_event_number", how="left")
    out["recall_date"] = out["event_date_initiated"].fillna(out["recall_initiation_date"])
    return out


def explode_recall_submissions(recalls: pd.DataFrame) -> pd.DataFrame:
    """One row per (recall, premarket submission number) for joining to 510(k)/PMA."""
    k = recalls[["res_event_number", "recall_class", "recall_date", "product_code", "k_numbers"]].explode("k_numbers")
    k = k.rename(columns={"k_numbers": "submission_number"}).dropna(subset=["submission_number"])
    p = recalls[["res_event_number", "recall_class", "recall_date", "product_code", "pma_numbers"]].explode("pma_numbers")
    p = p.rename(columns={"pma_numbers": "submission_number"}).dropna(subset=["submission_number"])
    out = pd.concat([k, p], ignore_index=True)
    out["submission_number"] = out["submission_number"].astype(str).str.upper().str.strip()
    return out
