"""Rate-limited openFDA client for FAERS (``drug/event``) count and record queries.

openFDA facts this module relies on (see https://open.fda.gov/apis/):

* Base URL ``https://api.fda.gov/<endpoint>.json`` with query params ``search``,
  ``count``, ``limit``, ``skip``, ``sort`` and ``api_key``.
* ``count=<field>`` returns ``{"results": [{"term": ..., "count": ...}]}`` for
  string fields and ``[{"time": "YYYYMMDD", "count": ...}]`` for date fields.
  At most 1000 count buckets are returned per call.
* Record queries return at most ``limit=1000`` records per page.  ``skip`` is
  capped at 25 000, so deep pagination must use the ``Link: <url>; rel="next"``
  response header, which carries a ``search_after`` cursor.  This client follows
  that header automatically and falls back to ``skip`` when the header is absent.
* Rate limits: 40 requests/min and 1000/day without a key; 240 requests/min and
  120 000/day with a free key (``OPENFDA_API_KEY``).

Nothing here is specific to bias analysis; the same class is reused for the
device and label endpoints in sibling projects.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENFDA_BASE_URL = "https://api.fda.gov"

#: ``primarysource.qualification`` codes in FAERS / ICH E2B.
QUALIFICATION_LABELS: Dict[str, str] = {
    "1": "physician",
    "2": "pharmacist",
    "3": "other_health_professional",
    "4": "lawyer",
    "5": "consumer",
}

#: ``patient.patientsex`` codes.
SEX_LABELS: Dict[str, str] = {"0": "unknown", "1": "male", "2": "female"}

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


class OpenFDAError(RuntimeError):
    """Raised when openFDA returns a non-retryable error."""


@dataclass
class OpenFDAClient:
    """Small synchronous client with a sliding-window rate limiter and retries.

    Parameters
    ----------
    api_key:
        openFDA API key. Defaults to the ``OPENFDA_API_KEY`` environment variable.
    max_per_minute:
        Requests allowed per rolling 60 s window. 240 with a key, 40 without.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Retries on HTTP 429 / 5xx and connection errors (exponential backoff).
    session:
        Optional :class:`requests.Session` (injected in tests).
    """

    api_key: Optional[str] = None
    max_per_minute: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 5
    base_url: str = OPENFDA_BASE_URL
    session: Optional[requests.Session] = None
    _timestamps: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENFDA_API_KEY") or None
        if self.max_per_minute is None:
            self.max_per_minute = 240 if self.api_key else 40
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": "faers_bias/0.1 (research)"})

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

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        params = dict(params or {})
        if self.api_key and "api_key" not in params and "api_key=" not in url:
            params["api_key"] = self.api_key
        backoff = 1.0
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # network hiccup
                last_exc = exc
                logger.warning("request error (%s); retry %d", exc, attempt + 1)
            else:
                if resp.status_code == 404:
                    # openFDA returns 404 with {"error": {"code": "NOT_FOUND"}} when
                    # a search matches nothing. Treat as an empty result.
                    return resp
                if resp.status_code < 400:
                    return resp
                if resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning("HTTP %s from openFDA; retry %d", resp.status_code, attempt + 1)
                else:
                    raise OpenFDAError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        raise OpenFDAError(f"giving up after {self.max_retries} retries: {last_exc}")

    def _endpoint_url(self, endpoint: str) -> str:
        endpoint = endpoint.strip("/")
        return f"{self.base_url}/{endpoint}.json"

    # ----------------------------------------------------------------- counts
    def count(
        self,
        endpoint: str,
        search: Optional[str],
        count_field: str,
        limit: int = 1000,
        exact: bool = True,
    ) -> pd.DataFrame:
        """Run a ``count`` query and return a tidy DataFrame.

        For string fields the result has columns ``term, count``; for date
        fields (``receivedate``, ``date_received`` ...) it has ``time, count``
        with ``time`` parsed to ``datetime64``.

        ``exact=True`` appends ``.exact`` so multi-word terms are not tokenised
        (openFDA convention). Date fields never take ``.exact``.
        """
        field_name = count_field
        is_date = field_name.split(".")[-1].lower().endswith("date") or field_name.endswith("time")
        if exact and not is_date and not field_name.endswith(".exact"):
            field_name = field_name + ".exact"
        params: Dict[str, Any] = {"count": field_name, "limit": int(limit)}
        if search:
            params["search"] = search
        resp = self._get(self._endpoint_url(endpoint), params)
        if resp.status_code == 404:
            return pd.DataFrame(columns=["time" if is_date else "term", "count"])
        results = resp.json().get("results", [])
        df = pd.DataFrame(results)
        if df.empty:
            return pd.DataFrame(columns=["time" if is_date else "term", "count"])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d", errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
        return df

    # ---------------------------------------------------------------- records
    def iter_records(
        self,
        endpoint: str,
        search: Optional[str] = None,
        limit: int = 100,
        max_records: Optional[int] = None,
        sort: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate over records, following the ``Link`` header for deep pages.

        openFDA emits ``Link: <...&search_after=...>; rel="next"`` on record
        queries; following it is the only way past the 25 000 ``skip`` cap.
        """
        url = self._endpoint_url(endpoint)
        params: Dict[str, Any] = {"limit": int(min(limit, 1000))}
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        n = 0
        skip = 0
        next_url: Optional[str] = None
        followed_link = False  # once we are in search_after mode, never fall back to skip
        while True:
            if next_url is not None:
                resp = self._get(next_url, params=None)
            else:
                resp = self._get(url, {**params, "skip": skip})
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
                followed_link = True
                continue
            if followed_link:
                # last search_after page has no rel="next" -> exhausted
                return
            # skip-based fallback (only valid up to 25 000)
            skip += len(results)
            total = payload.get("meta", {}).get("results", {}).get("total", None)
            if skip >= 25000 or (total is not None and skip >= int(total)) or len(results) < params["limit"]:
                return
            next_url = None

    def fetch_records(self, endpoint: str, search: Optional[str] = None, **kw: Any) -> List[Dict[str, Any]]:
        """Materialise :meth:`iter_records` into a list."""
        return list(self.iter_records(endpoint, search=search, **kw))

    # --------------------------------------------------------- FAERS helpers
    @staticmethod
    def quote(term: str) -> str:
        """Quote a multi-word value for an openFDA ``search`` clause."""
        term = term.replace('"', "")
        return f'"{term}"'

    @staticmethod
    def date_range(start: str, end: str, field_name: str = "receivedate") -> str:
        """Build ``field:[YYYYMMDD TO YYYYMMDD]`` from ISO dates or YYYYMMDD strings."""
        s = start.replace("-", "")
        e = end.replace("-", "")
        return f"{field_name}:[{s} TO {e}]"

    def faers_drug_search(
        self,
        generic_name: str,
        roles: Sequence[str] = ("1", "2"),
        extra: Optional[str] = None,
    ) -> str:
        """Search clause restricting to reports listing ``generic_name``.

        ``patient.drug.drugcharacterization`` 1 = suspect, 2 = concomitant,
        3 = interacting. Default keeps suspect + concomitant, matching common
        practice; use ``roles=("1",)`` for suspect-only analyses.

        Note: openFDA cannot express "role of *this* drug" strictly (the two
        clauses are matched independently within a report); for strict
        suspect-only cohorts filter records client-side after ``iter_records``.
        """
        clause = f"patient.drug.openfda.generic_name:{self.quote(generic_name.upper())}"
        if roles:
            role_clause = "+".join(f"patient.drug.drugcharacterization:{r}" for r in roles)
            clause = f"{clause} AND ({role_clause})" if len(roles) == 1 else f"{clause} AND ({role_clause.replace('+', ' OR ')})"
        if extra:
            clause = f"{clause} AND {extra}"
        return clause

    def faers_sex_counts(self, generic_name: str, search_extra: Optional[str] = None) -> pd.DataFrame:
        """Report counts by ``patient.patientsex`` for a drug (labels mapped)."""
        df = self.count("drug/event", self.faers_drug_search(generic_name, extra=search_extra), "patient.patientsex", exact=False)
        if not df.empty:
            df["sex"] = df["term"].astype(str).map(SEX_LABELS).fillna("unknown")
        return df

    def faers_reporter_counts(self, generic_name: str, search_extra: Optional[str] = None) -> pd.DataFrame:
        """Report counts by ``primarysource.qualification`` for a drug."""
        df = self.count(
            "drug/event", self.faers_drug_search(generic_name, extra=search_extra), "primarysource.qualification", exact=False
        )
        if not df.empty:
            df["reporter"] = df["term"].astype(str).map(QUALIFICATION_LABELS).fillna("unknown")
        return df

    def faers_monthly_series(
        self,
        search: str,
        start: str = "2004-01-01",
        end: str = "2025-12-31",
        date_field: str = "receivedate",
    ) -> pd.DataFrame:
        """Monthly report counts for an arbitrary search clause.

        openFDA caps count output at 1000 buckets, so daily buckets over > 2.7
        years are truncated. We therefore issue one call per calendar year and
        aggregate to month-start ``period`` timestamps.
        """
        frames = []
        years = range(int(start[:4]), int(end[:4]) + 1)
        for y in years:
            ys = max(f"{y}-01-01", start)
            ye = min(f"{y}-12-31", end)
            clause = f"{search} AND {self.date_range(ys, ye, date_field)}" if search else self.date_range(ys, ye, date_field)
            df = self.count("drug/event", clause, date_field, exact=False)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["month", "count"])
        daily = pd.concat(frames, ignore_index=True)
        daily["month"] = daily["time"].dt.to_period("M").dt.to_timestamp()
        monthly = daily.groupby("month", as_index=False)["count"].sum()
        full_index = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
        monthly = monthly.set_index("month").reindex(full_index, fill_value=0).rename_axis("month").reset_index()
        return monthly

    def faers_reaction_counts(self, generic_name: str, limit: int = 1000, search_extra: Optional[str] = None) -> pd.DataFrame:
        """Top MedDRA PTs (``patient.reaction.reactionmeddrapt``) for a drug."""
        return self.count(
            "drug/event",
            self.faers_drug_search(generic_name, extra=search_extra),
            "patient.reaction.reactionmeddrapt",
            limit=limit,
        )

    def faers_two_by_two(
        self,
        generic_name: str,
        reaction_pt: str,
        search_extra: Optional[str] = None,
    ) -> Dict[str, int]:
        """Return the 2x2 counts (a, b, c, d) for a drug-event pair via 4 count calls.

        ``search_extra`` (e.g. ``patient.patientsex:2``) restricts all four cells,
        which is how sex-stratified tables are built.
        """
        drug = self.faers_drug_search(generic_name)
        rxn = f"patient.reaction.reactionmeddrapt:{self.quote(reaction_pt)}"
        base = f"({search_extra})" if search_extra else None

        def total(clause: Optional[str]) -> int:
            parts = [p for p in (clause, base) if p]
            s = " AND ".join(parts) if parts else None
            df = self.count("drug/event", s, "receivedate", exact=False)
            return int(df["count"].sum()) if not df.empty else 0

        n_drug = total(drug)
        a = total(f"{drug} AND {rxn}")
        n_rxn = total(rxn)
        n_all = total(None)
        b = n_drug - a
        c = n_rxn - a
        d = n_all - a - b - c
        return {"a": a, "b": b, "c": c, "d": d}


def flatten_faers_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one FAERS JSON report into a row (drugs and reactions as lists)."""
    patient = rec.get("patient", {}) or {}
    src = rec.get("primarysource", {}) or {}
    drugs = patient.get("drug", []) or []
    reactions = patient.get("reaction", []) or []
    generic_names: List[str] = []
    suspect_names: List[str] = []
    for d in drugs:
        names = (d.get("openfda", {}) or {}).get("generic_name") or []
        if not names and d.get("medicinalproduct"):
            names = [str(d["medicinalproduct"]).upper()]
        generic_names.extend(names)
        if str(d.get("drugcharacterization")) == "1":
            suspect_names.extend(names)
    return {
        "safetyreportid": rec.get("safetyreportid"),
        "receivedate": rec.get("receivedate"),
        "receiptdate": rec.get("receiptdate"),
        "serious": rec.get("serious"),
        "occurcountry": rec.get("occurcountry"),
        "qualification": str(src.get("qualification")) if src.get("qualification") is not None else None,
        "reporter": QUALIFICATION_LABELS.get(str(src.get("qualification")), "unknown"),
        "patientsex": str(patient.get("patientsex")) if patient.get("patientsex") is not None else None,
        "sex": SEX_LABELS.get(str(patient.get("patientsex")), "unknown"),
        "patientonsetage": patient.get("patientonsetage"),
        "patientonsetageunit": patient.get("patientonsetageunit"),
        "generic_names": sorted(set(generic_names)),
        "suspect_generic_names": sorted(set(suspect_names)),
        "reactions": sorted({r.get("reactionmeddrapt") for r in reactions if r.get("reactionmeddrapt")}),
    }


def records_to_frame(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Vectorise :func:`flatten_faers_record` over an iterable of reports."""
    return pd.DataFrame([flatten_faers_record(r) for r in records])
