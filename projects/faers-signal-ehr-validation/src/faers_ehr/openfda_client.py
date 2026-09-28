"""openFDA drug adverse-event (FAERS) client.

* Reads ``OPENFDA_API_KEY`` from the environment (never hard-code it). Without a key the API
  allows 40 requests/min and 1,000/day; with a key 240/min and 120,000/day.
* Throttles to the per-minute limit, retries on 429 / 5xx with exponential back-off and honours
  ``Retry-After``.
* Pagination: ``skip`` up to 25,000 records, then follows the ``Link: <...>; rel="next"`` header
  (``search_after``) that openFDA returns for deeper pages.
* 2x2 tables come from ``meta.results.total`` of four searches, i.e. four cheap requests per
  drug x outcome pair; no records are downloaded.

MedDRA preferred terms use British spelling in FAERS (HYPERKALAEMIA, HYPONATRAEMIA).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import pandas as pd
import requests

BASE_URL = "https://api.fda.gov/drug/event.json"

# Outcome -> MedDRA PTs (pre-registered lists; broad SMQ-like variants can be added per outcome)
OUTCOME_PT: dict[str, tuple[str, ...]] = {
    "hyperkalaemia": ("HYPERKALAEMIA", "BLOOD POTASSIUM INCREASED"),
    "hyponatraemia": ("HYPONATRAEMIA", "BLOOD SODIUM DECREASED"),
    "aki": ("ACUTE KIDNEY INJURY", "RENAL FAILURE ACUTE", "RENAL IMPAIRMENT", "BLOOD CREATININE INCREASED"),
    "hepatotoxicity": ("HEPATOTOXICITY", "DRUG-INDUCED LIVER INJURY", "HEPATOCELLULAR INJURY", "LIVER INJURY",
                       "HEPATIC FAILURE", "ALANINE AMINOTRANSFERASE INCREASED", "HEPATIC ENZYME INCREASED",
                       "TRANSAMINASES INCREASED"),
    "thrombocytopenia": ("THROMBOCYTOPENIA", "PLATELET COUNT DECREASED"),
    "neutropenia": ("NEUTROPENIA", "NEUTROPHIL COUNT DECREASED", "FEBRILE NEUTROPENIA", "AGRANULOCYTOSIS"),
    "qt_prolongation": ("ELECTROCARDIOGRAM QT PROLONGED", "LONG QT SYNDROME", "TORSADE DE POINTES"),
}

DRUG_FIELD = "patient.drug.openfda.generic_name"
RXCUI_FIELD = "patient.drug.openfda.rxcui"
EVENT_FIELD = "patient.reaction.reactionmeddrapt"


def _quote(term: str) -> str:
    """Quote a Lucene term for openFDA (spaces inside quotes are literal)."""
    return '"' + term.replace('"', "") + '"'


def drug_query(ingredient: str, suspect_only: bool = True, field_name: str = DRUG_FIELD) -> str:
    """Search fragment for reports listing ``ingredient``; ``suspect_only`` restricts the report
    to those where *some* drug is characterised as suspect (openFDA cannot bind the two
    conditions to the same drug entry, so this is an approximation - documented limitation)."""
    q = f"{field_name}:{_quote(ingredient.upper())}"
    if suspect_only:
        q = f"({q}+AND+patient.drug.drugcharacterization:1)"
    return q


def event_query(pts: Sequence[str]) -> str:
    return "(" + "+OR+".join(f"{EVENT_FIELD}:{_quote(p.upper())}" for p in pts) + ")"


@dataclass
class OpenFDAClient:
    api_key: str | None = field(default_factory=lambda: os.environ.get("OPENFDA_API_KEY"))
    requests_per_minute: int | None = None
    timeout: float = 60.0
    max_retries: int = 6
    session: requests.Session = field(default_factory=requests.Session)
    _last: list[float] = field(default_factory=list, repr=False)
    n_requests: int = 0

    def __post_init__(self) -> None:
        if self.requests_per_minute is None:
            self.requests_per_minute = 240 if self.api_key else 40

    # -- low level ---------------------------------------------------------------------------
    def _throttle(self) -> None:
        now = time.time()
        self._last = [t for t in self._last if now - t < 60.0]
        if len(self._last) >= self.requests_per_minute:
            time.sleep(max(0.0, 60.0 - (now - self._last[0]) + 0.05))
        self._last.append(time.time())

    def get(self, params: dict | None = None, url: str | None = None) -> dict:
        """One GET with throttling and retries. openFDA uses '+' as the AND/space separator, so
        the query string is assembled by hand (``requests`` would percent-encode the '+')."""
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key
        if url is None:
            url = BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        elif self.api_key and "api_key=" not in url:
            url += ("&" if "?" in url else "?") + f"api_key={self.api_key}"
        delay = 1.0
        for attempt in range(self.max_retries):
            self._throttle()
            self.n_requests += 1
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code == 404:  # openFDA returns 404 for "no matches"
                return {"meta": {"results": {"total": 0}}, "results": [], "_link_next": None}
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", delay))
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
            r.raise_for_status()
            out = r.json()
            out["_link_next"] = _next_link(r.headers.get("Link"))
            return out
        raise RuntimeError(f"openFDA: giving up after {self.max_retries} retries: {url[:120]}")

    # -- counts ------------------------------------------------------------------------------
    def total(self, search: str) -> int:
        """Number of reports matching ``search`` (from meta.results.total)."""
        out = self.get({"search": search, "limit": 1})
        return int(out.get("meta", {}).get("results", {}).get("total", 0))

    def count(self, search: str, count_field: str, exact: bool = True, limit: int = 1000) -> pd.DataFrame:
        """Aggregate counts of ``count_field`` values among reports matching ``search``."""
        cf = count_field + (".exact" if exact and not count_field.endswith(".exact") else "")
        out = self.get({"search": search, "count": cf, "limit": limit})
        return pd.DataFrame(out.get("results", []), columns=["term", "count"])

    def two_by_two(self, drug_search: str, event_search: str, base_search: str = "") -> dict[str, int]:
        """a: drug & event, b: drug & not event, c: event & not drug, d: neither, n: all."""
        def combine(*parts: str) -> str:
            return "+AND+".join(p for p in parts if p)

        n = self.total(base_search) if base_search else self.total("_exists_:safetyreportid")
        drug_total = self.total(combine(base_search, drug_search))
        event_total = self.total(combine(base_search, event_search))
        a = self.total(combine(base_search, drug_search, event_search))
        b, c = drug_total - a, event_total - a
        return {"a": a, "b": b, "c": c, "d": n - a - b - c, "n": n}

    def drug_event_table(self, ingredients: Iterable[str], outcomes: Iterable[str] | None = None,
                         suspect_only: bool = True, base_search: str = "", pt_map: dict[str, Sequence[str]] | None = None,
                         year_range: tuple[int, int] | None = None) -> pd.DataFrame:
        """2x2 tables for every ingredient x outcome (one row each).

        ``year_range`` restricts by receivedate (e.g. (2004, 2026)). ``pt_map`` overrides
        :data:`OUTCOME_PT`.
        """
        pt_map = pt_map or OUTCOME_PT
        outcomes = list(outcomes) if outcomes is not None else list(pt_map)
        if year_range:
            yr = f"receivedate:[{year_range[0]}0101+TO+{year_range[1]}1231]"
            base_search = f"{base_search}+AND+{yr}" if base_search else yr
        rows = []
        for ing in ingredients:
            dq = drug_query(ing, suspect_only)
            for oc in outcomes:
                eq = event_query(pt_map[oc])
                t = self.two_by_two(dq, eq, base_search)
                rows.append({"drug": ing.lower(), "outcome": oc, **t, "drug_query": dq, "event_query": eq})
        return pd.DataFrame(rows)

    def stratified_counts(self, ingredient: str, outcome: str, strata_field: str,
                          suspect_only: bool = True, pt_map: dict[str, Sequence[str]] | None = None) -> pd.DataFrame:
        """Report counts by a stratification field (e.g. ``primarysource.qualification``,
        ``patient.drug.drugcharacterization``) for drug-and-event vs drug-only reports, used for
        RQ3 (which reporting features predict false-positive signals)."""
        pt_map = pt_map or OUTCOME_PT
        dq = drug_query(ingredient, suspect_only)
        eq = event_query(pt_map[outcome])
        both = self.count(f"{dq}+AND+{eq}", strata_field).rename(columns={"count": "drug_and_event"})
        drug = self.count(dq, strata_field).rename(columns={"count": "drug_total"})
        out = drug.merge(both, on="term", how="left").fillna({"drug_and_event": 0})
        out.insert(0, "outcome", outcome)
        out.insert(0, "drug", ingredient.lower())
        return out

    def yearly_counts(self, ingredient: str, outcome: str | None = None, suspect_only: bool = True) -> pd.DataFrame:
        """Reports per receive-year (Weber-effect / stimulated-reporting diagnostics)."""
        search = drug_query(ingredient, suspect_only)
        if outcome:
            search += "+AND+" + event_query(OUTCOME_PT[outcome])
        df = self.count(search, "receivedate", exact=False)
        if df.empty:
            return df
        df["year"] = df["term"].astype(str).str[:4].astype(int)
        return df.groupby("year", as_index=False)["count"].sum()

    # -- records -----------------------------------------------------------------------------
    def iter_reports(self, search: str, limit: int = 100, max_records: int | None = None,
                     sort: str | None = None) -> Iterator[dict]:
        """Stream raw reports (skip pagination, then Link/search_after beyond 25,000)."""
        params = {"search": search, "limit": min(limit, 1000)}
        if sort:
            params["sort"] = sort
        n, skip, url = 0, 0, None
        while True:
            out = self.get(params if url is None else None, url=url)
            for rec in out.get("results", []):
                yield rec
                n += 1
                if max_records and n >= max_records:
                    return
            if not out.get("results"):
                return
            if out.get("_link_next"):
                url = out["_link_next"]
                continue
            skip += params["limit"]
            if skip > 25000:
                return
            params["skip"] = skip
            url = None


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
    return m.group(1) if m else None


def reports_to_frame(reports: Iterable[dict]) -> pd.DataFrame:
    """Flatten raw reports to one row per (report, drug, reaction) for local analysis."""
    rows = []
    for r in reports:
        sid = r.get("safetyreportid")
        ver = r.get("safetyreportversion")
        rec = r.get("receivedate")
        qual = (r.get("primarysource") or {}).get("qualification")
        pat = r.get("patient", {})
        for d in pat.get("drug", []):
            of = d.get("openfda", {})
            for rx in pat.get("reaction", []):
                rows.append({"safetyreportid": sid, "version": ver, "receivedate": rec, "qualification": qual,
                             "generic_name": ";".join(of.get("generic_name", [])) or None,
                             "rxcui": ";".join(of.get("rxcui", [])) or None,
                             "medicinalproduct": d.get("medicinalproduct"),
                             "characterization": d.get("drugcharacterization"),
                             "indication": d.get("drugindication"),
                             "reaction": rx.get("reactionmeddrapt"), "outcome_code": rx.get("reactionoutcome")})
    return pd.DataFrame(rows)


def deduplicate_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest version per safetyreportid (FAERS follow-ups share the case id)."""
    if df.empty:
        return df
    d = df.copy()
    d["version"] = pd.to_numeric(d["version"], errors="coerce").fillna(0)
    latest = d.groupby("safetyreportid")["version"].transform("max")
    return d[d["version"] == latest].drop_duplicates()
