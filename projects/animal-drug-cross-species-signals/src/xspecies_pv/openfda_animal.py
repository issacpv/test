"""openFDA client for ``animalandveterinary/event`` (and count queries on ``drug/event``).

The animal endpoint has no reliable cursor beyond ``skip`` (capped at 25,000), so records are
harvested in date windows of ``original_receive_date``. Flatteners turn nested reports into one
row per (report, drug, reaction) so pandas can do the rest. The HTTP session is injectable so
that the flatteners and paging logic are testable offline.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

LOG = logging.getLogger(__name__)
BASE_URL = "https://api.fda.gov"
ANIMAL_ENDPOINT = "animalandveterinary/event"
HUMAN_ENDPOINT = "drug/event"
MAX_SKIP = 25_000
DATE_FMT = "%Y%m%d"


def _q(term: str) -> str:
    term = term.replace('"', "")
    return f'"{term}"' if any(ch in term for ch in " -/,()+'") else term


class OpenFDAAnimalClient:
    """Retrying, paced client. ``api_key`` defaults to ``OPENFDA_API_KEY``."""

    def __init__(self, api_key: Optional[str] = None, session: Any = None, timeout: float = 60.0,
                 min_interval: float = 0.26, max_retries: int = 5) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENFDA_API_KEY")
        if session is None:
            if requests is None:  # pragma: no cover
                raise ImportError("requests is required for live openFDA calls")
            session = requests.Session()
            session.headers.update({"User-Agent": "xspecies_pv/0.1 (research)"})
        self.session = session
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{BASE_URL}/{endpoint}.json?" + "&".join(f"{k}={v}" for k, v in params.items())
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    return {"meta": {"results": {"total": 0}}, "results": []}
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

    # ------------------------------------------------------------------ paging
    def iter_window(self, endpoint: str, search: str, limit: int = 1000) -> Iterator[Dict[str, Any]]:
        skip = 0
        while True:
            data = self.get(endpoint, {"search": search, "limit": limit, "skip": skip})
            rows = data.get("results", [])
            if not rows:
                return
            yield from rows
            skip += len(rows)
            total = int(data.get("meta", {}).get("results", {}).get("total", 0))
            if skip >= total:
                return
            if skip >= MAX_SKIP:
                LOG.warning("skip ceiling hit for %s (total=%d); use a narrower window", search, total)
                return

    def iter_animal_events(self, start: date, end: date, species: Optional[List[str]] = None,
                           step_days: int = 30, max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Animal reports received in ``[start, end]``, in ``step_days`` windows, optionally by species."""
        n = 0
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=step_days - 1), end)
            clauses = [f"original_receive_date:[{cur.strftime(DATE_FMT)}+TO+{nxt.strftime(DATE_FMT)}]"]
            if species:
                clauses.append("animal.species:(" + "+".join(_q(s) for s in species) + ")")
            for rec in self.iter_window(ANIMAL_ENDPOINT, "+AND+".join(clauses)):
                yield rec
                n += 1
                if max_records is not None and n >= max_records:
                    return
            cur = nxt + timedelta(days=1)

    # ------------------------------------------------------------------ counts
    def count(self, endpoint: str, search: str, field: str, limit: int = 1000) -> pd.DataFrame:
        rows = self.get(endpoint, {"search": search, "count": field, "limit": limit}).get("results", [])
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["term", "count"])

    def daily_counts(self, endpoint: str, search: str, date_field: str, start: date, end: date) -> pd.DataFrame:
        frames = []
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
            return pd.DataFrame({"date": pd.to_datetime([]), "count": pd.Series([], dtype=int)})
        out = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
        out["date"] = pd.to_datetime(out["date"], format=DATE_FMT)
        return out.groupby("date", as_index=False)["count"].sum()


def faers_search(ingredient: Optional[str] = None, pt: Optional[str] = None, suspect_only: bool = True) -> str:
    parts: List[str] = []
    if ingredient:
        parts.append(f"patient.drug.openfda.generic_name:{_q(ingredient.upper())}")
        if suspect_only:
            parts.append("patient.drug.drugcharacterization:1")
    if pt:
        parts.append(f"patient.reaction.reactionmeddrapt:{_q(pt)}")
    return "+AND+".join(parts)


# --------------------------------------------------------------------------- flatteners
def _first(x: Any) -> Any:
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _dose_fields(dose: Any) -> Dict[str, Any]:
    """openFDA serialises the dose as ``{"numerator": "5", "numerator_unit": "mg", "denominator": ..}``
    or, in some records, as a free-text string. Return (value, unit) as best effort."""
    if dose is None:
        return {"dose_value": None, "dose_unit": None, "dose_denominator": None, "dose_denominator_unit": None}
    if isinstance(dose, dict):
        num = dose.get("numerator")
        try:
            num = float(num) if num not in (None, "") else None
        except (TypeError, ValueError):
            num = None
        den = dose.get("denominator")
        try:
            den = float(den) if den not in (None, "") else None
        except (TypeError, ValueError):
            den = None
        return {"dose_value": num, "dose_unit": dose.get("numerator_unit"), "dose_denominator": den,
                "dose_denominator_unit": dose.get("denominator_unit")}
    return {"dose_value": None, "dose_unit": str(dose), "dose_denominator": None, "dose_denominator_unit": None}


def flatten_animal_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per (drug, reaction) of a CVM report; reports without reactions yield one row with NaN reaction."""
    animal = rec.get("animal", {}) or {}
    weight = animal.get("weight", {}) or {}
    age = animal.get("age", {}) or {}
    base = {
        "report_id": rec.get("report_id") or rec.get("unique_aer_id_number"),
        "unique_aer_id_number": rec.get("unique_aer_id_number"),
        "receive_date": rec.get("original_receive_date"),
        "onset_date": rec.get("onset_date"),
        "species": animal.get("species"),
        "breed": (animal.get("breed", {}) or {}).get("breed_component") if isinstance(animal.get("breed"), dict) else animal.get("breed"),
        "gender": animal.get("gender"),
        "age_value": age.get("min"),
        "age_unit": age.get("unit"),
        "weight_min": weight.get("min"),
        "weight_max": weight.get("max"),
        "weight_unit": weight.get("unit"),
        "serious_ae": rec.get("serious_ae"),
        "primary_reporter": rec.get("primary_reporter"),
        "health_prior": rec.get("health_assessment_prior_to_exposure", {}).get("condition") if isinstance(rec.get("health_assessment_prior_to_exposure"), dict) else None,
        "outcome": ";".join(str(o.get("medical_status")) for o in (rec.get("outcome") or []) if isinstance(o, dict)),
        "n_animals_treated": rec.get("number_of_animals_treated"),
    }
    drugs = rec.get("drug") or [{}]
    reactions = rec.get("reaction") or [{}]
    rows: List[Dict[str, Any]] = []
    for di, d in enumerate(drugs):
        actives = d.get("active_ingredients") or [{}]
        for a in actives:
            drow = dict(base)
            drow.update({
                "drug_index": di,
                "brand_name": d.get("brand_name"),
                "active_ingredient": a.get("name"),
                "route": d.get("route"),
                "dosage_form": d.get("dosage_form"),
                "used_according_to_label": d.get("used_according_to_label"),
                "manufacturer": (d.get("manufacturer") or {}).get("name") if isinstance(d.get("manufacturer"), dict) else d.get("manufacturer"),
            })
            drow.update(_dose_fields(a.get("dose")))
            for r in reactions:
                row = dict(drow)
                row.update({
                    "veddra_term": r.get("veddra_term_name"),
                    "veddra_code": r.get("veddra_term_code"),
                    "veddra_version": r.get("veddra_version"),
                    "n_affected": r.get("number_of_animals_affected"),
                    "accuracy": r.get("accuracy"),
                })
                rows.append(row)
    return rows


def flatten_faers_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per (suspect drug, reaction) of a FAERS report (for record-level pulls)."""
    patient = rec.get("patient", {}) or {}
    base = {
        "safetyreportid": rec.get("safetyreportid"),
        "receive_date": rec.get("receivedate"),
        "serious": rec.get("serious"),
        "reporter_qualification": (rec.get("primarysource") or {}).get("qualification"),
        "sex": patient.get("patientsex"),
        "age": patient.get("patientonsetage"),
        "age_unit": patient.get("patientonsetageunit"),
        "weight_kg": patient.get("patientweight"),
    }
    rows: List[Dict[str, Any]] = []
    for d in patient.get("drug") or []:
        if str(d.get("drugcharacterization")) not in ("1", "2"):
            continue
        generic = _first((d.get("openfda") or {}).get("generic_name")) or d.get("medicinalproduct")
        for r in patient.get("reaction") or [{}]:
            row = dict(base)
            row.update({"ingredient": generic, "characterization": d.get("drugcharacterization"),
                        "pt": r.get("reactionmeddrapt"), "outcome": r.get("reactionoutcome")})
            rows.append(row)
    return rows


def records_to_frame(records: List[Dict[str, Any]], flattener=flatten_animal_record) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.extend(flattener(rec))
    return pd.DataFrame(rows)
