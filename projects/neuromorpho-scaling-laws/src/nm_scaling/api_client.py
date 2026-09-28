"""Client for the NeuroMorpho.org REST API v1.

Endpoints (see https://neuromorpho.org/apiReference.html):

* ``GET  /api/neuron?page=P&size=S``                     all neurons, paginated
* ``POST /api/neuron/select?page=P&size=S`` (JSON body)  filtered; body maps field -> list of values
* ``GET  /api/neuron/select?q=field:value&fq=...``      filtered (GET form)
* ``GET  /api/neuron/fields/{field}``                    distinct values of a field
* ``GET  /api/morphometry/name/{neuron_name}``           L-Measure summary morphometrics
* ``https://neuromorpho.org/dableFiles/{archive}/CNG version/{name}.CNG.swc``  standardised SWC

The client retries on 429/5xx with exponential backoff and honours the environment variable
``NEUROMORPHO_VERIFY_SSL`` (set to ``0`` only if the server's TLS chain is temporarily broken).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests

LOG = logging.getLogger(__name__)

BASE_URL = "https://neuromorpho.org/api"
DABLE_URL = "https://neuromorpho.org/dableFiles"

#: Metadata fields that encode provenance / protocol ("batch") rather than biology.
BATCH_FIELDS: Tuple[str, ...] = (
    "archive",
    "reconstruction_software",
    "shrinkage_reported",
    "shrinkage_corrected",
    "protocol",
    "slicing_thickness",
    "slicing_direction",
    "objective_type",
    "magnification",
    "stain",
    "original_format",
    "physical_Integrity",
    "structural_domains",
    "experiment_condition",
    "deposition_date",
)

#: Metadata fields that encode biology.
BIO_FIELDS: Tuple[str, ...] = (
    "species",
    "scientific_name",
    "strain",
    "brain_region",
    "cell_type",
    "age_classification",
    "min_age",
    "max_age",
    "gender",
    "domain",
)

_LIST_FIELDS = ("brain_region", "cell_type", "structural_domains", "attributes")


def _verify_default() -> bool:
    return os.environ.get("NEUROMORPHO_VERIFY_SSL", "1").strip() not in {"0", "false", "False", "no"}


class NeuroMorphoClient:
    """Thin, resumable client for the NeuroMorpho.org REST API v1.

    Parameters
    ----------
    base_url : API root (default ``https://neuromorpho.org/api``).
    timeout : per-request timeout in seconds.
    max_retries : retries on HTTP 429 / 5xx / connection errors.
    backoff : base of the exponential backoff (seconds).
    verify_ssl : TLS verification; ``None`` reads ``NEUROMORPHO_VERIFY_SSL``.
    """

    def __init__(self, base_url: str = BASE_URL, timeout: float = 60.0, max_retries: int = 5,
                 backoff: float = 2.0, verify_ssl: Optional[bool] = None,
                 session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.verify_ssl = _verify_default() if verify_ssl is None else verify_ssl
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "nm_scaling/0.1"})

    # ------------------------------------------------------------------ low level
    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                 json_body: Optional[Dict[str, Any]] = None, stream: bool = False,
                 absolute: bool = False) -> requests.Response:
        url = path if absolute else f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(method, url, params=params, json=json_body, timeout=self.timeout,
                                            verify=self.verify_ssl, stream=stream)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code} for {url}", response=resp)
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                wait = self.backoff ** attempt
                LOG.warning("request %s %s failed (%s); retry %d/%d in %.1fs", method, url, exc, attempt + 1,
                            self.max_retries, wait)
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params).json()

    def _post_json(self, path: str, body: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("POST", path, params=params, json_body=body).json()

    # ------------------------------------------------------------------ metadata
    def fields(self) -> List[str]:
        """Names of queryable neuron fields (``GET /neuron/fields``)."""
        data = self._get_json("neuron/fields")
        return list(data.get("Neuron Fields", data.get("fields", [])))

    def field_values(self, field: str) -> List[str]:
        """Distinct values of a neuron field, e.g. ``field_values('species')``."""
        data = self._get_json(f"neuron/fields/{field}")
        return list(data.get("fields", []))

    @staticmethod
    def _unwrap_page(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        embedded = data.get("_embedded", {})
        # the resource key differs between endpoints (neuronResources / morphometryResources ...)
        records: List[Dict[str, Any]] = []
        for value in embedded.values():
            if isinstance(value, list):
                records.extend(value)
        page = data.get("page", {})
        return records, {k: int(page.get(k, 0)) for k in ("size", "totalElements", "totalPages", "number")}

    def iter_pages(self, filters: Optional[Dict[str, Sequence[str]]] = None, page_size: int = 500,
                   start_page: int = 0, max_pages: Optional[int] = None,
                   endpoint: str = "neuron") -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """Yield ``(page_number, records)`` for ``/neuron`` (no filters) or ``/neuron/select``.

        ``filters`` maps a field name to a list of accepted values, e.g.
        ``{"species": ["mouse", "rat"], "brain_region": ["neocortex"]}``.
        """
        page_size = max(1, min(int(page_size), 500))
        page = start_page
        n_pages_done = 0
        while True:
            params = {"page": page, "size": page_size}
            if filters:
                data = self._post_json(f"{endpoint}/select", {k: list(v) for k, v in filters.items()}, params=params)
            else:
                data = self._get_json(endpoint, params=params)
            records, info = self._unwrap_page(data)
            LOG.debug("%s page %d/%d: %d records", endpoint, page, info.get("totalPages", -1), len(records))
            if not records:
                return
            yield page, records
            n_pages_done += 1
            page += 1
            if max_pages is not None and n_pages_done >= max_pages:
                return
            if info.get("totalPages") and page >= info["totalPages"]:
                return

    def iter_neurons(self, filters: Optional[Dict[str, Sequence[str]]] = None, page_size: int = 500,
                     max_pages: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield neuron metadata records one at a time."""
        for _, records in self.iter_pages(filters, page_size=page_size, max_pages=max_pages):
            yield from records

    def count(self, filters: Optional[Dict[str, Sequence[str]]] = None) -> int:
        """Total number of records matching ``filters`` (one request of size 1)."""
        params = {"page": 0, "size": 1}
        data = (self._post_json("neuron/select", {k: list(v) for k, v in filters.items()}, params=params)
                if filters else self._get_json("neuron", params=params))
        return self._unwrap_page(data)[1]["totalElements"]

    def get_neuron(self, neuron_name: str) -> Dict[str, Any]:
        return self._get_json(f"neuron/name/{quote(neuron_name)}")

    # ------------------------------------------------------------------ morphometry
    def get_morphometry(self, neuron_name: str) -> Dict[str, Any]:
        """NeuroMorpho's L-Measure summary for one neuron (``GET /morphometry/name/{name}``)."""
        return self._get_json(f"morphometry/name/{quote(neuron_name)}")

    def iter_morphometry(self, filters: Dict[str, Sequence[str]], page_size: int = 500,
                         max_pages: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Paginated morphometry records for a filter (``POST /morphometry/select``)."""
        for _, records in self.iter_pages(filters, page_size=page_size, max_pages=max_pages, endpoint="morphometry"):
            yield from records

    # ------------------------------------------------------------------ SWC files
    @staticmethod
    def swc_url(archive: str, neuron_name: str) -> str:
        """URL of the standardised CNG.swc for a neuron."""
        return f"{DABLE_URL}/{quote(archive.lower())}/CNG%20version/{quote(neuron_name)}.CNG.swc"

    def download_swc(self, neuron: Dict[str, Any], out_dir: Path, overwrite: bool = False) -> Path:
        """Download the CNG.swc of ``neuron`` (a metadata record) into ``out_dir/<archive>/``."""
        archive = str(neuron["archive"])
        name = str(neuron["neuron_name"])
        target = Path(out_dir) / archive / f"{name}.CNG.swc"
        if target.exists() and not overwrite and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        resp = self._request("GET", self.swc_url(archive, name), absolute=True, stream=True)
        tmp = target.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        tmp.replace(target)
        return target


# ---------------------------------------------------------------------- helpers
def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten list-valued metadata fields into ``field`` (joined) and ``field_0`` (first entry)."""
    out: Dict[str, Any] = {}
    for key, value in rec.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            items = [str(v) for v in value]
            out[key] = "|".join(items)
            out[f"{key}_0"] = items[0] if items else None
            if key in _LIST_FIELDS:
                out[f"n_{key}"] = len(items)
        elif isinstance(value, dict):
            continue
        else:
            out[key] = value
    return out


def neurons_to_dataframe(records: Sequence[Dict[str, Any]]):
    """Convert API records into a flat ``pandas.DataFrame`` with typed batch columns."""
    import pandas as pd

    df = pd.DataFrame([flatten_record(r) for r in records])
    if df.empty:
        return df
    for col in ("slicing_thickness", "magnification", "min_age", "max_age", "soma_surface", "surface", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    if "shrinkage_corrected" in df.columns:
        df["shrinkage_corrected_flag"] = df["shrinkage_corrected"].astype(str).str.lower().str.startswith("corr")
    if "brain_region_0" in df.columns:
        df["brain_region_top"] = df["brain_region_0"]
    return df


__all__ = ["NeuroMorphoClient", "BATCH_FIELDS", "BIO_FIELDS", "flatten_record", "neurons_to_dataframe"]
