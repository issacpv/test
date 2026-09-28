"""Paginated client for the MRIQC WebAPI.

The crowdsourced IQM database (Esteban et al., 2019, *Sci Data*) is served by
a python-eve REST API at ``https://mriqc.nimh.nih.gov/api/v1/<modality>`` with
``modality`` in ``{T1w, T2w, bold}``.  Eve conventions used here:

* ``?page=<k>&max_results=<n>`` for pagination (``_meta.total`` gives the
  total count, ``_links.next`` exists while more pages remain);
* ``?where=<JSON>`` for server-side filtering, e.g.
  ``{"bids_meta.MagneticFieldStrength": 3}`` or
  ``{"provenance.version": {"$regex": "^24"}}``;
* records hold ``bids_meta`` (whitelisted BIDS sidecar fields plus hashed
  ``subject_id``/``session_id``), ``provenance`` (``md5sum``, ``version``,
  ``software``, ``settings``) and the IQMs at top level.

No authentication is needed for GET.  Uploads (POST) require a token and are
out of scope.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import requests

API_ROOT = "https://mriqc.nimh.nih.gov/api/v1"

# IQMs reported by MRIQC (Esteban et al., 2017, PLoS ONE); kept as reference
# lists so downstream code can select columns robustly.
IQM_T1W: List[str] = [
    "cjv", "cnr", "efc", "fber", "fwhm_avg", "fwhm_x", "fwhm_y", "fwhm_z",
    "icvs_csf", "icvs_gm", "icvs_wm", "inu_med", "inu_range",
    "qi_1", "qi_2", "rpve_csf", "rpve_gm", "rpve_wm",
    "snr_csf", "snr_gm", "snr_wm", "snr_total", "snrd_csf", "snrd_gm", "snrd_wm", "snrd_total",
    "summary_bg_mean", "summary_bg_stdv", "summary_bg_p05", "summary_bg_p95",
    "summary_csf_mean", "summary_gm_mean", "summary_wm_mean",
    "tpm_overlap_csf", "tpm_overlap_gm", "tpm_overlap_wm", "wm2max",
]
IQM_BOLD: List[str] = [
    "aor", "aqi", "dummy_trs", "dvars_nstd", "dvars_std", "dvars_vstd", "efc", "fber",
    "fd_mean", "fd_num", "fd_perc", "fwhm_avg", "fwhm_x", "fwhm_y", "fwhm_z",
    "gcor", "gsr_x", "gsr_y", "snr", "tsnr", "summary_bg_mean", "summary_fg_mean",
]
BIDS_META = [
    "bids_meta.MagneticFieldStrength", "bids_meta.Manufacturer", "bids_meta.ManufacturersModelName",
    "bids_meta.RepetitionTime", "bids_meta.EchoTime", "bids_meta.FlipAngle", "bids_meta.modality",
    "bids_meta.subject_id", "bids_meta.session_id", "bids_meta.run_id", "bids_meta.task_id",
]


@dataclass
class MRIQCClient:
    """Iterate over WebAPI records with pagination, retry and optional caching."""

    root: str = API_ROOT
    max_results: int = 1000
    sleep: float = 0.5
    timeout: float = 120.0
    max_retries: int = 5

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def _get(self, modality: str, page: int, where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        params: Dict[str, Any] = {"max_results": self.max_results, "page": page}
        if where:
            params["where"] = json.dumps(where)
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(f"{self.root}/{modality}", params=params, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def count(self, modality: str = "T1w", where: Optional[Dict[str, Any]] = None) -> int:
        body = self._get(modality, 1, where)
        return int(body.get("_meta", {}).get("total", len(body.get("_items", []))))

    def iter_records(self, modality: str = "T1w", where: Optional[Dict[str, Any]] = None,
                     max_pages: Optional[int] = None, start_page: int = 1) -> Iterator[Dict[str, Any]]:
        """Yield raw records page by page until ``_links.next`` disappears."""
        page = start_page
        while True:
            body = self._get(modality, page, where)
            items = body.get("_items", [])
            for it in items:
                yield it
            has_next = "next" in body.get("_links", {}) and len(items) > 0
            if not has_next or (max_pages and page - start_page + 1 >= max_pages):
                return
            page += 1
            time.sleep(self.sleep)

    def fetch(self, modality: str = "T1w", where: Optional[Dict[str, Any]] = None,
              max_pages: Optional[int] = None, cache: Optional[Path] = None) -> pd.DataFrame:
        """Fetch records into a flat DataFrame; optionally append raw JSON lines to ``cache``."""
        records: List[Dict[str, Any]] = []
        fh = open(cache, "a") if cache else None
        try:
            for rec in self.iter_records(modality, where, max_pages):
                records.append(rec)
                if fh:
                    fh.write(json.dumps(rec) + "\n")
        finally:
            if fh:
                fh.close()
        return records_to_frame(records, modality)


def records_to_frame(records: List[Dict[str, Any]], modality: str = "T1w") -> pd.DataFrame:
    """Flatten nested WebAPI records (``bids_meta.*``, ``provenance.*``, IQMs).

    Drops the ``provenance.settings`` sub-tree (large, rarely useful) and
    coerces IQM columns to float.  Adds ``modality`` and keeps ``_id``,
    ``_created``, ``_updated``.
    """
    if not records:
        return pd.DataFrame()
    slim = []
    for rec in records:
        rec = dict(rec)
        prov = dict(rec.get("provenance") or {})
        prov.pop("settings", None)
        prov.pop("warnings", None)
        rec["provenance"] = prov
        rec.pop("_links", None)
        rec.pop("_etag", None)
        slim.append(rec)
    df = pd.json_normalize(slim, sep=".")
    df["modality"] = modality
    iqm_cols = [c for c in (IQM_T1W if modality != "bold" else IQM_BOLD) if c in df.columns]
    df[iqm_cols] = df[iqm_cols].apply(pd.to_numeric, errors="coerce")
    for c in ("bids_meta.MagneticFieldStrength", "bids_meta.RepetitionTime", "bids_meta.EchoTime"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "_created" in df.columns:
        df["_created"] = pd.to_datetime(df["_created"], errors="coerce", utc=True)
    return df


def dedupe_records(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    """Drop repeated uploads of the same image.

    Duplicates are defined by ``provenance.md5sum`` + ``provenance.version``
    (the same file processed by different MRIQC versions is kept, because
    IQM definitions changed across versions).  Sorted by ``_created`` first so
    ``keep="last"`` retains the most recent upload.
    """
    if df.empty or "provenance.md5sum" not in df.columns:
        return df
    key = ["provenance.md5sum"] + (["provenance.version"] if "provenance.version" in df.columns else [])
    if "_created" in df.columns:
        df = df.sort_values("_created")
    return df.drop_duplicates(subset=key, keep=keep).reset_index(drop=True)


def major_version(v: Any) -> Optional[str]:
    """``'23.1.0'`` -> ``'23'``; None for missing."""
    if not isinstance(v, str) or not v:
        return None
    return v.split(".")[0]


def summarise_by_scanner(df: pd.DataFrame, iqms: Optional[List[str]] = None) -> pd.DataFrame:
    """Median IQMs by (Manufacturer, field strength, MRIQC major version)."""
    iqms = iqms or [c for c in IQM_T1W + IQM_BOLD if c in df.columns]
    g = df.assign(mriqc_major=df.get("provenance.version", pd.Series(index=df.index)).map(major_version))
    keys = [k for k in ("bids_meta.Manufacturer", "bids_meta.MagneticFieldStrength", "mriqc_major") if k in g.columns]
    out = g.groupby(keys, dropna=False)[iqms].median()
    out["n"] = g.groupby(keys, dropna=False).size()
    return out.reset_index()


def load_jsonl(path: Path, modality: str) -> pd.DataFrame:
    """Reload a cache written by :meth:`MRIQCClient.fetch`."""
    with open(path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return records_to_frame(records, modality)
