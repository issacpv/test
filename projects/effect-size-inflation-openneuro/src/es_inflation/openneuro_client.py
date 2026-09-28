"""Minimal, dependency-light client for the OpenNeuro GraphQL API.

The public endpoint is ``https://openneuro.org/crn/graphql``. The fields used here follow the
OpenNeuro schema (``Dataset``, ``Snapshot``, ``Summary``, ``Description``); if the server rejects a
field (schemas evolve), the client automatically retries with a minimal field set, and
``introspect_type`` prints what the live server offers.

A second, independent route is provided through the ``OpenNeuroDatasets`` GitHub organisation,
which mirrors every dataset's ``participants.tsv`` and ``dataset_description.json`` in plain git.

Nothing here needs credentials.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
import requests

GRAPHQL_URL = "https://openneuro.org/crn/graphql"
GITHUB_RAW = "https://raw.githubusercontent.com/OpenNeuroDatasets/{ds_id}/{branch}/{path}"

# Full query: everything the audit needs in one page.
DATASETS_QUERY = """
query DatasetsPage($first: Int!, $after: String) {
  datasets(first: $first, after: $after) {
    edges {
      node {
        id
        name
        created
        public
        latestSnapshot {
          tag
          created
          summary {
            subjects
            sessions
            modalities
            tasks
            totalFiles
            size
          }
          description {
            Name
            DatasetDOI
            ReferencesAndLinks
            Authors
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

# Fallback query if the server rejects any of the richer fields above.
MINIMAL_DATASETS_QUERY = """
query DatasetsPageMinimal($first: Int!, $after: String) {
  datasets(first: $first, after: $after) {
    edges {
      node {
        id
        name
        created
        latestSnapshot {
          tag
          summary { subjects modalities }
          description { Name DatasetDOI ReferencesAndLinks }
        }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

INTROSPECT_QUERY = """
query TypeFields($name: String!) {
  __type(name: $name) {
    fields { name type { name kind ofType { name kind ofType { name } } } }
  }
}
"""

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\],;]+", re.IGNORECASE)


class OpenNeuroAPIError(RuntimeError):
    """Raised when the GraphQL server returns errors or an unexpected payload."""


def extract_dois(text: str | Iterable[str] | None) -> List[str]:
    """Return unique DOIs found in a string or an iterable of strings (order preserved).

    Trailing punctuation that commonly follows a DOI in free text is stripped.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        text = " ".join(str(t) for t in text)
    out: List[str] = []
    for m in _DOI_RE.findall(text):
        doi = m.rstrip(".,;:")
        doi = doi.lower()
        if doi not in out:
            out.append(doi)
    return out


def flatten_dataset_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one ``datasets.edges[].node`` object into a flat record.

    ``n_subjects`` is the length of ``latestSnapshot.summary.subjects`` (the list of subject labels
    in the BIDS snapshot), which is the ground-truth n for the dataset as deposited.
    """
    snap = node.get("latestSnapshot") or {}
    summary = snap.get("summary") or {}
    desc = snap.get("description") or {}
    subjects = summary.get("subjects") or []
    sessions = summary.get("sessions") or []
    refs = desc.get("ReferencesAndLinks") or []
    if isinstance(refs, str):
        refs = [refs]
    authors = desc.get("Authors") or []
    if isinstance(authors, str):
        authors = [authors]
    dataset_doi = desc.get("DatasetDOI") or ""
    ref_dois = extract_dois(refs)
    return {
        "dataset_id": node.get("id"),
        "name": node.get("name") or desc.get("Name") or "",
        "created": node.get("created"),
        "public": node.get("public"),
        "latest_tag": snap.get("tag"),
        "snapshot_created": snap.get("created"),
        "n_subjects": int(len(subjects)),
        "n_sessions": int(len(sessions)),
        "modalities": "|".join(summary.get("modalities") or []),
        "tasks": "|".join(summary.get("tasks") or []),
        "total_files": summary.get("totalFiles"),
        "size_bytes": summary.get("size"),
        "dataset_doi": dataset_doi,
        "reference_dois": "|".join(ref_dois),
        "n_references": len(refs),
        "n_authors": len(authors),
    }


class OpenNeuroClient:
    """Paginating GraphQL client with retries, optional on-disk page cache and schema fallback.

    Parameters
    ----------
    url : GraphQL endpoint.
    session : a ``requests.Session`` (or any object with a compatible ``post``); injectable for tests.
    page_size : datasets per page (the server caps this; 100 is safe).
    max_retries, backoff : retry policy for network/5xx errors.
    cache_dir : if given, each page's JSON is written there (``page_0000.json`` ...).
    """

    def __init__(
        self,
        url: str = GRAPHQL_URL,
        session: Optional[Any] = None,
        page_size: int = 100,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff: float = 1.5,
        cache_dir: Optional[Path] = None,
        sleep_between_pages: float = 0.2,
    ) -> None:
        self.url = url
        self.session = session or requests.Session()
        self.page_size = int(page_size)
        self.timeout = timeout
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sleep_between_pages = sleep_between_pages
        self._query_in_use = DATASETS_QUERY

    # ------------------------------------------------------------------ raw
    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST one GraphQL query; return ``data``. Raise ``OpenNeuroAPIError`` on GraphQL errors."""
        payload = {"query": query, "variables": variables or {}}
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(self.url, json=payload, timeout=self.timeout)
                status = getattr(resp, "status_code", 200)
                if status in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {status}")
                body = resp.json()
                if body.get("errors"):
                    raise OpenNeuroAPIError(json.dumps(body["errors"])[:2000])
                return body.get("data") or {}
            except OpenNeuroAPIError:
                raise
            except Exception as exc:  # noqa: BLE001 - network layer
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff ** attempt)
        raise OpenNeuroAPIError(f"request failed after {self.max_retries + 1} attempts: {last_exc}")

    def introspect_type(self, type_name: str) -> List[str]:
        """Return ``name: Type`` strings for the fields of a schema type (for ``--introspect``)."""
        data = self.query(INTROSPECT_QUERY, {"name": type_name})
        t = data.get("__type") or {}
        out = []
        for f in t.get("fields") or []:
            ty = f.get("type") or {}
            name = ty.get("name") or (ty.get("ofType") or {}).get("name") or ((ty.get("ofType") or {}).get("ofType") or {}).get("name")
            out.append(f"{f['name']}: {name} ({ty.get('kind')})")
        return out

    # --------------------------------------------------------------- paging
    def iter_dataset_nodes(self, max_datasets: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield raw dataset nodes across all pages (``first``/``after`` cursor pagination)."""
        after: Optional[str] = None
        n_yielded = 0
        page_idx = 0
        while True:
            variables = {"first": self.page_size, "after": after}
            try:
                data = self.query(self._query_in_use, variables)
            except OpenNeuroAPIError as exc:
                if self._query_in_use is DATASETS_QUERY and _looks_like_schema_error(str(exc)):
                    self._query_in_use = MINIMAL_DATASETS_QUERY
                    data = self.query(self._query_in_use, variables)
                else:
                    raise
            conn = data.get("datasets") or {}
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                (self.cache_dir / f"page_{page_idx:04d}.json").write_text(json.dumps(conn))
            edges = conn.get("edges") or []
            for edge in edges:
                node = (edge or {}).get("node")
                if not node:
                    continue
                yield node
                n_yielded += 1
                if max_datasets is not None and n_yielded >= max_datasets:
                    return
            page_info = conn.get("pageInfo") or {}
            if not page_info.get("hasNextPage") or not edges:
                return
            after = page_info.get("endCursor")
            page_idx += 1
            if self.sleep_between_pages:
                time.sleep(self.sleep_between_pages)

    def fetch_datasets_table(self, max_datasets: Optional[int] = None) -> pd.DataFrame:
        """Return a DataFrame with one flattened row per dataset (see ``flatten_dataset_node``)."""
        rows = [flatten_dataset_node(n) for n in self.iter_dataset_nodes(max_datasets=max_datasets)]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates("dataset_id").sort_values("dataset_id").reset_index(drop=True)
        return df


def _looks_like_schema_error(msg: str) -> bool:
    msg = msg.lower()
    return "cannot query field" in msg or "unknown argument" in msg or "validation" in msg


# ------------------------------------------------------------ GitHub route
def count_subjects_from_participants_tsv(text: str) -> int:
    """Count participant rows in a BIDS ``participants.tsv`` (header + one row per subject).

    Blank lines and rows whose first column does not start with ``sub-`` are ignored, which guards
    against comment lines and trailing whitespace; duplicate ids are counted once.
    """
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, comment=None, skip_blank_lines=True)
    if df.empty or df.shape[1] == 0:
        return 0
    first = df.iloc[:, 0].astype(str).str.strip()
    ids = first[first.str.startswith("sub-")]
    return int(ids.nunique())


def iter_dataset_ids(start: int = 1, stop: int = 7000) -> Iterator[str]:
    """Yield OpenNeuro accession ids ``ds000001 ... ds00NNNN``."""
    for i in range(start, stop + 1):
        yield f"ds{i:06d}"


def fetch_dataset_from_github(
    ds_id: str,
    session: Optional[Any] = None,
    branches: Iterable[str] = ("master", "main"),
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Fetch ``participants.tsv`` + ``dataset_description.json`` from the GitHub mirror.

    Returns a record with the same keys as ``flatten_dataset_node`` (missing fields are ``None``),
    or ``None`` if the dataset has no mirror. ``n_subjects`` is ``None`` when the dataset has no
    ``participants.tsv`` (it is optional in BIDS).
    """
    session = session or requests.Session()
    desc: Optional[Dict[str, Any]] = None
    branch_used: Optional[str] = None
    for branch in branches:
        r = session.get(GITHUB_RAW.format(ds_id=ds_id, branch=branch, path="dataset_description.json"), timeout=timeout)
        if getattr(r, "status_code", 404) == 200:
            try:
                desc = r.json()
            except ValueError:
                desc = json.loads(r.text)
            branch_used = branch
            break
    if desc is None:
        return None
    n_subjects: Optional[int] = None
    r = session.get(GITHUB_RAW.format(ds_id=ds_id, branch=branch_used, path="participants.tsv"), timeout=timeout)
    if getattr(r, "status_code", 404) == 200:
        try:
            n_subjects = count_subjects_from_participants_tsv(r.text)
        except Exception:  # noqa: BLE001 - malformed TSV
            n_subjects = None
    refs = desc.get("ReferencesAndLinks") or []
    if isinstance(refs, str):
        refs = [refs]
    authors = desc.get("Authors") or []
    return {
        "dataset_id": ds_id,
        "name": desc.get("Name") or "",
        "created": None,
        "public": True,
        "latest_tag": None,
        "snapshot_created": None,
        "n_subjects": n_subjects,
        "n_sessions": None,
        "modalities": "",
        "tasks": "",
        "total_files": None,
        "size_bytes": None,
        "dataset_doi": desc.get("DatasetDOI") or "",
        "reference_dois": "|".join(extract_dois(refs)),
        "n_references": len(refs),
        "n_authors": len(authors) if isinstance(authors, list) else 1,
    }
