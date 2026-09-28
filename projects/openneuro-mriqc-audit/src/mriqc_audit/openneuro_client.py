"""OpenNeuro GraphQL client and MD5 linkage helpers.

The public GraphQL endpoint is ``https://openneuro.org/crn/graphql``.  Field
names below were checked against the server schema
(``packages/openneuro-server/src/graphql/schema/*.ts`` in
https://github.com/OpenNeuroOrg/openneuro):

* ``datasets(first, after, orderBy, filterBy, modality)`` returns a connection
  with ``edges { node }`` and ``pageInfo { hasNextPage endCursor }``
  (``first`` is capped at 100 by the server).
* ``Dataset { id name created public latestSnapshot metadata }``
* ``Snapshot { tag created description { Name DatasetDOI License }
  summary { modalities subjects sessions tasks size totalFiles dataProcessed
  subjectMetadata { participantId age sex group } } files(recursive) { filename urls size annexed } }``
* ``Metadata { studyDomain species studyLongitudinal dataProcessed
  associatedPaperDOI affirmedDefaced }`` (curated dataset-level metadata).

Linking IQMs to OpenNeuro files
-------------------------------
MRIQC WebAPI records carry ``provenance.md5sum`` = MD5 of the *bytes* of the
input NIfTI (nipype ``hash_infile``), while ``bids_meta.subject_id`` is
SHA-256 hashed.  OpenNeuro stores annexed files under git-annex **MD5E** keys
``MD5E-s<size>--<md5>.nii.gz``.  A plain ``git clone`` of
``https://github.com/OpenNeuroDatasets/<dsid>`` (no annex content) therefore
exposes every file's MD5 through its symlink target, which is what
:func:`annex_md5_index` reads.  Joining on MD5 recovers dataset, subject and
therefore ``participants.tsv`` age/sex/group for each crowdsourced IQM record
without downloading a single image.
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests

GQL_URL = "https://openneuro.org/crn/graphql"

DATASETS_QUERY = """
query ListDatasets($first: Int!, $after: String, $modality: String) {
  datasets(first: $first, after: $after, modality: $modality) {
    edges {
      node {
        id
        name
        created
        public
        metadata {
          studyDomain
          species
          studyLongitudinal
          dataProcessed
          associatedPaperDOI
          affirmedDefaced
        }
        latestSnapshot {
          tag
          created
          description { Name DatasetDOI License }
          summary {
            modalities
            subjects
            sessions
            tasks
            size
            totalFiles
            dataProcessed
            subjectMetadata { participantId age sex group }
          }
        }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

SNAPSHOT_FILES_QUERY = """
query SnapshotFiles($datasetId: ID!, $tag: String!) {
  snapshot(datasetId: $datasetId, tag: $tag) {
    id
    tag
    files(recursive: true) { filename size urls annexed }
  }
}
"""

_ANNEX_KEY_RE = re.compile(r"MD5E-s(?P<size>\d+)--(?P<md5>[0-9a-f]{32})(?P<ext>\.[^/]*)?$")


@dataclass
class OpenNeuroClient:
    """Small GraphQL client with retry/backoff and cursor pagination.

    Parameters
    ----------
    url
        GraphQL endpoint.
    token
        Optional API key (OpenNeuro account -> "API key"); only needed for
        private datasets.  Read from ``OPENNEURO_API_KEY`` if not given.
    sleep
        Seconds to wait between pages (be polite to the shared server).
    """

    url: str = GQL_URL
    token: Optional[str] = None
    sleep: float = 0.2
    timeout: float = 60.0
    max_retries: int = 5

    def __post_init__(self) -> None:
        self.session = requests.Session()
        tok = self.token or os.environ.get("OPENNEURO_API_KEY")
        if tok:
            self.session.cookies.set("accessToken", tok)

    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST one GraphQL query; raises on transport or GraphQL errors."""
        payload = {"query": query, "variables": variables or {}}
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout)
                if r.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} from server")
                r.raise_for_status()
                body = r.json()
                if body.get("errors"):
                    raise RuntimeError(f"GraphQL errors: {body['errors']}")
                return body["data"]
            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
                last = e  # noqa: F841
        raise RuntimeError("unreachable")

    # ---------------------------------------------------------------- datasets
    def iter_datasets(self, page_size: int = 100, modality: Optional[str] = None,
                      max_datasets: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield dataset nodes across all pages (cursor pagination)."""
        after: Optional[str] = None
        n = 0
        while True:
            data = self.query(DATASETS_QUERY, {"first": min(page_size, 100), "after": after, "modality": modality})
            conn = data["datasets"]
            for edge in conn["edges"]:
                node = edge["node"]
                if node is None:
                    continue
                yield node
                n += 1
                if max_datasets and n >= max_datasets:
                    return
            if not conn["pageInfo"]["hasNextPage"]:
                return
            after = conn["pageInfo"]["endCursor"]
            time.sleep(self.sleep)

    def list_datasets(self, **kwargs) -> pd.DataFrame:
        """Dataset-level table: one row per dataset (see :func:`flatten_dataset`)."""
        return pd.DataFrame([flatten_dataset(n) for n in self.iter_datasets(**kwargs)])

    def subject_metadata(self, **kwargs) -> pd.DataFrame:
        """Participant-level table from ``summary.subjectMetadata`` across datasets."""
        rows: List[Dict[str, Any]] = []
        for node in self.iter_datasets(**kwargs):
            snap = node.get("latestSnapshot") or {}
            summ = snap.get("summary") or {}
            for s in summ.get("subjectMetadata") or []:
                rows.append({"dataset_id": node["id"], "tag": snap.get("tag"), **s})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ files
    def snapshot_files(self, dataset_id: str, tag: str) -> pd.DataFrame:
        data = self.query(SNAPSHOT_FILES_QUERY, {"datasetId": dataset_id, "tag": tag})
        snap = data["snapshot"] or {}
        df = pd.DataFrame(snap.get("files") or [])
        if not df.empty:
            df["urls"] = df["urls"].apply(lambda u: (u or [None])[0])
        return df

    def fetch_participants_tsv(self, dataset_id: str, tag: str) -> Optional[pd.DataFrame]:
        """Download ``participants.tsv`` of a snapshot (None if absent)."""
        files = self.snapshot_files(dataset_id, tag)
        if files.empty:
            return None
        hit = files[files["filename"] == "participants.tsv"]
        if hit.empty or not hit.iloc[0]["urls"]:
            return None
        r = self.session.get(hit.iloc[0]["urls"], timeout=self.timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str, na_values=["n/a", "NA", ""])
        df.insert(0, "dataset_id", dataset_id)
        return df


def flatten_dataset(node: Dict[str, Any]) -> Dict[str, Any]:
    """One tidy row per dataset node returned by :data:`DATASETS_QUERY`."""
    snap = node.get("latestSnapshot") or {}
    summ = snap.get("summary") or {}
    desc = snap.get("description") or {}
    meta = node.get("metadata") or {}
    subj_meta = summ.get("subjectMetadata") or []
    ages = pd.to_numeric(pd.Series([s.get("age") for s in subj_meta]), errors="coerce")
    return {
        "dataset_id": node.get("id"),
        "name": node.get("name") or desc.get("Name"),
        "created": node.get("created"),
        "public": node.get("public"),
        "tag": snap.get("tag"),
        "snapshot_created": snap.get("created"),
        "doi": desc.get("DatasetDOI"),
        "license": desc.get("License"),
        "modalities": ",".join(summ.get("modalities") or []),
        "n_subjects": len(summ.get("subjects") or []),
        "n_sessions": len(summ.get("sessions") or []),
        "n_tasks": len(summ.get("tasks") or []),
        "size_bytes": summ.get("size"),
        "total_files": summ.get("totalFiles"),
        "data_processed": summ.get("dataProcessed"),
        "n_participants_tsv": len(subj_meta),
        "age_median": float(ages.median()) if ages.notna().any() else None,
        "age_min": float(ages.min()) if ages.notna().any() else None,
        "age_max": float(ages.max()) if ages.notna().any() else None,
        "frac_pediatric": float((ages < 18).mean()) if ages.notna().any() else None,
        "n_groups": len({s.get("group") for s in subj_meta if s.get("group")}),
        "study_domain": meta.get("studyDomain"),
        "species": meta.get("species"),
        "study_longitudinal": meta.get("studyLongitudinal"),
        "associated_paper_doi": meta.get("associatedPaperDOI"),
        "affirmed_defaced": meta.get("affirmedDefaced"),
    }


# ------------------------------------------------------------------ MD5 linkage
def parse_annex_key(target: str) -> Optional[Tuple[str, int]]:
    """Extract ``(md5, size)`` from a git-annex MD5E key or symlink target.

    >>> parse_annex_key('.git/annex/objects/Xk/9Q/MD5E-s1234--0123456789abcdef0123456789abcdef.nii.gz/MD5E-s1234--0123456789abcdef0123456789abcdef.nii.gz')
    ('0123456789abcdef0123456789abcdef', 1234)
    """
    m = _ANNEX_KEY_RE.search(target)
    if not m:
        return None
    return m.group("md5"), int(m.group("size"))


def annex_md5_index(dataset_dir: Path, suffixes: Tuple[str, ...] = (".nii.gz", ".nii")) -> pd.DataFrame:
    """Walk a git-cloned OpenNeuro dataset and index annexed images by MD5.

    Works on a *bare* ``git clone`` (symlinks present, annex content absent) or
    on a datalad clone.  Returns columns ``filename``, ``md5``, ``size``.
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    for root, _dirs, files in os.walk(dataset_dir):
        if ".git" in Path(root).parts:
            continue
        for f in files:
            if not f.endswith(suffixes):
                continue
            p = Path(root) / f
            target = None
            if p.is_symlink():
                target = os.readlink(p)
            else:
                # git may check out annex pointer files (unlocked) as text
                try:
                    with p.open("rb") as fh:
                        head = fh.read(200)
                    if head.startswith(b"/annex/objects/") or b"MD5E-s" in head:
                        target = head.decode("utf-8", "ignore")
                except OSError:
                    target = None
            if not target:
                continue
            parsed = parse_annex_key(target)
            if parsed:
                rows.append({"filename": str(p.relative_to(dataset_dir)), "md5": parsed[0], "size": parsed[1]})
    return pd.DataFrame(rows, columns=["filename", "md5", "size"])


def bids_entities(filename: str) -> Dict[str, str]:
    """Parse ``sub-01/ses-1/anat/sub-01_ses-1_run-2_T1w.nii.gz`` into entities."""
    name = Path(filename).name
    ent: Dict[str, str] = {}
    for part in name.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            ent[k] = v
        else:
            ent["suffix"] = part.split(".")[0]
    return ent


def link_iqms_to_openneuro(iqms: pd.DataFrame, md5_index: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Inner-join MRIQC records (``provenance.md5sum``) with an annex MD5 index."""
    idx = md5_index.copy()
    idx["dataset_id"] = dataset_id
    ents = idx["filename"].apply(bids_entities).apply(pd.Series)
    idx = pd.concat([idx, ents.add_prefix("bids_")], axis=1)
    return iqms.merge(idx, left_on="provenance.md5sum", right_on="md5", how="inner")
