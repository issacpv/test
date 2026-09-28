"""OpenNeuro GraphQL registry search for transcranial electrical stimulation datasets.

The public endpoint ``https://openneuro.org/crn/graphql`` exposes ``datasets``
with cursor pagination. We page through all public datasets, keep those whose
name / README mention tES keywords, and record whether a T1w (``anat``)
modality is present so that individual head models can be built.

Only :func:`fetch_all_datasets` touches the network; parsing and filtering
are pure functions and are unit-tested on synthetic payloads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

TES_PATTERNS = [
    r"\btdcs\b",
    r"\btacs\b",
    r"\btrns\b",
    r"\btes\b",
    r"\bhd-?tdcs\b",
    r"transcranial (direct|alternating|random noise|electric(al)?)",
    r"electrical stimulation",
]
_TES_RE = re.compile("|".join(TES_PATTERNS), flags=re.IGNORECASE)

QUERY = """
query ($cursor: String) {
  datasets(first: 100, after: $cursor, filterBy: {}, orderBy: {created: descending}) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        name
        latestSnapshot {
          tag
          readme
          summary { modalities subjects sessions tasks }
          description { Name }
        }
      }
    }
  }
}
"""


@dataclass
class DatasetRecord:
    """Minimal description of an OpenNeuro dataset relevant to the registry."""

    accession: str
    name: str
    tag: str = ""
    n_subjects: int = 0
    n_sessions: int = 0
    modalities: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    readme: str = ""

    @property
    def has_t1w(self) -> bool:
        """True when the snapshot summary lists an anatomical (T1w/MRI) modality."""
        mods = {m.lower() for m in self.modalities}
        return bool(mods & {"t1w", "anat", "mri"})

    @property
    def is_tes(self) -> bool:
        return bool(_TES_RE.search(f"{self.name}\n{self.readme}"))


def parse_dataset_nodes(payload: dict[str, Any]) -> list[DatasetRecord]:
    """Convert one GraphQL response page into :class:`DatasetRecord` objects.

    Parameters
    ----------
    payload
        The JSON-decoded response (``{"data": {"datasets": {...}}}``).
    """
    edges = payload.get("data", {}).get("datasets", {}).get("edges", [])
    out: list[DatasetRecord] = []
    for edge in edges:
        node = edge.get("node") or {}
        snap = node.get("latestSnapshot") or {}
        summary = snap.get("summary") or {}
        desc = snap.get("description") or {}
        subjects = summary.get("subjects") or []
        sessions = summary.get("sessions") or []
        out.append(
            DatasetRecord(
                accession=str(node.get("id", "")),
                name=str(node.get("name") or desc.get("Name") or ""),
                tag=str(snap.get("tag") or ""),
                n_subjects=len(subjects),
                n_sessions=len(sessions),
                modalities=list(summary.get("modalities") or []),
                tasks=list(summary.get("tasks") or []),
                readme=str(snap.get("readme") or ""),
            )
        )
    return out


def page_info(payload: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(has_next_page, end_cursor)`` from a GraphQL page."""
    info = payload.get("data", {}).get("datasets", {}).get("pageInfo", {})
    return bool(info.get("hasNextPage")), info.get("endCursor")


def filter_tes_datasets(records: Iterable[DatasetRecord]) -> list[DatasetRecord]:
    """Keep datasets whose name or README mention tES keywords."""
    return [r for r in records if r.is_tes]


def registry_frame(records: Iterable[DatasetRecord]) -> pd.DataFrame:
    """Tabulate records for ``data/registry.csv`` with empty hand-curation columns."""
    rows = [
        {
            "accession": r.accession,
            "name": r.name,
            "tag": r.tag,
            "n_subjects": r.n_subjects,
            "n_sessions": r.n_sessions,
            "modalities": ";".join(r.modalities),
            "tasks": ";".join(r.tasks),
            "has_t1w": r.has_t1w,
            "montage": "",
            "current_mA": "",
            "target_roi": "",
            "outcome_column": "",
        }
        for r in records
    ]
    cols = [
        "accession", "name", "tag", "n_subjects", "n_sessions", "modalities", "tasks",
        "has_t1w", "montage", "current_mA", "target_roi", "outcome_column",
    ]
    return pd.DataFrame(rows, columns=cols).sort_values("accession").reset_index(drop=True)


def fetch_all_datasets(max_pages: int | None = None, timeout: float = 60.0) -> list[DatasetRecord]:
    """Page through the OpenNeuro GraphQL API and return all dataset records.

    Requires network access; ``requests`` is imported lazily so the rest of the
    module stays importable offline.
    """
    import requests

    records: list[DatasetRecord] = []
    cursor: str | None = None
    page = 0
    while True:
        resp = requests.post(GRAPHQL_URL, json={"query": QUERY, "variables": {"cursor": cursor}}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records.extend(parse_dataset_nodes(payload))
        has_next, cursor = page_info(payload)
        page += 1
        if not has_next or (max_pages is not None and page >= max_pages):
            break
    return records
