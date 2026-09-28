"""ClinicalTrials.gov v2 API client for digital-health RCTs and their IPD-sharing statements.

The registry step needs no data-use agreement and is a result in itself: how
many completed digital-health RCTs say they will share IPD, where, and under
which conditions.  The v2 API (https://clinicaltrials.gov/data-api/api) returns
JSON with a ``protocolSection`` per study; pagination is by ``nextPageToken``.
"""
from __future__ import annotations

import re
import time
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

CT_API = "https://clinicaltrials.gov/api/v2/studies"

#: search phrases for digital-health interventions (intervention field)
DIGITAL_TERMS: Sequence[str] = (
    "smartphone app",
    "mobile app",
    "mobile application",
    "digital therapeutic",
    "internet-delivered",
    "web-based intervention",
    "text message",
    "SMS",
    "remote monitoring",
    "telemonitoring",
    "wearable",
    "chatbot",
    "virtual reality",
    "digital intervention",
)
DIGITAL_RE = re.compile(r"\b(app|apps|smartphone|mobile|digital|internet|web[- ]based|online|sms|text messag|e-?health|m-?health|telemonitor|remote monitor|wearable|chatbot|virtual reality|video game|computeri[sz]ed)\b", re.I)

FIELDS: Sequence[str] = (
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "StartDate",
    "PrimaryCompletionDate",
    "StudyType",
    "Phase",
    "DesignAllocation",
    "DesignInterventionModel",
    "EnrollmentCount",
    "LeadSponsorName",
    "LeadSponsorClass",
    "InterventionType",
    "InterventionName",
    "Condition",
    "IPDSharing",
    "IPDSharingDescription",
    "IPDSharingInfoType",
    "IPDSharingTimeFrame",
    "IPDSharingAccessCriteria",
    "IPDSharingURL",
    "HasResults",
)

REPO_PATTERNS: Sequence[tuple[str, str]] = (
    ("vivli", r"vivli"),
    ("yoda", r"\byoda\b|yale open data"),
    ("csdr", r"clinicalstudydatarequest|\bcsdr\b"),
    ("nda", r"\bnda\b|nimh data archive|nda\.nih\.gov"),
    ("osf", r"\bosf\.io|open science framework"),
    ("zenodo", r"zenodo"),
    ("dryad", r"dryad"),
    ("figshare", r"figshare"),
    ("icpsr", r"icpsr"),
    ("github", r"github"),
    ("mendeley", r"mendeley data"),
    ("on_request", r"upon (reasonable )?request|on request|contact(ing)? the (pi|principal investigator|corresponding author)"),
)


def build_params(term: str, start_year: int = 2015, status: Sequence[str] = ("COMPLETED",), study_type: str = "INTERVENTIONAL", page_size: int = 100, ipd_yes_only: bool = False) -> Dict[str, str]:
    """Query parameters for one intervention search term."""
    q_term = f"AREA[StartDate]RANGE[{start_year}-01-01,MAX]"
    if ipd_yes_only:
        q_term += " AND AREA[IPDSharing]Yes"
    return {
        "query.intr": term,
        "query.term": q_term,
        "filter.overallStatus": ",".join(status),
        "filter.advanced": f"AREA[StudyType]{study_type}",
        "fields": ",".join(FIELDS),
        "pageSize": str(page_size),
        "countTotal": "true",
        "format": "json",
    }


def fetch_studies(params: Dict[str, str], max_pages: int = 50, session: Optional[requests.Session] = None, sleep_s: float = 0.3) -> List[dict]:
    """Page through the v2 ``/studies`` endpoint; returns raw study dicts."""
    sess = session or requests.Session()
    out: List[dict] = []
    token: Optional[str] = None
    for _ in range(max_pages):
        p = dict(params)
        if token:
            p["pageToken"] = token
        resp = sess.get(CT_API, params=p, timeout=60, headers={"User-Agent": "dh-ipd-registry"})
        if resp.status_code == 429:
            time.sleep(5)
            continue
        resp.raise_for_status()
        payload = resp.json()
        out.extend(payload.get("studies", []))
        token = payload.get("nextPageToken")
        if not token:
            break
        time.sleep(sleep_s)
    return out


def _get(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def flatten_study(study: dict) -> Dict[str, object]:
    """One flat record per study from the v2 JSON."""
    ps = study.get("protocolSection", {})
    ipd = ps.get("ipdSharingStatementModule", {}) or {}
    interventions = _get(ps, "armsInterventionsModule", "interventions", default=[]) or []
    return {
        "nct_id": _get(ps, "identificationModule", "nctId"),
        "title": _get(ps, "identificationModule", "briefTitle"),
        "official_title": _get(ps, "identificationModule", "officialTitle"),
        "status": _get(ps, "statusModule", "overallStatus"),
        "start_date": _get(ps, "statusModule", "startDateStruct", "date"),
        "completion_date": _get(ps, "statusModule", "primaryCompletionDateStruct", "date"),
        "study_type": _get(ps, "designModule", "studyType"),
        "phases": ";".join(_get(ps, "designModule", "phases", default=[]) or []),
        "allocation": _get(ps, "designModule", "designInfo", "allocation"),
        "enrollment": _get(ps, "designModule", "enrollmentInfo", "count"),
        "sponsor": _get(ps, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "sponsor_class": _get(ps, "sponsorCollaboratorsModule", "leadSponsor", "class"),
        "intervention_types": ";".join(sorted({i.get("type", "") for i in interventions})),
        "intervention_names": ";".join(i.get("name", "") for i in interventions),
        "conditions": ";".join(_get(ps, "conditionsModule", "conditions", default=[]) or []),
        "ipd_sharing": ipd.get("ipdSharing"),
        "ipd_description": ipd.get("description"),
        "ipd_info_types": ";".join(ipd.get("infoTypes", []) or []),
        "ipd_time_frame": ipd.get("timeFrame"),
        "ipd_access_criteria": ipd.get("accessCriteria"),
        "ipd_url": ipd.get("url"),
        "has_results": study.get("hasResults"),
    }


def classify_repository(text: Optional[str]) -> str:
    """Name the repository/mechanism mentioned in an IPD-sharing statement (first match wins)."""
    t = (text or "").lower()
    for name, pat in REPO_PATTERNS:
        if re.search(pat, t):
            return name
    return "none" if t.strip() else "empty"


def is_digital_intervention(row: Dict[str, object]) -> bool:
    """Heuristic: digital keywords in intervention names/title, and not a drug-only trial."""
    text = " ".join(str(row.get(k, "") or "") for k in ("intervention_names", "title", "official_title"))
    types = str(row.get("intervention_types", "") or "")
    return bool(DIGITAL_RE.search(text)) and ("DRUG" not in types or "DEVICE" in types or "BEHAVIORAL" in types)


def studies_to_frame(studies: Iterable[dict]) -> pd.DataFrame:
    """Flatten, de-duplicate by NCT id, add year, repository class and digital flag."""
    df = pd.DataFrame([flatten_study(s) for s in studies])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)
    df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
    df["ipd_repository"] = [classify_repository(f"{a or ''} {b or ''} {c or ''}") for a, b, c in zip(df["ipd_description"], df["ipd_url"], df["ipd_access_criteria"])]
    df["digital"] = [is_digital_intervention(r) for r in df.to_dict("records")]
    df["ipd_yes"] = df["ipd_sharing"].astype(str).str.upper().eq("YES")
    return df


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))


def sharing_summary(df: pd.DataFrame, by: Sequence[str] = ("sponsor_class",)) -> pd.DataFrame:
    """Share of trials with IPD sharing = YES, by strata, with Wilson CIs."""
    rows = []
    for key, g in df.groupby(list(by), dropna=False):
        n, k = len(g), int(g["ipd_yes"].sum())
        lo, hi = wilson(k, n)
        rec = {b: (key[i] if isinstance(key, tuple) else key) for i, b in enumerate(by)}
        rec.update({"n": n, "n_ipd_yes": k, "share_yes": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi})
        for repo in ("vivli", "yoda", "csdr", "nda", "osf", "on_request"):
            rec[f"n_{repo}"] = int((g["ipd_repository"] == repo).sum())
        rows.append(rec)
    return pd.DataFrame(rows)
