"""Building the audit corpus: GitHub search, shallow clones and paper links.

Two real APIs are used (both read credentials from environment variables and
handle pagination / rate limits):

* GitHub REST API v3 (``GITHUB_TOKEN``): repository and code search for dataset
  markers such as ``ptbxl_database.csv`` or ``chb01_03.edf``.  Code search
  requires authentication; repository search works unauthenticated at a very
  low rate limit.
* Semantic Scholar Graph API (``S2_API_KEY`` optional): papers citing a dataset
  descriptor (e.g. the PTB-XL Scientific Data paper), whose abstracts/PDF links
  are then regex-mined for GitHub URLs.

Nothing here executes downloaded code; repositories are cloned with
``--depth 1`` and scanned statically by :mod:`leakscan.detector`.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

GITHUB_API = "https://api.github.com"
S2_API = "https://api.semanticscholar.org/graph/v1"

#: dataset marker -> GitHub search phrases (code search) and repo search terms
DATASET_QUERIES: Dict[str, Dict[str, List[str]]] = {
    "chbmit": {
        "code": ['"chb01_03.edf"', '"physionet.org/content/chbmit"', '"chb-mit"'],
        "repo": ["chb-mit seizure", "chbmit seizure detection"],
    },
    "ptbxl": {
        "code": ['"ptbxl_database.csv"', '"scp_statements.csv"'],
        "repo": ["ptb-xl ecg classification", "ptbxl"],
    },
    "mimic": {
        "code": ['"mimiciv" "icustays"', '"mimic-iv" "chartevents"', '"MIMIC-III" "CHARTEVENTS"'],
        "repo": ["mimic-iv mortality prediction", "mimic-iii benchmark"],
    },
    "bonn": {"code": ['"Bonn" "EEG" "Z.zip"', '"bonn" eeg "F.zip"'], "repo": ["bonn eeg dataset seizure"]},
    "deap": {"code": ['"data_preprocessed_python"'], "repo": ["deap eeg emotion recognition"]},
    "sleep-edf": {"code": ['"sleep-cassette"', '"SC4001E0-PSG.edf"'], "repo": ["sleep-edf sleep staging"]},
}

#: DOIs of dataset descriptor papers used for citation mining via Semantic Scholar
DATASET_DOIS: Dict[str, str] = {
    "ptbxl": "10.1038/s41597-020-0495-6",  # Wagner et al. 2020, Sci Data
    "mimic-iv": "10.1038/s41597-022-01899-x",  # Johnson et al. 2023, Sci Data
    "eicu": "10.1038/sdata.2018.178",  # Pollard et al. 2018, Sci Data
    "helsinki": "10.1038/sdata.2019.39",  # Stevenson et al. 2019, Sci Data
}

GITHUB_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def _gh_headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "leakscan-audit"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _sleep_for_rate_limit(resp: requests.Response) -> None:
    reset = resp.headers.get("X-RateLimit-Reset")
    if reset:
        wait = max(0.0, float(reset) - time.time()) + 1.0
        time.sleep(min(wait, 120.0))
    else:
        time.sleep(30.0)


def github_search(kind: str, query: str, token: Optional[str] = None, max_pages: int = 10, per_page: int = 100, session: Optional[requests.Session] = None) -> Iterator[Dict[str, object]]:
    """Iterate over GitHub search results (``kind`` = ``'repositories'`` or ``'code'``).

    Handles pagination (``page`` parameter, up to GitHub's 1,000-result cap) and
    secondary rate limits (sleeps until ``X-RateLimit-Reset``).
    """
    token = token or os.environ.get("GITHUB_TOKEN")
    if kind == "code" and not token:
        raise RuntimeError("GitHub code search requires GITHUB_TOKEN in the environment.")
    sess = session or requests.Session()
    for page in range(1, max_pages + 1):
        resp = sess.get(f"{GITHUB_API}/search/{kind}", params={"q": query, "per_page": per_page, "page": page}, headers=_gh_headers(token), timeout=60)
        if resp.status_code in (403, 429):
            _sleep_for_rate_limit(resp)
            resp = sess.get(f"{GITHUB_API}/search/{kind}", params={"q": query, "per_page": per_page, "page": page}, headers=_gh_headers(token), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("items", [])
        for it in items:
            if kind == "code":
                repo = it.get("repository", {})
                yield {"full_name": repo.get("full_name"), "html_url": repo.get("html_url"), "path": it.get("path"), "query": query}
            else:
                yield {
                    "full_name": it.get("full_name"),
                    "html_url": it.get("html_url"),
                    "stars": it.get("stargazers_count"),
                    "created_at": it.get("created_at"),
                    "pushed_at": it.get("pushed_at"),
                    "language": it.get("language"),
                    "query": query,
                }
        if len(items) < per_page:
            break
        time.sleep(2.0)  # be polite; search API allows 30 req/min authenticated


def semantic_scholar_citations(doi: str, api_key: Optional[str] = None, limit: int = 1000, session: Optional[requests.Session] = None) -> List[Dict[str, object]]:
    """All papers citing ``doi`` (title, year, externalIds, abstract, open-access PDF) with offset pagination."""
    api_key = api_key or os.environ.get("S2_API_KEY")
    sess = session or requests.Session()
    headers = {"x-api-key": api_key} if api_key else {}
    fields = "title,year,externalIds,abstract,openAccessPdf,venue"
    out: List[Dict[str, object]] = []
    offset = 0
    page = 500
    while offset < limit:
        resp = sess.get(f"{S2_API}/paper/DOI:{doi}/citations", params={"fields": fields, "offset": offset, "limit": min(page, limit - offset)}, headers=headers, timeout=60)
        if resp.status_code == 429:
            time.sleep(5.0)
            continue
        resp.raise_for_status()
        data = resp.json().get("data", [])
        for d in data:
            cp = d.get("citingPaper", {})
            out.append(cp)
        if "next" not in resp.json() or not data:
            break
        offset = int(resp.json()["next"])
        time.sleep(1.0)
    return out


def extract_repo_links(text: str) -> List[str]:
    """Unique ``owner/repo`` slugs mentioned in free text (abstract, README, PDF text)."""
    slugs = []
    for m in GITHUB_URL_RE.finditer(text or ""):
        slug = f"{m.group(1)}/{m.group(2)}".rstrip(".")
        if slug.endswith(".git"):
            slug = slug[:-4]
        if slug not in slugs:
            slugs.append(slug)
    return slugs


def clone_shallow(full_name: str, dest_root: str | Path, depth: int = 1, timeout: int = 600) -> Optional[Path]:
    """``git clone --depth 1`` of ``owner/repo`` into ``dest_root/owner__repo``; returns the path or None."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / full_name.replace("/", "__")
    if dest.exists():
        return dest
    url = f"https://github.com/{full_name}.git"
    try:
        subprocess.run(["git", "clone", "--depth", str(depth), "--quiet", url, str(dest)], check=True, timeout=timeout, capture_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return dest
