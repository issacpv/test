#!/usr/bin/env python3
"""Acquire the connectome, E-field and stimulation-dataset inputs for tms_target.

What this script does automatically
-----------------------------------
1. **Searches OpenNeuro** for candidate TMS / tDCS datasets via the public GraphQL
   API and writes a candidate table to ``data/openneuro/candidates.json``. This is
   the only fully automatic, credential-free step, and it is the one worth
   automating: the OpenNeuro catalogue changes monthly, so a hard-coded dataset
   list goes stale. **Every hit must still be screened by hand** -- a keyword match
   on "TMS" catches datasets that merely mention it in a task name.
2. **Downloads** selected OpenNeuro datasets through ``openneuro-py`` or
   ``datalad`` if either is installed, restricted to the modalities we need.

What requires credentials or a manual step
------------------------------------------
* **HCP S1200** -- free registration and acceptance of the Open Access Data Use
  Terms at https://db.humanconnectome.org . Group-average products (parcellated FC,
  the MSM-All dense connectome) are downloaded either from ConnectomeDB or from the
  ``hcp-openaccess`` S3 bucket with the AWS keys ConnectomeDB issues. The script
  reads ``HCP_AWS_ACCESS_KEY_ID`` / ``HCP_AWS_SECRET_ACCESS_KEY`` from the
  environment and prints ready-to-run ``aws s3 cp`` commands; it never stores keys.
* **SimNIBS / ROAST** -- installed software, not downloads.

Examples
--------
    python scripts/download_data.py --dry-run
    python scripts/download_data.py --search-openneuro --query TMS --query tDCS
    python scripts/download_data.py --download-openneuro ds004024 --modality eeg
    python scripts/download_data.py --hcp-instructions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

LOG = logging.getLogger("download_data")

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
HCP_S3_BUCKET = "s3://hcp-openaccess/HCP_1200"

#: Datasets that are known-relevant starting points. Verify each against the live
#: OpenNeuro record before use -- versions and file counts change.
SEED_DATASETS: dict[str, str] = {
    "ds004024": (
        "TMS-EEG-MRI-fMRI-DWI, paired associative stimulation and connectivity "
        "(Shirley Ryan AbilityLab). Tasks: ccPAS, rest, spTMS. 13 participants, "
        "BIDS 1.6.0, ~1 TB - use --modality to restrict the download."
    ),
    "ds005498": "Single-pulse TMS-fMRI (concurrent). Check the latest snapshot tag.",
}

DATASET_QUERY = """
query($query: JSON!, $first: Int!, $after: String) {
  datasets: advancedSearch(query: $query, first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        created
        public
        latestSnapshot {
          tag
          size
          description { Name Authors }
          summary { modalities subjects tasks sessions }
        }
      }
    }
  }
}
"""

HCP_INSTRUCTIONS = f"""
================================================================================
HCP S1200 -- free registration, Open Access Data Use Terms
================================================================================
1. Register at https://db.humanconnectome.org and accept the WU-Minn HCP Open
   Access Data Use Terms. Then, in ConnectomeDB, generate AWS credentials
   ("Account Settings" -> "Amazon S3 credentials").

2. Export them (this script reads them but never writes them anywhere):

       export HCP_AWS_ACCESS_KEY_ID=...
       export HCP_AWS_SECRET_ACCESS_KEY=...

3. What to fetch. The group-average products are what a *normative* connectome
   analysis needs, and they are small compared to per-subject data:

   a) S1200 group-average dense functional connectome (MSM-All registered,
      n=1003 or the n=812 r227 subset). Tens of GB; read it row-wise only.

       aws s3 cp --recursive \\
         {HCP_S3_BUCKET}/../HCP_Resources/GroupAvg/HCP_S1200_GroupAvg_v1/ \\
         data/hcp/group_avg/ --exclude "*" --include "*dconn*"

      The S1200 Group Average release is also downloadable directly from
      ConnectomeDB, which is usually easier than guessing S3 prefixes. Start at
      https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release

   b) Per-subject rfMRI dense/parcellated timeseries for the *individualized*
      connectome arm (this is the large download; budget ~1 TB for 100 subjects,
      or use the ICA-FIX parcellated timeseries which are far smaller):

       aws s3 ls {HCP_S3_BUCKET}/
       aws s3 cp {HCP_S3_BUCKET}/<subject>/MNINonLinear/Results/ \\
         data/hcp/subjects/<subject>/ --recursive \\
         --exclude "*" --include "*Atlas_MSMAll_hp2000_clean.ptseries.nii"

   c) Diffusion / tractography for structural connectivity. HCP ships preprocessed
      diffusion; the streamline matrices must be generated yourself (MRtrix3 or
      DSI Studio) or obtained from a published derivative.

4. Parcellation. Use HCP-MMP1 (Glasser et al. 2016, Nature) or a Schaefer
   parcellation, and keep parcel centroid coordinates -- the spatial nulls in
   tms_target.nulls require them.

NOTE ON SCOPE: the group-average arm is tractable on a laptop. The per-subject
individualized arm is a serious data-engineering commitment; consider starting
with the ~100 subject "HCP100" unrelated subset.
"""

EFIELD_INSTRUCTIONS = """
================================================================================
E-field modelling tools (software installs, not data downloads)
================================================================================
SimNIBS >= 4 (FEM, includes the 'ernie' example head model):
    pip install simnibs          # or the platform installer (the supported route)
    charm <subject_id> <T1.nii.gz> [<T2.nii.gz>]      # build m2m_<subject>/
    simnibs_python run_simulation.py                  # writes .msh with normE

    Load the result with tms_target.efield.load_efield_msh(path, tissue_tags=(2,))
    (tag 2 is grey matter in the SimNIBS convention).

ROAST (tES, MATLAB/Octave, writes NIfTI):
    https://www.parralab.org/roast/
    Load with tms_target.efield.load_efield_nifti(path, mask_path=gm_mask).

For the HCP-derived head models, run charm on the HCP T1w/T2w images of the same
subjects used for the individualized connectomes, so field and connectivity share
a subject and a coordinate space.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data"), help="output root (default: data)")
    p.add_argument("--search-openneuro", action="store_true", help="query the OpenNeuro GraphQL API")
    p.add_argument(
        "--query", action="append", default=None,
        help="keyword to search on OpenNeuro; repeatable. Default: TMS, tDCS, tACS.",
    )
    p.add_argument("--first", type=int, default=25, help="results per page (default 25)")
    p.add_argument("--max-results", type=int, default=100, help="stop after this many hits per query")
    p.add_argument(
        "--download-openneuro", action="append", default=None, metavar="DSID",
        help="OpenNeuro accession number to download (e.g. ds004024); repeatable",
    )
    p.add_argument(
        "--modality", action="append", default=None,
        help="restrict an OpenNeuro download to these BIDS modality dirs (eeg, func, anat, dwi)",
    )
    p.add_argument("--sample", action="store_true", help="download only the first 2 subjects of a dataset")
    p.add_argument("--hcp-instructions", action="store_true", help="print HCP acquisition steps and exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan without network access")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def search_openneuro(
    keyword: str, *, first: int = 25, max_results: int = 100, timeout: float = 60.0
) -> list[dict[str, Any]]:
    """Search OpenNeuro for datasets matching a keyword, following pagination.

    Args:
        keyword: Free-text query passed to OpenNeuro's advanced search.
        first: Page size.
        max_results: Stop after this many results.
        timeout: Per-request timeout.

    Returns:
        A list of dataset nodes, each with id, snapshot tag and summary.

    Raises:
        requests.HTTPError: If the API returns an error status.
        RuntimeError: If the response contains GraphQL errors.
    """
    query_body = {"query": {"query_string": {"query": keyword}}}
    out: list[dict[str, Any]] = []
    after: str | None = None
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "User-Agent": "tms_target/0.1"})

    while len(out) < max_results:
        payload = {
            "query": DATASET_QUERY,
            "variables": {"query": query_body["query"], "first": first, "after": after},
        }
        resp = session.post(OPENNEURO_GRAPHQL, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            raise RuntimeError(f"OpenNeuro GraphQL errors: {body['errors']}")
        block = (body.get("data") or {}).get("datasets") or {}
        edges = block.get("edges") or []
        if not edges:
            break
        for edge in edges:
            node = edge.get("node")
            if node:
                out.append(node)
        info = block.get("pageInfo") or {}
        if not info.get("hasNextPage"):
            break
        after = info.get("endCursor")
    return out[:max_results]


def summarize_hit(node: dict[str, Any]) -> dict[str, Any]:
    """Flatten an OpenNeuro dataset node into a screening row."""
    snap = node.get("latestSnapshot") or {}
    desc = snap.get("description") or {}
    summary = snap.get("summary") or {}
    subjects = summary.get("subjects") or []
    return {
        "id": node.get("id"),
        "name": desc.get("Name"),
        "tag": snap.get("tag"),
        "size_gb": round((snap.get("size") or 0) / 1e9, 2),
        "modalities": summary.get("modalities"),
        "tasks": summary.get("tasks"),
        "n_subjects": len(subjects) if isinstance(subjects, list) else None,
        "url": f"https://openneuro.org/datasets/{node.get('id')}",
    }


def download_openneuro(
    dataset_id: str, out_dir: Path, *, modalities: list[str] | None = None, sample: bool = False
) -> int:
    """Download an OpenNeuro dataset with ``openneuro-py``, falling back to ``datalad``.

    Args:
        dataset_id: Accession number, e.g. ``ds004024``.
        out_dir: Destination directory.
        modalities: BIDS modality directories to include, e.g. ``["eeg", "func"]``.
        sample: Restrict to the first two subjects.

    Returns:
        A process exit code (0 on success).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / dataset_id

    if shutil.which("openneuro-py") or _module_available("openneuro"):
        cmd = [sys.executable, "-m", "openneuro", "download", f"--dataset={dataset_id}",
               f"--target-dir={target}"]
        if modalities:
            for m in modalities:
                cmd.append(f"--include=*/{m}/*")
        if sample:
            cmd += ["--include=sub-01/*", "--include=sub-02/*",
                    "--include=dataset_description.json", "--include=participants.tsv"]
        LOG.info("running: %s", " ".join(cmd))
        return subprocess.call(cmd)

    if shutil.which("datalad"):
        LOG.info("openneuro-py not found; using datalad")
        rc = subprocess.call(
            ["datalad", "clone", f"https://github.com/OpenNeuroDatasets/{dataset_id}.git", str(target)]
        )
        if rc != 0:
            return rc
        patterns = [f"*/{m}/*" for m in (modalities or [])] or ["."]
        for pattern in patterns:
            rc = rc or subprocess.call(["datalad", "get", "-d", str(target), pattern])
        return rc

    LOG.error(
        "neither openneuro-py nor datalad is installed. Install one:\n"
        "    pip install openneuro-py\n"
        "  or\n"
        "    pip install datalad && datalad clone "
        f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
    )
    return 3


def _module_available(name: str) -> bool:
    """Return True if a module can be imported."""
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def check_hcp_credentials() -> bool:
    """Report whether HCP S3 credentials are present in the environment."""
    have = bool(os.environ.get("HCP_AWS_ACCESS_KEY_ID")) and bool(
        os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    )
    if have:
        LOG.info("HCP AWS credentials found in the environment")
    else:
        LOG.warning(
            "HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY not set; "
            "group-average downloads will need them (or use ConnectomeDB directly)"
        )
    return have


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.hcp_instructions:
        print(HCP_INSTRUCTIONS)
        print(EFIELD_INSTRUCTIONS)
        return 0

    queries = args.query or ["TMS", "tDCS", "tACS"]
    print("Seed datasets (verify each against the live OpenNeuro record):")
    for dsid, note in SEED_DATASETS.items():
        print(f"  {dsid}: {note}")
    print(HCP_INSTRUCTIONS)
    print(EFIELD_INSTRUCTIONS)

    if args.dry_run:
        print(json.dumps({"openneuro_queries": queries,
                          "would_download": args.download_openneuro or [],
                          "out": str(args.out)}, indent=2))
        print("\n--dry-run: no network access performed.")
        return 0

    check_hcp_credentials()
    exit_code = 0

    if args.search_openneuro:
        results: dict[str, list[dict[str, Any]]] = {}
        for keyword in queries:
            try:
                hits = search_openneuro(
                    keyword, first=args.first, max_results=args.max_results, timeout=args.timeout
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error("OpenNeuro search for %r failed: %s", keyword, exc)
                exit_code = 1
                continue
            rows = [summarize_hit(h) for h in hits]
            results[keyword] = rows
            LOG.info("query %r: %d datasets", keyword, len(rows))
            for row in rows[:10]:
                print(f"  {row['id']:<12} {row['size_gb']:>8} GB  {str(row['modalities']):<28} {row['name']}")

        dest = args.out / "openneuro" / "candidates.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
        LOG.info("wrote %s -- SCREEN THESE BY HAND before downloading", dest)

    for dsid in args.download_openneuro or []:
        rc = download_openneuro(
            dsid, args.out / "openneuro", modalities=args.modality, sample=args.sample
        )
        if rc != 0:
            LOG.error("download of %s exited %d", dsid, rc)
            exit_code = exit_code or rc

    if not args.search_openneuro and not args.download_openneuro:
        print("\nNothing to do. Pass --search-openneuro, --download-openneuro DSID, "
              "--hcp-instructions or --dry-run.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
