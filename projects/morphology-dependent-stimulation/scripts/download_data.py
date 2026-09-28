#!/usr/bin/env python3
"""Acquire the open morphology and head-model inputs for morph_stim.

Sources
-------
NeuroMorpho.Org   REST API, fully open, no registration. Human and mouse cortical
                  pyramidal / interneuron reconstructions. This is the only part
                  the script downloads in bulk.
Allen Cell Types  Open, no registration. SWC reconstructions and biophysical
                  model packages are fetched through ``allensdk`` if installed;
                  otherwise the script prints the exact commands.
SimNIBS examples  The ``ernie`` example head model ships with SimNIBS itself
                  (``simnibs`` >= 4). We only print instructions, because the
                  download is a multi-GB installer-managed asset.
MIDA              Free registration with the IT'IS Foundation. Manual download.

Examples
--------
Smoke test without touching the network::

    python scripts/download_data.py --dry-run

Small sample of each cohort (fast, ~40 files)::

    python scripts/download_data.py --sample

Full open cohorts::

    python scripts/download_data.py --all --out data

Only the human cohort, capped::

    python scripts/download_data.py --cohort human --limit 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from morph_stim.neuromorpho import (  # noqa: E402
    HUMAN_CORTICAL_QUERY,
    MOUSE_CORTICAL_QUERY,
    NeuroMorphoClient,
    NeuroMorphoQuery,
)

LOG = logging.getLogger("download_data")

#: Diseased / aged arm. ``experiment_condition`` is sparsely populated, so this
#: cohort is expected to be small and is reported as exploratory.
DISEASE_QUERY = NeuroMorphoQuery(
    species=("human",),
    brain_region=("neocortex",),
    extra=(("experiment_condition", ("alzheimer's disease", "epilepsy", "aging")),),
)

COHORTS: dict[str, NeuroMorphoQuery] = {
    "human": HUMAN_CORTICAL_QUERY,
    "mouse": MOUSE_CORTICAL_QUERY,
    "disease": DISEASE_QUERY,
}

ALLEN_HINT = """
Allen Cell Types Database (open, no registration)
-------------------------------------------------
Morphologies + biophysical models, human and mouse:

    pip install allensdk
    python - <<'PY'
    from allensdk.core.cell_types_cache import CellTypesCache
    from allensdk.api.queries.biophysical_api import BiophysicalApi

    ctc = CellTypesCache(manifest_file='data/allen/manifest.json')
    cells = ctc.get_cells(species=['Homo Sapiens'])          # or ['Mus musculus']
    for cell in cells:
        ctc.get_reconstruction(cell['id'])                   # SWC
        ctc.get_ephys_features()

    # One biophysical (all-active) model package, ready to run under NEURON:
    BiophysicalApi().cache_data(<neuronal_model_id>, working_directory='data/allen/model')
    PY

Model ids are listed at https://celltypes.brain-map.org (filter
Models -> 'Biophysical - all active').
"""

SIMNIBS_HINT = """
Head models (only needed for the field-orientation priors, not for the sweep)
----------------------------------------------------------------------------
SimNIBS >= 4 example dataset, including subject 'ernie' and the MNI152 head:

    pip install simnibs                     # or the platform installer
    python -m simnibs.examples.download     # see 'Datasets' in the SimNIBS docs
    # -> simnibs_examples/ernie/m2m_ernie/ernie.msh

MIDA (multimodal imaging-based detailed anatomical model) requires free
registration with the IT'IS Foundation; download the NIfTI label volume manually
into data/head_models/mida/ .
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"), help="output root (default: data)")
    p.add_argument(
        "--cohort",
        action="append",
        choices=sorted(COHORTS),
        help="cohort to fetch; repeatable. Default: human and mouse.",
    )
    p.add_argument("--all", action="store_true", help="fetch every cohort including 'disease'")
    p.add_argument("--sample", action="store_true", help="cap each cohort at --sample-size records")
    p.add_argument("--sample-size", type=int, default=15, help="records per cohort in --sample mode")
    p.add_argument("--limit", type=int, default=None, help="hard cap on records per cohort")
    p.add_argument("--metadata-only", action="store_true", help="search but do not download SWC files")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit without network access")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between HTTP requests (be polite)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cohorts = sorted(COHORTS) if args.all else (args.cohort or ["human", "mouse"])
    limit = args.limit if args.limit is not None else (args.sample_size if args.sample else None)

    print(ALLEN_HINT)
    print(SIMNIBS_HINT)

    plan = {c: {"clauses": COHORTS[c].clauses(), "limit": limit} for c in cohorts}
    print("NeuroMorpho plan:\n" + json.dumps(plan, indent=2))
    if args.dry_run:
        print("\n--dry-run: no network access performed.")
        return 0

    client = NeuroMorphoClient(cache_dir=args.out / "neuromorpho", delay_s=args.delay)
    try:
        health = client.health()
        LOG.info("NeuroMorpho health: %s", health)
    except Exception as exc:  # noqa: BLE001
        LOG.error("cannot reach NeuroMorpho.Org (%s). Check network/proxy and retry.", exc)
        return 2

    manifest: dict[str, object] = {}
    failures = 0
    for cohort in cohorts:
        query = COHORTS[cohort]
        LOG.info("cohort %s: %s", cohort, query.clauses())
        try:
            records = client.search(query, limit=limit)
        except Exception as exc:  # noqa: BLE001
            LOG.error("search failed for %s: %s", cohort, exc)
            failures += 1
            continue
        LOG.info("cohort %s: %d records", cohort, len(records))

        downloaded = 0
        if not args.metadata_only:
            for rec in records:
                try:
                    client.fetch_swc(rec)
                    downloaded += 1
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("SWC download failed for %s: %s", rec.get("neuron_name"), exc)
        manifest[cohort] = {
            "n_records": len(records),
            "n_swc_downloaded": downloaded,
            "clauses": query.clauses(),
        }

    out_manifest = args.out / "neuromorpho" / "manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("wrote %s", out_manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
