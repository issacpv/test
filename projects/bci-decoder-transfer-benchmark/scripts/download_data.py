#!/usr/bin/env python
"""Download open motor-imagery EEG datasets via MOABB and/or PhysioNet EEGMMIDB directly.

Examples
--------
List MOABB motor-imagery datasets (needs ``moabb``)::

    python scripts/download_data.py --list

Smoke test (subject 1 of a few small MOABB datasets + 3 EEGMMIDB runs of subject 1)::

    python scripts/download_data.py --sample

Download selected MOABB datasets fully (resumable)::

    python scripts/download_data.py --moabb BNCI2014_001 PhysionetMI

Download EEGMMIDB runs directly from PhysioNet (open, no credentials)::

    python scripts/download_data.py --physionet --subjects 1 2 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("download_data")

PHYSIONET_BASE = "https://physionet.org/files/eegmmidb/1.0.0"
MI_RUNS = (4, 8, 12)          # imagined left vs right fist
MI_RUNS_FEET = (6, 10, 14)    # imagined both fists vs both feet
SAMPLE_MOABB = ["BNCI2014_001", "BNCI2014_004", "Zhou2016"]


# --------------------------------------------------------------------------- MOABB
def moabb_mi_datasets() -> List[Any]:
    """Instantiate every MOABB dataset whose paradigm is imagery."""
    import moabb.datasets as mds  # type: ignore

    out = []
    for name in dir(mds):
        cls = getattr(mds, name)
        if not isinstance(cls, type) or name.startswith("Base") or name.startswith("_"):
            continue
        try:
            ds = cls()
        except Exception:  # noqa: BLE001 - some classes need arguments or extra deps
            continue
        if getattr(ds, "paradigm", "") == "imagery":
            out.append(ds)
    return out


def describe(ds: Any) -> Dict[str, Any]:
    return {
        "code": getattr(ds, "code", ds.__class__.__name__),
        "class": ds.__class__.__name__,
        "n_subjects": len(getattr(ds, "subject_list", [])),
        "n_sessions": getattr(ds, "n_sessions", None),
        "events": getattr(ds, "event_id", None),
        "interval": getattr(ds, "interval", None),
        "doi": getattr(ds, "doi", None),
    }


def download_moabb(names: List[str], subjects: Optional[List[int]] = None) -> None:
    import moabb.datasets as mds  # type: ignore

    for name in names:
        cls = getattr(mds, name, None)
        if cls is None:
            LOG.error("no MOABB dataset named %s", name)
            continue
        ds = cls()
        subs = subjects or ds.subject_list
        LOG.info("%s: downloading %d subjects", name, len(subs))
        try:
            # download() fetches all files; get_data() also parses them (slower but validates)
            ds.download(subject_list=subs, verbose=False)
        except TypeError:
            ds.download(subject_list=subs)
        except Exception as exc:  # noqa: BLE001
            LOG.error("%s failed: %s", name, exc)


# --------------------------------------------------------------------------- PhysioNet
def download_physionet(out_dir: Path, subjects: List[int], runs: List[int]) -> int:
    """Fetch ``S{sub:03d}R{run:02d}.edf`` (+ ``.event``) from PhysioNet; skips existing files."""
    n = 0
    for sub in subjects:
        sdir = out_dir / f"S{sub:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            for ext in (".edf", ".edf.event"):
                fname = f"S{sub:03d}R{run:02d}{ext}"
                target = sdir / fname
                if target.exists() and target.stat().st_size > 0:
                    continue
                url = f"{PHYSIONET_BASE}/S{sub:03d}/{fname}"
                try:
                    with requests.get(url, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        tmp = target.with_suffix(target.suffix + ".part")
                        with tmp.open("wb") as fh:
                            for chunk in r.iter_content(1 << 20):
                                fh.write(chunk)
                        tmp.replace(target)
                    n += 1
                    LOG.info("downloaded %s", target)
                except requests.RequestException as exc:
                    LOG.warning("failed %s: %s", url, exc)
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--list", action="store_true", help="list MOABB motor-imagery datasets")
    ap.add_argument("--moabb", nargs="*", default=None, help="MOABB dataset class names, or 'all'")
    ap.add_argument("--physionet", action="store_true", help="download EEGMMIDB runs directly from PhysioNet")
    ap.add_argument("--subjects", nargs="*", type=int, default=None)
    ap.add_argument("--feet-runs", action="store_true", help="also fetch fists-vs-feet runs (6, 10, 14)")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(args.out / "mne_data"))
    (args.out / "mne_data").mkdir(exist_ok=True)

    have_moabb = True
    try:
        import moabb  # type: ignore  # noqa: F401
    except ImportError:
        have_moabb = False

    if args.list:
        if not have_moabb:
            LOG.error("moabb is not installed: pip install moabb")
            return 1
        rows = [describe(ds) for ds in moabb_mi_datasets()]
        (args.out / "datasets.json").write_text(json.dumps(rows, indent=2, default=str))
        for r in rows:
            print(f"{r['class']:22s} subjects={r['n_subjects']:4d} sessions={r['n_sessions']} events={r['events']}")
        return 0

    if args.sample:
        if have_moabb:
            download_moabb(SAMPLE_MOABB, subjects=[1])
        else:
            LOG.warning("moabb not installed; skipping MOABB sample")
        n = download_physionet(args.out / "eegmmidb", args.subjects or [1], list(MI_RUNS))
        LOG.info("PhysioNet sample: %d files", n)
        return 0

    if args.moabb is not None:
        if not have_moabb:
            LOG.error("moabb is not installed: pip install moabb")
            return 1
        names = args.moabb
        if names == ["all"] or not names:
            names = [ds.__class__.__name__ for ds in moabb_mi_datasets()]
        download_moabb(names, subjects=args.subjects)

    if args.physionet:
        runs = list(MI_RUNS) + (list(MI_RUNS_FEET) if args.feet_runs else [])
        subs = args.subjects or list(range(1, 110))
        n = download_physionet(args.out / "eegmmidb", subs, runs)
        LOG.info("PhysioNet: %d files downloaded", n)

    if not (args.moabb is not None or args.physionet):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
