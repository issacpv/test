#!/usr/bin/env python3
"""Acquire LIDC-IDRI, LUNA16, Duke DLCS and (optionally) NLST for lidc_uq.

Access summary
--------------
LIDC-IDRI     **Open**, no registration. 1018 thoracic CT cases with up to four
              radiologists' annotations including malignancy 1-5. Images via the
              TCIA NBIA REST API or the NBIA Data Retriever; the XML annotations
              come with the images and are read by ``pylidc``.
LUNA16        **Open**. A curated 888-scan subset of LIDC (slice thickness <= 2.5 mm,
              nodules >= 3 mm accepted by >= 3 of 4 readers) with fixed
              cross-validation folds. Hosted on Zenodo.
Duke DLCS     **Open**. 2061 screening LDCT scans, ~3187 semi-automatically
              annotated nodules with 3D bounding boxes; the released subset is
              1613 volumes / 2487 nodules. Zenodo.
NLST          **Application required** via NCI CDAS. Not downloadable by script.

This script automates the metadata/manifest steps and the small open downloads, and
prints exact commands for everything else. It never downloads the full ~125 GB LIDC
image set by default -- use ``--images`` with ``--limit`` and check your disk first.

Examples
--------
    python scripts/download_data.py --dry-run
    python scripts/download_data.py --lidc-manifest --limit 20
    python scripts/download_data.py --lidc-manifest --images --limit 5
    python scripts/download_data.py --configure-pylidc /data/LIDC-IDRI
    python scripts/download_data.py --instructions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests

LOG = logging.getLogger("download_data")

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
LIDC_COLLECTION = "LIDC-IDRI"

INSTRUCTIONS = """
================================================================================
1. LIDC-IDRI (open, no registration) -- the primary dataset
================================================================================
Two routes. The **NBIA Data Retriever is the officially supported one** and is far
more reliable for the full collection than scripted REST calls:

  a) Manifest + NBIA Data Retriever (recommended for the full 125 GB set)
     - Open https://www.cancerimagingarchive.net/collection/lidc-idri/
     - Download the collection's .tcia manifest file
     - Install the NBIA Data Retriever, open the manifest, download to
       data/LIDC-IDRI/

  b) REST API (good for a small sample; this script does it)
     GET {base}/getSeries?Collection=LIDC-IDRI&format=JSON
     GET {base}/getImage?SeriesInstanceUID=<uid>        -> a zip of DICOM files

The radiologist annotations ship as XML alongside the DICOM and are parsed by
pylidc. There is also a DICOM-SR re-encoding of the LIDC annotations on TCIA
("Standardized representation of the LIDC annotations using DICOM") if you prefer
DICOM-native tooling over pylidc.

Configure pylidc afterwards (it needs an absolute path):

    python scripts/download_data.py --configure-pylidc /abs/path/to/LIDC-IDRI

which writes ~/.pylidcrc:

    [dicom]
    path = /abs/path/to/LIDC-IDRI
    warn = True

Then:

    python -c "import pylidc as pl; print(pl.query(pl.Scan).count())"

================================================================================
2. LUNA16 (open) -- for the detection/FP arm and the standard folds
================================================================================
Hosted on Zenodo (record 3723295). Ten subset archives plus annotations and
candidates CSVs:

    pip install zenodo-get
    zenodo_get 3723295 -o data/luna16
    # or fetch individual files:
    #   annotations.csv, candidates_V2.csv, subset0.zip ... subset9.zip

LUNA16 images are .mhd/.raw, not DICOM. Keep LUNA16's official 10-fold split if
you report detection numbers, so they are comparable with the literature.

================================================================================
3. Duke Lung Cancer Screening (DLCS) dataset (open, 2024/2025)
================================================================================
Zenodo DOI 10.5281/zenodo.10782891; dataset paper: "The Duke Lung Cancer Screening
(DLCS) Dataset: A Reference Dataset of Annotated Low-dose Screening Thoracic CT",
Radiology: Artificial Intelligence, doi:10.1148/ryai.240248

    pip install zenodo-get
    zenodo_get 10.5281/zenodo.10782891 -o data/dlcs

Important for this project: DLCS annotations are **3D bounding boxes produced
semi-automatically** (radiologist spot-checked at >90% accuracy), grouped 1-4 by
how much human confirmation they received. There is **no multi-reader malignancy
rating**, so DLCS cannot supply a disagreement target -- it is the *transfer*
cohort, where we test whether LIDC-trained uncertainty still ranks correctly, using
the diagnostic labels as the reference. Record the annotation group per nodule; it
is a confounder for any uncertainty comparison.

================================================================================
4. NLST (application required) -- optional, for screening-scale operating points
================================================================================
NCI Cancer Data Access System (CDAS): https://cdas.cancer.gov/nlst/
  - Submit a data request describing the project
  - Approval typically takes weeks; imaging is a separate request from the
    clinical/outcome tables
  - Reference: Aberle et al. 2011, New England Journal of Medicine, "Reduced
    lung-cancer mortality with low-dose computed tomographic screening"

Do not attempt to script this. Once approved, CDAS provides its own download tool.
""".format(base=TCIA_BASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data"), help="output root (default: data)")
    p.add_argument("--instructions", action="store_true", help="print all acquisition steps and exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan without network access")
    p.add_argument(
        "--lidc-manifest", action="store_true",
        help="query the TCIA API for LIDC-IDRI series metadata and write a manifest",
    )
    p.add_argument(
        "--images", action="store_true",
        help="also download DICOM series zips (use with --limit; the full set is ~125 GB)",
    )
    p.add_argument("--limit", type=int, default=None, help="cap the number of series handled")
    p.add_argument("--sample", action="store_true", help="shorthand for --limit 5 --images")
    p.add_argument(
        "--configure-pylidc", type=Path, default=None, metavar="DICOM_DIR",
        help="write ~/.pylidcrc pointing at this LIDC-IDRI DICOM directory",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def fetch_lidc_series(*, limit: int | None = None, timeout: float = 120.0) -> list[dict[str, Any]]:
    """Fetch LIDC-IDRI series metadata from the TCIA NBIA REST API.

    Args:
        limit: Maximum series to return.
        timeout: Request timeout.

    Returns:
        List of series metadata dicts, each containing at least
        ``SeriesInstanceUID`` and ``PatientID``.

    Raises:
        requests.HTTPError: If the API returns an error status.
        ValueError: If the response is not a JSON list.
    """
    url = f"{TCIA_BASE}/getSeries"
    params = {"Collection": LIDC_COLLECTION, "format": "JSON"}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError(f"unexpected response type {type(payload).__name__} from {url}")
    series = [s for s in payload if isinstance(s, dict)]
    LOG.info("TCIA reports %d series in %s", len(series), LIDC_COLLECTION)
    return series[:limit] if limit is not None else series


def download_series(
    series_uid: str, out_dir: Path, *, timeout: float = 120.0, chunk_bytes: int = 1 << 20
) -> Path:
    """Download one DICOM series as a zip from the TCIA API.

    Args:
        series_uid: ``SeriesInstanceUID``.
        out_dir: Destination directory.
        timeout: Request timeout.
        chunk_bytes: Streaming chunk size.

    Returns:
        Path to the written zip file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{series_uid}.zip"
    if target.exists() and target.stat().st_size > 0:
        LOG.debug("already have %s", target.name)
        return target
    url = f"{TCIA_BASE}/getImage"
    with requests.get(url, params={"SeriesInstanceUID": series_uid}, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        tmp = target.with_suffix(".zip.part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    fh.write(chunk)
        tmp.replace(target)
    return target


def configure_pylidc(dicom_dir: Path) -> Path:
    """Write ``~/.pylidcrc`` pointing at a LIDC-IDRI DICOM directory.

    Args:
        dicom_dir: Absolute path to the extracted LIDC-IDRI collection.

    Returns:
        Path to the written config file.

    Raises:
        FileNotFoundError: If ``dicom_dir`` does not exist.
        ValueError: If ``dicom_dir`` is not absolute -- pylidc silently fails to
            find scans when given a relative path.
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_absolute():
        raise ValueError(f"pylidc requires an absolute path, got {dicom_dir}")
    if not dicom_dir.exists():
        raise FileNotFoundError(dicom_dir)
    config = Path(os.path.expanduser("~")) / ".pylidcrc"
    config.write_text(f"[dicom]\npath = {dicom_dir}\nwarn = True\n", encoding="utf-8")
    LOG.info("wrote %s -> %s", config, dicom_dir)
    return config


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.instructions:
        print(INSTRUCTIONS)
        return 0

    if args.sample:
        args.images = True
        args.limit = args.limit or 5
        args.lidc_manifest = True

    print(INSTRUCTIONS)

    if args.dry_run:
        print(json.dumps({
            "out": str(args.out),
            "lidc_manifest": bool(args.lidc_manifest),
            "images": bool(args.images),
            "limit": args.limit,
            "configure_pylidc": str(args.configure_pylidc) if args.configure_pylidc else None,
        }, indent=2))
        print("\n--dry-run: no network access performed.")
        return 0

    exit_code = 0

    if args.configure_pylidc is not None:
        try:
            configure_pylidc(args.configure_pylidc)
        except (ValueError, FileNotFoundError) as exc:
            LOG.error("could not configure pylidc: %s", exc)
            exit_code = 2

    if args.lidc_manifest:
        try:
            series = fetch_lidc_series(limit=args.limit, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001
            LOG.error(
                "TCIA API request failed (%s). Use the NBIA Data Retriever route "
                "described above instead - it is the supported path for the full "
                "collection.", exc,
            )
            return exit_code or 2

        manifest_path = args.out / "lidc" / "series_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(series, indent=1), encoding="utf-8")
        LOG.info("wrote %s (%d series)", manifest_path, len(series))

        if args.images:
            if args.limit is None:
                LOG.warning(
                    "--images without --limit would fetch the whole ~125 GB collection; "
                    "refusing. Pass --limit N, or use the NBIA Data Retriever."
                )
                return exit_code or 1
            image_dir = args.out / "lidc" / "zips"
            for i, s in enumerate(series, start=1):
                uid = s.get("SeriesInstanceUID")
                if not uid:
                    continue
                try:
                    path = download_series(str(uid), image_dir, timeout=args.timeout)
                    LOG.info("[%d/%d] %s (%.1f MB)", i, len(series), path.name,
                             path.stat().st_size / 1e6)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("series %s failed: %s", uid, exc)
                    exit_code = exit_code or 1

    if not any((args.lidc_manifest, args.configure_pylidc)):
        print("\nNothing to do. Pass --lidc-manifest, --configure-pylidc DIR, "
              "--instructions or --dry-run.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
