#!/usr/bin/env python
"""Fetch atlases, maps and the claims-corpus candidates for the spatial-null audit.

Examples
--------
    python scripts/download_data.py --parcellations --out data/atlases
    python scripts/download_data.py --neuromaps --sample --out data/maps
    python scripts/download_data.py --neurovault data/claims/claims.csv --out data/maps/neurovault
    python scripts/download_data.py --pubmed-candidates --out data/claims

Everything is open; no credentials are needed. HCP and ENIGMA inputs are described in data/README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_null_audit.claims import CLAIM_COLUMNS  # noqa: E402

CBIG_RAW = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
    "Schaefer2018_LocalGlobal/Parcellations"
)
SCHAEFER_RES = (100, 200, 400, 1000)
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NEUROVAULT = "https://neurovault.org/api"

PUBMED_QUERY = (
    '("spin test"[tiab] OR "spatial permutation"[tiab] OR "BrainSMASH"[tiab] OR '
    '"spatial autocorrelation"[tiab] OR "spin permutation"[tiab]) AND '
    '("brain map"[tiab] OR "cortical map"[tiab] OR gradient[tiab] OR receptor[tiab] OR "cortical thickness"[tiab]) '
    'AND ("2018"[dp] : "2026"[dp])'
)


def _get(url: str, dest: Path, timeout: float = 120) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                print(f"  {url}: HTTP {r.status_code}")
                return False
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {url}: {exc}")
        return False


def cmd_parcellations(out: Path, sample: bool) -> None:
    res_list = SCHAEFER_RES[:1] if sample else SCHAEFER_RES
    for n in res_list:
        base = f"Schaefer2018_{n}Parcels_7Networks_order"
        ok = _get(f"{CBIG_RAW}/MNI/Centroid_coordinates/{base}_FSLMNI152_1mm.Centroid_RAS.csv", out / f"{base}_FSLMNI152_1mm.Centroid_RAS.csv")
        print(f"centroids {n}: {'ok' if ok else 'failed'}")
        for hemi in ("lh", "rh"):
            ok = _get(f"{CBIG_RAW}/FreeSurfer5.3/fsaverage/label/{hemi}.{base}.annot", out / f"{hemi}.{base}.annot")
            print(f"annot {hemi} {n}: {'ok' if ok else 'failed'}")
    try:
        from neuromaps.datasets import fetch_atlas  # type: ignore

        for atlas, density in (("fsaverage", "10k"), ("fsLR", "32k")):
            paths = fetch_atlas(atlas, density, data_dir=str(out / "spheres"))
            print(f"neuromaps {atlas} {density}: {paths['sphere']}")
    except ImportError:
        print("neuromaps not installed: sphere surfaces skipped (MNI-centroid projection will be used; see data/README.md)")


def cmd_neuromaps(out: Path, sample: bool) -> None:
    try:
        from neuromaps.datasets import available_annotations, fetch_annotation  # type: ignore
    except ImportError:
        print("pip install neuromaps first")
        return
    annots = available_annotations()
    print(f"{len(annots)} annotations available")
    pd.DataFrame(annots, columns=["source", "desc", "space", "den"]).to_csv(out / "neuromaps_annotations.csv", index=False)
    for i, (source, desc, space, den) in enumerate(annots):
        if sample and i >= 5:
            break
        try:
            fetch_annotation(source=source, desc=desc, space=space, den=den, data_dir=str(out / "neuromaps"))
            print(f"  fetched {source}/{desc}/{space}/{den}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {source}/{desc}: {exc}")


def cmd_neurovault(claims_csv: Path, out: Path) -> None:
    claims = pd.read_csv(claims_csv)
    ids = set()
    for col in ("map_a", "map_b"):
        for v in claims[col].dropna().astype(str):
            if v.startswith("neurovault:"):
                ids.add(v.split(":", 1)[1])
    print(f"{len(ids)} NeuroVault images referenced")
    for img_id in sorted(ids):
        try:
            meta = requests.get(f"{NEUROVAULT}/images/{img_id}/", params={"format": "json"}, timeout=60).json()
            ok = _get(meta["file"], out / f"{img_id}.nii.gz")
            print(f"  image {img_id}: {'ok' if ok else 'failed'} ({meta.get('name')})")
        except Exception as exc:  # noqa: BLE001
            print(f"  image {img_id}: {exc}")
        time.sleep(0.2)


def cmd_pubmed_candidates(out: Path, sample: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("NCBI_API_KEY")
    params = {"db": "pubmed", "term": PUBMED_QUERY, "retmode": "json", "retmax": 50 if sample else 5000}
    if key:
        params["api_key"] = key
    r = requests.get(f"{NCBI}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    pmids = r.json()["esearchresult"]["idlist"]
    print(f"{len(pmids)} candidate PMIDs")
    rows = []
    for i in range(0, len(pmids), 100):
        chunk = pmids[i : i + 100]
        rs = requests.get(f"{NCBI}/esummary.fcgi", params={"db": "pubmed", "id": ",".join(chunk), "retmode": "json", **({"api_key": key} if key else {})}, timeout=60)
        rs.raise_for_status()
        res = rs.json().get("result", {})
        for pmid in chunk:
            item = res.get(pmid, {})
            doi = next((a["value"] for a in item.get("articleids", []) if a.get("idtype") == "doi"), "")
            rows.append({"pmid": pmid, "doi": doi, "year": (item.get("pubdate") or "")[:4], "title": item.get("title"), "journal": item.get("fulljournalname")})
        time.sleep(0.11 if key else 0.35)
    pd.DataFrame(rows).to_csv(out / "pubmed_candidates.csv", index=False)
    template = out / "claims.csv"
    if not template.exists():
        pd.DataFrame(columns=CLAIM_COLUMNS).to_csv(template, index=False)
        print(f"wrote empty claims template -> {template}")
    print(f"wrote {len(rows)} candidates -> {out / 'pubmed_candidates.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--parcellations", action="store_true")
    p.add_argument("--neuromaps", action="store_true")
    p.add_argument("--neurovault", metavar="CLAIMS_CSV")
    p.add_argument("--pubmed-candidates", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.parcellations:
        cmd_parcellations(out, args.sample)
    if args.neuromaps:
        cmd_neuromaps(out, args.sample)
    if args.neurovault:
        cmd_neurovault(Path(args.neurovault), out)
    if args.pubmed_candidates:
        cmd_pubmed_candidates(out, args.sample)
    if not any([args.parcellations, args.neuromaps, args.neurovault, args.pubmed_candidates]):
        p.print_help()


if __name__ == "__main__":
    main()
