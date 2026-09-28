#!/usr/bin/env python3
"""Fetch the open inputs for imgtx_nulls.

Usage
-----
    python scripts/download_data.py ahba [--sample]                 # AHBA microarray via abagen
    python scripts/download_data.py expression --atlas glasser|schaefer400 [--n-rois 400]
    python scripts/download_data.py geometry --atlas glasser|schaefer400
    python scripts/download_data.py maps                            # neuromaps annotations -> CSV
    python scripts/download_data.py genesets [--sample]             # GO obo + NCBI gene2go
    python scripts/download_data.py hcp-aging                       # prints NDA instructions

Nothing here needs credentials except HCP-Aging (NDA) and the Glasser
parcellation from BALSA (free registration); both are documented, not bypassed.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"

GO_OBO = "https://purl.obolibrary.org/obo/go/go-basic.obo"
GENE2GO = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"

MAPS = [
    ("hcps1200", "myelinmap", "fsLR", "32k"),
    ("hcps1200", "thickness", "fsLR", "32k"),
    ("margulies2016", "fcgradient01", "fsLR", "32k"),
    ("hcps1200", "megalpha", "fsLR", "4k"),
    ("hcps1200", "megbeta", "fsLR", "4k"),
    ("hcps1200", "megtheta", "fsLR", "4k"),
]


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"exists: {dest}")
        return
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for part in r.iter_content(chunk):
                fh.write(part)
    print(f"downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def cmd_ahba(sample: bool) -> None:
    try:
        import abagen
    except ImportError:
        sys.exit("pip install abagen")
    donors = ["9861"] if sample else "all"
    files = abagen.fetch_microarray(donors=donors, data_dir=str(DATA / "abagen"), verbose=1)
    print(f"fetched microarray for donors: {list(files)}")


def _parcellation(atlas: str, n_rois: int):
    """Return (atlas, atlas_info) suitable for abagen.get_expression_data."""
    parc_dir = DATA / "parcellations"
    if atlas == "glasser":
        lh = parc_dir / "Glasser_MMP1.0.L.32k_fs_LR.label.gii"
        rh = parc_dir / "Glasser_MMP1.0.R.32k_fs_LR.label.gii"
        info = parc_dir / "Glasser_MMP1.0_atlas_info.csv"
        if not (lh.exists() and rh.exists() and info.exists()):
            sys.exit("Glasser label GIFTIs + atlas_info CSV not found; see data/README.md section 2 (BALSA)")
        return (str(lh), str(rh)), str(info)
    if atlas == "schaefer400":
        try:
            from netneurotools import datasets as nntdata
        except ImportError:
            sys.exit("pip install netneurotools")
        sch = nntdata.fetch_schaefer2018("fslr32k", data_dir=str(parc_dir))[f"{n_rois}Parcels7Networks"]
        return (sch.lh, sch.rh), None
    raise ValueError(atlas)


def cmd_expression(atlas: str, n_rois: int) -> None:
    from imgtx_nulls.expression import fetch_parcellated_expression
    parc, info = _parcellation(atlas, n_rois)
    out = DATA / "expression"
    out.mkdir(parents=True, exist_ok=True)
    expr = fetch_parcellated_expression(parc, atlas_info=info, data_dir=str(DATA / "abagen"),
                                        cache=out / f"{atlas}_expression.parquet")
    print(f"expression table: {expr.shape} -> {out}")
    donors = fetch_parcellated_expression(parc, atlas_info=info, data_dir=str(DATA / "abagen"), return_donors=True)
    ddir = out / f"{atlas}_donors"
    ddir.mkdir(exist_ok=True)
    for d, df in donors.items():
        df.to_parquet(ddir / f"{d}.parquet")
    print(f"per-donor tables -> {ddir}")


def cmd_geometry(atlas: str, n_rois: int) -> None:
    from imgtx_nulls.maps import parcel_centroids_from_gifti, geodesic_distance_matrix
    import numpy as np
    parc, _ = _parcellation(atlas, n_rois)
    parc_dir = DATA / "parcellations"
    spheres = [parc_dir / "S1200.L.sphere.32k_fs_LR.surf.gii", parc_dir / "S1200.R.sphere.32k_fs_LR.surf.gii"]
    if not all(s.exists() for s in spheres):
        try:
            from neuromaps.datasets import fetch_fslr
            fs = fetch_fslr(density="32k", data_dir=str(parc_dir))
            spheres = [Path(fs["sphere"][0]), Path(fs["sphere"][1])]
        except ImportError:
            sys.exit("Need sphere surfaces: pip install neuromaps or place S1200 sphere GIFTIs in data/parcellations")
    cents = [parcel_centroids_from_gifti(s, p) for s, p in zip(spheres, parc)]
    import pandas as pd
    lh, rh = cents
    lh["hemi"], rh["hemi"] = "L", "R"
    cen = pd.concat([lh, rh])
    cen.to_csv(parc_dir / f"centroids_{atlas}.csv")
    D = geodesic_distance_matrix(cen[["x", "y", "z"]].to_numpy(), radius=100.0)
    np.save(parc_dir / f"greatcircle_{atlas}.npy", D)
    print(f"{len(cen)} centroids -> {parc_dir}. For exact surface geodesics use brainsmash.workbench.geo.cortex.")


def cmd_maps() -> None:
    try:
        from neuromaps import datasets
    except ImportError:
        sys.exit("pip install neuromaps")
    out = DATA / "maps" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for src, desc, space, den in MAPS:
        try:
            files = datasets.fetch_annotation(source=src, desc=desc, space=space, den=den, data_dir=str(out))
            print(f"{src}/{desc}: {files}")
        except Exception as e:  # noqa: BLE001
            print(f"{src}/{desc}: {e}", file=sys.stderr)
    print("Parcellate with imgtx_nulls.maps.fetch_neuromaps_annotation(..., parcellation=(lh, rh)) once a parcellation is in place.")


def cmd_genesets(sample: bool) -> None:
    out = DATA / "genesets"
    _download(GO_OBO, out / "go-basic.obo")
    if sample:
        print("--sample: skipping gene2go (≈ 100 MB)")
        return
    _download(GENE2GO, out / "gene2go.gz")
    human = out / "gene2go_human.tsv"
    if not human.exists():
        with gzip.open(out / "gene2go.gz", "rt") as fin, open(human, "w") as fout:
            for line in fin:
                if line.startswith("#") or line.startswith("9606\t"):
                    fout.write(line)
        print(f"human subset -> {human}")


def cmd_hcp_aging() -> None:
    print(
        "HCP-Aging requires an NDA Data Use Certification (collection 2847).\n"
        "  pip install nda-tools\n"
        "  downloadcmd -dp <PACKAGE_ID> -d data/hcp_aging -u $NDA_USERNAME -p $NDA_PASSWORD\n"
        "Select the MSMAll thickness / MyelinMap dscalar (or pscalar) files, then compute parcel-wise age\n"
        "effects and store only the group-level maps under data/maps/. See data/README.md section 4."
    )


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["ahba", "expression", "geometry", "maps", "genesets", "hcp-aging"])
    p.add_argument("--sample", action="store_true")
    p.add_argument("--atlas", default="glasser", choices=["glasser", "schaefer400"])
    p.add_argument("--n-rois", type=int, default=400)
    a = p.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    {"ahba": lambda: cmd_ahba(a.sample), "expression": lambda: cmd_expression(a.atlas, a.n_rois),
     "geometry": lambda: cmd_geometry(a.atlas, a.n_rois), "maps": cmd_maps,
     "genesets": lambda: cmd_genesets(a.sample), "hcp-aging": cmd_hcp_aging}[a.what]()


if __name__ == "__main__":
    main()
