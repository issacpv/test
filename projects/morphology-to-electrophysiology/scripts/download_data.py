#!/usr/bin/env python
"""Download Allen Cell Types data (and optionally NeuroMorpho morphologies) for morph2ephys.

Examples
--------
    python scripts/download_data.py --sample                       # 10 mouse + 5 human cells
    python scripts/download_data.py --species mouse human          # all reconstructed cells
    python scripts/download_data.py --species mouse human --nwb    # + NWB sweep files (large)
    python scripts/download_data.py --neuromorpho --species mouse human --nm-max-pages 4

Requires ``allensdk`` for the Allen part. The NeuroMorpho part only needs ``requests``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morph2ephys.allen_fetch import HAVE_ALLENSDK, AllenCellTypesFetcher, fetch_neuromorpho_swcs  # noqa: E402

LOG = logging.getLogger("download_data")


def download_allen(out_dir: Path, species: List[str], sample: bool, nwb: bool) -> int:
    if not HAVE_ALLENSDK:
        LOG.error("allensdk is not installed: pip install allensdk (see data/README.md)")
        return 1
    fetcher = AllenCellTypesFetcher(out_dir / "allen")
    table = fetcher.build_table(species=species, require_reconstruction=True)
    LOG.info("cells with reconstruction: %s", table.groupby("species").size().to_dict())
    fetcher.save_tables(out_dir / "allen")
    if sample:
        per_species = {"mouse": 10, "human": 5}
        picks = []
        for sp, g in table.groupby("species"):
            picks.extend(g["id"].head(per_species.get(sp.split()[0].lower(), 5)).tolist())
    else:
        picks = table["id"].tolist()
    n_ok = 0
    for i, specimen_id in enumerate(picks, 1):
        try:
            fetcher.download_reconstruction(int(specimen_id))
            if nwb:
                fetcher.download_ephys_nwb(int(specimen_id))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            LOG.warning("specimen %s failed: %s", specimen_id, exc)
        if i % 50 == 0:
            LOG.info("%d/%d specimens done", i, len(picks))
    LOG.info("downloaded %d/%d reconstructions%s", n_ok, len(picks), " (+NWB)" if nwb else "")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--species", nargs="*", default=["mouse", "human"])
    ap.add_argument("--sample", action="store_true", help="small smoke test (10 mouse + 5 human)")
    ap.add_argument("--nwb", action="store_true", help="also download NWB sweep files")
    ap.add_argument("--neuromorpho", action="store_true", help="download NeuroMorpho neocortical SWCs instead")
    ap.add_argument("--nm-max-pages", type=int, default=2, help="pages of 500 per species for NeuroMorpho")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.neuromorpho:
        n = fetch_neuromorpho_swcs(args.out / "neuromorpho", species=args.species,
                                   max_pages=1 if args.sample else args.nm_max_pages,
                                   max_files=20 if args.sample else None)
        LOG.info("NeuroMorpho: %d SWC files", n)
        return 0
    return download_allen(args.out, args.species, args.sample, args.nwb)


if __name__ == "__main__":
    raise SystemExit(main())
