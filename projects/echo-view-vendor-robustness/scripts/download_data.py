#!/usr/bin/env python
"""Stage echocardiography datasets for echo_view.

Most sources need manual registration/DUA (EchoNet, CAMUS, TMED-2, TTE47); this
script documents them and downloads the credentialed MIMIC-IV-ECHO sample.

Credentials for MIMIC-IV-ECHO read from PHYSIONET_USERNAME / PHYSIONET_PASSWORD.

Examples
--------
python scripts/download_data.py --dataset info
python scripts/download_data.py --dataset mimic-iv-echo --out data/mimic-iv-echo --sample
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

MIMIC_ECHO_ROOT = "https://physionet.org/files/mimic-iv-echo/0.1/"
MIMIC_ECHO_META = ["record_list.csv"]  # study manifest; DICOMs under files/

REGISTRATION_INFO = {
    "echonet-dynamic": "https://echonet.github.io/dynamic/  (register + DUA; EchoNet-Dynamic.zip)",
    "camus": "https://www.creatis.insa-lyon.fr/Challenge/camus/  (register)",
    "tmed2": "https://tmed.cs.tufts.edu/tmed_v2.html  (register + DUA)",
    "tte47": "https://thrive-centre.com/datasets/TTE47  (open benchmark)",
}


def _auth() -> tuple[str, str]:
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    if not u or not p:
        sys.exit("Set PHYSIONET_USERNAME / PHYSIONET_PASSWORD for MIMIC-IV-ECHO.")
    return u, p


def download(url: str, dest: Path, auth) -> bool:
    if requests is None:
        sys.exit("pip install requests")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=180) as r:
        if r.status_code != 200:
            print(f"  [fail {r.status_code}] {url}")
            return False
        with open(dest, "wb") as f:
            for c in r.iter_content(1 << 20):
                if c:
                    f.write(c)
    print(f"  [ok] {dest}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["info", "mimic-iv-echo"])
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args(argv)

    if args.dataset == "info":
        print("Manual-registration datasets:")
        for k, v in REGISTRATION_INFO.items():
            print(f"  {k}: {v}")
        return 0

    auth = _auth()
    for rel in MIMIC_ECHO_META:
        download(MIMIC_ECHO_ROOT + rel, args.out / Path(rel).name, auth)
    if args.sample:
        print("  For a DICOM sample, pick a study path from record_list.csv and fetch it, e.g.:")
        print(f"    {MIMIC_ECHO_ROOT}files/pXX/pXXXXXXXX/sXXXXXXXX/<file>.dcm")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
