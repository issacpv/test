#!/usr/bin/env python
"""Fetch Synthea exports and PhysioNet demo databases; print/run credentialed downloads.

Examples
--------
    python scripts/download_data.py --synthea-sample --out data/synthea
    python scripts/download_data.py --synthea-generate 5000 --seed 1 --out data/synthea
    python scripts/download_data.py --mimic-demo --out data/mimic-iv-demo
    python scripts/download_data.py --eicu-demo --out data/eicu-demo
    PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=... python scripts/download_data.py --mimic-full --out data/mimiciv --run

Credentialed downloads read credentials from environment variables and are only executed with --run.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

SYNTHEA_SAMPLE_URLS = [
    "https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip",
    "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip",
]
SYNTHEA_JAR_URL = "https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"
PHYSIONET_OPEN = {
    "mimic-demo": "https://physionet.org/files/mimic-iv-demo/2.2/",
    "eicu-demo": "https://physionet.org/files/eicu-crd-demo/2.0.1/",
}
PHYSIONET_CREDENTIALED = {
    "mimic-full": "https://physionet.org/files/mimiciv/3.1/",
    "eicu-full": "https://physionet.org/files/eicu-crd/2.0/",
}


def synthea_sample(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for url in SYNTHEA_SAMPLE_URLS:
        try:
            print(f"downloading {url}")
            r = requests.get(url, timeout=300)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(out)
            (out / "VERSION.txt").write_text(f"source={url}\n")
            csvs = list(out.rglob("patients.csv"))
            print(f"extracted; patients.csv at {csvs[0] if csvs else 'NOT FOUND'}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")
    print("Could not fetch a sample export; use --synthea-generate or download manually from https://synthea.mitre.org/downloads")


def synthea_generate(out: Path, n_patients: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    jar = out / "synthea-with-dependencies.jar"
    if not jar.exists():
        print(f"downloading {SYNTHEA_JAR_URL}")
        with requests.get(SYNTHEA_JAR_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(jar, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    if shutil.which("java") is None:
        print("java not found; install Java 11+ and re-run")
        sys.exit(1)
    cmd = [
        "java", "-jar", str(jar), "-p", str(n_patients), "-s", str(seed),
        "--exporter.csv.export=true", "--exporter.fhir.export=false", "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false", f"--exporter.baseDirectory={out}",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    (out / "VERSION.txt").write_text(f"jar={SYNTHEA_JAR_URL}\nn_patients={n_patients}\nseed={seed}\n")


def wget_recursive(url: str, out: Path, user: str | None = None, password: str | None = None, run: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-r", "-N", "-c", "-np", "-P", str(out)]
    if user:
        cmd += ["--user", user, "--password", "********"]
    cmd.append(url)
    print(" ".join(cmd))
    if not run:
        print("(dry run; add --run to execute)")
        return
    if user:
        cmd[cmd.index("********")] = password or ""
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data")
    p.add_argument("--synthea-sample", action="store_true")
    p.add_argument("--synthea-generate", type=int, metavar="N_PATIENTS")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mimic-demo", action="store_true")
    p.add_argument("--eicu-demo", action="store_true")
    p.add_argument("--mimic-full", action="store_true")
    p.add_argument("--eicu-full", action="store_true")
    p.add_argument("--run", action="store_true", help="execute credentialed wget instead of printing it")
    args = p.parse_args()
    out = Path(args.out)

    if args.synthea_sample:
        synthea_sample(out)
    if args.synthea_generate:
        synthea_generate(out, args.synthea_generate, args.seed)
    if args.mimic_demo:
        wget_recursive(PHYSIONET_OPEN["mimic-demo"], out, run=True)
    if args.eicu_demo:
        wget_recursive(PHYSIONET_OPEN["eicu-demo"], out, run=True)
    for flag, key in (("mimic_full", "mimic-full"), ("eicu_full", "eicu-full")):
        if getattr(args, flag):
            user = os.environ.get("PHYSIONET_USERNAME")
            pw = os.environ.get("PHYSIONET_PASSWORD")
            if args.run and not (user and pw):
                print("set PHYSIONET_USERNAME and PHYSIONET_PASSWORD (credentialed access + signed DUA required)")
                sys.exit(1)
            wget_recursive(PHYSIONET_CREDENTIALED[key], out, user=user or "$PHYSIONET_USERNAME", password=pw, run=args.run)
    if not any([args.synthea_sample, args.synthea_generate, args.mimic_demo, args.eicu_demo, args.mimic_full, args.eicu_full]):
        p.print_help()


if __name__ == "__main__":
    main()
