#!/usr/bin/env python
"""Stage the echo datasets used by echo_uq.

All real datasets need a registration / DUA (Stanford AIMI for EchoNet-*, Creatis
for CAMUS, Tufts for TMED-2), so this script:

  --sample            writes a synthetic beating-LV dataset with known EF to data/synthetic/
  --dataset camus     downloads CAMUS from the URL you received after registering (env CAMUS_URL)
  --dataset echonet-dynamic | echonet-pediatric | echonet-lvh | tmed2
                      prints the exact steps and checks whether the data are already in place
  --verify            counts videos / studies found under --out

Examples
--------
python scripts/download_data.py --sample --out data --n 40
CAMUS_URL='https://.../CAMUS_public.zip' python scripts/download_data.py --dataset camus --out data
python scripts/download_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INSTRUCTIONS = {
    "echonet-dynamic": ("https://echonet.github.io/dynamic/", "FileList.csv", "Videos/*.avi",
                        "Register on the Stanford AIMI shared-datasets portal, accept the research use agreement, "
                        "download the zip and extract so that data/echonet-dynamic/FileList.csv exists."),
    "echonet-pediatric": ("https://echonet.github.io/pediatric/", "A4C/FileList.csv", "*/Videos/*.avi",
                          "Same portal as EchoNet-Dynamic; extract to data/echonet-pediatric/{A4C,PSAX}/."),
    "echonet-lvh": ("https://echonet.github.io/lvh/", "MeasurementsList.csv", "**/*.avi",
                    "Same portal; PLAX videos + MeasurementsList.csv under data/echonet-lvh/."),
    "camus": ("https://www.creatis.insa-lyon.fr/Challenge/camus/", "training", "**/*_sequence.mhd",
              "Register, receive the archive URL by e-mail, then export CAMUS_URL and re-run with --dataset camus."),
    "tmed2": ("https://tmed.cs.tufts.edu/tmed_v2.html", "labeled", "**/*.png",
              "Sign the Tufts data use agreement, download and extract to data/tmed2/."),
}


def write_synthetic(out: Path, n: int, seed: int) -> None:
    from echo_uq.video import synthetic_lv_video

    rng = np.random.default_rng(seed)
    vdir = out / "synthetic" / "Videos"
    vdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        ef = float(np.clip(rng.normal(0.58, 0.12), 0.15, 0.80))
        jitter = 0.15 if rng.random() < 0.25 else 0.0
        noise = float(rng.choice([8.0, 12.0, 30.0]))
        echo = synthetic_lv_video(n_frames=96, ef=ef, a_ed=float(rng.uniform(32, 42)), b_ed=float(rng.uniform(16, 24)),
                                  theta=float(rng.uniform(0.2, 0.5)), heart_rate=float(rng.uniform(55, 110)),
                                  rr_jitter=jitter, noise=noise, seed=int(rng.integers(0, 1 << 31)))
        name = f"SYN{i:05d}"
        np.savez_compressed(vdir / f"{name}.npz", frames=echo.frames, masks=echo.masks)
        rows.append({"FileName": name, "EF": round(100 * echo.ef_true, 2), "EDV": round(echo.edv_true, 1), "ESV": round(echo.esv_true, 1),
                     "FrameHeight": 112, "FrameWidth": 112, "FPS": echo.fps, "NumberOfFrames": echo.frames.shape[0],
                     "Split": "TRAIN" if i < 0.6 * n else "VAL" if i < 0.8 * n else "TEST",
                     "Irregular": int(jitter > 0), "Noise": noise})
    with open(out / "synthetic" / "FileList.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {n} synthetic videos to {vdir} and FileList.csv")


def download_camus(out: Path) -> None:
    url = os.environ.get("CAMUS_URL")
    if not url:
        print(INSTRUCTIONS["camus"][3])
        return
    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")
    dest = out / "camus"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "camus_download.zip"
    if not archive.exists():
        print(f"downloading CAMUS archive -> {archive}")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(archive, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        print(f"extracted to {dest}")
    else:
        print("downloaded file is not a zip archive; extract it manually (may be .tar.gz or .rar)")


def verify(out: Path) -> None:
    for name, (_, marker, pattern, _) in INSTRUCTIONS.items():
        d = out / name
        n = len(list(d.glob(pattern))) if d.exists() else 0
        ok = (d / marker).exists()
        print(f"{name:18s} marker={'yes' if ok else 'NO ':3s} files={n}")
    syn = out / "synthetic"
    if syn.exists():
        print(f"{'synthetic':18s} marker={'yes' if (syn / 'FileList.csv').exists() else 'NO '} files={len(list(syn.glob('Videos/*.npz')))}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(INSTRUCTIONS))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample", action="store_true", help="generate a synthetic smoke-test dataset")
    ap.add_argument("--n", type=int, default=40, help="number of synthetic videos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.out)
        return
    if args.sample:
        write_synthetic(args.out, args.n, args.seed)
        return
    if args.dataset is None:
        ap.error("--dataset, --sample or --verify is required")
    if args.dataset == "camus":
        download_camus(args.out)
        return
    url, marker, pattern, steps = INSTRUCTIONS[args.dataset]
    d = args.out / args.dataset
    print(f"{args.dataset}: {url}\n{steps}")
    print(f"present: {(d / marker).exists()} ({len(list(d.glob(pattern))) if d.exists() else 0} files)")


if __name__ == "__main__":
    main()
