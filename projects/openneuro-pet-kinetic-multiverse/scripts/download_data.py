#!/usr/bin/env python
"""Discover and fetch PET-BIDS datasets from OpenNeuro for the kinetic multiverse.

``--discover`` builds ``data/index/pet_datasets.csv`` from the OpenNeuro GraphQL API
(no credentials) including tracer, frame count and blood-data availability read
from the ``*_pet.json`` sidecars. ``--fetch DSID`` downloads pet/ and anat/ folders
via anonymous S3. ``--simulate`` writes synthetic test-retest TACs offline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GRAPHQL = "https://openneuro.org/crn/graphql"
FILE_URL = "https://openneuro.org/crn/datasets/{ds}/snapshots/{tag}/files/{path}"

Q_DATASETS = """
query($after: String) {
  datasets(first: 50, after: $after, modality: "pet") {
    pageInfo { hasNextPage endCursor }
    edges { node { id latestSnapshot { tag description { Name } summary { subjects modalities sessions } } } }
  }
}
"""
Q_FILES = """
query($id: ID!, $tag: String!, $tree: String) {
  snapshot(datasetId: $id, tag: $tag) { files(tree: $tree) { id filename size directory } }
}
"""


def _post(query: str, variables: dict, retries: int = 3) -> dict:
    import requests

    for i in range(retries):
        r = requests.post(GRAPHQL, json={"query": query, "variables": variables}, timeout=120)
        if r.ok and "errors" not in r.json():
            return r.json()["data"]
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"GraphQL failed: {r.status_code} {r.text[:200]}")


def _files(ds: str, tag: str, tree: str | None = None) -> list[dict]:
    return (_post(Q_FILES, {"id": ds, "tag": tag, "tree": tree}).get("snapshot") or {}).get("files") or []


def _get_json(ds: str, tag: str, path: str) -> dict | None:
    import requests

    r = requests.get(FILE_URL.format(ds=ds, tag=tag, path=path.replace("/", ":")), timeout=120)
    if not r.ok:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def _walk_pet_sidecars(ds: str, tag: str, max_subjects: int = 3) -> tuple[list[dict], bool]:
    """Return sidecar JSONs for up to ``max_subjects`` subjects and whether blood TSVs exist."""
    sidecars, blood = [], False
    top = _files(ds, tag)
    subs = [f for f in top if f["filename"].startswith("sub-") and f.get("directory")][:max_subjects]
    for s in subs:
        entries = _files(ds, tag, s["id"])
        # sessions or direct pet folder
        folders = [e for e in entries if e.get("directory") and (e["filename"].startswith("ses-") or e["filename"] == "pet")]
        for fo in folders:
            inner = _files(ds, tag, fo["id"])
            if fo["filename"].startswith("ses-"):
                inner = [i for i in inner if i.get("directory") and i["filename"] == "pet"]
                inner = _files(ds, tag, inner[0]["id"]) if inner else []
                prefix = f"{s['filename']}/{fo['filename']}/pet/"
            else:
                prefix = f"{s['filename']}/pet/"
            for f in inner:
                if f["filename"].endswith("_pet.json"):
                    js = _get_json(ds, tag, prefix + f["filename"])
                    if js:
                        sidecars.append(js)
                if f["filename"].endswith("_blood.tsv"):
                    blood = True
    return sidecars, blood


def discover(sample: bool = False) -> list[dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from pet_multiverse.tacs import classify_tracer

    rows, after = [], None
    while True:
        block = _post(Q_DATASETS, {"after": after})["datasets"]
        for e in block["edges"]:
            n = e["node"]
            snap = n.get("latestSnapshot") or {}
            tag = snap.get("tag")
            if not tag:
                continue
            try:
                sidecars, blood = _walk_pet_sidecars(n["id"], tag)
            except Exception as exc:  # noqa: BLE001
                print("sidecar walk failed", n["id"], exc)
                sidecars, blood = [], False
            tracers = sorted({str(s.get("TracerName", "")) for s in sidecars if s.get("TracerName")})
            n_frames = max((len(s.get("FrameTimesStart", []) or []) for s in sidecars), default=0)
            durations = sorted({round(sum(s.get("FrameDuration", []) or []) / 60.0, 1) for s in sidecars})
            rows.append({
                "id": n["id"], "tag": tag, "name": (snap.get("description") or {}).get("Name", ""),
                "n_subjects": len((snap.get("summary") or {}).get("subjects") or []),
                "n_sessions": len((snap.get("summary") or {}).get("sessions") or []),
                "tracers": ";".join(tracers), "tracer_class": ";".join(classify_tracer(t) for t in tracers) or "unknown",
                "n_frames": n_frames, "dynamic": int(n_frames > 1), "scan_minutes": ";".join(map(str, durations)),
                "blood_tsv": int(blood),
                "decay_corrected": ";".join(sorted({str(s.get("ImageDecayCorrected")) for s in sidecars})),
            })
            print(rows[-1]["id"], rows[-1]["tracers"], "frames", n_frames, "blood", blood)
            if sample and len(rows) >= 5:
                break
        if sample and len(rows) >= 5 or not block["pageInfo"]["hasNextPage"]:
            break
        after = block["pageInfo"]["endCursor"]
    out = DATA / "index"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pet_datasets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / "pet_datasets.csv", len(rows))
    return rows


def fetch(ds: str, include_anat: bool = True) -> None:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install boto3") from exc
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    out = DATA / "openneuro" / ds
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket="openneuro.org", Prefix=f"{ds}/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(ds) + 1:]
            top_level = "/" not in rel
            keep = top_level or "/pet/" in f"/{rel}" or (include_anat and "/anat/" in f"/{rel}")
            if not keep or rel.startswith("derivatives/"):
                continue
            dest = out / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file("openneuro.org", obj["Key"], str(dest))
            print("downloaded", dest)


def simulate() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from pet_multiverse.models import simulate_test_retest_dataset

    out = DATA / "sample"
    out.mkdir(parents=True, exist_ok=True)
    tacs, meta = simulate_test_retest_dataset(n_subjects=12, seed=0)
    tacs.to_csv(out / "simulated_tacs.csv", index=False)
    (out / "simulated_truth.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / "simulated_tacs.csv", tacs.shape)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--fetch", metavar="DSID")
    ap.add_argument("--no-anat", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args(argv)
    if a.simulate:
        simulate()
    elif a.discover:
        discover(sample=a.sample)
    elif a.fetch:
        fetch(a.fetch, include_anat=not a.no_anat)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
