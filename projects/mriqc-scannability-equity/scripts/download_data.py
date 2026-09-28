#!/usr/bin/env python
"""OpenNeuro corpus downloader for mriqc-scannability-equity.

Uses the public OpenNeuro GraphQL API (no credentials) to build a dataset index and
fetch small metadata files (participants.tsv, participants.json,
dataset_description.json, shipped MRIQC group tables). Raw images are fetched via
S3 (``--raw``). ``--simulate`` writes a synthetic corpus without network access.
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
FILES_URL = "https://openneuro.org/crn/datasets/{ds}/snapshots/{tag}/files/{path}"

DATASETS_QUERY = """
query($after: String) {
  datasets(first: 100, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges { node {
      id
      latestSnapshot {
        tag
        description { Name }
        summary { subjects modalities tasks dataProcessed }
      }
    } }
  }
}
"""

FILES_QUERY = """
query($id: ID!, $tag: String!, $tree: String) {
  snapshot(datasetId: $id, tag: $tag) { files(tree: $tree) { id filename size directory } }
}
"""


def _post(query: str, variables: dict, retries: int = 3) -> dict:
    import requests

    for i in range(retries):
        r = requests.post(GRAPHQL, json={"query": query, "variables": variables}, timeout=120)
        if r.ok:
            js = r.json()
            if "errors" in js:
                raise RuntimeError(js["errors"])
            return js["data"]
        time.sleep(2 * (i + 1))
    r.raise_for_status()
    return {}


def build_index(out: Path, sample: bool = False) -> list[dict]:
    rows, after = [], None
    while True:
        data = _post(DATASETS_QUERY, {"after": after})
        block = data["datasets"]
        for e in block["edges"]:
            n = e["node"]
            snap = n.get("latestSnapshot") or {}
            summ = snap.get("summary") or {}
            mods = [m.lower() for m in (summ.get("modalities") or [])]
            if not any(m in mods for m in ("t1w", "bold", "anat", "func", "mri")):
                continue
            rows.append({
                "id": n["id"], "tag": snap.get("tag"), "name": (snap.get("description") or {}).get("Name", ""),
                "n_subjects": len(summ.get("subjects") or []), "modalities": ";".join(mods),
                "n_tasks": len(summ.get("tasks") or []),
            })
        if sample and len(rows) >= 10:
            rows = rows[:10]
            break
        if not block["pageInfo"]["hasNextPage"]:
            break
        after = block["pageInfo"]["endCursor"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "datasets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(f"indexed {len(rows)} datasets → {out / 'datasets.csv'}")
    return rows


def _list_files(ds: str, tag: str, tree: str | None = None) -> list[dict]:
    data = _post(FILES_QUERY, {"id": ds, "tag": tag, "tree": tree})
    return (data.get("snapshot") or {}).get("files") or []


def _fetch(ds: str, tag: str, path: str, dest: Path) -> bool:
    import requests

    if dest.exists():
        return True
    r = requests.get(FILES_URL.format(ds=ds, tag=tag, path=path.replace("/", ":")), timeout=300)
    if not r.ok:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def fetch_participants(index_rows: list[dict], sample: bool = False) -> None:
    wanted = ["participants.tsv", "participants.json", "dataset_description.json"]
    for i, row in enumerate(index_rows):
        ds, tag = row["id"], row["tag"]
        if not tag:
            continue
        out = DATA / "openneuro" / ds
        try:
            files = _list_files(ds, tag)
        except Exception as exc:  # noqa: BLE001
            print("skip", ds, exc)
            continue
        names = {f["filename"]: f for f in files}
        for w in wanted:
            if w in names:
                ok = _fetch(ds, tag, w, out / w)
                print(("ok   " if ok else "fail ") + f"{ds}/{w}")
        # shipped MRIQC derivatives
        if "derivatives" in names and names["derivatives"].get("directory"):
            try:
                deriv = _list_files(ds, tag, names["derivatives"]["id"])
                for d in deriv:
                    if d.get("directory") and "mriqc" in d["filename"].lower():
                        inner = _list_files(ds, tag, d["id"])
                        for f in inner:
                            if f["filename"] in ("group_T1w.tsv", "group_bold.tsv"):
                                _fetch(ds, tag, f"derivatives/{d['filename']}/{f['filename']}",
                                       out / "derivatives" / "mriqc" / f["filename"])
                                print("ok   ", ds, "derivatives/mriqc/", f["filename"])
            except Exception as exc:  # noqa: BLE001
                print("derivatives listing failed", ds, exc)
        if sample and i >= 9:
            break


def fetch_raw(ds: str, modality: str) -> None:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install boto3") from exc
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    suffix = "_T1w.nii.gz" if modality == "anat" else "_bold.nii.gz"
    out = DATA / "openneuro" / ds
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="openneuro.org", Prefix=f"{ds}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(ds) + 1:]
            keep = rel.endswith(suffix) or rel.endswith((".json", ".tsv")) and "/" not in rel
            if not keep or "derivatives" in rel:
                continue
            dest = out / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file("openneuro.org", key, str(dest))
            print("downloaded", dest)


def fetch_webapi(modality: str, pages: int) -> None:
    import requests

    out = DATA / "webapi"
    out.mkdir(parents=True, exist_ok=True)
    for p in range(1, pages + 1):
        r = requests.get(f"https://mriqc.nimh.nih.gov/api/v1/{modality}", params={"max_results": 1000, "page": p}, timeout=120)
        r.raise_for_status()
        (out / f"{modality}_page{p}.json").write_text(json.dumps(r.json()))
        print("saved page", p)


def simulate() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from scannability.simulate import simulate_corpus

    out = DATA / "sample"
    out.mkdir(parents=True, exist_ok=True)
    df = simulate_corpus(n_datasets=30, seed=0)
    df.to_csv(out / "corpus.csv", index=False)
    print("wrote", out / "corpus.csv", df.shape)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--participants", action="store_true")
    ap.add_argument("--raw", metavar="DSID")
    ap.add_argument("--modality", default="anat", choices=["anat", "func"])
    ap.add_argument("--webapi", choices=["T1w", "bold"])
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args(argv)
    if a.simulate:
        simulate()
        return 0
    if a.webapi:
        fetch_webapi(a.webapi, a.pages)
        return 0
    if a.raw:
        fetch_raw(a.raw, a.modality)
        return 0
    idx_path = DATA / "index" / "datasets.csv"
    if a.index or (a.participants and not idx_path.exists()):
        rows = build_index(DATA / "index", sample=a.sample)
    elif a.participants:
        with open(idx_path) as fh:
            rows = list(csv.DictReader(fh))
    else:
        ap.print_help()
        return 0
    if a.participants:
        fetch_participants(rows, sample=a.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
