#!/usr/bin/env python3
"""Download the two open neonatal EEG cohorts used by neoclock from Zenodo.

Examples
--------
    HELSINKI_ZENODO_RECORD=2547147 python scripts/download_data.py --dataset helsinki --sample
    HIE_ZENODO_RECORD=1234567 python scripts/download_data.py --dataset hie
    python scripts/download_data.py --dataset hie --search        # find the record via the Zenodo search API
    python scripts/download_data.py --build-manifests

Both datasets are open (CC-BY); no credentials are involved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ZENODO_RECORD = "https://zenodo.org/api/records/{record}"
ZENODO_SEARCH = "https://zenodo.org/api/records"
HIE_TITLE = "Neonatal EEG graded for severity of background abnormalities in hypoxic-ischaemic encephalopathy"
HELSINKI_TITLE = "A dataset of neonatal EEG recordings with seizure annotations"


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "neoclock/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "neoclock/0.1"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if existing and resp.status != 206:
                existing = 0
            with open(dest, "ab" if existing else "wb") as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        if exc.code == 416:
            return
        raise
    _log(f"saved {dest.relative_to(ROOT)}")


def zenodo_search(query: str, size: int = 25, max_pages: int = 10) -> Iterator[dict]:
    """Iterate Zenodo search hits (paginated)."""
    for page in range(1, max_pages + 1):
        q = urllib.parse.urlencode({"q": query, "size": size, "page": page, "sort": "bestmatch"})
        data = _get_json(f"{ZENODO_SEARCH}?{q}")
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return
        for h in hits:
            yield h
        if len(hits) < size:
            return


def find_record_by_title(title: str) -> Optional[str]:
    words = [w for w in re.findall(r"[a-z]+", title.lower()) if len(w) > 3]
    best, best_score = None, 0
    for h in zenodo_search(f'"{title}"'):
        t = str(h.get("metadata", {}).get("title", "")).lower()
        score = sum(w in t for w in words)
        if score > best_score:
            best, best_score = str(h.get("id")), score
    return best if best_score >= max(3, len(words) // 2) else None


def download_zenodo_record(record: str, dest: Path, sample: bool, sample_n: int = 2) -> None:
    meta = _get_json(ZENODO_RECORD.format(record=record))
    files = meta.get("files", [])
    _log(f"record {record}: '{meta.get('metadata', {}).get('title', '')}' with {len(files)} files")
    signal = sorted(f for f in files if f["key"].lower().endswith((".edf", ".mat", ".zip", ".bdf")))
    others = [f for f in files if f not in signal]
    todo = others + (signal[:sample_n] if sample else signal)
    for f in todo:
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        if url:
            _download_file(url, dest / f["key"])


def download_helsinki(sample: bool, search: bool) -> None:
    record = os.environ.get("HELSINKI_ZENODO_RECORD")
    if not record and search:
        record = find_record_by_title(HELSINKI_TITLE)
    if not record:
        sys.exit("Set HELSINKI_ZENODO_RECORD (DATASETS.md lists 2547147) or pass --search.")
    download_zenodo_record(record, DATA / "helsinki", sample)


def download_hie(sample: bool, search: bool) -> None:
    record = os.environ.get("HIE_ZENODO_RECORD")
    if not record and search:
        record = find_record_by_title(HIE_TITLE)
    if not record:
        sys.exit("Set HIE_ZENODO_RECORD to the Zenodo record id of the O'Toole et al. (2023) dataset, or pass --search.")
    download_zenodo_record(record, DATA / "hie_grading", sample)


# --------------------------------------------------------------------------- #
def _edf_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        hdr = fh.read(256)
        if len(hdr) < 256:
            return {}
        n_rec = int(hdr[236:244].decode(errors="ignore").strip() or 0)
        rec_dur = float(hdr[244:252].decode(errors="ignore").strip() or 0)
        n_ch = int(hdr[252:256].decode(errors="ignore").strip() or 0)
        fh.seek(256 + n_ch * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80))
        nsamp = [int(fh.read(8).decode(errors="ignore").strip() or 0) for _ in range(n_ch)]
    return dict(fs=(nsamp[0] / rec_dur) if (nsamp and rec_dur) else "", n_channels=n_ch, duration_s=n_rec * rec_dur)


def _find_col(cols: List[str], *patterns: str) -> Optional[str]:
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I):
                return c
    return None


def _clinical_table(path: Path) -> Dict[str, dict]:
    """Helsinki clinical_information.csv -> {record_id: {ga_weeks, pma_weeks}} with tolerant column matching."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    cols = list(rows[0].keys())
    id_col = _find_col(cols, r"^id$", r"eeg", r"record", r"subject", r"patient") or cols[0]
    ga_col = _find_col(cols, r"gestation", r"\bga\b")
    pna_col = _find_col(cols, r"postnatal", r"\bpna\b", r"age at (eeg|recording)", r"days")
    pma_col = _find_col(cols, r"postmenstrual", r"\bpma\b", r"corrected")
    _log(f"clinical columns: id={id_col} ga={ga_col} pna={pna_col} pma={pma_col}")
    out = {}
    for r in rows:
        rid = str(r.get(id_col, "")).strip()
        rid = rid if rid.lower().startswith("eeg") else f"eeg{rid}"
        ga = _to_float(r.get(ga_col)) if ga_col else None
        pma = _to_float(r.get(pma_col)) if pma_col else None
        if pma is None and ga is not None and pna_col:
            pna = _to_float(r.get(pna_col))
            if pna is not None:
                pma = ga + (pna / 7.0 if pna > 4 else pna)  # days if large, weeks if small
        out[rid] = dict(ga_weeks=ga if ga is not None else "", pma_weeks=pma if pma is not None else "")
    return out


def _to_float(v) -> Optional[float]:
    try:
        x = float(str(v).strip())
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def build_manifests() -> None:
    man = DATA / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    events: List[dict] = []

    hel = DATA / "helsinki"
    clin = _clinical_table(hel / "clinical_information.csv")
    for edf in sorted(hel.glob("eeg*.edf"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        h = _edf_header(edf)
        c = clin.get(edf.stem, {})
        records.append(dict(dataset="helsinki", record_id=edf.stem, subject_id=edf.stem, path=str(edf),
                            fs=h.get("fs", ""), n_channels=h.get("n_channels", ""), duration_s=h.get("duration_s", ""),
                            ga_weeks=c.get("ga_weeks", ""), pma_weeks=c.get("pma_weeks", ""), grade=""))
    for ann in sorted(hel.glob("annotations_2017_*.csv")):
        annot = ann.stem[-1]
        with open(ann, newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            continue
        for j in range(len(rows[0])):
            vals = []
            for r in rows:
                if j < len(r) and r[j] not in ("", "NaN"):
                    try:
                        vals.append(int(float(r[j])))
                    except ValueError:
                        break
            on = None
            for t, v in enumerate(vals + [0]):
                if v == 1 and on is None:
                    on = t
                elif v == 0 and on is not None:
                    events.append(dict(dataset="helsinki", record_id=f"eeg{j + 1}", onset_s=on, offset_s=t, annotator=annot))
                    on = None

    hie = DATA / "hie_grading"
    grades: Dict[str, str] = {}
    for tab in list(hie.glob("*.csv")) + list(hie.glob("*.tsv")):
        with open(tab, newline="", encoding="utf-8", errors="ignore") as fh:
            rdr = csv.DictReader(fh, delimiter="\t" if tab.suffix == ".tsv" else ",")
            cols = rdr.fieldnames or []
            gcol = _find_col(cols, r"grade")
            fcol = _find_col(cols, r"file", r"id", r"epoch", r"record") or (cols[0] if cols else None)
            if not gcol or not fcol:
                continue
            for r in rdr:
                grades[Path(str(r[fcol])).stem] = str(r[gcol]).strip()
    for f in sorted(hie.rglob("*")):
        if f.suffix.lower() in (".edf", ".mat", ".bdf"):
            h = _edf_header(f) if f.suffix.lower() == ".edf" else {}
            records.append(dict(dataset="hie", record_id=f.stem, subject_id=re.sub(r"[_-]?(epoch|ep)?\d+$", "", f.stem),
                                path=str(f), fs=h.get("fs", ""), n_channels=h.get("n_channels", ""),
                                duration_s=h.get("duration_s", ""), ga_weeks="", pma_weeks="", grade=grades.get(f.stem, "")))

    for name, rows, cols in (
        ("records.csv", records, ["dataset", "record_id", "subject_id", "path", "fs", "n_channels", "duration_s",
                                  "ga_weeks", "pma_weeks", "grade"]),
        ("events.csv", events, ["dataset", "record_id", "onset_s", "offset_s", "annotator"]),
    ):
        with open(man / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        _log(f"wrote {name}: {len(rows)} rows")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["helsinki", "hie", "all"])
    p.add_argument("--sample", action="store_true")
    p.add_argument("--search", action="store_true", help="resolve the Zenodo record id via the search API")
    p.add_argument("--build-manifests", action="store_true")
    args = p.parse_args(argv)
    if args.dataset in ("helsinki", "all"):
        download_helsinki(args.sample, args.search)
    if args.dataset in ("hie", "all"):
        download_hie(args.sample, args.search)
    if args.build_manifests:
        build_manifests()
    if not args.dataset and not args.build_manifests:
        p.print_help()


if __name__ == "__main__":
    main()
