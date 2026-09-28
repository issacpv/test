#!/usr/bin/env python
"""Fetch the paper corpus and MIMIC demo databases; print credentialed and model download commands.

Examples
--------
    python scripts/download_data.py --papers --sample --out data/papers
    python scripts/download_data.py --johnson2017 --out data/papers
    python scripts/download_data.py --mimic-demo --out data/mimic-iv-demo
    PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=... python scripts/download_data.py --mimic-full --out data/mimiciv --run
    python scripts/download_data.py --models

No MIMIC data is ever sent anywhere by this script; credentialed downloads only run with --run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_QUERY = (
    '("MIMIC-IV"[tiab] OR "MIMIC IV"[tiab]) AND (mortality[tiab] OR readmission[tiab] OR "length of stay"[tiab]) '
    'AND (cohort[tiab] OR patients[tiab]) AND ("2020"[dp] : "2026"[dp])'
)
JOHNSON_README = "https://raw.githubusercontent.com/alistairewj/reproducibility-mimic/master/README.md"
PHYSIONET = {
    "mimic-demo": ("https://physionet.org/files/mimic-iv-demo/2.2/", False),
    "mimic3-demo": ("https://physionet.org/files/mimiciii-demo/1.4/", False),
    "mimic-full": ("https://physionet.org/files/mimiciv/3.1/", True),
    "mimic3-full": ("https://physionet.org/files/mimiciii/1.4/", True),
}
PAPER_LIST_COLUMNS = ["paper_id", "doi", "pmid", "pmcid", "database", "mimic_version", "reported_n", "reported_prevalence", "unit", "johnson2017_reproduced_n", "notes"]
MODEL_SUGGESTIONS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "8B"),
    ("Qwen/Qwen2.5-14B-Instruct", "14B"),
    ("Qwen/Qwen2.5-32B-Instruct", "32B"),
    ("mistralai/Mistral-Small-24B-Instruct-2501", "24B"),
    ("google/gemma-2-9b-it", "9B"),
    ("meta-llama/Llama-3.3-70B-Instruct", "70B"),
]


def cmd_papers(out: Path, sample: bool, fetch_fulltext: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "pmc").mkdir(exist_ok=True)
    key = os.environ.get("NCBI_API_KEY")
    extra = {"api_key": key} if key else {}
    r = requests.get(f"{NCBI}/esearch.fcgi", params={"db": "pubmed", "term": PUBMED_QUERY, "retmode": "json", "retmax": 25 if sample else 3000, **extra}, timeout=60)
    r.raise_for_status()
    pmids = r.json()["esearchresult"]["idlist"]
    print(f"{len(pmids)} PubMed candidates")
    rows = []
    for i in range(0, len(pmids), 100):
        chunk = pmids[i : i + 100]
        rs = requests.get(f"{NCBI}/esummary.fcgi", params={"db": "pubmed", "id": ",".join(chunk), "retmode": "json", **extra}, timeout=60)
        rs.raise_for_status()
        res = rs.json().get("result", {})
        for pmid in chunk:
            item = res.get(pmid, {})
            ids = {a.get("idtype"): a.get("value") for a in item.get("articleids", [])}
            rows.append({"pmid": pmid, "doi": ids.get("doi", ""), "pmcid": ids.get("pmc", ""), "year": (item.get("pubdate") or "")[:4], "title": item.get("title"), "journal": item.get("fulljournalname")})
        time.sleep(0.11 if key else 0.35)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pubmed_candidates.csv", index=False)
    print(f"wrote {out / 'pubmed_candidates.csv'}")
    if fetch_fulltext:
        n_ok = 0
        for pmcid in df["pmcid"].dropna().astype(str):
            if not pmcid.startswith("PMC"):
                continue
            dest = out / "pmc" / f"{pmcid}.xml"
            if dest.exists():
                n_ok += 1
                continue
            try:
                rf = requests.get(f"{NCBI}/efetch.fcgi", params={"db": "pmc", "id": pmcid[3:], "retmode": "xml", **extra}, timeout=120)
                if rf.status_code == 200 and "<article" in rf.text:
                    dest.write_text(rf.text)
                    n_ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  {pmcid}: {exc}")
            time.sleep(0.11 if key else 0.35)
        print(f"{n_ok} PMC open-access full texts in {out / 'pmc'}")
    template = out / "paper_list.csv"
    if not template.exists():
        pd.DataFrame(columns=PAPER_LIST_COLUMNS).to_csv(template, index=False)
        print(f"wrote paper-list template -> {template}")


def cmd_johnson2017(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = requests.get(JOHNSON_README, timeout=60)
    if r.status_code == 200:
        (out / "johnson2017_README.md").write_text(r.text)
        print(f"saved {out / 'johnson2017_README.md'}")
    print("Clone the repository for the reproduction SQL and study table:\n  git clone --recursive https://github.com/alistairewj/reproducibility-mimic.git data/reproducibility-mimic")
    template = out / "paper_list.csv"
    if not template.exists():
        pd.DataFrame(columns=PAPER_LIST_COLUMNS).to_csv(template, index=False)


def wget_recursive(url: str, out: Path, credentialed: bool, run: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-r", "-N", "-c", "-np", "-P", str(out)]
    user = os.environ.get("PHYSIONET_USERNAME")
    pw = os.environ.get("PHYSIONET_PASSWORD")
    if credentialed:
        cmd += ["--user", user or "$PHYSIONET_USERNAME", "--password", "********"]
    cmd.append(url)
    print(" ".join(cmd))
    if credentialed and not run:
        print("(dry run; add --run to execute; requires PhysioNet credentialing + signed DUA)")
        return
    if credentialed:
        if not (user and pw):
            print("set PHYSIONET_USERNAME and PHYSIONET_PASSWORD")
            sys.exit(1)
        cmd[cmd.index("********")] = pw
    subprocess.run(cmd, check=True)


def cmd_models() -> None:
    print("Suggested open-weight models (check each licence on Hugging Face):")
    for name, size in MODEL_SUGGESTIONS:
        local = "models/" + name.split("/")[-1].lower()
        print(f"  [{size}] huggingface-cli download {name} --local-dir {local}")
    print("\nServe locally (OpenAI-compatible), e.g.:")
    print("  python -m vllm.entrypoints.openai.api_server --model models/llama-3.1-8b-instruct --port 8000 --max-model-len 16384")
    print("  # or: llama-server -m models/<model>.gguf --port 8000")
    print("Then: export COHORT_REPRO_BASE_URL=http://localhost:8000/v1 COHORT_REPRO_MODEL=<served name>")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--papers", action="store_true")
    p.add_argument("--no-fulltext", action="store_true")
    p.add_argument("--johnson2017", action="store_true")
    p.add_argument("--mimic-demo", action="store_true")
    p.add_argument("--mimic3-demo", action="store_true")
    p.add_argument("--mimic-full", action="store_true")
    p.add_argument("--mimic3-full", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--models", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    did = False
    if args.papers:
        cmd_papers(out, args.sample, fetch_fulltext=not args.no_fulltext)
        did = True
    if args.johnson2017:
        cmd_johnson2017(out)
        did = True
    for flag, key in (("mimic_demo", "mimic-demo"), ("mimic3_demo", "mimic3-demo"), ("mimic_full", "mimic-full"), ("mimic3_full", "mimic3-full")):
        if getattr(args, flag):
            url, cred = PHYSIONET[key]
            wget_recursive(url, out, cred, args.run or not cred)
            did = True
    if args.models:
        cmd_models()
        did = True
    if not did:
        p.print_help()


if __name__ == "__main__":
    main()
