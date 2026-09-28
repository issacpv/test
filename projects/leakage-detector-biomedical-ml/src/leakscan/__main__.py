"""Command-line interface.

    python -m leakscan scan PATH [--json out.json] [--md out.md] [--subject-level]
    python -m leakscan audit repos.csv --clones data/repos --out outputs/findings.csv

``repos.csv`` needs a ``full_name`` column (``owner/repo``) and optionally a
``dataset`` column used for stratified prevalence tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .corpus import clone_shallow
from .detector import scan_path
from .report import findings_to_frame, prevalence_table, summarise_repo, to_markdown


def _cmd_scan(args: argparse.Namespace) -> int:
    results = scan_path(args.path, subject_level=True if args.subject_level else None)
    if args.json:
        Path(args.json).write_text(json.dumps([[f.to_dict() for f in r.findings] for r in results], indent=2))
    md = to_markdown(results, min_severity=args.min_severity)
    if args.md:
        Path(args.md).write_text(md)
    else:
        print(md)
    worst = max((r.max_severity for r in results), default="info")
    return 1 if worst in ("high", "medium") and args.fail_on_findings else 0


def _cmd_audit(args: argparse.Namespace) -> int:
    repos = pd.read_csv(args.repos)
    frames = []
    for _, row in repos.iterrows():
        dest = clone_shallow(row["full_name"], args.clones)
        if dest is None:
            print(f"[audit] clone failed: {row['full_name']}", file=sys.stderr)
            continue
        res = scan_path(dest, subject_level=True if args.subject_level else None)
        df = findings_to_frame(res, repo=row["full_name"])
        if "dataset" in repos.columns:
            df["dataset"] = row["dataset"]
        frames.append(df)
    if not frames:
        print("no results", file=sys.stderr)
        return 1
    all_df = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out, index=False)
    repo_df = summarise_repo(all_df, min_severity=args.min_severity)
    if "dataset" in all_df.columns:
        repo_df = repo_df.merge(all_df.groupby("repo")["dataset"].first().reset_index(), on="repo", how="left")
        prev = prevalence_table(repo_df, by="dataset")
    else:
        prev = prevalence_table(repo_df)
    repo_df.to_csv(str(Path(args.out).with_name("repo_summary.csv")), index=False)
    prev.to_csv(str(Path(args.out).with_name("prevalence.csv")), index=False)
    print(prev.to_string(index=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="leakscan")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a file or directory")
    s.add_argument("path")
    s.add_argument("--json")
    s.add_argument("--md")
    s.add_argument("--subject-level", action="store_true", help="assert the data are subject-level (grade random splits high)")
    s.add_argument("--min-severity", default="low", choices=["info", "low", "medium", "high"])
    s.add_argument("--fail-on-findings", action="store_true")
    s.set_defaults(func=_cmd_scan)
    a = sub.add_parser("audit", help="clone and scan a list of repositories")
    a.add_argument("repos", help="CSV with full_name[,dataset]")
    a.add_argument("--clones", default="data/repos")
    a.add_argument("--out", default="outputs/findings.csv")
    a.add_argument("--subject-level", action="store_true")
    a.add_argument("--min-severity", default="medium", choices=["low", "medium", "high"])
    a.set_defaults(func=_cmd_audit)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except BrokenPipeError:  # e.g. `python -m leakscan scan . | head`
        sys.exit(0)
