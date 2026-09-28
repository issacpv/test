#!/usr/bin/env python3
"""Fetch OpenNeuro task-fMRI datasets and their fMRIPrep derivatives.

Everything here is open (CC0) - no credentials are needed. Three routes:

* ``openneuro-py`` (preferred; ``pip install openneuro-py``)
* AWS CLI against the public bucket (``aws s3 sync --no-sign-request s3://openneuro.org/<id> ...``)
* DataLad (``datalad install https://github.com/OpenNeuroDatasets/<id>``)

Examples
--------
    python scripts/download_data.py --check                     # which registry datasets ship fMRIPrep derivatives?
    python scripts/download_data.py --sample                    # ds002785: participants, events, confounds for 2 subjects
    python scripts/download_data.py --dataset ds001734 --confounds-only
    python scripts/download_data.py --dataset ds002785 --task workingmemory --bold --n-subjects 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from fmri_multiverse import fetch  # noqa: E402


def check_registry() -> None:
    print(f"{'dataset':<10} {'registry':<8} {'live check':<12} name")
    for ds in fetch.DATASETS.values():
        live = fetch.has_fmriprep_derivatives(ds.dataset_id)
        live_s = {True: "fmriprep", False: "none", None: "offline?"}[live]
        print(f"{ds.dataset_id:<10} {ds.derivatives_in_dataset:<8} {live_s:<12} {ds.name}")


def build_includes(task: Optional[str], bold: bool, n_subjects: Optional[int], confounds_only: bool) -> List[str]:
    task_glob = f"*task-{task}*" if task else "*"
    inc = ["participants.tsv", "participants.json", "dataset_description.json", "task-*_bold.json", f"sub-*/func/{task_glob}_events.tsv", f"sub-*/func/{task_glob}_bold.json"]
    inc.append(f"derivatives/fmriprep*/sub-*/func/{task_glob}desc-confounds_*.tsv")
    inc.append(f"derivatives/fmriprep*/sub-*/func/{task_glob}desc-confounds_*.json")
    if bold and not confounds_only:
        inc.append(f"derivatives/fmriprep*/sub-*/func/{task_glob}space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        inc.append(f"derivatives/fmriprep*/sub-*/func/{task_glob}space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    if n_subjects:
        # openneuro-py / aws globs cannot express "first N subjects"; restrict by sub-0? patterns instead
        width = 2 if n_subjects < 100 else 3
        subs = [f"sub-{i:0{width}d}" for i in range(1, n_subjects + 1)] + [f"sub-{i:04d}" for i in range(1, n_subjects + 1)]
        inc = [p.replace("sub-*", "{" + ",".join(subs) + "}") if "sub-*" in p else p for p in inc]
    return inc


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="query OpenNeuro for fMRIPrep derivatives in every registry dataset")
    ap.add_argument("--sample", action="store_true", help="small smoke-test download from ds002785")
    ap.add_argument("--dataset", help="OpenNeuro accession, e.g. ds001734")
    ap.add_argument("--task")
    ap.add_argument("--bold", action="store_true", help="also fetch preprocessed BOLD + masks (large)")
    ap.add_argument("--confounds-only", action="store_true")
    ap.add_argument("--n-subjects", type=int)
    ap.add_argument("--datalad", action="store_true", help="use DataLad instead of openneuro-py/S3")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        check_registry()
        return
    if args.sample:
        inc = build_includes("workingmemory", bold=False, n_subjects=2, confounds_only=True)
        try:
            cmd = fetch.download("ds002785", args.out, include=inc, dry_run=args.dry_run)
            print("ran:", " ".join(cmd))
        except RuntimeError as exc:
            print(exc)
            print("Fallback listing via anonymous S3 (boto3):")
            try:
                for k in fetch.list_s3_prefix("ds002785", "derivatives/fmriprep/sub-0001/", max_keys=20):
                    print("  ", k)
            except Exception as e:  # noqa: BLE001
                print("  ", e)
        return
    if not args.dataset:
        ap.print_help()
        return
    inc = build_includes(args.task, args.bold, args.n_subjects, args.confounds_only)
    if args.datalad:
        cmds = fetch.datalad_install(args.dataset, args.out, get=[p for p in inc if not p.startswith("*")], dry_run=args.dry_run)
        for c in cmds:
            print(" ".join(c))
    else:
        cmd = fetch.download(args.dataset, args.out, include=inc, dry_run=args.dry_run)
        print("ran:" if not args.dry_run else "would run:", " ".join(cmd))


if __name__ == "__main__":
    main()
