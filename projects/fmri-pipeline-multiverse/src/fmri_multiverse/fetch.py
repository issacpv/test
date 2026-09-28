"""Dataset registry and fetchers for OpenNeuro task-fMRI datasets with
fMRIPrep derivatives.

Three access routes are supported, in order of preference:

1. ``openneuro-py`` (``pip install openneuro-py``): resumable, glob filters.
2. Public S3 bucket ``s3://openneuro.org/<dataset>/`` (anonymous, boto3 or
   the AWS CLI with ``--no-sign-request``).
3. DataLad (``datalad install https://github.com/OpenNeuroDatasets/<id>``).

The OpenNeuro GraphQL endpoint is used to list snapshot files and to
discover whether a dataset ships ``derivatives/fmriprep``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
OPENNEURO_BUCKET = "openneuro.org"


@dataclass
class Finding:
    """A published contrast whose robustness is scored."""

    task: str
    contrast: str  # nilearn-style expression over trial_type columns
    region: str  # a-priori ROI (atlas label or MNI sphere) where the effect was reported
    source: str  # paper reporting the effect


@dataclass
class DatasetSpec:
    dataset_id: str
    name: str
    n_subjects: int
    tasks: Sequence[str]
    findings: Sequence[Finding]
    derivatives_in_dataset: str  # "yes" | "verify" | "no"
    fmriprep_version: Optional[str] = None
    notes: str = ""
    reference: str = ""


#: Candidate datasets. ``derivatives_in_dataset`` = "verify" means the
#: registry author could not confirm derivative availability; run
#: ``has_fmriprep_derivatives(dataset_id)`` before planning compute.
DATASETS: Dict[str, DatasetSpec] = {
    "ds000030": DatasetSpec(
        "ds000030", "UCLA Consortium for Neuropsychiatric Phenomics (CNP)", 272,
        ["stopsignal", "bart", "scap", "taskswitch", "pamenc", "pamret", "rest"],
        [
            Finding("stopsignal", "STOP_SUCCESS - GO", "right inferior frontal gyrus / pre-SMA", "Poldrack et al. 2016 Sci Data; Gorgolewski et al. 2017"),
            Finding("bart", "ACCEPT - REJECT", "ventral striatum", "Poldrack et al. 2016 Sci Data"),
            Finding("scap", "load4 - load1", "dorsolateral prefrontal / intraparietal", "Poldrack et al. 2016 Sci Data"),
        ],
        "yes", fmriprep_version="1.1.x (dataset-shipped)", notes="Also MRIQC outputs; 4 diagnostic groups (use controls or model group).",
        reference="Poldrack et al. (2016) Sci Data 3:160110",
    ),
    "ds001734": DatasetSpec(
        "ds001734", "NARPS mixed-gambles", 108, ["MGT"],
        [
            Finding("MGT", "gain", "ventromedial PFC / ventral striatum (H1-H4)", "Botvinik-Nezer et al. 2020 Nature"),
            Finding("MGT", "loss", "amygdala / vmPFC (H5-H9)", "Botvinik-Nezer et al. 2020 Nature"),
        ],
        "yes", fmriprep_version="1.1.4 (dataset-shipped)", notes="Two groups (equalRange / equalIndifference); nine pre-specified hypotheses with 70-team results as ground truth.",
        reference="Botvinik-Nezer et al. (2019) Sci Data 6:106; (2020) Nature 582:84",
    ),
    "ds002785": DatasetSpec(
        "ds002785", "AOMIC-PIOP1", 216, ["workingmemory", "emomatching", "faces", "gstroop", "anticipation", "restingstate"],
        [
            Finding("workingmemory", "active - passive", "dorsolateral PFC / intraparietal sulcus", "Snoek et al. 2021 Sci Data"),
            Finding("emomatching", "emotion - control", "amygdala / fusiform", "Snoek et al. 2021 Sci Data"),
            Finding("gstroop", "incongruent - congruent", "dorsal ACC", "Snoek et al. 2021 Sci Data"),
        ],
        "yes", fmriprep_version="1.3.2 (dataset-shipped)", notes="Physio (RETROICOR) regressors included; multi-echo not used.",
        reference="Snoek et al. (2021) Sci Data 8:85",
    ),
    "ds002790": DatasetSpec(
        "ds002790", "AOMIC-PIOP2", 226, ["workingmemory", "emomatching", "stopsignal", "restingstate"],
        [
            Finding("workingmemory", "active - passive", "dorsolateral PFC / intraparietal sulcus", "Snoek et al. 2021 Sci Data"),
            Finding("stopsignal", "stop - go", "right IFG / pre-SMA", "Snoek et al. 2021 Sci Data"),
        ],
        "yes", fmriprep_version="1.3.2 (dataset-shipped)", reference="Snoek et al. (2021) Sci Data 8:85",
    ),
    "ds003097": DatasetSpec(
        "ds003097", "AOMIC-ID1000", 928, ["moviewatching"],
        [Finding("moviewatching", "inter-subject correlation", "visual / auditory cortex", "Snoek et al. 2021 Sci Data")],
        "yes", fmriprep_version="1.3.2 (dataset-shipped)", notes="Naturalistic only; no event-related contrasts. Optional ISC arm.",
        reference="Snoek et al. (2021) Sci Data 8:85",
    ),
    "ds000117": DatasetSpec(
        "ds000117", "Wakeman & Henson multimodal faces", 16, ["facerecognition"],
        [Finding("facerecognition", "faces - scrambled", "fusiform face area / occipital face area", "Wakeman & Henson 2015 Sci Data")],
        "verify", notes="Small n; fMRIPrep derivatives may be under the OpenNeuroDerivatives organisation rather than in the dataset.",
        reference="Wakeman & Henson (2015) Sci Data 2:150001",
    ),
    "ds000228": DatasetSpec(
        "ds000228", "Richardson pixar theory-of-mind (children + adults)", 155, ["pixar"],
        [Finding("pixar", "mental - pain (reverse correlation)", "TPJ / mPFC", "Richardson et al. 2018 Nat Commun")],
        "verify", notes="Naturalistic movie with event annotations; high motion in children - good for fragility modelling.",
        reference="Richardson et al. (2018) Nat Commun 9:1027",
    ),
}


# --------------------------------------------------------------------------- #
# OpenNeuro GraphQL
# --------------------------------------------------------------------------- #
def _graphql(query: str, variables: Optional[dict] = None, timeout: int = 30) -> dict:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(OPENNEURO_GRAPHQL, data=payload, headers={"Content-Type": "application/json", "User-Agent": "fmri_multiverse/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    if "errors" in data:
        raise RuntimeError(f"OpenNeuro GraphQL error: {data['errors']}")
    return data["data"]


def latest_snapshot_tag(dataset_id: str) -> str:
    """Tag of the latest snapshot (e.g. ``'1.0.5'``)."""
    q = "query($id: ID!) { dataset(id: $id) { latestSnapshot { tag } } }"
    return _graphql(q, {"id": dataset_id})["dataset"]["latestSnapshot"]["tag"]


def list_snapshot_files(dataset_id: str, tag: Optional[str] = None, tree: Optional[str] = None) -> List[dict]:
    """List files (one directory level) of a snapshot via GraphQL.

    Returns dicts with ``filename, size, directory, id, urls``. Pass the
    ``id`` of a directory entry as ``tree`` to descend.
    """
    tag = tag or latest_snapshot_tag(dataset_id)
    q = (
        "query($id: ID!, $tag: String!, $tree: ID) { snapshot(datasetId: $id, tag: $tag) "
        "{ files(tree: $tree) { id filename size directory urls } } }"
    )
    return _graphql(q, {"id": dataset_id, "tag": tag, "tree": tree})["snapshot"]["files"]


def has_fmriprep_derivatives(dataset_id: str, tag: Optional[str] = None) -> Optional[bool]:
    """True/False if the snapshot has a ``derivatives/fmriprep`` directory; None on network failure."""
    try:
        top = list_snapshot_files(dataset_id, tag)
        deriv = [f for f in top if f["filename"] == "derivatives" and f.get("directory")]
        if not deriv:
            return False
        sub = list_snapshot_files(dataset_id, tag, tree=deriv[0]["id"])
        return any(f["filename"].startswith("fmriprep") for f in sub)
    except Exception:  # noqa: BLE001 - network problems are expected offline
        return None


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def s3_uri(dataset_id: str) -> str:
    return f"s3://{OPENNEURO_BUCKET}/{dataset_id}/"


def download(
    dataset_id: str,
    out_dir: Path,
    include: Sequence[str] = ("participants.tsv", "*events.tsv", "derivatives/fmriprep/*desc-confounds_*.tsv"),
    tag: Optional[str] = None,
    dry_run: bool = False,
) -> List[str]:
    """Download selected files with ``openneuro-py`` (preferred) or the AWS CLI.

    Returns the command that was (or would be) executed. ``include`` uses
    openneuro-py glob semantics; for the AWS CLI route the globs are turned
    into ``--include`` filters after ``--exclude "*"``.
    """
    out_dir = Path(out_dir) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("openneuro-py"):
        cmd = ["openneuro-py", "download", "--dataset", dataset_id, "--target-dir", str(out_dir)]
        if tag:
            cmd += ["--tag", tag]
        for pat in include:
            cmd += ["--include", pat]
    elif shutil.which("aws"):
        cmd = ["aws", "s3", "sync", "--no-sign-request", s3_uri(dataset_id), str(out_dir), "--exclude", "*"]
        for pat in include:
            cmd += ["--include", pat if pat.startswith("*") else f"*{pat}"]
    else:
        raise RuntimeError("Install openneuro-py (pip install openneuro-py) or the AWS CLI.")
    if not dry_run:
        subprocess.run(cmd, check=True)
    return cmd


def datalad_install(dataset_id: str, out_dir: Path, get: Sequence[str] = (), dry_run: bool = False) -> List[List[str]]:
    """``datalad install`` the OpenNeuroDatasets mirror and ``datalad get`` selected paths."""
    out_dir = Path(out_dir) / dataset_id
    cmds = [["datalad", "install", "-s", f"https://github.com/OpenNeuroDatasets/{dataset_id}.git", str(out_dir)]]
    for path in get:
        cmds.append(["datalad", "get", "-d", str(out_dir), str(out_dir / path)])
    if not dry_run:
        for c in cmds:
            subprocess.run(c, check=True)
    return cmds


def list_s3_prefix(dataset_id: str, prefix: str = "derivatives/fmriprep/", max_keys: int = 1000) -> List[str]:
    """Anonymous listing of the public bucket (boto3 required)."""
    try:
        import boto3  # type: ignore
        from botocore import UNSIGNED  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install boto3") from exc
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    keys: List[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=OPENNEURO_BUCKET, Prefix=f"{dataset_id}/{prefix}"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= max_keys:
                return keys
    return keys


# --------------------------------------------------------------------------- #
# Local discovery
# --------------------------------------------------------------------------- #
@dataclass
class RunFiles:
    subject: str
    task: str
    run: Optional[str]
    bold: Optional[Path]
    confounds: Optional[Path]
    events: Optional[Path]
    mask: Optional[Path] = None


def find_runs(dataset_root: Path, task: str, space: str = "MNI152NLin2009cAsym") -> List[RunFiles]:
    """Pair fMRIPrep preprocessed BOLD, confounds and raw events for a task."""
    root = Path(dataset_root)
    deriv = root / "derivatives" / "fmriprep"
    if not deriv.exists():
        candidates = list((root / "derivatives").glob("fmriprep*")) if (root / "derivatives").exists() else []
        deriv = candidates[0] if candidates else deriv
    runs: List[RunFiles] = []
    for conf in sorted(deriv.glob(f"sub-*/func/*task-{task}*desc-confounds_*.tsv")) + sorted(deriv.glob(f"sub-*/ses-*/func/*task-{task}*desc-confounds_*.tsv")):
        stem = conf.name.split("_desc-confounds")[0]
        sub = stem.split("_")[0]
        run = next((p.split("-")[1] for p in stem.split("_") if p.startswith("run-")), None)
        bold = next(iter(conf.parent.glob(f"{stem}_space-{space}_desc-preproc_bold.nii.gz")), None)
        mask = next(iter(conf.parent.glob(f"{stem}_space-{space}_desc-brain_mask.nii.gz")), None)
        ev_dir = root / sub / "func" if (root / sub / "func").exists() else root / sub
        events = next(iter(ev_dir.glob(f"{stem}_events.tsv")), None) if ev_dir.exists() else None
        runs.append(RunFiles(sub, task, run, bold, conf, events, mask))
    return runs


def dataset_characteristics(dataset_root: Path, task: str) -> Dict[str, object]:
    """Acquisition/design features used by the fragility model.

    Reads BIDS JSON sidecars (TR, multiband, field strength, voxel size) and
    events (design type, number of trials) without loading images.
    """
    root = Path(dataset_root)
    out: Dict[str, object] = {"dataset": root.name, "task": task}
    sidecars = list(root.glob(f"task-{task}_bold.json")) + list(root.glob(f"sub-*/func/*task-{task}*_bold.json"))
    if sidecars:
        meta = json.loads(sidecars[0].read_text())
        out["tr"] = meta.get("RepetitionTime")
        out["multiband"] = meta.get("MultibandAccelerationFactor")
        out["field_strength"] = meta.get("MagneticFieldStrength")
        out["echo_time"] = meta.get("EchoTime")
        out["manufacturer"] = meta.get("Manufacturer")
    events = list(root.glob(f"sub-*/func/*task-{task}*_events.tsv"))
    out["n_subjects_with_events"] = len({e.name.split("_")[0] for e in events})
    if events:
        try:
            import pandas as pd

            ev = pd.read_csv(events[0], sep="\t")
            out["n_events_per_run"] = int(len(ev))
            out["median_duration"] = float(ev["duration"].median()) if "duration" in ev else None
            out["design"] = "block" if out.get("median_duration") and out["median_duration"] > 10 else "event"
            out["n_conditions"] = int(ev["trial_type"].nunique()) if "trial_type" in ev else None
        except Exception:  # noqa: BLE001
            pass
    return out
