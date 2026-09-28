"""MRIQC group-table loading and exclusion rules.

MRIQC writes ``group_T1w.tsv`` and ``group_bold.tsv`` with one row per image
(``bids_name`` like ``sub-01_ses-1_T1w`` or ``sub-01_task-rest_bold``) and one
column per IQM. Rules are expressed as a list of ``(iqm, direction, value, kind)``
where ``kind`` is ``'abs'`` (absolute threshold) or ``'pct'`` (dataset-relative
percentile); a subject is excluded if any rule fires.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["gt", "lt"]

T1W_IQMS = ("cjv", "cnr", "efc", "snr_total", "snrd_total", "fber", "qi_1", "qi_2", "wm2max", "inu_med",
            "summary_bg_mean", "rpve_gm", "rpve_wm")
BOLD_IQMS = ("fd_mean", "fd_perc", "fd_num", "tsnr", "dvars_std", "efc", "gsr_x", "gsr_y", "aor", "aqi", "snr", "fwhm_avg")


@dataclass(frozen=True)
class Rule:
    iqm: str
    direction: Direction   # 'gt': exclude if value > threshold; 'lt': exclude if value < threshold
    value: float
    kind: Literal["abs", "pct"] = "abs"

    def fires(self, x: pd.Series, dataset: pd.Series | None = None) -> pd.Series:
        if self.kind == "abs":
            thr = pd.Series(self.value, index=x.index, dtype=float)
        else:
            if dataset is None:
                q = np.nanpercentile(x, self.value)
                thr = pd.Series(q, index=x.index, dtype=float)
            else:
                thr = x.groupby(dataset).transform(lambda s: np.nanpercentile(s, self.value) if s.notna().any() else np.nan)
        out = (x > thr) if self.direction == "gt" else (x < thr)
        return out.fillna(False)


@dataclass
class RuleSet:
    name: str
    rules: list[Rule] = field(default_factory=list)

    def apply(self, df: pd.DataFrame, dataset_col: str = "dataset") -> pd.Series:
        fired = pd.Series(False, index=df.index)
        for r in self.rules:
            if r.iqm in df.columns:
                fired |= r.fires(df[r.iqm], df[dataset_col] if dataset_col in df.columns else None)
        return fired.astype(int)


def literature_rulesets() -> dict[str, RuleSet]:
    """Exclusion rules commonly used in the literature (thresholds are parameters, not truths)."""
    return {
        "bold_fd_strict": RuleSet("bold_fd_strict", [Rule("fd_mean", "gt", 0.2), Rule("fd_perc", "gt", 20.0)]),
        "bold_fd_moderate": RuleSet("bold_fd_moderate", [Rule("fd_mean", "gt", 0.3), Rule("fd_perc", "gt", 30.0)]),
        "bold_fd_lenient": RuleSet("bold_fd_lenient", [Rule("fd_mean", "gt", 0.5)]),
        "t1w_relative10": RuleSet("t1w_relative10", [Rule("cjv", "gt", 90.0, "pct"), Rule("cnr", "lt", 10.0, "pct"),
                                                     Rule("efc", "gt", 90.0, "pct")]),
        "t1w_absolute": RuleSet("t1w_absolute", [Rule("cjv", "gt", 0.55), Rule("cnr", "lt", 2.5), Rule("efc", "gt", 0.6)]),
        "bold_relative10": RuleSet("bold_relative10", [Rule("fd_mean", "gt", 90.0, "pct"), Rule("tsnr", "lt", 10.0, "pct")]),
    }


def fd_threshold_sweep(thresholds=(0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)) -> list[RuleSet]:
    """One rule set per mean-FD threshold, for exclusion-elasticity analyses."""
    return [RuleSet(f"fd_mean_gt_{str(t).replace('.', 'p')}", [Rule("fd_mean", "gt", float(t))]) for t in thresholds]


_BIDS_RE = re.compile(r"(sub-[A-Za-z0-9]+)(?:_ses-([A-Za-z0-9]+))?")


def parse_bids_name(name: str) -> tuple[str, str | None]:
    m = _BIDS_RE.search(str(name))
    if not m:
        return str(name), None
    return m.group(1), m.group(2)


def load_group_table(path: Path | str, dataset_id: str, modality: Literal["T1w", "bold"]) -> pd.DataFrame:
    """Load a MRIQC group TSV and add dataset/participant/session/modality columns."""
    df = pd.read_csv(path, sep="\t")
    subs = df["bids_name"].map(parse_bids_name)
    df["participant_id"] = [s[0] for s in subs]
    df["session"] = [s[1] for s in subs]
    df["dataset"] = dataset_id
    df["modality"] = modality
    return df


def load_corpus_iqms(mriqc_root: Path | str) -> pd.DataFrame:
    """Concatenate ``<root>/<dsid>/group_{T1w,bold}.tsv`` for all datasets."""
    root = Path(mriqc_root)
    frames = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mod in ("T1w", "bold"):
            p = ds_dir / f"group_{mod}.tsv"
            if p.exists():
                frames.append(load_group_table(p, ds_dir.name, mod))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_participant(iqms: pd.DataFrame, modality: str, agg: str = "worst") -> pd.DataFrame:
    """Collapse runs/sessions to one row per participant per dataset.

    ``agg='worst'`` takes the worst run per IQM in the direction that matters for the
    common rules (max for fd/cjv/efc/dvars, min for cnr/snr/tsnr); ``'mean'`` averages.
    """
    sub = iqms[iqms["modality"] == modality]
    cols = [c for c in (T1W_IQMS if modality == "T1w" else BOLD_IQMS) if c in sub.columns]
    if agg == "mean":
        return sub.groupby(["dataset", "participant_id"])[cols].mean().reset_index()
    worse_high = {"fd_mean", "fd_perc", "fd_num", "cjv", "efc", "dvars_std", "gsr_x", "gsr_y", "aor", "aqi", "inu_med", "qi_1", "qi_2"}
    aggs = {c: ("max" if c in worse_high else "min") for c in cols}
    return sub.groupby(["dataset", "participant_id"]).agg(aggs).reset_index()


def zscore_within_dataset(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_z"] = out.groupby("dataset")[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=1) or 1.0))
    return out


def merge_iqms_participants(iqms_pp: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Inner-join per-participant IQMs with harmonized demographics on (dataset, participant_id)."""
    p = participants.copy()
    p["participant_id"] = p["participant_id"].where(p["participant_id"].str.startswith("sub-"), "sub-" + p["participant_id"])
    return iqms_pp.merge(p, on=["dataset", "participant_id"], how="inner")


def apply_rulesets(df: pd.DataFrame, rulesets: dict[str, RuleSet] | list[RuleSet]) -> pd.DataFrame:
    out = df.copy()
    items = rulesets.values() if isinstance(rulesets, dict) else rulesets
    for rs in items:
        out[f"excl_{rs.name}"] = rs.apply(out)
    return out
