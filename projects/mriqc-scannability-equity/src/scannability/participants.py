"""Harmonize `participants.tsv` files from many BIDS datasets.

Handles the common conventions for age (numbers, ranges ``20-25``, ``89+``,
months), sex (``M``/``F``/``male``/``female``/``1``/``2``) and group / diagnosis
columns with a keyword dictionary mapping to coarse classes.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

AGE_COLS = ("age", "Age", "AGE", "age_years", "age_at_scan", "age_at_baseline", "Age_at_scan", "ageAtScan")
SEX_COLS = ("sex", "Sex", "SEX", "gender", "Gender", "biological_sex")
GROUP_COLS = ("group", "Group", "diagnosis", "Diagnosis", "dx", "DX", "condition", "Condition",
              "participant_type", "subject_type", "patient", "status", "cohort", "clinical_group")

COARSE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "control": ("control", "healthy", "hc", "ctrl", "typical", "td", "normal", "nc", "cn", "hv"),
    "neurological": ("stroke", "epilep", "parkinson", "pd", "dementia", "alzheim", "ad", "mci", "ms",
                     "sclerosis", "tumor", "tumour", "glioma", "tbi", "brain injury", "aphasia", "als",
                     "huntington", "migraine", "lesion", "dystonia", "tremor"),
    "psychiatric": ("schizo", "schz", "scz", "sz", "bipolar", "bd", "depress", "mdd", "anxiety", "ptsd", "ocd",
                    "psychosis", "psychotic", "addiction", "alcohol", "cocaine", "cannabis", "eating",
                    "anorexia", "borderline", "personality"),
    "neurodevelopmental": ("autism", "asd", "adhd", "dyslexia", "developmental", "tourette", "preterm",
                           "fragile", "down syndrome", "intellectual"),
}


def parse_age(value: object) -> float:
    """Parse an age cell into years (NaN if unparsable).

    Handles ``'23'``, ``23.5``, ``'20-25'`` (midpoint), ``'89+'``, ``'18 months'`` and ``'n/a'``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().lower()
    if s in {"", "n/a", "na", "nan", "none", "unknown", "-", "?"}:
        return np.nan
    months = re.match(r"^(\d+(?:\.\d+)?)\s*(m|mo|mos|month|months)$", s)
    if months:
        return float(months.group(1)) / 12.0
    rng = re.match(r"^(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)$", s)
    if rng:
        return (float(rng.group(1)) + float(rng.group(2))) / 2.0
    plus = re.match(r"^(\d+(?:\.\d+)?)\s*\+$", s)
    if plus:
        return float(plus.group(1))
    num = re.match(r"^[^\d]*(\d+(?:\.\d+)?)", s)
    if num:
        v = float(num.group(1))
        return v if 0 <= v <= 120 else np.nan
    return np.nan


def parse_sex(value: object) -> str:
    """Return 'F', 'M' or 'U' (unknown)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "U"
    s = str(value).strip().lower()
    if s in {"f", "female", "woman", "women", "2", "fem"}:
        return "F"
    if s in {"m", "male", "man", "men", "1"}:
        return "M"
    return "U"


def coarse_group(value: object) -> str:
    """Map a free-text group/diagnosis label to a coarse class."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    s = re.sub(r"[_\-/]", " ", str(value).strip().lower())
    if s in {"", "n/a", "na", "nan", "none"}:
        return "unknown"
    tokens = set(s.split())
    for cls in ("control", "neurological", "psychiatric", "neurodevelopmental"):
        for kw in COARSE_KEYWORDS[cls]:
            if " " in kw:
                if kw in s:
                    return cls
            elif kw in tokens or (len(kw) > 3 and kw in s):
                return cls
    if any(t in tokens for t in ("patient", "patients", "case", "clinical", "disorder")):
        return "other_clinical"
    return "unknown"


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def harmonize_participants(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Return a tidy table: dataset, participant_id, age, sex, group_raw, group."""
    pid_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    out = pd.DataFrame({"dataset": dataset_id, "participant_id": df[pid_col].astype(str)})
    a, s, g = _first_present(df, AGE_COLS), _first_present(df, SEX_COLS), _first_present(df, GROUP_COLS)
    out["age"] = df[a].map(parse_age) if a else np.nan
    out["sex"] = df[s].map(parse_sex) if s else "U"
    out["group_raw"] = df[g].astype(str) if g else np.nan
    out["group"] = df[g].map(coarse_group) if g else "unknown"
    # single-group datasets without a group column are usually controls; flag rather than assume
    out["group_source"] = "column" if g else "missing"
    return out


def load_corpus_participants(root: Path | str) -> pd.DataFrame:
    """Read every ``<root>/<dsid>/participants.tsv`` and harmonize them."""
    root = Path(root)
    frames = []
    for tsv in sorted(root.glob("*/participants.tsv")):
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str, na_values=["n/a", "N/A", "NA", ""])
        except Exception:  # noqa: BLE001
            continue
        frames.append(harmonize_participants(df, tsv.parent.name))
    if not frames:
        return pd.DataFrame(columns=["dataset", "participant_id", "age", "sex", "group_raw", "group", "group_source"])
    return pd.concat(frames, ignore_index=True)


def mapping_review_table(corpus: pd.DataFrame) -> pd.DataFrame:
    """Unique raw group labels with their coarse mapping and counts (for manual review)."""
    t = corpus.dropna(subset=["group_raw"]).groupby(["group_raw", "group"]).size().reset_index(name="n")
    return t.sort_values("n", ascending=False).reset_index(drop=True)
