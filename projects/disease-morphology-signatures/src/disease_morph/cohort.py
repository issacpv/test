"""Condition taxonomy and archive-matched cohort construction for NeuroMorpho records.

NeuroMorpho's ``experiment_condition`` is a curated but free-form vocabulary
("Control", "APP/PS1", "aged", "pilocarpine model of epilepsy", ...). This module
maps every value to one of a small set of classes with a version-controlled keyword
table, then pairs non-control records with control records from the *same archive*
(and, when possible, the same species / region / cell class), because the
disease-vs-control contrast is only identified within a laboratory's protocol.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

CONDITION_CLASSES = ("control", "ad_model", "aging", "epilepsy", "other")

# Order matters: the first class whose pattern matches wins. Patterns are
# case-insensitive regular expressions applied to each condition string.
KEYWORDS: Dict[str, Sequence[str]] = {
    "control": [r"^control$", r"^normal$", r"^wild[- ]?type$", r"^wt$", r"^untreated$", r"^sham$",
                r"^vehicle$", r"^naive$", r"^healthy$"],
    "ad_model": [r"alzheimer", r"\bapp\b", r"app/ps1", r"appswe", r"ps1", r"psen1", r"3xtg", r"5xfad",
                 r"tg2576", r"\btau\b", r"\bps19\b", r"amyloid", r"a-?beta", r"\bhapp\b", r"\bapoe4?\b"],
    "aging": [r"\bag(ed|ing)\b", r"\bold\b", r"senescen", r"elderly", r"\b(1[8-9]|2[0-9]|3[0-9]) ?(mo|month)",
              r"\baged? [0-9]+ ?(y|yr|years)"],
    "epilepsy": [r"epilep", r"pilocarpin", r"kainat", r"kainic", r"status epilepticus", r"\bse\b",
                 r"kindl", r"seizure", r"pentylenetetrazol", r"\bptz\b", r"temporal lobe", r"\btle\b"],
}

_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in KEYWORDS.items()}


def map_condition(condition: object) -> str:
    """Map one condition value (string or list of strings) to a condition class.

    A record that carries both a disease keyword and "control" (e.g. a disease
    dataset's own controls, curated as ["Control", "APP/PS1 littermate"]) is
    classified as control only if *no* disease pattern matches any element.
    """
    if condition is None:
        return "other"
    values: List[str] = list(condition) if isinstance(condition, (list, tuple)) else [str(condition)]
    values = [v for v in values if v is not None and str(v).strip()]
    if not values:
        return "other"
    text_blob = " | ".join(map(str, values))
    for cls in ("ad_model", "aging", "epilepsy"):
        if any(p.search(text_blob) for p in _COMPILED[cls]):
            return cls
    if any(p.search(v.strip()) for v in map(str, values) for p in _COMPILED["control"]):
        return "control"
    return "other"


def _first(x: object) -> str:
    if isinstance(x, (list, tuple)):
        return str(x[0]) if x else ""
    return "" if x is None else str(x)


def _cell_class(cell_type: object) -> str:
    """Coarse cell class from NeuroMorpho's cell_type list."""
    t = " ".join(map(str, cell_type)) if isinstance(cell_type, (list, tuple)) else str(cell_type or "")
    t = t.lower()
    if "granule" in t:
        return "granule"
    if "pyramidal" in t or "principal" in t:
        return "pyramidal"
    if "interneuron" in t or "basket" in t or "chandelier" in t or "martinotti" in t or "gaba" in t:
        return "interneuron"
    if "purkinje" in t:
        return "purkinje"
    if "medium spiny" in t or "msn" in t:
        return "medium_spiny"
    if "glia" in t or "astro" in t or "microglia" in t:
        return "glia"
    return "other"


def records_to_frame(records: Iterable[Dict]) -> pd.DataFrame:
    """Flatten NeuroMorpho JSON records into the columns this project uses."""
    rows = []
    for r in records:
        rows.append({
            "neuron_name": r.get("neuron_name"),
            "archive": r.get("archive"),
            "species": r.get("species"),
            "region_top": _first(r.get("brain_region")),
            "cell_type_raw": r.get("cell_type"),
            "cell_class": _cell_class(r.get("cell_type")),
            "experiment_condition": r.get("experiment_condition"),
            "min_age": r.get("min_age"),
            "max_age": r.get("max_age"),
            "age_classification": r.get("age_classification"),
            "gender": r.get("gender"),
            "strain": r.get("strain"),
            "reconstruction_software": r.get("reconstruction_software"),
            "shrinkage_corrected": r.get("shrinkage_corrected"),
            "slicing_thickness": r.get("slicing_thickness"),
            "stain": r.get("stain"),
            "protocol": r.get("protocol"),
            "physical_integrity": r.get("physical_Integrity"),
            "structural_domains": r.get("structural_domains"),
            "reference_pmid": _first(r.get("reference_pmid")),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["condition_class"] = df["experiment_condition"].map(map_condition)
    return df


def dendrites_complete(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: dendrites reconstructed completely (NeuroMorpho integrity flag)."""
    integ = df["physical_integrity"].fillna("").astype(str).str.lower()
    dom = df["structural_domains"].fillna("").astype(str).str.lower()
    return integ.str.contains("dendrites complete") & dom.str.contains("dendrit")


def build_cohort(records: Iterable[Dict], min_per_group: int = 5,
                 match_on: Sequence[str] = ("species", "cell_class")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cohort table, condition-mapping audit).

    The cohort table has one row per record with ``condition_class``, an
    ``in_contrast`` flag (True when the record belongs to an archive x matching-key
    stratum that has at least ``min_per_group`` cases and controls) and a
    ``contrast_id`` string naming that stratum.
    """
    df = records_to_frame(records)
    if df.empty:
        return df, pd.DataFrame(columns=["experiment_condition", "condition_class", "n"])
    df = df[dendrites_complete(df)].copy()
    keys = ["archive", *match_on]
    df["stratum"] = df[keys].astype(str).agg("|".join, axis=1)
    df["in_contrast"] = False
    df["contrast_id"] = ""
    # A contrast stratum = archive x matching keys. Its controls are shared by every
    # disease class in the stratum that reaches the minimum group size.
    for stratum, g in df.groupby("stratum"):
        n_ctrl = int((g["condition_class"] == "control").sum())
        if n_ctrl < min_per_group:
            continue
        ok_classes = [cls for cls in ("ad_model", "aging", "epilepsy")
                      if int((g["condition_class"] == cls).sum()) >= min_per_group]
        if not ok_classes:
            continue
        idx = g.index[g["condition_class"].isin([*ok_classes, "control"])]
        df.loc[idx, "in_contrast"] = True
        df.loc[idx, "contrast_id"] = stratum
    audit = (df.assign(experiment_condition=df["experiment_condition"].astype(str))
               .groupby(["experiment_condition", "condition_class"]).size().rename("n").reset_index()
               .sort_values("n", ascending=False))
    return df, audit


def condition_report(cohort: pd.DataFrame) -> pd.DataFrame:
    """Archive x condition_class count table restricted to identifiable contrasts."""
    if cohort.empty:
        return pd.DataFrame()
    sub = cohort[cohort["in_contrast"]]
    return pd.crosstab(sub["archive"], sub["condition_class"])
