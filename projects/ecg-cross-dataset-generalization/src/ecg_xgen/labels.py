"""SNOMED-CT label harmoniser with explicit mapping policies.

The harmonised target set is a small, clinically interpretable vocabulary that
every source can express.  Three *policies* turn source labels into it; the
choice of policy is an experimental factor in the transfer grid, not a
preprocessing detail:

``strict``      one-to-one concept matches only (e.g. RBBB <- 59118001 only)
``lenient``     Challenge-2021 equivalence groups and clinically-merged subtypes
                (CRBBB/IRBBB -> RBBB, SVPB -> PAC, VPB -> PVC)
``superclass``  coarse groups (RHYTHM / CONDUCTION / MORPHOLOGY / NORMAL)

SNOMED codes below follow the PhysioNet/CinC Challenge 2021 ``dx_mapping_scored.csv``
and ``dx_mapping_unscored.csv``; verify against those files after download
(``ecg_xgen.labels.check_against_challenge_table``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

HARMONISED_CLASSES: tuple[str, ...] = (
    "NORM", "AF", "AFL", "AVB1", "RBBB", "LBBB", "PVC", "PAC", "SB", "STACH", "LQT", "LVH",
)
SUPERCLASSES: tuple[str, ...] = ("NORMAL", "RHYTHM", "CONDUCTION", "MORPHOLOGY")

# SNOMED-CT code -> harmonised class, STRICT policy.
SNOMED_STRICT: dict[str, str] = {
    "426783006": "NORM",   # sinus rhythm
    "164889003": "AF",     # atrial fibrillation
    "164890007": "AFL",    # atrial flutter
    "270492004": "AVB1",   # 1st degree AV block
    "59118001": "RBBB",    # right bundle branch block
    "164909002": "LBBB",   # left bundle branch block
    "427172004": "PVC",    # premature ventricular contractions
    "284470004": "PAC",    # premature atrial contraction
    "426177001": "SB",     # sinus bradycardia
    "427084000": "STACH",  # sinus tachycardia
    "111975006": "LQT",    # prolonged QT interval
    "164873001": "LVH",    # left ventricular hypertrophy
}

# Additional codes folded in under the LENIENT policy.
SNOMED_LENIENT_EXTRA: dict[str, str] = {
    "713427006": "RBBB",   # complete RBBB (Challenge: equivalent to RBBB)
    "713426002": "RBBB",   # incomplete RBBB (merged clinically)
    "733534002": "LBBB",   # complete LBBB
    "63593006": "PAC",     # supraventricular premature beats (equiv. PAC)
    "17338001": "PVC",     # ventricular premature beats (equiv. PVC)
    "426627000": "SB",     # bradycardia (unspecified)
}

SUPERCLASS_OF: dict[str, str] = {
    "NORM": "NORMAL",
    "AF": "RHYTHM", "AFL": "RHYTHM", "SB": "RHYTHM", "STACH": "RHYTHM", "PAC": "RHYTHM", "PVC": "RHYTHM",
    "AVB1": "CONDUCTION", "RBBB": "CONDUCTION", "LBBB": "CONDUCTION",
    "LQT": "MORPHOLOGY", "LVH": "MORPHOLOGY",
}

# PTB-XL SCP statement -> harmonised class (subset relevant to the target set; see scp_statements.csv).
PTBXL_SCP: dict[str, str] = {
    "NORM": "NORM", "SR": "NORM",
    "AFIB": "AF", "AFLT": "AFL", "1AVB": "AVB1",
    "CRBBB": "RBBB", "IRBBB": "RBBB", "CLBBB": "LBBB", "ILBBB": "LBBB",
    "PVC": "PVC", "PAC": "PAC", "SBRAD": "SB", "STACH": "STACH", "LNGQT": "LQT", "LVH": "LVH",
}
PTBXL_STRICT_ONLY: set[str] = {"CRBBB", "CLBBB", "SR", "NORM"}  # IRBBB/ILBBB only under lenient
PTBXL_LENIENT_ONLY: set[str] = {"IRBBB", "ILBBB"}

# CODE-15% column -> harmonised class (labels come from the Telehealth Network of Minas Gerais reports).
CODE15_COLUMNS: dict[str, str] = {
    "1dAVb": "AVB1", "RBBB": "RBBB", "LBBB": "LBBB", "SB": "SB", "ST": "STACH", "AF": "AF", "normal_ecg": "NORM",
}


@dataclass(frozen=True)
class MappingPolicy:
    """A named mapping from source vocabularies to the harmonised vocabulary."""

    name: str
    snomed: Mapping[str, str]
    ptbxl: Mapping[str, str]
    code15: Mapping[str, str]
    classes: tuple[str, ...] = HARMONISED_CLASSES

    def index(self, cls: str) -> int:
        return self.classes.index(cls)


def get_policy(name: str) -> MappingPolicy:
    """Return one of ``strict``, ``lenient`` or ``superclass``."""
    if name == "strict":
        ptb = {k: v for k, v in PTBXL_SCP.items() if k not in PTBXL_LENIENT_ONLY}
        return MappingPolicy("strict", dict(SNOMED_STRICT), ptb, dict(CODE15_COLUMNS))
    if name == "lenient":
        sn = {**SNOMED_STRICT, **SNOMED_LENIENT_EXTRA}
        return MappingPolicy("lenient", sn, dict(PTBXL_SCP), dict(CODE15_COLUMNS))
    if name == "superclass":
        sn = {k: SUPERCLASS_OF[v] for k, v in {**SNOMED_STRICT, **SNOMED_LENIENT_EXTRA}.items()}
        ptb = {k: SUPERCLASS_OF[v] for k, v in PTBXL_SCP.items()}
        c15 = {k: SUPERCLASS_OF[v] for k, v in CODE15_COLUMNS.items()}
        return MappingPolicy("superclass", sn, ptb, c15, classes=SUPERCLASSES)
    raise ValueError(f"unknown policy {name!r}")


def harmonize(codes: Iterable[str], policy: MappingPolicy, source: str = "snomed") -> np.ndarray:
    """Multi-hot vector over ``policy.classes`` for one record's raw labels.

    ``source`` selects the vocabulary: ``"snomed"`` (Challenge / Chapman /
    Ningbo / Georgia / CPSC headers), ``"ptbxl"`` (SCP codes) or ``"code15"``
    (column names whose value is 1).
    """
    table = {"snomed": policy.snomed, "ptbxl": policy.ptbxl, "code15": policy.code15}[source]
    y = np.zeros(len(policy.classes), dtype=np.int8)
    for c in codes:
        c = str(c).strip()
        if c in table:
            y[policy.classes.index(table[c])] = 1
    # a record with any abnormality is not NORM even if a "sinus rhythm" code is present
    if "NORM" in policy.classes:
        i = policy.classes.index("NORM")
        if y[i] and y.sum() > 1:
            y[i] = 0
    if "NORMAL" in policy.classes:
        i = policy.classes.index("NORMAL")
        if y[i] and y.sum() > 1:
            y[i] = 0
    return y


def harmonize_many(records: Sequence[Iterable[str]], policy: MappingPolicy, source: str = "snomed") -> np.ndarray:
    """Stack :func:`harmonize` over records -> (n, n_classes) int8."""
    if len(records) == 0:
        return np.zeros((0, len(policy.classes)), dtype=np.int8)
    return np.stack([harmonize(r, policy, source) for r in records])


def code15_row_to_codes(row: Mapping[str, object]) -> list[str]:
    """Convert one ``exams.csv`` row (dict-like) into a list of positive column names."""
    return [col for col in CODE15_COLUMNS if col in row and bool(row[col])]


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two binary label vectors (agreement between mapping policies)."""
    a = np.asarray(a).astype(int).ravel()
    b = np.asarray(b).astype(int).ravel()
    if a.size == 0:
        return float("nan")
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def policy_agreement(records: Sequence[Iterable[str]], source: str, p1: MappingPolicy, p2: MappingPolicy) -> dict[str, float]:
    """Per-class Cohen's kappa between two policies that share the same class vocabulary."""
    if p1.classes != p2.classes:
        raise ValueError("policies must share the class vocabulary")
    y1, y2 = harmonize_many(records, p1, source), harmonize_many(records, p2, source)
    return {c: cohen_kappa(y1[:, i], y2[:, i]) for i, c in enumerate(p1.classes)}


def agreement_mask(records: Sequence[Iterable[str]], source: str, p1: MappingPolicy, p2: MappingPolicy) -> np.ndarray:
    """Boolean mask of records whose harmonised vector is identical under two policies (label-noise-aware eval)."""
    y1, y2 = harmonize_many(records, p1, source), harmonize_many(records, p2, source)
    return np.all(y1 == y2, axis=1)


def check_against_challenge_table(dx_mapping_csv: str) -> list[str]:
    """Return codes in our tables that are absent from the Challenge ``dx_mapping_*.csv`` (sanity check)."""
    import csv

    with open(dx_mapping_csv, newline="") as f:
        known = {row["SNOMEDCTCode"].strip() for row in csv.DictReader(f)}
    ours = set(SNOMED_STRICT) | set(SNOMED_LENIENT_EXTRA)
    return sorted(ours - known)
