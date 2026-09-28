"""Claim registry: schema, validation, practice survey and the null-robustness index.

A *claim* is one published scalar association between two cortical maps. The registry CSV
(``data/claims/claims.csv``) has one row per claim with the columns in ``CLAIM_COLUMNS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

CLAIM_COLUMNS: List[str] = [
    "claim_id",  # short unique id, e.g. "hansen2022_5ht1a_thickness"
    "doi",
    "year",
    "map_a",  # "neuromaps:<source>/<desc>/<space>/<den>", "neurovault:<image_id>", or a file path
    "map_b",
    "space",  # fsaverage | fsLR | MNI152
    "parcellation",  # e.g. schaefer, desikan, glasser, vertex
    "n_parcels",  # integer, or number of vertices for vertex-wise claims
    "hemispheres",  # both | left | right
    "reported_r",
    "reported_p",
    "reported_stat",  # pearson | spearman
    "null_method",  # none | spin_vertex | spin_parcel | brainsmash | moran | eigenstrapping | other
    "n_perm",
    "sa_check_reported",  # 1 if the paper reports surrogate-vs-original autocorrelation agreement
    "notes",
]

NULL_METHODS = {"none", "spin_vertex", "spin_parcel", "brainsmash", "moran", "eigenstrapping", "other"}
SPACES = {"fsaverage", "fsLR", "MNI152"}
HEMIS = {"both", "left", "right"}


@dataclass
class Claim:
    """One registered claim (typed view of a registry row)."""

    claim_id: str
    doi: str
    year: int
    map_a: str
    map_b: str
    space: str
    parcellation: str
    n_parcels: int
    hemispheres: str
    reported_r: float
    reported_p: float
    reported_stat: str = "pearson"
    null_method: str = "spin_parcel"
    n_perm: Optional[int] = None
    sa_check_reported: int = 0
    notes: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "Claim":
        n_perm = row.get("n_perm")
        n_perm = None if n_perm in (None, "") or (isinstance(n_perm, float) and np.isnan(n_perm)) else int(n_perm)
        return cls(
            claim_id=str(row["claim_id"]),
            doi=str(row["doi"]),
            year=int(row["year"]),
            map_a=str(row["map_a"]),
            map_b=str(row["map_b"]),
            space=str(row["space"]),
            parcellation=str(row["parcellation"]),
            n_parcels=int(row["n_parcels"]),
            hemispheres=str(row.get("hemispheres", "both")),
            reported_r=float(row["reported_r"]),
            reported_p=float(row["reported_p"]),
            reported_stat=str(row.get("reported_stat", "pearson")),
            null_method=str(row.get("null_method", "other")),
            n_perm=n_perm,
            sa_check_reported=int(row.get("sa_check_reported", 0) or 0),
            notes=str(row.get("notes", "") or ""),
        )

    @property
    def seed(self) -> int:
        """Deterministic per-claim seed (stable across Python sessions, unlike ``hash``)."""
        h = 0
        for ch in self.claim_id:
            h = (h * 131 + ord(ch)) % (2**31 - 1)
        return h


def validate_claims(df: pd.DataFrame) -> List[str]:
    """Return a list of human-readable problems (empty list means valid)."""
    problems: List[str] = []
    missing = [c for c in CLAIM_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems
    if df["claim_id"].duplicated().any():
        problems.append("duplicate claim_id values")
    for i, row in df.iterrows():
        cid = row["claim_id"]
        if row["null_method"] not in NULL_METHODS:
            problems.append(f"{cid}: unknown null_method {row['null_method']!r}")
        if row["space"] not in SPACES:
            problems.append(f"{cid}: unknown space {row['space']!r}")
        if row["hemispheres"] not in HEMIS:
            problems.append(f"{cid}: unknown hemispheres {row['hemispheres']!r}")
        try:
            r = float(row["reported_r"])
            if not -1 <= r <= 1:
                problems.append(f"{cid}: reported_r out of range")
        except (TypeError, ValueError):
            problems.append(f"{cid}: reported_r not numeric")
        try:
            p = float(row["reported_p"])
            if not 0 <= p <= 1:
                problems.append(f"{cid}: reported_p out of range")
        except (TypeError, ValueError):
            problems.append(f"{cid}: reported_p not numeric")
        try:
            if int(row["n_parcels"]) < 2:
                problems.append(f"{cid}: n_parcels < 2")
        except (TypeError, ValueError):
            problems.append(f"{cid}: n_parcels not integer")
    return problems


def load_claims(path: str | Path, strict: bool = True) -> pd.DataFrame:
    """Load and validate the registry CSV. With ``strict`` any problem raises ``ValueError``."""
    df = pd.read_csv(path)
    problems = validate_claims(df)
    if problems and strict:
        raise ValueError("invalid claims registry:\n  " + "\n  ".join(problems))
    return df


def claims_from_frame(df: pd.DataFrame) -> List[Claim]:
    """Typed claims from a validated DataFrame."""
    return [Claim.from_row(r) for r in df.to_dict("records")]


def summarize_claims(df: pd.DataFrame) -> Dict[str, object]:
    """Practice-survey statistics: null usage, permutation counts, parcellation, reporting habits."""
    out: Dict[str, object] = {
        "n_claims": int(len(df)),
        "by_null_method": df["null_method"].value_counts().to_dict(),
        "by_space": df["space"].value_counts().to_dict(),
        "by_parcellation": df["parcellation"].value_counts().to_dict(),
        "median_n_parcels": float(df["n_parcels"].median()),
        "n_perm_reported_frac": float(df["n_perm"].notna().mean()),
        "median_n_perm": float(df["n_perm"].dropna().median()) if df["n_perm"].notna().any() else float("nan"),
        "sa_check_reported_frac": float(pd.to_numeric(df["sa_check_reported"], errors="coerce").fillna(0).mean()),
        "reported_p_bins": pd.cut(
            pd.to_numeric(df["reported_p"], errors="coerce"),
            bins=[-1e-12, 0.001, 0.01, 0.05, 1.0],
            labels=["<0.001", "0.001-0.01", "0.01-0.05", ">0.05"],
        )
        .value_counts()
        .to_dict(),
    }
    return out


def null_robustness_index(pvalues: Mapping[str, float], alpha: float = 0.05, exclude: Iterable[str] = ("naive",)) -> float:
    """Fraction of (spatial) null families under which the claim is significant at ``alpha``."""
    keys = [k for k in pvalues if k not in set(exclude)]
    if not keys:
        return float("nan")
    return float(np.mean([pvalues[k] < alpha for k in keys]))


def classify_claim(reported_p: float, audited_p: Mapping[str, float], alpha: float = 0.05, exclude: Iterable[str] = ("naive",)) -> str:
    """'confirmed' if significant under every spatial null, 'flipped' if under none, else 'fragile'.

    Claims reported as non-significant are labelled 'reported_null' and not classified further.
    """
    if reported_p >= alpha:
        return "reported_null"
    idx = null_robustness_index(audited_p, alpha=alpha, exclude=exclude)
    if np.isnan(idx):
        return "unclear"
    if idx == 1.0:
        return "confirmed"
    if idx == 0.0:
        return "flipped"
    return "fragile"
