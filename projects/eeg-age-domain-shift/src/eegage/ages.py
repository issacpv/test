"""Age parsing, binning and age-distance utilities.

Sources of age per corpus:

* TUH corpora: the EDF *local patient identification* field (header bytes 8-88)
  contains a token ``Age:NN``; :func:`parse_edf_header_age` reads it without any
  EDF library.
* CHB-MIT: ``SUBJECT-INFO`` (case, gender, age).
* Siena: ``subject_info.csv`` (a column whose name contains ``age``).
* Helsinki neonates: age 0 by construction.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# (lower, upper, label), upper exclusive, years
AGE_BINS: List[Tuple[float, float, str]] = [
    (0.0, 1.0, "0-1"),
    (1.0, 6.0, "1-6"),
    (6.0, 12.0, "6-12"),
    (12.0, 18.0, "12-18"),
    (18.0, 40.0, "18-40"),
    (40.0, 65.0, "40-65"),
    (65.0, 120.0, "65+"),
]
BIN_LABELS = [b[2] for b in AGE_BINS]


def parse_tuh_patient_field(text: str) -> Optional[float]:
    """Extract an age in years from a TUH-style patient field (``... Age:47 ...``)."""
    m = re.search(r"Age:\s*(\d{1,3})", text)
    if not m:
        return None
    age = float(m.group(1))
    return age if 0 <= age <= 110 else None


def parse_edf_header_age(path: Path) -> Optional[float]:
    """Read the 80-byte local patient identification field of an EDF file and parse ``Age:NN``."""
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(88)
    except OSError:
        return None
    if len(hdr) < 88:
        return None
    return parse_tuh_patient_field(hdr[8:88].decode("ascii", errors="ignore"))


def parse_chbmit_subject_info(text: str) -> Dict[str, Dict[str, float]]:
    """Parse CHB-MIT ``SUBJECT-INFO`` (``chb01<TAB>F<TAB>11``) into {case: {"sex": .., "age": ..}}."""
    out: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        toks = re.split(r"[\t,;]+|\s{2,}", line.strip())
        toks = [t for t in toks if t]
        if len(toks) >= 3 and toks[0].lower().startswith("chb"):
            try:
                out[toks[0]] = {"sex": toks[1].upper()[:1], "age": float(toks[2])}
            except ValueError:
                continue
    return out


def parse_siena_subject_info(csv_text: str) -> Dict[str, float]:
    """Parse Siena ``subject_info.csv`` into {patient_id: age}; tolerant to column naming."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return {}
    cols = {c.lower().strip(): c for c in reader.fieldnames}
    id_col = next((cols[c] for c in cols if "patient" in c or c in ("id", "subject")), reader.fieldnames[0])
    age_col = next((cols[c] for c in cols if "age" in c and "stage" not in c), None)
    out: Dict[str, float] = {}
    if age_col is None:
        return out
    for row in reader:
        try:
            out[str(row[id_col]).strip()] = float(row[age_col])
        except (TypeError, ValueError):
            continue
    return out


def age_bin(age: Optional[float], bins: Sequence[Tuple[float, float, str]] = AGE_BINS) -> str:
    """Label of the bin containing ``age`` (years), or ``""`` if missing/out of range."""
    if age is None or not np.isfinite(age):
        return ""
    for lo, hi, lab in bins:
        if lo <= age < hi:
            return lab
    return ""


def bin_center(label: str, bins: Sequence[Tuple[float, float, str]] = AGE_BINS) -> float:
    """Geometric-ish centre of a bin (midpoint in log(age+1) space, back-transformed)."""
    for lo, hi, lab in bins:
        if lab == label:
            return float(np.expm1(0.5 * (np.log1p(lo) + np.log1p(min(hi, 100.0)))))
    raise KeyError(label)


def age_distance(a: float, b: float, kind: str = "log") -> float:
    """Distance between two ages: ``|log(a+1) - log(b+1)|`` (default) or ``|a-b|`` in years."""
    if kind == "log":
        return float(abs(np.log1p(a) - np.log1p(b)))
    return float(abs(a - b))


def bin_distance_matrix(labels: Sequence[str], kind: str = "log") -> pd.DataFrame:
    """Pairwise age distance between bin centres."""
    c = [bin_center(l) for l in labels]
    M = np.array([[age_distance(x, y, kind) for y in c] for x in c])
    return pd.DataFrame(M, index=list(labels), columns=list(labels))


def bin_counts(ages: Sequence[float], subject_ids: Sequence[str]) -> pd.DataFrame:
    """Subjects and rows per bin (one subject counted once)."""
    df = pd.DataFrame({"age": list(ages), "subject": list(subject_ids)})
    df["bin"] = [age_bin(a) for a in df["age"]]
    g = df[df["bin"] != ""].groupby("bin")
    out = pd.DataFrame({"n_subjects": g["subject"].nunique(), "n_rows": g.size()})
    return out.reindex([l for l in BIN_LABELS if l in out.index])
