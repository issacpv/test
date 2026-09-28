"""Harmonised preoperative features across INSPIRE, MOVER and MIMIC-IV.

The ontology is a dict concept -> (regex over the site's lab label, unit hint).
Each site supplies a dictionary of its own lab labels (MIMIC ``d_labitems``,
INSPIRE ``item_name`` values, MOVER lab names); ``verify_ontology`` refuses a
mapping whose label does not match the concept regex so that a mis-mapped
variable cannot silently become "concept shift".
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PREOP_LABS: dict[str, tuple[str, str]] = {
    "hemoglobin": (r"h(a?e)?moglobin|^hb$|^hgb$", "g/dL"),
    "wbc": (r"white blood|^wbc$|leukocyte", "K/uL"),
    "platelets": (r"platelet|^plt$", "K/uL"),
    "sodium": (r"^sodium|^na$", "mEq/L"),
    "potassium": (r"^potassium|^k$", "mEq/L"),
    "creatinine": (r"^creatinine|^cr$|^crea", "mg/dL"),
    "bun": (r"urea nitrogen|^bun$", "mg/dL"),
    "glucose": (r"^glucose|^glu$", "mg/dL"),
    "albumin": (r"^albumin|^alb$", "g/dL"),
    "inr": (r"^inr", "ratio"),
    "bilirubin_total": (r"bilirubin,? total|^tbil|total bilirubin", "mg/dL"),
    "ast": (r"^ast|aspartate", "IU/L"),
}

SURGERY_GROUPS: dict[str, str] = {
    "0": "central_nervous", "1": "peripheral_nervous", "2": "heart_great_vessels", "3": "upper_arteries", "4": "lower_arteries",
    "5": "upper_veins", "6": "lower_veins", "7": "lymphatic", "8": "eye", "9": "ear_nose_sinus", "B": "respiratory",
    "C": "mouth_throat", "D": "gastrointestinal", "F": "hepatobiliary_pancreas", "G": "endocrine", "H": "skin_breast",
    "J": "skin_breast", "K": "musculoskeletal", "L": "musculoskeletal", "M": "musculoskeletal", "N": "musculoskeletal",
    "P": "musculoskeletal", "Q": "musculoskeletal", "R": "musculoskeletal", "S": "musculoskeletal", "T": "urinary",
    "U": "female_reproductive", "V": "male_reproductive", "W": "anatomical_regions", "X": "anatomical_regions", "Y": "anatomical_regions",
}


def verify_ontology(site_map: dict[str, str], site_labels: dict[str, str]) -> list[str]:
    """Check that every mapped site item's label matches its concept regex.

    ``site_map``: concept -> site item id (string).  ``site_labels``: site item id -> label.
    Returns a list of problems (empty = OK).  Raises KeyError for unknown concepts.
    """
    problems = []
    for concept, item in site_map.items():
        pattern = PREOP_LABS[concept][0]
        label = site_labels.get(str(item))
        if label is None:
            problems.append(f"{concept}: item {item} not in site dictionary")
        elif not re.search(pattern, label.strip().lower()):
            problems.append(f"{concept}: item {item} label '{label}' does not match /{pattern}/")
    return problems


def last_value_before(labs: pd.DataFrame, enc: pd.DataFrame, window_days: float = 30.0) -> pd.DataFrame:
    """Last lab value per concept within ``window_days`` before op_start, plus missing indicators.

    ``labs``: subject_id, charttime, concept, value.  ``enc``: op_id, subject_id, op_start.
    Returns a wide DataFrame indexed by op_id with columns <concept> and <concept>_missing.
    """
    from .cohorts import timedelta_hours

    m = enc[["op_id", "subject_id", "op_start"]].merge(labs, on="subject_id", how="inner")
    h = timedelta_hours(m["op_start"], m["charttime"])
    m = m[(h > 0) & (h <= 24.0 * window_days)].assign(h=h[(h > 0) & (h <= 24.0 * window_days)])
    m = m.sort_values("h").groupby(["op_id", "concept"], as_index=False).first()
    wide = m.pivot(index="op_id", columns="concept", values="value")
    wide = wide.reindex(index=enc["op_id"], columns=list(PREOP_LABS))
    for c in list(PREOP_LABS):
        wide[f"{c}_missing"] = wide[c].isna().astype(int)
    return wide


# Abbreviated Quan et al. (2005) prefix mapping.  Cross-check against mimic-code charlson.sql before use.
_CHARLSON: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {
    "mi": (1, ("410", "412"), ("I21", "I22", "I252")),
    "chf": (1, ("428",), ("I50", "I110", "I130", "I132")),
    "pvd": (1, ("441", "4439", "7854"), ("I70", "I71", "I739", "I790")),
    "cvd": (1, tuple(str(c) for c in range(430, 439)), ("I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69", "G45", "G46")),
    "dementia": (1, ("290",), ("F00", "F01", "F02", "F03", "G30")),
    "copd": (1, tuple(str(c) for c in range(490, 506)), ("J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67")),
    "rheum": (1, ("7100", "7101", "714"), ("M05", "M06", "M32", "M33", "M34")),
    "pud": (1, ("531", "532", "533", "534"), ("K25", "K26", "K27", "K28")),
    "mild_liver": (1, ("5712", "5715", "5716"), ("K700", "K701", "K702", "K703", "K73", "K74", "B18")),
    "diabetes": (1, ("250",), ("E10", "E11", "E12", "E13", "E14")),
    "hemiplegia": (2, ("342", "344"), ("G81", "G82")),
    "renal": (2, ("585", "586"), ("N18", "N19")),
    "cancer": (2, tuple(str(c) for c in list(range(140, 173)) + list(range(174, 196)) + list(range(200, 209))), tuple(f"C{i:02d}" for i in list(range(0, 27)) + list(range(30, 35)) + list(range(37, 42)) + [43] + list(range(45, 59)) + list(range(60, 77)) + list(range(81, 86)) + [88] + list(range(90, 98)))),
    "severe_liver": (3, ("4560", "4561", "4562", "5722", "5723", "5724"), ("I85", "I864", "K72", "K766", "K767")),
    "metastatic": (6, ("196", "197", "198", "199"), ("C77", "C78", "C79", "C80")),
    "hiv": (6, ("042",), ("B20", "B21", "B22", "B24")),
}


def charlson_from_icd(diag: pd.DataFrame, key: str = "hadm_id") -> pd.DataFrame:
    """Charlson comorbidity flags and index per ``key`` from ICD-9/10 codes.

    ``diag``: key, icd_code, icd_version.  Metastatic overrides cancer and severe
    liver overrides mild liver when computing the index (no double counting).
    """
    d = diag.copy()
    d["code"] = d["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    v9 = d["icd_version"].astype(int) == 9
    for name, (_, p9, p10) in _CHARLSON.items():
        d[name] = np.where(v9, d["code"].str.startswith(p9), d["code"].str.startswith(p10))
    flags = d.groupby(key)[list(_CHARLSON)].any()
    idx = pd.Series(0, index=flags.index, dtype=int)
    for name, (w, _, _) in _CHARLSON.items():
        f = flags[name].copy()
        if name == "cancer":
            f &= ~flags["metastatic"]
        if name == "mild_liver":
            f &= ~flags["severe_liver"]
        idx += w * f.astype(int)
    flags["charlson_index"] = idx
    return flags.reset_index()


def surgery_group_from_code(code: str, system: str) -> str:
    """Map a procedure code to one of the body-system groups.

    ``system`` in {'icd10pcs', 'icd9', 'cpt'}.  ICD-9 chapters and CPT ranges are
    mapped coarsely; unknown -> 'other'.
    """
    c = str(code).replace(".", "").upper()
    if system == "icd10pcs" and len(c) == 7 and c[0] == "0":
        return SURGERY_GROUPS.get(c[1], "other")
    if system == "icd9" and c[:2].isdigit():
        ch = int(c[:2])
        if 1 <= ch <= 5:
            return "central_nervous"
        if 6 <= ch <= 7:
            return "endocrine"
        if 8 <= ch <= 16:
            return "eye"
        if 18 <= ch <= 20:
            return "ear_nose_sinus"
        if 21 <= ch <= 29:
            return "mouth_throat"
        if 30 <= ch <= 34:
            return "respiratory"
        if 35 <= ch <= 39:
            return "heart_great_vessels"
        if 40 <= ch <= 41:
            return "lymphatic"
        if 42 <= ch <= 54:
            return "gastrointestinal"
        if 55 <= ch <= 59:
            return "urinary"
        if 60 <= ch <= 64:
            return "male_reproductive"
        if 65 <= ch <= 71:
            return "female_reproductive"
        if 76 <= ch <= 84:
            return "musculoskeletal"
        if 85 <= ch <= 86:
            return "skin_breast"
        return "other"
    if system == "cpt" and c[:5].isdigit():
        n = int(c[:5])
        if 10000 <= n < 20000:
            return "skin_breast"
        if 20000 <= n < 30000:
            return "musculoskeletal"
        if 30000 <= n < 33000:
            return "respiratory"
        if 33000 <= n < 40000:
            return "heart_great_vessels"
        if 40000 <= n < 50000:
            return "gastrointestinal"
        if 50000 <= n < 54000:
            return "urinary"
        if 54000 <= n < 56000:
            return "male_reproductive"
        if 56000 <= n < 60000:
            return "female_reproductive"
        if 60000 <= n < 61000:
            return "endocrine"
        if 61000 <= n < 65000:
            return "central_nervous"
        if 65000 <= n < 69000:
            return "eye"
        if 69000 <= n < 70000:
            return "ear_nose_sinus"
        return "other"
    return "other"


def assemble_preop_features(enc: pd.DataFrame, labs_wide: pd.DataFrame, charlson: pd.DataFrame) -> pd.DataFrame:
    """Join demographics, emergency flag, anaesthesia type, surgery group, labs and Charlson into one design table.

    ``enc`` needs op_id, hadm_id, age, sex ('F'/'M'), emergency (0/1), anesthesia ('general'/'regional'/'sedation'),
    surgery_group.  Categorical columns are one-hot encoded with a fixed category list so
    that every site yields identical columns.
    """
    X = enc[["op_id", "hadm_id", "age", "sex", "emergency", "anesthesia", "surgery_group"]].copy()
    X["female"] = (X.pop("sex") == "F").astype(int)
    X["anesthesia"] = pd.Categorical(X["anesthesia"], categories=["general", "regional", "sedation"])
    groups = sorted(set(SURGERY_GROUPS.values()) | {"other"})
    X["surgery_group"] = pd.Categorical(X["surgery_group"], categories=groups)
    X = pd.get_dummies(X, columns=["anesthesia", "surgery_group"], dtype=int)
    X = X.merge(labs_wide.reset_index(), on="op_id", how="left")
    X = X.merge(charlson[["hadm_id", "charlson_index"]], on="hadm_id", how="left")
    X["charlson_index"] = X["charlson_index"].fillna(0).astype(int)
    return X.set_index("op_id").drop(columns=["hadm_id"])
