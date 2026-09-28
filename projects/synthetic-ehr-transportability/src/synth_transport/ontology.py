"""Shared encounter-level ontology for MIMIC-IV, eICU-CRD and Synthea, plus the Synthea loader.

Every concept in ``FEATURE_SPEC`` has a definition for each of the three sources so that the same
feature matrix can be built from all of them. MIMIC-IV and eICU definitions are expressed as the
identifiers used in the respective tables (ICD prefixes, ``itemid``, ``labname``); the SQL that
applies them lives in ``MIMIC_SQL`` / ``EICU_SQL`` templates (DuckDB dialect). Synthea definitions
use SNOMED-CT condition codes and LOINC observation codes and are applied by
``load_synthea_encounters``.

Feature blocks
--------------
* ``demographics``      age, sex
* ``context``           emergency admission flag, prior encounters in the previous 365 days
* ``conditions``        six chronic-condition flags active at admission
* ``labs``              first value within 24 h for ten common labs
* ``observation``       per-lab measurement counts in 24 h and missingness indicators (this block is
                        the *observation process*, ablated in RQ2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SOURCES = ("mimic_iv", "eicu", "synthea")

# --------------------------------------------------------------- conditions
# ICD-9 / ICD-10 prefixes (MIMIC-IV diagnoses_icd; eICU diagnosis.icd9code) and Synthea SNOMED codes.
# Synthea codes should be checked against the module JSONs of the pinned release.
CONDITIONS: Dict[str, Dict[str, Sequence[str]]] = {
    "diabetes": {"icd9": ("250",), "icd10": ("E10", "E11", "E13"), "snomed": ("44054006", "46635009")},
    "hypertension": {"icd9": ("401", "402", "403", "404", "405"), "icd10": ("I10", "I11", "I12", "I13", "I15"), "snomed": ("59621000",)},
    "heart_failure": {"icd9": ("428",), "icd10": ("I50",), "snomed": ("88805009", "84114007")},
    "copd": {"icd9": ("491", "492", "496"), "icd10": ("J41", "J42", "J43", "J44"), "snomed": ("13645005", "185086009", "87433001")},
    "ckd": {"icd9": ("585",), "icd10": ("N18",), "snomed": ("431855005", "431856006", "433144002", "431857002", "46177005")},
    "cancer": {"icd9": tuple(str(i) for i in range(140, 209)), "icd10": tuple(f"C{i:02d}" for i in range(0, 98)), "snomed": ("363406005", "254837009", "93761005", "254637007", "363346000")},
}

# --------------------------------------------------------------------- labs
# MIMIC-IV hosp.labevents itemid; eICU lab.labname; Synthea observations LOINC code.
LABS: Dict[str, Dict[str, object]] = {
    "creatinine": {"mimic_itemid": (50912,), "eicu_labname": ("creatinine",), "loinc": ("2160-0", "38483-4"), "unit": "mg/dL"},
    "sodium": {"mimic_itemid": (50983,), "eicu_labname": ("sodium",), "loinc": ("2951-2",), "unit": "mEq/L"},
    "potassium": {"mimic_itemid": (50971,), "eicu_labname": ("potassium",), "loinc": ("2823-3",), "unit": "mEq/L"},
    "glucose": {"mimic_itemid": (50931,), "eicu_labname": ("glucose",), "loinc": ("2345-7",), "unit": "mg/dL"},
    "hemoglobin": {"mimic_itemid": (51222,), "eicu_labname": ("Hgb",), "loinc": ("718-7",), "unit": "g/dL"},
    "wbc": {"mimic_itemid": (51301,), "eicu_labname": ("WBC x 1000",), "loinc": ("6690-2",), "unit": "K/uL"},
    "platelets": {"mimic_itemid": (51265,), "eicu_labname": ("platelets x 1000",), "loinc": ("777-3",), "unit": "K/uL"},
    "bicarbonate": {"mimic_itemid": (50882,), "eicu_labname": ("bicarbonate",), "loinc": ("1963-8", "20565-8"), "unit": "mEq/L"},
    "bun": {"mimic_itemid": (51006,), "eicu_labname": ("BUN",), "loinc": ("3094-0", "6299-2"), "unit": "mg/dL"},
    "lactate": {"mimic_itemid": (50813,), "eicu_labname": ("lactate",), "loinc": ("2524-7",), "unit": "mmol/L"},
}

FEATURE_SPEC: Dict[str, Dict[str, object]] = {
    "age": {"block": "demographics", "mimic_iv": "patients.anchor_age + (admittime year - anchor_year)", "eicu": "patient.age ('> 89' -> 90)", "synthea": "encounter START - patients.BIRTHDATE"},
    "sex_male": {"block": "demographics", "mimic_iv": "patients.gender == 'M'", "eicu": "patient.gender == 'Male'", "synthea": "patients.GENDER == 'M'"},
    "emergency": {"block": "context", "mimic_iv": "admissions.admission_type in (EW EMER., URGENT, DIRECT EMER.)", "eicu": "patient.hospitaladmitsource == 'Emergency Department'", "synthea": "encounters.ENCOUNTERCLASS == 'emergency' within 24 h before inpatient START"},
    "prior_encounters_365d": {"block": "context", "mimic_iv": "count admissions with admittime in [admittime-365d, admittime)", "eicu": "not available (single-stay records) -> NaN", "synthea": "count encounters with START in [START-365d, START)"},
}
for _c in CONDITIONS:
    FEATURE_SPEC[f"cond_{_c}"] = {"block": "conditions", "mimic_iv": f"diagnoses_icd prefix {CONDITIONS[_c]['icd9'] + CONDITIONS[_c]['icd10']}", "eicu": "diagnosis.icd9code / pastHistory prefixes", "synthea": f"conditions.CODE in {CONDITIONS[_c]['snomed']} active at START"}
for _l in LABS:
    FEATURE_SPEC[f"lab_{_l}"] = {"block": "labs", "mimic_iv": f"labevents itemid {LABS[_l]['mimic_itemid']} first in 24 h", "eicu": f"lab.labname {LABS[_l]['eicu_labname']} first in 24 h", "synthea": f"observations LOINC {LABS[_l]['loinc']} first in encounter"}
    FEATURE_SPEC[f"n_{_l}"] = {"block": "observation", "mimic_iv": "count in 24 h", "eicu": "count in 24 h", "synthea": "count in encounter"}
    FEATURE_SPEC[f"miss_{_l}"] = {"block": "observation", "mimic_iv": "lab missing in 24 h", "eicu": "lab missing in 24 h", "synthea": "lab missing in encounter"}

OUTCOMES: Dict[str, Dict[str, str]] = {
    "mortality": {"mimic_iv": "admissions.hospital_expire_flag", "eicu": "patient.hospitaldischargestatus == 'Expired'", "synthea": "patients.DEATHDATE within [START, STOP]"},
    "readmit_30d": {"mimic_iv": "next admittime - dischtime <= 30 d", "eicu": "not available -> NaN", "synthea": "next inpatient START - STOP <= 30 d"},
}

# DuckDB SQL templates (documentation of the exact definitions; run them in a notebook/ETL script).
MIMIC_SQL = """
-- encounter-level table for MIMIC-IV (hosp module); {itemids} = comma-separated lab itemids
WITH adm AS (
  SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime, a.hospital_expire_flag,
         p.gender, p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year) AS age,
         a.admission_type IN ('EW EMER.', 'URGENT', 'DIRECT EMER.') AS emergency
  FROM admissions a JOIN patients p USING (subject_id)
  WHERE p.anchor_age >= 18
),
labs AS (
  SELECT l.hadm_id, l.itemid, l.valuenum, l.charttime,
         ROW_NUMBER() OVER (PARTITION BY l.hadm_id, l.itemid ORDER BY l.charttime) AS rn,
         COUNT(*) OVER (PARTITION BY l.hadm_id, l.itemid) AS n_meas
  FROM labevents l JOIN adm USING (hadm_id)
  WHERE l.itemid IN ({itemids}) AND l.charttime BETWEEN adm.admittime AND adm.admittime + INTERVAL 24 HOUR
        AND l.valuenum IS NOT NULL
)
SELECT * FROM adm LEFT JOIN (SELECT * FROM labs WHERE rn = 1) USING (hadm_id);
"""

EICU_SQL = """
-- encounter-level table for eICU-CRD; {labnames} = quoted, comma-separated lab names
WITH pt AS (
  SELECT patientunitstayid, patienthealthsystemstayid, gender = 'Male' AS sex_male,
         CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age,
         hospitaladmitsource = 'Emergency Department' AS emergency,
         hospitaldischargestatus = 'Expired' AS mortality, hospitalid
  FROM patient WHERE unitvisitnumber = 1
),
labs AS (
  SELECT l.patientunitstayid, l.labname, l.labresult, l.labresultoffset,
         ROW_NUMBER() OVER (PARTITION BY l.patientunitstayid, l.labname ORDER BY l.labresultoffset) AS rn,
         COUNT(*) OVER (PARTITION BY l.patientunitstayid, l.labname) AS n_meas
  FROM lab l WHERE l.labname IN ({labnames}) AND l.labresultoffset BETWEEN 0 AND 1440
)
SELECT * FROM pt LEFT JOIN (SELECT * FROM labs WHERE rn = 1) USING (patientunitstayid);
"""


def feature_names(blocks: Optional[Sequence[str]] = None) -> List[str]:
    """Names of features in the given blocks (all blocks when ``None``)."""
    return [k for k, v in FEATURE_SPEC.items() if blocks is None or v["block"] in set(blocks)]


def verify_spec() -> List[str]:
    """Return problems in the ontology (every concept must define all three sources)."""
    problems = []
    for name, spec in FEATURE_SPEC.items():
        for src in SOURCES:
            if src not in spec or not spec[src]:
                problems.append(f"{name}: missing definition for {src}")
        if spec.get("block") not in {"demographics", "context", "conditions", "labs", "observation"}:
            problems.append(f"{name}: unknown block {spec.get('block')!r}")
    for lab, spec in LABS.items():
        for key in ("mimic_itemid", "eicu_labname", "loinc"):
            if not spec.get(key):
                problems.append(f"lab {lab}: missing {key}")
    return problems


# ------------------------------------------------------------- Synthea loader
def _read_csv(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str)


def load_synthea_encounters(folder: str | Path, encounter_class: str = "inpatient", min_age: int = 18) -> pd.DataFrame:
    """Build the encounter-level feature table from Synthea CSV exports.

    Uses ``patients.csv``, ``encounters.csv``, ``conditions.csv`` and ``observations.csv``. One row per
    inpatient encounter of an adult patient; columns follow ``FEATURE_SPEC`` plus outcomes
    ``mortality`` and ``readmit_30d`` and identifiers ``encounter_id``, ``patient_id``.
    """
    folder = Path(folder)
    patients = _read_csv(folder, "patients")
    enc = _read_csv(folder, "encounters")
    cond = _read_csv(folder, "conditions")
    obs = _read_csv(folder, "observations")

    for df, cols in ((patients, ["BIRTHDATE", "DEATHDATE"]), (enc, ["START", "STOP"]), (cond, ["START", "STOP"]), (obs, ["DATE"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    enc = enc.merge(patients[["Id", "BIRTHDATE", "DEATHDATE", "GENDER"]].rename(columns={"Id": "PATIENT"}), on="PATIENT", how="left")
    enc = enc.sort_values(["PATIENT", "START"]).reset_index(drop=True)
    inpat = enc[enc["ENCOUNTERCLASS"].str.lower() == encounter_class].copy()
    inpat["age"] = (inpat["START"] - inpat["BIRTHDATE"]).dt.days / 365.25
    inpat = inpat[inpat["age"] >= min_age].copy()
    inpat["sex_male"] = (inpat["GENDER"].str.upper() == "M").astype(int)

    # context: emergency encounter within 24 h before, prior encounters in 365 d
    emerg = enc[enc["ENCOUNTERCLASS"].str.lower() == "emergency"][["PATIENT", "START"]].rename(columns={"START": "E_START"})
    all_enc = enc[["PATIENT", "START"]].rename(columns={"START": "A_START"})
    emergency_flag, prior_counts = [], []
    for _, row in inpat.iterrows():
        e = emerg[emerg["PATIENT"] == row["PATIENT"]]
        emergency_flag.append(int(((row["START"] - e["E_START"]) <= pd.Timedelta(hours=24)).where((row["START"] - e["E_START"]) >= pd.Timedelta(0)).fillna(False).any()))
        a = all_enc[all_enc["PATIENT"] == row["PATIENT"]]
        delta = row["START"] - a["A_START"]
        prior_counts.append(int(((delta > pd.Timedelta(0)) & (delta <= pd.Timedelta(days=365))).sum()))
    inpat["emergency"] = emergency_flag
    inpat["prior_encounters_365d"] = prior_counts

    # conditions active at START
    cond["CODE"] = cond["CODE"].astype(str)
    for name, spec in CONDITIONS.items():
        sub = cond[cond["CODE"].isin([str(c) for c in spec["snomed"]])]
        flags = []
        for _, row in inpat.iterrows():
            s = sub[sub["PATIENT"] == row["PATIENT"]]
            active = (s["START"] <= row["START"]) & (s["STOP"].isna() | (s["STOP"] >= row["START"]))
            flags.append(int(active.any()))
        inpat[f"cond_{name}"] = flags

    # labs within the encounter (first value; count; missing)
    obs["CODE"] = obs["CODE"].astype(str)
    obs["VALUE_NUM"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.sort_values("DATE")
    for lab, spec in LABS.items():
        sub = obs[obs["CODE"].isin(list(spec["loinc"])) & obs["VALUE_NUM"].notna()]
        grp = sub.groupby("ENCOUNTER")["VALUE_NUM"]
        first = grp.first()
        count = grp.size()
        inpat[f"lab_{lab}"] = inpat["Id"].map(first)
        inpat[f"n_{lab}"] = inpat["Id"].map(count).fillna(0).astype(int)
        inpat[f"miss_{lab}"] = inpat[f"lab_{lab}"].isna().astype(int)

    # outcomes
    inpat["mortality"] = ((inpat["DEATHDATE"] >= inpat["START"]) & (inpat["DEATHDATE"] <= inpat["STOP"] + pd.Timedelta(days=1))).astype(int)
    nxt = inpat.groupby("PATIENT")["START"].shift(-1)
    inpat["readmit_30d"] = ((nxt - inpat["STOP"]) <= pd.Timedelta(days=30)).fillna(False).astype(int)

    out = inpat.rename(columns={"Id": "encounter_id", "PATIENT": "patient_id"})
    cols = ["encounter_id", "patient_id"] + feature_names() + list(OUTCOMES)
    return out[cols].reset_index(drop=True)


# ------------------------------------------------------- feature matrix
def build_feature_matrix(
    df: pd.DataFrame,
    blocks: Optional[Sequence[str]] = None,
    impute: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Select feature columns, median-impute numeric labs, return ``(X, medians)``.

    ``blocks`` restricts to feature blocks (e.g. drop ``'observation'`` for the ablation). Medians
    computed on the training source must be passed as ``impute`` when transforming other sources so
    that no target information leaks into the features.
    """
    cols = [c for c in feature_names(blocks) if c in df.columns]
    X = df[cols].astype(float).copy()
    medians = dict(impute) if impute is not None else {c: float(X[c].median()) for c in cols}
    for c in cols:
        fill = medians.get(c, 0.0)
        if not np.isfinite(fill):
            fill = 0.0
            medians[c] = 0.0
        X[c] = X[c].fillna(fill)
    return X, medians
