"""DuckDB SQL templates that turn MIMIC-IV / eICU raw files into the long measurement table.

Output schema (both databases): ``stay_id, itemid, t_hours, hod, valuenum, priority``
plus stay-level covariates in a separate ``stays`` table. ``priority`` is
``labevents.priority`` ('STAT'/'ROUTINE') on MIMIC-IV; eICU has no priority
field, so it is NULL there and STAT-based features are skipped.

Nothing here downloads data; ``root`` must point at a local, credentialed copy.
"""
from __future__ import annotations

from typing import Iterable

# Common MIMIC-IV lab itemids (hosp.d_labitems). Verify with ``verify_itemids`` before use.
MIMIC_LAB_ITEMS: dict[str, int] = {
    "creatinine": 50912, "bun": 51006, "sodium": 50983, "potassium": 50971, "chloride": 50902,
    "bicarbonate": 50882, "glucose": 50931, "calcium": 50893, "magnesium": 50960, "phosphate": 50970,
    "hemoglobin": 51222, "hematocrit": 51221, "wbc": 51301, "platelets": 51265,
    "lactate": 50813, "ph": 50820, "pco2": 50818, "po2": 50821,
    "alt": 50861, "ast": 50878, "bilirubin_total": 50885, "albumin": 50862,
    "inr": 51237, "pt": 51274, "ptt": 51275, "troponin_t": 51003,
}

# eICU lab.labname values for the same concepts (exact strings in eICU-CRD v2.0).
EICU_LAB_NAMES: dict[str, str] = {
    "creatinine": "creatinine", "bun": "BUN", "sodium": "sodium", "potassium": "potassium", "chloride": "chloride",
    "bicarbonate": "bicarbonate", "glucose": "glucose", "calcium": "calcium", "magnesium": "magnesium",
    "phosphate": "phosphate", "hemoglobin": "Hgb", "hematocrit": "Hct", "wbc": "WBC x 1000",
    "platelets": "platelets x 1000", "lactate": "lactate", "ph": "pH", "pco2": "paCO2", "po2": "paO2",
    "alt": "ALT (SGPT)", "ast": "AST (SGOT)", "bilirubin_total": "total bilirubin", "albumin": "albumin",
    "inr": "PT - INR", "pt": "PT", "ptt": "PTT", "troponin_t": "troponin - T",
}


def mimic_stays_sql(root: str) -> str:
    """Adult first ICU stays with demographics, insurance, language, race, era and outcomes."""
    return f"""
    SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.first_careunit,
           hour(i.intime) AS admit_hour,
           p.gender, p.anchor_age AS age, p.anchor_year_group,
           a.race, a.insurance, a.language, a.hospital_expire_flag,
           date_diff('hour', i.intime, a.deathtime) AS hours_to_death,
           date_diff('hour', i.intime, i.outtime) AS icu_los_hours
    FROM read_csv_auto('{root}/icu/icustays.csv.gz') i
    JOIN read_csv_auto('{root}/hosp/patients.csv.gz') p USING (subject_id)
    JOIN read_csv_auto('{root}/hosp/admissions.csv.gz') a USING (hadm_id)
    QUALIFY row_number() OVER (PARTITION BY i.subject_id ORDER BY i.intime) = 1
    """


def mimic_labs_sql(root: str, itemids: Iterable[int], window_hours: float = 24.0, pre_hours: float = 6.0) -> str:
    """Lab measurements (charttime = specimen collection) relative to ICU admission, with priority."""
    ids = ",".join(str(i) for i in itemids)
    return f"""
    WITH stays AS (SELECT stay_id, hadm_id, intime FROM read_csv_auto('{root}/icu/icustays.csv.gz'))
    SELECT s.stay_id, l.itemid,
           date_diff('second', s.intime, l.charttime) / 3600.0 AS t_hours,
           hour(l.charttime) AS hod,
           l.valuenum, l.priority, l.flag,
           date_diff('second', l.charttime, l.storetime) / 3600.0 AS result_delay_hours
    FROM read_csv_auto('{root}/hosp/labevents.csv.gz') l
    JOIN stays s ON l.hadm_id = s.hadm_id
    WHERE l.itemid IN ({ids})
      AND l.charttime >= s.intime - INTERVAL {int(pre_hours)} HOUR
      AND l.charttime <= s.intime + INTERVAL {int(window_hours)} HOUR
    """


def mimic_lab_orders_sql(root: str, window_hours: float = 24.0) -> str:
    """Provider order entry rows of type 'Lab' (order time precedes specimen collection)."""
    return f"""
    WITH stays AS (SELECT stay_id, hadm_id, intime FROM read_csv_auto('{root}/icu/icustays.csv.gz'))
    SELECT s.stay_id, o.poe_id, o.order_type, o.order_subtype, o.order_status,
           date_diff('second', s.intime, o.ordertime) / 3600.0 AS order_t_hours, hour(o.ordertime) AS order_hod
    FROM read_csv_auto('{root}/hosp/poe.csv.gz') o
    JOIN stays s ON o.hadm_id = s.hadm_id
    WHERE o.order_type = 'Lab'
      AND o.ordertime BETWEEN s.intime - INTERVAL 6 HOUR AND s.intime + INTERVAL {int(window_hours)} HOUR
    """


def eicu_stays_sql(root: str) -> str:
    """eICU unit stays with hospital identifiers (for cross-hospital ordering-policy analysis)."""
    return f"""
    SELECT p.patientunitstayid AS stay_id, p.patienthealthsystemstayid, p.hospitalid, p.unittype,
           CAST(split_part(p.unitadmittime24, ':', 1) AS INTEGER) AS admit_hour,
           p.gender, p.age, p.ethnicity AS race,
           p.unitdischargestatus, p.hospitaldischargestatus, p.unitdischargeoffset / 60.0 AS icu_los_hours,
           h.numbedscategory, h.teachingstatus, h.region
    FROM read_csv_auto('{root}/patient.csv.gz') p
    LEFT JOIN read_csv_auto('{root}/hospital.csv.gz') h USING (hospitalid)
    WHERE p.unitvisitnumber = 1
    """


def eicu_labs_sql(root: str, labnames: Iterable[str], window_hours: float = 24.0, pre_hours: float = 6.0) -> str:
    names = ",".join("'" + n.replace("'", "''") + "'" for n in labnames)
    return f"""
    SELECT l.patientunitstayid AS stay_id, l.labname AS itemid,
           l.labresultoffset / 60.0 AS t_hours,
           (CAST(split_part(p.unitadmittime24, ':', 1) AS INTEGER) + floor(l.labresultoffset / 60.0)) % 24 AS hod,
           l.labresult AS valuenum, NULL AS priority
    FROM read_csv_auto('{root}/lab.csv.gz') l
    JOIN read_csv_auto('{root}/patient.csv.gz') p USING (patientunitstayid)
    WHERE l.labname IN ({names})
      AND l.labresultoffset BETWEEN {-int(pre_hours * 60)} AND {int(window_hours * 60)}
    """


def run_query(sql: str, database: str = ":memory:"):
    """Execute with DuckDB (imported lazily) and return a pandas DataFrame."""
    try:
        import duckdb  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install duckdb to run the extraction queries") from e
    con = duckdb.connect(database)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def verify_itemids(d_labitems_path: str, items: dict[str, int] = MIMIC_LAB_ITEMS) -> dict[str, str]:
    """Return {concept: label} from ``d_labitems`` for each itemid so the mapping can be eyeballed."""
    import pandas as pd

    d = pd.read_csv(d_labitems_path).set_index("itemid")["label"]
    return {k: str(d.get(v, "MISSING")) for k, v in items.items()}
