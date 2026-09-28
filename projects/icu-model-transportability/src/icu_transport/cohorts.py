"""Cohort, stay-table and shared-ontology extraction for four ICU databases.

The module renders DuckDB SQL directly over the raw ``.csv(.gz)`` / ``.csv`` files distributed
by PhysioNet (MIMIC-IV v3.1, eICU-CRD v2.0, HiRID v1.1.1) and by Amsterdam Medical Data
Science (AmsterdamUMCdb v1.0.2). No database server is needed::

    import duckdb
    from icu_transport import cohorts
    con = duckdb.connect()
    stays = cohorts.load_stays(con, "mimiciv", root="data/raw/physionet.org/files/mimiciv/3.1")
    hr = cohorts.load_concept(con, "mimiciv", "hr", root=..., max_hours=48)

Every concept in :data:`ONTOLOGY` carries one source per database. Ids for MIMIC-IV are
high-confidence (they match ricu / YAIB); ids for eICU, HiRID and AUMCdb are taken from the
same dictionaries but flagged ``verified=False`` until :func:`verify_ontology` has been run
against the actual item dictionaries shipped with each database. ``load_concept`` refuses to
extract an unverified source unless ``allow_unverified=True``.

Time is always expressed as **hours since ICU admission** (float), which is the only time axis
that exists in all four databases (eICU has offsets only; AUMCdb has relative milliseconds).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

DBS = ("mimiciv", "eicu", "hirid", "aumcdb")

ROOT_ENV = {
    "mimiciv": "ICU_MIMICIV_ROOT",
    "eicu": "ICU_EICU_ROOT",
    "hirid": "ICU_HIRID_ROOT",
    "aumcdb": "ICU_AUMCDB_ROOT",
}

AGE_BANDS = ("18-39", "40-49", "50-59", "60-69", "70-79", "80+")

# ----------------------------------------------------------------------------------------------
# Source / concept specification
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """How one concept is stored in one database.

    ``where`` / ``value`` / ``time`` are SQL fragments over the event table alias ``e`` and the
    stay table alias ``s``. ``end_time`` marks interval-valued sources (infusions, ventilation).
    """

    table: str
    where: str
    value: str
    time: str
    join: str
    ids: tuple = ()
    dictionary: str = ""  # which dictionary verify_ontology should check ("chart", "lab", ...)
    end_time: str | None = None
    verified: bool = False
    note: str = ""


@dataclass(frozen=True)
class Concept:
    name: str
    unit: str
    label_regex: str  # case-insensitive regex the dictionary label must match
    lo: float
    hi: float
    sources: dict[str, tuple[Source, ...]] = field(default_factory=dict)
    carry_forward_h: float = 24.0  # maximum forward-fill horizon on the hourly grid
    kind: str = "numeric"  # numeric | interval | event


# -- helpers that build Source objects for each database schema -------------------------------

_MIMIC_T = "date_diff('minute', s.intime, e.charttime) / 60.0"


def _mimic_chart(ids: Sequence[int], factor: float = 1.0, offset: float = 0.0, **kw: Any) -> Source:
    return Source("icu/chartevents.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))}) AND e.valuenum IS NOT NULL",
                  f"e.valuenum * {factor} + {offset}", _MIMIC_T, "e.stay_id = s.stay_id",
                  ids=tuple(ids), dictionary="chart", verified=True, **kw)


def _mimic_lab(ids: Sequence[int], factor: float = 1.0, **kw: Any) -> Source:
    return Source("hosp/labevents.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))}) AND e.valuenum IS NOT NULL",
                  f"e.valuenum * {factor}", _MIMIC_T, "e.hadm_id = s.hadm_id",
                  ids=tuple(ids), dictionary="lab", verified=True, **kw)


def _mimic_output(ids: Sequence[int], **kw: Any) -> Source:
    return Source("icu/outputevents.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))}) AND e.value IS NOT NULL",
                  "e.value", _MIMIC_T, "e.stay_id = s.stay_id", ids=tuple(ids), dictionary="chart",
                  verified=True, **kw)


def _mimic_input(ids: Sequence[int], value: str = "e.rate", **kw: Any) -> Source:
    return Source("icu/inputevents.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))}) AND {value} IS NOT NULL",
                  value, "date_diff('minute', s.intime, e.starttime) / 60.0", "e.stay_id = s.stay_id",
                  ids=tuple(ids), dictionary="chart",
                  end_time="date_diff('minute', s.intime, e.endtime) / 60.0", verified=True, **kw)


def _mimic_proc(ids: Sequence[int], **kw: Any) -> Source:
    return Source("icu/procedureevents.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))})", "1.0",
                  "date_diff('minute', s.intime, e.starttime) / 60.0", "e.stay_id = s.stay_id",
                  ids=tuple(ids), dictionary="chart",
                  end_time="date_diff('minute', s.intime, e.endtime) / 60.0", verified=True, **kw)


def _eicu_periodic(col: str, **kw: Any) -> Source:
    return Source("vitalPeriodic.csv.gz", f"e.{col} IS NOT NULL", f"e.{col}", "e.observationoffset / 60.0",
                  "e.patientunitstayid = s.stay_id", ids=(col,), dictionary="vitalPeriodic", **kw)


def _eicu_aperiodic(col: str, **kw: Any) -> Source:
    return Source("vitalAperiodic.csv.gz", f"e.{col} IS NOT NULL", f"e.{col}", "e.observationoffset / 60.0",
                  "e.patientunitstayid = s.stay_id", ids=(col,), dictionary="vitalAperiodic", **kw)


def _eicu_lab(names: Sequence[str], factor: float = 1.0, **kw: Any) -> Source:
    quoted = ", ".join("'" + n.replace("'", "''") + "'" for n in names)
    return Source("lab.csv.gz", f"e.labname IN ({quoted}) AND e.labresult IS NOT NULL", f"e.labresult * {factor}",
                  "e.labresultoffset / 60.0", "e.patientunitstayid = s.stay_id", ids=tuple(names),
                  dictionary="lab", **kw)


def _eicu_nurse(label: str, valname: str, **kw: Any) -> Source:
    return Source("nurseCharting.csv.gz",
                  f"e.nursingchartcelltypevallabel = '{label}' AND e.nursingchartcelltypevalname = '{valname}'",
                  "TRY_CAST(e.nursingchartvalue AS DOUBLE)", "e.nursingchartoffset / 60.0",
                  "e.patientunitstayid = s.stay_id", ids=(label, valname), dictionary="nurseCharting", **kw)


def _eicu_io(path_like: str, **kw: Any) -> Source:
    return Source("intakeOutput.csv.gz", f"e.cellpath LIKE '{path_like}' AND e.cellvaluenumeric IS NOT NULL",
                  "e.cellvaluenumeric", "e.intakeoutputoffset / 60.0", "e.patientunitstayid = s.stay_id",
                  ids=(path_like,), dictionary="intakeOutput", **kw)


def _eicu_infusion(pattern: str, **kw: Any) -> Source:
    return Source("infusionDrug.csv.gz", f"e.drugname ILIKE '{pattern}' AND TRY_CAST(e.drugrate AS DOUBLE) IS NOT NULL",
                  "TRY_CAST(e.drugrate AS DOUBLE)", "e.infusionoffset / 60.0", "e.patientunitstayid = s.stay_id",
                  ids=(pattern,), dictionary="infusionDrug", **kw)


_HIRID_T = "date_diff('minute', s.intime, e.datetime) / 60.0"


def _hirid_obs(ids: Sequence[int], factor: float = 1.0, **kw: Any) -> Source:
    return Source("raw_stage/observation_tables/csv/part-*.csv",
                  f"e.variableid IN ({', '.join(map(str, ids))}) AND e.value IS NOT NULL", f"e.value * {factor}",
                  _HIRID_T, "e.patientid = s.stay_id", ids=tuple(ids), dictionary="variable", **kw)


def _hirid_pharma(ids: Sequence[int], **kw: Any) -> Source:
    return Source("raw_stage/pharma_records/csv/part-*.csv",
                  f"e.pharmaid IN ({', '.join(map(str, ids))}) AND e.givendose IS NOT NULL", "e.givendose",
                  "date_diff('minute', s.intime, e.givenat) / 60.0", "e.patientid = s.stay_id", ids=tuple(ids),
                  dictionary="pharma", **kw)


_AUMC_T = "(e.measuredat - s.intime_ms) / 3600000.0"


def _aumc_numeric(ids: Sequence[int], factor: float = 1.0, **kw: Any) -> Source:
    return Source("numericitems.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))}) AND e.value IS NOT NULL",
                  f"e.value * {factor}", _AUMC_T, "e.admissionid = s.stay_id", ids=tuple(ids),
                  dictionary="numericitems", **kw)


def _aumc_list(ids: Sequence[int], **kw: Any) -> Source:
    return Source("listitems.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))})", "TRY_CAST(e.valueid AS DOUBLE)",
                  _AUMC_T, "e.admissionid = s.stay_id", ids=tuple(ids), dictionary="listitems", **kw)


def _aumc_drug(ids: Sequence[int], **kw: Any) -> Source:
    return Source("drugitems.csv.gz", f"e.itemid IN ({', '.join(map(str, ids))}) AND e.rate IS NOT NULL", "e.rate",
                  "(e.start - s.intime_ms) / 3600000.0", "e.admissionid = s.stay_id", ids=tuple(ids),
                  dictionary="drugitems", end_time="(e.stop - s.intime_ms) / 3600000.0", **kw)


def _c(name: str, unit: str, rx: str, lo: float, hi: float, *, mimiciv=(), eicu=(), hirid=(), aumcdb=(),
       carry: float = 24.0, kind: str = "numeric") -> Concept:
    def tup(x):
        return tuple(x) if isinstance(x, (list, tuple)) else (x,)

    return Concept(name, unit, rx, lo, hi,
                   {"mimiciv": tup(mimiciv), "eicu": tup(eicu), "hirid": tup(hirid), "aumcdb": tup(aumcdb)},
                   carry, kind)


# ----------------------------------------------------------------------------------------------
# The shared ontology (38 concepts). Ids follow ricu / YAIB; non-MIMIC ids need verify_ontology.
# ----------------------------------------------------------------------------------------------

ONTOLOGY: dict[str, Concept] = {c.name: c for c in [
    _c("hr", "bpm", r"heart ?rate|hartfrequentie", 20, 300,
       mimiciv=_mimic_chart([220045]), eicu=_eicu_periodic("heartrate"),
       hirid=_hirid_obs([200]), aumcdb=_aumc_numeric([6640]), carry=6),
    _c("sbp", "mmHg", r"systolic|systolisch|arterial blood pressure systolic", 30, 300,
       mimiciv=_mimic_chart([220050, 220179]),
       eicu=(_eicu_periodic("systemicsystolic"), _eicu_aperiodic("noninvasivesystolic")),
       hirid=_hirid_obs([100, 600]), aumcdb=_aumc_numeric([6641, 6678]), carry=6),
    _c("dbp", "mmHg", r"diastolic|diastolisch", 10, 200,
       mimiciv=_mimic_chart([220051, 220180]),
       eicu=(_eicu_periodic("systemicdiastolic"), _eicu_aperiodic("noninvasivediastolic")),
       hirid=_hirid_obs([120, 620]), aumcdb=_aumc_numeric([6643, 6680]), carry=6),
    _c("map", "mmHg", r"mean|gemiddeld", 20, 250,
       mimiciv=_mimic_chart([220052, 220181]),
       eicu=(_eicu_periodic("systemicmean"), _eicu_aperiodic("noninvasivemean")),
       hirid=_hirid_obs([110, 610]), aumcdb=_aumc_numeric([6642, 6679]), carry=6),
    _c("resp", "/min", r"respirat|ademfrequentie", 2, 80,
       mimiciv=_mimic_chart([220210, 224690]), eicu=_eicu_periodic("respiration"),
       hirid=_hirid_obs([300]), aumcdb=_aumc_numeric([8874]), carry=6),
    _c("spo2", "%", r"spo2|o2 saturation|sao2|saturatie", 40, 100,
       mimiciv=_mimic_chart([220277]), eicu=_eicu_periodic("sao2"),
       hirid=_hirid_obs([4000]), aumcdb=_aumc_numeric([6709]), carry=6),
    _c("temp", "C", r"temperature|temp", 25, 45,
       mimiciv=(_mimic_chart([223762]), _mimic_chart([223761], factor=5 / 9, offset=-32 * 5 / 9)),
       eicu=_eicu_periodic("temperature"), hirid=_hirid_obs([410, 400]), aumcdb=_aumc_numeric([8658, 8659, 8662]),
       carry=12),
    _c("gcs_eye", "score", r"gcs|glasgow|eye|ogen|e\.?m\.?v", 1, 4,
       mimiciv=_mimic_chart([220739]), eicu=_eicu_nurse("Glasgow coma score", "Eyes"),
       hirid=_hirid_obs([10000300]), aumcdb=_aumc_list([6732]), carry=24),
    _c("gcs_verbal", "score", r"gcs|glasgow|verbal|verbaal|e\.?m\.?v", 1, 5,
       mimiciv=_mimic_chart([223900]), eicu=_eicu_nurse("Glasgow coma score", "Verbal"),
       hirid=_hirid_obs([10000200]), aumcdb=_aumc_list([6735]), carry=24),
    _c("gcs_motor", "score", r"gcs|glasgow|motor|e\.?m\.?v", 1, 6,
       mimiciv=_mimic_chart([223901]), eicu=_eicu_nurse("Glasgow coma score", "Motor"),
       hirid=_hirid_obs([10000100]), aumcdb=_aumc_list([6734]), carry=24),
    _c("fio2", "%", r"fio2|inspired o2|o2 ?%", 21, 100,
       mimiciv=_mimic_chart([223835]), eicu=_eicu_lab(["FiO2"]), hirid=_hirid_obs([2010]),
       aumcdb=_aumc_numeric([6699]), carry=12),
    _c("urine", "mL", r"urine|foley|void|nephrostomy|ureteral|suprapubic|straight cath|gu irrigant|condom", 0, 5000,
       mimiciv=_mimic_output([226559, 226560, 226561, 226563, 226564, 226565, 226567, 226557, 226558, 227489]),
       eicu=_eicu_io("%Output (ml)|Urine%"), hirid=_hirid_obs([30005010]),
       aumcdb=_aumc_numeric([8794, 8796, 8798, 8800, 8803]), carry=0, kind="event"),
    _c("creatinine", "mg/dL", r"creatinine|kreatinine", 0.1, 30,
       mimiciv=_mimic_lab([50912]), eicu=_eicu_lab(["creatinine"]),
       hirid=_hirid_obs([20000600], factor=1 / 88.42), aumcdb=_aumc_numeric([6836], factor=1 / 88.42)),
    _c("potassium", "mmol/L", r"potassium|kalium", 1, 12,
       mimiciv=_mimic_lab([50971, 50822]), eicu=_eicu_lab(["potassium"]),
       hirid=_hirid_obs([20000500]), aumcdb=_aumc_numeric([9927, 6835])),
    _c("sodium", "mmol/L", r"sodium|natrium", 90, 200,
       mimiciv=_mimic_lab([50983, 50824]), eicu=_eicu_lab(["sodium"]),
       hirid=_hirid_obs([20000400]), aumcdb=_aumc_numeric([9924, 6840])),
    _c("chloride", "mmol/L", r"chloride|chloor", 50, 160,
       mimiciv=_mimic_lab([50902]), eicu=_eicu_lab(["chloride"]),
       hirid=_hirid_obs([24000439]), aumcdb=_aumc_numeric([9930])),
    _c("bicarbonate", "mmol/L", r"bicarbonate|hco3|bicarbonaat", 2, 60,
       mimiciv=_mimic_lab([50882]), eicu=_eicu_lab(["bicarbonate"]),
       hirid=_hirid_obs([20004200]), aumcdb=_aumc_numeric([9992])),
    _c("bun", "mg/dL", r"urea nitrogen|bun|ureum", 1, 300,
       mimiciv=_mimic_lab([51006]), eicu=_eicu_lab(["BUN"]),
       hirid=_hirid_obs([20004100], factor=2.8), aumcdb=_aumc_numeric([9943], factor=2.8)),
    _c("glucose", "mg/dL", r"glucose", 10, 2000,
       mimiciv=_mimic_lab([50931, 50809]), eicu=_eicu_lab(["glucose", "bedside glucose"]),
       hirid=_hirid_obs([20005110], factor=18.016), aumcdb=_aumc_numeric([9947], factor=18.016)),
    _c("lactate", "mmol/L", r"lactate|lactaat", 0.1, 40,
       mimiciv=_mimic_lab([50813]), eicu=_eicu_lab(["lactate"]),
       hirid=_hirid_obs([24000524]), aumcdb=_aumc_numeric([10053])),
    _c("wbc", "10^9/L", r"white blood|wbc|leuco", 0.1, 200,
       mimiciv=_mimic_lab([51301]), eicu=_eicu_lab(["WBC x 1000"]),
       hirid=_hirid_obs([20000700]), aumcdb=_aumc_numeric([9965])),
    _c("platelets", "10^9/L", r"platelet|thrombo", 1, 2000,
       mimiciv=_mimic_lab([51265]), eicu=_eicu_lab(["platelets x 1000"]),
       hirid=_hirid_obs([20000110]), aumcdb=_aumc_numeric([9964])),
    _c("hemoglobin", "g/dL", r"hemoglobin|hgb|hb", 2, 25,
       mimiciv=_mimic_lab([51222]), eicu=_eicu_lab(["Hgb"]),
       hirid=_hirid_obs([24000836], factor=0.1), aumcdb=_aumc_numeric([9960], factor=1.611)),
    _c("bilirubin", "mg/dL", r"bilirubin|bilirubine", 0.05, 80,
       mimiciv=_mimic_lab([50885]), eicu=_eicu_lab(["total bilirubin"]),
       hirid=_hirid_obs([20004300], factor=1 / 17.1), aumcdb=_aumc_numeric([9945], factor=1 / 17.1)),
    _c("alt", "U/L", r"alanine|alt|sgpt|alat", 1, 20000,
       mimiciv=_mimic_lab([50861]), eicu=_eicu_lab(["ALT (SGPT)"]),
       hirid=_hirid_obs([24000610]), aumcdb=_aumc_numeric([9941])),
    _c("ast", "U/L", r"aspartate|ast|sgot|asat", 1, 20000,
       mimiciv=_mimic_lab([50878]), eicu=_eicu_lab(["AST (SGOT)"]),
       hirid=_hirid_obs([24000330]), aumcdb=_aumc_numeric([9940])),
    _c("inr", "ratio", r"inr", 0.5, 20,
       mimiciv=_mimic_lab([51237]), eicu=_eicu_lab(["PT - INR"]),
       hirid=_hirid_obs([20004410]), aumcdb=_aumc_numeric([11893])),
    _c("ph", "", r"\bph\b", 6.5, 8.0,
       mimiciv=_mimic_lab([50820]), eicu=_eicu_lab(["pH"]),
       hirid=_hirid_obs([20000300]), aumcdb=_aumc_numeric([12310, 6848])),
    _c("pao2", "mmHg", r"po2|pao2|o2 ?\(?arterial|zuurstof", 10, 700,
       mimiciv=_mimic_lab([50821]), eicu=_eicu_lab(["paO2"]),
       hirid=_hirid_obs([20000200]), aumcdb=_aumc_numeric([21214, 7433], factor=7.50062)),
    _c("paco2", "mmHg", r"pco2|paco2|koolzuur", 5, 200,
       mimiciv=_mimic_lab([50818]), eicu=_eicu_lab(["paCO2"]),
       hirid=_hirid_obs([20001200]), aumcdb=_aumc_numeric([21213, 6846], factor=7.50062)),
    _c("albumin", "g/dL", r"albumin", 0.5, 7,
       mimiciv=_mimic_lab([50862]), eicu=_eicu_lab(["albumin"]),
       hirid=_hirid_obs([24000605], factor=0.1), aumcdb=_aumc_numeric([9937], factor=0.1)),
    _c("magnesium", "mg/dL", r"magnesium", 0.2, 10,
       mimiciv=_mimic_lab([50960]), eicu=_eicu_lab(["magnesium"]),
       hirid=_hirid_obs([24000230], factor=2.431), aumcdb=_aumc_numeric([9952], factor=2.431)),
    _c("calcium", "mg/dL", r"calcium", 2, 20,
       mimiciv=_mimic_lab([50893]), eicu=_eicu_lab(["calcium"]),
       hirid=_hirid_obs([20005100], factor=4.008), aumcdb=_aumc_numeric([9933], factor=4.008)),
    _c("crp", "mg/L", r"c-reactive|crp", 0, 1000,
       mimiciv=_mimic_lab([50889]), eicu=_eicu_lab(["CRP"]),
       hirid=_hirid_obs([20002200]), aumcdb=_aumc_numeric([6825])),
    _c("norepinephrine", "mcg/kg/min", r"norepinephrine|noradrenaline", 0, 5,
       mimiciv=_mimic_input([221906]), eicu=_eicu_infusion("norepinephrine%"),
       hirid=_hirid_pharma([1000462, 1000656, 1000657, 1000658]), aumcdb=_aumc_drug([7229]),
       carry=0, kind="interval"),
    _c("vasopressin", "units/min", r"vasopressin", 0, 1,
       mimiciv=_mimic_input([222315]), eicu=_eicu_infusion("vasopressin%"),
       hirid=_hirid_pharma([112, 113]), aumcdb=_aumc_drug([12467]), carry=0, kind="interval"),
    _c("epinephrine", "mcg/kg/min", r"epinephrine|adrenaline", 0, 5,
       mimiciv=_mimic_input([221289]), eicu=_eicu_infusion("epinephrine%"),
       hirid=_hirid_pharma([71, 1000750, 1000649, 1000650, 1000655]), aumcdb=_aumc_drug([6962]),
       carry=0, kind="interval"),
    _c("mech_vent", "flag", r"ventilat|beademing", 0, 1,
       mimiciv=_mimic_proc([225792, 225794]), eicu=_eicu_periodic("respiration"),  # placeholder: use apacheApsVar.vent
       hirid=_hirid_obs([15001552]), aumcdb=_aumc_numeric([12290]), carry=0, kind="interval"),
]}

# eICU has no clean ventilation-flag time series in vitalPeriodic; the placeholder above must be
# replaced by respiratoryCharting / apacheApsVar-based logic before use.
ONTOLOGY["mech_vent"].sources["eicu"] = ()

# ----------------------------------------------------------------------------------------------
# Stay tables
# ----------------------------------------------------------------------------------------------

_AGE_BAND_SQL = """CASE WHEN {age} < 40 THEN '18-39' WHEN {age} < 50 THEN '40-49' WHEN {age} < 60 THEN '50-59'
        WHEN {age} < 70 THEN '60-69' WHEN {age} < 80 THEN '70-79' ELSE '80+' END"""


def stay_sql(db: str, root: str) -> str:
    """SQL producing one row per ICU stay with a harmonised column set.

    Columns: stay_id, subject_id, hadm_id, intime (TIMESTAMP or NULL), intime_ms (AUMCdb only),
    los_hours, age, age_band, sex ('F'/'M'), race (NULL where unavailable), death_hours (hours
    from ICU admission to death; NULL if survived / unknown), hospital_id, stay_rank, db.
    """
    r = root.rstrip("/")
    if db == "mimiciv":
        return f"""
        SELECT icu.stay_id, icu.subject_id, icu.hadm_id, icu.intime, NULL::DOUBLE AS intime_ms,
               icu.los * 24.0 AS los_hours,
               pat.anchor_age + (EXTRACT(year FROM icu.intime) - pat.anchor_year) AS age,
               {_AGE_BAND_SQL.format(age='(pat.anchor_age + (EXTRACT(year FROM icu.intime) - pat.anchor_year))')} AS age_band,
               pat.gender AS sex, adm.race AS race,
               CASE WHEN adm.deathtime IS NOT NULL THEN date_diff('minute', icu.intime, adm.deathtime) / 60.0
                    WHEN pat.dod IS NOT NULL THEN date_diff('minute', icu.intime, CAST(pat.dod AS TIMESTAMP)) / 60.0
               END AS death_hours,
               NULL::VARCHAR AS hospital_id,
               ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS stay_rank,
               'mimiciv' AS db
        FROM read_csv_auto('{r}/icu/icustays.csv.gz') icu
        JOIN read_csv_auto('{r}/hosp/patients.csv.gz') pat USING (subject_id)
        JOIN read_csv_auto('{r}/hosp/admissions.csv.gz') adm USING (hadm_id)
        """
    if db == "eicu":
        age = "CASE WHEN p.age = '> 89' THEN 90 ELSE TRY_CAST(p.age AS INTEGER) END"
        return f"""
        SELECT p.patientunitstayid AS stay_id, p.uniquepid AS subject_id, p.patienthealthsystemstayid AS hadm_id,
               NULL::TIMESTAMP AS intime, NULL::DOUBLE AS intime_ms,
               p.unitdischargeoffset / 60.0 AS los_hours,
               {age} AS age, {_AGE_BAND_SQL.format(age=age)} AS age_band,
               CASE WHEN p.gender = 'Female' THEN 'F' WHEN p.gender = 'Male' THEN 'M' END AS sex,
               p.ethnicity AS race,
               CASE WHEN p.unitdischargestatus = 'Expired' THEN p.unitdischargeoffset / 60.0
                    WHEN p.hospitaldischargestatus = 'Expired' THEN p.hospitaldischargeoffset / 60.0 END AS death_hours,
               CAST(p.hospitalid AS VARCHAR) AS hospital_id,
               -- eICU has no absolute dates: ordering across hospital stays is approximate
               ROW_NUMBER() OVER (PARTITION BY p.uniquepid
                                  ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid, p.unitvisitnumber) AS stay_rank,
               'eicu' AS db
        FROM read_csv_auto('{r}/patient.csv.gz') p
        """
    if db == "hirid":
        # HiRID has no discharge time; los_hours / death_hours use the last observation time,
        # which is the convention used by ricu / YAIB.
        return f"""
        WITH last_obs AS (
            SELECT patientid, MAX(datetime) AS last_dt
            FROM read_csv_auto('{r}/raw_stage/observation_tables/csv/part-*.csv', union_by_name=true)
            GROUP BY patientid)
        SELECT g.patientid AS stay_id, g.patientid AS subject_id, NULL::BIGINT AS hadm_id,
               g.admissiontime AS intime, NULL::DOUBLE AS intime_ms,
               date_diff('minute', g.admissiontime, l.last_dt) / 60.0 AS los_hours,
               g.age AS age, {_AGE_BAND_SQL.format(age='g.age')} AS age_band,
               g.sex AS sex, NULL::VARCHAR AS race,
               CASE WHEN g.discharge_status = 'dead' THEN date_diff('minute', g.admissiontime, l.last_dt) / 60.0 END AS death_hours,
               NULL::VARCHAR AS hospital_id, 1 AS stay_rank, 'hirid' AS db
        FROM read_csv_auto('{r}/reference_data/general_table.csv') g
        LEFT JOIN last_obs l USING (patientid)
        """
    if db == "aumcdb":
        # All AUMCdb times are milliseconds relative to the patient's first admission.
        return f"""
        SELECT a.admissionid AS stay_id, a.patientid AS subject_id, NULL::BIGINT AS hadm_id,
               NULL::TIMESTAMP AS intime, CAST(a.admittedat AS DOUBLE) AS intime_ms,
               a.lengthofstay AS los_hours,
               NULL::DOUBLE AS age, a.agegroup AS age_band,
               CASE WHEN a.gender = 'Vrouw' THEN 'F' WHEN a.gender = 'Man' THEN 'M' END AS sex,
               NULL::VARCHAR AS race,
               CASE WHEN a.dateofdeath IS NOT NULL THEN (a.dateofdeath - a.admittedat) / 3600000.0 END AS death_hours,
               NULL::VARCHAR AS hospital_id, a.admissioncount AS stay_rank, 'aumcdb' AS db
        FROM read_csv_auto('{r}/admissions.csv.gz') a
        """
    raise ValueError(f"unknown db {db!r}; expected one of {DBS}")


def concept_sql(db: str, concept: Concept | str, root: str, max_hours: float | None = None,
                allow_unverified: bool = False) -> str:
    """Render the UNION of all sources of ``concept`` in ``db`` as one long-format query.

    Output columns: stay_id, hours, value[, end_hours for interval concepts].
    """
    c = ONTOLOGY[concept] if isinstance(concept, str) else concept
    srcs = c.sources.get(db, ())
    if not srcs:
        raise KeyError(f"concept {c.name!r} has no source in {db!r}")
    parts = []
    for src in srcs:
        if not src.verified and not allow_unverified:
            raise RuntimeError(
                f"{db}:{c.name} source ids {src.ids} are not verified. Run verify_ontology() and set "
                f"verified=True (or pass allow_unverified=True for exploration)."
            )
        end = f", {src.end_time} AS end_hours" if src.end_time else ", NULL::DOUBLE AS end_hours"
        tlimit = f" AND ({src.time}) <= {max_hours}" if max_hours is not None else ""
        parts.append(
            f"SELECT s.stay_id, {src.time} AS hours, {src.value} AS value{end}\n"
            f"FROM read_csv_auto('{root.rstrip('/')}/{src.table}', union_by_name=true) e\n"
            f"JOIN stays s ON {src.join}\nWHERE {src.where}{tlimit}"
        )
    return f"WITH stays AS ({stay_sql(db, root)})\n" + "\nUNION ALL\n".join(parts)


def _root(db: str, root: str | None) -> str:
    root = root or os.environ.get(ROOT_ENV[db])
    if not root:
        raise ValueError(f"no root for {db}: pass root= or set {ROOT_ENV[db]}")
    return root


def load_stays(con: Any, db: str, root: str | None = None, adults_only: bool = True,
               first_stay_only: bool = True, min_los_hours: float = 6.0) -> pd.DataFrame:
    """Load the harmonised stay table with the YAIB-style inclusion criteria applied."""
    df = con.execute(stay_sql(db, _root(db, root))).df()
    if adults_only and df["age"].notna().any():
        df = df[df["age"].fillna(18) >= 18]
    if first_stay_only:
        df = df[df["stay_rank"] == 1]
    if min_los_hours:
        df = df[df["los_hours"].fillna(np.inf) >= min_los_hours]
    return df.reset_index(drop=True)


def load_concept(con: Any, db: str, concept: str, root: str | None = None, max_hours: float | None = None,
                 allow_unverified: bool = False, plausibility: bool = True) -> pd.DataFrame:
    """Long-format concept table ``(stay_id, hours, value, end_hours, concept)`` for one database."""
    c = ONTOLOGY[concept]
    df = con.execute(concept_sql(db, c, _root(db, root), max_hours, allow_unverified)).df()
    if plausibility and c.kind == "numeric":
        df = df[(df["value"] >= c.lo) & (df["value"] <= c.hi)]
    df["concept"] = c.name
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Ontology verification against each database's own dictionary
# ----------------------------------------------------------------------------------------------

def _dictionary_sql(db: str, dictionary: str, root: str) -> str | None:
    r = root.rstrip("/")
    if db == "mimiciv":
        return {"chart": f"SELECT itemid AS id, label FROM read_csv_auto('{r}/icu/d_items.csv.gz')",
                "lab": f"SELECT itemid AS id, label FROM read_csv_auto('{r}/hosp/d_labitems.csv.gz')"}.get(dictionary)
    if db == "eicu":
        return {"lab": f"SELECT DISTINCT labname AS id, labname AS label FROM read_csv_auto('{r}/lab.csv.gz')",
                "vitalPeriodic": f"SELECT column_name AS id, column_name AS label FROM (DESCRIBE SELECT * FROM read_csv_auto('{r}/vitalPeriodic.csv.gz'))",
                "vitalAperiodic": f"SELECT column_name AS id, column_name AS label FROM (DESCRIBE SELECT * FROM read_csv_auto('{r}/vitalAperiodic.csv.gz'))",
                "nurseCharting": f"SELECT DISTINCT nursingchartcelltypevalname AS id, nursingchartcelltypevallabel || ' ' || nursingchartcelltypevalname AS label FROM read_csv_auto('{r}/nurseCharting.csv.gz')",
                "infusionDrug": f"SELECT DISTINCT drugname AS id, drugname AS label FROM read_csv_auto('{r}/infusionDrug.csv.gz')",
                "intakeOutput": f"SELECT DISTINCT cellpath AS id, cellpath AS label FROM read_csv_auto('{r}/intakeOutput.csv.gz')"}.get(dictionary)
    if db == "hirid":
        return {"variable": f"SELECT id, \"variable name\" AS label FROM read_csv_auto('{r}/reference_data/hirid_variable_reference.csv', normalize_names=false)",
                "pharma": f"SELECT \"variableid\" AS id, \"variable name\" AS label FROM read_csv_auto('{r}/reference_data/pharma_reference.csv', normalize_names=false)"}.get(dictionary)
    if db == "aumcdb":
        return {"numericitems": f"SELECT DISTINCT itemid AS id, item AS label FROM read_csv_auto('{r}/numericitems.csv.gz')",
                "listitems": f"SELECT DISTINCT itemid AS id, item AS label FROM read_csv_auto('{r}/listitems.csv.gz')",
                "drugitems": f"SELECT DISTINCT itemid AS id, item AS label FROM read_csv_auto('{r}/drugitems.csv.gz')"}.get(dictionary)
    return None


def verify_ontology(con: Any, db: str, root: str | None = None, concepts: Iterable[str] | None = None) -> pd.DataFrame:
    """Check every source id of ``db`` against the database's own item dictionary.

    Returns one row per (concept, id) with ``status`` in {"ok", "label_mismatch", "missing_id",
    "no_dictionary"} and the dictionary label found. A concept is safe to extract when all of
    its ids are ``ok``; the returned frame is meant to be inspected and used to flip the
    ``verified`` flags in :data:`ONTOLOGY` (or to fix the ids).
    """
    root = _root(db, root)
    names = list(concepts) if concepts is not None else list(ONTOLOGY)
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for name in names:
        c = ONTOLOGY[name]
        rx = re.compile(c.label_regex, re.IGNORECASE)
        for src in c.sources.get(db, ()):
            sql = _dictionary_sql(db, src.dictionary, root)
            if sql is None:
                rows.append((name, None, "no_dictionary", None, src.verified))
                continue
            if src.dictionary not in cache:
                cache[src.dictionary] = con.execute(sql).df()
            d = cache[src.dictionary]
            for i in src.ids:
                if src.dictionary in ("infusionDrug", "intakeOutput"):
                    pat = i.replace("%", ".*")
                    hit = d[d["label"].astype(str).str.contains(pat, case=False, regex=True, na=False)]
                else:
                    hit = d[d["id"].astype(str) == str(i)]
                if hit.empty:
                    rows.append((name, i, "missing_id", None, src.verified))
                else:
                    label = str(hit["label"].iloc[0])
                    rows.append((name, i, "ok" if rx.search(label) else "label_mismatch", label, src.verified))
    return pd.DataFrame(rows, columns=["concept", "id", "status", "label", "verified"])


# ----------------------------------------------------------------------------------------------
# Hourly grid, windows, SOFA and labels (database-agnostic; operate on the long tables)
# ----------------------------------------------------------------------------------------------

def hourly_grid(long: pd.DataFrame, stays: pd.DataFrame, n_hours: int = 48) -> pd.DataFrame:
    """Pivot long concept rows to an (stay_id, hour) x concept grid with carry-forward.

    Numeric concepts take the last value within the hour then forward-fill up to
    ``carry_forward_h`` hours; event concepts (urine) are summed per hour with no fill;
    interval concepts (infusions, ventilation) are expanded to every hour they cover.
    Returns a wide frame indexed by (stay_id, hour) with one column per concept plus
    ``<concept>__n`` measurement counts for numeric concepts.
    """
    idx = pd.MultiIndex.from_product([stays["stay_id"].unique(), range(n_hours)], names=["stay_id", "hour"])
    out = pd.DataFrame(index=idx)
    for name, g in long.groupby("concept"):
        c = ONTOLOGY[name]
        g = g[(g["hours"] >= 0) & (g["hours"] < n_hours)] if c.kind != "interval" else g
        if c.kind == "interval":
            recs = []
            for r in g.itertuples(index=False):
                end = r.end_hours if pd.notna(r.end_hours) else r.hours + 1
                h0, h1 = max(int(np.floor(r.hours)), 0), min(int(np.ceil(end)), n_hours)
                recs.extend((r.stay_id, h, r.value) for h in range(h0, h1))
            if not recs:
                out[name] = 0.0
                continue
            ex = pd.DataFrame(recs, columns=["stay_id", "hour", "value"])
            s = ex.groupby(["stay_id", "hour"])["value"].max()
            out[name] = s.reindex(idx).fillna(0.0)
            continue
        g = g.assign(hour=np.floor(g["hours"]).astype(int))
        if c.kind == "event":
            out[name] = g.groupby(["stay_id", "hour"])["value"].sum().reindex(idx)
            continue
        last = g.sort_values("hours").groupby(["stay_id", "hour"])["value"].last().reindex(idx)
        cnt = g.groupby(["stay_id", "hour"])["value"].size().reindex(idx).fillna(0).astype(int)
        limit = int(c.carry_forward_h) if c.carry_forward_h else 0
        filled = last.groupby(level="stay_id").ffill(limit=limit) if limit > 0 else last
        out[name] = filled
        out[name + "__n"] = cnt
    return out


def window_features(grid: pd.DataFrame, start: int = 0, end: int = 24, include_counts: bool = True) -> pd.DataFrame:
    """Summarise the hourly grid over hours [start, end) into one row per stay.

    Produces mean/min/max/last per concept, the number of hours with a value, and (optionally)
    the total measurement count ``<concept>__count`` (the "observation-process" feature set).
    """
    g = grid[(grid.index.get_level_values("hour") >= start) & (grid.index.get_level_values("hour") < end)]
    value_cols = [c for c in g.columns if not c.endswith("__n")]
    agg = g[value_cols].groupby(level="stay_id").agg(["mean", "min", "max", "last"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    if include_counts:
        cnt = g[[c for c in g.columns if c.endswith("__n")]].groupby(level="stay_id").sum()
        cnt.columns = [c.replace("__n", "__count") for c in cnt.columns]
        agg = agg.join(cnt)
    miss = g[value_cols].isna().groupby(level="stay_id").mean()
    miss.columns = [f"{c}__missing" for c in miss.columns]
    return agg.join(miss)


def sofa_hourly(grid: pd.DataFrame) -> pd.Series:
    """Hourly SOFA (Vincent et al., 1996) from the grid; urine-based renal points need 24-h sums.

    Missing components score 0 (the usual convention for retrospective Sepsis-3 labelling).
    """
    def col(name: str) -> pd.Series:
        return grid[name] if name in grid else pd.Series(np.nan, index=grid.index)

    pf = col("pao2") / (col("fio2") / 100.0)
    vent = col("mech_vent").fillna(0) > 0
    resp = pd.Series(0, index=grid.index)
    resp[pf < 400] = 1
    resp[pf < 300] = 2
    resp[(pf < 200) & vent] = 3
    resp[(pf < 100) & vent] = 4

    plt = col("platelets")
    coag = pd.Series(0, index=grid.index)
    coag[plt < 150] = 1
    coag[plt < 100] = 2
    coag[plt < 50] = 3
    coag[plt < 20] = 4

    bili = col("bilirubin")
    liver = pd.Series(0, index=grid.index)
    liver[bili >= 1.2] = 1
    liver[bili >= 2.0] = 2
    liver[bili >= 6.0] = 3
    liver[bili >= 12.0] = 4

    mp = col("map")
    ne = col("norepinephrine").fillna(0)
    ep = col("epinephrine").fillna(0)
    cardio = pd.Series(0, index=grid.index)
    cardio[mp < 70] = 1
    cardio[(ne > 0) | (ep > 0)] = 3
    cardio[(ne > 0.1) | (ep > 0.1)] = 4

    gcs = col("gcs_eye") + col("gcs_verbal") + col("gcs_motor")
    cns = pd.Series(0, index=grid.index)
    cns[gcs <= 14] = 1
    cns[gcs <= 12] = 2
    cns[gcs <= 9] = 3
    cns[gcs <= 5] = 4

    cr = col("creatinine")
    uo24 = col("urine").groupby(level="stay_id").transform(lambda s: s.rolling(24, min_periods=1).sum())
    renal = pd.Series(0, index=grid.index)
    renal[cr >= 1.2] = 1
    renal[cr >= 2.0] = 2
    renal[(cr >= 3.5) | (uo24 < 500)] = 3
    renal[(cr >= 5.0) | (uo24 < 200)] = 4
    return (resp + coag + liver + cardio + cns + renal).rename("sofa")


def label_mortality(stays: pd.DataFrame, pred_hour: float = 24.0, horizon_h: float = 48.0) -> pd.Series:
    """1 if death occurs in (pred_hour, pred_hour + horizon_h] hours after ICU admission."""
    d = stays.set_index("stay_id")["death_hours"]
    return ((d > pred_hour) & (d <= pred_hour + horizon_h)).astype(int).rename("mortality")


def kdigo_creatinine_stage(creat: pd.DataFrame, baseline_window_h: float = 168.0) -> pd.DataFrame:
    """KDIGO creatinine stage per measurement (stay_id, hours, value -> stage).

    Baseline = minimum creatinine in the preceding ``baseline_window_h`` hours (else first value).
    Stage 1: +0.3 mg/dL within 48 h or >= 1.5x baseline; stage 2: >= 2x; stage 3: >= 3x or >= 4.0.
    """
    rows = []
    for sid, g in creat.sort_values("hours").groupby("stay_id"):
        h, v = g["hours"].to_numpy(), g["value"].to_numpy()
        for i in range(len(v)):
            prev = (h >= h[i] - baseline_window_h) & (h < h[i])
            base = v[prev].min() if prev.any() else v[0]
            prev48 = (h >= h[i] - 48) & (h < h[i])
            rise48 = v[i] - (v[prev48].min() if prev48.any() else v[i])
            ratio = v[i] / base if base > 0 else 1.0
            stage = 0
            if rise48 >= 0.3 or ratio >= 1.5:
                stage = 1
            if ratio >= 2.0:
                stage = 2
            if ratio >= 3.0 or (v[i] >= 4.0 and (rise48 >= 0.3 or ratio >= 1.5)):
                stage = 3
            rows.append((sid, h[i], v[i], base, stage))
    return pd.DataFrame(rows, columns=["stay_id", "hours", "value", "baseline", "stage"])


def label_aki(creat: pd.DataFrame, urine_hourly: pd.Series | None, weight_kg: pd.Series | None,
              pred_hour: float = 24.0, horizon_h: float = 48.0) -> pd.Series:
    """KDIGO stage >= 1 AKI (creatinine or 6-h urine output < 0.5 mL/kg/h) in (pred, pred+horizon]."""
    st = kdigo_creatinine_stage(creat)
    st = st[(st["hours"] > pred_hour) & (st["hours"] <= pred_hour + horizon_h) & (st["stage"] >= 1)]
    pos = set(st["stay_id"])
    if urine_hourly is not None and weight_kg is not None:
        uo = urine_hourly.groupby(level="stay_id").transform(lambda s: s.rolling(6, min_periods=6).sum())
        w = weight_kg.reindex(uo.index.get_level_values("stay_id")).to_numpy()
        rate = uo.to_numpy() / (6.0 * w)
        hrs = uo.index.get_level_values("hour").to_numpy()
        m = (rate < 0.5) & (hrs > pred_hour) & (hrs <= pred_hour + horizon_h)
        pos |= set(uo.index.get_level_values("stay_id")[m])
    ids = creat["stay_id"].unique()
    return pd.Series([int(s in pos) for s in ids], index=pd.Index(ids, name="stay_id"), name="aki")


def suspected_infection_time(abx: pd.DataFrame, cultures: pd.DataFrame) -> pd.Series:
    """Sepsis-3 suspicion time per stay (Seymour et al., 2016): earliest of antibiotic / culture
    when a culture is taken within 24 h before or 72 h after an antibiotic start.
    ``abx`` and ``cultures`` are long frames with columns (stay_id, hours)."""
    out = {}
    cul = {k: g["hours"].to_numpy() for k, g in cultures.groupby("stay_id")}
    for sid, g in abx.groupby("stay_id"):
        ch = cul.get(sid)
        if ch is None:
            continue
        for a in np.sort(g["hours"].to_numpy()):
            ok = ch[(ch >= a - 24) & (ch <= a + 72)]
            if ok.size:
                out[sid] = float(min(a, ok.min()))
                break
    return pd.Series(out, name="t_susp", dtype=float).rename_axis("stay_id")


def sepsis3_onset(sofa: pd.Series, t_susp: pd.Series, lookback_h: float = 48.0, lookahead_h: float = 24.0) -> pd.Series:
    """Sepsis-3 onset hour per stay: first hour in [t_susp - 48 h, t_susp + 24 h] where SOFA rises
    >= 2 above the minimum SOFA seen before that window (0 if unobserved)."""
    out = {}
    for sid, t in t_susp.items():
        if sid not in sofa.index.get_level_values("stay_id"):
            continue
        s = sofa.xs(sid, level="stay_id")
        hrs = s.index.to_numpy()
        base = s[hrs < t - lookback_h].min() if (hrs < t - lookback_h).any() else 0
        win = s[(hrs >= t - lookback_h) & (hrs <= t + lookahead_h)]
        hit = win[win - base >= 2]
        if not hit.empty:
            out[sid] = float(hit.index[0])
    return pd.Series(out, name="sepsis_onset", dtype=float).rename_axis("stay_id")


def label_sepsis(onset: pd.Series, stay_ids: Iterable, pred_hour: float = 24.0, horizon_h: float = 12.0,
                 exclude_prior: bool = True) -> pd.Series:
    """1 if Sepsis-3 onset in (pred, pred+horizon]; stays with onset <= pred are dropped (NaN) when
    ``exclude_prior`` so that prevalent cases do not contaminate the incident-onset task."""
    ids = pd.Index(list(stay_ids), name="stay_id")
    y = pd.Series(0.0, index=ids, name="sepsis")
    on = onset.reindex(ids)
    y[(on > pred_hour) & (on <= pred_hour + horizon_h)] = 1.0
    if exclude_prior:
        y[on <= pred_hour] = np.nan
    return y


def harmonise_race(race: pd.Series) -> pd.Series:
    """Map MIMIC-IV ``race`` and eICU ``ethnicity`` strings to 5 shared categories."""
    def f(x: Any) -> str:
        s = str(x).upper()
        if "HISPANIC" in s or "LATINO" in s:
            return "Hispanic"
        if "BLACK" in s or "AFRICAN" in s:
            return "Black"
        if "ASIAN" in s:
            return "Asian"
        if "WHITE" in s or "CAUCASIAN" in s:
            return "White"
        return "Other/Unknown"

    return race.map(f)
