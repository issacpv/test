"""Deterministic compiler from the cohort DSL to SQL (DuckDB) and to pandas, for MIMIC-IV / MIMIC-III.

Conventions follow the MIMIC Code Repository:
* MIMIC-IV age at ICU admission = ``anchor_age + (year(intime) - anchor_year)``;
* MIMIC-III age = years between ``dob`` and ``intime``; ages > 89 are shifted to 300+ in the data and
  are set to 91.4 (the MIMIC-III documented median for that group);
* "first ICU stay" = earliest ``intime`` per subject (default) or per hospital admission;
* ICU length of stay ``los`` is in fractional days in both databases.

Both executors produce a table with one row per unit (``stay_id``/``icustay_id``, ``hadm_id`` or
``subject_id``) plus the outcome column when an outcome is defined.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .cohort_spec import CohortDefinition, Criterion

__all__ = ["SCHEMA", "compile_sql", "PandasExecutor", "run_sql"]

SCHEMA: Dict[str, Dict[str, str]] = {
    "mimic-iv": {
        "stay_id": "stay_id",
        "icd_code": "icd_code",
        "icd_version": "icd_version",
        "age_expr": "p.anchor_age + (EXTRACT(year FROM i.intime) - p.anchor_year)",
        "dod": "p.dod",
    },
    "mimic-iii": {
        "stay_id": "icustay_id",
        "icd_code": "icd9_code",
        "icd_version": None,  # MIMIC-III is ICD-9 only
        "age_expr": "CASE WHEN date_diff('day', p.dob, i.intime) / 365.25 > 150 THEN 91.4 ELSE date_diff('day', p.dob, i.intime) / 365.25 END",
        "dod": "p.dod",
    },
}


def _s(x: Any) -> str:
    return "'" + str(x).replace("'", "''") + "'"


# ----------------------------------------------------------------- SQL
def compile_sql(defn: CohortDefinition) -> str:
    """Return a single DuckDB SQL statement selecting the cohort with its outcome."""
    sch = SCHEMA[defn.database]
    sid = sch["stay_id"]
    where: List[str] = []
    ctes: List[str] = []

    base = f"""
base AS (
  SELECT i.{sid} AS stay_id, i.hadm_id, i.subject_id, i.intime, i.outtime, i.los * 24.0 AS icu_los_hours,
         i.first_careunit, a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag, a.admission_type,
         date_diff('hour', a.admittime, a.dischtime) AS hosp_los_hours,
         {sch['age_expr']} AS age, {sch['dod']} AS dod,
         ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS stay_rank_subject,
         ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS stay_rank_hadm,
         ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY a.admittime) AS adm_rank_subject
  FROM icustays i JOIN admissions a ON a.hadm_id = i.hadm_id JOIN patients p ON p.subject_id = i.subject_id
)"""
    ctes.append(base)

    for k, c in enumerate(defn.criteria):
        p = c.params
        kind = c.kind
        if kind == "age_min":
            where.append(f"age >= {float(p['years'])}")
        elif kind == "age_max":
            where.append(f"age <= {float(p['years'])}")
        elif kind == "first_icu_stay_only":
            col = "stay_rank_hadm" if p.get("per", "subject") == "hadm" else "stay_rank_subject"
            where.append(f"{col} = 1")
        elif kind == "first_admission_only":
            where.append("adm_rank_subject = 1")
        elif kind == "min_icu_los_hours":
            where.append(f"icu_los_hours >= {float(p['hours'])}")
        elif kind == "max_icu_los_hours":
            where.append(f"icu_los_hours <= {float(p['hours'])}")
        elif kind == "min_hospital_los_hours":
            where.append(f"hosp_los_hours >= {float(p['hours'])}")
        elif kind == "care_unit_in":
            where.append("first_careunit IN (" + ", ".join(_s(u) for u in p["units"]) + ")")
        elif kind == "care_unit_not_in":
            where.append("first_careunit NOT IN (" + ", ".join(_s(u) for u in p["units"]) + ")")
        elif kind == "admission_type_in":
            where.append("admission_type IN (" + ", ".join(_s(t) for t in p["types"]) + ")")
        elif kind in ("require_icd_prefix", "exclude_icd_prefix"):
            # same normalisation as PandasExecutor: strip dots, upper-case, then prefix match
            conds = " OR ".join(f"UPPER(REPLACE(d.{sch['icd_code']}, '.', '')) LIKE {_s(str(pref).replace('.', '').upper() + '%')}" for pref in p["prefixes"])
            ver = ""
            if sch["icd_version"] and p.get("icd_version"):
                ver = f" AND d.{sch['icd_version']} = {int(p['icd_version'])}"
            ctes.append(f"icd_{k} AS (SELECT DISTINCT d.hadm_id FROM diagnoses_icd d WHERE ({conds}){ver})")
            op = "IN" if kind == "require_icd_prefix" else "NOT IN"
            where.append(f"hadm_id {op} (SELECT hadm_id FROM icd_{k})")
        elif kind == "require_lab_measured":
            ids = ", ".join(str(int(i)) for i in p["itemids"])
            win = float(p.get("window_hours", 24))
            ctes.append(
                f"lab_{k} AS (SELECT DISTINCT b.stay_id FROM base b JOIN labevents l ON l.hadm_id = b.hadm_id "
                f"WHERE l.itemid IN ({ids}) AND l.charttime >= b.intime AND l.charttime <= b.intime + INTERVAL {int(win)} HOUR AND l.valuenum IS NOT NULL)"
            )
            where.append(f"stay_id IN (SELECT stay_id FROM lab_{k})")
        elif kind == "exclude_death_within_hours":
            anchor = "admittime" if p.get("from", "icu_intime") == "hosp_admittime" else "intime"
            where.append(f"NOT (deathtime IS NOT NULL AND date_diff('hour', {anchor}, deathtime) < {float(p['hours'])})")
        elif kind == "exclude_missing_outcome":
            where.append("hospital_expire_flag IS NOT NULL")
        else:
            raise ValueError(f"unsupported criterion kind {kind!r}")

    outcome_sql = "NULL AS outcome"
    if defn.outcome is not None:
        ok = defn.outcome.kind
        if ok == "in_hospital_mortality":
            outcome_sql = "CAST(hospital_expire_flag AS INTEGER) AS outcome"
        elif ok == "icu_mortality":
            outcome_sql = "CAST(deathtime IS NOT NULL AND deathtime <= outtime AS INTEGER) AS outcome"
        elif ok in ("mortality_30d", "mortality_90d", "mortality_1y"):
            days = {"mortality_30d": 30, "mortality_90d": 90, "mortality_1y": 365}[ok]
            outcome_sql = f"CAST(dod IS NOT NULL AND date_diff('day', intime, dod) <= {days} AS INTEGER) AS outcome"
        elif ok == "icu_los_gt_hours":
            outcome_sql = f"CAST(icu_los_hours > {float(defn.outcome.hours)} AS INTEGER) AS outcome"
        elif ok == "readmission_30d":
            ctes.append("readm AS (SELECT subject_id, admittime, LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admit, dischtime FROM admissions)")
            outcome_sql = "CAST((SELECT date_diff('day', r.dischtime, r.next_admit) <= 30 FROM readm r WHERE r.subject_id = base.subject_id AND r.admittime = base.admittime) AS INTEGER) AS outcome"
        else:
            raise ValueError(f"unsupported outcome {ok!r}")

    unit_select = {"icustay": "stay_id, hadm_id, subject_id", "hadm": "stay_id, hadm_id, subject_id", "subject": "stay_id, hadm_id, subject_id"}[defn.unit]
    sql = "WITH " + ",\n".join(ctes) + f"\nSELECT {unit_select}, intime, age, icu_los_hours, {outcome_sql}\nFROM base\n"
    if where:
        sql += "WHERE " + "\n  AND ".join(where) + "\n"
    if defn.unit != "icustay":
        # hadm / subject units: keep the earliest surviving stay per unit so the unit key is unique
        key = "hadm_id" if defn.unit == "hadm" else "subject_id"
        sql += f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY intime) = 1\n"
    return sql.strip() + ";"


def run_sql(sql: str, con: Any) -> pd.DataFrame:
    """Execute compiled SQL on a DuckDB connection (tables: icustays, admissions, patients, ...)."""
    return con.execute(sql).df()


# -------------------------------------------------------------- pandas
class PandasExecutor:
    """Apply a definition to in-memory MIMIC tables with the same semantics as ``compile_sql``.

    Parameters are DataFrames named as the MIMIC tables: ``patients``, ``admissions``, ``icustays``,
    optional ``diagnoses_icd`` and ``labevents``. Datetime columns are parsed if they are strings.
    """

    def __init__(self, database: str, patients: pd.DataFrame, admissions: pd.DataFrame, icustays: pd.DataFrame, diagnoses_icd: Optional[pd.DataFrame] = None, labevents: Optional[pd.DataFrame] = None) -> None:
        self.database = database
        self.sch = SCHEMA[database]
        self.patients = patients.copy()
        self.admissions = admissions.copy()
        self.icustays = icustays.copy()
        self.diagnoses_icd = diagnoses_icd.copy() if diagnoses_icd is not None else None
        self.labevents = labevents.copy() if labevents is not None else None
        for df, cols in ((self.admissions, ["admittime", "dischtime", "deathtime"]), (self.icustays, ["intime", "outtime"]), (self.patients, ["dod", "dob"])):
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        if self.labevents is not None and "charttime" in self.labevents.columns:
            self.labevents["charttime"] = pd.to_datetime(self.labevents["charttime"], errors="coerce")

    def _base(self) -> pd.DataFrame:
        sid = self.sch["stay_id"]
        i = self.icustays.rename(columns={sid: "stay_id"})
        b = i.merge(self.admissions, on=["hadm_id", "subject_id"], how="inner", suffixes=("", "_adm")).merge(self.patients, on="subject_id", how="inner", suffixes=("", "_pat"))
        b["icu_los_hours"] = b["los"].astype(float) * 24.0
        b["hosp_los_hours"] = (b["dischtime"] - b["admittime"]).dt.total_seconds() / 3600.0
        if self.database == "mimic-iv":
            b["age"] = b["anchor_age"].astype(float) + (b["intime"].dt.year - b["anchor_year"].astype(float))
        else:
            yrs = (b["intime"] - b["dob"]).dt.days / 365.25
            b["age"] = np.where(yrs > 150, 91.4, yrs)
        b = b.sort_values(["subject_id", "intime"])
        b["stay_rank_subject"] = b.groupby("subject_id").cumcount() + 1
        b["stay_rank_hadm"] = b.groupby("hadm_id").cumcount() + 1
        adm_rank = self.admissions.sort_values(["subject_id", "admittime"]).assign(adm_rank_subject=lambda d: d.groupby("subject_id").cumcount() + 1)[["hadm_id", "adm_rank_subject"]]
        b = b.merge(adm_rank, on="hadm_id", how="left")
        return b

    def _icd_hadms(self, prefixes: List[str], icd_version: Optional[int]) -> set:
        if self.diagnoses_icd is None:
            raise ValueError("diagnoses_icd table required for ICD criteria")
        d = self.diagnoses_icd
        code = d[self.sch["icd_code"]].astype(str).str.replace(".", "", regex=False).str.upper()
        mask = pd.Series(False, index=d.index)
        for pref in prefixes:
            mask |= code.str.startswith(str(pref).replace(".", "").upper())
        if self.sch["icd_version"] and icd_version is not None:
            mask &= d[self.sch["icd_version"]].astype(int) == int(icd_version)
        return set(d.loc[mask, "hadm_id"])

    def _apply(self, b: pd.DataFrame, c: Criterion) -> pd.DataFrame:
        p = c.params
        k = c.kind
        if k == "age_min":
            return b[b["age"] >= float(p["years"])]
        if k == "age_max":
            return b[b["age"] <= float(p["years"])]
        if k == "first_icu_stay_only":
            col = "stay_rank_hadm" if p.get("per", "subject") == "hadm" else "stay_rank_subject"
            return b[b[col] == 1]
        if k == "first_admission_only":
            return b[b["adm_rank_subject"] == 1]
        if k == "min_icu_los_hours":
            return b[b["icu_los_hours"] >= float(p["hours"])]
        if k == "max_icu_los_hours":
            return b[b["icu_los_hours"] <= float(p["hours"])]
        if k == "min_hospital_los_hours":
            return b[b["hosp_los_hours"] >= float(p["hours"])]
        if k == "care_unit_in":
            return b[b["first_careunit"].isin(list(p["units"]))]
        if k == "care_unit_not_in":
            return b[~b["first_careunit"].isin(list(p["units"]))]
        if k == "admission_type_in":
            return b[b["admission_type"].isin(list(p["types"]))]
        if k == "require_icd_prefix":
            return b[b["hadm_id"].isin(self._icd_hadms(p["prefixes"], p.get("icd_version")))]
        if k == "exclude_icd_prefix":
            return b[~b["hadm_id"].isin(self._icd_hadms(p["prefixes"], p.get("icd_version")))]
        if k == "require_lab_measured":
            if self.labevents is None:
                raise ValueError("labevents table required")
            win = pd.Timedelta(hours=float(p.get("window_hours", 24)))
            l = self.labevents[self.labevents["itemid"].isin([int(i) for i in p["itemids"]]) & self.labevents["valuenum"].notna()]
            m = b[["stay_id", "hadm_id", "intime"]].merge(l[["hadm_id", "charttime"]], on="hadm_id")
            ok = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["intime"] + win)]["stay_id"].unique()
            return b[b["stay_id"].isin(ok)]
        if k == "exclude_death_within_hours":
            anchor = "admittime" if p.get("from", "icu_intime") == "hosp_admittime" else "intime"
            early = b["deathtime"].notna() & ((b["deathtime"] - b[anchor]).dt.total_seconds() / 3600.0 < float(p["hours"]))
            return b[~early]
        if k == "exclude_missing_outcome":
            return b[b["hospital_expire_flag"].notna()]
        raise ValueError(f"unsupported criterion kind {k!r}")

    def _outcome(self, b: pd.DataFrame, defn: CohortDefinition) -> pd.Series:
        if defn.outcome is None:
            return pd.Series(np.nan, index=b.index)
        ok = defn.outcome.kind
        if ok == "in_hospital_mortality":
            return b["hospital_expire_flag"].astype(int)
        if ok == "icu_mortality":
            return (b["deathtime"].notna() & (b["deathtime"] <= b["outtime"])).astype(int)
        if ok in ("mortality_30d", "mortality_90d", "mortality_1y"):
            days = {"mortality_30d": 30, "mortality_90d": 90, "mortality_1y": 365}[ok]
            return (b["dod"].notna() & ((b["dod"] - b["intime"]).dt.days <= days)).astype(int)
        if ok == "icu_los_gt_hours":
            return (b["icu_los_hours"] > float(defn.outcome.hours)).astype(int)
        if ok == "readmission_30d":
            adm = self.admissions.sort_values(["subject_id", "admittime"]).copy()
            adm["next_admit"] = adm.groupby("subject_id")["admittime"].shift(-1)
            adm["readm"] = ((adm["next_admit"] - adm["dischtime"]).dt.days <= 30).fillna(False).astype(int)
            return b["hadm_id"].map(adm.set_index("hadm_id")["readm"]).fillna(0).astype(int)
        raise ValueError(f"unsupported outcome {ok!r}")

    def run(self, defn: CohortDefinition) -> pd.DataFrame:
        """Return the cohort table (one row per unit) with an ``outcome`` column."""
        b = self._base()
        for c in defn.criteria:
            b = self._apply(b, c)
        b = b.copy()
        b["outcome"] = self._outcome(b, defn)
        if defn.unit == "hadm":
            b = b.sort_values("intime").drop_duplicates("hadm_id")
        elif defn.unit == "subject":
            b = b.sort_values("intime").drop_duplicates("subject_id")
        cols = ["stay_id", "hadm_id", "subject_id", "intime", "age", "icu_los_hours", "outcome"]
        return b[cols].reset_index(drop=True)
