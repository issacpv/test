"""Cohort and outcome construction for MIMIC-IV-ED.

Two equivalent implementations are provided:

* :func:`build_cohort_duckdb` - SQL over the raw ``.csv.gz`` files (fast, ~450k
  stays in seconds, no database server).  Requires ``duckdb``.
* :func:`build_cohort_pandas` - pure pandas, used by the tests and for the
  open demo; identical semantics.

Outcome definitions follow Xie et al. (2022, Sci Data; nliulab/mimic4ed-benchmark):

``hospitalization``   ED disposition ``ADMITTED`` and a linked ``hadm_id``
``critical_outcome``  ICU admission (``icustays.intime``) within 12 h of ED ``outtime``
                      OR in-hospital death (``admissions.deathtime``) within 12 h of ED ``outtime``
``revisit_72h``       another ED stay of the same subject starting within 72 h of ``outtime``

Extensions for the equity / drift analyses:

``age``               ``anchor_age + (year(intime) - anchor_year)`` (adults: >= 18)
``era``               ``anchor_year_group`` of the patient (the only calendar-anchored field)
``language``/``insurance`` from ``admissions`` of the same visit, else the subject's most recent admission
``n_meds_pyxis``      number of dispensing events during the stay (resource-use proxy)

Differences from Xie et al. are limited to the extra columns; inclusion criteria
(adult, non-missing ``acuity``) are reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CRITICAL_WINDOW_H = 12.0
REVISIT_WINDOW_H = 72.0

RACE_MAP: dict[str, str] = {
    "WHITE": "White", "BLACK": "Black", "HISPANIC": "Hispanic", "ASIAN": "Asian",
}
"""Collapse MIMIC free-text race strings by prefix into 6 categories (White, Black, Hispanic, Asian, Other, Unknown)."""


def collapse_race(race: pd.Series) -> pd.Series:
    """Map MIMIC race strings (e.g. 'WHITE - RUSSIAN', 'BLACK/AFRICAN AMERICAN') to 6 categories."""
    s = race.fillna("UNKNOWN").astype(str).str.upper()
    out = pd.Series("Other", index=s.index, dtype=object)
    for prefix, label in RACE_MAP.items():
        out[s.str.startswith(prefix)] = label
    out[s.str.contains("HISPANIC|LATINO", regex=True)] = "Hispanic"
    out[s.str.contains("UNKNOWN|UNABLE|DECLINED|PATIENT DECLINED", regex=True)] = "Unknown"
    return out


# ----------------------------------------------------------------------------
# DuckDB implementation
# ----------------------------------------------------------------------------
COHORT_SQL = """
WITH ed AS (
    SELECT e.subject_id, e.hadm_id, e.stay_id, e.intime, e.outtime, e.gender, e.race,
           e.arrival_transport, e.disposition,
           t.temperature, t.heartrate, t.resprate, t.o2sat, t.sbp, t.dbp, t.pain, t.acuity, t.chiefcomplaint
    FROM read_csv_auto('{ed}/edstays.csv.gz') e
    LEFT JOIN read_csv_auto('{ed}/triage.csv.gz') t USING (stay_id)
),
pat AS (
    SELECT subject_id, anchor_age, anchor_year, anchor_year_group, dod
    FROM read_csv_auto('{hosp}/patients.csv.gz')
),
adm AS (
    SELECT subject_id, hadm_id, admittime, deathtime, language, insurance
    FROM read_csv_auto('{hosp}/admissions.csv.gz')
),
icu AS (
    SELECT subject_id, hadm_id, intime AS icu_intime FROM read_csv_auto('{icu}/icustays.csv.gz')
),
pyx AS (
    SELECT stay_id, COUNT(*) AS n_meds_pyxis FROM read_csv_auto('{ed}/pyxis.csv.gz') GROUP BY stay_id
),
base AS (
    SELECT ed.*, pat.anchor_age, pat.anchor_year, pat.anchor_year_group AS era, pat.dod,
           pat.anchor_age + (EXTRACT(year FROM ed.intime) - pat.anchor_year) AS age,
           adm.deathtime, adm.language, adm.insurance,
           COALESCE(pyx.n_meds_pyxis, 0) AS n_meds_pyxis
    FROM ed
    LEFT JOIN pat USING (subject_id)
    LEFT JOIN adm ON ed.hadm_id = adm.hadm_id
    LEFT JOIN pyx USING (stay_id)
),
icu12 AS (
    SELECT b.stay_id, MIN(i.icu_intime) AS first_icu
    FROM base b JOIN icu i ON b.hadm_id = i.hadm_id
    WHERE i.icu_intime >= b.outtime AND i.icu_intime <= b.outtime + INTERVAL '{crit_h} hours'
    GROUP BY b.stay_id
),
nxt AS (
    SELECT stay_id, LEAD(intime) OVER (PARTITION BY subject_id ORDER BY intime) AS next_intime FROM ed
)
SELECT b.*,
       CASE WHEN b.disposition = 'ADMITTED' AND b.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS hospitalization,
       CASE WHEN icu12.first_icu IS NOT NULL
              OR (b.deathtime IS NOT NULL AND b.deathtime >= b.outtime
                  AND b.deathtime <= b.outtime + INTERVAL '{crit_h} hours') THEN 1 ELSE 0 END AS critical_outcome,
       CASE WHEN nxt.next_intime IS NOT NULL
              AND nxt.next_intime <= b.outtime + INTERVAL '{rev_h} hours' THEN 1 ELSE 0 END AS revisit_72h
FROM base b
LEFT JOIN icu12 USING (stay_id)
LEFT JOIN nxt USING (stay_id)
WHERE b.age >= 18 AND b.acuity IS NOT NULL
"""


@dataclass
class DataPaths:
    """Locations of the raw MIMIC directories (see data/README.md)."""

    ed: Path
    hosp: Path
    icu: Path

    @classmethod
    def from_root(cls, root: Path | str, demo: bool = False) -> "DataPaths":
        root = Path(root)
        if demo:
            return cls(root / "mimic-iv-ed-demo/2.2/ed", root / "mimic-iv-demo/2.2/hosp", root / "mimic-iv-demo/2.2/icu")
        return cls(root / "mimic-iv-ed/2.2/ed", root / "mimiciv/3.1/hosp", root / "mimiciv/3.1/icu")


def build_cohort_duckdb(paths: DataPaths, con: Any | None = None) -> pd.DataFrame:
    """Run :data:`COHORT_SQL` with DuckDB and post-process (race collapse, language propagation)."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install duckdb, or use build_cohort_pandas") from exc
    con = con or duckdb.connect()
    sql = COHORT_SQL.format(ed=paths.ed, hosp=paths.hosp, icu=paths.icu, crit_h=int(CRITICAL_WINDOW_H), rev_h=int(REVISIT_WINDOW_H))
    df = con.execute(sql).df()
    adm = con.execute(f"SELECT subject_id, admittime, language, insurance FROM read_csv_auto('{paths.hosp}/admissions.csv.gz')").df()
    return _postprocess(df, adm)


# ----------------------------------------------------------------------------
# pandas implementation (identical semantics)
# ----------------------------------------------------------------------------
def _ensure_columns(df: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
    """Return ``df`` with at least ``cols`` present (empty tables from the demo may have no columns)."""
    out = pd.DataFrame(columns=cols) if df is None or len(df.columns) == 0 else df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_cohort_pandas(edstays: pd.DataFrame, triage: pd.DataFrame, patients: pd.DataFrame,
                        admissions: pd.DataFrame, icustays: pd.DataFrame, pyxis: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the adult ED cohort with Xie et al. outcomes from in-memory tables."""
    admissions = _ensure_columns(admissions, ["subject_id", "hadm_id", "admittime", "deathtime", "language", "insurance"])
    icustays = _ensure_columns(icustays, ["subject_id", "hadm_id", "intime"])
    ed = edstays.merge(triage.drop(columns=[c for c in ("subject_id",) if c in triage.columns]), on="stay_id", how="left")
    for c in ("intime", "outtime"):
        ed[c] = pd.to_datetime(ed[c])
    pat = patients[["subject_id", "anchor_age", "anchor_year", "anchor_year_group", "dod"]].rename(columns={"anchor_year_group": "era"})
    df = ed.merge(pat, on="subject_id", how="left")
    df["age"] = df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])
    adm = admissions[["hadm_id", "deathtime", "language", "insurance"]].copy()
    adm["deathtime"] = pd.to_datetime(adm["deathtime"])
    df = df.merge(adm, on="hadm_id", how="left")
    if pyxis is not None and len(pyxis):
        n = pyxis.groupby("stay_id").size().rename("n_meds_pyxis")
        df = df.merge(n, left_on="stay_id", right_index=True, how="left")
    df["n_meds_pyxis"] = df.get("n_meds_pyxis", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # outcomes
    df["hospitalization"] = ((df["disposition"] == "ADMITTED") & df["hadm_id"].notna()).astype(int)
    icu = icustays[["hadm_id", "intime"]].rename(columns={"intime": "icu_intime"}).copy()
    icu["icu_intime"] = pd.to_datetime(icu["icu_intime"])
    j = df[["stay_id", "hadm_id", "outtime"]].merge(icu, on="hadm_id", how="inner")
    win = pd.Timedelta(hours=CRITICAL_WINDOW_H)
    j = j[(j["icu_intime"] >= j["outtime"]) & (j["icu_intime"] <= j["outtime"] + win)]
    icu_flag = df["stay_id"].isin(j["stay_id"])
    death_flag = df["deathtime"].notna() & (df["deathtime"] >= df["outtime"]) & (df["deathtime"] <= df["outtime"] + win)
    df["critical_outcome"] = (icu_flag | death_flag).astype(int)
    df = df.sort_values(["subject_id", "intime"])
    nxt = df.groupby("subject_id")["intime"].shift(-1)
    df["revisit_72h"] = (nxt.notna() & (nxt <= df["outtime"] + pd.Timedelta(hours=REVISIT_WINDOW_H))).astype(int)

    df = df[(df["age"] >= 18) & df["acuity"].notna()].reset_index(drop=True)
    return _postprocess(df, admissions[["subject_id", "admittime", "language", "insurance"]])


def _postprocess(df: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """Race collapse, language/insurance propagation from the subject's latest admission, derived flags."""
    df = df.copy()
    df["race6"] = collapse_race(df["race"])
    latest = admissions.copy()
    latest["admittime"] = pd.to_datetime(latest["admittime"])
    latest = latest.sort_values("admittime").groupby("subject_id").tail(1).set_index("subject_id")
    for col in ("language", "insurance"):
        fallback = df["subject_id"].map(latest[col])
        df[col] = df[col].where(df[col].notna(), fallback)
    df["nonenglish"] = np.where(df["language"].isna(), np.nan, (df["language"].astype(str).str.upper() != "ENGLISH").astype(float))
    df["acuity"] = df["acuity"].astype(int)
    df["age_band"] = pd.cut(df["age"], [17, 39, 64, 200], labels=["18-39", "40-64", "65+"]).astype(str)
    df["ed_los_h"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    return df


STRUCTURED_FEATURES: tuple[str, ...] = (
    "age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain_num", "acuity", "n_meds_pyxis",
)


def structured_matrix(df: pd.DataFrame, features: tuple[str, ...] = STRUCTURED_FEATURES) -> tuple[np.ndarray, list[str]]:
    """Numeric design matrix for the Xie-style structured model (pain parsed to numeric; NaN kept for imputers)."""
    d = df.copy()
    d["pain_num"] = pd.to_numeric(d.get("pain"), errors="coerce")
    d["pain_num"] = d["pain_num"].where(d["pain_num"].between(0, 10))
    cols = [c for c in features if c in d.columns]
    X = d[cols].astype(float).to_numpy()
    for tr in ("AMBULANCE", "WALK IN", "HELICOPTER"):
        X = np.c_[X, (d["arrival_transport"].astype(str).str.upper() == tr).astype(float).to_numpy()]
        cols.append(f"arrival_{tr.lower().replace(' ', '_')}")
    return X, cols
