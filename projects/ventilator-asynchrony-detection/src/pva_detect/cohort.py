"""Charted-ventilator extraction templates (MIMIC-IV, eICU, HiRID) and the exposure-outcome analysis.

Variable identifiers below are the usual ones but MUST be verified against each database's
dictionary (``d_items``, ``respiratorycharting.respchartvaluelabel``, ``hirid_variable_reference.csv``)
with the helper lookups before any extraction is trusted.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# MIMIC-IV icu.chartevents itemids commonly used for ventilator settings/observations (verify with d_items)
MIMIC_VENT_ITEMS: dict[str, int] = {
    "rr_set": 224688, "rr_spont": 224689, "rr_total": 224690, "vt_set": 224684, "vt_obs": 224685,
    "minute_volume": 224687, "pip": 224695, "plateau": 224696, "peep_set": 220339, "vent_mode": 223849,
}

# Regex hints for HiRID's variable reference ("Variable Name" column) and eICU respiratoryCharting labels
HIRID_HINTS: dict[str, str] = {
    "rr_set": r"respiratory rate.*set|set.*respiratory rate|RR\s*set",
    "rr_total": r"respiratory rate.*(?:measured|total|patient)|^respiratory rate$",
    "rr_monitor": r"respiratory rate.*(?:monitor|impedance|ECG)",
    "vt_obs": r"tidal volume.*(?:exp|measured|observed)|^tidal volume$",
    "pip": r"peak.*pressure",
    "peep_set": r"PEEP",
    "minute_volume": r"minute volume",
    "vent_mode": r"ventilat.*mode",
}
EICU_HINTS: dict[str, str] = {
    "rr_set": r"vent rate|set rate|RR.*set",
    "rr_total": r"total RR|RR \(patient\)|resp rate total",
    "vt_obs": r"exhaled TV|tidal volume.*(?:observed|exhaled)",
    "pip": r"peak insp",
    "peep_set": r"PEEP",
    "minute_volume": r"minute volume|exhaled MV",
    "vent_mode": r"mode",
}
# Bedside-monitor (impedance) respiratory rate: MIMIC-IV chartevents 220210 'Respiratory Rate';
# eICU vitalPeriodic.respiration; HiRID monitor RR (look up with HIRID_HINTS['rr_monitor']).
MIMIC_MONITOR_RR_ITEM = 220210


def mimic_vent_sql(root: str, items: dict[str, int] = MIMIC_VENT_ITEMS) -> str:
    ids = ",".join(str(v) for v in items.values())
    return f"""
    SELECT c.stay_id, c.itemid, date_diff('second', i.intime, c.charttime) AS t_s, c.valuenum, c.value
    FROM read_csv_auto('{root}/icu/chartevents.csv.gz') c
    JOIN read_csv_auto('{root}/icu/icustays.csv.gz') i USING (stay_id)
    WHERE c.itemid IN ({ids})
    """


def eicu_resp_sql(root: str) -> str:
    return f"""
    SELECT patientunitstayid AS stay_id, respchartoffset * 60 AS t_s, respchartvaluelabel AS label, respchartvalue AS value
    FROM read_csv_auto('{root}/respiratoryCharting.csv.gz')
    """


def lookup_variables(reference: pd.DataFrame, hints: dict[str, str], name_col: str, id_col: str) -> pd.DataFrame:
    """Candidate (concept, variable id, name) rows whose name matches each regex hint; for manual verification."""
    rows = []
    for concept, rx in hints.items():
        m = reference[name_col].astype(str).str.contains(rx, case=False, regex=True, na=False)
        for _, r in reference.loc[m].iterrows():
            rows.append({"concept": concept, "variable_id": r[id_col], "name": r[name_col]})
    return pd.DataFrame(rows, columns=["concept", "variable_id", "name"])


def pivot_charted(long: pd.DataFrame, concept_map: dict[str, int | str], id_col: str = "itemid") -> pd.DataFrame:
    """Long (stay_id, t_s, id, valuenum) -> wide (stay_id, t_s, rr_set, rr_total, vt_obs, pip, minute_volume, ...)."""
    inv = {v: k for k, v in concept_map.items()}
    df = long[long[id_col].isin(inv)].assign(concept=lambda d: d[id_col].map(inv))
    wide = df.pivot_table(index=["stay_id", "t_s"], columns="concept", values="valuenum", aggfunc="mean").reset_index()
    wide.columns.name = None
    return wide.sort_values(["stay_id", "t_s"], ignore_index=True)


def exposure_table(windows: pd.DataFrame, ai_col: str = "ai_pred", threshold: float = 10.0) -> pd.DataFrame:
    """Per-stay exposure: mean predicted AI, fraction of windows with AI >= threshold, ventilated hours."""
    g = windows.groupby("stay_id")
    return pd.DataFrame({
        "ai_mean": g[ai_col].mean(),
        "frac_windows_high": g[ai_col].apply(lambda s: float((s >= threshold).mean())),
        "n_windows": g.size(),
    }).reset_index()


def logistic_association(df: pd.DataFrame, exposure: str, outcome: str, covariates: Iterable[str] = (),
                         cluster: str | None = None) -> dict[str, float]:
    """Odds ratio (per unit exposure) with 95% CI from a logistic regression (statsmodels), optional cluster-robust SE."""
    import statsmodels.api as sm

    cols = [exposure, *covariates]
    X = sm.add_constant(df[cols].astype(float))
    y = df[outcome].astype(float)
    model = sm.Logit(y, X)
    if cluster is not None:
        res = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": df[cluster].to_numpy()})
    else:
        res = model.fit(disp=0)
    ci = res.conf_int().loc[exposure]
    return {"or": float(np.exp(res.params[exposure])), "ci_low": float(np.exp(ci[0])), "ci_high": float(np.exp(ci[1])),
            "p": float(res.pvalues[exposure]), "n": int(len(df))}


def parse_vent_mode(value: str) -> str:
    """Coarse ventilator mode class from free-text mode strings (MIMIC 'CMV/ASSIST', 'PSV/SBT', eICU, HiRID)."""
    v = str(value).upper()
    if re.search(r"PSV|PS\b|PRESSURE SUPPORT|CPAP|SBT|SPONT", v):
        return "spontaneous"
    if re.search(r"CMV|ASSIST|AC\b|A/C|VC|PC\b|PRVC|SIMV|APRV|BILEVEL|MMV", v):
        return "controlled"
    return "unknown"
