"""Lab- and ECG-defined adverse outcomes in MIMIC-IV (DuckDB SQL templates + incident logic).

Each outcome has (i) the ``labevents`` itemids that define it, (ii) a *baseline-normal* rule
(so that only *incident* events count) and (iii) an *event* rule. The SQL pulls the relevant
lab rows; :func:`incident_outcome` applies the rules relative to each subject's exposure index
time. All thresholds are parameters so that sensitivity analyses are one call away.

MIMIC-IV v3.1 ``hosp/d_labitems`` itemids used (verify with ``verify_itemids``):
    50971 Potassium (blood)          50822 Potassium, whole blood (blood gas)
    50983 Sodium (blood)             50824 Sodium, whole blood
    50912 Creatinine                 50861 ALT   50878 AST   50885 Bilirubin, total   50863 Alk phos
    51265 Platelet count             52075 Absolute neutrophil count   51301 White blood cells
MIMIC-IV-ECG ``machine_measurements.csv``: rr_interval, qrs_onset, t_end (ms) -> QT, QTc (Bazett).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "potassium": (50971, 50822),
    "sodium": (50983, 50824),
    "creatinine": (50912,),
    "alt": (50861,),
    "ast": (50878,),
    "bilirubin": (50885,),
    "alp": (50863,),
    "platelets": (51265,),
    "anc": (52075,),
    "wbc": (51301,),
}

LAB_LABEL_REGEX = {
    "potassium": r"potassium", "sodium": r"sodium", "creatinine": r"creatinine", "alt": r"alanine aminotransferase",
    "ast": r"asparate aminotransferase|aspartate aminotransferase", "bilirubin": r"bilirubin, total",
    "alp": r"alkaline phosphatase", "platelets": r"platelet count", "anc": r"absolute neutrophil", "wbc": r"white blood cells",
}


@dataclass
class OutcomeDefinition:
    name: str
    labs: tuple[str, ...]
    window_days: float
    baseline_days: float
    is_event: Callable[[pd.DataFrame], pd.Series]      # row-wise rule on a wide lab frame
    baseline_ok: Callable[[pd.DataFrame], pd.Series]   # rule on the baseline wide frame
    params: dict[str, Any] = field(default_factory=dict)


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """Long (subject_id, charttime, lab, valuenum) -> wide columns per lab (mean if duplicates)."""
    return df.pivot_table(index=["subject_id", "charttime"], columns="lab", values="valuenum", aggfunc="mean").reset_index()


def _has(w: pd.DataFrame, col: str) -> pd.Series:
    return w[col] if col in w else pd.Series(np.nan, index=w.index)


def make_outcomes(k_hi: float = 5.5, na_lo: float = 130.0, alt_uln: float = 40.0, bili_uln: float = 1.2,
                  plt_lo: float = 100.0, plt_drop: float = 0.5, anc_lo: float = 1.0,
                  windows: dict[str, float] | None = None) -> dict[str, OutcomeDefinition]:
    """Pre-registered outcome definitions (thresholds exposed for sensitivity analyses)."""
    win = {"hyperkalaemia": 7, "hyponatraemia": 7, "aki": 7, "hepatotoxicity": 14, "thrombocytopenia": 14,
           "neutropenia": 14}
    win.update(windows or {})
    defs = {
        "hyperkalaemia": OutcomeDefinition(
            "hyperkalaemia", ("potassium",), win["hyperkalaemia"], 7,
            is_event=lambda w: _has(w, "potassium") >= k_hi,
            baseline_ok=lambda w: _has(w, "potassium") < 5.0, params={"k_hi": k_hi}),
        "hyponatraemia": OutcomeDefinition(
            "hyponatraemia", ("sodium",), win["hyponatraemia"], 7,
            is_event=lambda w: _has(w, "sodium") < na_lo,
            baseline_ok=lambda w: _has(w, "sodium") >= 135, params={"na_lo": na_lo}),
        "aki": OutcomeDefinition(  # KDIGO stage >= 1 vs the baseline creatinine (handled in incident_outcome)
            "aki", ("creatinine",), win["aki"], 7,
            is_event=lambda w: pd.Series(False, index=w.index),  # placeholder: creatinine uses kdigo_event
            baseline_ok=lambda w: _has(w, "creatinine") < 1.5, params={"kdigo": True}),
        "hepatotoxicity": OutcomeDefinition(
            "hepatotoxicity", ("alt", "bilirubin"), win["hepatotoxicity"], 7,
            is_event=lambda w: _has(w, "alt") >= 3 * alt_uln,
            baseline_ok=lambda w: (_has(w, "alt") < 2 * alt_uln) & (_has(w, "bilirubin").fillna(0) < 2 * bili_uln),
            params={"alt_uln": alt_uln, "bili_uln": bili_uln, "severe": "alt>=3xULN and bilirubin>=2xULN"}),
        "thrombocytopenia": OutcomeDefinition(
            "thrombocytopenia", ("platelets",), win["thrombocytopenia"], 7,
            is_event=lambda w: _has(w, "platelets") < plt_lo,
            baseline_ok=lambda w: _has(w, "platelets") >= 150, params={"plt_lo": plt_lo, "plt_drop": plt_drop}),
        "neutropenia": OutcomeDefinition(
            "neutropenia", ("anc",), win["neutropenia"], 7,
            is_event=lambda w: _has(w, "anc") < anc_lo,
            baseline_ok=lambda w: _has(w, "anc") >= 1.5, params={"anc_lo": anc_lo}),
    }
    return defs


# ----------------------------------------------------------------------------------------------
# SQL
# ----------------------------------------------------------------------------------------------


def labs_sql(root: str, labs: tuple[str, ...], subject_filter_sql: str | None = None) -> str:
    """DuckDB SQL: (subject_id, hadm_id, charttime, lab, valuenum) for the requested labs.

    ``subject_filter_sql`` may be a sub-query returning subject_id to restrict the scan
    (labevents has ~120 M rows; DuckDB pushes the IN filter into the CSV scan).
    """
    r = root.rstrip("/")
    cases = " ".join(f"WHEN e.itemid IN ({', '.join(map(str, LAB_ITEMIDS[l]))}) THEN '{l}'" for l in labs)
    ids = ", ".join(str(i) for l in labs for i in LAB_ITEMIDS[l])
    flt = f" AND e.subject_id IN ({subject_filter_sql})" if subject_filter_sql else ""
    return f"""
    SELECT e.subject_id, e.hadm_id, e.charttime, CASE {cases} END AS lab, e.valuenum
    FROM read_csv_auto('{r}/hosp/labevents.csv.gz') e
    WHERE e.itemid IN ({ids}) AND e.valuenum IS NOT NULL{flt}
    """


def exposures_sql(root: str, ingredient_lookup_table: str = "ingredient_lookup") -> str:
    """First inpatient *administration* per (subject, ingredient) with the previous administration
    time of the same ingredient (for the new-user washout). Expects a DuckDB table
    ``ingredient_lookup(ndc, drug, ingredient)`` (from ``rxnorm.RxNavClient.map_prescriptions``).
    """
    r = root.rstrip("/")
    return f"""
    WITH rx AS (
        SELECT p.subject_id, p.hadm_id, p.pharmacy_id, p.ndc, p.drug, l.ingredient
        FROM read_csv_auto('{r}/hosp/prescriptions.csv.gz') p
        JOIN {ingredient_lookup_table} l ON l.ndc = p.ndc AND l.drug = p.drug
        WHERE l.ingredient IS NOT NULL),
    adm AS (
        SELECT e.subject_id, e.hadm_id, e.charttime, rx.ingredient
        FROM read_csv_auto('{r}/hosp/emar.csv.gz') e
        JOIN rx USING (pharmacy_id)
        WHERE e.event_txt = 'Administered')
    SELECT subject_id, hadm_id, ingredient, charttime AS index_time,
           LAG(charttime) OVER (PARTITION BY subject_id, ingredient ORDER BY charttime) AS prev_admin_time,
           ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id, ingredient ORDER BY charttime) AS admin_rank
    FROM adm
    """


def qtc_sql(ecg_root: str, formula: str = "bazett") -> str:
    """QT and QTc (ms) from MIMIC-IV-ECG machine measurements.

    QT = t_end - qrs_onset; RR from rr_interval (ms). Bazett: QTc = QT / sqrt(RR/1000);
    Fridericia: QT / cbrt(RR/1000). Rows with QRS >= 120 ms (qrs_end - qrs_onset) are flagged
    so that bundle-branch-block ECGs can be excluded.
    """
    r = ecg_root.rstrip("/")
    corr = "SQRT(rr_interval / 1000.0)" if formula == "bazett" else "POWER(rr_interval / 1000.0, 1.0/3.0)"
    return f"""
    SELECT subject_id, study_id, ecg_time, rr_interval, (t_end - qrs_onset) AS qt_ms,
           (t_end - qrs_onset) / {corr} AS qtc_ms,
           (qrs_end - qrs_onset) AS qrs_ms,
           ((qrs_end - qrs_onset) >= 120) AS wide_qrs
    FROM read_csv_auto('{r}/machine_measurements.csv')
    WHERE t_end IS NOT NULL AND qrs_onset IS NOT NULL AND rr_interval IS NOT NULL AND rr_interval > 0
    """


def verify_itemids(con: Any, root: str) -> pd.DataFrame:
    """Check LAB_ITEMIDS against hosp/d_labitems labels (fluid = Blood expected)."""
    import re

    d = con.execute(f"SELECT itemid, label, fluid, category FROM read_csv_auto('{root.rstrip('/')}/hosp/d_labitems.csv.gz')").df()
    rows = []
    for lab, ids in LAB_ITEMIDS.items():
        rx = re.compile(LAB_LABEL_REGEX[lab], re.IGNORECASE)
        for i in ids:
            hit = d[d["itemid"] == i]
            label = hit["label"].iloc[0] if len(hit) else None
            rows.append({"lab": lab, "itemid": i, "label": label, "fluid": hit["fluid"].iloc[0] if len(hit) else None,
                         "status": "missing" if label is None else ("ok" if rx.search(str(label)) else "label_mismatch")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# incident-outcome logic (pandas)
# ----------------------------------------------------------------------------------------------


def kdigo_event(creat_post: pd.Series, baseline: float) -> pd.Series:
    """KDIGO creatinine criterion vs the baseline value: >= 1.5x baseline or >= 0.3 mg/dL rise."""
    return (creat_post >= 1.5 * baseline) | (creat_post - baseline >= 0.3)


def incident_outcome(labs_long: pd.DataFrame, index: pd.DataFrame, definition: OutcomeDefinition,
                     require_baseline: bool = True, require_followup_lab: bool = True) -> pd.DataFrame:
    """Per exposure row: baseline value(s), whether baseline is normal, first event time, outcome.

    ``labs_long``: subject_id, charttime, lab, valuenum. ``index``: subject_id, index_time (+ any
    id columns, carried through). Follow-up = (index_time, index_time + window_days]. Returns the
    index frame with columns baseline_<lab>, baseline_ok, n_followup_labs, event_time, outcome
    (1/0) and eligible (baseline ok and, if required, at least one follow-up lab). Rows that are
    not eligible must be excluded from the cohort rather than counted as non-events.
    """
    d = labs_long[labs_long["lab"].isin(definition.labs)].copy()
    d["charttime"] = pd.to_datetime(d["charttime"])
    out = index.copy()
    out["index_time"] = pd.to_datetime(out["index_time"])
    res = []
    by_subject = {s: g.sort_values("charttime") for s, g in d.groupby("subject_id")}
    for row in out.itertuples(index=False):
        g = by_subject.get(row.subject_id)
        rec = {"baseline_ok": False, "n_followup_labs": 0, "event_time": pd.NaT, "outcome": 0}
        if g is None:
            res.append(rec)
            continue
        t0 = row.index_time
        pre = g[(g["charttime"] <= t0) & (g["charttime"] > t0 - pd.Timedelta(days=definition.baseline_days))]
        post = g[(g["charttime"] > t0) & (g["charttime"] <= t0 + pd.Timedelta(days=definition.window_days))]
        base = {}
        for lab in definition.labs:
            p = pre[pre["lab"] == lab]
            base[lab] = float(p["valuenum"].iloc[-1]) if len(p) else np.nan  # last value before index
            rec[f"baseline_{lab}"] = base[lab]
        bw = pd.DataFrame([base])
        rec["baseline_ok"] = bool(definition.baseline_ok(bw).fillna(False).iloc[0])
        rec["n_followup_labs"] = int(len(post))
        if len(post):
            w = _wide(post).sort_values("charttime")
            if definition.params.get("kdigo"):
                ev = kdigo_event(_has(w, "creatinine"), base.get("creatinine", np.nan))
            else:
                ev = definition.is_event(w)
                if "plt_drop" in definition.params and np.isfinite(base.get("platelets", np.nan)):
                    ev = ev | (_has(w, "platelets") <= (1 - definition.params["plt_drop"]) * base["platelets"])
            ev = ev.fillna(False)
            if ev.any():
                rec["event_time"] = w.loc[ev, "charttime"].iloc[0]
                rec["outcome"] = 1
        res.append(rec)
    res_df = pd.DataFrame(res, index=out.index)
    out = pd.concat([out, res_df], axis=1)
    elig = pd.Series(True, index=out.index)
    if require_baseline:
        elig &= out["baseline_ok"]
    if require_followup_lab:
        elig &= out["n_followup_labs"] > 0
    out["eligible"] = elig
    return out


def qtc_outcome(qtc: pd.DataFrame, index: pd.DataFrame, window_days: float = 3.0, baseline_days: float = 7.0,
                qtc_threshold: float = 500.0, delta_threshold: float = 60.0, exclude_wide_qrs: bool = True) -> pd.DataFrame:
    """QTc >= 500 ms or delta-QTc >= 60 ms vs the last pre-index ECG (needs a baseline ECG)."""
    q = qtc.copy()
    q["ecg_time"] = pd.to_datetime(q["ecg_time"])
    if exclude_wide_qrs and "wide_qrs" in q:
        q = q[~q["wide_qrs"].astype(bool)]
    out = index.copy()
    out["index_time"] = pd.to_datetime(out["index_time"])
    res = []
    by_subject = {s: g.sort_values("ecg_time") for s, g in q.groupby("subject_id")}
    for row in out.itertuples(index=False):
        g = by_subject.get(row.subject_id)
        rec = {"baseline_qtc": np.nan, "max_qtc": np.nan, "delta_qtc": np.nan, "outcome": 0, "eligible": False}
        if g is not None:
            t0 = row.index_time
            pre = g[(g["ecg_time"] <= t0) & (g["ecg_time"] > t0 - pd.Timedelta(days=baseline_days))]
            post = g[(g["ecg_time"] > t0) & (g["ecg_time"] <= t0 + pd.Timedelta(days=window_days))]
            if len(pre) and len(post):
                b = float(pre["qtc_ms"].iloc[-1])
                m = float(post["qtc_ms"].max())
                rec.update(baseline_qtc=b, max_qtc=m, delta_qtc=m - b, eligible=bool(b < qtc_threshold),
                           outcome=int((m >= qtc_threshold) or (m - b >= delta_threshold)))
        res.append(rec)
    return pd.concat([out, pd.DataFrame(res, index=out.index)], axis=1)
