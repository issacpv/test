"""Build dose, level and course tables from MIMIC-IV, and simulate synthetic courses.

Three dose sources exist in MIMIC-IV and are kept apart through a ``source`` column:

* ``inputevents`` (ICU module): ``starttime``, ``endtime``, ``amount``, ``amountuom``, ``rate``, ``patientweight``.
  Highest fidelity for infusion timing while the patient is in the ICU.
* ``emar`` + ``emar_detail`` (hosp module): barcode-scanned administrations with ``charttime`` (administration
  time), ``dose_given`` / ``dose_given_unit`` and, sometimes, ``infusion_rate``. Available for admissions after
  the eMAR go-live only; infusion duration must often be imputed.
* ``prescriptions`` (hosp module): ordered doses with ``starttime``/``stoptime`` (order validity, not
  administration); used only as a scheduled-time proxy.

Item ids are never hard-coded: :func:`resolve_itemids` selects rows of ``d_items`` / ``d_labitems`` by a regex
on the label and the caller records the result.  Times inside the PK code are floating-point hours relative
to the course start (``time_h``).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .pk_models import ModelSpec, concentration

MG_PER_UNIT = {"mg": 1.0, "g": 1000.0, "gm": 1000.0, "gram": 1000.0, "grams": 1000.0, "mcg": 0.001, "ug": 0.001}
LEVEL_UNITS_MG_L = {"mg/l", "ug/ml", "mcg/ml", "µg/ml"}


# ------------------------------------------------------------------------------------------------- item ids
def resolve_itemids(dictionary: pd.DataFrame, pattern: str, label_col: str = "label", id_col: str = "itemid") -> pd.DataFrame:
    """Rows of a dictionary table whose label matches ``pattern`` (case-insensitive regex).

    Works for ``d_items`` (icu) and ``d_labitems`` (hosp). The caller should print/save the result so that the
    ids used in a run are documented.
    """
    mask = dictionary[label_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    return dictionary.loc[mask, [id_col, label_col]].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ doses
def _to_mg(amount: pd.Series, unit: pd.Series) -> pd.Series:
    factor = unit.astype(str).str.strip().str.lower().map(MG_PER_UNIT)
    return amount.astype(float) * factor


def doses_from_inputevents(inputevents: pd.DataFrame, itemids: Iterable[int]) -> pd.DataFrame:
    """Dose events from ICU ``inputevents`` rows for the given item ids.

    Returns columns ``subject_id, hadm_id, stay_id, start, end, amount_mg, weight_kg, source``. Rows with unknown
    units or non-positive amounts are dropped; ``end <= start`` rows are kept and handled as boluses downstream.
    """
    ids = set(int(i) for i in itemids)
    d = inputevents[inputevents["itemid"].isin(ids)].copy()
    d["amount_mg"] = _to_mg(d["amount"], d["amountuom"])
    d = d[d["amount_mg"].notna() & (d["amount_mg"] > 0)]
    out = pd.DataFrame(
        {
            "subject_id": d["subject_id"].astype(int),
            "hadm_id": d["hadm_id"],
            "stay_id": d.get("stay_id"),
            "start": pd.to_datetime(d["starttime"]),
            "end": pd.to_datetime(d["endtime"]),
            "amount_mg": d["amount_mg"].astype(float),
            "weight_kg": d["patientweight"].astype(float) if "patientweight" in d else np.nan,
            "source": "inputevents",
        }
    )
    return out.sort_values(["subject_id", "start"]).reset_index(drop=True)


def doses_from_emar(
    emar: pd.DataFrame,
    emar_detail: pd.DataFrame,
    medication_pattern: str = r"vancomycin",
    hours_per_gram: float = 1.0,
    keep_events: Sequence[str] = ("Administered", "Delayed Administered"),
) -> pd.DataFrame:
    """Dose events from ``emar`` joined to ``emar_detail``.

    ``emar_detail`` may hold several rows per ``emar_id`` (one per product); rows with ``dose_given`` are used.
    Infusion duration is ``dose_given / infusion_rate`` when a rate in mg/h (or mg/min) is present, otherwise
    ``hours_per_gram * dose_g`` (the common 1 g/h convention; make this a study variable, see README).
    """
    e = emar[emar["medication"].astype(str).str.contains(medication_pattern, case=False, regex=True, na=False)]
    e = e[e["event_txt"].isin(list(keep_events))]
    det = emar_detail[emar_detail["dose_given"].notna()][["emar_id", "dose_given", "dose_given_unit", "infusion_rate", "infusion_rate_unit"]]
    m = e.merge(det, on="emar_id", how="inner")
    m["amount_mg"] = _to_mg(pd.to_numeric(m["dose_given"], errors="coerce"), m["dose_given_unit"])
    m = m[m["amount_mg"].notna() & (m["amount_mg"] > 0)].copy()
    rate = pd.to_numeric(m["infusion_rate"], errors="coerce")
    unit = m["infusion_rate_unit"].astype(str).str.lower()
    rate_mg_h = np.where(unit.str.contains("min"), rate * 60.0, rate)
    dur_h = np.where(np.isfinite(rate_mg_h) & (rate_mg_h > 0), m["amount_mg"] / rate_mg_h, hours_per_gram * m["amount_mg"] / 1000.0)
    m["start"] = pd.to_datetime(m["charttime"])
    m["end"] = m["start"] + pd.to_timedelta(dur_h, unit="h")
    m["duration_imputed"] = ~(np.isfinite(rate_mg_h) & (rate_mg_h > 0))
    out = m[["subject_id", "hadm_id", "start", "end", "amount_mg", "duration_imputed"]].copy()
    out["stay_id"] = np.nan
    out["weight_kg"] = np.nan
    out["source"] = "emar"
    return out.sort_values(["subject_id", "start"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- levels
def levels_from_labevents(labevents: pd.DataFrame, d_labitems: pd.DataFrame, pattern: str = r"vancomycin") -> pd.DataFrame:
    """Drug levels (mg/L) from ``labevents`` for lab items whose label matches ``pattern``."""
    items = resolve_itemids(d_labitems, pattern)
    lv = labevents[labevents["itemid"].isin(items["itemid"])].copy()
    lv["value_mg_L"] = pd.to_numeric(lv["valuenum"], errors="coerce")
    unit_ok = lv["valueuom"].astype(str).str.lower().isin(LEVEL_UNITS_MG_L) | lv["valueuom"].isna()
    lv = lv[lv["value_mg_L"].notna() & (lv["value_mg_L"] > 0) & unit_ok]
    out = pd.DataFrame(
        {
            "subject_id": lv["subject_id"].astype(int),
            "hadm_id": lv["hadm_id"],
            "charttime": pd.to_datetime(lv["charttime"]),
            "value_mg_L": lv["value_mg_L"].astype(float),
            "itemid": lv["itemid"],
        }
    )
    return out.sort_values(["subject_id", "charttime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- courses
def assemble_courses(doses: pd.DataFrame, gap_hours: float = 72.0) -> pd.DataFrame:
    """Assign a ``course_id`` to doses: a new course starts when the gap to the previous dose end exceeds ``gap_hours``.

    Returns the dose table with ``course_id`` (``"<subject_id>-<k>"``), ``course_start`` and ``time_h``
    (hours from course start to dose start) and ``end_h``.
    """
    d = doses.sort_values(["subject_id", "start"]).reset_index(drop=True).copy()
    course_ids = []
    starts = []
    for sid, grp in d.groupby("subject_id", sort=False):
        k = 0
        prev_end = None
        cur_start = None
        for idx, row in grp.iterrows():
            if prev_end is None or (row["start"] - prev_end) > pd.Timedelta(hours=gap_hours):
                k += 1
                cur_start = row["start"]
            course_ids.append((idx, f"{sid}-{k}"))
            starts.append((idx, cur_start))
            prev_end = max(row["end"], prev_end) if prev_end is not None else row["end"]
    d["course_id"] = pd.Series(dict(course_ids))
    d["course_start"] = pd.Series(dict(starts))
    d["time_h"] = (d["start"] - d["course_start"]).dt.total_seconds() / 3600.0
    d["end_h"] = (d["end"] - d["course_start"]).dt.total_seconds() / 3600.0
    return d


def attach_levels(doses_with_courses: pd.DataFrame, levels: pd.DataFrame, tail_hours: float = 72.0, trough_window_h: float = 1.0) -> pd.DataFrame:
    """Attach levels to courses and infer trough status from the dosing schedule.

    A level belongs to the course whose ``[course_start, last dose end + tail_hours]`` window contains it.
    ``trough_inferred`` is True when the next dose in the course starts within ``trough_window_h`` after the draw;
    ``during_infusion`` flags draws inside any dose interval (a data-quality stratum, see README).
    """
    rows = []
    for cid, grp in doses_with_courses.groupby("course_id", sort=False):
        sid = int(grp["subject_id"].iloc[0])
        c_start = grp["course_start"].iloc[0]
        c_end = grp["end"].max() + pd.Timedelta(hours=tail_hours)
        lv = levels[(levels["subject_id"] == sid) & (levels["charttime"] >= c_start) & (levels["charttime"] <= c_end)]
        for _, r in lv.iterrows():
            t = r["charttime"]
            later = grp[grp["start"] > t]
            next_gap_h = (later["start"].min() - t).total_seconds() / 3600.0 if len(later) else np.nan
            during = bool(((grp["start"] <= t) & (grp["end"] >= t)).any())
            rows.append(
                {
                    "course_id": cid,
                    "subject_id": sid,
                    "charttime": t,
                    "time_h": (t - c_start).total_seconds() / 3600.0,
                    "value": float(r["value_mg_L"]),
                    "hours_to_next_dose": next_gap_h,
                    "trough_inferred": bool(np.isfinite(next_gap_h) and 0.0 < next_gap_h <= trough_window_h),
                    "during_infusion": during,
                }
            )
    cols = ["course_id", "subject_id", "charttime", "time_h", "value", "hours_to_next_dose", "trough_inferred", "during_infusion"]
    return pd.DataFrame(rows, columns=cols).sort_values(["course_id", "time_h"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------- simulation
def simulate_course(
    spec: ModelSpec,
    covariates: Dict[str, float],
    eta_true: Sequence[float],
    n_doses: int = 6,
    interval_h: float = 12.0,
    dose_mg: float = 1000.0,
    infusion_h: float = 1.0,
    level_times_h: Optional[Sequence[float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, pd.DataFrame]:
    """Simulate a regular dosing course and noisy levels from ``spec`` with known random effects.

    Returns ``{"doses": DataFrame(start, end, amount_mg), "levels": DataFrame(time_h, value, true_value)}`` with
    times in hours from the first dose. Residual error follows the model's proportional + additive scheme.
    """
    rng = np.random.default_rng() if rng is None else rng
    starts = np.arange(n_doses) * interval_h
    doses = pd.DataFrame({"start": starts, "end": starts + infusion_h, "amount_mg": dose_mg})
    if level_times_h is None:
        level_times_h = [interval_h * 2 - 0.5, interval_h * 4 - 0.5]  # troughs before dose 3 and dose 5
    lt = np.asarray(level_times_h, dtype=float)
    params = spec.individual_params(covariates, np.asarray(eta_true, dtype=float))
    true = concentration(spec, params, doses, lt)
    sd = np.sqrt((spec.sigma_prop * true) ** 2 + spec.sigma_add**2)
    obs = np.clip(true + rng.normal(0, sd), 0.1, None)
    levels = pd.DataFrame({"time_h": lt, "value": obs, "true_value": true})
    return {"doses": doses, "levels": levels, "params": params}
