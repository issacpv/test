"""Build an offline-RL dataset (states, discretised actions, rewards) from ICU ventilation records.

Pipeline: SQL (DuckDB) -> long event table (stay_id, time_h, variable, value) -> 4-h wide bins ->
forward fill -> action discretisation (VT/kg PBW x PEEP x FiO2) -> `Trajectories`.

The SQL templates target the `mimic-code` derived schema for MIMIC-IV and the raw eICU-CRD tables.
They are strings so they can be inspected/ported; `run_sql` needs `duckdb` (lazy import).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------- SQL
# MIMIC-IV (mimic-code derived concepts in DuckDB). One row per charted value, hours since ventilation start.
MIMIC_VENT_SQL = """
WITH vent AS (
    SELECT v.stay_id, v.starttime, v.endtime, i.subject_id, i.hadm_id, i.intime
    FROM mimiciv_derived.ventilation v
    JOIN mimiciv_icu.icustays i USING (stay_id)
    WHERE v.ventilation_status = 'InvasiveVent'
      AND date_diff('hour', v.starttime, v.endtime) >= 24
    QUALIFY row_number() OVER (PARTITION BY v.stay_id ORDER BY v.starttime) = 1
),
settings AS (
    SELECT s.stay_id, s.charttime, 'tidal_volume_set' AS variable, s.tidal_volume_set AS value FROM mimiciv_derived.ventilator_setting s
    UNION ALL SELECT stay_id, charttime, 'tidal_volume_observed', tidal_volume_observed FROM mimiciv_derived.ventilator_setting
    UNION ALL SELECT stay_id, charttime, 'peep', peep FROM mimiciv_derived.ventilator_setting
    UNION ALL SELECT stay_id, charttime, 'fio2', fio2 / 100.0 FROM mimiciv_derived.ventilator_setting
    UNION ALL SELECT stay_id, charttime, 'plateau_pressure', plateau_pressure FROM mimiciv_derived.ventilator_setting
    UNION ALL SELECT stay_id, charttime, 'resp_rate_set', respiratory_rate_set FROM mimiciv_derived.ventilator_setting
    UNION ALL SELECT stay_id, charttime, 'minute_volume', minute_volume FROM mimiciv_derived.ventilator_setting
),
gas AS (
    SELECT b.hadm_id, b.charttime, 'po2' AS variable, b.po2 AS value FROM mimiciv_derived.bg b WHERE b.specimen = 'ART.'
    UNION ALL SELECT hadm_id, charttime, 'pco2', pco2 FROM mimiciv_derived.bg WHERE specimen = 'ART.'
    UNION ALL SELECT hadm_id, charttime, 'ph', ph FROM mimiciv_derived.bg WHERE specimen = 'ART.'
    UNION ALL SELECT hadm_id, charttime, 'lactate', lactate FROM mimiciv_derived.bg WHERE specimen = 'ART.'
),
vitals AS (
    SELECT stay_id, charttime, 'heart_rate' AS variable, heart_rate AS value FROM mimiciv_derived.vitalsign
    UNION ALL SELECT stay_id, charttime, 'mbp', mbp FROM mimiciv_derived.vitalsign
    UNION ALL SELECT stay_id, charttime, 'resp_rate', resp_rate FROM mimiciv_derived.vitalsign
    UNION ALL SELECT stay_id, charttime, 'spo2', spo2 / 100.0 FROM mimiciv_derived.vitalsign
    UNION ALL SELECT stay_id, charttime, 'temperature', temperature FROM mimiciv_derived.vitalsign
)
SELECT v.stay_id, date_diff('minute', v.starttime, e.charttime) / 60.0 AS time_h, e.variable, e.value
FROM vent v
JOIN (
    SELECT stay_id, charttime, variable, value FROM settings
    UNION ALL SELECT stay_id, charttime, variable, value FROM vitals
    UNION ALL SELECT vv.stay_id, g.charttime, g.variable, g.value FROM gas g JOIN vent vv USING (hadm_id)
) e USING (stay_id)
WHERE e.value IS NOT NULL AND e.charttime BETWEEN v.starttime AND LEAST(v.endtime, v.starttime + INTERVAL 7 DAY)
"""

MIMIC_OUTCOME_SQL = """
SELECT i.stay_id, i.subject_id, p.gender, p.anchor_age AS age,
       CASE WHEN a.hospital_expire_flag = 1 THEN 1 ELSE 0 END AS died,
       (SELECT result_value::DOUBLE FROM mimiciv_hosp.omr o WHERE o.subject_id = i.subject_id AND o.result_name = 'Height (Inches)'
        ORDER BY abs(date_diff('day', o.chartdate, i.intime::DATE)) LIMIT 1) * 2.54 AS height_cm
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a USING (hadm_id)
JOIN mimiciv_hosp.patients p USING (subject_id)
"""

# eICU-CRD: offsets are minutes from ICU admission; respiratorycharting holds ventilator settings.
EICU_VENT_SQL = """
WITH rc AS (
    SELECT patientunitstayid, respchartoffset / 60.0 AS time_h,
           CASE respchartvaluelabel
                WHEN 'Tidal Volume (set)' THEN 'tidal_volume_set'
                WHEN 'Tidal Volume Observed (VT)' THEN 'tidal_volume_observed'
                WHEN 'PEEP' THEN 'peep'
                WHEN 'FiO2' THEN 'fio2'
                WHEN 'Plateau Pressure' THEN 'plateau_pressure'
                WHEN 'Total RR' THEN 'resp_rate'
                WHEN 'Mean Airway Pressure' THEN 'mean_airway_pressure'
           END AS variable,
           TRY_CAST(replace(respchartvalue, '%', '') AS DOUBLE) AS value
    FROM respiratorycharting
    WHERE respchartvaluelabel IN ('Tidal Volume (set)','Tidal Volume Observed (VT)','PEEP','FiO2','Plateau Pressure','Total RR','Mean Airway Pressure')
),
vp AS (
    SELECT patientunitstayid, observationoffset / 60.0 AS time_h, 'heart_rate' AS variable, heartrate::DOUBLE AS value FROM vitalperiodic
    UNION ALL SELECT patientunitstayid, observationoffset / 60.0, 'spo2', sao2::DOUBLE / 100.0 FROM vitalperiodic
    UNION ALL SELECT patientunitstayid, observationoffset / 60.0, 'mbp', systemicmean::DOUBLE FROM vitalperiodic
),
lb AS (
    SELECT patientunitstayid, labresultoffset / 60.0 AS time_h,
           CASE labname WHEN 'paO2' THEN 'po2' WHEN 'paCO2' THEN 'pco2' WHEN 'pH' THEN 'ph' WHEN 'lactate' THEN 'lactate' END AS variable,
           labresult::DOUBLE AS value
    FROM lab WHERE labname IN ('paO2','paCO2','pH','lactate')
)
SELECT patientunitstayid AS stay_id, time_h, variable,
       CASE WHEN variable = 'fio2' AND value > 1 THEN value / 100.0 ELSE value END AS value
FROM (SELECT * FROM rc UNION ALL SELECT * FROM vp UNION ALL SELECT * FROM lb)
WHERE variable IS NOT NULL AND value IS NOT NULL AND time_h BETWEEN 0 AND 168
"""

EICU_OUTCOME_SQL = """
SELECT patientunitstayid AS stay_id, hospitalid, gender, TRY_CAST(age AS INTEGER) AS age,
       admissionheight AS height_cm,
       CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS died
FROM patient
"""


def run_sql(con, sql: str) -> pd.DataFrame:
    """Execute SQL on an open DuckDB connection (`duckdb.connect('data/derived/mimic_derived.duckdb')`)."""
    return con.execute(sql).fetch_df()


# ---------------------------------------------------------------------------------------- PBW
def predicted_body_weight(height_cm: np.ndarray, sex: Sequence[str]) -> np.ndarray:
    """ARDSNet predicted body weight (kg): male 50 + 0.91*(h-152.4); female 45.5 + 0.91*(h-152.4)."""
    h = np.asarray(height_cm, float)
    male = np.asarray([str(s).upper().startswith("M") for s in sex])
    pbw = np.where(male, 50.0, 45.5) + 0.91 * (h - 152.4)
    return np.clip(pbw, 25.0, 120.0)


# ---------------------------------------------------------------------------------------- binning
STATE_FEATURES: Tuple[str, ...] = (
    "fio2", "peep", "vt_per_kg", "plateau_pressure", "resp_rate", "minute_volume", "spo2", "po2", "pco2", "ph",
    "heart_rate", "mbp", "temperature", "lactate", "age", "male", "hours_vent",
)
ACTION_VARIABLES: Tuple[str, str, str] = ("vt_per_kg", "peep", "fio2")
DEFAULT_ACTION_EDGES: Dict[str, Tuple[float, ...]] = {
    "vt_per_kg": (6.0, 8.0),   # < 6 | 6–8 | > 8 mL/kg PBW
    "peep": (8.0, 12.0),       # < 8 | 8–12 | > 12 cmH2O
    "fio2": (0.4, 0.6),        # < 0.4 | 0.4–0.6 | > 0.6
}


def bin_events(events: pd.DataFrame, bin_hours: float = 4.0, max_hours: float = 168.0,
               agg: Mapping[str, str] | str = "last") -> pd.DataFrame:
    """Long (stay_id, time_h, variable, value) -> wide (stay_id, t) x variables using per-bin aggregation.

    `agg` is 'last' (last charted value in the bin, appropriate for settings), 'mean', or a dict per variable.
    """
    e = events.dropna(subset=["value"]).copy()
    e = e[(e["time_h"] >= 0) & (e["time_h"] < max_hours)]
    e["t"] = (e["time_h"] // bin_hours).astype(int)
    e = e.sort_values(["stay_id", "variable", "time_h"])
    if isinstance(agg, str):
        wide = e.groupby(["stay_id", "t", "variable"])["value"].agg(agg).unstack("variable")
    else:
        parts = []
        for var, a in agg.items():
            sub = e[e["variable"] == var]
            parts.append(sub.groupby(["stay_id", "t"])["value"].agg(a).rename(var))
        wide = pd.concat(parts, axis=1)
    return wide.reset_index()


def forward_fill(wide: pd.DataFrame, max_gap_bins: int = 2, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Within-stay forward fill limited to `max_gap_bins`, after re-indexing to a complete t grid per stay."""
    cols = [c for c in (cols or [c for c in wide.columns if c not in ("stay_id", "t")]) if c in wide]
    out = []
    for sid, g in wide.groupby("stay_id"):
        g = g.set_index("t").reindex(range(int(g["t"].min()), int(g["t"].max()) + 1))
        g["stay_id"] = sid
        g[cols] = g[cols].ffill(limit=max_gap_bins)
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def discretize_actions(vt_per_kg: np.ndarray, peep: np.ndarray, fio2: np.ndarray,
                       edges: Mapping[str, Sequence[float]] = DEFAULT_ACTION_EDGES) -> np.ndarray:
    """Map continuous settings to a single action index a = vt_bin * (nP*nF) + peep_bin * nF + fio2_bin."""
    vb = np.digitize(np.asarray(vt_per_kg, float), edges["vt_per_kg"])
    pb = np.digitize(np.asarray(peep, float), edges["peep"])
    fb = np.digitize(np.asarray(fio2, float), edges["fio2"])
    nP, nF = len(edges["peep"]) + 1, len(edges["fio2"]) + 1
    return (vb * nP * nF + pb * nF + fb).astype(int)


def action_grid(edges: Mapping[str, Sequence[float]] = DEFAULT_ACTION_EDGES) -> pd.DataFrame:
    """Table of action index -> (vt_bin, peep_bin, fio2_bin) with representative (mid-point-ish) values."""
    def reps(ed: Sequence[float], lo: float, hi: float) -> List[float]:
        pts = [lo] + list(ed) + [hi]
        return [(pts[i] + pts[i + 1]) / 2 for i in range(len(pts) - 1)]
    vt_r, pe_r, fi_r = reps(edges["vt_per_kg"], 4.0, 10.0), reps(edges["peep"], 5.0, 16.0), reps(edges["fio2"], 0.21, 1.0)
    rows = []
    for vb, vt in enumerate(vt_r):
        for pb, pe in enumerate(pe_r):
            for fb, fi in enumerate(fi_r):
                rows.append({"action": vb * len(pe_r) * len(fi_r) + pb * len(fi_r) + fb, "vt_bin": vb, "peep_bin": pb,
                             "fio2_bin": fb, "vt_per_kg": vt, "peep": pe, "fio2": fi})
    return pd.DataFrame(rows).set_index("action")


def n_actions(edges: Mapping[str, Sequence[float]] = DEFAULT_ACTION_EDGES) -> int:
    return int(np.prod([len(v) + 1 for v in edges.values()]))


# ---------------------------------------------------------------------------------------- trajectories
@dataclass
class Trajectories:
    """Flat transition arrays; `traj_id`/`t` index trajectories. `state` is the discrete state (optional)."""

    obs: np.ndarray            # (N, d) continuous features (may contain NaN before imputation)
    action: np.ndarray         # (N,) int
    reward: np.ndarray         # (N,) float
    next_obs: np.ndarray       # (N, d)
    done: np.ndarray           # (N,) bool
    traj_id: np.ndarray        # (N,)
    t: np.ndarray              # (N,) int
    state: Optional[np.ndarray] = None       # (N,) int discrete state
    next_state: Optional[np.ndarray] = None  # (N,) int
    feature_names: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        return len(self.action)

    def episodes(self) -> List[np.ndarray]:
        """Row indices of each trajectory in time order."""
        order = np.lexsort((self.t, self.traj_id))
        ids = self.traj_id[order]
        cuts = np.flatnonzero(np.diff(ids)) + 1
        return [order[s] for s in np.split(np.arange(len(order)), cuts)]

    def returns(self, gamma: float = 1.0) -> np.ndarray:
        return np.array([np.sum(self.reward[ep] * gamma ** np.arange(len(ep))) for ep in self.episodes()])

    def subset(self, idx: np.ndarray) -> "Trajectories":
        return Trajectories(self.obs[idx], self.action[idx], self.reward[idx], self.next_obs[idx], self.done[idx],
                            self.traj_id[idx], self.t[idx], None if self.state is None else self.state[idx],
                            None if self.next_state is None else self.next_state[idx], self.feature_names)


def build_trajectories(wide: pd.DataFrame, outcomes: pd.DataFrame, features: Sequence[str] = STATE_FEATURES,
                       edges: Mapping[str, Sequence[float]] = DEFAULT_ACTION_EDGES, bin_hours: float = 4.0,
                       terminal_reward: float = 1.0, shaping: Optional[Mapping[str, Tuple[float, float, float]]] = None,
                       min_bins: int = 6) -> Trajectories:
    """Assemble transitions from a forward-filled wide table and a per-stay outcome table.

    `outcomes` columns: stay_id, died (0/1), and optionally height_cm, gender, age (for PBW / demographics).
    Reward: terminal +/- `terminal_reward` (survival / death) at the last bin, 0 elsewhere; optional shaping
    {feature: (low, high, penalty)} subtracts `penalty` per bin when the feature is outside [low, high].
    """
    w = wide.copy()
    o = outcomes.set_index("stay_id")
    w = w[w["stay_id"].isin(o.index)]
    if "height_cm" in o and "gender" in o:
        pbw = pd.Series(predicted_body_weight(o["height_cm"].fillna(o["height_cm"].median()).to_numpy(), o["gender"].fillna("M")),
                        index=o.index)
        vt = w["tidal_volume_set"] if "tidal_volume_set" in w else w.get("tidal_volume_observed")
        w["vt_per_kg"] = vt.to_numpy() / pbw.reindex(w["stay_id"]).to_numpy() if vt is not None else np.nan
    if "age" in o:
        w["age"] = o["age"].reindex(w["stay_id"]).to_numpy()
    if "gender" in o:
        w["male"] = o["gender"].reindex(w["stay_id"]).astype(str).str.upper().str.startswith("M").astype(float).to_numpy()
    w["hours_vent"] = w["t"] * bin_hours
    for f in features:
        if f not in w:
            w[f] = np.nan
    obs_l, act_l, rew_l, nobs_l, done_l, tid_l, t_l = [], [], [], [], [], [], []
    for sid, g in w.sort_values(["stay_id", "t"]).groupby("stay_id"):
        if len(g) < min_bins:
            continue
        X = g[list(features)].to_numpy(float)
        a = discretize_actions(g["vt_per_kg"].to_numpy(float), g["peep"].to_numpy(float), g["fio2"].to_numpy(float), edges)
        r = np.zeros(len(g))
        if shaping:
            for f, (lo, hi, pen) in shaping.items():
                v = g[f].to_numpy(float)
                r -= pen * ((v < lo) | (v > hi)).astype(float)
        r[-1] += -terminal_reward if int(o.loc[sid, "died"]) == 1 else terminal_reward
        nX = np.vstack([X[1:], X[-1:]])
        d = np.zeros(len(g), bool)
        d[-1] = True
        obs_l.append(X); act_l.append(a); rew_l.append(r); nobs_l.append(nX); done_l.append(d)
        tid_l.append(np.full(len(g), sid)); t_l.append(np.arange(len(g)))
    if not obs_l:
        raise ValueError("no stays with >= min_bins bins")
    return Trajectories(np.vstack(obs_l), np.concatenate(act_l), np.concatenate(rew_l), np.vstack(nobs_l),
                        np.concatenate(done_l), np.concatenate(tid_l), np.concatenate(t_l), feature_names=tuple(features))


def impute_and_scale(traj: Trajectories, medians: Optional[np.ndarray] = None,
                     scale: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[Trajectories, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Median-impute NaNs and z-score using training statistics (fit on `traj` if not given)."""
    import warnings

    X = traj.obs.copy()
    if medians is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (feature absent at this site) -> 0
            med = np.nanmedian(X, axis=0)
    else:
        med = medians
    med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(X), med, X)
    nX = np.where(np.isnan(traj.next_obs), med, traj.next_obs)
    mu, sd = (X.mean(0), X.std(0) + 1e-8) if scale is None else scale
    out = Trajectories((X - mu) / sd, traj.action, traj.reward, (nX - mu) / sd, traj.done, traj.traj_id, traj.t,
                       traj.state, traj.next_state, traj.feature_names)
    return out, med, (mu, sd)


def discretize_states(traj: Trajectories, k: int = 500, seed: int = 0, model=None) -> Tuple[Trajectories, object]:
    """k-means state clustering (fit on obs, applied to next_obs) for tabular estimators."""
    from sklearn.cluster import KMeans
    if model is None:
        model = KMeans(n_clusters=min(k, max(2, traj.n // 5)), n_init=3, random_state=seed).fit(traj.obs)
    s = model.predict(traj.obs)
    ns = model.predict(traj.next_obs)
    return Trajectories(traj.obs, traj.action, traj.reward, traj.next_obs, traj.done, traj.traj_id, traj.t, s, ns,
                        traj.feature_names), model
