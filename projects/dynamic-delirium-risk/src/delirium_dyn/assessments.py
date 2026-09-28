"""CAM-ICU / RASS parsing, 12-hour window states and a synthetic stay simulator.

Window state rule (per ICU stay, fixed windows from ICU admission):

* ``delirium``   any CAM-ICU positive in the window;
* ``normal``     otherwise, any CAM-ICU negative;
* ``coma``       otherwise, any "unable to assess" (UTA) or *all* RASS values <= -4;
* ``unscreened`` otherwise (patient arousable at least once but no CAM-ICU charted);
* ``discharged`` / ``dead`` after the stay ends (terminal).

The four-feature CAM-ICU rule (positive iff feature 1 and feature 2 and (feature 3 or feature 4)) is applied
when only features are charted; a charted summary result takes precedence.  Charted strings are mapped with
regexes so that MIMIC-IV and eICU spellings can share the code; the mapping used should be versioned under
``data/value_maps/``.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

POS_RE = re.compile(r"\b(yes|positive|present|pos)\b", re.I)
NEG_RE = re.compile(r"\b(no|negative|absent|neg)\b", re.I)
UTA_RE = re.compile(r"(uta|unable|not\s*assess|n/?a)", re.I)

FEATURE_PATTERNS: Dict[str, str] = {
    "f1": r"ms\s*change|mental\s*status|acute\s*onset|fluctuat",
    "f2": r"inattention",
    "f3": r"rass\s*loc|altered\s*loc|level\s*of\s*consciousness",
    "f4": r"disorgani[sz]ed",
}
SUMMARY_PATTERN = r"^delirium\s*assessment$|cam-?icu\s*(result|overall|score)?$"
RASS_PATTERN = r"richmond|rass"
CAM_ANY_PATTERN = r"cam-?icu|delirium\s*assessment"


def resolve_itemids(d_items: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Rows of ``d_items`` whose ``label`` matches ``pattern`` (case-insensitive regex)."""
    m = d_items["label"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    return d_items.loc[m, ["itemid", "label"]].reset_index(drop=True)


def parse_cam_value(value: object) -> Optional[str]:
    """Map a charted CAM-ICU string to ``'positive' | 'negative' | 'uta'`` or ``None``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if UTA_RE.search(s):
        return "uta"
    if POS_RE.search(s):
        return "positive"
    if NEG_RE.search(s):
        return "negative"
    return None


def parse_rass(value: object) -> Optional[int]:
    """Extract an integer RASS in [-5, 4] from a charted value (numeric or e.g. ``'-3 Moderate sedation'``)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m = re.search(r"([+-]?\d)", str(value))
    if not m:
        return None
    v = int(m.group(1))
    return v if -5 <= v <= 4 else None


def assessment_timeline(chartevents: pd.DataFrame, d_items: pd.DataFrame) -> pd.DataFrame:
    """Long table of parsed assessments: ``stay_id, charttime, kind, value``.

    ``kind`` is ``'cam'`` (value in positive/negative/uta) or ``'rass'`` (integer). Feature rows charted at the
    same ``charttime`` are combined with the four-feature rule when no summary result exists at that time.
    """
    cam_items = resolve_itemids(d_items, CAM_ANY_PATTERN)
    rass_items = resolve_itemids(d_items, RASS_PATTERN)
    ce = chartevents[chartevents["itemid"].isin(pd.concat([cam_items["itemid"], rass_items["itemid"]]))].copy()
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    label_of = dict(zip(d_items["itemid"], d_items["label"].astype(str)))
    ce["label"] = ce["itemid"].map(label_of)
    rows = []
    # RASS
    r = ce[ce["itemid"].isin(rass_items["itemid"])]
    for stay, t, v in zip(r["stay_id"], r["charttime"], r["value"] if "value" in r else r["valuenum"]):
        p = parse_rass(v)
        if p is not None:
            rows.append((stay, t, "rass", p))
    # CAM summary and features
    c = ce[ce["itemid"].isin(cam_items["itemid"])].copy()
    c["is_summary"] = c["label"].str.contains(SUMMARY_PATTERN, case=False, regex=True)
    for (stay, t), grp in c.groupby(["stay_id", "charttime"], sort=False):
        summ = grp[grp["is_summary"]]
        val = None
        if len(summ):
            parsed = [parse_cam_value(v) for v in summ["value"]]
            parsed = [p for p in parsed if p]
            if "positive" in parsed:
                val = "positive"
            elif "negative" in parsed:
                val = "negative"
            elif "uta" in parsed:
                val = "uta"
        if val is None:
            feats: Dict[str, Optional[str]] = {}
            for lab, v in zip(grp["label"], grp["value"]):
                for f, pat in FEATURE_PATTERNS.items():
                    if re.search(pat, lab, re.I):
                        feats[f] = parse_cam_value(v)
            if feats:
                if "uta" in feats.values():
                    val = "uta"
                elif all(k in feats for k in ("f1", "f2")) and any(k in feats for k in ("f3", "f4")):
                    pos = feats["f1"] == "positive" and feats["f2"] == "positive" and (feats.get("f3") == "positive" or feats.get("f4") == "positive")
                    val = "positive" if pos else "negative"
        if val is not None:
            rows.append((stay, t, "cam", val))
    out = pd.DataFrame(rows, columns=["stay_id", "charttime", "kind", "value"])
    return out.sort_values(["stay_id", "charttime"]).reset_index(drop=True)


def window_states(
    timeline: pd.DataFrame,
    icustays: pd.DataFrame,
    window_hours: int = 12,
    deathtime: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Per-stay fixed windows with the state rule from the module docstring.

    ``icustays`` needs ``stay_id, intime, outtime``; ``deathtime`` is an optional Series indexed by ``stay_id``.
    One terminal row (``discharged`` or ``dead``) is appended after the last window of each stay.
    """
    tl = timeline.copy()
    tl["charttime"] = pd.to_datetime(tl["charttime"])
    rows = []
    for _, st in icustays.iterrows():
        sid = st["stay_id"]
        t_in, t_out = pd.to_datetime(st["intime"]), pd.to_datetime(st["outtime"])
        n_win = int(np.ceil((t_out - t_in).total_seconds() / 3600.0 / window_hours))
        n_win = max(n_win, 1)
        s = tl[tl["stay_id"] == sid]
        for w in range(n_win):
            w0 = t_in + pd.Timedelta(hours=w * window_hours)
            w1 = w0 + pd.Timedelta(hours=window_hours)
            sw = s[(s["charttime"] >= w0) & (s["charttime"] < w1)]
            cam = sw[sw["kind"] == "cam"]["value"]
            rass = pd.to_numeric(sw[sw["kind"] == "rass"]["value"], errors="coerce").dropna()
            if (cam == "positive").any():
                state = "delirium"
            elif (cam == "negative").any():
                state = "normal"
            elif (cam == "uta").any() or (len(rass) > 0 and (rass <= -4).all()):
                state = "coma"
            else:
                state = "unscreened"
            rows.append(
                {
                    "stay_id": sid, "window_idx": w, "w_start": w0, "w_end": w1, "state": state,
                    "n_cam": int(len(cam)), "n_rass": int(len(rass)),
                    "rass_min": float(rass.min()) if len(rass) else np.nan,
                    "rass_max": float(rass.max()) if len(rass) else np.nan,
                    "rass_mean": float(rass.mean()) if len(rass) else np.nan,
                }
            )
        died = deathtime is not None and sid in deathtime.index and pd.notna(deathtime.loc[sid]) and pd.to_datetime(deathtime.loc[sid]) <= t_out + pd.Timedelta(hours=window_hours)
        rows.append({"stay_id": sid, "window_idx": n_win, "w_start": t_in + pd.Timedelta(hours=n_win * window_hours), "w_end": pd.NaT,
                     "state": "dead" if died else "discharged", "n_cam": 0, "n_rass": 0, "rass_min": np.nan, "rass_max": np.nan, "rass_mean": np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ simulator
def simulate_stays(
    n_stays: int = 50,
    window_hours: int = 12,
    max_windows: int = 20,
    p_screen_base: float = 0.7,
    benzo_effect: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, pd.DataFrame]:
    """Simulate stays with a sedation-dependent state Markov chain and an informative screening process.

    Hidden state per window in {normal, delirium, coma}; a benzodiazepine "policy" variable (0/1 per window)
    multiplies the normal->delirium and coma->delirium hazards by ``exp(benzo_effect)``; deep sedation
    (propofol policy) drives normal->coma.  Screening probability is lower after a coma window and higher after
    a positive screen (informative missingness).  Returns ``windows`` (true and observed states),
    ``sedation`` (per-window benzo/propofol/dex doses) and ``icustays``.
    """
    rng = np.random.default_rng() if rng is None else rng
    win_rows, sed_rows, stay_rows = [], [], []
    t0 = pd.Timestamp("2150-01-01")
    for sid in range(1, n_stays + 1):
        n_win = int(rng.integers(3, max_windows + 1))
        state = "coma" if rng.random() < 0.3 else "normal"
        prev_obs = "normal"
        intime = t0 + pd.Timedelta(days=int(rng.integers(0, 365)))
        for w in range(n_win):
            benzo = int(rng.random() < 0.3)
            propofol = int(rng.random() < 0.4)
            dex = int(rng.random() < 0.2)
            # transitions
            if state == "normal":
                p_del = 0.10 * np.exp(benzo_effect * benzo)
                p_coma = 0.05 + 0.15 * propofol
                u = rng.random()
                state = "delirium" if u < p_del else ("coma" if u < p_del + p_coma else "normal")
            elif state == "delirium":
                p_norm = 0.25 + 0.1 * dex
                p_coma = 0.05 + 0.1 * propofol
                u = rng.random()
                state = "normal" if u < p_norm else ("coma" if u < p_norm + p_coma else "delirium")
            else:  # coma
                p_del = 0.15 * np.exp(benzo_effect * benzo)
                p_norm = 0.15
                u = rng.random()
                state = "delirium" if u < p_del else ("normal" if u < p_del + p_norm else "coma")
            # screening (informative)
            p_screen = p_screen_base * (0.6 if prev_obs == "coma" else 1.0) * (1.2 if prev_obs == "delirium" else 1.0)
            screened = rng.random() < min(p_screen, 0.98)
            if state == "coma":
                observed = "coma" if screened else "unscreened"
            else:
                observed = state if screened else "unscreened"
            prev_obs = observed
            win_rows.append({"stay_id": sid, "window_idx": w, "w_start": intime + pd.Timedelta(hours=w * window_hours),
                             "w_end": intime + pd.Timedelta(hours=(w + 1) * window_hours), "state": observed, "true_state": state,
                             "n_cam": int(screened and state != "coma"), "n_rass": 1,
                             "rass_min": -4.0 if state == "coma" else float(rng.integers(-3, 2)),
                             "rass_max": np.nan, "rass_mean": np.nan})
            sed_rows.append({"stay_id": sid, "window_idx": w, "benzo_mg": 2.0 * benzo * rng.uniform(0.5, 2.0),
                             "propofol_mg": 800.0 * propofol * rng.uniform(0.5, 2.0), "dex_mcg": 300.0 * dex * rng.uniform(0.5, 2.0)})
        died = rng.random() < 0.1
        win_rows.append({"stay_id": sid, "window_idx": n_win, "w_start": intime + pd.Timedelta(hours=n_win * window_hours), "w_end": pd.NaT,
                         "state": "dead" if died else "discharged", "true_state": "dead" if died else "discharged",
                         "n_cam": 0, "n_rass": 0, "rass_min": np.nan, "rass_max": np.nan, "rass_mean": np.nan})
        stay_rows.append({"stay_id": sid, "intime": intime, "outtime": intime + pd.Timedelta(hours=n_win * window_hours)})
    windows = pd.DataFrame(win_rows)
    windows["rass_max"] = windows["rass_max"].fillna(windows["rass_min"])
    windows["rass_mean"] = windows["rass_mean"].fillna(windows["rass_min"])
    return {"windows": windows, "sedation": pd.DataFrame(sed_rows), "icustays": pd.DataFrame(stay_rows)}
