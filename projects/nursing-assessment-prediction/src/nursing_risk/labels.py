"""HAPI label definitions (ICD and nursing documentation) and the label multiverse."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ICD-10-CM L89.<site 2 digits><stage digit>: stage digit 0 unstageable, 1-4 stage, 6 deep tissue, 9 unspecified.
# ICD-9-CM 707.0x pressure ulcer by site; 707.2x stage (707.20 unspecified, 707.21-707.24 stage I-IV, 707.25 unstageable).
_STAGE_RE = re.compile(r"(?:stage\s*)?(\d)|unstageable|deep tissue|dti", re.IGNORECASE)


def icd_pi_stage(code: str, version: int) -> int | None:
    """Return the pressure-ulcer stage encoded by an ICD code (None if not a PI code).

    Stage 0 = unstageable / deep-tissue / unspecified (present but stage unknown).
    """
    c = str(code).replace(".", "").upper()
    if int(version) == 10 and c.startswith("L89"):
        if len(c) >= 5 and c[4].isdigit():
            d = int(c[4])
            return d if 1 <= d <= 4 else 0
        return 0
    if int(version) == 9 and c.startswith("707"):
        if c.startswith("7072") and len(c) >= 5 and c[4].isdigit():
            d = int(c[4])
            return d if 1 <= d <= 4 else 0
        if c.startswith("7070"):
            return 0
    return None


def icd_labels(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Per hadm_id: icd_any (any PI code) and icd_stage2plus (any code with stage >= 2)."""
    d = diagnoses[["hadm_id", "icd_code", "icd_version"]].copy()
    d["stage"] = [icd_pi_stage(c, v) for c, v in zip(d["icd_code"], d["icd_version"])]
    d = d.dropna(subset=["stage"])
    g = d.groupby("hadm_id")["stage"]
    out = pd.DataFrame({"icd_any": g.size() > 0, "icd_max_stage": g.max()})
    out["icd_stage2plus"] = out["icd_max_stage"] >= 2
    return out.reset_index()


def parse_stage_value(value: object) -> int | None:
    """Parse a charted skin-assessment value into a stage integer (0 = unstageable/DTI); None if not a PI."""
    s = str(value).strip().lower()
    if s in {"", "nan", "none", "intact", "no", "not applicable", "n/a"}:
        return None
    if "unstageable" in s or "deep tissue" in s or "dti" in s:
        return 0
    m = re.search(r"(\d)", s)
    if m:
        d = int(m.group(1))
        return d if 1 <= d <= 4 else 0
    if "stage" in s or "ulcer" in s or "injury" in s:
        return 0
    return None


def nursing_first_pi(chartevents: pd.DataFrame, resolved: pd.DataFrame, min_stage: int = 1) -> pd.DataFrame:
    """First charted pressure-injury time per stay from skin_pi_stage items (stage >= min_stage; stage 0 counts when min_stage == 0).

    ``chartevents``: stay_id, hadm_id, charttime, itemid, value.  Returns stay_id, hadm_id, first_pi_time, first_pi_stage.
    """
    ids = set(resolved.loc[resolved["concept"] == "skin_pi_stage", "itemid"])
    ce = chartevents[chartevents["itemid"].isin(ids)].copy()
    ce["stage"] = ce["value"].map(parse_stage_value)
    ce = ce.dropna(subset=["stage"])
    ce = ce[(ce["stage"] >= min_stage) | ((min_stage == 0) & (ce["stage"] == 0))]
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    first = ce.sort_values("charttime").groupby("stay_id").first().reset_index()
    return first[["stay_id", "hadm_id", "charttime", "stage"]].rename(columns={"charttime": "first_pi_time", "stage": "first_pi_stage"})


def hospital_acquired_flag(first_pi: pd.DataFrame, admissions: pd.DataFrame, min_hours: float = 24.0) -> pd.DataFrame:
    """Mark first PI documentation as hospital-acquired if it occurs > min_hours after hospital admission.

    Adds ``hours_from_admit``, ``hospital_acquired`` and ``probable_poa`` (documented within min_hours).
    """
    a = admissions[["hadm_id", "admittime"]].copy()
    a["admittime"] = pd.to_datetime(a["admittime"])
    f = first_pi.merge(a, on="hadm_id", how="left")
    f["hours_from_admit"] = (f["first_pi_time"] - f["admittime"]).dt.total_seconds() / 3600.0
    f["hospital_acquired"] = f["hours_from_admit"] > min_hours
    f["probable_poa"] = f["hours_from_admit"] <= min_hours
    return f


def label_multiverse(stays: pd.DataFrame, icd: pd.DataFrame, nursing_stage1: pd.DataFrame, nursing_stage2: pd.DataFrame, admissions: pd.DataFrame, thresholds_h: tuple[float, ...] = (24.0, 48.0)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build every label definition per ICU stay and summarise prevalence and agreement.

    Definitions: icd_any, icd_stage2plus, nurs1_ha{T}, nurs2_ha{T} for T in thresholds_h,
    icd_and_nurs1_ha24.  Returns (per-stay label table, summary with prevalence, Jaccard and
    Cohen's kappa against icd_any, and the probable-POA fraction among ICD-coded stays).
    """
    st = stays[["stay_id", "hadm_id"]].drop_duplicates("stay_id").copy()
    st = st.merge(icd[["hadm_id", "icd_any", "icd_stage2plus"]], on="hadm_id", how="left").fillna({"icd_any": False, "icd_stage2plus": False})
    defs = ["icd_any", "icd_stage2plus"]
    for name, nurs in (("nurs1", nursing_stage1), ("nurs2", nursing_stage2)):
        for T in thresholds_h:
            ha = hospital_acquired_flag(nurs, admissions, min_hours=T)
            col = f"{name}_ha{int(T)}"
            st[col] = st["stay_id"].isin(set(ha.loc[ha["hospital_acquired"], "stay_id"]))
            defs.append(col)
    st["icd_and_nurs1_ha24"] = st["icd_any"].astype(bool) & st["nurs1_ha24"]
    defs.append("icd_and_nurs1_ha24")
    poa = hospital_acquired_flag(nursing_stage1, admissions, min_hours=thresholds_h[0])
    poa_stays = set(poa.loc[poa["probable_poa"], "stay_id"])
    icd_stays = st.loc[st["icd_any"].astype(bool), "stay_id"]
    poa_frac = float(icd_stays.isin(poa_stays).mean()) if len(icd_stays) else np.nan
    rows = []
    ref = st["icd_any"].astype(bool).to_numpy()
    for d in defs:
        x = st[d].astype(bool).to_numpy()
        inter, union = np.sum(x & ref), np.sum(x | ref)
        po = np.mean(x == ref)
        pe = np.mean(x) * np.mean(ref) + (1 - np.mean(x)) * (1 - np.mean(ref))
        kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
        rows.append({"definition": d, "n_positive": int(x.sum()), "prevalence": float(x.mean()), "jaccard_vs_icd_any": float(inter / union) if union else np.nan, "kappa_vs_icd_any": float(kappa), "probable_poa_frac_of_icd": poa_frac})
    return st, pd.DataFrame(rows)


def inpatient_fall_icd10(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Weak in-hospital fall label: any W00-W19 external-cause code together with Y92.23x (hospital as place) on the same admission."""
    d = diagnoses[diagnoses["icd_version"].astype(int) == 10][["hadm_id", "icd_code"]].copy()
    c = d["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    d["fall"] = c.str.match(r"^W(0\d|1\d)")
    d["hospital_place"] = c.str.startswith("Y9223")
    g = d.groupby("hadm_id")[["fall", "hospital_place"]].any()
    g["inpatient_fall"] = g["fall"] & g["hospital_place"]
    return g.reset_index()
