"""Visit-level cohort construction for MIMIC-IV-ED with language from MIMIC-IV admissions."""
from __future__ import annotations

import numpy as np
import pandas as pd

LANGUAGE_GROUPS: dict[str, tuple[str, ...]] = {
    # group -> substrings (lower-case) matched against admissions.language; verify against the
    # actual value set with `scripts/download_data.py --language-audit` (values changed in v3.0)
    "english": ("english",),
    "spanish": ("spanish",),
    "portuguese": ("portuguese", "cape verdean", "kabuverdianu"),
    "chinese": ("chinese", "mandarin", "cantonese"),
    "russian": ("russian",),
    "haitian_creole": ("haitian", "creole"),
    "vietnamese": ("vietnamese",),
    "arabic": ("arabic",),
}
UNKNOWN_LANGUAGE_VALUES = {"?", "", "unknown", "nan", "none"}

RACE_GROUPS: dict[str, tuple[str, ...]] = {
    "hispanic_latino": ("hispanic", "latino", "south american", "portuguese"),
    "black": ("black",),
    "asian": ("asian",),
    "white": ("white",),
}
UNKNOWN_RACE = ("unknown", "unable to obtain", "patient declined", "?")

LWBS_DISPOSITIONS = {"LEFT WITHOUT BEING SEEN", "ELOPED", "LEFT AGAINST MEDICAL ADVICE"}


def harmonize_language(value: object) -> str:
    """Map an admissions.language value to a language group ('unknown' / 'other' otherwise)."""
    v = str(value).strip().lower()
    if v in UNKNOWN_LANGUAGE_VALUES:
        return "unknown"
    for group, subs in LANGUAGE_GROUPS.items():
        if any(s in v for s in subs):
            return group
    return "other"


def harmonize_race(value: object) -> str:
    """Map a MIMIC race string to 6 groups. Hispanic/Latino takes precedence over colour terms."""
    v = str(value).strip().lower()
    if v == "nan" or any(u in v for u in UNKNOWN_RACE):
        return "unknown"
    for group, subs in RACE_GROUPS.items():
        if any(s in v for s in subs):
            return group
    return "other"


def language_by_subject(admissions: pd.DataFrame) -> pd.DataFrame:
    """Modal harmonised language per subject across admissions, with a conflict flag.

    'unknown' never wins over a known language.  Returns subject_id, language_group, language_conflict.
    """
    a = admissions[["subject_id", "language"]].copy()
    a["lg"] = a["language"].map(harmonize_language)
    known = a[a["lg"] != "unknown"]
    mode = known.groupby("subject_id")["lg"].agg(lambda s: s.value_counts().idxmax())
    n_distinct = known.groupby("subject_id")["lg"].nunique()
    out = pd.DataFrame({"language_group": mode, "language_conflict": n_distinct > 1})
    all_subj = a["subject_id"].unique()
    out = out.reindex(all_subj)
    out["language_group"] = out["language_group"].fillna("unknown")
    out["language_conflict"] = out["language_conflict"].fillna(False).astype(bool)
    return out.rename_axis("subject_id").reset_index()


def ed_return_within(edstays: pd.DataFrame, hours: float = 72.0) -> pd.Series:
    """1 if the same subject has another ED stay starting within ``hours`` of this stay's outtime."""
    e = edstays[["stay_id", "subject_id", "intime", "outtime"]].sort_values(["subject_id", "intime"]).copy()
    e["next_intime"] = e.groupby("subject_id")["intime"].shift(-1)
    gap_h = (pd.to_datetime(e["next_intime"]) - pd.to_datetime(e["outtime"])).dt.total_seconds() / 3600.0
    ret = ((gap_h >= 0) & (gap_h <= hours)).astype(int)
    return pd.Series(ret.to_numpy(), index=e["stay_id"].to_numpy(), name="ed_return").reindex(edstays["stay_id"]).fillna(0).astype(int)


def build_visit_table(
    edstays: pd.DataFrame,
    triage: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    pyxis: pd.DataFrame | None = None,
    critical_icu_hours: float = 12.0,
) -> pd.DataFrame:
    """One row per ED stay with exposures, covariates and outcomes.

    Outcomes: admitted, lwbs, ed_los_h, critical (ICU-in within ``critical_icu_hours`` of
    ED outtime or in-hospital death on the linked admission), ed_return (72 h),
    time_to_first_med_h (pyxis, if given).  Age at visit = anchor_age + (visit year - anchor_year).
    """
    e = edstays.copy()
    for c in ("intime", "outtime"):
        e[c] = pd.to_datetime(e[c])
    e = e.merge(triage.drop(columns=[c for c in ("subject_id",) if c in triage.columns]), on="stay_id", how="left")
    e = e.merge(patients[["subject_id", "anchor_age", "anchor_year", "anchor_year_group"]], on="subject_id", how="left")
    e["age"] = e["anchor_age"] + (e["intime"].dt.year - e["anchor_year"])
    lang = language_by_subject(admissions)
    e = e.merge(lang, on="subject_id", how="left")
    e["language_group"] = e["language_group"].fillna("unknown")
    e["nep"] = (~e["language_group"].isin(["english", "unknown"])).astype(int)
    e["race_group"] = e["race"].map(harmonize_race)
    adm = admissions[["hadm_id", "insurance", "marital_status", "hospital_expire_flag"]].drop_duplicates("hadm_id")
    e = e.merge(adm, on="hadm_id", how="left")
    e["admitted"] = e["disposition"].astype(str).str.upper().eq("ADMITTED").astype(int)
    e["lwbs"] = e["disposition"].astype(str).str.upper().isin(LWBS_DISPOSITIONS).astype(int)
    e["ed_los_h"] = (e["outtime"] - e["intime"]).dt.total_seconds() / 3600.0
    icu = icustays[["hadm_id", "intime"]].rename(columns={"intime": "icu_intime"})
    icu["icu_intime"] = pd.to_datetime(icu["icu_intime"])
    first_icu = icu.groupby("hadm_id")["icu_intime"].min().reset_index()
    e = e.merge(first_icu, on="hadm_id", how="left")
    icu_h = (e["icu_intime"] - e["outtime"]).dt.total_seconds() / 3600.0
    e["icu_12h"] = ((icu_h >= -1.0) & (icu_h <= critical_icu_hours)).fillna(False).astype(int)
    e["critical"] = ((e["icu_12h"] == 1) | (e["hospital_expire_flag"].fillna(0) == 1)).astype(int)
    e["ed_return"] = ed_return_within(edstays).to_numpy()
    e["hour"] = e["intime"].dt.hour
    e["weekend"] = (e["intime"].dt.dayofweek >= 5).astype(int)
    e["pain_missing"] = e["pain"].isna().astype(int) if "pain" in e.columns else np.nan
    vit = [c for c in ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp") if c in e.columns]
    e["n_missing_vitals"] = e[vit].isna().sum(axis=1) if vit else np.nan
    if pyxis is not None and len(pyxis):
        p = pyxis[["stay_id", "charttime"]].copy()
        p["charttime"] = pd.to_datetime(p["charttime"])
        first_med = p.groupby("stay_id")["charttime"].min().rename("first_med_time")
        e = e.merge(first_med, on="stay_id", how="left")
        e["time_to_first_med_h"] = (e["first_med_time"] - e["intime"]).dt.total_seconds() / 3600.0
    return e


def selection_audit(visits: pd.DataFrame, group_col: str = "language_group") -> pd.DataFrame:
    """Compare triage characteristics of visits with known vs unknown language (selection into the analysable cohort)."""
    cols = [c for c in ("age", "acuity", "admitted", "critical", "ed_los_h", "pain_missing", "n_missing_vitals") if c in visits.columns]
    known = visits[group_col] != "unknown"
    a = visits.loc[known, cols].agg(["mean", "median"]).T.add_prefix("known_")
    b = visits.loc[~known, cols].agg(["mean", "median"]).T.add_prefix("unknown_")
    out = a.join(b)
    out["n_known"], out["n_unknown"] = int(known.sum()), int((~known).sum())
    return out
