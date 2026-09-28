"""OASIS-3 table construction: ID parsing, session matching, synthetic sample.

OASIS-3 identifiers encode the participant and the day offset from entry:
``OAS30001_MR_d0129`` (MR session), ``OAS30001_AV1451_d1234`` (tau PET),
``OAS30001_PIB_d0100`` (amyloid PET), and clinical rows are keyed by
``OAS30001_ClinicalData_d0120`` or by ``(Subject, days_to_visit)``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_ID_RE = re.compile(r"^(?P<subject>OAS3\d{4})_(?P<modality>[A-Za-z0-9]+)_d(?P<day>\d{4,5})$")


def parse_oasis_id(session_id: str) -> tuple[str, str, int]:
    """Split an OASIS-3 session ID into ``(subject, modality, day)``.

    >>> parse_oasis_id("OAS30001_AV1451_d1234")
    ('OAS30001', 'AV1451', 1234)
    """
    m = _ID_RE.match(str(session_id).strip())
    if not m:
        raise ValueError(f"not an OASIS-3 session id: {session_id!r}")
    return m.group("subject"), m.group("modality"), int(m.group("day"))


def add_id_columns(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Add ``subject``, ``modality`` and ``day`` columns parsed from ``id_col``."""
    parsed = df[id_col].map(parse_oasis_id)
    out = df.copy()
    out["subject"] = [p[0] for p in parsed]
    out["modality"] = [p[1] for p in parsed]
    out["day"] = [p[2] for p in parsed]
    return out


def apoe_e4_count(genotype: object) -> float:
    """Count ε4 alleles from an APOE genotype coded as e.g. ``34``, ``'3/4'`` or ``'E3E4'``."""
    if genotype is None or (isinstance(genotype, float) and np.isnan(genotype)):
        return np.nan
    digits = re.findall(r"[234]", str(genotype))
    if len(digits) != 2:
        return np.nan
    return float(sum(d == "4" for d in digits))


def match_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str = "subject",
    left_day: str = "day",
    right_day: str = "day",
    max_gap: int,
    suffix: str,
) -> pd.DataFrame:
    """For each row of ``left`` find the row of ``right`` (same subject) nearest in days.

    Rows with no partner within ``max_gap`` days keep NaNs for the right-hand columns.
    The right-hand day column is kept as ``day_<suffix>`` and the gap as ``gap_<suffix>``.
    """
    right = right.rename(columns={right_day: f"day_{suffix}"})
    pieces = []
    for _, row in left.iterrows():
        cand = right[right[on] == row[on]]
        if cand.empty:
            pieces.append(pd.Series(dtype=object))
            continue
        gaps = (cand[f"day_{suffix}"] - row[left_day]).abs()
        j = gaps.idxmin()
        if gaps.loc[j] > max_gap:
            pieces.append(pd.Series(dtype=object))
            continue
        best = cand.loc[j].drop(labels=[on])
        best[f"gap_{suffix}"] = int(gaps.loc[j])
        pieces.append(best)
    matched = pd.DataFrame(pieces, index=left.index)
    return pd.concat([left, matched], axis=1)


def build_pet_mr_table(
    tau: pd.DataFrame,
    mr: pd.DataFrame,
    clinical: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    mr_gap: int = 365,
    clin_gap: int = 180,
    one_per_subject: bool = True,
) -> pd.DataFrame:
    """Join tau PET sessions to the nearest MR session and clinical visit.

    Parameters
    ----------
    tau : DataFrame with ``tau_id`` (``OAS3xxxx_AV1451_dNNNN``) plus ROI SUVR columns.
    mr : DataFrame with ``mr_id`` and any imaging feature columns.
    clinical : DataFrame with ``subject``, ``day``, ``cdr``, ``cdr_sb``, ``mmse``.
    demographics : DataFrame with ``subject``, ``age_at_entry``, ``sex``, ``education``, ``apoe``.
    """
    t = add_id_columns(tau, "tau_id")
    m = add_id_columns(mr, "mr_id").drop(columns=["modality"])
    t = match_nearest(t, m, max_gap=mr_gap, suffix="mr")
    t = match_nearest(t, clinical, max_gap=clin_gap, suffix="clin")
    t = t.merge(demographics, on="subject", how="left")
    t["age"] = t["age_at_entry"] + t["day"] / 365.25
    t["apoe_e4"] = t["apoe"].map(apoe_e4_count)
    t["has_mr"] = t["mr_id"].notna()
    t["has_clin"] = t["gap_clin"].notna()
    if one_per_subject:
        t = t.sort_values(["subject", "day"]).groupby("subject", as_index=False).first()
    return t.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic OASIS-3-like tables (for --sample mode and tests)
# ---------------------------------------------------------------------------
TAU_ROIS = [
    "ctx-lh-entorhinal", "ctx-rh-entorhinal", "Left-Amygdala", "Right-Amygdala",
    "ctx-lh-parahippocampal", "ctx-rh-parahippocampal", "ctx-lh-fusiform", "ctx-rh-fusiform",
    "ctx-lh-inferiortemporal", "ctx-rh-inferiortemporal", "ctx-lh-middletemporal", "ctx-rh-middletemporal",
    "ctx-lh-precuneus", "ctx-rh-precuneus", "ctx-lh-inferiorparietal", "ctx-rh-inferiorparietal",
    "ctx-lh-lateraloccipital", "ctx-rh-lateraloccipital", "ctx-lh-superiorfrontal", "ctx-rh-superiorfrontal",
]


def simulate_cohort(n_subjects: int = 200, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Simulate OASIS-3-like tau, MR, clinical and demographic tables.

    The generative model encodes the confound structure the project audits:
    tau load rises with age and APOE ε4, atrophy is downstream of tau *and*
    of age/vascular load, and cognition depends on both.
    """
    rng = np.random.default_rng(seed)
    subjects = [f"OAS3{i:04d}" for i in range(1, n_subjects + 1)]
    age0 = rng.uniform(55, 88, n_subjects)
    sex = rng.choice(["F", "M"], n_subjects)
    apoe = rng.choice(["33", "34", "23", "44", "22"], n_subjects, p=[0.55, 0.28, 0.1, 0.05, 0.02])
    e4 = np.array([apoe_e4_count(g) for g in apoe])
    educ = rng.integers(8, 21, n_subjects)

    latent_tau = -1.2 + 0.06 * (age0 - 70) + 0.9 * e4 + rng.normal(0, 1.0, n_subjects)
    tau_mtl = 1.05 + 0.12 * np.exp(latent_tau) / (1 + np.exp(latent_tau)) * 2 + rng.normal(0, 0.05, n_subjects)
    tau_neo = 1.00 + 0.10 * np.clip(latent_tau - 0.5, 0, None) + rng.normal(0, 0.05, n_subjects)
    vascular = rng.gamma(2.0, 1.0, n_subjects) * (1 + 0.02 * (age0 - 70))
    atrophy = 1.0 * (tau_mtl - 1.1) * 10 + 0.02 * (age0 - 70) + 0.15 * vascular + rng.normal(0, 0.4, n_subjects)
    cdr_sb = np.clip(0.4 * atrophy + 0.6 * (tau_neo - 1.0) * 10 + rng.normal(0, 0.5, n_subjects), 0, 18)
    mmse = np.clip(30 - 0.9 * cdr_sb - rng.gamma(1.0, 0.7, n_subjects), 0, 30).round()

    day_pet = rng.integers(0, 3000, n_subjects)
    tau_rows = {"tau_id": [f"{s}_AV1451_d{d:04d}" for s, d in zip(subjects, day_pet)]}
    for roi in TAU_ROIS:
        base = tau_mtl if any(k in roi for k in ("entorhinal", "Amygdala", "parahippocampal", "fusiform", "temporal")) else tau_neo
        tau_rows[f"suvr_{roi}"] = base + rng.normal(0, 0.03, n_subjects)
    tau_df = pd.DataFrame(tau_rows)

    day_mr = day_pet + rng.integers(-200, 200, n_subjects)
    day_mr = np.clip(day_mr, 0, None)
    mr = pd.DataFrame({"mr_id": [f"{s}_MR_d{d:04d}" for s, d in zip(subjects, day_mr)]})
    etiv = rng.normal(1500e3, 150e3, n_subjects) * np.where(sex == "M", 1.08, 0.95)
    mr["eTIV"] = etiv
    mr["Left-Hippocampus"] = (4200 - 300 * atrophy + rng.normal(0, 150, n_subjects)) * etiv / 1500e3
    mr["Right-Hippocampus"] = (4300 - 300 * atrophy + rng.normal(0, 150, n_subjects)) * etiv / 1500e3
    mr["WM-hypointensities"] = np.exp(7.0 + 0.6 * vascular + rng.normal(0, 0.3, n_subjects))
    for hemi in ("lh", "rh"):
        for region in ("entorhinal", "inferiortemporal", "middletemporal", "fusiform", "precuneus",
                       "superiorfrontal", "inferiorparietal", "lateraloccipital", "parahippocampal", "posteriorcingulate"):
            temporal = region in ("entorhinal", "inferiortemporal", "middletemporal", "fusiform", "parahippocampal")
            base = 3.2 if temporal else 2.5
            slope = 0.15 if temporal else 0.05
            mr[f"{hemi}_{region}_thickness"] = base - slope * atrophy - 0.004 * (age0 - 70) + rng.normal(0, 0.06, n_subjects)
    mr["scanner"] = rng.choice(["TrioTim", "Biograph_mMR"], n_subjects, p=[0.7, 0.3])
    mr.loc[mr["scanner"] == "Biograph_mMR", [c for c in mr.columns if c.endswith("_thickness")]] += 0.05

    clin = pd.DataFrame({
        "subject": subjects,
        "day": np.clip(day_pet + rng.integers(-120, 120, n_subjects), 0, None),
        "cdr": np.where(cdr_sb < 0.5, 0.0, np.where(cdr_sb < 4.5, 0.5, 1.0)),
        "cdr_sb": cdr_sb.round(1),
        "mmse": mmse,
    })
    demo = pd.DataFrame({"subject": subjects, "age_at_entry": age0.round(2), "sex": sex, "education": educ, "apoe": apoe})
    return {"tau": tau_df, "mr": mr, "clinical": clin, "demographics": demo}


def write_synthetic_oasis3_tables(out_dir: Path | str, n_subjects: int = 120, seed: int = 0) -> dict[str, Path]:
    """Write the simulated tables as CSVs mirroring the OASIS-3 file roles."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tabs = simulate_cohort(n_subjects, seed)
    paths = {}
    for name, df in tabs.items():
        p = out / f"synthetic_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths


def subjects_with_tau(tau: pd.DataFrame, id_col: str = "tau_id") -> list[str]:
    """Unique subject IDs that have at least one tau PET session (for targeted downloads)."""
    return sorted({parse_oasis_id(x)[0] for x in tau[id_col]})


def write_subject_list(subjects: Iterable[str], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(subjects) + "\n")
