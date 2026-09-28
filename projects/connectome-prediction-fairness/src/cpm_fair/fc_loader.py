"""Functional-connectivity loading utilities for HCP-style data.

Supports
--------
* CIFTI parcellated time series (``*.ptseries.nii``) and dense time series
  (``*.dtseries.nii``) via :mod:`nibabel` (optional dependency),
* plain-text node time series (HCP ``HCP_PTN1200`` ``NodeTimeseries`` files,
  one column per node, one row per TR) and ``.npy`` arrays,
* Fisher-z Pearson FC, vectorisation of the upper triangle,
* merging of the HCP *unrestricted* and *restricted* behavioural CSVs and
  the definition of demographic subgroups.

All loaders return arrays shaped ``(n_timepoints, n_nodes)``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional dependency
    import nibabel as nib  # type: ignore
except ImportError:  # pragma: no cover - exercised only without nibabel
    nib = None

HCP_REST_RUNS: Tuple[str, ...] = (
    "rfMRI_REST1_LR",
    "rfMRI_REST1_RL",
    "rfMRI_REST2_LR",
    "rfMRI_REST2_RL",
)

#: Column names in the HCP S1200 behavioural tables that this project uses.
#: ``restricted`` columns require the Restricted Data application.
HCP_COLUMNS: Dict[str, str] = {
    "subject": "Subject",
    "sex": "Gender",  # unrestricted (F/M)
    "age_range": "Age",  # unrestricted, 5-year bins
    "cognition": "CogTotalComp_Unadj",  # unrestricted, NIH Toolbox composite
    "fluid": "CogFluidComp_Unadj",
    "crystallised": "CogCrystalComp_Unadj",
    "motion": "Movement_RelativeRMS_mean",  # per-run in the imaging tables
    "family": "Family_ID",  # restricted
    "race": "Race",  # restricted
    "ethnicity": "Ethnicity",  # restricted
    "income": "SSAGA_Income",  # restricted (8 ordinal levels)
    "education": "SSAGA_Educ",  # restricted (years)
    "age_exact": "Age_in_Yrs",  # restricted
}


# --------------------------------------------------------------------------- #
# Time-series loaders
# --------------------------------------------------------------------------- #
def load_cifti_timeseries(path: str | Path) -> np.ndarray:
    """Load a CIFTI ``.dtseries.nii`` / ``.ptseries.nii`` as ``(T, N)``.

    Parameters
    ----------
    path
        Path to a CIFTI-2 file.

    Returns
    -------
    numpy.ndarray
        Float32 array with time along axis 0.
    """
    if nib is None:
        raise ImportError("nibabel is required to read CIFTI files: pip install nibabel")
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D CIFTI series, got shape {data.shape}")
    return data


def load_text_timeseries(path: str | Path) -> np.ndarray:
    """Load a whitespace/tab separated node time-series file as ``(T, N)``."""
    arr = np.loadtxt(str(path))
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32)


def load_timeseries(path: str | Path) -> np.ndarray:
    """Dispatch on file suffix (``.nii`` -> CIFTI, ``.npy``, otherwise text)."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return load_cifti_timeseries(p)
    if name.endswith(".npy"):
        return np.load(p).astype(np.float32)
    return load_text_timeseries(p)


# --------------------------------------------------------------------------- #
# FC computation
# --------------------------------------------------------------------------- #
def timeseries_to_fc(
    ts: np.ndarray, fisher_z: bool = True, standardize: bool = True
) -> np.ndarray:
    """Pearson functional connectivity ``(N, N)`` with zero diagonal.

    Parameters
    ----------
    ts
        ``(T, N)`` time series.
    fisher_z
        Apply ``arctanh`` to off-diagonal correlations.
    standardize
        Z-score each node's time series before correlating (no effect on
        Pearson r, but guards against constant columns via ``nan`` handling).
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError("ts must be (T, N)")
    if standardize:
        sd = ts.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan
        ts = (ts - ts.mean(axis=0)) / sd
    fc = np.corrcoef(ts, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 0.0)
    if fisher_z:
        fc = np.arctanh(np.clip(fc, -0.999999, 0.999999))
        np.fill_diagonal(fc, 0.0)
    return fc


def average_runs_fc(
    run_timeseries: Iterable[np.ndarray], fisher_z: bool = True
) -> np.ndarray:
    """Average Fisher-z FC matrices across runs (the standard HCP practice)."""
    mats = [timeseries_to_fc(ts, fisher_z=fisher_z) for ts in run_timeseries]
    if not mats:
        raise ValueError("no runs supplied")
    return np.mean(np.stack(mats, axis=0), axis=0)


def vectorize_upper(fc: np.ndarray) -> np.ndarray:
    """Return the strict upper triangle of a symmetric matrix as a vector."""
    n = fc.shape[0]
    iu = np.triu_indices(n, k=1)
    return fc[iu]


def unvectorize_upper(vec: np.ndarray, n_nodes: int) -> np.ndarray:
    """Inverse of :func:`vectorize_upper` (symmetric, zero diagonal)."""
    fc = np.zeros((n_nodes, n_nodes), dtype=float)
    iu = np.triu_indices(n_nodes, k=1)
    if vec.shape[0] != iu[0].shape[0]:
        raise ValueError("vector length does not match n_nodes")
    fc[iu] = vec
    fc = fc + fc.T
    return fc


# --------------------------------------------------------------------------- #
# Subject tables
# --------------------------------------------------------------------------- #
def load_hcp_subject_table(
    unrestricted_csv: str | Path,
    restricted_csv: Optional[str | Path] = None,
    subject_col: str = HCP_COLUMNS["subject"],
) -> pd.DataFrame:
    """Merge the HCP unrestricted and (optional) restricted behavioural tables.

    The restricted table is needed for ``Family_ID`` (family-aware CV),
    ``Race``/``Ethnicity`` and SES variables. Without it, a ``family``
    column equal to the subject ID is created and a warning emitted, because
    HCP contains ~ 450 families with >1 member and splitting them across
    folds leaks information.
    """
    unres = pd.read_csv(unrestricted_csv)
    unres[subject_col] = unres[subject_col].astype(str)
    if restricted_csv is not None:
        res = pd.read_csv(restricted_csv)
        res[subject_col] = res[subject_col].astype(str)
        overlap = [c for c in res.columns if c in unres.columns and c != subject_col]
        res = res.drop(columns=overlap)
        df = unres.merge(res, on=subject_col, how="left")
    else:
        df = unres.copy()
    fam_col = HCP_COLUMNS["family"]
    if fam_col in df.columns:
        df["family"] = df[fam_col].astype(str)
    else:
        warnings.warn(
            "No Family_ID column: family-aware CV will treat every subject as its "
            "own family. Obtain HCP restricted data to avoid leakage.",
            stacklevel=2,
        )
        df["family"] = df[subject_col]
    return df


def define_subgroups(
    df: pd.DataFrame,
    race_col: str = HCP_COLUMNS["race"],
    ethnicity_col: Optional[str] = HCP_COLUMNS["ethnicity"],
    min_n: int = 30,
    other_label: str = "Other/Multiple",
) -> pd.Series:
    """Build a categorical race/ethnicity subgroup label.

    Follows the convention of Li et al. (2022): Hispanic/Latino ethnicity is
    treated as its own category when the column is present, then race
    categories with fewer than ``min_n`` members are collapsed into
    ``other_label``. The label is a *social*, not biological, category and is
    used only to audit model behaviour.
    """
    if race_col not in df.columns:
        raise KeyError(f"{race_col} not in table (restricted data required)")
    grp = df[race_col].astype(str).str.strip().replace({"nan": "Unknown"})
    if ethnicity_col is not None and ethnicity_col in df.columns:
        hisp = df[ethnicity_col].astype(str).str.contains("Hispanic", case=False, na=False) & ~df[
            ethnicity_col
        ].astype(str).str.contains("Not", case=False, na=False)
        grp = grp.where(~hisp, "Hispanic/Latino")
    counts = grp.value_counts()
    rare = counts[counts < min_n].index
    grp = grp.where(~grp.isin(rare), other_label)
    return grp.astype("category")


def ses_index(
    df: pd.DataFrame,
    income_col: str = HCP_COLUMNS["income"],
    education_col: str = HCP_COLUMNS["education"],
) -> pd.Series:
    """Composite SES z-score = mean of z(income level) and z(years education)."""
    parts = []
    for col in (income_col, education_col):
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            parts.append((x - x.mean()) / x.std(ddof=1))
    if not parts:
        raise KeyError("neither income nor education columns are present")
    return pd.concat(parts, axis=1).mean(axis=1)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def build_fc_dataset(
    subject_ids: Sequence[str],
    path_fn: Callable[[str, str], Path],
    runs: Sequence[str] = HCP_REST_RUNS,
    loader: Callable[[str | Path], np.ndarray] = load_timeseries,
    fisher_z: bool = True,
    min_runs: int = 1,
) -> Tuple[np.ndarray, List[str]]:
    """Stack vectorised run-averaged FC for many subjects.

    Parameters
    ----------
    subject_ids
        Subject identifiers.
    path_fn
        ``path_fn(subject, run) -> Path`` to that run's time-series file.
    runs
        Run names to average (missing files are skipped).
    loader
        Function reading a time-series file to ``(T, N)``.
    min_runs
        Subjects with fewer available runs are dropped.

    Returns
    -------
    X, kept_ids
        ``X`` is ``(n_subjects, n_edges)``; ``kept_ids`` the subjects retained.
    """
    rows: List[np.ndarray] = []
    kept: List[str] = []
    for sid in subject_ids:
        series = []
        for run in runs:
            p = Path(path_fn(str(sid), run))
            if p.exists():
                series.append(loader(p))
        if len(series) < min_runs:
            warnings.warn(f"subject {sid}: only {len(series)} runs, skipped", stacklevel=2)
            continue
        fc = average_runs_fc(series, fisher_z=fisher_z)
        rows.append(vectorize_upper(fc))
        kept.append(str(sid))
    if not rows:
        raise RuntimeError("no subjects loaded")
    return np.vstack(rows), kept
