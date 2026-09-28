"""Dataset loaders, ECG-echo pairing and leakage-safe splits.

PTB-XL and EchoNet-Dynamic ship CSV manifests; MIMIC-IV-ECG and MIMIC-IV-ECHO
are linked by ``subject_id`` and acquisition time. Signal reading is lazy
(``wfdb``) so the module imports without the data present.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

PTBXL_TRAIN_FOLDS = tuple(range(1, 9))
PTBXL_VAL_FOLD = 9
PTBXL_TEST_FOLD = 10


def load_ptbxl_metadata(root: Path | str) -> pd.DataFrame:
    """Load ``ptbxl_database.csv`` with parsed SCP codes and diagnostic superclass flags.

    Adds boolean columns ``NORM, MI, STTC, CD, HYP`` (superclasses) and the
    ``fold`` column (``strat_fold``).
    """
    root = Path(root)
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    code_to_class = scp["diagnostic_class"].to_dict()
    for cls in sorted(set(code_to_class.values())):
        df[cls] = df["scp_codes"].apply(lambda d, c=cls: any(code_to_class.get(k) == c for k in d))
    df["fold"] = df["strat_fold"].astype(int)
    return df


def ptbxl_split(df: pd.DataFrame) -> dict[str, pd.Index]:
    """Official PTB-XL split: folds 1-8 train, 9 validation, 10 test (patient-disjoint)."""
    return {
        "train": df.index[df["fold"].isin(PTBXL_TRAIN_FOLDS)],
        "val": df.index[df["fold"] == PTBXL_VAL_FOLD],
        "test": df.index[df["fold"] == PTBXL_TEST_FOLD],
    }


def load_ptbxl_signal(root: Path | str, row: pd.Series, sampling_rate: int = 100) -> tuple[np.ndarray, list[str]]:
    """Read one record with wfdb; returns ``(signal [T, 12], lead_names)``."""
    import wfdb

    key = "filename_lr" if sampling_rate == 100 else "filename_hr"
    rec = wfdb.rdrecord(str(Path(root) / row[key]))
    return rec.p_signal.astype(np.float32), list(rec.sig_name)


def load_echonet_filelist(root: Path | str) -> pd.DataFrame:
    """Load EchoNet-Dynamic ``FileList.csv`` (EF, ESV, EDV, FPS, NumberOfFrames, Split)."""
    df = pd.read_csv(Path(root) / "FileList.csv")
    df["FileName"] = df["FileName"].astype(str).str.replace(r"\.avi$", "", regex=True)
    return df


def pair_ecg_echo(
    ecg: pd.DataFrame,
    echo: pd.DataFrame,
    subject_col: str = "subject_id",
    ecg_time_col: str = "ecg_time",
    echo_time_col: str = "echo_time",
    max_days: float = 7.0,
    one_per_echo: bool = True,
) -> pd.DataFrame:
    """Pair each echo with the nearest-in-time ECG of the same subject within ``max_days``.

    Returns the echo rows with ``ecg_index`` and ``delay_days`` (ECG time minus
    echo time; negative = ECG before echo). Echoes without a candidate are
    dropped. If ``one_per_echo`` is False, all ECGs within the window are kept.
    """
    e = ecg.reset_index().rename(columns={"index": "ecg_index"})
    e[ecg_time_col] = pd.to_datetime(e[ecg_time_col])
    h = echo.reset_index().rename(columns={"index": "echo_index"})
    h[echo_time_col] = pd.to_datetime(h[echo_time_col])
    merged = h.merge(e, on=subject_col, how="inner", suffixes=("", "_ecg"))
    merged["delay_days"] = (merged[ecg_time_col] - merged[echo_time_col]).dt.total_seconds() / 86400.0
    merged = merged[merged["delay_days"].abs() <= max_days].copy()
    if one_per_echo:
        merged["abs_delay"] = merged["delay_days"].abs()
        merged = merged.sort_values(["echo_index", "abs_delay"]).drop_duplicates("echo_index").drop(columns="abs_delay")
    return merged.reset_index(drop=True)


def patient_level_split(
    subject_ids: np.ndarray | pd.Series,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> np.ndarray:
    """Assign rows to ``train/val/test`` so that all rows of a subject share a split."""
    ids = np.asarray(subject_ids)
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    label = {}
    for i, s in enumerate(unique):
        label[s] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return np.array([label[s] for s in ids])
