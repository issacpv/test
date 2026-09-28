"""Confound strategies for fMRIPrep outputs.

The strategy families follow Ciric et al. (2017, NeuroImage), Parkes et al.
(2018, NeuroImage) and the fMRIPrep/Nilearn benchmark of Wang et al. (2024,
PLoS Comput Biol). Each strategy is defined declaratively and can be applied
either through ``nilearn.interfaces.fmriprep.load_confounds`` (if nilearn is
installed) or directly on the ``*_desc-confounds_timeseries.tsv`` with pandas
(:func:`select_confounds`), which makes the multiverse runnable on parcel
time series without any imaging dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


@dataclass(frozen=True)
class ConfoundStrategy:
    name: str
    motion: str = "basic"  # basic (6) | derivatives (12) | full (24) | none
    wm_csf: str = "none"  # none | basic (2) | full (8)
    global_signal: str = "none"  # none | basic (1) | full (4)
    compcor: Optional[str] = None  # None | anat_combined | anat_separated | temporal
    n_compcor: int = 5
    high_pass_cosine: bool = True
    scrub_fd: Optional[float] = None  # framewise-displacement threshold (mm)
    scrub_dvars: Optional[float] = None
    min_segment: int = 5  # Power et al. 2014: drop segments shorter than this
    ica_aroma: bool = False
    description: str = ""

    def nilearn_kwargs(self) -> Dict[str, object]:
        """Keyword arguments for ``nilearn.interfaces.fmriprep.load_confounds``."""
        strategy: List[str] = []
        kw: Dict[str, object] = {}
        if self.motion != "none":
            strategy.append("motion")
            kw["motion"] = self.motion
        if self.wm_csf != "none":
            strategy.append("wm_csf")
            kw["wm_csf"] = self.wm_csf
        if self.global_signal != "none":
            strategy.append("global_signal")
            kw["global_signal"] = self.global_signal
        if self.compcor:
            strategy.append("compcor")
            kw["compcor"] = self.compcor
            kw["n_compcor"] = self.n_compcor
        if self.high_pass_cosine:
            strategy.append("high_pass")
        if self.scrub_fd is not None:
            strategy.append("scrub")
            kw["scrub"] = self.min_segment
            kw["fd_threshold"] = self.scrub_fd
            kw["std_dvars_threshold"] = self.scrub_dvars if self.scrub_dvars is not None else 3
        if self.ica_aroma:
            strategy.append("ica_aroma")
            kw["ica_aroma"] = "full"
        kw["strategy"] = strategy
        return kw


STRATEGIES: Dict[str, ConfoundStrategy] = {
    "hmp6": ConfoundStrategy("hmp6", motion="basic", high_pass_cosine=True, description="6 motion parameters + cosine drifts"),
    "hmp24": ConfoundStrategy("hmp24", motion="full", description="24 motion (Friston) + cosine"),
    "hmp24_phys2": ConfoundStrategy("hmp24_phys2", motion="full", wm_csf="basic", description="24HMP + mean WM/CSF"),
    "hmp24_phys8": ConfoundStrategy("hmp24_phys8", motion="full", wm_csf="full", description="24HMP + 8 physiological (Satterthwaite/Ciric '36P' minus GSR)"),
    "hmp24_phys8_gsr4": ConfoundStrategy("hmp24_phys8_gsr4", motion="full", wm_csf="full", global_signal="full", description="36P: 24HMP + 8Phys + 4GSR"),
    "acompcor": ConfoundStrategy("acompcor", motion="full", compcor="anat_combined", n_compcor=5, description="24HMP + 5 aCompCor (combined WM+CSF mask) + cosine"),
    "acompcor_gsr": ConfoundStrategy("acompcor_gsr", motion="full", compcor="anat_combined", n_compcor=5, global_signal="basic", description="aCompCor + GSR"),
    "hmp24_phys8_scrub": ConfoundStrategy("hmp24_phys8_scrub", motion="full", wm_csf="full", scrub_fd=0.5, description="36P-like with FD>0.5 mm scrubbing (Power)"),
    "aroma": ConfoundStrategy("aroma", motion="none", wm_csf="basic", ica_aroma=True, description="ICA-AROMA non-aggressive + WM/CSF (requires AROMA outputs)"),
    "aroma_gsr": ConfoundStrategy("aroma_gsr", motion="none", wm_csf="basic", global_signal="basic", ica_aroma=True, description="AROMA + GSR"),
}


# --------------------------------------------------------------------------- #
# Direct TSV implementation
# --------------------------------------------------------------------------- #
def _expand(df: pd.DataFrame, cols: Sequence[str], level: str) -> List[str]:
    """Return the column names for basic / derivatives / full expansions."""
    out = list(cols)
    if level in ("derivatives", "full"):
        out += [f"{c}_derivative1" for c in cols]
    if level == "full":
        out += [f"{c}_power2" for c in cols] + [f"{c}_derivative1_power2" for c in cols]
    missing = [c for c in out if c not in df.columns]
    if missing:
        # older fMRIPrep versions lack derivative/power columns: compute them
        for c in missing:
            base = c.replace("_derivative1_power2", "").replace("_derivative1", "").replace("_power2", "")
            if base not in df.columns:
                raise KeyError(f"confound column {base} missing")
            series = df[base].astype(float)
            if "_derivative1" in c:
                series = series.diff().fillna(0.0)
            if "_power2" in c:
                series = series**2
            df[c] = series
    return out


def _compcor_columns(df: pd.DataFrame, kind: str, n: int, meta: Optional[dict] = None) -> List[str]:
    if kind == "temporal":
        cols = [c for c in df.columns if re.match(r"t_comp_cor_\d+", c)]
        return cols[:n]
    cols = [c for c in df.columns if re.match(r"a_comp_cor_\d+", c)]
    if meta:
        # fMRIPrep >= 1.4 JSON sidecar says which mask each component belongs to
        by_mask: Dict[str, List[str]] = {}
        for c in cols:
            m = meta.get(c, {}).get("Mask")
            if m:
                by_mask.setdefault(m, []).append(c)
        if kind == "anat_combined" and "combined" in by_mask:
            return by_mask["combined"][:n]
        if kind == "anat_separated" and {"WM", "CSF"} <= set(by_mask):
            return by_mask["WM"][:n] + by_mask["CSF"][:n]
    return cols[:n]


def select_confounds(
    confounds: pd.DataFrame | str | Path,
    strategy: ConfoundStrategy | str,
    meta: Optional[dict] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply a strategy to an fMRIPrep confounds table.

    Parameters
    ----------
    confounds
        DataFrame or path to ``*_desc-confounds_timeseries.tsv`` (or the older
        ``*_desc-confounds_regressors.tsv``).
    strategy
        :class:`ConfoundStrategy` or a key of :data:`STRATEGIES`.
    meta
        Optional parsed JSON sidecar (maps aCompCor components to masks).

    Returns
    -------
    (regressors, sample_mask)
        ``regressors`` has one column per nuisance regressor (NaNs in the
        first row from derivatives are filled with 0); ``sample_mask`` is a
        boolean array with ``False`` for volumes removed by scrubbing.
    """
    if isinstance(strategy, str):
        strategy = STRATEGIES[strategy]
    df = pd.read_csv(confounds, sep="\t") if not isinstance(confounds, pd.DataFrame) else confounds.copy()
    df = df.replace("n/a", np.nan)
    cols: List[str] = []
    if strategy.ica_aroma:
        cols += [c for c in df.columns if c.startswith("aroma_motion_")]
        if not cols:
            raise KeyError("no aroma_motion_* columns: run fMRIPrep with --use-aroma (<= 23.0) or fMRIPost-AROMA")
    if strategy.motion != "none":
        cols += _expand(df, MOTION6, strategy.motion)
    if strategy.wm_csf != "none":
        cols += _expand(df, ["white_matter", "csf"], strategy.wm_csf)
    if strategy.global_signal != "none":
        cols += _expand(df, ["global_signal"], strategy.global_signal)
    if strategy.compcor:
        cols += _compcor_columns(df, strategy.compcor, strategy.n_compcor, meta)
    if strategy.high_pass_cosine:
        cols += [c for c in df.columns if c.startswith("cosine")]
    reg = df[cols].astype(float).fillna(0.0)
    mask = np.ones(len(df), dtype=bool)
    if strategy.scrub_fd is not None:
        fd = pd.to_numeric(df.get("framewise_displacement"), errors="coerce").fillna(0.0).to_numpy()
        bad = fd > strategy.scrub_fd
        if strategy.scrub_dvars is not None and "std_dvars" in df:
            dv = pd.to_numeric(df["std_dvars"], errors="coerce").fillna(0.0).to_numpy()
            bad |= dv > strategy.scrub_dvars
        mask = ~bad
        mask = _drop_short_segments(mask, strategy.min_segment)
    return reg, mask


def _drop_short_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Remove surviving segments shorter than ``min_len`` volumes."""
    mask = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i < min_len:
                mask[i:j] = False
            i = j
        else:
            i += 1
    return mask


def motion_summary(confounds: pd.DataFrame | str | Path, fd_threshold: float = 0.5) -> Dict[str, float]:
    """Mean/max FD and fraction of volumes above threshold (for QC and fragility features)."""
    df = pd.read_csv(confounds, sep="\t") if not isinstance(confounds, pd.DataFrame) else confounds
    fd = pd.to_numeric(df.get("framewise_displacement"), errors="coerce").dropna().to_numpy()
    if len(fd) == 0:
        return {"mean_fd": np.nan, "max_fd": np.nan, "frac_fd_above": np.nan, "n_volumes": float(len(df))}
    return {"mean_fd": float(fd.mean()), "max_fd": float(fd.max()), "frac_fd_above": float((fd > fd_threshold).mean()), "n_volumes": float(len(df))}


def load_confounds_for(bold_path: str | Path, strategy: ConfoundStrategy | str, use_nilearn: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load confounds for a preprocessed BOLD file.

    Tries nilearn's ``load_confounds`` (which resolves the TSV and JSON
    sidecar from the BOLD filename) and falls back to :func:`select_confounds`
    on the sibling confounds TSV.
    """
    if isinstance(strategy, str):
        strategy = STRATEGIES[strategy]
    bold_path = Path(bold_path)
    if use_nilearn:
        try:
            from nilearn.interfaces.fmriprep import load_confounds  # type: ignore

            reg, mask = load_confounds(str(bold_path), **strategy.nilearn_kwargs())
            n = len(reg)
            m = np.zeros(n, dtype=bool) if mask is not None else np.ones(n, dtype=bool)
            if mask is not None:
                m[np.asarray(mask)] = True
            return reg, m
        except ImportError:
            pass
    stem = bold_path.name.split("_space-")[0]
    cands = list(bold_path.parent.glob(f"{stem}_desc-confounds_timeseries.tsv")) + list(bold_path.parent.glob(f"{stem}_desc-confounds_regressors.tsv"))
    if not cands:
        raise FileNotFoundError(f"no confounds TSV next to {bold_path}")
    meta = None
    js = cands[0].with_suffix(".json")
    if js.exists():
        import json

        meta = json.loads(js.read_text())
    return select_confounds(cands[0], strategy, meta=meta)
