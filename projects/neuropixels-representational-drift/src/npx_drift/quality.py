"""Unit-quality filtering, strata and stability covariates; CCF area-group mapping."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

#: Allen Brain Observatory default unit filters (applied as strict inequalities).
ALLEN_DEFAULT_CRITERIA: Dict[str, tuple] = {
    "isi_violations": ("<", 0.5),
    "amplitude_cutoff": ("<", 0.1),
    "presence_ratio": (">", 0.95),
}

_AREA_GROUPS: Dict[str, Sequence[str]] = {
    "cortex": ("VIS", "VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl", "VISli", "VISpl", "VISmma",
               "VISmmp", "RSP", "ACA", "MOs", "MOp", "PL", "ILA", "ORB", "SSp", "SSs", "AUD", "TEa", "ECT", "PTLp"),
    "thalamus": ("LGd", "LGv", "LP", "LD", "MGd", "MGv", "MGm", "PO", "VPM", "VPL", "VAL", "VM", "MD",
                 "AV", "AM", "AD", "CL", "PF", "RT", "TH", "IGL", "SGN", "POL", "Eth", "PoT", "PIL", "LT"),
    "hippocampal": ("CA1", "CA2", "CA3", "DG", "DG-mo", "DG-po", "DG-sg", "SUB", "ProS", "POST", "PRE", "PAR", "HPF", "ENT", "ENTl", "ENTm"),
    "midbrain": ("SC", "SCs", "SCm", "SCig", "SCsg", "SCzo", "SCop", "SCiw", "SCdg", "MRN", "APN", "NOT", "MB", "PAG", "IC", "PPT"),
}


def area_group(acronym: str) -> str:
    """Map a CCF acronym (e.g. ``'VISp'``, ``'LGd'``, ``'CA1'``, ``'DG-mo'``) to a coarse group."""
    a = str(acronym)
    for group, prefixes in _AREA_GROUPS.items():
        if a in prefixes:
            return group
    for group, prefixes in _AREA_GROUPS.items():
        if any(a.startswith(p) for p in prefixes if len(p) >= 2):
            return group
    if a.startswith("VIS"):
        return "cortex"
    return "other"


def filter_units(units: pd.DataFrame, criteria: Optional[Dict[str, tuple]] = None,
                 ibl_label_col: str = "label", require_area: bool = True) -> pd.DataFrame:
    """Return the subset of ``units`` passing quality criteria.

    Allen-style metrics are compared with ``criteria`` (default
    :data:`ALLEN_DEFAULT_CRITERIA`); if an IBL ``label`` column is present,
    units with ``label == 1`` are kept instead. Missing metric columns are ignored.
    """
    criteria = ALLEN_DEFAULT_CRITERIA if criteria is None else criteria
    keep = np.ones(len(units), dtype=bool)
    if ibl_label_col in units:
        keep &= (units[ibl_label_col].astype(float) == 1).to_numpy()
    else:
        for col, (op, thr) in criteria.items():
            if col not in units:
                continue
            v = units[col].astype(float).to_numpy()
            keep &= (v < thr) if op == "<" else (v > thr)
    if require_area and "area" in units:
        keep &= units["area"].astype(str).ne("unknown").to_numpy()
    return units.loc[keep]


def quality_strata(units: pd.DataFrame, metrics: Sequence[str] = ("presence_ratio", "amplitude_cutoff", "isi_violations"),
                   n_bins: int = 3) -> pd.Series:
    """Composite quality tercile (or ``n_bins``-tile) per unit: higher = better isolation/stability."""
    score = np.zeros(len(units))
    n = 0
    for m in metrics:
        if m not in units:
            continue
        v = units[m].astype(float).rank(pct=True).to_numpy()
        score += v if m == "presence_ratio" else (1 - v)
        n += 1
    if n == 0:
        return pd.Series(np.zeros(len(units), int), index=units.index, name="quality_stratum")
    score /= n
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1)[1:-1])
    return pd.Series(np.searchsorted(edges, score, side="right"), index=units.index, name="quality_stratum")


def unit_stability_covariates(spike_times: Dict[int, np.ndarray], t_start: float, t_end: float,
                              n_blocks: int = 4, bin_s: float = 60.0,
                              amplitudes: Optional[Dict[int, np.ndarray]] = None) -> pd.DataFrame:
    """Per-unit within-session stability covariates.

    Columns: ``rate_cv`` (coefficient of variation of firing rate across ``n_blocks``
    equal time blocks), ``presence_ratio_bins`` (fraction of ``bin_s`` bins with >= 1
    spike), ``rate_slope`` (linear trend of block rates, Hz per block, normalized by
    mean rate), and ``amp_slope`` (normalized linear trend of spike amplitude if
    ``amplitudes`` is given).
    """
    edges = np.linspace(t_start, t_end, n_blocks + 1)
    nb = max(int((t_end - t_start) // bin_s), 1)
    rows = []
    for u, st in spike_times.items():
        st = st[(st >= t_start) & (st < t_end)]
        counts = np.histogram(st, edges)[0] / np.diff(edges)
        mean = counts.mean()
        cv = counts.std() / mean if mean > 0 else np.nan
        slope = np.polyfit(np.arange(n_blocks), counts, 1)[0] / mean if mean > 0 and n_blocks > 1 else np.nan
        pres = np.mean(np.histogram(st, np.linspace(t_start, t_end, nb + 1))[0] > 0)
        amp_slope = np.nan
        if amplitudes is not None and u in amplitudes and len(amplitudes[u]) == len(spike_times[u]) and len(st) > 10:
            a = amplitudes[u][(spike_times[u] >= t_start) & (spike_times[u] < t_end)]
            amp_slope = np.polyfit(st, a, 1)[0] * (t_end - t_start) / max(np.mean(a), 1e-12)
        rows.append(dict(unit_id=u, rate_cv=cv, presence_ratio_bins=pres, rate_slope=slope, amp_slope=amp_slope,
                         mean_rate=mean))
    return pd.DataFrame(rows).set_index("unit_id")


def add_area_groups(units: pd.DataFrame, col: str = "area") -> pd.DataFrame:
    out = units.copy()
    out["area_group"] = out[col].astype(str).map(area_group)
    return out
