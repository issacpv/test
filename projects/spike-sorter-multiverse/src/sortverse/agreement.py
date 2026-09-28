"""Unit matching across sorters, consensus/orphan labelling and quality metrics.

A *sorting* is a ``dict`` mapping unit id -> sorted 1-D array of spike times in seconds. Agreement follows
SpikeInterface's definition: ``n_match / (n_a + n_b - n_match)`` with matches counted within ``delta`` seconds.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

Sorting = Mapping[Hashable, np.ndarray]


def count_matches(a: np.ndarray, b: np.ndarray, delta: float = 0.4e-3) -> int:
    """Number of spikes in ``a`` that have a spike in ``b`` within +/- delta (each spike used at most once)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0
    i = j = n = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= delta:
            n += 1
            i += 1
            j += 1
        elif d > 0:
            j += 1
        else:
            i += 1
    return n


def agreement_matrix(sa: Sorting, sb: Sorting, delta: float = 0.4e-3) -> pd.DataFrame:
    """Agreement scores for every (unit_a, unit_b) pair."""
    ua = list(sa)
    ub = list(sb)
    M = np.zeros((len(ua), len(ub)))
    for i, a in enumerate(ua):
        ta = np.sort(sa[a])
        for j, b in enumerate(ub):
            tb = np.sort(sb[b])
            m = count_matches(ta, tb, delta)
            denom = len(ta) + len(tb) - m
            M[i, j] = m / denom if denom > 0 else 0.0
    return pd.DataFrame(M, index=ua, columns=ub)


def match_units(agreement: pd.DataFrame, min_agreement: float = 0.5) -> pd.DataFrame:
    """One-to-one unit pairing by Hungarian assignment on (1 - agreement); pairs below threshold are dropped."""
    if agreement.size == 0:
        return pd.DataFrame(columns=["unit_a", "unit_b", "agreement"])
    cost = 1.0 - agreement.to_numpy()
    r, c = linear_sum_assignment(cost)
    rows = [{"unit_a": agreement.index[i], "unit_b": agreement.columns[j], "agreement": float(agreement.iat[i, j])}
            for i, j in zip(r, c) if agreement.iat[i, j] >= min_agreement]
    return pd.DataFrame(rows, columns=["unit_a", "unit_b", "agreement"])


def consensus_table(sortings: Mapping[str, Sorting], delta: float = 0.4e-3, min_agreement: float = 0.5,
                    reference: Optional[str] = None) -> pd.DataFrame:
    """Label every unit of every sorter by how many sorters found it.

    Units are chained through the reference sorter: for each non-reference sorter, units matched to a reference
    unit inherit that reference unit's group; unmatched units form their own group. ``n_sorters`` per group is
    the number of distinct sorters contributing. Orphans have n_sorters == 1.
    """
    names = list(sortings)
    reference = names[0] if reference is None else reference
    ref = sortings[reference]
    group_of: Dict[Tuple[str, Hashable], int] = {}
    next_group = 0
    for u in ref:
        group_of[(reference, u)] = next_group
        next_group += 1
    for name in names:
        if name == reference:
            continue
        pairs = match_units(agreement_matrix(ref, sortings[name], delta), min_agreement)
        matched = dict(zip(pairs["unit_b"], pairs["unit_a"]))
        for u in sortings[name]:
            if u in matched:
                group_of[(name, u)] = group_of[(reference, matched[u])]
            else:
                group_of[(name, u)] = next_group
                next_group += 1
    df = pd.DataFrame([{"sorter": s, "unit": u, "group": g, "n_spikes": len(sortings[s][u])}
                       for (s, u), g in group_of.items()])
    n_sorters = df.groupby("group")["sorter"].nunique()
    df["n_sorters"] = df["group"].map(n_sorters)
    df["orphan"] = df["n_sorters"] == 1
    return df


# ------------------------------------------------------------------------------------------------------
# Quality metrics (Hill, Mehta & Kleinfeld, 2011; as implemented in the Allen / SpikeInterface pipelines)
# ------------------------------------------------------------------------------------------------------

def isi_violations(spike_times: np.ndarray, duration_s: float, isi_threshold: float = 1.5e-3,
                   min_isi: float = 0.0) -> Dict[str, float]:
    """ISI-violation ratio (estimated contamination) and raw violation rate."""
    t = np.sort(np.asarray(spike_times, dtype=float))
    n = len(t)
    if n < 2 or duration_s <= 0:
        return {"isi_violations_ratio": np.nan, "isi_violations_rate": np.nan, "n_violations": 0}
    isis = np.diff(t)
    n_viol = int(np.sum(isis < isi_threshold))
    violation_time = 2.0 * n * (isi_threshold - min_isi)
    total_rate = n / duration_s
    violation_rate = n_viol / violation_time
    return {"isi_violations_ratio": float(violation_rate / total_rate), "isi_violations_rate": float(n_viol / n),
            "n_violations": n_viol}


def presence_ratio(spike_times: np.ndarray, duration_s: float, n_bins: int = 100) -> float:
    """Fraction of equal time bins containing at least one spike."""
    t = np.asarray(spike_times, dtype=float)
    if duration_s <= 0:
        return np.nan
    counts, _ = np.histogram(t, bins=n_bins, range=(0.0, duration_s))
    return float(np.mean(counts > 0))


def amplitude_cutoff(amplitudes: np.ndarray, n_bins: int = 100, smoothing_bins: int = 3) -> float:
    """Approximate fraction of missed spikes from the truncated low tail of the amplitude histogram.

    The histogram peak is located and the high-side tail is mirrored to the low side; the mass beyond the
    lowest observed amplitude is the estimated missing fraction (capped at 0.5)."""
    a = np.asarray(amplitudes, dtype=float)
    if len(a) < 10:
        return np.nan
    h, edges = np.histogram(a, bins=n_bins, density=True)
    k = np.ones(smoothing_bins) / smoothing_bins
    hs = np.convolve(h, k, mode="same")
    peak = int(np.argmax(hs))
    # height of the pdf at the lowest observed amplitude
    low_val = hs[0]
    # find where the high-side tail falls to that height
    above = np.flatnonzero(hs[peak:] <= low_val)
    if len(above) == 0:
        return 0.5
    g = peak + above[0]
    width = edges[1] - edges[0]
    missing = float(np.sum(hs[g:]) * width)
    return float(min(missing, 0.5))


def firing_rate(spike_times: np.ndarray, duration_s: float) -> float:
    return float(len(spike_times) / duration_s) if duration_s > 0 else np.nan


def qc_table(sorting: Sorting, duration_s: float, amplitudes: Optional[Mapping[Hashable, np.ndarray]] = None
             ) -> pd.DataFrame:
    """Per-unit QC metrics."""
    rows = []
    for u, t in sorting.items():
        row = {"unit": u, "n_spikes": len(t), "firing_rate": firing_rate(t, duration_s),
               "presence_ratio": presence_ratio(t, duration_s), **isi_violations(t, duration_s)}
        if amplitudes is not None and u in amplitudes:
            row["amplitude_cutoff"] = amplitude_cutoff(amplitudes[u])
            row["amplitude_median"] = float(np.median(amplitudes[u]))
        rows.append(row)
    return pd.DataFrame(rows)


QC_RULES: Dict[str, Dict[str, float]] = {
    # Allen Visual Coding default filters
    "allen_default": {"isi_violations_ratio_max": 0.5, "amplitude_cutoff_max": 0.1, "presence_ratio_min": 0.9},
    "lenient": {"isi_violations_ratio_max": 1.0, "amplitude_cutoff_max": 0.3, "presence_ratio_min": 0.5},
    "none": {"isi_violations_ratio_max": np.inf, "amplitude_cutoff_max": np.inf, "presence_ratio_min": 0.0},
}


def apply_qc(qc: pd.DataFrame, rule: str = "allen_default") -> np.ndarray:
    """Boolean mask of units passing a named QC rule (missing metrics are treated as passing)."""
    r = QC_RULES[rule]
    ok = np.ones(len(qc), dtype=bool)
    if "isi_violations_ratio" in qc:
        ok &= ~(qc["isi_violations_ratio"] > r["isi_violations_ratio_max"]).to_numpy()
    if "amplitude_cutoff" in qc:
        ok &= ~(qc["amplitude_cutoff"] > r["amplitude_cutoff_max"]).to_numpy()
    if "presence_ratio" in qc:
        ok &= ~(qc["presence_ratio"] < r["presence_ratio_min"]).to_numpy()
    return ok


def perturb_sorting(sorting: Sorting, rng: np.random.Generator, jitter_s: float = 0.1e-3, drop_frac: float = 0.1,
                    add_noise_units: int = 0, duration_s: float = 60.0, merge_pairs: Sequence[Tuple[Hashable, Hashable]] = ()
                    ) -> Dict[Hashable, np.ndarray]:
    """Simulate another sorter's output: jitter, missed spikes, extra noise units and merges (for tests/nulls)."""
    out: Dict[Hashable, np.ndarray] = {}
    merged = {b: a for a, b in merge_pairs}
    for u, t in sorting.items():
        t = np.asarray(t, dtype=float)
        keep = rng.random(len(t)) >= drop_frac
        tt = np.sort(t[keep] + rng.normal(0.0, jitter_s, keep.sum()))
        target = merged.get(u, u)
        out[target] = np.sort(np.concatenate([out.get(target, np.array([])), tt]))
    for k in range(add_noise_units):
        out[f"noise{k}"] = np.sort(rng.uniform(0.0, duration_s, size=int(rng.integers(50, 200))))
    return out
