"""CGM data-quality audit for benchmark integrity (esp. OpenAPS Data Commons).

Detects artefacts that can inflate or distort forecasting benchmarks:
- gaps and their length distribution;
- interpolated/imputed fraction;
- calibration jumps (abrupt level shifts inconsistent with physiology);
- duplicate/overlapping timestamps;
- flat-line (stuck-sensor) segments.

The audit produces per-stream flags so evaluation can be run on artefact-filtered
data and compared to naive inclusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRID_MIN = 5
MAX_PHYSIOLOGIC_RATE = 4.0  # mg/dL per minute; larger single-step jumps are suspect


def gap_report(grid: pd.DataFrame, freq_min: int = GRID_MIN) -> dict:
    """Summarise missing/imputed CGM in a gridded stream."""
    imp = grid["imputed"].to_numpy(bool) if "imputed" in grid else np.zeros(len(grid), bool)
    # run-length of imputed stretches
    runs = []
    c = 0
    for v in imp:
        if v:
            c += 1
        elif c:
            runs.append(c)
            c = 0
    if c:
        runs.append(c)
    return {
        "imputed_fraction": float(imp.mean()),
        "n_gaps": len(runs),
        "max_gap_min": int(max(runs) * freq_min) if runs else 0,
        "median_gap_min": float(np.median(runs) * freq_min) if runs else 0.0,
    }


def calibration_jumps(glucose: np.ndarray, freq_min: int = GRID_MIN,
                      rate_thresh: float = MAX_PHYSIOLOGIC_RATE) -> dict:
    """Count single-step glucose changes exceeding a physiologic rate limit."""
    g = np.asarray(glucose, float)
    rate = np.abs(np.diff(g)) / freq_min
    n_jumps = int(np.sum(rate > rate_thresh))
    return {"n_calibration_jumps": n_jumps,
            "jump_fraction": float(n_jumps / max(len(rate), 1)),
            "max_rate_mgdl_per_min": float(np.nanmax(rate)) if len(rate) else 0.0}


def flatline_segments(glucose: np.ndarray, min_len: int = 6, tol: float = 0.5) -> dict:
    """Detect stuck-sensor flat lines (>=min_len samples with ~constant value)."""
    g = np.asarray(glucose, float)
    flat = np.abs(np.diff(g)) < tol
    runs, c, longest = 0, 0, 0
    for v in flat:
        if v:
            c += 1
        else:
            if c + 1 >= min_len:
                runs += 1
                longest = max(longest, c + 1)
            c = 0
    if c + 1 >= min_len:
        runs += 1
        longest = max(longest, c + 1)
    return {"n_flatline_segments": runs, "longest_flatline": int(longest)}


def duplicate_timestamps(times) -> int:
    """Number of duplicate timestamps (overlapping/merged streams)."""
    t = pd.to_datetime(pd.Series(times))
    return int(t.duplicated().sum())


def audit_stream(grid: pd.DataFrame) -> dict:
    """Full per-stream artefact report combining the individual checks."""
    report = {}
    report.update(gap_report(grid))
    report.update(calibration_jumps(grid["glucose"].to_numpy(float)))
    report.update(flatline_segments(grid["glucose"].to_numpy(float)))
    report["n_duplicate_timestamps"] = duplicate_timestamps(grid["time"]) if "time" in grid else 0
    return report
