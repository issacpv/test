"""Decision impact of damping: NIBP-ABP discrepancy, threshold crossings and event enrichment by damping class.

All functions take *minute-level* (or window-level) tables with a ``damping_class`` column
(``adequate`` / ``underdamped`` / ``overdamped`` / ``unlabelled``) produced by
``flush.propagate_labels`` or by the flush-free classifier in ``sqi``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

CLASSES = ("adequate", "underdamped", "overdamped")


def pair_nibp_abp(abp_windows: pd.DataFrame, nibp: pd.DataFrame, tolerance_s: float = 90.0) -> pd.DataFrame:
    """For every NIBP reading, the median ABP window values within +-tolerance; returns paired deltas.

    ``abp_windows`` needs ``t_center, sbp, dbp, map, damping_class``; ``nibp`` needs ``t, sbp, dbp, map``.
    """
    rows = []
    t_abp = abp_windows["t_center"].to_numpy(float)
    for _, r in nibp.iterrows():
        m = np.abs(t_abp - float(r["t"])) <= tolerance_s
        if not m.any():
            continue
        sub = abp_windows[m]
        cls = sub["damping_class"].mode().iloc[0]
        rows.append({"t": float(r["t"]), "damping_class": cls,
                     "d_sbp": float(sub["sbp"].median() - r["sbp"]),
                     "d_dbp": float(sub["dbp"].median() - r["dbp"]),
                     "d_map": float(sub["map"].median() - r["map"])})
    return pd.DataFrame(rows, columns=["t", "damping_class", "d_sbp", "d_dbp", "d_map"])


def discrepancy_by_class(paired: pd.DataFrame) -> pd.DataFrame:
    """Bias and SD of ABP-minus-NIBP differences per damping class (the clinical signature of damping)."""
    g = paired.groupby("damping_class")
    out = g[["d_sbp", "d_dbp", "d_map"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n"] = g.size()
    return out.reset_index()


def threshold_crossing_share(minutes: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Share of sub-threshold minutes (hypotension flags) falling in each damping class vs the class's exposure share.

    ``enrichment`` = share of flagged minutes in the class / share of all minutes in the class.
    """
    thresholds = thresholds or {"map": 65.0, "sbp": 90.0}
    exposure = minutes["damping_class"].value_counts(normalize=True)
    rows = []
    for var, thr in thresholds.items():
        flagged = minutes[minutes[var] < thr]
        share = flagged["damping_class"].value_counts(normalize=True)
        for cls in exposure.index:
            rows.append({"variable": var, "threshold": thr, "damping_class": cls,
                         "n_flagged": int((flagged["damping_class"] == cls).sum()),
                         "flag_share": float(share.get(cls, 0.0)), "exposure_share": float(exposure[cls]),
                         "enrichment": float(share.get(cls, 0.0) / exposure[cls])})
    return pd.DataFrame(rows)


def event_rate_ratio(events_t: np.ndarray, minutes: pd.DataFrame, reference: str = "adequate") -> pd.DataFrame:
    """Events per exposure-hour by damping class and the rate ratio vs the reference class.

    ``events_t`` are event times (s), e.g. vasopressor rate changes from MIMIC ``inputevents``;
    ``minutes`` needs ``t_center`` (s) and ``damping_class`` at 60-s resolution.
    """
    t = minutes["t_center"].to_numpy(float)
    cls = minutes["damping_class"].to_numpy()
    idx = np.clip(np.searchsorted(t, np.asarray(events_t, float)), 0, len(t) - 1)
    ev_cls = cls[idx]
    rows = []
    for c in pd.unique(cls):
        hours = float((cls == c).sum()) / 60.0
        n_ev = int((ev_cls == c).sum())
        rows.append({"damping_class": c, "events": n_ev, "exposure_h": hours, "rate_per_h": n_ev / hours if hours else np.nan})
    df = pd.DataFrame(rows).set_index("damping_class")
    ref_rate = df.loc[reference, "rate_per_h"] if reference in df.index else np.nan
    df["rate_ratio"] = df["rate_per_h"] / ref_rate
    return df.reset_index()


def cluster_bootstrap(df: pd.DataFrame, group: str, stat: Callable[[pd.DataFrame], float],
                      n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Record-level bootstrap of any statistic: (point, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    ids = df[group].unique()
    parts = {i: g for i, g in df.groupby(group)}
    point = float(stat(df))
    vals = [float(stat(pd.concat([parts[i] for i in rng.choice(ids, len(ids), replace=True)], ignore_index=True)))
            for _ in range(n_boot)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)
