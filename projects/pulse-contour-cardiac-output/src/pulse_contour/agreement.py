"""Method-comparison statistics for cardiac output / stroke volume validation.

* ``bland_altman``            bias and limits of agreement; with ``subject`` given, the
                              repeated-measures variance decomposition of Bland & Altman
                              (J Biopharm Stat, 2007) is used.
* ``percentage_error``        1.96 SD / mean reference (Critchley & Critchley, J Clin Monit Comput 1999);
                              the conventional interchangeability threshold is 30%.
* ``four_quadrant``           trending concordance rate with an exclusion zone (Critchley et al., 2010).
* ``polar_concordance``       polar-plot angular agreement (Critchley et al., Anesth Analg 2010).
* ``bootstrap_by_subject``    cluster bootstrap CIs for any statistic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BlandAltman:
    bias: float
    sd: float
    loa_low: float
    loa_high: float
    n: int
    n_subjects: int
    var_between: float
    var_within: float


def bland_altman(ref: np.ndarray, est: np.ndarray, subject: np.ndarray | None = None) -> BlandAltman:
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    m = np.isfinite(ref) & np.isfinite(est)
    d = est[m] - ref[m]
    if subject is None:
        sd = float(d.std(ddof=1)) if d.size > 1 else float("nan")
        return BlandAltman(float(d.mean()), sd, float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd),
                           int(d.size), int(d.size), sd ** 2, 0.0)
    s = np.asarray(subject)[m]
    df = pd.DataFrame({"d": d, "s": s})
    g = df.groupby("s")["d"]
    k = g.ngroups
    n_i = g.size().to_numpy(float)
    N = n_i.sum()
    grand = d.mean()
    means = g.mean().to_numpy()
    ss_between = float(np.sum(n_i * (means - grand) ** 2))
    ss_within = float(np.sum((d - df["s"].map(g.mean()).to_numpy()) ** 2))
    msb = ss_between / (k - 1) if k > 1 else 0.0
    msw = ss_within / (N - k) if N > k else 0.0
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1) if k > 1 else 1.0
    var_between = max((msb - msw) / n0, 0.0)
    var_total = var_between + msw
    sd = float(np.sqrt(var_total))
    return BlandAltman(float(grand), sd, float(grand - 1.96 * sd), float(grand + 1.96 * sd), int(N), int(k), var_between, msw)


def percentage_error(ba: BlandAltman, ref_mean: float) -> float:
    """Critchley & Critchley percentage error (%) = 100 * 1.96 SD / mean reference."""
    return float(100.0 * 1.96 * ba.sd / ref_mean)


def trend_deltas(df: pd.DataFrame, subject: str, time: str, ref: str, est: str) -> pd.DataFrame:
    """Consecutive-pair changes per subject: ``subject, d_ref, d_est``."""
    rows = []
    for sid, sub in df.sort_values([subject, time]).groupby(subject):
        r, e = sub[ref].to_numpy(float), sub[est].to_numpy(float)
        for i in range(1, len(sub)):
            rows.append({subject: sid, "d_ref": r[i] - r[i - 1], "d_est": e[i] - e[i - 1]})
    return pd.DataFrame(rows, columns=[subject, "d_ref", "d_est"])


def four_quadrant(d_ref: np.ndarray, d_est: np.ndarray, exclusion: float = 0.0) -> dict[str, float]:
    """Concordance rate (%) of the sign of changes, ignoring pairs inside the central exclusion zone."""
    d_ref, d_est = np.asarray(d_ref, float), np.asarray(d_est, float)
    keep = (np.abs(d_ref) > exclusion) & (np.abs(d_est) > exclusion) & np.isfinite(d_ref) & np.isfinite(d_est)
    if keep.sum() == 0:
        return {"concordance_pct": float("nan"), "n_included": 0, "n_total": int(d_ref.size)}
    conc = np.sign(d_ref[keep]) == np.sign(d_est[keep])
    return {"concordance_pct": float(100 * conc.mean()), "n_included": int(keep.sum()), "n_total": int(d_ref.size)}


def polar_concordance(d_ref: np.ndarray, d_est: np.ndarray, exclusion: float = 0.0,
                      angle_limit_deg: float = 30.0) -> dict[str, float]:
    """Polar plot statistics: mean angular bias and % of points within +-angle_limit of the 0-degree axis."""
    d_ref, d_est = np.asarray(d_ref, float), np.asarray(d_est, float)
    mean_change = (d_ref + d_est) / 2
    keep = (np.abs(mean_change) > exclusion) & np.isfinite(d_ref) & np.isfinite(d_est)
    if keep.sum() == 0:
        return {"angular_bias_deg": float("nan"), "radial_loa_pct": float("nan"), "n_included": 0}
    # angle from the line of identity (45 deg in the d_ref/d_est plane); third-quadrant points
    # (both methods decreasing) are mirrored onto the 0-degree axis as in Critchley's polar plot
    theta = np.degrees(np.arctan2(d_est[keep], d_ref[keep])) - 45.0
    theta = (theta + 180.0) % 360.0 - 180.0
    theta = np.where(theta > 90.0, theta - 180.0, np.where(theta < -90.0, theta + 180.0, theta))
    return {"angular_bias_deg": float(theta.mean()), "radial_loa_pct": float(100 * (np.abs(theta) <= angle_limit_deg).mean()),
            "n_included": int(keep.sum())}


def bootstrap_by_subject(stat: Callable[[pd.DataFrame], float], df: pd.DataFrame, subject: str,
                         n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap over subjects; returns (point estimate, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    ids = df[subject].unique()
    groups = {i: g for i, g in df.groupby(subject)}
    point = float(stat(df))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        vals.append(float(stat(pd.concat([groups[i] for i in pick], ignore_index=True))))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)
