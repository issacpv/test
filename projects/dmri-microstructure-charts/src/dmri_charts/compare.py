"""Model-comparison metrics for normative charts.

A chart is graded on: (1) age sensitivity, (2) reliability of individual deviations, (3) bias of
deviations when the protocol changes, (4) sensitivity to clinical deviation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .normative import NormativeModel


def age_explained_variance(model: NormativeModel, age, y, sex=None, site=None) -> Dict[str, float]:
    """R^2 of the fitted age curve (sex/site held at reference) and the normalised late-life slope.

    ``slope_late_per_decade_sd`` is the change of the mean curve per decade after age 40 in
    units of the model's SD at 40 (an effect size of aging).
    """
    age, y = np.asarray(age, float), np.asarray(y, float)
    mu, sigma = model.predict(age, sex, site)
    ss_res = ((y - mu) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    grid = np.array([40.0, 50.0, 60.0, 70.0])
    inside = (grid >= model.knots_[0]) & (grid <= model.knots_[-1])
    if inside.sum() >= 2:
        mg, sg = model.predict(grid[inside])
        slope = np.polyfit(grid[inside], mg, 1)[0] * 10.0 / sg[0]
    else:
        slope = np.nan
    return {"r2_age": float(r2), "slope_late_per_decade_sd": float(slope)}


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1) (two-way random, absolute agreement, single measurement) for two sessions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    Y = np.column_stack([a[ok], b[ok]])
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * ((Y.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((Y.mean(0) - grand) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def protocol_transfer_bias(z_ref: np.ndarray, z_alt: np.ndarray) -> Dict[str, float]:
    """Paired comparison of z-scores from the full protocol vs an emulated protocol."""
    z_ref, z_alt = np.asarray(z_ref, float), np.asarray(z_alt, float)
    ok = np.isfinite(z_ref) & np.isfinite(z_alt)
    d = z_alt[ok] - z_ref[ok]
    sd = d.std(ddof=1) if ok.sum() > 1 else np.nan
    return {
        "mean_shift": float(d.mean()),
        "sd_shift": float(sd),
        "cohen_d": float(d.mean() / sd) if sd and sd > 0 else np.nan,
        "corr": float(np.corrcoef(z_ref[ok], z_alt[ok])[0, 1]) if ok.sum() > 2 else np.nan,
        "loa_low": float(d.mean() - 1.96 * sd),
        "loa_high": float(d.mean() + 1.96 * sd),
        "n": int(ok.sum()),
    }


def deviation_auc(z_controls: np.ndarray, z_patients: np.ndarray, direction: str = "auto") -> Dict[str, float]:
    """AUC of |z| (or signed z) for patients vs controls via the Mann-Whitney U statistic."""
    zc, zp = np.asarray(z_controls, float), np.asarray(z_patients, float)
    zc, zp = zc[np.isfinite(zc)], zp[np.isfinite(zp)]
    if direction == "abs":
        xc, xp = np.abs(zc), np.abs(zp)
    elif direction == "auto":
        sign = 1.0 if zp.mean() >= zc.mean() else -1.0
        xc, xp = sign * zc, sign * zp
    else:
        xc, xp = zc, zp
    u = stats.mannwhitneyu(xp, xc, alternative="greater").statistic
    auc = float(u / (len(xc) * len(xp)))
    return {"auc": auc, "frac_patients_abnormal": float((np.abs(zp) > 1.96).mean()), "frac_controls_abnormal": float((np.abs(zc) > 1.96).mean())}


def rank_models(table: pd.DataFrame, higher_is_better: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
    """Rank-sum leaderboard. ``table`` has one row per model and metric columns.

    ``higher_is_better`` defaults to: r2_age, icc, auc, corr -> True; cohen_d (absolute) -> False.
    """
    defaults = {"r2_age": True, "slope_late_per_decade_sd": None, "icc": True, "auc": True, "corr": True, "cohen_d": False, "mean_shift": False}
    hib = {**defaults, **(higher_is_better or {})}
    ranks = pd.DataFrame(index=table.index)
    for col in table.columns:
        if col not in hib or hib[col] is None:
            continue
        vals = table[col].abs() if col in ("cohen_d", "mean_shift", "slope_late_per_decade_sd") else table[col]
        ranks[f"rank_{col}"] = vals.rank(ascending=not hib[col])
    ranks["rank_sum"] = ranks.sum(axis=1)
    return pd.concat([table, ranks], axis=1).sort_values("rank_sum")
