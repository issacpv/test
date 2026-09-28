"""Population sweeps, probabilistic VTA radius and factorial variance decomposition.

The deliverable of the project is a long table with one row per (morphology, diameter, distance,
orientation draw, conductor model, pulse) and its threshold. Everything here consumes that
table; :func:`threshold_sweep` produces it for small designs (large designs are run with the
same function in parallel chunks).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .axon_model import Pulse, build_axon_on_path, threshold_current
from .lead_field import LeadField
from .morph_paths import place_path, straight_fiber


def threshold_sweep(field: LeadField, paths: Dict[str, np.ndarray], diameters_um: Sequence[float],
                    distances_mm: Sequence[float], n_orient: int = 1, pulse: Pulse = Pulse(), z_mm: Optional[float] = None,
                    scale: float = 1.0, seed: int = 0, hi: float = 20.0, rotate: bool = True) -> pd.DataFrame:
    """Thresholds for every (morphology, diameter, distance, orientation) combination.

    ``paths`` maps morphology id -> path (mm). Straight-fibre controls can be added by the caller
    (e.g. ``{"straight_parallel": straight_fiber(...)}`` with ``rotate=False``).
    """
    rng = np.random.default_rng(seed)
    z = field.contact_centre()[2] if z_mm is None else z_mm
    rows: List[Dict[str, float]] = []
    for mid, path in paths.items():
        for D in diameters_um:
            for r in distances_mm:
                for k in range(n_orient):
                    placed = place_path(path, r, z, rng=rng, scale=scale, rotate=rotate)
                    axon = build_axon_on_path(placed, D)
                    ve = field.unit_potential(axon.nodes)
                    th = threshold_current(axon, ve, pulse, hi=hi)
                    rows.append({"morphology": mid, "diameter_um": D, "distance_mm": r, "orientation": k,
                                 "threshold": th, "n_nodes": axon.n})
    return pd.DataFrame(rows)


def activation_probability_curve(sweep: pd.DataFrame, amplitude: float, by: str = "distance_mm") -> pd.DataFrame:
    """P(threshold <= amplitude) per distance (over all other factors), with Wilson 95% CI."""
    g = sweep.groupby(by)["threshold"]
    n = g.size()
    k = g.apply(lambda s: float(np.sum(s <= amplitude)))
    p = k / n
    zc = 1.96
    denom = 1 + zc ** 2 / n
    centre = (p + zc ** 2 / (2 * n)) / denom
    half = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / denom
    return pd.DataFrame({by: n.index, "p_active": p.values, "ci_lo": (centre - half).values,
                         "ci_hi": (centre + half).values, "n": n.values})


def probabilistic_vta_radius(sweep: pd.DataFrame, amplitude: float, probs: Sequence[float] = (0.05, 0.5, 0.95),
                             by: str = "distance_mm") -> Dict[str, float]:
    """Radii at which the activation probability crosses given levels (linear interpolation).

    Also returns the per-draw radius quantiles: for each (morphology, diameter, orientation) the
    largest distance at which threshold <= amplitude, whose distribution gives the spread of
    individual VTA radii.
    """
    curve = activation_probability_curve(sweep, amplitude, by).sort_values(by)
    r, p = curve[by].values, curve["p_active"].values
    out: Dict[str, float] = {}
    for q in probs:
        # activation probability decreases with distance; find first crossing below q
        idx = np.where(p < q)[0]
        if len(idx) == 0:
            out[f"r_p{int(q * 100)}"] = float(r[-1])
        elif idx[0] == 0:
            out[f"r_p{int(q * 100)}"] = 0.0
        else:
            i = idx[0]
            out[f"r_p{int(q * 100)}"] = float(np.interp(q, [p[i], p[i - 1]], [r[i], r[i - 1]]))
    keys = [c for c in ("morphology", "diameter_um", "orientation") if c in sweep.columns]
    per = sweep[sweep["threshold"] <= amplitude].groupby(keys)[by].max() if keys else pd.Series(dtype=float)
    if len(per):
        out["draw_r_q05"], out["draw_r_q50"], out["draw_r_q95"] = (float(x) for x in np.percentile(per.values, [5, 50, 95]))
    return out


def efield_isoline_radius(field: LeadField, amplitude: float, threshold_V_per_mm: float = 0.2,
                          r_max: float = 10.0, dr: float = 0.02, z_mm: Optional[float] = None,
                          azimuth_rad: float = 0.0) -> float:
    """Distance from the lead axis at which |E| falls below the Lead-DBS style threshold (0.2 V/mm)."""
    z = field.contact_centre()[2] if z_mm is None else z_mm
    rs = np.arange(field.spec.radius_mm + dr, r_max, dr)
    pts = np.column_stack([rs * np.cos(azimuth_rad), rs * np.sin(azimuth_rad), np.full_like(rs, z)])
    e = field.efield_magnitude(pts, amplitude)
    below = np.where(e < threshold_V_per_mm)[0]
    return float(rs[below[0]]) if len(below) else float(r_max)


def variance_decomposition(df: pd.DataFrame, value: str, factors: Sequence[str], log: bool = True,
                           n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """ANOVA-style first-order (main-effect) and pairwise-interaction variance fractions.

    For a (near-)balanced factorial design the main-effect fraction of factor F,
    ``Var(E[y | F]) / Var(y)``, equals the first-order Sobol index. ``residual`` is what the
    listed main effects and pairwise interactions do not explain. Bootstrap resamples rows.
    """
    d = df[[value] + list(factors)].replace([np.inf, -np.inf], np.nan).dropna()
    y = np.log(d[value].values) if log else d[value].values
    d = d.assign(_y=y)
    tot = float(np.var(y))

    def _fracs(dd: pd.DataFrame) -> Dict[str, float]:
        t = float(np.var(dd["_y"]))
        if t <= 0:
            return {f: np.nan for f in factors}
        out = {}
        for f in factors:
            out[f] = float(np.var(dd.groupby(f)["_y"].transform("mean"))) / t
        for f1, f2 in itertools.combinations(factors, 2):
            joint = float(np.var(dd.groupby([f1, f2])["_y"].transform("mean"))) / t
            out[f"{f1}x{f2}"] = max(joint - out[f1] - out[f2], 0.0)
        out["residual"] = max(1.0 - sum(v for k, v in out.items()), 0.0)
        return out

    point = _fracs(d)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        bs = d.sample(n=len(d), replace=True, random_state=int(rng.integers(0, 2 ** 31 - 1)))
        for k, v in _fracs(bs).items():
            boots[k].append(v)
    rows = []
    for k, v in point.items():
        b = np.array(boots[k], dtype=float)
        lo, hi = (np.nanpercentile(b, [2.5, 97.5]) if n_boot else (np.nan, np.nan))
        rows.append({"term": k, "fraction": v, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    res.attrs["total_variance"] = tot
    return res


def lognormal_vs_normal(thresholds: np.ndarray) -> Dict[str, float]:
    """AIC comparison of log-normal vs normal fits and the coefficient of variation (H1)."""
    from scipy import stats

    x = np.asarray(thresholds, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    mu, sd = x.mean(), x.std(ddof=1)
    ll_n = stats.norm.logpdf(x, mu, sd).sum()
    lmu, lsd = np.log(x).mean(), np.log(x).std(ddof=1)
    ll_ln = stats.lognorm.logpdf(x, s=lsd, scale=np.exp(lmu)).sum()
    return {"cv": float(sd / mu), "aic_normal": float(4 - 2 * ll_n), "aic_lognormal": float(4 - 2 * ll_ln),
            "skew": float(stats.skew(x)), "n": int(len(x))}
