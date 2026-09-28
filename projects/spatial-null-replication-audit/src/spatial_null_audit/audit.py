"""Audit driver: p-values under each null for one map pair, and smoothness-matched FPR calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import nulls
from .claims import classify_claim, null_robustness_index

__all__ = ["AuditResult", "null_pvalue", "audit_pair", "calibrated_fpr", "flip_table"]

DEFAULT_NULLS: Sequence[str] = ("naive", "spin", "spin_corrected", "moran", "variogram")


def null_pvalue(r_obs: float, r_null: np.ndarray, two_sided: bool = True) -> float:
    """Permutation p-value with the +1 correction (Phipson & Smyth, 2010)."""
    r_null = np.asarray(r_null, dtype=float)
    if two_sided:
        b = np.sum(np.abs(r_null) >= abs(r_obs))
    else:
        b = np.sum(r_null >= r_obs)
    return float((b + 1) / (r_null.size + 1))


@dataclass
class AuditResult:
    """Everything the audit records for one claim / map pair."""

    r_pearson: float
    r_spearman: float
    n_parcels: int
    p_values: Dict[str, float] = field(default_factory=dict)
    null_sd: Dict[str, float] = field(default_factory=dict)
    sa_deviation: Dict[str, float] = field(default_factory=dict)  # mean |I_surr - I_x| per null
    lambda_a: float = float("nan")
    lambda_b: float = float("nan")
    morans_i_a: float = float("nan")
    morans_i_b: float = float("nan")

    def robustness_index(self, alpha: float = 0.05) -> float:
        return null_robustness_index(self.p_values, alpha=alpha)

    def classify(self, reported_p: float, alpha: float = 0.05) -> str:
        return classify_claim(reported_p, self.p_values, alpha=alpha)

    def as_dict(self) -> Dict[str, object]:
        d = {
            "r_pearson": self.r_pearson,
            "r_spearman": self.r_spearman,
            "n_parcels": self.n_parcels,
            "lambda_a": self.lambda_a,
            "lambda_b": self.lambda_b,
            "morans_i_a": self.morans_i_a,
            "morans_i_b": self.morans_i_b,
            "robustness_index": self.robustness_index(),
        }
        d.update({f"p_{k}": v for k, v in self.p_values.items()})
        d.update({f"sd_{k}": v for k, v in self.null_sd.items()})
        d.update({f"sadev_{k}": v for k, v in self.sa_deviation.items()})
        return d


def _make_surrogates(
    name: str,
    x: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    coords: Optional[np.ndarray],
    dist: Optional[np.ndarray],
    hemi: Optional[np.ndarray],
    sa_tolerance: float,
    mems: Optional[np.ndarray],
) -> np.ndarray:
    if name == "naive":
        return nulls.naive_permutation(x, n_perm, rng)
    if name == "spin":
        if coords is None:
            raise ValueError("spin nulls need coords")
        return nulls.spin_surrogates(x, coords, n_perm, hemi=hemi, rng=rng)
    if name == "spin_corrected":
        if coords is None or dist is None:
            raise ValueError("spin_corrected needs coords and dist")
        return nulls.spin_surrogates(x, coords, n_perm, hemi=hemi, rng=rng, sa_tolerance=sa_tolerance, dist=dist)
    if name == "moran":
        if dist is None:
            raise ValueError("moran needs dist")
        return nulls.moran_surrogates(x, dist, n_perm, rng=rng, mems=mems)
    if name == "variogram":
        if dist is None:
            raise ValueError("variogram needs dist")
        return nulls.variogram_surrogates(x, dist, n_perm, rng=rng)
    raise ValueError(f"unknown null {name!r}")


def audit_pair(
    x: np.ndarray,
    y: np.ndarray,
    coords: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    hemi: Optional[np.ndarray] = None,
    null_names: Iterable[str] = DEFAULT_NULLS,
    n_perm: int = 1000,
    seed: int = 0,
    sa_tolerance: float = 0.05,
    extra_nulls: Optional[Dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]]] = None,
) -> AuditResult:
    """Correlate ``x`` with ``y`` and test it under every requested null (surrogates of ``x``).

    ``extra_nulls`` lets callers plug in surrogate generators (e.g. an eigenstrapping wrapper) with
    the signature ``f(x, n_perm, rng) -> (n_perm, n)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if dist is None and coords is not None:
        dist = nulls.sphere_distance(coords)
    rng = np.random.default_rng(seed)
    r_obs = float(np.corrcoef(x, y)[0, 1])
    res = AuditResult(r_pearson=r_obs, r_spearman=float(stats.spearmanr(x, y).correlation), n_parcels=int(x.size))
    if dist is not None:
        res.lambda_a = nulls.autocorr_length(x, dist)["lambda"]
        res.lambda_b = nulls.autocorr_length(y, dist)["lambda"]
        res.morans_i_a = nulls.morans_i(x, dist)
        res.morans_i_b = nulls.morans_i(y, dist)
    mems = nulls.moran_eigenvectors(dist)[1] if (dist is not None and "moran" in set(null_names)) else None
    generators: Dict[str, Callable[[], np.ndarray]] = {}
    for name in null_names:
        generators[name] = (lambda nm=name: _make_surrogates(nm, x, n_perm, rng, coords, dist, hemi, sa_tolerance, mems))
    for name, f in (extra_nulls or {}).items():
        generators[name] = (lambda ff=f: ff(x, n_perm, rng))
    yc = (y - y.mean()) / y.std()
    for name, gen in generators.items():
        surr = gen()
        sc = (surr - surr.mean(axis=1, keepdims=True)) / np.maximum(surr.std(axis=1, keepdims=True), 1e-12)
        r_null = sc @ yc / x.size
        res.p_values[name] = null_pvalue(r_obs, r_null)
        res.null_sd[name] = float(r_null.std())
        if dist is not None:
            i_x = res.morans_i_a
            sample = surr[: min(50, surr.shape[0])]
            res.sa_deviation[name] = float(np.mean([abs(nulls.morans_i(s, dist) - i_x) for s in sample]))
    return res


def calibrated_fpr(
    dist: np.ndarray,
    coords: Optional[np.ndarray],
    length_scales: Sequence[float],
    n_sims: int = 200,
    n_perm: int = 200,
    null_names: Iterable[str] = ("naive", "spin", "moran", "variogram"),
    alpha: float = 0.05,
    seed: int = 0,
    hemi: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Empirical false-positive rate of each null on pairs of *independent* GRFs.

    For each ``length_scale`` (in units of ``dist``), ``n_sims`` independent pairs are drawn from
    ``exp(-d / length_scale)`` covariance; each null's p-value is computed with ``n_perm`` surrogates.
    Returns one row per (length_scale, null) with ``fpr`` and a Wilson 95% CI.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    null_names = list(null_names)
    for ls in length_scales:
        fields = nulls.gaussian_random_field(dist, ls, rng, n_maps=2 * n_sims)
        rejects = {nm: 0 for nm in null_names}
        for i in range(n_sims):
            res = audit_pair(fields[2 * i], fields[2 * i + 1], coords=coords, dist=dist, hemi=hemi, null_names=null_names, n_perm=n_perm, seed=int(rng.integers(2**31 - 1)))
            for nm in null_names:
                rejects[nm] += int(res.p_values[nm] < alpha)
        for nm in null_names:
            k, n = rejects[nm], n_sims
            lo, hi = _wilson(k, n)
            rows.append({"length_scale": ls, "null": nm, "fpr": k / n, "ci_low": lo, "ci_high": hi, "n_sims": n})
    return pd.DataFrame(rows)


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def flip_table(claims: pd.DataFrame, results: Dict[str, AuditResult], alpha: float = 0.05) -> pd.DataFrame:
    """Join the registry with audit results: reported p, p per null, robustness index, class."""
    rows = []
    for _, c in claims.iterrows():
        r = results.get(c["claim_id"])
        if r is None:
            continue
        row = {"claim_id": c["claim_id"], "reported_r": c["reported_r"], "reported_p": c["reported_p"], "null_method": c["null_method"], "n_parcels": c["n_parcels"]}
        row.update(r.as_dict())
        row["class"] = r.classify(float(c["reported_p"]), alpha=alpha)
        rows.append(row)
    return pd.DataFrame(rows)
