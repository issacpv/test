"""Reproducibility statistics for laminar profiles.

* :func:`icc_2_1` per depth bin.
* :func:`variance_components` – lab / subject-within-lab / residual variance (statsmodels MixedLM,
  with a method-of-moments fallback for balanced designs).
* :func:`reproducibility_index` – chance-corrected mean pairwise concordance of normalized profiles.
* :func:`snr_degradation_curve` – classification accuracy of a profile shape as thermal noise is added,
  emulating lower field strength / resolution from real high-SNR data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import profiles as pf


def icc_2_1(data: np.ndarray) -> float:
    Y = np.asarray(data, float)
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * np.sum((Y.mean(1) - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((Y.mean(0) - grand) ** 2) / (k - 1)
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + grand
    ms_e = np.sum(resid**2) / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def icc_per_bin(profiles_by_session: np.ndarray) -> np.ndarray:
    """``profiles_by_session`` = (n_subjects, n_sessions, n_bins) → ICC(2,1) per bin."""
    P = np.asarray(profiles_by_session, float)
    return np.array([icc_2_1(P[:, :, b]) for b in range(P.shape[2])])


@dataclass
class VarianceComponents:
    lab: float
    subject: float
    residual: float

    @property
    def total(self) -> float:
        return self.lab + self.subject + self.residual

    def fractions(self) -> dict[str, float]:
        t = self.total
        return {"lab": self.lab / t, "subject": self.subject / t, "residual": self.residual / t}


def _moments_nested(y: np.ndarray, lab: np.ndarray, subject: np.ndarray) -> VarianceComponents:
    """Method-of-moments nested ANOVA (approximate for unbalanced designs)."""
    labs = np.unique(lab)
    grand = y.mean()
    ss_lab = ss_sub = ss_res = 0.0
    n_sub_total = 0
    n_per_lab = []
    n_per_sub = []
    for L in labs:
        ml = lab == L
        yl = y[ml]
        ss_lab += yl.size * (yl.mean() - grand) ** 2
        n_per_lab.append(yl.size)
        for S in np.unique(subject[ml]):
            ms = ml & (subject == S)
            ys = y[ms]
            ss_sub += ys.size * (ys.mean() - yl.mean()) ** 2
            ss_res += np.sum((ys - ys.mean()) ** 2)
            n_sub_total += 1
            n_per_sub.append(ys.size)
    N = y.size
    df_lab, df_sub, df_res = len(labs) - 1, n_sub_total - len(labs), N - n_sub_total
    ms_lab = ss_lab / max(df_lab, 1)
    ms_sub = ss_sub / max(df_sub, 1)
    ms_res = ss_res / max(df_res, 1)
    n0 = np.mean(n_per_sub)     # average replicates per subject
    m0 = np.mean(n_per_lab)     # average observations per lab
    var_res = ms_res
    var_sub = max((ms_sub - ms_res) / n0, 0.0)
    var_lab = max((ms_lab - ms_sub) / m0, 0.0)
    return VarianceComponents(float(var_lab), float(var_sub), float(var_res))


def variance_components(y: np.ndarray, lab: np.ndarray, subject: np.ndarray, method: str = "auto") -> VarianceComponents:
    """Nested random-effects decomposition of one profile summary (e.g. superficial-deep contrast).

    ``subject`` labels must be unique across labs (or are made unique internally).
    ``method``: 'mixedlm' (REML via statsmodels), 'moments', or 'auto' (mixedlm with fallback).
    """
    y = np.asarray(y, float)
    lab = np.asarray(lab).astype(str)
    subject = np.char.add(np.char.add(lab, "/"), np.asarray(subject).astype(str))
    if method in ("auto", "mixedlm"):
        try:
            import pandas as pd
            import statsmodels.formula.api as smf

            df = pd.DataFrame({"y": y, "lab": lab, "subject": subject})
            model = smf.mixedlm("y ~ 1", df, groups="lab", re_formula="1", vc_formula={"subject": "0 + C(subject)"})
            fit = model.fit(reml=True, method=["lbfgs"], maxiter=500)
            var_lab = float(fit.cov_re.iloc[0, 0])
            var_sub = float(fit.vcomp[0]) if len(fit.vcomp) else 0.0
            var_res = float(fit.scale)
            if np.all(np.isfinite([var_lab, var_sub, var_res])):
                return VarianceComponents(max(var_lab, 0.0), max(var_sub, 0.0), var_res)
        except Exception:  # pragma: no cover - fall back below
            if method == "mixedlm":
                raise
    return _moments_nested(y, lab, subject)


def mean_pairwise(profiles_: np.ndarray, metric=pf.concordance_ccc) -> float:
    P = np.asarray(profiles_, float)
    n = P.shape[0]
    vals = [metric(P[i], P[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else float("nan")


def reproducibility_index(profiles_: np.ndarray, n_null: int = 200, rng: np.random.Generator | None = None,
                          normalize: str = "zscore") -> dict[str, float]:
    """Chance-corrected reproducibility of a set of profiles (rows = independent acquisitions).

    index = (observed - null) / (1 - null), where null is the mean pairwise CCC after shuffling the
    depth order independently within each profile (destroys shape, keeps the value distribution).
    """
    rng = np.random.default_rng() if rng is None else rng
    P = np.stack([pf.normalize_profile(p, normalize) for p in np.asarray(profiles_, float)])
    obs = mean_pairwise(P)
    null = np.empty(n_null)
    for i in range(n_null):
        Q = np.stack([p[rng.permutation(p.size)] for p in P])
        null[i] = mean_pairwise(Q)
    mu = float(null.mean())
    idx = (obs - mu) / (1 - mu) if mu < 1 else float("nan")
    p = float((1 + np.sum(null >= obs)) / (1 + n_null))
    return {"observed_ccc": obs, "null_mean": mu, "index": float(idx), "p_value": p}


def degrade_profile(true_profile: np.ndarray, n_voxels_per_bin: int, tsnr: float, contrast_pct: float,
                    n_timepoints: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a measured profile at a given voxel tSNR.

    Each bin has ``n_voxels_per_bin`` voxels whose contrast estimate has SE ≈ (100/tsnr)/√(n_timepoints/4)
    in % signal units (a block design with half the time in each condition); voxel estimates are averaged.
    """
    se_voxel = (100.0 / tsnr) / np.sqrt(n_timepoints / 4.0)
    p = np.asarray(true_profile, float) * contrast_pct
    noise = se_voxel * rng.standard_normal((n_voxels_per_bin, p.size)) / np.sqrt(1.0)
    return (p[None, :] + noise).mean(axis=0)


def snr_degradation_curve(true_profile: np.ndarray, tsnr_levels: np.ndarray, classifier, n_rep: int = 200,
                          n_voxels_per_bin: int = 50, contrast_pct: float = 2.0, n_timepoints: int = 300,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Accuracy of ``classifier(profile) -> bool`` against ``classifier(true_profile)`` at each tSNR."""
    rng = np.random.default_rng() if rng is None else rng
    target = classifier(np.asarray(true_profile, float))
    acc = np.empty(len(tsnr_levels))
    for i, s in enumerate(tsnr_levels):
        hits = 0
        for _ in range(n_rep):
            meas = degrade_profile(true_profile, n_voxels_per_bin, s, contrast_pct, n_timepoints, rng)
            hits += classifier(meas) == target
        acc[i] = hits / n_rep
    return acc


def tsnr_for_accuracy(tsnr_levels: np.ndarray, accuracy: np.ndarray, target: float = 0.8) -> float:
    """Smallest tSNR at which accuracy ≥ target (linear interpolation on log-tSNR); NaN if never reached."""
    t, a = np.asarray(tsnr_levels, float), np.asarray(accuracy, float)
    order = np.argsort(t)
    t, a = t[order], a[order]
    above = np.flatnonzero(a >= target)
    if above.size == 0:
        return float("nan")
    k = above[0]
    if k == 0:
        return float(t[0])
    x0, x1, y0, y1 = np.log(t[k - 1]), np.log(t[k]), a[k - 1], a[k]
    return float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))
