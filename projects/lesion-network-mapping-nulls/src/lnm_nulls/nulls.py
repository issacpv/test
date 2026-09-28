"""Null families for lesion network mapping and spatial surrogates.

N1  label_permutation_null      – permute symptom scores across patients (conditions on lesions).
N2  synthetic_lesion_null       – random spheres (volume-matched) with the observed scores.
N3  (see lesion_sampling.shuffle_lesion_locations)  – real lesions translated at random.
N4  matched_resampling_null     – lesions resampled from a pool matched on volume/hemisphere/territory.
N5  connectome_bootstrap        – map uncertainty from resampled connectome subjects.
N6  moran_spectral_surrogates   – spatial-autocorrelation-preserving surrogates of a parcel map.

Also: bias atlas (expected map under a lesion null) and prior-leakage R².
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lesion_sampling as ls
from . import lnm


@dataclass
class NullResult:
    observed_t: np.ndarray       # (n_parcels,)
    null_max: np.ndarray         # (n_perm,) max |t| per null realisation
    null_t: np.ndarray | None    # (n_perm, n_parcels) optional full null maps
    p_fwer: np.ndarray           # (n_parcels,)
    p_uncorrected: np.ndarray    # (n_parcels,) parcel-wise empirical p

    def significant(self, alpha: float = 0.05) -> np.ndarray:
        return self.p_fwer <= alpha


def _summarise(obs: np.ndarray, null_t: np.ndarray, keep_full: bool) -> NullResult:
    null_abs = np.abs(null_t)
    null_max = np.nanmax(null_abs, axis=1)
    obs_abs = np.abs(obs)
    p_fwer = (1 + np.sum(null_max[:, None] >= obs_abs[None, :], axis=0)) / (1 + null_max.size)
    p_unc = (1 + np.sum(null_abs >= obs_abs[None, :], axis=0)) / (1 + null_max.size)
    return NullResult(obs, null_max, null_t if keep_full else None, p_fwer, p_unc)


def label_permutation_null(maps: np.ndarray, scores: np.ndarray, covariates: np.ndarray | None,
                           n_perm: int, rng: np.random.Generator, strata: np.ndarray | None = None,
                           keep_full: bool = False) -> NullResult:
    """N1: permute scores (within ``strata`` if given) and recompute the regression-LNM t map."""
    obs = lnm.symptom_association(maps, scores, covariates)
    null_t = np.empty((n_perm, maps.shape[1]))
    scores = np.asarray(scores, float)
    for i in range(n_perm):
        if strata is None:
            perm = rng.permutation(scores)
        else:
            perm = scores.copy()
            for s in np.unique(strata):
                idx = np.flatnonzero(strata == s)
                perm[idx] = scores[rng.permutation(idx)]
        null_t[i] = lnm.symptom_association(maps, perm, covariates)
    return _summarise(obs, null_t, keep_full)


def synthetic_lesion_null(lesions: list[np.ndarray], scores: np.ndarray, covariates: np.ndarray | None,
                          fc_vp: np.ndarray, brain_mask: np.ndarray, n_perm: int, rng: np.random.Generator,
                          generator: str = "sphere", keep_full: bool = False,
                          voxel_index: np.ndarray | None = None) -> NullResult:
    """N2/N3: replace lesions by synthetic (``sphere``) or location-shuffled (``shuffle``) ones,
    keep the observed scores, recompute maps and the t map."""
    maps = lnm.lesion_network_maps(lesions, fc_vp, voxel_index)
    obs = lnm.symptom_association(maps, scores, covariates)
    vols = np.array([m.sum() for m in lesions])
    null_t = np.empty((n_perm, maps.shape[1]))
    for i in range(n_perm):
        if generator == "sphere":
            fake = ls.random_sphere_lesions(len(lesions), brain_mask, vols, rng)
        elif generator == "shuffle":
            fake = ls.shuffle_lesion_locations(lesions, brain_mask, rng)
        else:
            raise ValueError(generator)
        fmaps = lnm.lesion_network_maps(fake, fc_vp, voxel_index)
        null_t[i] = lnm.symptom_association(fmaps, scores, covariates)
    return _summarise(obs, null_t, keep_full)


def matched_resampling_null(lesions: list[np.ndarray], scores: np.ndarray, covariates: np.ndarray | None,
                            fc_vp: np.ndarray, pool: list[np.ndarray], pool_features: list[ls.LesionFeatures],
                            target_features: list[ls.LesionFeatures], n_perm: int, rng: np.random.Generator,
                            keep_full: bool = False, voxel_index: np.ndarray | None = None,
                            pool_maps: np.ndarray | None = None) -> NullResult:
    """N4: resample anatomically matched lesions from ``pool`` for every patient, keep the scores."""
    maps = lnm.lesion_network_maps(lesions, fc_vp, voxel_index)
    obs = lnm.symptom_association(maps, scores, covariates)
    if pool_maps is None:
        pool_maps = lnm.lesion_network_maps(pool, fc_vp, voxel_index)
    null_t = np.empty((n_perm, maps.shape[1]))
    for i in range(n_perm):
        picks = ls.matched_resample(pool, pool_features, target_features, rng)
        null_t[i] = lnm.symptom_association(pool_maps[picks], scores, covariates)
    return _summarise(obs, null_t, keep_full)


def connectome_bootstrap(lesions: list[np.ndarray], subject_fc_vp: list[np.ndarray], n_boot: int,
                         rng: np.random.Generator, voxel_index: np.ndarray | None = None) -> np.ndarray:
    """N5: bootstrap over connectome subjects; returns ``(n_boot, n_lesions, n_parcels)`` maps."""
    S = len(subject_fc_vp)
    out = np.empty((n_boot, len(lesions), subject_fc_vp[0].shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, S, S)
        fc = np.mean([subject_fc_vp[i] for i in idx], axis=0)
        out[b] = lnm.lesion_network_maps(lesions, fc, voxel_index)
    return out


def bias_atlas(null_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected map (mean) and SD across null lesion sets, ``null_maps`` = (n_null, n_parcels)."""
    return np.nanmean(null_maps, axis=0), np.nanstd(null_maps, axis=0)


def prior_leakage_r2(observed_map: np.ndarray, bias_map: np.ndarray) -> float:
    """Variance of an observed parcel map explained by the cohort's lesion-prior (bias) map."""
    x, y = np.asarray(bias_map, float), np.asarray(observed_map, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return float(r**2)


# ---------------------------------------------------------------------------
# N6: Moran spectral randomization (Wagner & Dray, 2015) for parcel maps
# ---------------------------------------------------------------------------
def spatial_weights(coords: np.ndarray, kernel: str = "inverse", bandwidth: float | None = None) -> np.ndarray:
    """Symmetric spatial weight matrix from parcel centroids (zero diagonal)."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if kernel == "inverse":
        W = 1.0 / d
    elif kernel == "gaussian":
        bw = np.median(d[np.isfinite(d)]) if bandwidth is None else bandwidth
        W = np.exp(-(d**2) / (2 * bw**2))
    else:
        raise ValueError(kernel)
    W[~np.isfinite(W)] = 0.0
    return W


def morans_i(x: np.ndarray, W: np.ndarray) -> float:
    z = x - x.mean()
    return float((z.size / W.sum()) * (z @ W @ z) / (z @ z))


def moran_spectral_surrogates(x: np.ndarray, W: np.ndarray, n_surr: int, rng: np.random.Generator,
                              procedure: str = "singleton") -> np.ndarray:
    """Surrogate maps with (approximately) the same Moran's I and the same variance as ``x``.

    ``singleton``: random sign flips of the Moran eigenvector coefficients (exactly preserves
    Moran's I). ``pair``: random rotations within random pairs of eigenvectors (Wagner & Dray, 2015).
    Returns ``(n_surr, n)``.
    """
    n = x.size
    H = np.eye(n) - np.ones((n, n)) / n
    M = H @ W @ H
    lam, U = np.linalg.eigh(M)
    keep = np.abs(lam) > 1e-8 * np.abs(lam).max()
    U = U[:, keep]
    z = (x - x.mean()) / x.std()
    a = U.T @ z
    out = np.empty((n_surr, n))
    for s in range(n_surr):
        if procedure == "singleton":
            a_s = a * rng.choice([-1.0, 1.0], size=a.size)
        elif procedure == "pair":
            a_s = a.copy()
            idx = rng.permutation(a.size)
            for i in range(0, a.size - 1, 2):
                j, k = idx[i], idx[i + 1]
                th = rng.uniform(0, 2 * np.pi)
                r = np.hypot(a[j], a[k])
                a_s[j], a_s[k] = r * np.cos(th), r * np.sin(th)
        else:
            raise ValueError(procedure)
        y = U @ a_s
        out[s] = (y - y.mean()) / y.std() * x.std() + x.mean()
    return out


def surrogate_corr_pvalue(x: np.ndarray, y: np.ndarray, W: np.ndarray, n_surr: int,
                          rng: np.random.Generator) -> tuple[float, float]:
    """Spearman correlation of two parcel maps with a Moran-surrogate p-value (two-sided)."""
    from scipy.stats import spearmanr

    r_obs = spearmanr(x, y).statistic
    surr = moran_spectral_surrogates(x, W, n_surr, rng)
    r_null = np.array([spearmanr(s, y).statistic for s in surr])
    p = (1 + np.sum(np.abs(r_null) >= abs(r_obs))) / (1 + n_surr)
    return float(r_obs), float(p)
