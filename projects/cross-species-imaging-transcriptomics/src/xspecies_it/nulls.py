"""Null models for region-level brain maps: spatial surrogates and gene-ensemble nulls.

``variogram_surrogates`` re-implements the variogram-matching idea of Burt et al. (2020,
NeuroImage; BrainSMASH) for arbitrary point sets in 3-D (e.g. CCF region centroids), so the
same null can be applied to mouse subcortex + cortex where spin tests are undefined.
``gene_ensemble_null`` follows Fulcher, Arnatkeviciute & Fornito (2021, Nat Commun): the
statistic of a gene set is compared against random sets matched on a nuisance
property (mean expression level and/or spatial autocorrelation bin).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


# ------------------------------------------------------------------ spatial autocorrelation
def morans_i(x: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    """Moran's I with a k-nearest-neighbour binary weight matrix."""
    x = np.asarray(x, float)
    D = squareform(pdist(np.asarray(coords, float)))
    np.fill_diagonal(D, np.inf)
    W = np.zeros_like(D)
    idx = np.argsort(D, axis=1)[:, :k]
    for i, nb in enumerate(idx):
        W[i, nb] = 1.0
    W = np.maximum(W, W.T)
    z = x - x.mean()
    num = (W * np.outer(z, z)).sum()
    den = (z ** 2).sum()
    return float(len(x) / W.sum() * num / den)


def empirical_variogram(x: np.ndarray, D: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Semivariance gamma(h) = 0.5 * mean[(x_i - x_j)^2] for pairs in each distance bin."""
    iu = np.triu_indices_from(D, k=1)
    d = D[iu]
    diff2 = 0.5 * (x[iu[0]] - x[iu[1]]) ** 2
    which = np.digitize(d, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = which == b
        if m.any():
            out[b] = diff2[m].mean()
    return out


def variogram_surrogates(
    x: np.ndarray,
    coords: np.ndarray,
    n_surr: int = 1000,
    kernel_scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    n_bins: int = 25,
    max_dist_frac: float = 0.5,
    seed: int = 0,
    return_fit: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Generate surrogate maps that preserve the empirical variogram of ``x``.

    Algorithm (Burt et al., 2020, simplified): permute ``x``; smooth the permuted map with
    a Gaussian distance kernel at several scales; for each scale, linearly regress the
    target variogram on the surrogate's variogram (gamma_target ~ a + b * gamma_surr) and
    keep the scale with the smallest residual; return ``sqrt(b) * smoothed + noise`` with
    the intercept realised as white noise so the surrogate's variogram matches the target's
    in both nugget and range.

    Parameters
    ----------
    x : (n,) map values.
    coords : (n, d) coordinates (mm).
    kernel_scales : Gaussian kernel widths as multiples of the median nearest-neighbour distance.
    return_fit : also return the variogram-match correlation per surrogate.

    Returns
    -------
    surrogates : (n_surr, n) array.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    coords = np.asarray(coords, float)
    n = len(x)
    D = squareform(pdist(coords))
    D_off = D.copy()
    np.fill_diagonal(D_off, np.inf)
    nn = D_off.min(axis=1)
    d0 = float(np.median(nn))
    if not np.isfinite(d0) or d0 <= 0:
        raise ValueError("degenerate coordinates: zero median nearest-neighbour distance")
    dmax = np.percentile(D[np.triu_indices(n, 1)], 100 * max_dist_frac)
    bins = np.linspace(0, dmax, n_bins + 1)
    gamma_t = empirical_variogram(x, D, bins)
    ok = ~np.isnan(gamma_t)
    kernels = [np.exp(-0.5 * (D / (s * d0)) ** 2) for s in kernel_scales]
    kernels = [K / K.sum(axis=1, keepdims=True) for K in kernels]

    surr = np.empty((n_surr, n))
    fit_r = np.empty(n_surr)
    x_std = x.std()
    for i in range(n_surr):
        perm = rng.permutation(x)
        best = None
        for K in kernels:
            s = K @ perm
            if s.std() < 1e-12:
                continue  # kernel too wide: constant map, cannot match a variogram
            s = (s - s.mean()) / s.std() * x_std
            g = empirical_variogram(s, D, bins)
            m = ok & ~np.isnan(g)
            if m.sum() < 3 or np.nanstd(g[m]) < 1e-12:
                continue
            b, a, r, _, _ = stats.linregress(g[m], gamma_t[m])
            resid = np.mean((gamma_t[m] - (a + b * g[m])) ** 2)
            if best is None or resid < best[0]:
                best = (resid, a, b, s, r)
        assert best is not None
        _, a, b, s, r = best
        b = max(b, 0.0)
        a = max(a, 0.0)
        y = np.sqrt(b) * s + np.sqrt(a) * rng.standard_normal(n)  # nugget as white noise
        y = (y - y.mean()) / (y.std() + 1e-12) * x_std + x.mean()
        surr[i] = y
        fit_r[i] = r
    if return_fit:
        return surr, fit_r
    return surr


def spatial_null_corr(
    x: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    n_surr: int = 1000,
    method: str = "spearman",
    seed: int = 0,
    surrogates: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Correlation between maps ``x`` and ``y`` with a variogram-surrogate p-value (surrogates of ``x``)."""
    corr = stats.spearmanr if method == "spearman" else stats.pearsonr
    r_obs = float(corr(x, y)[0])
    if surrogates is None:
        surrogates = variogram_surrogates(x, coords, n_surr=n_surr, seed=seed)
    r_null = np.array([corr(s, y)[0] for s in surrogates])
    p = (np.sum(np.abs(r_null) >= abs(r_obs)) + 1) / (len(r_null) + 1)
    return {"r": r_obs, "p_spatial": float(p), "null_mean": float(r_null.mean()), "null_sd": float(r_null.std())}


# ------------------------------------------------------------------ gene-ensemble nulls
def gene_ensemble_null(
    gene_scores: np.ndarray,
    gene_set_mask: np.ndarray,
    match_on: Optional[np.ndarray] = None,
    n_perm: int = 5000,
    n_bins: int = 10,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> Dict[str, float]:
    """Compare a gene-set statistic against random gene sets matched on a nuisance variable.

    Parameters
    ----------
    gene_scores : (n_genes,) per-gene association scores (e.g. Spearman rho with a map).
    gene_set_mask : boolean (n_genes,) membership of the gene set of interest.
    match_on : optional (n_genes,) nuisance property (e.g. mean expression or Moran's I).
        Random sets are drawn bin-by-bin so their nuisance distribution matches the set's.
    n_perm : number of random sets.

    Returns
    -------
    dict with observed statistic, null mean/sd, two-sided p, z-score.
    """
    rng = np.random.default_rng(seed)
    gene_scores = np.asarray(gene_scores, float)
    mask = np.asarray(gene_set_mask, bool)
    k = int(mask.sum())
    if k == 0:
        raise ValueError("empty gene set")
    obs = float(statistic(gene_scores[mask]))
    idx_all = np.arange(len(gene_scores))
    if match_on is None:
        draws = [rng.choice(idx_all, size=k, replace=False) for _ in range(n_perm)]
    else:
        match_on = np.asarray(match_on, float)
        edges = np.quantile(match_on, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        bin_of = np.clip(np.digitize(match_on, edges) - 1, 0, n_bins - 1)
        need = np.bincount(bin_of[mask], minlength=n_bins)
        pools = [idx_all[bin_of == b] for b in range(n_bins)]
        draws = []
        for _ in range(n_perm):
            d = []
            for b in range(n_bins):
                if need[b] == 0:
                    continue
                pool = pools[b]
                d.append(rng.choice(pool, size=min(need[b], len(pool)), replace=len(pool) < need[b]))
            draws.append(np.concatenate(d))
    null = np.array([statistic(gene_scores[d]) for d in draws])
    p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (n_perm + 1)
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return {"stat": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()), "p_ensemble": float(p), "z": float(z)}


def panel_aware_enrichment(
    gene_scores: np.ndarray,
    gene_set_mask: np.ndarray,
    panel_mask: np.ndarray,
    match_on: Optional[np.ndarray] = None,
    n_perm: int = 5000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Enrichment of a gene set computed with (a) genome-wide and (b) panel-restricted backgrounds.

    With a targeted panel (e.g. the 500-gene MERFISH panel chosen to discriminate cell
    types), using all genes as background inflates enrichment for any cell-type-related
    set. The panel-aware version restricts both the set and the random draws to the panel.
    Returns both results so the inflation can be reported.
    """
    panel_mask = np.asarray(panel_mask, bool)
    genome = gene_ensemble_null(gene_scores, gene_set_mask, match_on, n_perm=n_perm, seed=seed)
    ps = gene_scores[panel_mask]
    pm = np.asarray(gene_set_mask, bool)[panel_mask]
    mo = None if match_on is None else np.asarray(match_on)[panel_mask]
    panel = gene_ensemble_null(ps, pm, mo, n_perm=n_perm, seed=seed) if pm.any() else {"p_ensemble": np.nan}
    return {"genome_background": genome, "panel_background": panel}


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out
