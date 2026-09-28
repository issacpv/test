"""Classical twin-design statistics: ICCs, Falconer, ACE maximum likelihood, twin identification, power.

Inputs are pair matrices ``mz`` and ``dz`` of shape ``(n_pairs, 2)`` holding one (covariate
adjusted, standardised) phenotype per twin. Ordering within a pair is arbitrary, so ICCs use
double entry and the ACE likelihood is symmetric.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats


# --------------------------------------------------------------------------- correlations
def twin_icc(pairs: np.ndarray) -> float:
    """Double-entry intraclass correlation of a (n_pairs, 2) matrix."""
    p = np.asarray(pairs, dtype=float)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 3:
        return float("nan")
    x = np.concatenate([p[:, 0], p[:, 1]])
    y = np.concatenate([p[:, 1], p[:, 0]])
    return float(np.corrcoef(x, y)[0, 1])


def falconer(r_mz: float, r_dz: float) -> Dict[str, float]:
    """Falconer's estimates: h2 = 2(rMZ - rDZ), c2 = 2rDZ - rMZ, e2 = 1 - rMZ (unconstrained)."""
    return {"h2": 2 * (r_mz - r_dz), "c2": 2 * r_dz - r_mz, "e2": 1 - r_mz}


# --------------------------------------------------------------------------- ACE ML
def _pair_loglik(pairs: np.ndarray, mu: float, var: float, r: float) -> float:
    p = np.asarray(pairs, dtype=float)
    if len(p) == 0:
        return 0.0
    cov = var * np.array([[1.0, r], [r, 1.0]])
    try:
        return float(stats.multivariate_normal(mean=[mu, mu], cov=cov, allow_singular=False).logpdf(p).sum())
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf


def _ace_components(theta: np.ndarray, model: str) -> Tuple[float, float, float]:
    a2 = theta[0] ** 2 if "A" in model else 0.0
    c2 = theta[1] ** 2 if "C" in model else 0.0
    e2 = theta[2] ** 2
    return a2, c2, e2


def fit_ace(mz: np.ndarray, dz: np.ndarray, model: str = "ACE") -> Dict[str, float]:
    """Maximum-likelihood ACE-family fit (``model`` in {"ACE", "AE", "CE", "E"}).

    Parameterisation: standard deviations a, c, e (squared to keep variances non-negative), a common
    mean. Returns standardised h2, c2, e2, total variance, log-likelihood, number of free parameters.
    """
    if model not in ("ACE", "AE", "CE", "E"):
        raise ValueError("model must be ACE, AE, CE or E")
    mz = np.asarray(mz, dtype=float)
    dz = np.asarray(dz, dtype=float)
    allv = np.concatenate([mz.ravel(), dz.ravel()])
    mu0, sd0 = float(np.nanmean(allv)), float(np.nanstd(allv))

    def nll(x: np.ndarray) -> float:
        mu = x[3]
        a2, c2, e2 = _ace_components(x[:3], model)
        var = a2 + c2 + e2
        if var <= 1e-10:
            return 1e12
        r_mz = (a2 + c2) / var
        r_dz = (0.5 * a2 + c2) / var
        return -(_pair_loglik(mz, mu, var, r_mz) + _pair_loglik(dz, mu, var, r_dz))

    x0 = np.array([sd0 * 0.6, sd0 * 0.4, sd0 * 0.6, mu0])
    best = None
    for start in (x0, x0 * np.array([1.2, 0.5, 0.8, 1.0]), x0 * np.array([0.5, 1.2, 1.0, 1.0])):
        res = optimize.minimize(nll, start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    a2, c2, e2 = _ace_components(best.x[:3], model)
    var = a2 + c2 + e2
    k = 1 + ("A" in model) + ("C" in model) + 1  # mean + free variance components
    return {"h2": a2 / var, "c2": c2 / var, "e2": e2 / var, "var": var, "mean": float(best.x[3]),
            "loglik": float(-best.fun), "n_params": k, "model": model, "converged": bool(best.success)}


def ace_model_comparison(mz: np.ndarray, dz: np.ndarray) -> Dict[str, object]:
    """Fit ACE, AE, CE, E; LRTs of dropping A (ACE vs CE) and C (ACE vs AE); AIC per model."""
    fits = {m: fit_ace(mz, dz, m) for m in ("ACE", "AE", "CE", "E")}
    ll = {m: f["loglik"] for m, f in fits.items()}

    def lrt(full: str, reduced: str) -> Dict[str, float]:
        stat = max(0.0, 2 * (ll[full] - ll[reduced]))
        df = fits[full]["n_params"] - fits[reduced]["n_params"]
        # variance components are bounded at zero: the null distribution is a 50:50 mixture of chi2(0) and chi2(df)
        p = 0.5 * stats.chi2.sf(stat, df) if df == 1 else stats.chi2.sf(stat, df)
        return {"stat": float(stat), "df": int(df), "p": float(p)}

    aic = {m: 2 * f["n_params"] - 2 * f["loglik"] for m, f in fits.items()}
    return {"fits": fits, "lrt_drop_A": lrt("ACE", "CE"), "lrt_drop_C": lrt("ACE", "AE"), "aic": aic,
            "best_by_aic": min(aic, key=aic.get)}


def ace_bootstrap(mz: np.ndarray, dz: np.ndarray, n_boot: int = 500, model: str = "ACE", seed: int = 0,
                  alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """Pair-level bootstrap percentile CIs for h2, c2, e2."""
    rng = np.random.default_rng(seed)
    mz = np.asarray(mz, dtype=float)
    dz = np.asarray(dz, dtype=float)
    out = {"h2": [], "c2": [], "e2": []}
    for _ in range(n_boot):
        bm = mz[rng.integers(0, len(mz), len(mz))]
        bd = dz[rng.integers(0, len(dz), len(dz))]
        f = fit_ace(bm, bd, model)
        for k in out:
            out[k].append(f[k])
    return {k: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))) for k, v in out.items()}


def permutation_mz_dz_difference(mz: np.ndarray, dz: np.ndarray, n_perm: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Permutation test of rMZ - rDZ by shuffling zygosity labels across pairs."""
    rng = np.random.default_rng(seed)
    mz = np.asarray(mz, dtype=float)
    dz = np.asarray(dz, dtype=float)
    obs = twin_icc(mz) - twin_icc(dz)
    allp = np.vstack([mz, dz])
    n_mz = len(mz)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(allp))
        null[i] = twin_icc(allp[perm[:n_mz]]) - twin_icc(allp[perm[n_mz:]])
    return {"observed": float(obs), "p_perm": float((1 + np.sum(null >= obs)) / (n_perm + 1))}


def reliability_ceiling(h2: float, icc_retest: float) -> Dict[str, float]:
    """Heritability is bounded by reliability; report the disattenuated estimate h2 / ICC."""
    return {"h2": h2, "ceiling": icc_retest, "h2_disattenuated": h2 / icc_retest if icc_retest > 0 else np.nan}


# --------------------------------------------------------------------------- fingerprinting
def twin_identification(features: np.ndarray, pairs: Sequence[Tuple[int, int]], n_perm: int = 1000, seed: int = 0
                        ) -> Dict[str, float]:
    """Nearest-neighbour co-twin identification from feature vectors (rows = subjects).

    For each twin, the co-twin is "identified" when it is the most correlated other subject.
    Accuracy is compared with a permutation null that re-pairs subjects at random.
    """
    rng = np.random.default_rng(seed)
    F = np.asarray(features, dtype=float)
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-12)
    C = np.corrcoef(F)
    np.fill_diagonal(C, -np.inf)
    nn = np.argmax(C, axis=1)

    def acc(prs: Sequence[Tuple[int, int]]) -> float:
        hits = [nn[a] == b for a, b in prs] + [nn[b] == a for a, b in prs]
        return float(np.mean(hits))

    obs = acc(pairs)
    members = np.array([i for p in pairs for i in p])
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(members)
        null[i] = acc(list(zip(perm[0::2], perm[1::2])))
    return {"accuracy": obs, "chance": 1.0 / (len(F) - 1), "null_mean": float(null.mean()),
            "p_perm": float((1 + np.sum(null >= obs)) / (n_perm + 1))}


# --------------------------------------------------------------------------- power
def power_simulation(n_mz: int, n_dz: int, h2: float, c2: float = 0.0, n_sim: int = 200, alpha: float = 0.05,
                     reliability: float = 1.0, seed: int = 0) -> Dict[str, float]:
    """Power of the LRT for A > 0 at given pair counts, true h2/c2 and measurement reliability.

    Measurement error (1 - reliability) is added as extra E variance, which attenuates the
    observed correlations exactly as in real data.
    """
    from .synthetic import simulate_twin_phenotypes

    rng = np.random.default_rng(seed)
    rejections = 0
    for i in range(n_sim):
        mz, dz = simulate_twin_phenotypes(n_mz, n_dz, h2 * reliability, c2 * reliability, seed=int(rng.integers(1 << 31)))
        cmp_ = ace_model_comparison(mz, dz)
        rejections += int(cmp_["lrt_drop_A"]["p"] < alpha)
    return {"power": rejections / n_sim, "n_mz": n_mz, "n_dz": n_dz, "h2": h2, "c2": c2, "reliability": reliability}


__all__ = ["twin_icc", "falconer", "fit_ace", "ace_model_comparison", "ace_bootstrap", "permutation_mz_dz_difference",
           "reliability_ceiling", "twin_identification", "power_simulation"]
