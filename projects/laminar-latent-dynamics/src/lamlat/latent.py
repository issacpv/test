"""Population latent structure per layer: FA dimensionality, subspace overlap, reduced-rank regression, nulls.

All functions take (n_time_bins, n_units) count/rate matrices. Cross-validation is blocked in time to respect
autocorrelation.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import orth, subspace_angles
from sklearn.decomposition import FactorAnalysis


def spike_count_matrix(spike_times: Sequence[np.ndarray], bin_edges: np.ndarray, sqrt_transform: bool = True) -> np.ndarray:
    """(n_bins, n_units) counts; optional square-root variance stabilisation."""
    M = np.column_stack([np.histogram(np.asarray(t, dtype=float), bins=bin_edges)[0] for t in spike_times]).astype(float)
    return np.sqrt(M) if sqrt_transform else M


def blocked_folds(n: int, n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous-block K-fold indices (train, test)."""
    idx = np.arange(n)
    blocks = np.array_split(idx, n_folds)
    return [(np.concatenate([b for j, b in enumerate(blocks) if j != k]), blocks[k]) for k in range(n_folds)]


def fa_cv_dimensionality(X: np.ndarray, dims: Iterable[int] = range(1, 11), n_folds: int = 5) -> pd.DataFrame:
    """Cross-validated FA log-likelihood per latent dimensionality (Williamson et al., 2016 style)."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0)
    rows = []
    for d in dims:
        if d >= X.shape[1]:
            break
        lls = []
        for tr, te in blocked_folds(len(X), n_folds):
            fa = FactorAnalysis(n_components=d, max_iter=500).fit(X[tr])
            lls.append(fa.score(X[te]))
        rows.append({"dim": int(d), "cv_loglik": float(np.mean(lls))})
    df = pd.DataFrame(rows)
    df.attrs["best_dim"] = int(df.loc[df["cv_loglik"].idxmax(), "dim"]) if len(df) else 0
    return df


def fa_subspace(X: np.ndarray, n_components: int) -> Dict[str, np.ndarray]:
    """Fit FA and return the orthonormalised loading subspace plus shared/private variance per unit."""
    X = np.asarray(X, dtype=float)
    fa = FactorAnalysis(n_components=n_components, max_iter=500).fit(X)
    L = fa.components_.T                     # (n_units, k)
    shared = np.sum(L ** 2, axis=1)
    private = fa.noise_variance_
    return {"basis": orth(L), "loadings": L, "shared_var": shared, "private_var": private,
            "percent_shared": shared / (shared + private + 1e-12)}


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spaces of A and B."""
    return subspace_angles(np.asarray(A, dtype=float), np.asarray(B, dtype=float))


def subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared cosine of the principal angles (1 = identical, 0 = orthogonal)."""
    return float(np.mean(np.cos(principal_angles(A, B)) ** 2))


def cross_layer_overlap(X_a: np.ndarray, X_b: np.ndarray, k: int) -> float:
    """Overlap between the top-k FA subspaces of two populations *in a shared coordinate system*.

    Populations have different units, so their loading subspaces live in different spaces. We therefore compare
    the *temporal* latent trajectories: FA scores (n_time, k) of each population, and compute the overlap of the
    column spaces of the two score matrices (how much of the time course of A's latents is spanned by B's).
    """
    fa_a = FactorAnalysis(n_components=k, max_iter=500).fit(np.asarray(X_a, dtype=float))
    fa_b = FactorAnalysis(n_components=k, max_iter=500).fit(np.asarray(X_b, dtype=float))
    Za = fa_a.transform(np.asarray(X_a, dtype=float))
    Zb = fa_b.transform(np.asarray(X_b, dtype=float))
    return subspace_overlap(Za - Za.mean(axis=0), Zb - Zb.mean(axis=0))


def reduced_rank_regression(X: np.ndarray, Y: np.ndarray, rank: int, ridge: float = 1e-6) -> np.ndarray:
    """RRR coefficient matrix B (p x q) of rank ``rank`` minimising ||Y - X B||^2 (with tiny ridge)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    B_ols = np.linalg.solve(XtX, X.T @ Y)
    Yhat = X @ B_ols
    # project onto the top-r right singular vectors of the fitted values
    _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
    V = Vt[:rank].T
    return B_ols @ V @ V.T


def rrr_cv_r2(X: np.ndarray, Y: np.ndarray, ranks: Iterable[int], n_folds: int = 5, ridge: float = 1e-3) -> pd.DataFrame:
    """Blocked-CV predictive R^2 (pooled over target units) as a function of RRR rank; rank 0 = full OLS."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    rows = []
    ranks = list(ranks)
    for r in ranks:
        ss_res = ss_tot = 0.0
        for tr, te in blocked_folds(len(X), n_folds):
            mx, my = X[tr].mean(axis=0), Y[tr].mean(axis=0)
            Xtr, Ytr = X[tr] - mx, Y[tr] - my
            Xte, Yte = X[te] - mx, Y[te] - my
            if r == 0:
                B = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(X.shape[1]), Xtr.T @ Ytr)
            else:
                B = reduced_rank_regression(Xtr, Ytr, r, ridge)
            pred = Xte @ B
            ss_res += float(np.sum((Yte - pred) ** 2))
            ss_tot += float(np.sum(Yte ** 2))
        rows.append({"rank": int(r), "cv_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan})
    return pd.DataFrame(rows)


def communication_dimensionality(cv_table: pd.DataFrame, fraction: float = 0.95) -> int:
    """Smallest rank whose CV R^2 reaches ``fraction`` of the best R^2 in the table (Semedo et al., 2019)."""
    best = cv_table["cv_r2"].max()
    ok = cv_table[(cv_table["rank"] > 0) & (cv_table["cv_r2"] >= fraction * best)]
    return int(ok["rank"].min()) if len(ok) else int(cv_table["rank"].max())


# ------------------------------------------------------------------------------------------------------
# Nulls and matching
# ------------------------------------------------------------------------------------------------------

def matched_indices(labels: np.ndarray, groups: Sequence[str], n: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """Equal-size random subsets of unit indices per group (n = smallest group size by default)."""
    rng = np.random.default_rng(0) if rng is None else rng
    labels = np.asarray(labels)
    idx = {g: np.flatnonzero(labels == g) for g in groups}
    n = min(len(v) for v in idx.values()) if n is None else n
    return {g: rng.choice(v, size=n, replace=False) for g, v in idx.items()}


def depth_shuffle_null(X: np.ndarray, labels: np.ndarray, group_a: str, group_b: str,
                       stat_fn: Callable[[np.ndarray, np.ndarray], float], n_perm: int = 200,
                       n_match: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> Dict[str, object]:
    """Observed statistic vs a null in which layer labels are permuted across units (unit counts preserved).

    ``stat_fn(X_a, X_b)`` takes the two (n_time, n_units) sub-matrices. Unit-count matching is applied in both the
    observed and null computations so that the comparison is fair."""
    rng = np.random.default_rng(0) if rng is None else rng
    labels = np.asarray(labels)
    m = matched_indices(labels, [group_a, group_b], n_match, rng)
    obs = float(stat_fn(X[:, m[group_a]], X[:, m[group_b]]))
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        mp = matched_indices(perm, [group_a, group_b], len(m[group_a]), rng)
        null[i] = stat_fn(X[:, mp[group_a]], X[:, mp[group_b]])
    z = (obs - null.mean()) / (null.std(ddof=1) + 1e-12)
    p = float((np.sum(null <= obs) + 1) / (n_perm + 1))  # one-sided: is observed overlap LOWER than null?
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "z": float(z),
            "p_lower": p, "null": null}


def split_unit_ceiling(X: np.ndarray, k: int, n_rep: int = 20, rng: Optional[np.random.Generator] = None) -> float:
    """Overlap between the FA subspaces of two random halves of the *same* population: the ceiling for
    between-layer overlap given finite sampling."""
    rng = np.random.default_rng(0) if rng is None else rng
    n_units = X.shape[1]
    vals = []
    for _ in range(n_rep):
        perm = rng.permutation(n_units)
        h = n_units // 2
        vals.append(cross_layer_overlap(X[:, perm[:h]], X[:, perm[h:2 * h]], k))
    return float(np.mean(vals))


def synthetic_two_populations(n_time: int, n_a: int, n_b: int, k_shared: int, k_private: int,
                              rng: np.random.Generator, shared_gain_b: float = 1.0, noise: float = 0.5
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two populations driven by ``k_shared`` common latents plus ``k_private`` private latents each."""
    Z = rng.normal(size=(n_time, k_shared))
    Pa = rng.normal(size=(n_time, k_private))
    Pb = rng.normal(size=(n_time, k_private))
    La, Lb = rng.normal(size=(k_shared, n_a)), rng.normal(size=(k_shared, n_b))
    Qa, Qb = rng.normal(size=(k_private, n_a)), rng.normal(size=(k_private, n_b))
    Xa = Z @ La + Pa @ Qa + noise * rng.normal(size=(n_time, n_a))
    Xb = shared_gain_b * (Z @ Lb) + Pb @ Qb + noise * rng.normal(size=(n_time, n_b))
    return Xa, Xb, Z
