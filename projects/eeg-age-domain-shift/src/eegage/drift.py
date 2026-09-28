"""Drift diagnostics between age bins and age-aware reweighting.

* :func:`mmd_rbf` - unbiased kernel MMD^2 with a median-heuristic RBF kernel and a
  permutation p-value (works on handcrafted features or foundation-model embeddings).
* :func:`centroid_distance` - cosine distance between bin centroids.
* :func:`age_probe_r2` - subject-grouped ridge probe: how much age an embedding encodes.
* :func:`age_importance_weights` - density-ratio weights so that a training set's age
  histogram matches a target's (the reweighting comparator to normative scaling).
* :func:`drift_vs_loss` - Spearman correlation between drift and transfer loss.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def _rbf_gram(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def mmd_rbf(Xa: np.ndarray, Xb: np.ndarray, gamma: float | None = None, n_perm: int = 200,
            seed: int = 0, max_n: int = 800) -> Tuple[float, float]:
    """Unbiased MMD^2 (RBF, median heuristic) and permutation p-value; subsamples to ``max_n`` per group."""
    rng = np.random.default_rng(seed)
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    if len(Xa) > max_n:
        Xa = Xa[rng.choice(len(Xa), max_n, replace=False)]
    if len(Xb) > max_n:
        Xb = Xb[rng.choice(len(Xb), max_n, replace=False)]
    Z = np.vstack([Xa, Xb])
    if gamma is None:
        sub = Z[rng.choice(len(Z), size=min(500, len(Z)), replace=False)]
        d2 = (sub ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None, :] - 2 * sub @ sub.T
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / med

    def stat(A: np.ndarray, B: np.ndarray) -> float:
        m, n = len(A), len(B)
        Kaa, Kbb, Kab = _rbf_gram(A, A, gamma), _rbf_gram(B, B, gamma), _rbf_gram(A, B, gamma)
        t1 = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1)) if m > 1 else 0.0
        t2 = (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1)) if n > 1 else 0.0
        return float(t1 + t2 - 2 * Kab.mean())

    obs = stat(Xa, Xb)
    m = len(Xa)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(Z))
        null[i] = stat(Z[perm[:m]], Z[perm[m:]])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


def centroid_distance(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Cosine distance between the mean vectors of two clouds (after global centering)."""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    mu = np.vstack([Xa, Xb]).mean(0)
    a, b = Xa.mean(0) - mu, Xb.mean(0) - mu
    na, nb = np.linalg.norm(a) + 1e-12, np.linalg.norm(b) + 1e-12
    return float(1.0 - (a @ b) / (na * nb))


def age_probe_r2(E: np.ndarray, ages: np.ndarray, groups: np.ndarray, n_splits: int = 5, alpha: float = 1.0) -> float:
    """Out-of-fold R^2 of a ridge regression predicting log(age+1) from embeddings, grouped by subject."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    E, ages, groups = np.asarray(E, float), np.asarray(ages, float), np.asarray(groups)
    t = np.log1p(ages)
    k = min(n_splits, len(np.unique(groups)))
    pred = np.zeros_like(t)
    for tr, te in GroupKFold(n_splits=k).split(E, t, groups):
        sc = StandardScaler().fit(E[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(E[tr]), t[tr])
        pred[te] = m.predict(sc.transform(E[te]))
    ss_res = ((t - pred) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def age_importance_weights(ages_src: np.ndarray, ages_tgt: np.ndarray, edges: Sequence[float] | None = None,
                           clip: float = 20.0) -> np.ndarray:
    """Per-source-sample weights ~ p_target(age)/p_source(age) from histograms in log-age (mean 1)."""
    a_s, a_t = np.log1p(np.asarray(ages_src, float)), np.log1p(np.asarray(ages_tgt, float))
    if edges is None:
        edges = np.linspace(min(a_s.min(), a_t.min()) - 1e-9, max(a_s.max(), a_t.max()) + 1e-9, 12)
    hs, _ = np.histogram(a_s, bins=edges)
    ht, _ = np.histogram(a_t, bins=edges)
    ps = (hs + 0.5) / (hs + 0.5).sum()
    pt = (ht + 0.5) / (ht + 0.5).sum()
    ratio = np.clip(pt / ps, 1.0 / clip, clip)
    idx = np.clip(np.digitize(a_s, edges) - 1, 0, len(ratio) - 1)
    w = ratio[idx]
    return w / w.mean()


def drift_vs_loss(drift: Sequence[float], loss: Sequence[float]) -> Dict[str, float]:
    """Spearman correlation between drift statistics and transfer losses across (train, test) cells."""
    d, l = np.asarray(drift, float), np.asarray(loss, float)
    ok = np.isfinite(d) & np.isfinite(l)
    if ok.sum() < 4:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    rho, p = stats.spearmanr(d[ok], l[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


def transfer_loss(M, diagonal_ref: bool = True):
    """Loss matrix: diag(train) - M[train, test] (how much is lost by leaving the training bin)."""
    import pandas as pd

    M = pd.DataFrame(M)
    ref = np.diag(M.values) if diagonal_ref else M.max(axis=1).values
    return pd.DataFrame(ref[:, None] - M.values, index=M.index, columns=M.columns)
