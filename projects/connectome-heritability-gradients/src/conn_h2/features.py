"""Region-wise connectome features whose heritability is mapped.

* **SC-FC coupling** (Gu et al., 2021, Nat Commun; Baum et al., 2020, PNAS):
  per-region correlation between the structural and functional connectivity
  profiles, optionally over structurally connected pairs only.
* **Multilinear communication model** (Vázquez-Rodríguez et al., 2019, PNAS):
  regional adjusted R^2 of FC predicted from SC, shortest-path length and
  communicability.
* **Dynamic FC states**: sliding-window FC, k-means states across subjects,
  per-subject fractional occupancy, dwell time and transition probabilities
  (Vidaurre et al., 2017, PNAS; Jun et al., 2022, NeuroImage).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import expm
from scipy.sparse.csgraph import shortest_path


# --------------------------------------------------------------------------- #
# SC-FC coupling
# --------------------------------------------------------------------------- #
def sc_fc_coupling(
    sc: np.ndarray, fc: np.ndarray, method: str = "spearman", log_sc: bool = True, mask_nonzero: bool = True
) -> np.ndarray:
    """Per-region correlation of SC and FC connectivity profiles.

    Parameters
    ----------
    sc
        ``(N, N)`` structural connectivity (streamline counts or densities).
    fc
        ``(N, N)`` functional connectivity (Fisher-z).
    method
        ``"spearman"`` or ``"pearson"``.
    log_sc
        Apply ``log1p`` to SC (streamline counts are heavy-tailed).
    mask_nonzero
        Correlate only over pairs with non-zero SC (as in Baum et al., 2020).

    Returns
    -------
    ``(N,)`` coupling values (NaN when fewer than 3 valid pairs).
    """
    sc = np.asarray(sc, dtype=float)
    fc = np.asarray(fc, dtype=float)
    n = sc.shape[0]
    if log_sc:
        sc = np.log1p(np.clip(sc, 0, None))
    out = np.full(n, np.nan)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        if mask_nonzero:
            keep &= sc[i] > 0
        if keep.sum() < 3:
            continue
        x, y = sc[i, keep], fc[i, keep]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        out[i] = stats.spearmanr(x, y).correlation if method == "spearman" else stats.pearsonr(x, y)[0]
    return out


def communicability(sc: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Weighted communicability ``expm(D^-1/2 A D^-1/2)`` (Crofts & Higham, 2009)."""
    A = np.asarray(sc, dtype=float)
    if normalize:
        d = A.sum(axis=1)
        d[d == 0] = 1.0
        d_half = d ** (-0.5)
        A = A * d_half[:, None] * d_half[None, :]
    C = expm(A)
    np.fill_diagonal(C, 0.0)
    return C


def shortest_path_lengths(sc: np.ndarray, transform: str = "inverse") -> np.ndarray:
    """Weighted shortest-path lengths with edge length = 1/w (or -log w)."""
    A = np.asarray(sc, dtype=float)
    with np.errstate(divide="ignore"):
        if transform == "inverse":
            L = np.where(A > 0, 1.0 / A, 0.0)
        elif transform == "log":
            Amax = A.max() if A.max() > 0 else 1.0
            L = np.where(A > 0, -np.log(A / (Amax * 1.0001)), 0.0)
        else:
            raise ValueError("transform must be 'inverse' or 'log'")
    D = shortest_path(L, method="D", directed=False)
    D[~np.isfinite(D)] = np.nan
    return D


def multilinear_coupling(sc: np.ndarray, fc: np.ndarray, predictors: Sequence[str] = ("sc", "spl", "comm")) -> np.ndarray:
    """Regional adjusted R^2 of FC profile regressed on communication predictors.

    Predictors (z-scored per region): ``sc`` = log SC, ``spl`` = shortest-path
    length, ``comm`` = communicability. Rows with NaN predictors are dropped.
    """
    sc = np.asarray(sc, dtype=float)
    fc = np.asarray(fc, dtype=float)
    n = sc.shape[0]
    mats: Dict[str, np.ndarray] = {}
    if "sc" in predictors:
        mats["sc"] = np.log1p(np.clip(sc, 0, None))
    if "spl" in predictors:
        mats["spl"] = shortest_path_lengths(sc)
    if "comm" in predictors:
        mats["comm"] = communicability(sc)
    r2 = np.full(n, np.nan)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        cols = []
        for name in predictors:
            v = mats[name][i]
            keep &= np.isfinite(v)
            cols.append(v)
        X = np.column_stack([c[keep] for c in cols])
        y = fc[i, keep]
        p = X.shape[1]
        m = keep.sum()
        if m <= p + 2:
            continue
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
        X1 = np.column_stack([np.ones(m), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ beta
        sst = ((y - y.mean()) ** 2).sum()
        if sst == 0:
            continue
        r2_raw = 1 - (resid**2).sum() / sst
        r2[i] = 1 - (1 - r2_raw) * (m - 1) / (m - p - 1)
    return r2


# --------------------------------------------------------------------------- #
# Dynamic FC states
# --------------------------------------------------------------------------- #
def sliding_window_fc(ts: np.ndarray, window: int = 60, step: int = 10) -> np.ndarray:
    """Vectorised upper-triangular Pearson FC per window -> ``(n_windows, n_edges)``."""
    ts = np.asarray(ts, dtype=float)
    T, N = ts.shape
    if window > T:
        raise ValueError("window longer than time series")
    iu = np.triu_indices(N, k=1)
    out = []
    for start in range(0, T - window + 1, step):
        seg = ts[start : start + window]
        c = np.corrcoef(seg, rowvar=False)
        c = np.nan_to_num(c)
        out.append(c[iu])
    return np.vstack(out)


def cluster_states(
    windows_per_subject: Sequence[np.ndarray], n_states: int = 4, seed: int = 0, n_init: int = 5
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """k-means clustering of windowed FC pooled over subjects.

    Returns ``(centroids (n_states, n_edges), labels_per_subject)``.
    """
    from sklearn.cluster import KMeans

    pooled = np.vstack(windows_per_subject)
    km = KMeans(n_clusters=n_states, n_init=n_init, random_state=seed).fit(pooled)
    labels, start = [], 0
    for w in windows_per_subject:
        labels.append(km.labels_[start : start + len(w)])
        start += len(w)
    return km.cluster_centers_, labels


def state_metrics(labels: np.ndarray, n_states: int) -> Dict[str, np.ndarray]:
    """Fractional occupancy, mean dwell time (windows) and transition matrix."""
    labels = np.asarray(labels, dtype=int)
    fo = np.bincount(labels, minlength=n_states) / len(labels)
    dwell = np.zeros(n_states)
    counts = np.zeros(n_states)
    run_len, cur = 1, labels[0]
    for lab in labels[1:]:
        if lab == cur:
            run_len += 1
        else:
            dwell[cur] += run_len
            counts[cur] += 1
            cur, run_len = lab, 1
    dwell[cur] += run_len
    counts[cur] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dwell = np.where(counts > 0, dwell / counts, 0.0)
    T = np.zeros((n_states, n_states))
    for a, b in zip(labels[:-1], labels[1:]):
        T[a, b] += 1
    row = T.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return {"fractional_occupancy": fo, "mean_dwell": mean_dwell, "transition_matrix": T / row}


def dynamic_state_features(
    ts_list: Sequence[np.ndarray], window: int = 60, step: int = 10, n_states: int = 4, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end: time series -> per-subject FO ``(S, K)``, dwell ``(S, K)``,
    flattened transition probabilities ``(S, K*K)``."""
    wins = [sliding_window_fc(ts, window, step) for ts in ts_list]
    _, labels = cluster_states(wins, n_states=n_states, seed=seed)
    fo, dw, tr = [], [], []
    for lab in labels:
        m = state_metrics(lab, n_states)
        fo.append(m["fractional_occupancy"])
        dw.append(m["mean_dwell"])
        tr.append(m["transition_matrix"].ravel())
    return np.vstack(fo), np.vstack(dw), np.vstack(tr)
