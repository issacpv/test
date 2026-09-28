"""Associations between EEG-age deviation and outcomes with permutation nulls."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def jonckheere_terpstra(x: np.ndarray, groups: np.ndarray, n_perm: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Jonckheere-Terpstra trend statistic for ordered groups (e.g. HIE grade), two-sided permutation p.

    JT = sum over group pairs (i < j) of the Mann-Whitney U counting x_j > x_i; a
    standardized version is returned together with its permutation p-value.
    """
    rng = np.random.default_rng(seed)
    x, g = np.asarray(x, float), np.asarray(groups)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    levels = np.unique(g)
    if levels.size < 2:
        return {"jt": float("nan"), "p": float("nan"), "n_groups": int(levels.size)}

    def _jt(xx: np.ndarray, gg: np.ndarray) -> float:
        s = 0.0
        for i in range(levels.size):
            for j in range(i + 1, levels.size):
                a, b = xx[gg == levels[i]], xx[gg == levels[j]]
                if a.size and b.size:
                    s += np.sum(b[:, None] > a[None, :]) + 0.5 * np.sum(b[:, None] == a[None, :])
        return float(s)

    obs = _jt(x, g)
    null = np.array([_jt(rng.permutation(x), g) for _ in range(n_perm)])
    mu, sd = null.mean(), null.std() + 1e-12
    p = float((np.sum(np.abs(null - mu) >= abs(obs - mu)) + 1) / (n_perm + 1))
    return {"jt": obs, "z": float((obs - mu) / sd), "p": p, "n_groups": int(levels.size)}


def spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return {"rho": float("nan"), "p": float("nan"), "n": int(x.size)}
    obs = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    return {"rho": float(obs), "p": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)), "n": int(x.size)}


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray, n_perm: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Spearman correlation of rank-residuals after regressing both variables on covariates."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    C = np.asarray(covariates, float)
    if C.ndim == 1:
        C = C[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(C).all(1)
    A = np.column_stack([np.ones(ok.sum()), C[ok]])
    rx = stats.rankdata(x[ok])
    ry = stats.rankdata(y[ok])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return spearman_perm(ex, ey, n_perm, seed)


def auroc_binary(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    score, y = np.asarray(score, float), np.asarray(y, int)
    ok = np.isfinite(score)
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok]))


def seizure_burden(events: Sequence[Tuple[float, float]], duration_s: float) -> float:
    """Seizure minutes per hour of recording from (onset_s, offset_s) events (overlaps merged)."""
    if duration_s <= 0:
        return float("nan")
    ev = sorted((float(a), float(b)) for a, b in events if b > a)
    merged = []
    for a, b in ev:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    total = sum(b - a for a, b in merged)
    return float(total / 60.0 / (duration_s / 3600.0))


def consensus_events(per_annotator: Sequence[Sequence[Tuple[float, float]]], duration_s: float, fs_ann: float = 1.0,
                     min_agree: int = None) -> Sequence[Tuple[float, float]]:
    """Intersection (default: all annotators) of per-second seizure annotations -> event list."""
    n = int(np.ceil(duration_s * fs_ann))
    votes = np.zeros(n, int)
    for evs in per_annotator:
        m = np.zeros(n, bool)
        for a, b in evs:
            m[int(a * fs_ann):int(np.ceil(b * fs_ann))] = True
        votes += m
    need = len(per_annotator) if min_agree is None else min_agree
    mask = np.r_[False, votes >= need, False].astype(int)
    d = np.diff(mask)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s / fs_ann, e / fs_ann) for s, e in zip(starts, ends)]


def bootstrap_ci(values: np.ndarray, stat=np.median, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    b = np.array([stat(rng.choice(v, v.size, replace=True)) for _ in range(n_boot)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
