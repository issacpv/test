"""scDRS-like per-cell polygenic scoring ("lite").

Follows the logic of scDRS (Zhang et al. 2022, Nat Genet) in simplified form:

1. Putative disease genes = top-N genes by MAGMA gene Z; weights = Z (capped).
2. Raw score per cell = weighted sum of per-gene standardised expression.
3. Control scores from `n_ctrl` random gene sets matched to the disease genes on mean expression
   (and, optionally, variance) bins, with the *same* weights.
4. Normalised score = (raw - mean_ctrl) / sd_ctrl per cell; Monte-Carlo p per cell.
5. Group (cell-type / domain) association: mean normalised score of the group compared with the
   distribution of the same statistic over control sets.

It is written for dense or sparse (cells x genes) matrices of moderate size; for atlas scale,
score chunks of cells with `score_cells` using pre-computed gene statistics (`gene_stats`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class GeneStats:
    mean: np.ndarray
    var: np.ndarray
    names: np.ndarray

    def index(self, genes: Sequence[str]) -> np.ndarray:
        pos = {g: i for i, g in enumerate(self.names)}
        return np.array([pos[g] for g in genes if g in pos], dtype=int)


def gene_stats(X, gene_names: Sequence[str]) -> GeneStats:
    """Per-gene mean and variance over cells, without densifying sparse input."""
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq = X.multiply(X).mean(axis=0)
        var = np.asarray(sq).ravel() - mean ** 2
    else:
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=0)
        var = X.var(axis=0)
    return GeneStats(mean=mean, var=np.clip(var, 1e-12, None), names=np.asarray(gene_names))


def gene_weights_from_z(gene_z: pd.Series, n_genes: int = 1000, z_cap: float = 10.0) -> Dict[str, float]:
    """Top-`n_genes` genes by MAGMA Z, weight = min(Z, z_cap), negative Z clipped to 0."""
    z = gene_z.dropna().sort_values(ascending=False).head(n_genes)
    return {g: float(min(max(v, 0.0), z_cap)) for g, v in z.items()}


def _match_control_sets(gs: GeneStats, target_idx: np.ndarray, n_ctrl: int, n_bins: int,
                        rng: np.random.Generator, match_var: bool = False) -> np.ndarray:
    """Return (n_ctrl, len(target_idx)) indices of control genes matched on expression bins."""
    n_genes = len(gs.names)
    mean_bin = pd.qcut(pd.Series(gs.mean).rank(method="first"), n_bins, labels=False).to_numpy()
    if match_var:
        var_bin = pd.qcut(pd.Series(gs.var).rank(method="first"), max(2, n_bins // 4), labels=False).to_numpy()
        joint = mean_bin * 1000 + var_bin
    else:
        joint = mean_bin
    pools: Dict[int, np.ndarray] = {}
    for b in np.unique(joint):
        pools[b] = np.flatnonzero(joint == b)
    target_set = set(target_idx.tolist())
    ctrl = np.empty((n_ctrl, len(target_idx)), dtype=int)
    for k in range(n_ctrl):
        for j, g in enumerate(target_idx):
            pool = pools[joint[g]]
            for _ in range(10):
                cand = rng.choice(pool)
                if cand not in target_set:
                    break
            ctrl[k, j] = cand
    return ctrl


def _weighted_sum(X, cols: np.ndarray, w: np.ndarray, gs: GeneStats) -> np.ndarray:
    """Sum_g w_g * (x_g - mean_g)/sd_g over selected columns, for every cell."""
    sd = np.sqrt(gs.var[cols])
    Xs = X[:, cols]
    Xs = Xs.toarray() if sparse.issparse(Xs) else np.asarray(Xs, dtype=float)
    return ((Xs - gs.mean[cols]) / sd) @ w


def score_cells(X, gene_names: Sequence[str], weights: Mapping[str, float], n_ctrl: int = 100,
                n_bins: int = 20, seed: int = 0, stats: Optional[GeneStats] = None,
                match_var: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    """Per-cell disease scores with matched control sets.

    Returns
    -------
    scores : DataFrame (n_cells) with columns raw_score, norm_score, mc_pval
    ctrl_norm : (n_cells, n_ctrl) normalised control scores (needed for group association)
    """
    rng = np.random.default_rng(seed)
    gs = stats if stats is not None else gene_stats(X, gene_names)
    genes = [g for g in weights if g in set(gs.names)]
    if len(genes) < 10:
        raise ValueError("Fewer than 10 weighted genes present in the expression matrix")
    idx = gs.index(genes)
    w = np.array([weights[g] for g in genes], dtype=float)
    w = w / w.sum()

    raw = _weighted_sum(X, idx, w, gs)
    ctrl_idx = _match_control_sets(gs, idx, n_ctrl, n_bins, rng, match_var)
    ctrl_raw = np.stack([_weighted_sum(X, ctrl_idx[k], w, gs) for k in range(n_ctrl)], axis=1)

    # scDRS-style: first centre/scale each control set to the raw score distribution, then per-cell z
    mu_c = ctrl_raw.mean(axis=0, keepdims=True)
    sd_c = ctrl_raw.std(axis=0, keepdims=True) + 1e-12
    ctrl_adj = (ctrl_raw - mu_c) / sd_c * raw.std() + raw.mean()
    cell_mu = ctrl_adj.mean(axis=1)
    cell_sd = ctrl_adj.std(axis=1) + 1e-12
    norm = (raw - cell_mu) / cell_sd
    ctrl_norm = (ctrl_adj - cell_mu[:, None]) / cell_sd[:, None]
    mc_p = (1 + (ctrl_norm >= norm[:, None]).sum(axis=1)) / (1 + n_ctrl)

    scores = pd.DataFrame({"raw_score": raw, "norm_score": norm, "mc_pval": mc_p})
    return scores, ctrl_norm


def group_association(norm_score: np.ndarray, ctrl_norm: np.ndarray, groups: Sequence,
                      min_cells: int = 20) -> pd.DataFrame:
    """Group-level association: observed mean normalised score vs control means (MC p) and z.

    Also reports the fraction of cells in the group with mc_pval-equivalent norm_score above the
    control 95th percentile ("frac_sig"), a heterogeneity-free proxy for scDRS's per-group stats.
    """
    g = pd.Categorical(np.asarray(groups))
    rows = []
    for lab in g.categories:
        sel = np.flatnonzero(g == lab)
        if len(sel) < min_cells:
            continue
        obs = norm_score[sel].mean()
        ctrl = ctrl_norm[sel].mean(axis=0)
        z = (obs - ctrl.mean()) / (ctrl.std() + 1e-12)
        p = (1 + (ctrl >= obs).sum()) / (1 + len(ctrl))
        thr = np.quantile(ctrl_norm[sel], 0.95)
        rows.append({"group": lab, "n_cells": len(sel), "mean_norm_score": obs, "assoc_z": z,
                     "assoc_mcp": p, "frac_sig": float((norm_score[sel] > thr).mean())})
    return pd.DataFrame(rows).set_index("group").sort_values("assoc_z", ascending=False)
