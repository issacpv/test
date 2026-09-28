"""Structural connectome construction: the 'multiverse' factors as pure functions of a matrix."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Construction:
    """One point of the connectome multiverse."""

    weight: str = "ncd"            # 'ncd' (normalised connection density) or 'ncs' (strength)
    hemispheres: str = "both"      # 'ipsi', 'both'
    symmetrise: str = "none"       # 'none', 'mean', 'max'
    log: bool = False
    density: Optional[float] = None  # keep the strongest fraction of edges (0-1); None = no threshold
    row_normalise: bool = False

    def name(self) -> str:
        return f"{self.weight}|{self.hemispheres}|sym={self.symmetrise}|log={int(self.log)}|dens={self.density}|rn={int(self.row_normalise)}"


def load_npz_connectome(path: Path | str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_csv_connectome(path: Path | str) -> pd.DataFrame:
    """Square CSV with region acronyms as index and columns (e.g. MouseLight-derived)."""
    df = pd.read_csv(path, index_col=0)
    return df.loc[df.index, df.index] if set(df.index) <= set(df.columns) else df


def bilateral(ipsi: np.ndarray, contra: np.ndarray) -> np.ndarray:
    """Build a 2N x 2N matrix [[ipsi, contra], [contra, ipsi]] from hemisphere-specific blocks."""
    return np.block([[ipsi, contra], [contra, ipsi]])


def apply(W: np.ndarray, c: Construction) -> np.ndarray:
    """Apply a construction to a raw (non-negative, possibly NaN) weight matrix."""
    W = np.array(W, float)
    W[~np.isfinite(W)] = 0.0
    np.fill_diagonal(W, 0.0)
    if c.log:
        W = np.log1p(W / max(W[W > 0].min(), 1e-12)) if (W > 0).any() else W
    if c.symmetrise == "mean":
        W = 0.5 * (W + W.T)
    elif c.symmetrise == "max":
        W = np.maximum(W, W.T)
    if c.density is not None:
        W = threshold_density(W, c.density)
    if c.row_normalise:
        s = W.sum(axis=1, keepdims=True)
        W = np.divide(W, s, out=np.zeros_like(W), where=s > 0)
    return W


def threshold_density(W: np.ndarray, density: float) -> np.ndarray:
    """Keep the strongest ``density`` fraction of off-diagonal edges (others set to 0).

    Exact top-k by rank (ties broken by position, so the result never overshoots by
    more than one edge). A symmetric input stays symmetric: pairs are ranked on the
    upper triangle and mirrored, so the kept count is rounded up to a whole pair.
    """
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    n = len(W)
    out = np.zeros_like(W, dtype=float)
    if np.allclose(W, W.T):
        iu = np.triu_indices(n, k=1)
        vals = W[iu]
        k = int(np.ceil(density * vals.size))
        keep = np.argsort(-vals, kind="stable")[:k]
        keep = keep[vals[keep] > 0]
        out[iu[0][keep], iu[1][keep]] = vals[keep]
        out = out + out.T
    else:
        off = np.where(~np.eye(n, dtype=bool))
        vals = W[off]
        k = int(np.ceil(density * vals.size))
        keep = np.argsort(-vals, kind="stable")[:k]
        keep = keep[vals[keep] > 0]
        out[off[0][keep], off[1][keep]] = vals[keep]
    return out


def edge_density(W: np.ndarray) -> float:
    off = ~np.eye(len(W), dtype=bool)
    return float((W[off] > 0).mean())


def spectral_normalise(W: np.ndarray) -> np.ndarray:
    """Scale so the largest eigenvalue magnitude is 1 (keeps couplings comparable across constructions)."""
    ev = np.abs(np.linalg.eigvals(W)).max()
    return W / ev if ev > 0 else W


def graph_summary(W: np.ndarray) -> Dict[str, float]:
    off = ~np.eye(len(W), dtype=bool)
    return {"n": int(len(W)), "density": edge_density(W), "mean_weight": float(W[off].mean()),
            "asymmetry": float(np.abs(W - W.T).sum() / max(np.abs(W).sum(), 1e-12)),
            "spectral_radius": float(np.abs(np.linalg.eigvals(W)).max())}


def homotopic_pairs(n_per_hemisphere: int) -> np.ndarray:
    """Index pairs (i, i + N) for a bilateral matrix built with :func:`bilateral`."""
    return np.column_stack([np.arange(n_per_hemisphere), np.arange(n_per_hemisphere) + n_per_hemisphere])


def multiverse(weights: Sequence[str] = ("ncd", "ncs"), hemis: Sequence[str] = ("ipsi", "both"),
               syms: Sequence[str] = ("none", "mean"), logs: Sequence[bool] = (False, True),
               densities: Sequence[Optional[float]] = (None, 0.2)) -> list:
    return [Construction(w, h, s, l, d) for w in weights for h in hemis for s in syms for l in logs for d in densities]
