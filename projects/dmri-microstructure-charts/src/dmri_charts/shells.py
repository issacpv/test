"""Gradient tables, shell identification and protocol emulation.

Protocol emulation subsamples an HCP-style multi-shell acquisition (b = 1000/2000/3000, 90
directions each, 18 b = 0) to schemes such as Cam-CAN (1000 x 30 + 2000 x 60), UK Biobank
(1000 x 50 + 2000 x 50) or HCP-Aging (1500/3000; only approximable by the nearest shells).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def read_bvals_bvecs(bval_path: str | Path, bvec_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read FSL-style ``bvals`` (1 x N) and ``bvecs`` (3 x N or N x 3) files; returns ``(N,), (N, 3)``."""
    bvals = np.loadtxt(bval_path).ravel()
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    if bvecs.shape != (len(bvals), 3):
        raise ValueError(f"bvecs shape {bvecs.shape} does not match {len(bvals)} bvals")
    norms = np.linalg.norm(bvecs, axis=1)
    nz = norms > 0
    bvecs[nz] = bvecs[nz] / norms[nz, None]
    return bvals.astype(float), bvecs.astype(float)


def identify_shells(bvals: np.ndarray, tol: float = 100.0, b0_threshold: float = 50.0) -> Dict[int, np.ndarray]:
    """Cluster b-values into shells; keys are shell b-values rounded to 50, values index arrays.

    b = 0 volumes are returned under key ``0``.
    """
    bvals = np.asarray(bvals, float)
    shells: Dict[int, List[int]] = {}
    centres: List[float] = []
    for i, b in enumerate(bvals):
        if b <= b0_threshold:
            shells.setdefault(0, []).append(i)
            continue
        match = next((c for c in centres if abs(b - c) <= tol), None)
        if match is None:
            centres.append(b)
            match = b
        key = int(round(match / 50.0) * 50)
        shells.setdefault(key, []).append(i)
    return {k: np.asarray(v, int) for k, v in shells.items()}


def _spread_subset(bvecs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point subset of ``n`` directions (antipodally symmetric distance)."""
    m = len(bvecs)
    if n >= m:
        return np.arange(m)
    chosen = [int(rng.integers(m))]
    d = np.abs(bvecs @ bvecs.T)  # cosine similarity magnitude
    while len(chosen) < n:
        sim = d[:, chosen].max(axis=1)
        sim[chosen] = np.inf
        chosen.append(int(np.argmin(sim)))
    return np.asarray(sorted(chosen))


def subsample_protocol(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    scheme: Dict[int, int],
    n_b0: Optional[int] = None,
    tol: float = 100.0,
    seed: int = 0,
) -> np.ndarray:
    """Indices emulating a target protocol.

    ``scheme`` maps shell b-value -> number of directions to keep (``None`` keeps all). Shells
    absent from the data raise ``KeyError`` (an emulation of b = 1500 from 1000/2000/3000 data is
    not possible; use the nearest available shell explicitly). Directions are chosen by greedy
    farthest-point sampling so that the subset stays well spread on the sphere.
    """
    rng = np.random.default_rng(seed)
    shells = identify_shells(bvals, tol=tol)
    keep: List[int] = []
    b0 = shells.get(0, np.array([], int))
    keep += list(b0 if n_b0 is None else b0[: n_b0])
    for b, n in scheme.items():
        if b not in shells:
            raise KeyError(f"shell b={b} not present; available: {sorted(shells)}")
        idx = shells[b]
        if n is None or n >= len(idx):
            keep += list(idx)
        else:
            sub = _spread_subset(np.asarray(bvecs)[idx], n, rng)
            keep += list(idx[sub])
    return np.asarray(sorted(keep), int)


def protocol_summary(bvals: np.ndarray, tol: float = 100.0) -> Dict[str, object]:
    shells = identify_shells(bvals, tol=tol)
    return {
        "n_volumes": int(len(bvals)),
        "n_b0": int(len(shells.get(0, []))),
        "shells": {int(k): int(len(v)) for k, v in shells.items() if k != 0},
        "max_b": int(max([k for k in shells if k != 0], default=0)),
        "multishell": int(len([k for k in shells if k != 0]) >= 2),
    }


# common target schemes (directions per shell), for convenience
SCHEMES: Dict[str, Dict[int, int]] = {
    "hcp_ya_full": {1000: 90, 2000: 90, 3000: 90},
    "camcan_like": {1000: 30, 2000: 60},
    "ukb_like": {1000: 50, 2000: 50},
    "single_shell_b1000_30": {1000: 30},
    "single_shell_b1000_15": {1000: 15},
}
