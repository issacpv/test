"""Phylogenetic generalised least squares (PGLS) helpers.

* ``parse_newick`` / ``newick_to_covariance``: Brownian-motion covariance from a Newick tree with
  branch lengths (e.g. exported from TimeTree in Myr): ``C_ij`` = shared path length from the root.
* ``divergence_times_to_covariance``: same from a table of pairwise divergence times.
* ``pgls``: GLS regression with Pagel's lambda estimated by maximum likelihood on a grid, with
  optional measurement-error variances added to the diagonal (species means with SEs).
* ``phylogenetic_signal``: lambda estimate and likelihood-ratio test against lambda = 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------- Newick
@dataclass
class Node:
    name: str = ""
    length: float = 0.0
    children: List["Node"] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    @property
    def is_leaf(self) -> bool:
        return not self.children


def parse_newick(newick: str) -> Node:
    """Minimal recursive-descent Newick parser supporting names, branch lengths and nesting."""
    s = newick.strip()
    if s.endswith(";"):
        s = s[:-1]
    pos = 0

    def _parse_node() -> Node:
        nonlocal pos
        node = Node()
        if pos < len(s) and s[pos] == "(":
            pos += 1
            while True:
                node.children.append(_parse_node())
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    continue
                if pos < len(s) and s[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"malformed Newick near position {pos}")
        # name
        start = pos
        while pos < len(s) and s[pos] not in ":,();":
            pos += 1
        node.name = s[start:pos].strip().strip("'\"")
        if pos < len(s) and s[pos] == ":":
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in ",();":
                pos += 1
            node.length = float(s[start:pos]) if s[start:pos].strip() else 0.0
        return node

    root = _parse_node()
    return root


def leaves(node: Node) -> List[Node]:
    if node.is_leaf:
        return [node]
    out: List[Node] = []
    for c in node.children:
        out.extend(leaves(c))
    return out


def newick_to_covariance(newick: str) -> Tuple[List[str], np.ndarray]:
    """Brownian-motion covariance matrix among tips: shared root-to-MRCA path length."""
    root = parse_newick(newick)
    tips = leaves(root)
    names = [t.name for t in tips]
    n = len(names)
    cov = np.zeros((n, n), dtype=float)
    idx = {t.name: i for i, t in enumerate(tips)}

    def _walk(node: Node, depth: float) -> List[str]:
        depth_here = depth + node.length
        if node.is_leaf:
            cov[idx[node.name], idx[node.name]] = depth_here
            return [node.name]
        subsets = [_walk(c, depth_here) for c in node.children]
        for a in range(len(subsets)):
            for b in range(a + 1, len(subsets)):
                for na in subsets[a]:
                    for nb in subsets[b]:
                        cov[idx[na], idx[nb]] = depth_here
                        cov[idx[nb], idx[na]] = depth_here
        return [x for sub in subsets for x in sub]

    _walk(root, 0.0)
    return names, cov


def divergence_times_to_covariance(names: Sequence[str], divergence_mya: Dict[Tuple[str, str], float],
                                   root_age: Optional[float] = None) -> np.ndarray:
    """Covariance from pairwise divergence times: ``C_ij = root_age - t_div(i, j)``, ``C_ii = root_age``."""
    names = list(names)
    n = len(names)

    def _t(a: str, b: str) -> float:
        if (a, b) in divergence_mya:
            return divergence_mya[(a, b)]
        if (b, a) in divergence_mya:
            return divergence_mya[(b, a)]
        raise KeyError(f"no divergence time for ({a}, {b})")

    if root_age is None:
        root_age = max(_t(a, b) for i, a in enumerate(names) for b in names[i + 1:])
    cov = np.full((n, n), float(root_age))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                cov[i, j] = root_age - _t(a, b)
    return cov


#: Approximate pairwise divergence times (Myr) for common NeuroMorpho species. PLACEHOLDERS:
#: replace with a TimeTree 5 export (data/phylo/timetree_species.nwk) before any analysis.
APPROX_DIVERGENCE_MYA: Dict[Tuple[str, str], float] = {
    ("Mus musculus", "Rattus norvegicus"): 20.0,
    ("Mus musculus", "Homo sapiens"): 87.0,
    ("Rattus norvegicus", "Homo sapiens"): 87.0,
    ("Homo sapiens", "Macaca mulatta"): 29.0,
    ("Mus musculus", "Macaca mulatta"): 87.0,
    ("Rattus norvegicus", "Macaca mulatta"): 87.0,
    ("Homo sapiens", "Felis catus"): 94.0,
    ("Mus musculus", "Felis catus"): 94.0,
    ("Rattus norvegicus", "Felis catus"): 94.0,
    ("Macaca mulatta", "Felis catus"): 94.0,
}


# ---------------------------------------------------------------------- PGLS
def pagel_lambda_transform(cov: np.ndarray, lam: float) -> np.ndarray:
    """Scale off-diagonal covariances by ``lam`` (lambda = 0: star phylogeny, 1: Brownian)."""
    c = np.array(cov, dtype=float, copy=True)
    d = np.diag(c).copy()
    c *= lam
    np.fill_diagonal(c, d)
    return c


@dataclass
class PGLSResult:
    params: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    lam: float
    loglik: float
    sigma2: float
    df_resid: int
    names: List[str]
    lam_profile: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [f"PGLS  lambda={self.lam:.3f}  logLik={self.loglik:.3f}  sigma2={self.sigma2:.4g}  df={self.df_resid}"]
        for n, b, se, t, p in zip(self.names, self.params, self.bse, self.tvalues, self.pvalues):
            lines.append(f"  {n:>14s}  {b: .4f}  se={se:.4f}  t={t: .3f}  p={p:.3g}")
        return "\n".join(lines)


def _gls_fit(y: np.ndarray, X: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """GLS via Cholesky whitening. Returns beta, sigma2 (ML), loglik, cov(beta)."""
    n, p = X.shape
    L = np.linalg.cholesky(C)
    Xw = np.linalg.solve(L, X)
    yw = np.linalg.solve(L, y)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    sigma2 = float(resid @ resid / n)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    loglik = -0.5 * (n * np.log(2 * np.pi * sigma2) + logdet + n)
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    cov_beta = XtX_inv * (resid @ resid / max(n - p, 1))
    return beta, sigma2, loglik, cov_beta


def pgls(y: Sequence[float], X: Optional[np.ndarray], C: np.ndarray, lam: Optional[float] = None,
         add_intercept: bool = True, se_y: Optional[Sequence[float]] = None,
         names: Optional[Sequence[str]] = None, lam_grid: Optional[np.ndarray] = None) -> PGLSResult:
    """Phylogenetic GLS with Pagel's lambda (ML on a grid unless ``lam`` is given).

    Parameters
    ----------
    y : species-level response (length n).
    X : n x p design matrix (without intercept), or ``None`` for intercept-only.
    C : n x n phylogenetic covariance (Brownian expectation), same species order as ``y``.
    se_y : optional standard errors of the species means; their squares are added to the
        diagonal (measurement-error PGLS).
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if X is None:
        X = np.empty((n, 0))
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        names = ["intercept"] + list(names or [f"x{i}" for i in range(X.shape[1] - 1)])
    else:
        names = list(names or [f"x{i}" for i in range(X.shape[1])])
    C = np.asarray(C, dtype=float)
    # scale C to unit mean diagonal so sigma2 is interpretable, keep measurement error separate
    scale = float(np.mean(np.diag(C)))
    C = C / scale
    me = np.zeros(n) if se_y is None else (np.asarray(se_y, dtype=float) ** 2)

    def _fit(l: float):
        Cl = pagel_lambda_transform(C, l)
        # measurement error is added after lambda scaling, relative to the phylogenetic variance
        return _gls_fit(y, X, Cl + np.diag(me / max(np.var(y), 1e-12)))

    profile = None
    if lam is None:
        grid = np.linspace(0.0, 1.0, 101) if lam_grid is None else np.asarray(lam_grid, dtype=float)
        lls = np.array([_fit(l)[2] for l in grid])
        profile = np.column_stack([grid, lls])
        lam = float(grid[int(np.argmax(lls))])
    beta, sigma2, loglik, cov_beta = _fit(lam)
    bse = np.sqrt(np.diag(cov_beta))
    df_resid = n - X.shape[1]
    tvals = beta / bse
    pvals = 2 * stats.t.sf(np.abs(tvals), df=max(df_resid, 1))
    return PGLSResult(params=beta, bse=bse, tvalues=tvals, pvalues=pvals, lam=float(lam), loglik=float(loglik),
                      sigma2=sigma2, df_resid=int(df_resid), names=names, lam_profile=profile)


def phylogenetic_signal(y: Sequence[float], C: np.ndarray, se_y: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Pagel's lambda for a trait (intercept-only PGLS) with an LR test against lambda = 0."""
    fit_ml = pgls(y, None, C, se_y=se_y)
    fit_0 = pgls(y, None, C, lam=0.0, se_y=se_y)
    lr = 2.0 * (fit_ml.loglik - fit_0.loglik)
    p = float(stats.chi2.sf(max(lr, 0.0), df=1))
    return {"lambda": fit_ml.lam, "loglik_ml": fit_ml.loglik, "loglik_lambda0": fit_0.loglik, "LR": float(lr), "p": p}


def species_level_table(df, feature: str, species_col: str = "scientific_name", min_n: int = 5):
    """Mean, SE and n of ``feature`` per species (input to PGLS); use lab-adjusted values upstream."""
    g = df.groupby(species_col)[feature]
    tab = g.agg(["mean", "sem", "count"]).rename(columns={"sem": "se", "count": "n"})
    return tab[tab["n"] >= min_n]


def align_species(names_cov: Sequence[str], C: np.ndarray, table) -> Tuple[List[str], np.ndarray, "object"]:
    """Restrict a covariance matrix and a species table to their common species, in the same order."""
    common = [n for n in names_cov if n in table.index]
    idx = [list(names_cov).index(n) for n in common]
    return common, C[np.ix_(idx, idx)], table.loc[common]


__all__ = [
    "Node", "parse_newick", "leaves", "newick_to_covariance", "divergence_times_to_covariance",
    "APPROX_DIVERGENCE_MYA", "pagel_lambda_transform", "PGLSResult", "pgls", "phylogenetic_signal",
    "species_level_table", "align_species",
]
