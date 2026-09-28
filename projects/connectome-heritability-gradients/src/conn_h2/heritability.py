"""Twin-based heritability estimation.

Three estimators of increasing sophistication, all operating on arrays of
MZ and DZ pair values shaped ``(n_pairs, 2)``:

* :func:`falconer_h2` - ``h2 = 2 (rMZ - rDZ)`` with ANOVA intraclass
  correlations (quick, biased when C and A trade off).
* :func:`defries_fulker` - DF regression on double-entered pairs with the
  proband x relatedness interaction giving ``h2`` (Rodgers & Kohler, 2005).
* :func:`ace_ml` - maximum-likelihood ACE structural model (bivariate normal
  per pair), with nested AE / CE / E models and likelihood-ratio tests. This
  is the model OpenMx / APACE fit; here it is solved directly with SciPy.

Plus family-structure helpers for the HCP restricted table, pair-resampling
bootstrap CIs, covariate residualisation and a regional map driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats


# --------------------------------------------------------------------------- #
# Pair construction
# --------------------------------------------------------------------------- #
@dataclass
class TwinPairs:
    """Row indices (into the subject table) of MZ, DZ and non-twin sibling pairs."""

    mz: np.ndarray  # (n_mz, 2)
    dz: np.ndarray  # (n_dz, 2)
    sib: np.ndarray  # (n_sib, 2)

    def summary(self) -> Dict[str, int]:
        return {"n_mz_pairs": len(self.mz), "n_dz_pairs": len(self.dz), "n_sib_pairs": len(self.sib)}


def build_twin_pairs(
    df: pd.DataFrame,
    zygosity_col: str = "ZygosityGT",
    fallback_zygosity_col: Optional[str] = "ZygositySR",
    family_col: str = "Family_ID",
    mother_col: str = "Mother_ID",
    father_col: str = "Father_ID",
    mz_labels: Sequence[str] = ("MZ",),
    dz_labels: Sequence[str] = ("DZ", "NotMZ"),
) -> TwinPairs:
    """Derive MZ / DZ / full-sibling pairs from the HCP restricted table.

    HCP conventions: ``ZygosityGT`` (genotype-confirmed: ``MZ``, ``DZ``, blank
    for non-twins) is preferred; ``ZygositySR`` (self-report: ``MZ``, ``NotMZ``,
    ``NotTwin``) is used when GT is missing. Twins share ``Family_ID`` and
    both parent IDs; full siblings share both parent IDs but are not twins.
    Only the first two twins per family are paired (HCP has no triplets).
    """
    df = df.reset_index(drop=True)
    zyg = df[zygosity_col].astype(str).str.strip() if zygosity_col in df.columns else pd.Series("", index=df.index)
    if fallback_zygosity_col and fallback_zygosity_col in df.columns:
        fb = df[fallback_zygosity_col].astype(str).str.strip()
        zyg = zyg.where(zyg.isin(list(mz_labels) + list(dz_labels)), fb)
    is_mz = zyg.isin(mz_labels)
    is_dz = zyg.isin(dz_labels)
    mz, dz, sib = [], [], []
    used = set()
    for fam, grp in df.groupby(family_col):
        idx = grp.index.to_numpy()
        for mask, store in ((is_mz, mz), (is_dz, dz)):
            tw = [i for i in idx if mask[i]]
            if len(tw) >= 2:
                store.append((tw[0], tw[1]))
                used.update((tw[0], tw[1]))
        # non-twin full siblings: all remaining pairs sharing both parents
        rest = [i for i in idx if i not in used]
        for a in range(len(rest)):
            for b in range(a + 1, len(rest)):
                i, j = rest[a], rest[b]
                same_parents = True
                for col in (mother_col, father_col):
                    if col in df.columns and str(df.at[i, col]) != str(df.at[j, col]):
                        same_parents = False
                if same_parents:
                    sib.append((i, j))
    to_arr = lambda x: np.asarray(x, dtype=int).reshape(-1, 2)  # noqa: E731
    return TwinPairs(to_arr(mz), to_arr(dz), to_arr(sib))


def pair_values(x: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Gather a ``(n_subjects,)`` phenotype into ``(n_pairs, 2)``."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x[pairs[:, 0]], x[pairs[:, 1]]])


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #
def icc_pairs(values: np.ndarray) -> float:
    """One-way ANOVA intraclass correlation ICC(1) for ``(n_pairs, 2)`` data."""
    v = np.asarray(values, dtype=float)
    v = v[np.all(np.isfinite(v), axis=1)]
    n = len(v)
    if n < 3:
        return np.nan
    k = 2
    grand = v.mean()
    pair_means = v.mean(axis=1)
    msb = k * ((pair_means - grand) ** 2).sum() / (n - 1)
    msw = ((v - pair_means[:, None]) ** 2).sum() / (n * (k - 1))
    denom = msb + (k - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else np.nan


def double_entry(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Double-entered (proband, co-twin) vectors."""
    v = np.asarray(values, dtype=float)
    return np.concatenate([v[:, 0], v[:, 1]]), np.concatenate([v[:, 1], v[:, 0]])


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def falconer_h2(mz: np.ndarray, dz: np.ndarray, clip: bool = True) -> Dict[str, float]:
    """Falconer's formula with ICC(1) twin correlations."""
    r_mz, r_dz = icc_pairs(mz), icc_pairs(dz)
    a2 = 2 * (r_mz - r_dz)
    c2 = 2 * r_dz - r_mz
    e2 = 1 - r_mz
    if clip:
        a2, c2, e2 = (float(np.clip(x, 0, 1)) for x in (a2, c2, e2))
    return {"h2": float(a2), "c2": float(c2), "e2": float(e2), "r_mz": r_mz, "r_dz": r_dz}


def defries_fulker(mz: np.ndarray, dz: np.ndarray) -> Dict[str, float]:
    """DeFries-Fulker regression on double-entered pairs.

    ``C = b0 + b1 P + b2 R + b3 (P x R)`` with ``R`` = 1 (MZ) or 0.5 (DZ).
    ``h2 = b3`` and ``c2 = b1`` (Rodgers & Kohler, 2005). Standard errors are
    naive (double entry duplicates observations); use :func:`bootstrap_ci`.
    """
    pm, cm = double_entry(mz)
    pd_, cd = double_entry(dz)
    P = np.concatenate([pm, pd_])
    C = np.concatenate([cm, cd])
    R = np.concatenate([np.ones(len(pm)), np.full(len(pd_), 0.5)])
    X = np.column_stack([np.ones(len(P)), P, R, P * R])
    ok = np.isfinite(P) & np.isfinite(C)
    beta, *_ = np.linalg.lstsq(X[ok], C[ok], rcond=None)
    return {"h2": float(beta[3]), "c2": float(beta[1]), "b0": float(beta[0]), "b_R": float(beta[2])}


def _nll_pairs(params: np.ndarray, mz: np.ndarray, dz: np.ndarray, model: str) -> float:
    a, c, e, mu = params
    a2 = a * a if "A" in model else 0.0
    c2 = c * c if "C" in model else 0.0
    e2 = e * e
    V = a2 + c2 + e2
    if V <= 1e-10 or e2 <= 1e-10:
        return 1e10
    nll = 0.0
    for data, cov in ((mz, a2 + c2), (dz, 0.5 * a2 + c2)):
        if len(data) == 0:
            continue
        S = np.array([[V, cov], [cov, V]])
        det = V * V - cov * cov
        if det <= 1e-12:
            return 1e10
        Sinv = np.array([[V, -cov], [-cov, V]]) / det
        d = data - mu
        quad = np.einsum("ij,jk,ik->i", d, Sinv, d)
        nll += 0.5 * (quad.sum() + len(data) * (np.log(det) + 2 * np.log(2 * np.pi)))
    return float(nll)


def ace_ml(mz: np.ndarray, dz: np.ndarray, models: Sequence[str] = ("ACE", "AE", "CE", "E")) -> pd.DataFrame:
    """Maximum-likelihood ACE model for (standardised or raw) twin data.

    Returns a table with one row per model: variance components as
    proportions (``a2, c2, e2``), ``loglik``, ``aic``, and the likelihood-ratio
    test p-value against the full ACE model (``lrt_p``). ``ACE`` and ``AE``
    are the models of interest; ``CE`` and ``E`` test whether A is needed.
    """
    mz = np.asarray(mz, dtype=float)
    dz = np.asarray(dz, dtype=float)
    mz = mz[np.all(np.isfinite(mz), axis=1)]
    dz = dz[np.all(np.isfinite(dz), axis=1)]
    allv = np.concatenate([mz.ravel(), dz.ravel()])
    sd, mu0 = allv.std(), allv.mean()
    rows = {}
    fal = falconer_h2(mz, dz)
    for model in models:
        a0 = np.sqrt(max(fal["h2"], 0.1)) * sd
        c0 = np.sqrt(max(fal["c2"], 0.1)) * sd
        e0 = np.sqrt(max(fal["e2"], 0.1)) * sd
        x0 = np.array([a0, c0, e0, mu0])
        res = optimize.minimize(_nll_pairs, x0, args=(mz, dz, model), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
        a, c, e, mu = res.x
        a2 = a * a if "A" in model else 0.0
        c2 = c * c if "C" in model else 0.0
        e2 = e * e
        V = a2 + c2 + e2
        n_par = 1 + ("A" in model) + ("C" in model) + 1
        rows[model] = {"a2": a2 / V, "c2": c2 / V, "e2": e2 / V, "var": V, "mu": mu, "loglik": -res.fun, "n_par": n_par, "aic": 2 * n_par + 2 * res.fun}
    out = pd.DataFrame(rows).T
    if "ACE" in out.index:
        full = out.loc["ACE", "loglik"]
        out["lrt_p"] = [
            stats.chi2.sf(2 * (full - out.loc[m, "loglik"]), df=max(out.loc["ACE", "n_par"] - out.loc[m, "n_par"], 1)) if m != "ACE" else np.nan
            for m in out.index
        ]
    return out


# --------------------------------------------------------------------------- #
# Uncertainty and correction
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    mz: np.ndarray,
    dz: np.ndarray,
    estimator: Callable[[np.ndarray, np.ndarray], Dict[str, float]] = falconer_h2,
    key: str = "h2",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap over *pairs* (within zygosity) for any estimator."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        mzb = mz[rng.integers(0, len(mz), len(mz))]
        dzb = dz[rng.integers(0, len(dz), len(dz))]
        vals.append(estimator(mzb, dzb)[key])
    vals = np.asarray(vals, dtype=float)
    point = estimator(mz, dz)[key]
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return {key: float(point), "ci_low": float(lo), "ci_high": float(hi), "boot_sd": float(np.nanstd(vals, ddof=1))}


def residualize_covariates(features: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Regress covariates (age, sex, motion, ...) out of every feature column."""
    F = np.asarray(features, dtype=float)
    C = np.column_stack([np.ones(len(F)), np.asarray(covariates, dtype=float)])
    beta, *_ = np.linalg.lstsq(C, F, rcond=None)
    return F - C @ beta


def reliability_ceiling(icc_retest: float) -> float:
    """Upper bound on h2 given test-retest reliability (h2 cannot exceed the
    reliable variance; Ge et al., 2017, PNAS)."""
    return float(np.clip(icc_retest, 0, 1))


def disattenuated_h2(h2: float, icc_retest: float) -> float:
    """h2 rescaled to the reliable variance fraction (clipped to [0, 1])."""
    if icc_retest <= 0:
        return np.nan
    return float(np.clip(h2 / icc_retest, 0, 1))


# --------------------------------------------------------------------------- #
# Regional maps
# --------------------------------------------------------------------------- #
def heritability_map(
    features: np.ndarray,
    pairs: TwinPairs,
    method: str = "falconer",
    covariates: Optional[np.ndarray] = None,
    n_boot: int = 0,
    seed: int = 0,
    standardize: bool = True,
) -> pd.DataFrame:
    """Estimate heritability of every column of ``features`` ``(n_subjects, n_regions)``.

    ``method`` in ``{"falconer", "df", "ace"}``. Returns one row per region
    with ``h2, c2, e2, r_mz, r_dz`` (+ bootstrap CI when ``n_boot > 0``; for
    ``ace`` also ``lrt_p_AE`` = p of dropping C and ``lrt_p_CE`` = p of dropping A).
    """
    F = np.asarray(features, dtype=float)
    if covariates is not None:
        F = residualize_covariates(F, covariates)
    if standardize:
        sd = F.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        F = (F - F.mean(axis=0)) / sd
    rows = []
    for j in range(F.shape[1]):
        mz = pair_values(F[:, j], pairs.mz)
        dz = pair_values(F[:, j], pairs.dz)
        if method == "falconer":
            est = falconer_h2(mz, dz)
            row = dict(est)
            if n_boot:
                row.update({k: v for k, v in bootstrap_ci(mz, dz, falconer_h2, "h2", n_boot, seed).items() if k != "h2"})
        elif method == "df":
            est = defries_fulker(mz, dz)
            row = {"h2": est["h2"], "c2": est["c2"], "e2": 1 - est["h2"] - est["c2"], "r_mz": icc_pairs(mz), "r_dz": icc_pairs(dz)}
            if n_boot:
                row.update({k: v for k, v in bootstrap_ci(mz, dz, defries_fulker, "h2", n_boot, seed).items() if k != "h2"})
        elif method == "ace":
            tab = ace_ml(mz, dz)
            row = {"h2": tab.loc["ACE", "a2"], "c2": tab.loc["ACE", "c2"], "e2": tab.loc["ACE", "e2"], "r_mz": icc_pairs(mz), "r_dz": icc_pairs(dz),
                   "h2_AE": tab.loc["AE", "a2"] if "AE" in tab.index else np.nan,
                   "lrt_p_AE": tab.loc["AE", "lrt_p"] if "AE" in tab.index else np.nan,
                   "lrt_p_CE": tab.loc["CE", "lrt_p"] if "CE" in tab.index else np.nan}
            if n_boot:
                row.update({k: v for k, v in bootstrap_ci(mz, dz, lambda a, b: {"h2": ace_ml(a, b, ("ACE",)).loc["ACE", "a2"]}, "h2", n_boot, seed).items() if k != "h2"})
        else:
            raise ValueError("method must be 'falconer', 'df' or 'ace'")
        row["region"] = j
        rows.append(row)
    return pd.DataFrame(rows).set_index("region")


def genetic_correlation_falconer(x: np.ndarray, y: np.ndarray, pairs: TwinPairs) -> Dict[str, float]:
    """Cross-trait twin correlations and a Falconer-style bivariate estimate.

    ``rG`` is approximated from cross-twin cross-trait correlations:
    ``covA = 2 (rMZ_xy - rDZ_xy)`` and ``rG = covA / sqrt(h2_x h2_y)``.
    This is a quick screen; confirm with a bivariate Cholesky ACE in OpenMx.
    """
    def cross_corr(pairs_idx: np.ndarray) -> float:
        a = np.concatenate([x[pairs_idx[:, 0]], x[pairs_idx[:, 1]]])
        b = np.concatenate([y[pairs_idx[:, 1]], y[pairs_idx[:, 0]]])
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 else np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = (x - np.nanmean(x)) / np.nanstd(x)
    y = (y - np.nanmean(y)) / np.nanstd(y)
    r_mz_xy = cross_corr(pairs.mz)
    r_dz_xy = cross_corr(pairs.dz)
    h2x = falconer_h2(pair_values(x, pairs.mz), pair_values(x, pairs.dz))["h2"]
    h2y = falconer_h2(pair_values(y, pairs.mz), pair_values(y, pairs.dz))["h2"]
    cov_a = 2 * (r_mz_xy - r_dz_xy)
    rg = cov_a / np.sqrt(h2x * h2y) if h2x > 0 and h2y > 0 else np.nan
    return {"r_mz_xy": r_mz_xy, "r_dz_xy": r_dz_xy, "cov_A": float(cov_a), "rG": float(np.clip(rg, -1, 1)) if np.isfinite(rg) else np.nan, "h2_x": h2x, "h2_y": h2y}
