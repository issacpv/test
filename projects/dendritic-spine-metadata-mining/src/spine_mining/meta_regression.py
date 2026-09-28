"""Random-effects meta-regression of log spine density on method and biology moderators.

* :func:`prepare_effects` - log-transform densities and derive sampling variances from SD/SEM and n
  (delta method; falls back to an imputed CV when uncertainty is missing).
* :func:`meta_regression` - method-of-moments tau^2 (DerSimonian-Laird generalised to
  moderators) + weighted least squares; optional cluster-robust SEs by lab.
* :func:`method_r2_permutation` - permutation null for the between-study variance explained by
  the method moderators, shuffling method labels within biological strata.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def prepare_effects(df: pd.DataFrame, value_col: str = "density_per_um", sd_col: str = "sd", n_col: str = "n",
                    sem_col: Optional[str] = "sem", default_cv: float = 0.35) -> pd.DataFrame:
    """Add ``yi`` (log density) and ``vi`` (sampling variance of the log mean) columns.

    Var(log mean) ~ (SD / (mean sqrt(n)))^2. When SD is missing but SEM is given, SD = SEM sqrt(n).
    When neither is given, a default coefficient of variation is imputed and the row is flagged.
    """
    out = df.copy()
    y = out[value_col].astype(float)
    out["yi"] = np.log(y)
    sd = out[sd_col].astype(float) if sd_col in out else pd.Series(np.nan, index=out.index)
    n = out[n_col].astype(float) if n_col in out else pd.Series(np.nan, index=out.index)
    if sem_col and sem_col in out:
        sem = out[sem_col].astype(float)
        sd = sd.where(sd.notna(), sem * np.sqrt(n))
    cv = sd / y
    imputed = cv.isna() | n.isna()
    cv = cv.where(~imputed, default_cv)
    n_eff = n.where(n.notna(), 5.0)
    out["vi"] = (cv ** 2) / n_eff
    out["vi_imputed"] = imputed
    return out


def _design_matrix(df: pd.DataFrame, moderators: Sequence[str]) -> pd.DataFrame:
    X = pd.get_dummies(df[list(moderators)], drop_first=True, dtype=float) if moderators else pd.DataFrame(index=df.index)
    X.insert(0, "intercept", 1.0)
    return X


def _tau2_mom(y: np.ndarray, v: np.ndarray, X: np.ndarray) -> float:
    W = np.diag(1.0 / v)
    P = W - W @ X @ np.linalg.pinv(X.T @ W @ X) @ X.T @ W
    q = float(y @ P @ y)
    df = len(y) - X.shape[1]
    tr = float(np.trace(P))
    return max(0.0, (q - df) / tr) if tr > 0 else 0.0


def meta_regression(df: pd.DataFrame, moderators: Sequence[str], cluster: Optional[str] = "lab", yi: str = "yi", vi: str = "vi"
                    ) -> Dict[str, object]:
    """Random-effects meta-regression with method-of-moments tau^2 and (optionally) cluster-robust SEs.

    Returns coefficients (log-ratio scale), SEs, z, p, tau^2, I^2, between-study R^2 relative to the
    intercept-only model, the pooled intercept-only estimate and the design column names.
    """
    d = df.dropna(subset=[yi, vi]).reset_index(drop=True)
    X = _design_matrix(d, moderators)
    Xa = X.to_numpy(dtype=float)
    y = d[yi].to_numpy(dtype=float)
    v = d[vi].to_numpy(dtype=float)
    tau2 = _tau2_mom(y, v, Xa)
    tau2_0 = _tau2_mom(y, v, np.ones((len(y), 1)))
    w = 1.0 / (v + tau2)
    W = np.diag(w)
    cov = np.linalg.pinv(Xa.T @ W @ Xa)
    beta = cov @ Xa.T @ W @ y
    resid = y - Xa @ beta
    if cluster and cluster in d.columns:
        # cluster-robust (sandwich) covariance by lab
        meat = np.zeros_like(cov)
        for g in d[cluster].unique():
            idx = np.flatnonzero(d[cluster].to_numpy() == g)
            u = (Xa[idx] * (w[idx] * resid[idx])[:, None]).sum(axis=0)
            meat += np.outer(u, u)
        G = d[cluster].nunique()
        cov_used = cov @ meat @ cov * (G / max(G - 1, 1))
    else:
        cov_used = cov
    se = np.sqrt(np.diag(cov_used))
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    w0 = 1.0 / v
    mu0 = float(np.sum(w0 * y) / np.sum(w0))
    Q = float(np.sum(w0 * (y - mu0) ** 2))
    I2 = float(max(0.0, (Q - (len(y) - 1)) / Q)) if Q > 0 else 0.0
    names = list(X.columns)
    return {"coef": pd.Series(beta, index=names), "se": pd.Series(se, index=names), "z": pd.Series(z, index=names),
            "p": pd.Series(p, index=names), "tau2": float(tau2), "tau2_null": float(tau2_0),
            "R2_between": float(1 - tau2 / tau2_0) if tau2_0 > 0 else np.nan, "I2": I2, "Q": Q, "n": int(len(y)),
            "columns": names, "ratio": pd.Series(np.exp(beta), index=names)}


def method_r2_permutation(df: pd.DataFrame, method_cols: Sequence[str], bio_cols: Sequence[str], n_perm: int = 200,
                          cluster: Optional[str] = "lab", seed: int = 0) -> Dict[str, float]:
    """Permutation null for the extra between-study R^2 of method moderators over biology moderators.

    Method labels are shuffled *within* biological strata (all combinations of ``bio_cols``) so the
    null preserves the confounding structure between biology and method availability.
    """
    rng = np.random.default_rng(seed)
    full = meta_regression(df, list(bio_cols) + list(method_cols), cluster)
    bio = meta_regression(df, list(bio_cols), cluster)
    obs = (bio["tau2"] - full["tau2"]) / bio["tau2"] if bio["tau2"] > 0 else np.nan
    strata = df[list(bio_cols)].astype(str).agg("|".join, axis=1) if bio_cols else pd.Series("all", index=df.index)
    null = np.empty(n_perm)
    for i in range(n_perm):
        d = df.copy()
        for s in strata.unique():
            idx = np.flatnonzero(strata.to_numpy() == s)
            perm = rng.permutation(idx)
            for c in method_cols:
                d.loc[d.index[idx], c] = df[c].to_numpy()[perm]
        f = meta_regression(d, list(bio_cols) + list(method_cols), cluster)
        null[i] = (bio["tau2"] - f["tau2"]) / bio["tau2"] if bio["tau2"] > 0 else np.nan
    null = null[np.isfinite(null)]
    return {"observed_R2_method": float(obs), "null_mean": float(null.mean()) if null.size else np.nan,
            "p_perm": float((1 + np.sum(null >= obs)) / (len(null) + 1)) if null.size else np.nan}


def simulate_literature(n_studies: int = 120, n_labs: int = 30, method_effects: Optional[Dict[str, float]] = None,
                        lab_sd: float = 0.15, study_sd: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Synthetic study table with known log-ratio method effects, lab random effects and reported SD/n."""
    rng = np.random.default_rng(seed)
    method_effects = method_effects or {"golgi": -0.5, "fluorescence": 0.0, "em": 0.3}
    methods = list(method_effects)
    species = ["mouse", "rat", "human"]
    comp = ["basal", "apical_oblique"]
    labs = [f"lab{i}" for i in range(n_labs)]
    lab_eff = dict(zip(labs, rng.normal(0, lab_sd, n_labs)))
    rows: List[Dict[str, object]] = []
    for i in range(n_studies):
        m = methods[rng.integers(len(methods))]
        sp = species[rng.integers(len(species))]
        c = comp[rng.integers(len(comp))]
        lab = labs[rng.integers(n_labs)]
        n = int(rng.integers(6, 40))
        mu = np.log(1.5) + method_effects[m] + (0.25 if sp == "human" else 0.0) + (0.1 if c == "apical_oblique" else 0.0) + lab_eff[lab]
        true_mean = np.exp(mu + rng.normal(0, study_sd))
        cv = 0.3
        obs_mean = true_mean * (1 + rng.normal(0, cv / np.sqrt(n)))
        rows.append({"study": f"s{i}", "lab": lab, "method": m, "species": sp, "compartment": c,
                     "density_per_um": max(obs_mean, 0.05), "sd": cv * true_mean, "n": n})
    return pd.DataFrame(rows)


__all__ = ["prepare_effects", "meta_regression", "method_r2_permutation", "simulate_literature"]
