"""Effect sizes, random-effects meta-analysis and signature similarity.

Implements the standard two-stage design: Hedges' g per archive contrast, then
DerSimonian-Laird or REML pooling with Hartung-Knapp-Sidik-Jonkman (HKSJ)
confidence intervals, plus Q / I^2 / tau^2, Egger's small-study test, and a
condition "signature" (vector of pooled g over morphometrics) with cosine
similarity and a within-archive permutation null.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def hedges_g(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Hedges' g (x minus y, small-sample corrected) and its sampling variance."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan"), float("nan")
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if sp2 <= 0:
        return float("nan"), float("nan")
    d = (x.mean() - y.mean()) / np.sqrt(sp2)
    df = nx + ny - 2
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    g = j * d
    var = (nx + ny) / (nx * ny) + g ** 2 / (2.0 * (nx + ny))
    return float(g), float(var)


@dataclass
class PooledEffect:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    tau2: float
    q: float
    q_p: float
    i2: float
    k: int
    method: str


def _tau2_dl(y: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    w = 1.0 / v
    mu_fe = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu_fe) ** 2))
    k = len(y)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    return tau2, q


def _tau2_reml(y: np.ndarray, v: np.ndarray, tol: float = 1e-8, max_iter: int = 200) -> float:
    """REML estimate of tau^2 by Fisher scoring (Viechtbauer, 2005 style iteration)."""
    tau2, _ = _tau2_dl(y, v)
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        den = np.sum(w ** 2)
        if den <= 0:
            return 0.0
        # REML fixed point: tau2 <- sum(w^2 ((y-mu)^2 - v)) / sum(w^2) + 1/sum(w)
        new = max(0.0, np.sum(w ** 2 * ((y - mu) ** 2 - v)) / den + 1.0 / np.sum(w))
        if abs(new - tau2) < tol:
            tau2 = new
            break
        tau2 = new
    return float(tau2)


def random_effects(effects: Sequence[float], variances: Sequence[float], method: str = "REML",
                   hksj: bool = True, alpha: float = 0.05) -> PooledEffect:
    """Pool per-archive effects with a random-effects model.

    ``method`` is "DL" or "REML". With ``hksj=True`` the confidence interval uses
    the Hartung-Knapp-Sidik-Jonkman variance and a t distribution with k-1 df,
    which is better calibrated when the number of archives is small.
    """
    y = np.asarray(effects, float)
    v = np.asarray(variances, float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    if k == 0:
        nan = float("nan")
        return PooledEffect(nan, nan, nan, nan, nan, nan, nan, nan, 0, method)
    tau2_dl, q = _tau2_dl(y, v)
    tau2 = _tau2_reml(y, v) if method.upper() == "REML" and k > 1 else tau2_dl
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    if hksj and k > 1:
        q_star = float(np.sum(w * (y - mu) ** 2) / (k - 1))
        se_hk = float(np.sqrt(max(q_star, 1.0) / np.sum(w)))  # truncated HKSJ (never narrower than RE)
        crit = stats.t.ppf(1 - alpha / 2, k - 1)
        ci = (mu - crit * se_hk, mu + crit * se_hk)
        se = se_hk
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
        ci = (mu - crit * se, mu + crit * se)
    q_p = float(stats.chi2.sf(q, k - 1)) if k > 1 else float("nan")
    i2 = float(max(0.0, (q - (k - 1)) / q) * 100.0) if q > 0 and k > 1 else 0.0
    return PooledEffect(mu, se, float(ci[0]), float(ci[1]), float(tau2), q, q_p, i2, k, method.upper())


def egger_test(effects: Sequence[float], variances: Sequence[float]) -> Dict[str, float]:
    """Egger regression of standardised effect on precision; intercept != 0 flags small-study effects."""
    y = np.asarray(effects, float)
    v = np.asarray(variances, float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    if len(y) < 3:
        return {"intercept": float("nan"), "p": float("nan"), "k": int(len(y))}
    se = np.sqrt(v)
    res = stats.linregress(1.0 / se, y / se)
    return {"intercept": float(res.intercept), "p": float(res.pvalue), "k": int(len(y))}


def contrast_effects(df: pd.DataFrame, outcome: str, condition_col: str = "condition_class",
                     contrast_col: str = "contrast_id", control_label: str = "control") -> pd.DataFrame:
    """Hedges' g of ``outcome`` (case minus control) within each contrast stratum."""
    rows = []
    for cid, g in df[df[contrast_col].astype(str) != ""].groupby(contrast_col):
        ctrl = g.loc[g[condition_col] == control_label, outcome].dropna()
        for cls, gc in g[g[condition_col] != control_label].groupby(condition_col):
            case = gc[outcome].dropna()
            eff, var = hedges_g(case, ctrl)
            rows.append({"contrast_id": cid, "condition_class": cls, "outcome": outcome,
                         "n_case": len(case), "n_control": len(ctrl), "g": eff, "var": var})
    return pd.DataFrame(rows)


def pooled_signature(df: pd.DataFrame, outcomes: Sequence[str], condition: str, **kw) -> pd.Series:
    """Vector of pooled g over ``outcomes`` for one condition class."""
    vals = {}
    for o in outcomes:
        eff = contrast_effects(df, o)
        eff = eff[eff["condition_class"] == condition]
        vals[o] = random_effects(eff["g"], eff["var"], **kw).estimate if len(eff) else float("nan")
    return pd.Series(vals, name=condition)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else float("nan")


def permute_within_archive(df: pd.DataFrame, rng: np.random.Generator, condition_col: str = "condition_class",
                           contrast_col: str = "contrast_id") -> pd.DataFrame:
    """Shuffle condition labels *within* each contrast stratum (preserves archive composition)."""
    out = df.copy()
    for cid, idx in out.groupby(contrast_col).groups.items():
        if str(cid) == "":
            continue
        idx = np.asarray(list(idx))
        out.loc[idx, condition_col] = rng.permutation(out.loc[idx, condition_col].to_numpy())
    return out


def signature_similarity_test(df: pd.DataFrame, outcomes: Sequence[str], cond_a: str, cond_b: str,
                              n_perm: int = 200, seed: int = 0,
                              stat: Callable[[Sequence[float], Sequence[float]], float] = cosine_similarity,
                              ) -> Dict[str, float]:
    """Observed cosine similarity of two condition signatures and a within-archive permutation p-value."""
    rng = np.random.default_rng(seed)
    obs = stat(pooled_signature(df, outcomes, cond_a), pooled_signature(df, outcomes, cond_b))
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = permute_within_archive(df, rng)
        null[i] = stat(pooled_signature(p, outcomes, cond_a), pooled_signature(p, outcomes, cond_b))
    null = null[np.isfinite(null)]
    p_val = float((np.sum(null >= obs) + 1) / (len(null) + 1)) if len(null) else float("nan")
    return {"observed": float(obs), "p_perm": p_val, "null_mean": float(null.mean()) if len(null) else float("nan"),
            "n_perm": int(len(null))}


def meta_regression(effects: pd.DataFrame, moderators: pd.DataFrame, tau2: Optional[float] = None) -> Dict[str, object]:
    """Weighted least squares of g on moderators with weights 1/(var + tau2) (method-of-moments tau2 if None)."""
    y = effects["g"].to_numpy(float)
    v = effects["var"].to_numpy(float)
    X = np.column_stack([np.ones(len(y)), moderators.to_numpy(float)])
    if tau2 is None:
        tau2, _ = _tau2_dl(y, v)
    w = 1.0 / (v + tau2)
    Wsq = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * Wsq[:, None], y * Wsq, rcond=None)
    resid = y - X @ beta
    cov = np.linalg.pinv((X * w[:, None]).T @ X)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    names = ["intercept", *moderators.columns]
    q_res = float(np.sum(w * resid ** 2))
    return {"coef": dict(zip(names, beta)), "se": dict(zip(names, se)), "p": dict(zip(names, p)),
            "tau2": float(tau2), "q_residual": q_res, "df_residual": int(len(y) - X.shape[1])}
