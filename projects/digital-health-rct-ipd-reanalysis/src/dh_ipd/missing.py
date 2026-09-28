"""Attrition sensitivity analyses for a single continuous follow-up outcome.

Digital-health trials lose 20-50% of participants by primary endpoint
(Eysenbach's "law of attrition"), rarely at random.  This module implements
the controlled multiple-imputation family recommended for such trials
(Carpenter, Roger & Kenward, 2013; Cro et al., 2020):

* ``method='mar'`` - arm-specific regression imputation (missing at random);
* ``method='j2r'`` - jump to reference: treated dropouts are imputed from the
  *control* arm's outcome model (no benefit after dropping out);
* ``method='delta'`` - MAR imputation minus ``delta`` (in outcome SD units) for
  treated dropouts; scanning ``delta`` gives a tipping-point analysis.

Imputations are proper (parameters drawn from their approximate posterior)
and combined with Rubin's rules.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .engagement import _design, _ols


def complete_case(df: pd.DataFrame, y: str = "y", z: str = "z", baseline: str = "baseline", covariates: Sequence[str] = ()) -> Dict[str, float]:
    d = df.dropna(subset=[y])
    beta, se = _ols(d[y].to_numpy(float), _design(d, z, baseline, covariates))
    return {"estimate": float(beta[1]), "se": float(se[1]), "n": float(len(d)), "missing_rate": float(df[y].isna().mean())}


def _draw_model(d: pd.DataFrame, y: str, cols: Sequence[str], rng: np.random.Generator):
    """Posterior draw of (beta, sigma) for y ~ 1 + cols on observed rows of one arm."""
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    yv = d[y].to_numpy(float)
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(p))
    beta_hat = XtX_inv @ X.T @ yv
    resid = yv - X @ beta_hat
    df_res = max(n - p, 1)
    sigma2 = float(resid @ resid) / df_res
    sigma2_draw = sigma2 * df_res / rng.chisquare(df_res)
    beta_draw = rng.multivariate_normal(beta_hat, sigma2_draw * XtX_inv)
    return beta_draw, np.sqrt(sigma2_draw)


def rubin_combine(estimates: Sequence[float], variances: Sequence[float]) -> Dict[str, float]:
    """Rubin's rules with Barnard-Rubin small-sample degrees of freedom."""
    q = np.asarray(estimates, float)
    u = np.asarray(variances, float)
    m = len(q)
    qbar = float(q.mean())
    W = float(u.mean())
    B = float(q.var(ddof=1)) if m > 1 else 0.0
    T = W + (1 + 1 / m) * B
    fmi = ((1 + 1 / m) * B) / T if T > 0 else 0.0
    df_old = (m - 1) / max(fmi**2, 1e-12) if m > 1 else np.inf
    se = float(np.sqrt(T))
    t = qbar / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=min(df_old, 1e6))) if np.isfinite(t) else np.nan
    return {"estimate": qbar, "se": se, "ci_low": qbar - 1.96 * se, "ci_high": qbar + 1.96 * se, "p": p, "fmi": float(fmi), "m": float(m)}


def multiple_imputation(
    df: pd.DataFrame,
    y: str = "y",
    z: str = "z",
    baseline: str = "baseline",
    covariates: Sequence[str] = (),
    method: str = "mar",
    delta: float = 0.0,
    m: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """Controlled multiple imputation of ``y`` followed by ANCOVA and Rubin's rules."""
    rng = np.random.default_rng(seed)
    cols = [baseline, *covariates]
    d = df.copy()
    obs = d[y].notna()
    sd_y = float(d.loc[obs, y].std())
    ests, vars_ = [], []
    for _ in range(m):
        yi = d[y].to_numpy(float).copy()
        models = {arm: _draw_model(d[(d[z] == arm) & obs], y, cols, rng) for arm in (0, 1)}
        for arm in (0, 1):
            miss = (d[z] == arm) & ~obs
            if not miss.any():
                continue
            ref_arm = 0 if (method == "j2r" and arm == 1) else arm
            beta, sigma = models[ref_arm]
            X = np.column_stack([np.ones(int(miss.sum()))] + [d.loc[miss, c].to_numpy(float) for c in cols])
            draw = X @ beta + rng.normal(0, sigma, int(miss.sum()))
            if method == "delta" and arm == 1:
                draw = draw - delta * sd_y
            yi[miss.to_numpy()] = draw
        Xd = _design(d, z, baseline, covariates)
        b, se = _ols(yi, Xd)
        ests.append(float(b[1]))
        vars_.append(float(se[1] ** 2))
    out = rubin_combine(ests, vars_)
    out.update({"method": method, "delta": delta, "missing_rate": float((~obs).mean())})
    return out


def tipping_point(df: pd.DataFrame, deltas: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0), y: str = "y", z: str = "z", baseline: str = "baseline", covariates: Sequence[str] = (), m: int = 20, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Delta-adjusted MI over a grid; the tipping point is the smallest delta with p > alpha."""
    rows = []
    for dlt in deltas:
        r = multiple_imputation(df, y, z, baseline, covariates, "delta", dlt, m, seed)
        rows.append({"delta": dlt, "estimate": r["estimate"], "se": r["se"], "p": r["p"], "significant": r["p"] < alpha})
    out = pd.DataFrame(rows)
    out.attrs["tipping_delta"] = float(out.loc[~out["significant"], "delta"].min()) if (~out["significant"]).any() else np.inf
    return out
