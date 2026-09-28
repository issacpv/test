"""Engagement-outcome analyses that respect randomisation.

All estimators take a tidy IPD frame with one row per participant and return
a dict with ``estimate``, ``se`` (where analytic), and bootstrap CIs via
:func:`bootstrap_estimates`.  The outcome model everywhere is ANCOVA
(``y ~ z + baseline [+ covariates]``) on complete cases; missing-data
sensitivity lives in :mod:`dh_ipd.missing`.

* :func:`itt` - intention-to-treat effect.
* :func:`naive_per_protocol` - the two comparisons papers usually report:
  treated engagers vs controls, and engagers vs non-engagers within the
  treated arm.  Both are confounded by whatever drives engagement.
* :func:`cace_wald` - complier average causal effect for one-sided
  non-compliance (controls cannot engage): ITT / P(engaged | treated).
* :func:`principal_score_effects` - stratum-specific effects under principal
  ignorability (Jo & Stuart, 2009; Ding & Lu, 2017): a model for P(S = s | X)
  is fitted in the treated arm, where S is observed, and used to re-weight
  controls to each stratum's covariate distribution.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression


def _ols(y: np.ndarray, X: np.ndarray, w: Optional[np.ndarray] = None):
    """Weighted least squares; returns (beta, se) with HC0 robust SEs."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    beta = np.linalg.solve(XtWX + 1e-10 * np.eye(X.shape[1]), Xw.T @ y)
    resid = y - X @ beta
    meat = (X * (w * resid)[:, None]).T @ (X * (w * resid)[:, None])
    bread = np.linalg.inv(XtWX + 1e-10 * np.eye(X.shape[1]))
    cov = bread @ meat @ bread
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def _design(df: pd.DataFrame, z: str, baseline: str, covariates: Sequence[str]) -> np.ndarray:
    cols = [df[z].to_numpy(float), df[baseline].to_numpy(float)] + [df[c].to_numpy(float) for c in covariates]
    return np.column_stack([np.ones(len(df))] + cols)


def itt(df: pd.DataFrame, y: str = "y", z: str = "z", baseline: str = "baseline", covariates: Sequence[str] = ()) -> Dict[str, float]:
    """ANCOVA intention-to-treat estimate on complete cases."""
    d = df.dropna(subset=[y])
    X = _design(d, z, baseline, covariates)
    beta, se = _ols(d[y].to_numpy(float), X)
    est, s = float(beta[1]), float(se[1])
    return {"estimate": est, "se": s, "ci_low": est - 1.96 * s, "ci_high": est + 1.96 * s, "p": float(2 * stats.norm.sf(abs(est / s))) if s > 0 else np.nan, "n": float(len(d))}


def naive_per_protocol(df: pd.DataFrame, y: str = "y", z: str = "z", engaged: str = "engaged", baseline: str = "baseline", covariates: Sequence[str] = ()) -> Dict[str, float]:
    """Engagers-vs-controls and engagers-vs-non-engagers (within treated) ANCOVA contrasts."""
    d = df.dropna(subset=[y])
    a = d[(d[z] == 0) | ((d[z] == 1) & (d[engaged] == 1))]
    beta_a, se_a = _ols(a[y].to_numpy(float), _design(a, z, baseline, covariates))
    tr = d[d[z] == 1].copy()
    tr["_e"] = tr[engaged].astype(float)
    beta_b, se_b = _ols(tr[y].to_numpy(float), _design(tr, "_e", baseline, covariates))
    return {
        "engagers_vs_controls": float(beta_a[1]),
        "engagers_vs_controls_se": float(se_a[1]),
        "engagers_vs_nonengagers": float(beta_b[1]),
        "engagers_vs_nonengagers_se": float(se_b[1]),
        "engagement_rate_treated": float(df.loc[df[z] == 1, engaged].mean()),
    }


def cace_wald(df: pd.DataFrame, y: str = "y", z: str = "z", engaged: str = "engaged", baseline: str = "baseline", covariates: Sequence[str] = ()) -> Dict[str, float]:
    """Wald / IV estimate of the effect among would-be engagers (one-sided non-compliance).

    Uses the ANCOVA ITT numerator and the engagement rate in the treated arm as
    denominator.  Controls have no access, so there are no "always-takers";
    the remaining exclusion restriction - that being *offered* the app without
    engaging has no effect - is the substantive assumption, and it fails for
    partial engagement (see ``simulate_ipd(nonengager_effect=...)``).  The
    principal-score estimator does not need it.  Delta-method SE treats the
    denominator as estimated.
    """
    it = itt(df, y, z, baseline, covariates)
    tr = df[df[z] == 1]
    pc = float(tr[engaged].mean())
    n_tr = len(tr)
    if pc <= 0:
        return {"estimate": np.nan, "se": np.nan, "compliance": pc}
    est = it["estimate"] / pc
    var = (it["se"] ** 2) / pc**2 + (it["estimate"] ** 2) * pc * (1 - pc) / (n_tr * pc**4)
    se = float(np.sqrt(var))
    return {"estimate": float(est), "se": se, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se, "compliance": pc, "itt": it["estimate"]}


def principal_score_effects(
    df: pd.DataFrame,
    y: str = "y",
    z: str = "z",
    stratum: str = "engaged",
    covariates: Sequence[str] = ("baseline", "age", "sex"),
    baseline: str = "baseline",
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Stratum-specific effects by principal-score weighting.

    ``stratum`` must be observed in the treated arm (integer levels, e.g. 0/1 or
    0/1/2 dose tertiles) and may be anything in controls (ignored).  Returns one
    row per stratum with the effect, its bootstrap CI and the stratum share.
    """
    d = df.dropna(subset=[y]).reset_index(drop=True)
    tr, co = d[d[z] == 1], d[d[z] == 0]
    levels = sorted(tr[stratum].astype(int).unique().tolist())
    Xtr = tr[list(covariates)].to_numpy(float)
    Xco = co[list(covariates)].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9

    def one(tr_: pd.DataFrame, co_: pd.DataFrame) -> Dict[int, float]:
        out = {}
        if len(levels) == 1:
            return {levels[0]: float(tr_[y].mean() - co_[y].mean())}
        ps = LogisticRegression(max_iter=1000)
        Xt = (tr_[list(covariates)].to_numpy(float) - mu) / sd
        Xc = (co_[list(covariates)].to_numpy(float) - mu) / sd
        ps.fit(Xt, tr_[stratum].astype(int))
        e = ps.predict_proba(Xc)  # P(S = s | X) for controls
        for j, s in enumerate(ps.classes_):
            w = e[:, j]
            # regression-adjusted contrast: treated in stratum s vs weighted controls, both adjusted for baseline
            yt = tr_.loc[tr_[stratum].astype(int) == s, y].to_numpy(float)
            bt = tr_.loc[tr_[stratum].astype(int) == s, baseline].to_numpy(float)
            yc = co_[y].to_numpy(float)
            bc = co_[baseline].to_numpy(float)
            # ANCOVA-style adjustment with a pooled baseline slope
            ypool = np.r_[yt, yc]
            bpool = np.r_[bt, bc]
            wpool = np.r_[np.ones(len(yt)), w * len(yt) / max(w.sum(), 1e-9)]
            zpool = np.r_[np.ones(len(yt)), np.zeros(len(yc))]
            X = np.column_stack([np.ones(len(ypool)), zpool, bpool])
            beta, _ = _ols(ypool, X, wpool)
            out[int(s)] = float(beta[1])
        return out

    point = one(tr, co)
    rng = np.random.default_rng(seed)
    boots: Dict[int, List[float]] = {s: [] for s in levels}
    for _ in range(n_boot):
        tb = tr.iloc[rng.integers(0, len(tr), len(tr))]
        cb = co.iloc[rng.integers(0, len(co), len(co))]
        try:
            res = one(tb, cb)
        except ValueError:
            continue
        for s in levels:
            if s in res:
                boots[s].append(res[s])
    rows = []
    for s in levels:
        b = np.asarray(boots[s])
        rows.append(
            {
                "stratum": s,
                "effect": point.get(s, np.nan),
                "ci_low": float(np.quantile(b, 0.025)) if len(b) else np.nan,
                "ci_high": float(np.quantile(b, 0.975)) if len(b) else np.nan,
                "share_treated": float((tr[stratum].astype(int) == s).mean()),
                "n_treated": int((tr[stratum].astype(int) == s).sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_estimates(fn: Callable[[pd.DataFrame], Dict[str, float]], df: pd.DataFrame, key: str = "estimate", n_boot: int = 500, seed: int = 0, cluster: Optional[str] = "trial") -> Dict[str, float]:
    """Cluster (trial) bootstrap CI for any estimator returning a dict with ``key``."""
    rng = np.random.default_rng(seed)
    vals = []
    if cluster and cluster in df.columns:
        clusters = df[cluster].unique()
        groups = {c: df[df[cluster] == c] for c in clusters}
        for _ in range(n_boot):
            pick = rng.choice(clusters, len(clusters), replace=True)
            b = pd.concat([groups[c] for c in pick], ignore_index=True)
            try:
                vals.append(fn(b)[key])
            except (ValueError, KeyError, np.linalg.LinAlgError):
                continue
    else:
        for _ in range(n_boot):
            b = df.iloc[rng.integers(0, len(df), len(df))]
            try:
                vals.append(fn(b)[key])
            except (ValueError, KeyError, np.linalg.LinAlgError):
                continue
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return {"estimate": fn(df)[key], "ci_low": float(np.quantile(v, 0.025)), "ci_high": float(np.quantile(v, 0.975)), "n_boot": float(len(v))}
