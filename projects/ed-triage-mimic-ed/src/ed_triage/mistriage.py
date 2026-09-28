"""Outcome-anchored ESI mis-triage definitions and equity metrics.

Definitions (adapted from Sax et al., 2023, JAMA Netw Open, to the fields
available in MIMIC-IV-ED):

* **under-triage**: ESI 3-5 *and* a critical outcome (ICU transfer or death
  within 12 h of ED departure).  A resource-anchored variant additionally
  counts ESI 4-5 with hospital admission.
* **over-triage**: ESI 1-2 *and* discharged from the ED (not admitted, no
  critical outcome).

Equity metrics: subgroup rates with Wilson CIs, crude risk ratios, adjusted
odds ratios from a self-contained IRLS logistic regression with Wald CIs
(``statsmodels`` is optional), equalised-odds gaps, and a *model-vs-nurse*
comparison at a matched over-triage rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def flag_mistriage(df: pd.DataFrame, resource_variant: bool = False) -> pd.DataFrame:
    """Add ``under_triage`` / ``over_triage`` columns (0/1) to a cohort frame with ``acuity``, ``critical_outcome``, ``hospitalization``."""
    d = df.copy()
    acuity = d["acuity"].astype(int)
    crit = d["critical_outcome"].astype(int) == 1
    hosp = d["hospitalization"].astype(int) == 1
    under = (acuity >= 3) & crit
    if resource_variant:
        under = under | ((acuity >= 4) & hosp)
    over = (acuity <= 2) & ~crit & ~hosp
    d["under_triage"] = under.astype(int)
    d["over_triage"] = over.astype(int)
    d["eligible_under"] = (crit).astype(int)          # denominator: critical outcomes
    d["eligible_over"] = (~crit & ~hosp).astype(int)   # denominator: low-acuity-by-outcome visits
    return d


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a proportion; returns (rate, low, high)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(p), float(centre - half), float(centre + half)


def subgroup_rates(df: pd.DataFrame, outcome: str, group: str, denominator: str | None = None) -> pd.DataFrame:
    """Rate of ``outcome`` per level of ``group`` (optionally among rows where ``denominator`` == 1), with Wilson CIs and risk ratios vs the largest level."""
    d = df if denominator is None else df[df[denominator] == 1]
    rows = []
    for lvl, sub in d.groupby(group, dropna=False):
        k, n = int(sub[outcome].sum()), int(len(sub))
        rate, lo, hi = wilson_ci(k, n)
        rows.append({"group": group, "level": lvl, "n": n, "events": k, "rate": rate, "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    ref = out.loc[0, "rate"]
    out["risk_ratio_vs_ref"] = out["rate"] / ref if ref and ref > 0 else np.nan
    out["reference"] = out.index == 0
    return out


# ----------------------------------------------------------------------------
# Adjusted odds ratios: self-contained IRLS logistic regression
# ----------------------------------------------------------------------------
@dataclass
class LogitFit:
    """Coefficients, standard errors and Wald CIs of a logistic regression."""

    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    converged: bool
    n: int

    def table(self, alpha: float = 0.05) -> pd.DataFrame:
        z = stats.norm.ppf(1 - alpha / 2)
        return pd.DataFrame({
            "term": self.names, "coef": self.coef, "se": self.se,
            "OR": np.exp(self.coef), "OR_low": np.exp(self.coef - z * self.se), "OR_high": np.exp(self.coef + z * self.se),
            "p": 2 * stats.norm.sf(np.abs(self.coef / self.se)),
        })


def fit_logit(X: np.ndarray, y: np.ndarray, names: list[str] | None = None, cluster: np.ndarray | None = None,
              max_iter: int = 100, tol: float = 1e-8, ridge: float = 1e-8) -> LogitFit:
    """IRLS logistic regression with an intercept; optional cluster-robust (sandwich) SEs by ``cluster`` id."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    Xd = np.c_[np.ones(n), X]
    names = ["intercept"] + (names or [f"x{i}" for i in range(p)])
    beta = np.zeros(p + 1)
    converged = False
    for _ in range(max_iter):
        eta = Xd @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        H = Xd.T @ (Xd * w[:, None]) + ridge * np.eye(p + 1)
        g = Xd.T @ (y - mu)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    eta = Xd @ beta
    mu = 1 / (1 + np.exp(-eta))
    w = mu * (1 - mu)
    Hinv = np.linalg.inv(Xd.T @ (Xd * w[:, None]) + ridge * np.eye(p + 1))
    if cluster is None:
        cov = Hinv
    else:
        cl = np.asarray(cluster)
        S = Xd * (y - mu)[:, None]
        meat = np.zeros((p + 1, p + 1))
        for c in np.unique(cl):
            sc = S[cl == c].sum(axis=0)
            meat += np.outer(sc, sc)
        G = len(np.unique(cl))
        cov = Hinv @ meat @ Hinv * (G / max(G - 1, 1))
    return LogitFit(names, beta, np.sqrt(np.diag(cov)), converged, n)


def design_matrix(df: pd.DataFrame, numeric: list[str], categorical: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Numeric columns (median-imputed, standardised) + one-hot categoricals with named reference levels."""
    cols, names = [], []
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        v = v.fillna(v.median())
        sd = v.std() if v.std() > 0 else 1.0
        cols.append(((v - v.mean()) / sd).to_numpy())
        names.append(c)
    for c, ref in categorical.items():
        s = df[c].astype(str)
        for lvl in sorted(s.unique()):
            if lvl == ref:
                continue
            cols.append((s == lvl).astype(float).to_numpy())
            names.append(f"{c}={lvl}")
    return np.column_stack(cols) if cols else np.zeros((len(df), 0)), names


def adjusted_odds_ratios(df: pd.DataFrame, outcome: str, exposure: str, reference: str,
                         covariates_numeric: list[str], covariates_categorical: dict[str, str] | None = None,
                         cluster: str | None = "subject_id") -> pd.DataFrame:
    """Adjusted OR table for ``exposure`` levels vs ``reference`` on ``outcome`` (rows restricted to non-missing exposure)."""
    d = df[df[exposure].notna()].copy()
    cats = {exposure: reference, **(covariates_categorical or {})}
    X, names = design_matrix(d, covariates_numeric, cats)
    fit = fit_logit(X, d[outcome].to_numpy(), names, cluster=d[cluster].to_numpy() if cluster else None)
    tab = fit.table()
    return tab[tab["term"].str.startswith(f"{exposure}=")].reset_index(drop=True)


def e_value(or_: float) -> float:
    """VanderWeele & Ding E-value for an odds ratio of a rare outcome (approximate when outcome is common)."""
    rr = or_ if or_ >= 1 else 1 / or_
    return float(rr + np.sqrt(rr * (rr - 1))) if rr > 1 else 1.0


# ----------------------------------------------------------------------------
# Model-vs-nurse comparison
# ----------------------------------------------------------------------------
def equalized_odds_gaps(y: np.ndarray, pred: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """TPR and FPR gaps (max-min over group levels) of a binary decision ``pred``."""
    y, pred, group = np.asarray(y).astype(int), np.asarray(pred).astype(int), np.asarray(group)
    tpr, fpr = [], []
    for g in pd.unique(group):
        m = group == g
        if (y[m] == 1).any():
            tpr.append(pred[m][y[m] == 1].mean())
        if (y[m] == 0).any():
            fpr.append(pred[m][y[m] == 0].mean())
    return {"tpr_gap": float(max(tpr) - min(tpr)) if len(tpr) > 1 else np.nan,
            "fpr_gap": float(max(fpr) - min(fpr)) if len(fpr) > 1 else np.nan}


def threshold_matching_overtriage(scores: np.ndarray, y: np.ndarray, nurse_high: np.ndarray) -> float:
    """Score threshold such that the model flags 'high acuity' with the same false-positive (over-triage) rate as ESI 1-2."""
    y, nurse_high = np.asarray(y).astype(int), np.asarray(nurse_high).astype(int)
    target_fpr = nurse_high[y == 0].mean() if (y == 0).any() else 0.0
    neg = np.sort(np.asarray(scores)[y == 0])[::-1]
    k = int(round(target_fpr * len(neg)))
    if k <= 0:
        return float(neg[0] + 1e-9) if len(neg) else 1.0
    return float(neg[min(k, len(neg)) - 1])


def compare_model_vs_nurse(df: pd.DataFrame, scores: np.ndarray, group: str, outcome: str = "critical_outcome") -> pd.DataFrame:
    """Under-triage rate per subgroup for nurse ESI (1-2 = high) vs a model at the matched over-triage rate."""
    y = df[outcome].to_numpy().astype(int)
    nurse_high = (df["acuity"].astype(int) <= 2).to_numpy().astype(int)
    thr = threshold_matching_overtriage(scores, y, nurse_high)
    model_high = (np.asarray(scores) >= thr).astype(int)
    rows = []
    for g, idx in df.groupby(group, dropna=False).indices.items():
        pos = y[idx] == 1
        if pos.sum() == 0:
            continue
        rows.append({
            "level": g, "n_critical": int(pos.sum()),
            "nurse_under_triage": float(1 - nurse_high[idx][pos].mean()),
            "model_under_triage": float(1 - model_high[idx][pos].mean()),
        })
    out = pd.DataFrame(rows)
    out.attrs["threshold"] = thr
    out.attrs["nurse_eo"] = equalized_odds_gaps(y, nurse_high, df[group].to_numpy())
    out.attrs["model_eo"] = equalized_odds_gaps(y, model_high, df[group].to_numpy())
    return out


def permutation_null_gap(y: np.ndarray, pred: np.ndarray, group: np.ndarray, strata: np.ndarray,
                         n_perm: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Observed TPR gap and permutation p-value, shuffling group labels *within* strata (e.g. ESI level)."""
    rng = np.random.default_rng(seed)
    group = np.asarray(group).copy()
    obs = equalized_odds_gaps(y, pred, group)["tpr_gap"]
    strata = np.asarray(strata)
    count = 0
    for _ in range(n_perm):
        g = group.copy()
        for s in np.unique(strata):
            m = np.where(strata == s)[0]
            g[m] = g[rng.permutation(m)]
        if equalized_odds_gaps(y, pred, g)["tpr_gap"] >= obs:
            count += 1
    return float(obs), float((count + 1) / (n_perm + 1))
