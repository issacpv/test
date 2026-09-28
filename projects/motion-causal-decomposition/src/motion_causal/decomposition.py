"""Causal decomposition of the FC -> phenotype association with respect to head motion.

Notation: subject *i*, run *r*; ``yhat_ir`` the run-level prediction of a model that was not
trained on subject *i*; ``m_ir`` the run's mean FD; ``y_i`` the phenotype.

1. Artifact sensitivity ``beta_m = d yhat / d m`` is identified from *within-subject* run-to-run
   variation (subject fixed effects), optionally with an instrument (run order / session) in a
   two-stage least-squares estimator: the instrument shifts motion but cannot change cognition.
2. Decomposition::

       Cov(yhat_i, y_i) = Cov(yhat_i - beta_m * m_i, y_i)  +  beta_m * Cov(m_i, y_i)
                          \_________ non-artifact ________/     \____ artifact-mediated ____/

   The non-artifact part is split further with a negative-control exposure (motion from an
   acquisition that cannot corrupt the FC used): the part of yhat explained by control motion
   is trait confounding; the remainder is the "clean" component.
3. :func:`robustness_value` gives the Cinelli-Hazlett (2020) robustness value of a coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class SensitivityResult:
    beta_m: float
    se: float
    n_subjects: int
    n_runs_used: int
    first_stage_f: Optional[float] = None
    method: str = "fe"

    @property
    def ci95(self):
        return (self.beta_m - 1.96 * self.se, self.beta_m + 1.96 * self.se)


def _demean_within(A: np.ndarray, subj: np.ndarray) -> np.ndarray:
    out = A.astype(float).copy()
    for s in np.unique(subj):
        idx = subj == s
        out[idx] -= np.nanmean(A[idx], axis=0)
    return out


def _cluster_se(x: np.ndarray, resid: np.ndarray, subj: np.ndarray) -> float:
    sxx = float((x ** 2).sum())
    meat = 0.0
    for s in np.unique(subj):
        idx = subj == s
        meat += float((x[idx] * resid[idx]).sum()) ** 2
    G = len(np.unique(subj))
    return float(np.sqrt(meat) / sxx * np.sqrt(G / max(G - 1, 1)))


def artifact_sensitivity(
    yhat_runs: np.ndarray,
    motion_runs: np.ndarray,
    instrument_runs: Optional[np.ndarray] = None,
) -> SensitivityResult:
    """Within-subject slope of run-level predictions on run-level motion.

    Parameters
    ----------
    yhat_runs, motion_runs
        ``(n_subjects, n_runs)``; NaN entries are dropped pairwise.
    instrument_runs
        Optional ``(n_subjects, n_runs)`` instrument (e.g. run index 0..R-1, or session 0/1).
        If given, a fixed-effects 2SLS estimate is returned together with the first-stage F.
    """
    Y = np.asarray(yhat_runs, float)
    M = np.asarray(motion_runs, float)
    n, R = Y.shape
    subj = np.repeat(np.arange(n), R)
    y, m = Y.ravel(), M.ravel()
    z = None if instrument_runs is None else np.asarray(instrument_runs, float).ravel()
    ok = np.isfinite(y) & np.isfinite(m) & (np.isfinite(z) if z is not None else True)
    y, m, subj = y[ok], m[ok], subj[ok]
    if z is not None:
        z = z[ok]
    # keep subjects with >= 2 usable runs
    counts = np.bincount(subj, minlength=n)
    keep = counts[subj] >= 2
    y, m, subj = y[keep], m[keep], subj[keep]
    if z is not None:
        z = z[keep]
    yd, md = _demean_within(y, subj), _demean_within(m, subj)
    if z is None:
        sxx = float((md ** 2).sum())
        if sxx <= 0:
            raise ValueError("no within-subject motion variation")
        beta = float((md * yd).sum() / sxx)
        resid = yd - beta * md
        se = _cluster_se(md, resid, subj)
        return SensitivityResult(beta, se, int(len(np.unique(subj))), int(len(y)), None, "fe")
    zd = _demean_within(z, subj)
    szz = float((zd ** 2).sum())
    if szz <= 0:
        raise ValueError("instrument has no within-subject variation")
    pi = float((zd * md).sum() / szz)
    m_fit = pi * zd
    res1 = md - m_fit
    G = len(np.unique(subj))
    f_stat = float(pi ** 2 * szz / (res1.var(ddof=1) if len(res1) > 2 else np.inf))
    beta = float((m_fit * yd).sum() / (m_fit ** 2).sum())
    resid = yd - beta * md  # structural residual
    se = _cluster_se(m_fit, resid, subj)
    return SensitivityResult(beta, se, int(G), int(len(y)), f_stat, "2sls")


@dataclass
class Decomposition:
    cov_total: float
    cov_artifact: float
    cov_nonartifact: float
    cov_trait: float = np.nan
    cov_clean: float = np.nan
    r_total: float = np.nan
    shares: Dict[str, float] = field(default_factory=dict)


def decompose_association(
    yhat: np.ndarray,
    y: np.ndarray,
    motion: np.ndarray,
    beta_m: float,
    control_motion: Optional[np.ndarray] = None,
) -> Decomposition:
    """Split ``Cov(yhat, y)`` into artifact-mediated, trait-confounded and clean components.

    ``motion`` is the subject-level motion of the runs used for ``yhat`` (mean FD); ``beta_m``
    the artifact sensitivity from :func:`artifact_sensitivity`. ``control_motion`` is the
    negative-control exposure (motion from another acquisition). The trait component is the
    covariance with ``y`` of the part of the artifact-free prediction linearly explained by the
    control motion; the clean component is the remainder.
    """
    yhat, y, motion = (np.asarray(a, float) for a in (yhat, y, motion))
    ok = np.isfinite(yhat) & np.isfinite(y) & np.isfinite(motion)
    if control_motion is not None:
        control_motion = np.asarray(control_motion, float)
        ok &= np.isfinite(control_motion)
    yhat, y, motion = yhat[ok], y[ok], motion[ok]
    cov = lambda a, b: float(np.cov(a, b, ddof=1)[0, 1])  # noqa: E731
    total = cov(yhat, y)
    artifact = beta_m * cov(motion, y)
    clean_pred = yhat - beta_m * motion
    nonart = cov(clean_pred, y)
    dec = Decomposition(total, artifact, nonart, r_total=float(np.corrcoef(yhat, y)[0, 1]))
    if control_motion is not None:
        c = control_motion[ok]
        cc = c - c.mean()
        b = float((cc * (clean_pred - clean_pred.mean())).sum() / (cc ** 2).sum()) if (cc ** 2).sum() > 0 else 0.0
        trait_pred = b * cc
        dec.cov_trait = cov(trait_pred, y)
        dec.cov_clean = cov(clean_pred - trait_pred, y)
    denom = total if total != 0 else np.nan
    dec.shares = {
        "artifact": artifact / denom,
        "nonartifact": nonart / denom,
        "trait": dec.cov_trait / denom,
        "clean": dec.cov_clean / denom,
    }
    return dec


def negative_control_exposure(
    yhat: np.ndarray,
    motion_primary: np.ndarray,
    motion_control: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compare the effect of same-run motion and of a negative-control motion on ``yhat``.

    Fits ``yhat ~ motion_primary + motion_control (+ covariates)`` by OLS. Under the negative-
    control logic, ``motion_control`` (which cannot corrupt the FC) captures the trait path; the
    excess of the primary coefficient over the control coefficient (both standardised to the
    same motion units) estimates the artifact path. Returns the two coefficients, their
    difference and HC1 standard errors.
    """
    yhat, mp, mc = (np.asarray(a, float) for a in (yhat, motion_primary, motion_control))
    ok = np.isfinite(yhat) & np.isfinite(mp) & np.isfinite(mc)
    cols = [np.ones(ok.sum()), mp[ok], mc[ok]]
    if covariates is not None:
        C = np.asarray(covariates, float)
        C = C[:, None] if C.ndim == 1 else C
        cols.extend(C[ok].T)
    X = np.column_stack(cols)
    yv = yhat[ok]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ yv
    e = yv - X @ beta
    n, k = X.shape
    V = XtX_inv @ (X.T @ (X * (e ** 2)[:, None])) @ XtX_inv * n / max(n - k, 1)
    se = np.sqrt(np.diag(V))
    diff_var = V[1, 1] + V[2, 2] - 2 * V[1, 2]
    return {
        "beta_primary": float(beta[1]),
        "se_primary": float(se[1]),
        "beta_control": float(beta[2]),
        "se_control": float(se[2]),
        "artifact_path": float(beta[1] - beta[2]),
        "se_artifact_path": float(np.sqrt(max(diff_var, 0))),
        "n": int(n),
    }


def robustness_value(t_stat: float, dof: int, q: float = 1.0) -> float:
    """Cinelli and Hazlett (2020) robustness value ``RV_q``.

    The minimum strength of association (partial R^2) an unmeasured confounder would need with
    both the treatment and the outcome to reduce the estimate by a fraction ``q`` (``q = 1``:
    to zero). ``f = |t| / sqrt(dof)`` is the partial Cohen's f of the coefficient.
    """
    f = q * abs(t_stat) / np.sqrt(dof)
    return float(0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2))


def e_value(rr: float) -> float:
    """VanderWeele and Ding (2017) E-value for a risk ratio ``rr >= 1`` (inverted if < 1)."""
    rr = rr if rr >= 1 else 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))
