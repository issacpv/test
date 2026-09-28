"""Active-comparator new-user cohorts, propensity-score matching and empirical calibration.

Target-trial emulation (Hernán & Robins, 2016 Am J Epidemiol) with an active comparator
(Lund, Richardson & Stürmer, 2015 Curr Epidemiol Rep):

    eligibility  : first administration of drug A or comparator B in the admission, no A or B in the
                   previous ``washout_days`` (across admissions), baseline outcome lab normal
    strategies   : initiate A vs initiate B
    assignment   : as observed; confounding handled by propensity-score matching / IPTW
    outcome      : incident lab-defined event within the window (see ``outcomes.py``)
    follow-up    : index -> event / discharge / death / window end
    estimand     : intention-to-treat-like risk ratio (first exposure), per-protocol as sensitivity

Empirical calibration (Schuemie et al., 2014 Stat Med) uses negative-control drug-outcome pairs
run through the same pipeline to estimate the systematic-error distribution and produce
calibrated p-values / CIs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class TargetTrialSpec:
    """Protocol components, kept next to the code so that the emulation is explicit."""

    treatment: str
    comparator: str
    outcome: str
    washout_days: int = 180
    follow_up_days: int = 7
    eligibility: tuple[str, ...] = ("age >= 18", "baseline lab available within 7 days and normal",
                                    "no exposure to treatment or comparator in washout")
    covariates: tuple[str, ...] = ("age", "sex", "admission_type", "icu_at_index", "baseline_lab",
                                   "egfr", "elixhauser_groups", "n_labs_prior_48h", "concomitant_risk_drugs")
    estimand: str = "risk ratio, first-exposure (ITT-like)"
    negative_controls: tuple[str, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------------------------------
# cohort assembly
# ----------------------------------------------------------------------------------------------


def new_user_cohort(exposures: pd.DataFrame, treatment: str, comparator: str, washout_days: int = 180) -> pd.DataFrame:
    """Active-comparator new users from an exposure table.

    ``exposures`` columns: subject_id, hadm_id, ingredient, index_time, prev_admin_time (NaT if
    none). One row per (subject, hadm, ingredient, admin_rank == 1) is expected; rows for other
    ingredients are ignored. A subject contributes the earliest qualifying admission; if both
    drugs start within the same admission, the earlier one defines the arm and the later one is
    a protocol deviation (kept, flagged ``both_drugs``).
    """
    e = exposures[exposures["ingredient"].isin([treatment, comparator])].copy()
    e["index_time"] = pd.to_datetime(e["index_time"])
    e["prev_admin_time"] = pd.to_datetime(e["prev_admin_time"])
    if "admin_rank" in e:
        e = e[e["admin_rank"] == 1]
    gap = (e["index_time"] - e["prev_admin_time"]).dt.days
    e = e[e["prev_admin_time"].isna() | (gap > washout_days)]
    e = e.sort_values(["subject_id", "index_time"])
    first = e.groupby("subject_id").head(1).copy()
    both = e.groupby(["subject_id", "hadm_id"])["ingredient"].nunique()
    first = first.merge(both.rename("n_drugs").reset_index(), on=["subject_id", "hadm_id"], how="left")
    first["both_drugs"] = first["n_drugs"] > 1
    first["treated"] = (first["ingredient"] == treatment).astype(int)
    return first.drop(columns=["n_drugs"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# propensity scores, matching, weighting, balance
# ----------------------------------------------------------------------------------------------


def propensity_scores(X: np.ndarray | pd.DataFrame, treated: np.ndarray, C: float = 1.0, clip: float = 1e-3) -> np.ndarray:
    """L2 logistic-regression propensity scores (standardised covariates), clipped away from 0/1."""
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    ps = LogisticRegression(C=C, max_iter=2000).fit(Xs, np.asarray(treated).astype(int)).predict_proba(Xs)[:, 1]
    return np.clip(ps, clip, 1 - clip)


def match_nearest(ps: np.ndarray, treated: np.ndarray, caliper_sd: float = 0.2, ratio: int = 1,
                  seed: int = 0) -> pd.DataFrame:
    """Greedy nearest-neighbour matching on the logit PS without replacement.

    Caliper = ``caliper_sd`` x SD of the logit PS (Austin, 2011). Treated units are processed in
    random order; each is matched to up to ``ratio`` unmatched controls within the caliper.
    Returns a frame with columns treated_idx, control_idx, pair_id.
    """
    lp = np.log(ps / (1 - ps))
    t = np.asarray(treated).astype(bool)
    cal = caliper_sd * np.std(lp)
    rng = np.random.default_rng(seed)
    t_idx = rng.permutation(np.flatnonzero(t))
    c_idx = np.flatnonzero(~t)
    order = np.argsort(lp[c_idx])
    c_sorted = c_idx[order]
    c_lp = lp[c_sorted]
    used = np.zeros(len(c_sorted), bool)
    rows = []
    for pid, i in enumerate(t_idx):
        for _ in range(ratio):
            pos = np.searchsorted(c_lp, lp[i])
            best, best_d = -1, np.inf
            for j in (pos - 1, pos, pos + 1):
                # expand outward from the insertion point until an unused control is found
                k = j
                step = -1 if j < pos else 1
                while 0 <= k < len(c_sorted) and used[k]:
                    k += step
                if 0 <= k < len(c_sorted):
                    d = abs(c_lp[k] - lp[i])
                    if d < best_d:
                        best, best_d = k, d
            if best >= 0 and best_d <= cal:
                used[best] = True
                rows.append({"treated_idx": int(i), "control_idx": int(c_sorted[best]), "pair_id": pid})
    return pd.DataFrame(rows, columns=["treated_idx", "control_idx", "pair_id"])


def iptw_weights(ps: np.ndarray, treated: np.ndarray, estimand: str = "ATT", trim: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    """Inverse-probability-of-treatment weights (ATT: treated 1, controls ps/(1-ps))."""
    ps = np.clip(ps, *trim)
    t = np.asarray(treated).astype(bool)
    if estimand == "ATT":
        return np.where(t, 1.0, ps / (1 - ps))
    return np.where(t, 1 / ps, 1 / (1 - ps))


def standardized_mean_differences(X: pd.DataFrame, treated: np.ndarray, weights: np.ndarray | None = None) -> pd.Series:
    """|SMD| per covariate (pooled SD from the unweighted groups); < 0.1 is the usual balance target."""
    t = np.asarray(treated).astype(bool)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float)
    out = {}
    for c in X.columns:
        x = X[c].to_numpy(float)
        m1 = np.average(x[t], weights=w[t])
        m0 = np.average(x[~t], weights=w[~t])
        s = np.sqrt((np.var(x[t], ddof=1) + np.var(x[~t], ddof=1)) / 2)
        out[c] = abs(m1 - m0) / s if s > 0 else 0.0
    return pd.Series(out, name="smd")


# ----------------------------------------------------------------------------------------------
# effect estimation
# ----------------------------------------------------------------------------------------------


def risk_ratio(outcome: np.ndarray, treated: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Weighted risk ratio and risk difference with log-scale Wald CI (weights = matching /
    IPTW; the CI ignores pairing and weight estimation - use :func:`bootstrap_effect` for
    matched or weighted designs)."""
    y = np.asarray(outcome, float)
    t = np.asarray(treated).astype(bool)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    r1 = np.average(y[t], weights=w[t])
    r0 = np.average(y[~t], weights=w[~t])
    n1, n0 = w[t].sum(), w[~t].sum()
    a, c = r1 * n1, r0 * n0
    if a == 0 or c == 0:
        a, c, n1, n0 = a + 0.5, c + 0.5, n1 + 1, n0 + 1
        r1, r0 = a / n1, c / n0
    rr = r1 / r0
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return {"risk_treated": float(r1), "risk_control": float(r0), "rr": float(rr), "log_rr": float(np.log(rr)),
            "se_log_rr": float(se), "rr_lo": float(rr * np.exp(-1.96 * se)), "rr_hi": float(rr * np.exp(1.96 * se)),
            "rd": float(r1 - r0), "n_treated": float(n1), "n_control": float(n0)}


def matched_risk_ratio(outcome: np.ndarray, matches: pd.DataFrame) -> dict[str, float]:
    y = np.asarray(outcome, float)
    idx = np.r_[matches["treated_idx"].to_numpy(), matches["control_idx"].to_numpy()]
    t = np.r_[np.ones(len(matches)), np.zeros(len(matches))]
    return risk_ratio(y[idx], t)


def bootstrap_effect(outcome: np.ndarray, treated: np.ndarray, X: pd.DataFrame, n_boot: int = 200,
                     caliper_sd: float = 0.2, seed: int = 0) -> dict[str, float]:
    """Bootstrap the whole PS + matching + RR pipeline (resampling subjects)."""
    rng = np.random.default_rng(seed)
    y, t = np.asarray(outcome, float), np.asarray(treated).astype(int)
    n = len(y)
    logs = []
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        if t[i].sum() < 5 or (1 - t[i]).sum() < 5:
            continue
        ps = propensity_scores(X.iloc[i], t[i])
        m = match_nearest(ps, t[i], caliper_sd, seed=b)
        if len(m) < 5:
            continue
        logs.append(matched_risk_ratio(y[i], m)["log_rr"])
    logs = np.asarray(logs)
    return {"rr": float(np.exp(np.median(logs))) if len(logs) else np.nan,
            "rr_lo": float(np.exp(np.percentile(logs, 2.5))) if len(logs) else np.nan,
            "rr_hi": float(np.exp(np.percentile(logs, 97.5))) if len(logs) else np.nan, "n_boot_ok": int(len(logs))}


# ----------------------------------------------------------------------------------------------
# empirical calibration (negative controls)
# ----------------------------------------------------------------------------------------------


@dataclass
class SystematicError:
    mu: float
    tau: float

    def calibrated_p(self, log_rr: float, se: float) -> float:
        """Two-sided p-value under the null that the true effect is 0 and the estimate is
        drawn from N(mu, tau^2 + se^2) (Schuemie et al., 2014)."""
        sd = np.sqrt(self.tau ** 2 + se ** 2)
        z = (log_rr - self.mu) / sd
        return float(2 * (1 - stats.norm.cdf(abs(z))))

    def calibrated_ci(self, log_rr: float, se: float) -> tuple[float, float]:
        """Calibrated 95% CI on the RR scale (shift by mu, widen by tau; the simple
        systematic-error model, without the effect-size-dependent extension)."""
        sd = np.sqrt(self.tau ** 2 + se ** 2)
        return (float(np.exp(log_rr - self.mu - 1.96 * sd)), float(np.exp(log_rr - self.mu + 1.96 * sd)))


def fit_systematic_error(log_rr_nc: np.ndarray, se_nc: np.ndarray) -> SystematicError:
    """Fit N(mu, tau^2) to negative-control estimates by maximum likelihood, accounting for
    each estimate's own standard error (true effect assumed 0 for negative controls)."""
    x = np.asarray(log_rr_nc, float)
    s = np.asarray(se_nc, float)
    m = np.isfinite(x) & np.isfinite(s)
    x, s = x[m], s[m]

    def nll(theta: np.ndarray) -> float:
        mu, log_tau = theta
        var = np.exp(2 * log_tau) + s ** 2
        return float(0.5 * np.sum(np.log(2 * np.pi * var) + (x - mu) ** 2 / var))

    res = optimize.minimize(nll, np.array([np.mean(x), np.log(max(np.std(x), 1e-3))]), method="Nelder-Mead")
    return SystematicError(float(res.x[0]), float(np.exp(res.x[1])))


def classify_pair(rr: float, rr_lo: float, rr_hi: float, p_cal: float, pos_rr: float = 1.5, neg_upper: float = 1.25,
                  alpha: float = 0.05) -> str:
    """EHR truth label for the PPV map: positive / negative / indeterminate."""
    if np.isfinite(rr) and rr >= pos_rr and p_cal < alpha:
        return "positive"
    if np.isfinite(rr_hi) and rr_hi < neg_upper:
        return "negative"
    return "indeterminate"
