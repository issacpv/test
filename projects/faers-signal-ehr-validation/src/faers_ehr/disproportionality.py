"""Disproportionality statistics with confidence / credibility intervals.

Given the 2x2 table for one drug-event pair

                 event     not event
    drug           a           b
    not drug       c           d

* PRR  = [a/(a+b)] / [c/(c+d)],  SE(ln PRR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
  (Evans, Waller & Davis, 2001) with the Yates-corrected chi-square; signal if PRR >= 2,
  chi2 >= 4 and a >= 3.
* ROR  = (a/b) / (c/d),  SE(ln ROR) = sqrt(1/a + 1/b + 1/c + 1/d)  (Rothman et al., 2004).
* IC   = log2((a + 0.5) / (E + 0.5)),  E = (a+b)(a+c)/n, with the 2.5% credibility bound from
  the Gamma(a + 0.5, rate E + 0.5) posterior (Norén, Hopstadius & Bate, 2013). Signal if IC025 > 0.
* EBGM / EB05 from the multi-item gamma-Poisson shrinker (DuMouchel, 1999): lambda_ij ~ mixture
  of two gammas, hyper-parameters fitted by marginal likelihood over the whole drug x event table
  (:func:`fit_mgps_prior`), posterior geometric mean and 5th percentile per cell. Signal if EB05 >= 2.

:func:`signal_table` computes everything for a DataFrame of 2x2 rows; :func:`signal_performance`
scores signal statistics against an external truth (e.g. EHR-confirmed pairs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

_Z = 1.959963984540054


# ----------------------------------------------------------------------------------------------
# frequentist
# ----------------------------------------------------------------------------------------------


def prr(a: float, b: float, c: float, d: float) -> dict[str, float]:
    a, b, c, d = map(float, (a, b, c, d))
    if a == 0 or c == 0 or (a + b) == 0 or (c + d) == 0:
        return {"prr": np.nan, "prr_lo": np.nan, "prr_hi": np.nan, "chi2": np.nan}
    val = (a / (a + b)) / (c / (c + d))
    se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    n = a + b + c + d
    expected = (a + b) * (a + c) / n
    # Yates-corrected chi-square on the 2x2
    num = (abs(a * d - b * c) - n / 2) ** 2 * n
    den = (a + b) * (c + d) * (a + c) * (b + d)
    chi2 = num / den if den > 0 else np.nan
    return {"prr": val, "prr_lo": val * np.exp(-_Z * se), "prr_hi": val * np.exp(_Z * se), "chi2": chi2,
            "expected": expected}


def ror(a: float, b: float, c: float, d: float) -> dict[str, float]:
    a, b, c, d = map(float, (a, b, c, d))
    if min(a, b, c, d) == 0:
        # Haldane-Anscombe correction keeps the estimate finite for sparse cells
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    val = (a / b) / (c / d)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {"ror": val, "ror_lo": val * np.exp(-_Z * se), "ror_hi": val * np.exp(_Z * se)}


# ----------------------------------------------------------------------------------------------
# BCPNN information component
# ----------------------------------------------------------------------------------------------


def bcpnn_ic(a: float, expected: float, shrinkage: float = 0.5) -> dict[str, float]:
    """IC with the Norén et al. (2013) shrinkage and exact gamma credibility bounds."""
    a, e = float(a), float(expected)
    ic = np.log2((a + shrinkage) / (e + shrinkage))
    post = stats.gamma(a + shrinkage, scale=1.0 / (e + shrinkage))
    return {"ic": ic, "ic025": np.log2(post.ppf(0.025)), "ic975": np.log2(post.ppf(0.975))}


# ----------------------------------------------------------------------------------------------
# MGPS / EBGM
# ----------------------------------------------------------------------------------------------


@dataclass
class MGPSPrior:
    """Two-component gamma mixture prior on lambda = observed/expected (shape, rate)."""

    alpha1: float = 0.2
    beta1: float = 0.1
    alpha2: float = 2.0
    beta2: float = 4.0
    p: float = 1 / 3

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha1, self.beta1, self.alpha2, self.beta2, self.p])


def _nb_logpmf(n: np.ndarray, e: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """log P(N = n | E = e) when lambda ~ Gamma(alpha, rate beta): negative binomial."""
    return (special.gammaln(alpha + n) - special.gammaln(alpha) - special.gammaln(n + 1)
            + alpha * np.log(beta / (beta + e)) + n * np.log(e / (beta + e)))


def _mixture_loglik(theta: np.ndarray, n: np.ndarray, e: np.ndarray) -> float:
    a1, b1, a2, b2 = np.exp(theta[:4])
    p = special.expit(theta[4])
    l1 = np.log(p) + _nb_logpmf(n, e, a1, b1)
    l2 = np.log1p(-p) + _nb_logpmf(n, e, a2, b2)
    return float(np.sum(np.logaddexp(l1, l2)))


def fit_mgps_prior(observed: np.ndarray, expected: np.ndarray, init: MGPSPrior | None = None,
                   min_expected: float = 1e-6) -> MGPSPrior:
    """Maximum marginal likelihood fit of the 5 MGPS hyper-parameters over all cells.

    ``observed`` / ``expected`` are the full drug x event table (including zero cells, which
    carry information about the prior). Optimisation is on log(alpha, beta) and logit(p).
    """
    n = np.asarray(observed, float)
    e = np.maximum(np.asarray(expected, float), min_expected)
    init = init or MGPSPrior()
    theta0 = np.r_[np.log(init.as_array()[:4]), special.logit(init.p)]
    res = optimize.minimize(lambda th: -_mixture_loglik(th, n, e), theta0, method="L-BFGS-B",
                            bounds=[(-8, 8)] * 4 + [(-8, 8)])
    a1, b1, a2, b2 = np.exp(res.x[:4])
    return MGPSPrior(float(a1), float(b1), float(a2), float(b2), float(special.expit(res.x[4])))


def ebgm(a: float, expected: float, prior: MGPSPrior) -> dict[str, float]:
    """Posterior geometric mean (EBGM), 5th and 95th percentiles (EB05, EB95) of lambda."""
    n, e = float(a), max(float(expected), 1e-6)
    l1 = np.log(prior.p) + _nb_logpmf(np.array(n), np.array(e), prior.alpha1, prior.beta1)
    l2 = np.log1p(-prior.p) + _nb_logpmf(np.array(n), np.array(e), prior.alpha2, prior.beta2)
    q = float(np.exp(l1 - np.logaddexp(l1, l2)))  # posterior weight of component 1
    g1 = stats.gamma(prior.alpha1 + n, scale=1.0 / (prior.beta1 + e))
    g2 = stats.gamma(prior.alpha2 + n, scale=1.0 / (prior.beta2 + e))
    e_log = q * (special.digamma(prior.alpha1 + n) - np.log(prior.beta1 + e)) + \
        (1 - q) * (special.digamma(prior.alpha2 + n) - np.log(prior.beta2 + e))

    def cdf(x: float) -> float:
        return q * g1.cdf(x) + (1 - q) * g2.cdf(x)

    def quantile(level: float) -> float:
        hi = max(g1.ppf(0.999), g2.ppf(0.999), 1.0)
        return float(optimize.brentq(lambda x: cdf(x) - level, 1e-12, hi))

    return {"ebgm": float(np.exp(e_log)), "eb05": quantile(0.05), "eb95": quantile(0.95), "post_w1": q}


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------


def expected_counts(table: pd.DataFrame) -> pd.Series:
    """E = (a+b)(a+c)/n per row of a 2x2 table frame (columns a, b, c, d or a, b, c, n)."""
    n = table["n"] if "n" in table else table[["a", "b", "c", "d"]].sum(axis=1)
    return (table["a"] + table["b"]) * (table["a"] + table["c"]) / n


def stratified_expected(long: pd.DataFrame, drug: str = "drug", event: str = "event", stratum: str = "stratum",
                        count: str = "n") -> pd.DataFrame:
    """Mantel-Haenszel-style expected counts summed over strata (e.g. sex x age x year).

    ``long`` has one row per (drug, event, stratum) with the report count; drug-marginals,
    event-marginals and totals are computed within stratum and E_ij = sum_k n_i.k n_.jk / n..k.
    Returns a frame with columns drug, event, observed, expected.
    """
    d = long.copy()
    d["n_i"] = d.groupby([stratum, drug])[count].transform("sum")
    d["n_j"] = d.groupby([stratum, event])[count].transform("sum")
    d["n_k"] = d.groupby(stratum)[count].transform("sum")
    d["e"] = d["n_i"] * d["n_j"] / d["n_k"]
    out = d.groupby([drug, event], as_index=False).agg(observed=(count, "sum"), expected=("e", "sum"))
    return out


def signal_table(table: pd.DataFrame, prior: MGPSPrior | None = None, fit_prior: bool = True) -> pd.DataFrame:
    """Add PRR/ROR/IC/EBGM columns and signal flags to a frame with columns a, b, c, d (and n)."""
    t = table.copy()
    if "n" not in t:
        t["n"] = t[["a", "b", "c", "d"]].sum(axis=1)
    t["expected"] = expected_counts(t)
    if prior is None:
        prior = fit_mgps_prior(t["a"].to_numpy(), t["expected"].to_numpy()) if fit_prior and len(t) >= 10 else MGPSPrior()
    rows = []
    for r in t.itertuples(index=False):
        row = {}
        row.update(prr(r.a, r.b, r.c, r.d))
        row.update(ror(r.a, r.b, r.c, r.d))
        row.update(bcpnn_ic(r.a, r.expected))
        row.update(ebgm(r.a, r.expected, prior))
        rows.append(row)
    stats_df = pd.DataFrame(rows, index=t.index).drop(columns=["expected"], errors="ignore")
    t = pd.concat([t, stats_df], axis=1)
    t["signal_prr"] = (t["prr"] >= 2) & (t["chi2"] >= 4) & (t["a"] >= 3)
    t["signal_ror"] = (t["ror_lo"] > 1) & (t["a"] >= 3)
    t["signal_ic"] = t["ic025"] > 0
    t["signal_ebgm"] = t["eb05"] >= 2
    t.attrs["mgps_prior"] = prior
    return t


# ----------------------------------------------------------------------------------------------
# scoring signals against an external truth
# ----------------------------------------------------------------------------------------------


def signal_performance(scores: pd.Series, truth: pd.Series, flags: pd.Series | None = None) -> dict[str, float]:
    """PPV / sensitivity / specificity of a binary flag, and AUROC / average precision of the
    continuous statistic, against a binary truth (1 = EHR-confirmed, 0 = EHR-refuted; NaN =
    indeterminate rows are dropped)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    m = truth.notna() & scores.notna()
    y = truth[m].astype(int).to_numpy()
    s = scores[m].to_numpy(float)
    out: dict[str, float] = {"n_pairs": int(m.sum()), "n_positive": int(y.sum())}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, s))
        out["average_precision"] = float(average_precision_score(y, s))
    if flags is not None:
        f = flags[m].astype(bool).to_numpy()
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
        out.update(ppv=tp / (tp + fp) if tp + fp else np.nan, sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                   specificity=tn / (tn + fp) if tn + fp else np.nan, n_signals=tp + fp)
        if tp + fp:
            lo, hi = wilson_interval(tp, tp + fp)
            out["ppv_lo"], out["ppv_hi"] = lo, hi
    return out


def wilson_interval(k: int, n: int, z: float = _Z) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))
