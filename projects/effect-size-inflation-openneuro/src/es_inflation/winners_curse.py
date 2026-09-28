"""Winner's-curse (selection-bias) estimators for reported effect sizes.

Model
-----
A study estimates an effect ``d_hat ~ N(d, se^2)`` and the estimate is *reported* only if it clears
a significance threshold ``|d_hat / se| > c`` (``c = z_{1 - alpha/2}`` for a two-sided test).
Everything below follows from that truncated-normal model:

* ``expected_reported_effect``   E[d_hat | selected]  (the biased mean of what gets published)
* ``type_m_error`` / ``type_s_error``  Gelman & Carlin (2014) exaggeration ratio and sign-error rate
* ``conditional_mle``            Zhong & Prentice (2008) conditional-likelihood MLE with a
                                 profile-likelihood confidence interval
* ``empirical_bayes_shrink``     normal-normal shrinkage across a corpus (no selection term)
* ``required_n``                 n for a target power at a (corrected) effect
* ``simulate_literature``        simulated studies + publication filter, for calibration checks

All functions are vectorised over numpy arrays where it makes sense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = [
    "z_from_alpha",
    "se_cohens_d",
    "se_fisher_z",
    "selection_probability",
    "expected_reported_effect",
    "expected_abs_reported_effect",
    "type_m_error",
    "type_s_error",
    "ConditionalMLEResult",
    "conditional_mle",
    "empirical_bayes_shrink",
    "required_n",
    "simulate_literature",
]


# ----------------------------------------------------------------- basics
def z_from_alpha(alpha: float = 0.05, two_sided: bool = True) -> float:
    """Critical z for a given alpha (two-sided by default)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return float(stats.norm.isf(alpha / 2 if two_sided else alpha))


def se_cohens_d(n1: int | np.ndarray, n2: Optional[int | np.ndarray] = None, d: float | np.ndarray = 0.0) -> np.ndarray:
    """Approximate standard error of Cohen's d.

    One-sample / paired (``n2 is None``): ``sqrt(1/n + d^2 / (2 n))``.
    Two-sample: ``sqrt(1/n1 + 1/n2 + d^2 / (2 (n1 + n2)))`` (Hedges & Olkin, 1985).
    """
    n1 = np.asarray(n1, dtype=float)
    d = np.asarray(d, dtype=float)
    if n2 is None:
        return np.sqrt(1.0 / n1 + d**2 / (2.0 * n1))
    n2 = np.asarray(n2, dtype=float)
    return np.sqrt(1.0 / n1 + 1.0 / n2 + d**2 / (2.0 * (n1 + n2)))


def se_fisher_z(n: int | np.ndarray) -> np.ndarray:
    """Standard error of Fisher's z-transformed correlation: ``1 / sqrt(n - 3)``."""
    n = np.asarray(n, dtype=float)
    return 1.0 / np.sqrt(n - 3.0)


# --------------------------------------------------- truncated-normal maths
def selection_probability(d: np.ndarray | float, se: np.ndarray | float, c: float) -> np.ndarray:
    """P(|d_hat| > c * se) = power of the two-sided test at true effect ``d``."""
    d = np.asarray(d, dtype=float)
    se = np.asarray(se, dtype=float)
    a = (c * se - d) / se
    b = (-c * se - d) / se
    return stats.norm.sf(a) + stats.norm.cdf(b)


def expected_reported_effect(d: np.ndarray | float, se: np.ndarray | float, c: float) -> np.ndarray:
    """E[d_hat | |d_hat| > c se] for ``d_hat ~ N(d, se^2)``.

    Uses the closed-form mean of a normal truncated to the two tails:
    ``int_{a}^{inf} x phi = mu (1 - Phi(alpha)) + sigma phi(alpha)`` and
    ``int_{-inf}^{b} x phi = mu Phi(beta) - sigma phi(beta)``.
    """
    d = np.asarray(d, dtype=float)
    se = np.asarray(se, dtype=float)
    alpha = (c * se - d) / se
    beta = (-c * se - d) / se
    upper = d * stats.norm.sf(alpha) + se * stats.norm.pdf(alpha)
    lower = d * stats.norm.cdf(beta) - se * stats.norm.pdf(beta)
    p = selection_probability(d, se, c)
    return (upper + lower) / p


def expected_abs_reported_effect(d: np.ndarray | float, se: np.ndarray | float, c: float) -> np.ndarray:
    """E[|d_hat| | |d_hat| > c se] (the quantity in Gelman & Carlin's Type M error)."""
    d = np.asarray(d, dtype=float)
    se = np.asarray(se, dtype=float)
    alpha = (c * se - d) / se
    beta = (-c * se - d) / se
    upper = d * stats.norm.sf(alpha) + se * stats.norm.pdf(alpha)
    lower = d * stats.norm.cdf(beta) - se * stats.norm.pdf(beta)
    p = selection_probability(d, se, c)
    return (upper - lower) / p


def type_m_error(d: np.ndarray | float, se: np.ndarray | float, alpha: float = 0.05) -> np.ndarray:
    """Exaggeration ratio E[|d_hat| | significant] / |d| (Gelman & Carlin, 2014). ``inf`` at d = 0."""
    d = np.asarray(d, dtype=float)
    c = z_from_alpha(alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        return expected_abs_reported_effect(d, se, c) / np.abs(d)


def type_s_error(d: np.ndarray | float, se: np.ndarray | float, alpha: float = 0.05) -> np.ndarray:
    """P(sign(d_hat) != sign(d) | significant). Defined for d != 0."""
    d = np.asarray(d, dtype=float)
    se = np.asarray(se, dtype=float)
    c = z_from_alpha(alpha)
    wrong_tail = np.where(d >= 0, stats.norm.cdf((-c * se - d) / se), stats.norm.sf((c * se - d) / se))
    return wrong_tail / selection_probability(d, se, c)


# -------------------------------------------------------- conditional MLE
@dataclass
class ConditionalMLEResult:
    """Output of ``conditional_mle``."""

    d_hat: float
    se: float
    threshold_z: float
    d_cond: float
    ci_low: float
    ci_high: float
    shrinkage: float  # d_cond / d_hat

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _cond_loglik(d: float, d_hat: float, se: float, c: float) -> float:
    return float(stats.norm.logpdf(d_hat, loc=d, scale=se) - np.log(selection_probability(d, se, c)))


def conditional_mle(d_hat: float, se: float, threshold_z: float, ci_level: float = 0.95) -> ConditionalMLEResult:
    """Conditional-likelihood MLE of the effect given that it was selected at ``|z| > threshold_z``.

    Maximises ``log phi((d_hat - d)/se) - log P(|d_hat| > c se; d)`` over ``d``
    (Zhong & Prentice, 2008, Biostatistics; Ghosh et al., 2008). The profile-likelihood CI is the
    set of ``d`` whose log-likelihood is within ``chi2_1(ci_level)/2`` of the maximum.

    If the reported effect did not clear the threshold (``|d_hat/se| <= c``) the conditional
    likelihood is not the right model; the function then returns the unconditional estimate.
    """
    d_hat = float(d_hat)
    se = float(se)
    c = float(threshold_z)
    if se <= 0:
        raise ValueError("se must be positive")
    if abs(d_hat / se) <= c:
        half = stats.norm.isf((1 - ci_level) / 2) * se
        return ConditionalMLEResult(d_hat, se, c, d_hat, d_hat - half, d_hat + half, 1.0)

    lo, hi = -abs(d_hat) - 6 * se, abs(d_hat) + 6 * se
    res = optimize.minimize_scalar(lambda d: -_cond_loglik(d, d_hat, se, c), bounds=(lo, hi), method="bounded")
    d_cond = float(res.x)
    ll_max = _cond_loglik(d_cond, d_hat, se, c)
    crit = stats.chi2.ppf(ci_level, df=1) / 2.0

    def g(d: float) -> float:
        return _cond_loglik(d, d_hat, se, c) - (ll_max - crit)

    ci_low, ci_high = lo, hi
    try:
        if g(lo) < 0:
            ci_low = float(optimize.brentq(g, lo, d_cond))
    except ValueError:
        pass
    try:
        if g(hi) < 0:
            ci_high = float(optimize.brentq(g, d_cond, hi))
    except ValueError:
        pass
    shrink = d_cond / d_hat if d_hat != 0 else np.nan
    return ConditionalMLEResult(d_hat, se, c, d_cond, ci_low, ci_high, float(shrink))


# ------------------------------------------------------- empirical Bayes
def empirical_bayes_shrink(d_hats: Sequence[float], ses: Sequence[float]) -> Tuple[np.ndarray, float, float]:
    """Normal-normal empirical-Bayes shrinkage of a set of effects (no selection term).

    ``d_i ~ N(mu, tau^2)``, ``d_hat_i | d_i ~ N(d_i, se_i^2)``. ``mu`` and ``tau^2`` are estimated by the
    DerSimonian-Laird moment estimator; posterior means are returned along with ``(mu, tau2)``.
    This is the *within-corpus* shrinkage benchmark; ``conditional_mle`` handles the selection term.
    """
    y = np.asarray(d_hats, dtype=float)
    s2 = np.asarray(ses, dtype=float) ** 2
    if y.size < 2:
        raise ValueError("need at least two studies")
    w = 1.0 / s2
    mu_fe = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu_fe) ** 2)
    cc = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (y.size - 1)) / cc)
    w_re = 1.0 / (s2 + tau2)
    mu = float(np.sum(w_re * y) / np.sum(w_re))
    if tau2 == 0.0:
        post = np.full_like(y, mu)
    else:
        post = (y / s2 + mu / tau2) / (1.0 / s2 + 1.0 / tau2)
    return post, mu, float(tau2)


# ------------------------------------------------------------ power tools
def required_n(d: float, power: float = 0.8, alpha: float = 0.05, two_sample: bool = False, n_max: int = 100_000) -> int:
    """Smallest per-group n giving at least ``power`` for effect ``d`` (two-sided z-test on d)."""
    if d == 0:
        return n_max
    c = z_from_alpha(alpha)
    lo, hi = 2, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        se = se_cohens_d(mid, mid if two_sample else None, d)
        if selection_probability(d, se, c) >= power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


# -------------------------------------------------------------- simulate
def simulate_literature(
    n_studies: int,
    true_effects: Sequence[float] | np.ndarray,
    sample_sizes: Sequence[int] | np.ndarray,
    alpha: float = 0.05,
    publish_only_significant: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate one-sample studies with given true effects and n, apply a publication filter.

    Returns a DataFrame with ``d_true, n, se, d_hat, z, significant, published``. With
    ``publish_only_significant=True`` only significant studies are kept (the winner's-curse regime).
    """
    rng = rng or np.random.default_rng(0)
    d_true = np.asarray(true_effects, dtype=float)
    ns = np.asarray(sample_sizes, dtype=int)
    if d_true.size != n_studies:
        d_true = rng.choice(d_true, size=n_studies, replace=True)
    if ns.size != n_studies:
        ns = rng.choice(ns, size=n_studies, replace=True)
    se = se_cohens_d(ns, None, d_true)
    d_hat = rng.normal(d_true, se)
    z = d_hat / se
    c = z_from_alpha(alpha)
    sig = np.abs(z) > c
    df = pd.DataFrame({"d_true": d_true, "n": ns, "se": se, "d_hat": d_hat, "z": z, "significant": sig})
    df["published"] = sig if publish_only_significant else True
    return df[df["published"]].reset_index(drop=True) if publish_only_significant else df
