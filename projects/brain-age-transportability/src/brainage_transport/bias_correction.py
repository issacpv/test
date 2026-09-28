"""Age-bias ("regression to the mean") corrections for brain-age delta.

Every regression-based brain-age model over-predicts young and under-predicts
old ages, so raw delta correlates negatively with age (Le et al., 2018; Liang
et al., 2019; Smith et al., 2019).  Several post-hoc corrections exist; all fit
a line on a *reference* sample and apply it elsewhere.  Their transport
behaviour differs:

``beheshti``  (Beheshti et al., 2019, NeuroImage: Clinical)
    Fit ``delta = a*age + b`` on the reference set; corrected delta is
    ``delta - (a*age + b)``.  Uses the test subject's chronological age.
``delange``   (de Lange & Cole, 2020, NeuroImage: Clinical)
    Fit ``pred = a*age + b``; corrected prediction is
    ``pred + (age - (a*age + b))``.  Algebraically identical to ``beheshti``
    (Zhang et al., 2023) - kept as a separate name for reporting.
``cole``      (Cole et al., 2018, Molecular Psychiatry)
    Fit ``pred = a*age + b``; corrected prediction is ``(pred - b)/a``.
    Does *not* use the test subject's age, so no age leakage, but inflates
    variance by ``1/a``.
``smith``     (Smith et al., 2019, NeuroImage)
    Like ``beheshti`` with a quadratic age term (``poly_degree=2``).
``none``
    Identity, for the uncorrected baseline.

Longitudinal consequence (the object of the audit)
--------------------------------------------------
For two visits of one person separated by ``dt`` years with raw delta change
``dD``, the corrected within-person change is ``dD - a*dt`` under
``beheshti``/``delange``/``smith`` but ``dD / a`` under ``cole``.  Because the
reference slope ``a`` is cohort- and scanner-specific, the *same* raw
trajectory can be labelled "accelerated" or "decelerated" ageing depending on
where the correction was calibrated.  :func:`longitudinal_change_under_method`
makes this explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

Method = Literal["none", "beheshti", "delange", "cole", "smith"]


def age_bias_slope(age: Sequence[float], delta: Sequence[float]) -> Dict[str, float]:
    """Slope / intercept / r of ``delta ~ age`` (the bias to be corrected)."""
    age = np.asarray(age, dtype=float)
    delta = np.asarray(delta, dtype=float)
    res = stats.linregress(age, delta)
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "r": float(res.rvalue), "p": float(res.pvalue)}


@dataclass
class BiasCorrector:
    """Fit a bias correction on a reference sample and apply it elsewhere.

    Parameters
    ----------
    method
        One of ``"none"``, ``"beheshti"``, ``"delange"``, ``"cole"``, ``"smith"``.
    poly_degree
        Polynomial degree of the age term (only ``beheshti``/``delange``/``smith``;
        ``smith`` forces 2).

    Notes
    -----
    Fit on *held-out* reference predictions (e.g. out-of-fold or a separate
    validation cohort).  Fitting on training-set predictions underestimates the
    bias because training residuals are shrunk (Butler et al., 2021).
    """

    method: Method = "beheshti"
    poly_degree: int = 1
    coef_: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.method == "smith":
            self.poly_degree = 2

    def fit(self, age: Sequence[float], pred: Sequence[float]) -> "BiasCorrector":
        age = np.asarray(age, dtype=float)
        pred = np.asarray(pred, dtype=float)
        if self.method == "none":
            self.coef_ = np.array([0.0])
            return self
        if self.method in ("beheshti", "smith"):
            # delta = poly(age)
            self.coef_ = np.polyfit(age, pred - age, self.poly_degree)
        elif self.method in ("delange", "cole"):
            # pred = poly(age)   (cole requires linear)
            deg = 1 if self.method == "cole" else self.poly_degree
            self.coef_ = np.polyfit(age, pred, deg)
        else:
            raise ValueError(f"unknown method {self.method!r}")
        return self

    def transform(self, age: Sequence[float], pred: Sequence[float]) -> np.ndarray:
        """Return the *corrected prediction* (corrected delta = result - age)."""
        if self.coef_ is None:
            raise RuntimeError("BiasCorrector not fitted")
        age = np.asarray(age, dtype=float)
        pred = np.asarray(pred, dtype=float)
        if self.method == "none":
            return pred
        if self.method in ("beheshti", "smith"):
            return pred - np.polyval(self.coef_, age)
        if self.method == "delange":
            return pred + (age - np.polyval(self.coef_, age))
        if self.method == "cole":
            a, b = self.coef_
            if abs(a) < 1e-8:
                raise ValueError("Cole correction undefined: reference slope is ~0")
            return (pred - b) / a
        raise ValueError(self.method)

    def fit_transform(self, age, pred) -> np.ndarray:
        return self.fit(age, pred).transform(age, pred)

    def corrected_delta(self, age, pred) -> np.ndarray:
        return self.transform(age, pred) - np.asarray(age, dtype=float)


def apply_all_methods(
    ref_age: Sequence[float], ref_pred: Sequence[float],
    target_age: Sequence[float], target_pred: Sequence[float],
    methods: Sequence[Method] = ("none", "beheshti", "cole", "smith"),
) -> pd.DataFrame:
    """Corrected deltas of a target cohort under each method (fit on reference).

    Returns a DataFrame with one column per method plus ``age``.
    """
    out = pd.DataFrame({"age": np.asarray(target_age, dtype=float)})
    for m in methods:
        bc = BiasCorrector(method=m).fit(ref_age, ref_pred)
        out[m] = bc.corrected_delta(target_age, target_pred)
    return out


def residual_bias_after_correction(age, corrected_delta) -> Dict[str, float]:
    """Remaining ``delta ~ age`` slope after correction (should be ~0 in-sample,
    but *not* when the correction is transported to a cohort with a different
    age range or scanner)."""
    return age_bias_slope(age, corrected_delta)


def longitudinal_change_under_method(
    raw_delta_change: Sequence[float], dt_years: Sequence[float], ref_slope: float, method: Method
) -> np.ndarray:
    """Within-person corrected delta change implied by each method.

    Parameters
    ----------
    raw_delta_change
        ``delta_visit2 - delta_visit1`` (uncorrected).
    dt_years
        Interval between visits.
    ref_slope
        Reference-set slope ``a`` (``delta ~ age`` for beheshti/smith-linear,
        ``pred ~ age`` for cole; note a_cole = 1 + a_beheshti).
    """
    dD = np.asarray(raw_delta_change, dtype=float)
    dt = np.asarray(dt_years, dtype=float)
    if method == "none":
        return dD
    if method in ("beheshti", "delange", "smith"):
        return dD - ref_slope * dt
    if method == "cole":
        # corrected pred = (pred-b)/a  ->  corrected delta change = (dPred)/a - dt
        # with dPred = dD + dt
        return (dD + dt) / ref_slope - dt
    raise ValueError(method)


def simulate_regression_to_mean(
    n: int = 2000, age_range=(40.0, 90.0), true_r2: float = 0.7, seed: int = 0
) -> pd.DataFrame:
    """Toy generator showing why raw delta correlates with age.

    A model that explains ``true_r2`` of age variance yields predictions
    shrunk towards the mean, i.e. ``pred = mean + sqrt(true_r2)*(age-mean)+noise``.
    Returns ``age``, ``pred``, ``delta``; ``delta ~ age`` has slope
    ``sqrt(true_r2)-1 < 0`` in expectation.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(*age_range, size=n)
    mu = age.mean()
    shrink = np.sqrt(true_r2)
    sd_age = age.std()
    noise_sd = sd_age * np.sqrt(true_r2 * (1 - true_r2))
    pred = mu + shrink * (age - mu) + rng.normal(0, noise_sd, n)
    return pd.DataFrame({"age": age, "pred": pred, "delta": pred - age})
