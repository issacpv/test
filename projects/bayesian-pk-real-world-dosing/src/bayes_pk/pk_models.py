"""Analytic linear compartment models for zero-order (infusion) dosing.

All times are in **hours** relative to an arbitrary origin (in practice the start of the course), amounts in
**mg**, volumes in **L**, clearances in **L/h**, concentrations in **mg/L**.

A dosing history is a :class:`pandas.DataFrame` with columns ``start``, ``end`` and ``amount_mg``; each row is
a constant-rate infusion of ``amount_mg`` over ``[start, end]``.  Boluses are represented as short infusions
(``end = start + BOLUS_MINUTES/60``).  Superposition (linearity) gives the concentration for any history.

The :class:`ModelSpec` holds a population model: typical parameter values (``theta``), log-normal
inter-individual variances (``omega``), residual-error parameters (``sigma_prop``, ``sigma_add``) and a
covariate function mapping ``(theta, covariates) -> individual typical parameters``.  Published models are to
be transcribed into ``ModelSpec`` objects from their papers (see README); :func:`illustrative_icu_prior` is a
**placeholder for testing only** and must not be mistaken for a published model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd

BOLUS_MINUTES = 2.0
_EPS = 1e-12

CovariateFn = Callable[[Mapping[str, float], Mapping[str, float]], Dict[str, float]]


@dataclass
class ModelSpec:
    """Population PK model description.

    Parameters
    ----------
    name:
        Short identifier (e.g. ``"thomson2009"``).
    ncmt:
        1 or 2 compartments.
    theta:
        Typical values and covariate coefficients. For ``ncmt == 1`` the covariate function must return
        ``{"CL", "V"}``; for ``ncmt == 2`` it must return ``{"CL", "V1", "Q", "V2"}``.
    omega:
        Variances of the log-normal inter-individual random effects, keyed by parameter name. Parameters not
        listed have no random effect.
    sigma_prop, sigma_add:
        Proportional (fraction) and additive (mg/L) residual-error SDs; combined error variance is
        ``(sigma_prop * C)^2 + sigma_add^2``.
    covariate_fn:
        Callable ``(theta, covariates) -> dict`` giving the individual *typical* parameters.
    citation:
        Where the numbers come from. Empty for illustrative specs.
    """

    name: str
    ncmt: int
    theta: Dict[str, float]
    omega: Dict[str, float]
    sigma_prop: float
    sigma_add: float
    covariate_fn: CovariateFn
    citation: str = ""
    eta_names: tuple = field(init=False)

    def __post_init__(self) -> None:
        if self.ncmt not in (1, 2):
            raise ValueError("ncmt must be 1 or 2")
        self.eta_names = tuple(sorted(self.omega))

    def individual_params(self, covariates: Mapping[str, float], eta: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Typical parameters for ``covariates`` multiplied by ``exp(eta)`` for parameters with random effects."""
        typ = dict(self.covariate_fn(self.theta, covariates))
        if eta is not None:
            eta = np.asarray(eta, dtype=float)
            if eta.shape != (len(self.eta_names),):
                raise ValueError(f"eta must have shape ({len(self.eta_names)},)")
            for name, e in zip(self.eta_names, eta):
                typ[name] = typ[name] * float(np.exp(e))
        return typ


# --------------------------------------------------------------------------------------------- dosing utils
def as_dose_frame(doses: pd.DataFrame) -> pd.DataFrame:
    """Validate a dose history and coerce boluses to short infusions. Returns a copy sorted by ``start``."""
    d = doses[["start", "end", "amount_mg"]].copy()
    d = d.astype(float)
    bolus = d["end"] <= d["start"]
    d.loc[bolus, "end"] = d.loc[bolus, "start"] + BOLUS_MINUTES / 60.0
    if (d["amount_mg"] < 0).any():
        raise ValueError("negative dose amounts")
    return d.sort_values("start").reset_index(drop=True)


# ------------------------------------------------------------------------------------- 1-compartment kinetics
def conc_1cmt_infusion(t: np.ndarray, doses: pd.DataFrame, CL: float, V: float) -> np.ndarray:
    """Concentration of a 1-compartment model under a set of zero-order infusions (superposition)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    d = as_dose_frame(doses)
    k = CL / V
    c = np.zeros_like(t)
    for start, end, amt in d.itertuples(index=False):
        dur = end - start
        rate = amt / dur
        tau = t - start
        during = (tau >= 0) & (t <= end)
        after = t > end
        c[during] += rate / CL * (1.0 - np.exp(-k * tau[during]))
        c[after] += rate / CL * (1.0 - np.exp(-k * dur)) * np.exp(-k * (t[after] - end))
    return c


# ------------------------------------------------------------------------------------- 2-compartment kinetics
def _macro_constants(CL: float, V1: float, Q: float, V2: float) -> tuple[float, float, float, float]:
    """Return ``(alpha, beta, A, B)`` for the unit-bolus response ``C(t) = A e^{-alpha t} + B e^{-beta t}``."""
    k10, k12, k21 = CL / V1, Q / V1, Q / V2
    s = k10 + k12 + k21
    p = k10 * k21
    disc = np.sqrt(max(s * s - 4.0 * p, 0.0))
    alpha = 0.5 * (s + disc)
    beta = 0.5 * (s - disc)
    A = (alpha - k21) / (V1 * (alpha - beta + _EPS))
    B = (k21 - beta) / (V1 * (alpha - beta + _EPS))
    return alpha, beta, A, B


def conc_2cmt_infusion(t: np.ndarray, doses: pd.DataFrame, CL: float, V1: float, Q: float, V2: float) -> np.ndarray:
    """Central-compartment concentration of a 2-compartment model under zero-order infusions."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    d = as_dose_frame(doses)
    alpha, beta, A, B = _macro_constants(CL, V1, Q, V2)
    c = np.zeros_like(t)
    for start, end, amt in d.itertuples(index=False):
        dur = end - start
        rate = amt / dur
        tau = t - start
        during = (tau >= 0) & (t <= end)
        after = t > end
        c[during] += rate * (A / alpha * (1.0 - np.exp(-alpha * tau[during])) + B / beta * (1.0 - np.exp(-beta * tau[during])))
        ta = t[after] - end
        c[after] += rate * (
            A / alpha * (1.0 - np.exp(-alpha * dur)) * np.exp(-alpha * ta)
            + B / beta * (1.0 - np.exp(-beta * dur)) * np.exp(-beta * ta)
        )
    return c


def concentration(spec: ModelSpec, params: Mapping[str, float], doses: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """Dispatch on ``spec.ncmt``."""
    if spec.ncmt == 1:
        return conc_1cmt_infusion(t, doses, params["CL"], params["V"])
    return conc_2cmt_infusion(t, doses, params["CL"], params["V1"], params["Q"], params["V2"])


def auc_interval(
    spec: ModelSpec, params: Mapping[str, float], doses: pd.DataFrame, t0: float, t1: float, n_grid: int = 4001
) -> float:
    """AUC (mg*h/L) of the model concentration over ``[t0, t1]`` by trapezoidal integration of the analytic curve.

    A dense grid (default 4001 points) makes the error negligible relative to residual variability; infusion
    boundaries are added to the grid so that the kink at each infusion end is integrated exactly.
    """
    d = as_dose_frame(doses)
    grid = np.linspace(t0, t1, n_grid)
    knots = np.concatenate([d["start"].to_numpy(), d["end"].to_numpy()])
    knots = knots[(knots > t0) & (knots < t1)]
    grid = np.unique(np.concatenate([grid, knots]))
    c = concentration(spec, params, d, grid)
    return float(np.trapezoid(c, grid)) if hasattr(np, "trapezoid") else float(np.trapz(c, grid))


# --------------------------------------------------------------------------------------- illustrative prior
def _illustrative_covariates(theta: Mapping[str, float], cov: Mapping[str, float]) -> Dict[str, float]:
    """CL scales linearly with creatinine clearance, volumes with weight. Illustrative shape only."""
    crcl = float(cov.get("crcl_ml_min", 100.0))
    wt = float(cov.get("weight_kg", 70.0))
    return {
        "CL": theta["CL_pop"] * (crcl / 100.0) ** theta["CL_crcl_exp"],
        "V1": theta["V1_pop"] * (wt / 70.0),
        "Q": theta["Q_pop"],
        "V2": theta["V2_pop"] * (wt / 70.0),
    }


def illustrative_icu_prior() -> ModelSpec:
    """A **placeholder** two-compartment vancomycin-like prior for unit tests and pipeline development.

    The numbers are round values in the physiological range and are NOT taken from any publication. Replace
    with transcribed published models (see ``data/models/``) before any scientific use.
    """
    return ModelSpec(
        name="illustrative_2cmt",
        ncmt=2,
        theta={"CL_pop": 3.0, "CL_crcl_exp": 0.8, "V1_pop": 30.0, "Q_pop": 5.0, "V2_pop": 40.0},
        omega={"CL": 0.15, "V1": 0.10},
        sigma_prop=0.15,
        sigma_add=1.0,
        covariate_fn=_illustrative_covariates,
        citation="ILLUSTRATIVE ONLY - not a published model",
    )


def illustrative_1cmt_prior() -> ModelSpec:
    """One-compartment placeholder (for tests of the 1-cmt code path)."""

    def cov_fn(theta: Mapping[str, float], cov: Mapping[str, float]) -> Dict[str, float]:
        crcl = float(cov.get("crcl_ml_min", 100.0))
        wt = float(cov.get("weight_kg", 70.0))
        return {"CL": theta["CL_pop"] * (crcl / 100.0), "V": theta["V_pop"] * (wt / 70.0)}

    return ModelSpec(
        name="illustrative_1cmt",
        ncmt=1,
        theta={"CL_pop": 3.5, "V_pop": 50.0},
        omega={"CL": 0.12, "V": 0.08},
        sigma_prop=0.15,
        sigma_add=1.0,
        covariate_fn=cov_fn,
        citation="ILLUSTRATIVE ONLY - not a published model",
    )
