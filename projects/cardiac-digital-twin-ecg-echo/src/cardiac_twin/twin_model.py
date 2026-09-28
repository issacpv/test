"""Interpretable cardiac state, forward models, unpaired alignment and counterfactuals.

The state ``z = (lvef, lvedv_ml, lv_mass_g, hr_bpm, qrs_ms)`` is observed
directly on the echo side (EF, EDV; mass from LVH labels) and indirectly on
the ECG side through ``ecg_forward``. Training data for the ECG->state
inverse model come from the prior + forward model (simulation-based
inference); the inferred EF score is then calibrated to the *unpaired* echo
cohort by 1-D optimal transport (quantile matching). The coefficients below
are population-level priors (documented as such) and are refitted on PTB-XL+
/ EchoNet-LVH in the full pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

STATE_COLS = ["lvef", "lvedv_ml", "lv_mass_g", "hr_bpm", "qrs_ms"]
ECG_COLS = ["hr_bpm", "qrs_ms", "axis_deg", "sokolow_lyon_mv"]


@dataclass(frozen=True)
class ForwardCoefficients:
    """Linear-Gaussian ECG forward model coefficients (priors; refit on real data)."""

    qrs_base_ms: float = 88.0
    qrs_per_ml_edv: float = 0.25        # QRS widens with LV dilation
    qrs_per_ef_point: float = -0.35      # lower EF -> longer QRS
    sokolow_base_mv: float = 1.6
    sokolow_per_g_mass: float = 0.012    # more mass -> more voltage
    axis_base_deg: float = 45.0
    axis_per_g_mass: float = -0.15       # LVH -> leftward axis
    noise_qrs_ms: float = 8.0
    noise_sokolow_mv: float = 0.5
    noise_axis_deg: float = 20.0
    noise_hr_bpm: float = 2.0


def sample_prior(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample plausible cardiac states with population-like correlations."""
    lvef = np.clip(rng.normal(58, 12, n), 10, 80)
    # dilated ventricles have lower EF: EDV increases as EF falls
    lvedv = np.clip(120 + (58 - lvef) * 2.5 + rng.normal(0, 25, n), 50, 400)
    mass = np.clip(150 + 0.3 * (lvedv - 120) + rng.normal(0, 35, n), 60, 400)
    hr = np.clip(rng.normal(75, 15, n), 35, 160)
    coef = ForwardCoefficients()
    qrs = np.clip(coef.qrs_base_ms + coef.qrs_per_ml_edv * (lvedv - 120) + coef.qrs_per_ef_point * (lvef - 58)
                  + rng.normal(0, 6, n), 60, 200)
    return pd.DataFrame({"lvef": lvef, "lvedv_ml": lvedv, "lv_mass_g": mass, "hr_bpm": hr, "qrs_ms": qrs})


def echo_forward(state: pd.DataFrame, rng: np.random.Generator, ef_noise: float = 5.0) -> pd.DataFrame:
    """Echo-side observation: EF, EDV, ESV with tracing noise (EF SD ~5 points)."""
    ef = np.clip(state["lvef"].to_numpy() + rng.normal(0, ef_noise, len(state)), 5, 85)
    edv = state["lvedv_ml"].to_numpy() * rng.lognormal(0, 0.08, len(state))
    return pd.DataFrame({"ef": ef, "edv_ml": edv, "esv_ml": edv * (1 - ef / 100.0)})


def ecg_forward(state: pd.DataFrame, rng: np.random.Generator, coef: ForwardCoefficients | None = None) -> pd.DataFrame:
    """ECG-side observation: features generated from the state plus noise."""
    c = coef or ForwardCoefficients()
    n = len(state)
    qrs = state["qrs_ms"].to_numpy() + rng.normal(0, c.noise_qrs_ms, n)
    sok = c.sokolow_base_mv + c.sokolow_per_g_mass * (state["lv_mass_g"].to_numpy() - 150) + rng.normal(0, c.noise_sokolow_mv, n)
    axis = c.axis_base_deg + c.axis_per_g_mass * (state["lv_mass_g"].to_numpy() - 150) + rng.normal(0, c.noise_axis_deg, n)
    hr = state["hr_bpm"].to_numpy() + rng.normal(0, c.noise_hr_bpm, n)
    return pd.DataFrame({"hr_bpm": hr, "qrs_ms": np.clip(qrs, 40, 250), "axis_deg": axis, "sokolow_lyon_mv": np.clip(sok, 0, 8)})


def quantile_align(source_scores: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    """1-D optimal-transport map: send each source score to the target quantile of the same rank.

    Valid for unpaired calibration only when both samples come from the same
    population (see README H5); returns the transported values.
    """
    s = np.asarray(source_scores, float)
    t = np.sort(np.asarray(target_values, float))
    ranks = (np.argsort(np.argsort(s)) + 0.5) / len(s)
    return np.quantile(t, ranks)


class UnpairedLatentAligner:
    """ECG-feature -> state regressor trained on simulated pairs, calibrated to an unpaired echo cohort."""

    def __init__(self, n_sim: int = 20000, alpha: float = 1.0, seed: int = 0) -> None:
        self.n_sim = n_sim
        self.alpha = alpha
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.echo_ef_reference: np.ndarray | None = None

    def fit(self, echo_ef: np.ndarray, coef: ForwardCoefficients | None = None) -> "UnpairedLatentAligner":
        """Fit the inverse model on prior/forward simulations and store the echo EF reference distribution."""
        rng = np.random.default_rng(self.seed)
        state = sample_prior(self.n_sim, rng)
        feats = ecg_forward(state, rng, coef)
        X = self.scaler.fit_transform(feats[ECG_COLS].to_numpy())
        self.model.fit(X, state[STATE_COLS].to_numpy())
        self.echo_ef_reference = np.asarray(echo_ef, float)
        return self

    def predict_state(self, ecg_features: pd.DataFrame, calibrate: bool = True) -> pd.DataFrame:
        """Predict the state from ECG features; EF is quantile-calibrated to the echo cohort if requested."""
        X = self.scaler.transform(ecg_features[ECG_COLS].to_numpy())
        pred = pd.DataFrame(self.model.predict(X), columns=STATE_COLS)
        if calibrate and self.echo_ef_reference is not None:
            pred["lvef_calibrated"] = quantile_align(pred["lvef"].to_numpy(), self.echo_ef_reference)
        return pred


def counterfactual(state: pd.DataFrame, rng: np.random.Generator, coef: ForwardCoefficients | None = None, **changes: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intervene on state columns (additive ``changes``) and regenerate ECG features.

    Returns ``(new_state, ecg_features_after)``. Downstream knock-on effects
    encoded in the prior (e.g. EF -> QRS) are applied for ``lvef`` and ``lvedv_ml``.
    """
    c = coef or ForwardCoefficients()
    new = state.copy()
    for k, v in changes.items():
        if k not in STATE_COLS:
            raise KeyError(f"unknown state column {k}")
        new[k] = new[k] + v
    d_ef = new["lvef"] - state["lvef"]
    d_edv = new["lvedv_ml"] - state["lvedv_ml"]
    new["qrs_ms"] = new["qrs_ms"] + c.qrs_per_ef_point * d_ef + c.qrs_per_ml_edv * d_edv
    return new, ecg_forward(new, rng, c)
