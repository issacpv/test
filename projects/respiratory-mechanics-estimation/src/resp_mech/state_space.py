"""Kalman filter / RTS smoother for time-varying elastance and resistance from charted rows.

State ``x = [E_L, R]`` (elastance in cmH2O/L, resistance in cmH2O/(L/s)),
random-walk dynamics with process noise proportional to elapsed time.
Observation rows (irregular in time, with missing values):

* ``Ppeak - PEEP = E_L * VT_L + R * Flow``   when Ppeak, PEEP, VT and a flow proxy are present
* ``Pplat - PEEP = E_L * VT_L``              when Pplat, PEEP and VT are present

Rows contribute whichever equations are available; rows with none are
propagated only (uncertainty grows). Output includes posterior mean and SD
of E and R, derived compliance and the inferred driving pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLS = ("time_s", "vt_ml", "flow_l_s", "ppeak", "pplat", "peep")


@dataclass
class KalmanMechanics:
    """Linear-Gaussian state-space estimator of [E_L, R].

    Parameters
    ----------
    e0, r0 : prior means (cmH2O/L, cmH2O/(L/s)).
    e_sd0, r_sd0 : prior SDs.
    q_e, q_r : process-noise SD per sqrt(hour) for E and R.
    obs_sd_peak, obs_sd_plat : measurement SDs (cmH2O) for peak and plateau equations.
    """

    e0: float = 30.0
    r0: float = 12.0
    e_sd0: float = 15.0
    r_sd0: float = 8.0
    q_e: float = 3.0
    q_r: float = 2.0
    obs_sd_peak: float = 2.0
    obs_sd_plat: float = 1.5

    def _observations(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hs, ys, rs = [], [], []
        vt_l = row["vt_ml"] / 1000.0 if np.isfinite(row["vt_ml"]) else np.nan
        if np.isfinite(vt_l) and np.isfinite(row["peep"]):
            if np.isfinite(row["pplat"]):
                hs.append([vt_l, 0.0])
                ys.append(row["pplat"] - row["peep"])
                rs.append(self.obs_sd_plat**2)
            if np.isfinite(row["ppeak"]) and np.isfinite(row["flow_l_s"]):
                hs.append([vt_l, row["flow_l_s"]])
                ys.append(row["ppeak"] - row["peep"])
                rs.append(self.obs_sd_peak**2)
        return np.asarray(hs, float).reshape(-1, 2), np.asarray(ys, float), np.asarray(rs, float)

    def filter(self, obs: pd.DataFrame, smooth: bool = True) -> pd.DataFrame:
        """Run the filter (and RTS smoother) over rows sorted by ``time_s``."""
        missing = [c for c in REQUIRED_COLS if c not in obs.columns]
        if missing:
            raise KeyError(f"missing columns: {missing}")
        df = obs.sort_values("time_s").reset_index(drop=True)
        n = len(df)
        x = np.array([self.e0, self.r0])
        P = np.diag([self.e_sd0**2, self.r_sd0**2])
        x_pred = np.zeros((n, 2))
        P_pred = np.zeros((n, 2, 2))
        x_filt = np.zeros((n, 2))
        P_filt = np.zeros((n, 2, 2))
        prev_t = df.loc[0, "time_s"]
        for k in range(n):
            dt_h = max(0.0, (df.loc[k, "time_s"] - prev_t) / 3600.0)
            prev_t = df.loc[k, "time_s"]
            Q = np.diag([self.q_e**2 * dt_h, self.q_r**2 * dt_h])
            P = P + Q
            x_pred[k], P_pred[k] = x, P
            H, y, r = self._observations(df.loc[k])
            if len(y):
                Rm = np.diag(r)
                S = H @ P @ H.T + Rm
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ (y - H @ x)
                P = (np.eye(2) - K @ H) @ P
                # keep physical positivity of the mean
                x = np.maximum(x, [1.0, 0.5])
            x_filt[k], P_filt[k] = x, P
        x_out, P_out = x_filt.copy(), P_filt.copy()
        if smooth and n > 1:
            for k in range(n - 2, -1, -1):
                G = P_filt[k] @ np.linalg.inv(P_pred[k + 1])
                x_out[k] = x_filt[k] + G @ (x_out[k + 1] - x_pred[k + 1])
                P_out[k] = P_filt[k] + G @ (P_out[k + 1] - P_pred[k + 1]) @ G.T
        out = df.copy()
        out["E_mean"] = x_out[:, 0]
        out["E_sd"] = np.sqrt(np.maximum(P_out[:, 0, 0], 0))
        out["R_mean"] = x_out[:, 1]
        out["R_sd"] = np.sqrt(np.maximum(P_out[:, 1, 1], 0))
        out["C_mean_ml"] = 1000.0 / out["E_mean"]
        vt_l = out["vt_ml"] / 1000.0
        out["dp_inferred"] = out["E_mean"] * vt_l
        out["dp_inferred_sd"] = out["E_sd"] * vt_l
        return out


def simulate_charting(
    e_true: np.ndarray,
    r_true: np.ndarray,
    times_s: np.ndarray,
    vt_ml: float = 450.0,
    flow_l_s: float = 0.75,
    peep: float = 8.0,
    pplat_fraction: float = 0.3,
    noise_sd: float = 1.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic charted rows from true E (cmH2O/L) and R trajectories.

    ``pplat_fraction`` rows get a charted plateau pressure (as in real
    databases where plateau checks are sporadic).
    """
    rng = np.random.default_rng(seed)
    n = len(times_s)
    vt_l = vt_ml / 1000.0
    ppeak = peep + e_true * vt_l + r_true * flow_l_s + rng.normal(0, noise_sd, n)
    pplat = peep + e_true * vt_l + rng.normal(0, noise_sd, n)
    has_plat = rng.random(n) < pplat_fraction
    pplat = np.where(has_plat, pplat, np.nan)
    return pd.DataFrame({
        "time_s": times_s, "vt_ml": vt_ml, "flow_l_s": flow_l_s, "ppeak": ppeak, "pplat": pplat, "peep": peep,
    })
