"""Open pulse-contour stroke-volume estimators and calibration.

All estimators return a quantity *proportional* to stroke volume; absolute
mL need a calibration factor from a reference method (one-point ratio or
regression), exactly as commercial calibrated devices do. The uncalibrated
form is what matters for trend tracking.

Implemented
-----------
* ``liljestrand``   PP / (SBP + DBP)                        (Liljestrand & Zander, 1928)
* ``herd``          MAP - DBP                                (Herd et al., 1966)
* ``systolic_area`` area under the systolic part of the pulse (Kouchoukos et al., 1970)
* ``windkessel2``   two-element Windkessel with per-beat diastolic time constant:
                    SV ~ C * [ (P_notch - P_dbp) + (1/tau) * integral_onset^notch P dt ]
* ``corrected_area`` systolic area with an empirical impedance correction fitted on calibration data
                    (the idea behind Wesseling's cZ; coefficients are *fitted*, not copied).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METHODS = ("liljestrand", "herd", "systolic_area", "windkessel2")


def liljestrand(f: pd.DataFrame) -> np.ndarray:
    return (f["pp"] / (f["sbp"] + f["dbp"])).to_numpy(dtype=float)


def herd(f: pd.DataFrame) -> np.ndarray:
    return (f["map"] - f["dbp"]).to_numpy(dtype=float)


def systolic_area(f: pd.DataFrame) -> np.ndarray:
    return f["sys_area"].to_numpy(dtype=float)


def windkessel2(f: pd.DataFrame) -> np.ndarray:
    """C-proportional SV from the 2-element Windkessel identity SV = C dP + (1/R) integral P dt with R = tau / C.

    The integral term uses the mean pressure over systole times t_sys as an approximation of
    integral_onset^notch P dt; beats without a valid tau fall back to the median tau of the record.
    """
    tau = f["tau"].to_numpy(dtype=float)
    tau = np.where(np.isfinite(tau), tau, np.nanmedian(tau) if np.isfinite(np.nanmedian(tau)) else 1.5)
    p_int = (f["sys_area"] + f["dbp"] * f["t_sys"]).to_numpy(dtype=float)  # integral of absolute pressure over systole
    return (f["notch_p"] - f["dbp"]).to_numpy(dtype=float) + p_int / tau


def estimate(f: pd.DataFrame, method: str) -> np.ndarray:
    """Proportional SV per beat for ``method`` in ``METHODS``."""
    fn = {"liljestrand": liljestrand, "herd": herd, "systolic_area": systolic_area, "windkessel2": windkessel2}[method]
    out = fn(f)
    out = np.where(f["ok"].to_numpy(bool), out, np.nan) if "ok" in f else out
    return out


@dataclass
class Calibration:
    scale: float
    offset: float
    method: str

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(x, dtype=float) + self.offset


def calibrate(est: np.ndarray, ref: np.ndarray, method: str = "ratio") -> Calibration:
    """``ratio``: one scale factor (mean ref / mean est), the usual one-point calibration.
    ``ols``: scale and offset by least squares (needs several reference points)."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    m = np.isfinite(est) & np.isfinite(ref)
    if m.sum() == 0:
        raise ValueError("no finite pairs to calibrate on")
    if method == "ratio":
        return Calibration(float(ref[m].mean() / est[m].mean()), 0.0, "ratio")
    if method == "ols":
        A = np.c_[est[m], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, ref[m], rcond=None)
        return Calibration(float(coef[0]), float(coef[1]), "ols")
    raise ValueError(method)


def fit_impedance_correction(f: pd.DataFrame, sv_ref: np.ndarray) -> np.ndarray:
    """Fit log(SV_ref / systolic_area) = b0 + b1*HR + b2*MAP by least squares.

    Returns the coefficient vector; ``corrected_area`` applies it. This is the empirical analogue
    of Wesseling's corrected characteristic impedance, learned on the calibration cohort.
    """
    area = f["sys_area"].to_numpy(float)
    y = np.log(np.asarray(sv_ref, float) / area)
    X = np.c_[np.ones(len(f)), f["hr"].to_numpy(float), f["map"].to_numpy(float)]
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    coef, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    return coef


def corrected_area(f: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(len(f)), f["hr"].to_numpy(float), f["map"].to_numpy(float)]
    return f["sys_area"].to_numpy(float) * np.exp(X @ coef)


def beat_to_window(f: pd.DataFrame, values: np.ndarray, fs: float, window_s: float = 20.0) -> pd.DataFrame:
    """Median-aggregate per-beat estimates into fixed windows: ``t_center, sv, hr, n_beats``."""
    t = f["onset"].to_numpy() / fs
    bins = np.floor(t / window_s).astype(int)
    df = pd.DataFrame({"bin": bins, "sv": values, "hr": f["hr"].to_numpy()})
    g = df.groupby("bin").agg(sv=("sv", "median"), hr=("hr", "median"), n_beats=("sv", "size"))
    g["t_center"] = (g.index + 0.5) * window_s
    return g.reset_index(drop=True)
