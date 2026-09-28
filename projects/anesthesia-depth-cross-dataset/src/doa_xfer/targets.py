"""Reference targets other than (and including) BIS.

* Age-adjusted MAC (Nickalls & Mapleson, 2003): MAC(age) = MAC40 * 10^(-0.00269 * (age - 40)).
  MAC40 values (vol%, in oxygen): sevoflurane 1.80, desflurane 6.6, isoflurane 1.17.
* Normalized propofol exposure: Ce / CE_REF with CE_REF = 3.0 ug/mL (an approximate
  effect-site concentration for loss of response in adults; used as a *scale*, not a
  threshold, so that propofol and volatile cases share a "fraction of hypnotic effect" axis).
* BIS lag: BIS is smoothed over tens of seconds; the lag relative to an EEG-derived
  trajectory is estimated by cross-correlation and applied before scoring.
* Phase labels from the clinical timestamps (pre-induction, induction, maintenance,
  emergence, post) and induction / emergence transition times from a depth trajectory.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

MAC40 = {"sevoflurane": 1.80, "desflurane": 6.6, "isoflurane": 1.17}
CE_REF_PROPOFOL = 3.0  # ug/mL


def mac_age_adjusted(age: float, agent: str = "sevoflurane") -> float:
    """Age-adjusted MAC (vol%) for a volatile agent."""
    return float(MAC40[agent] * 10 ** (-0.00269 * (float(age) - 40.0)))


def mac_fraction(exp_conc: np.ndarray, age: float, agent: str = "sevoflurane") -> np.ndarray:
    """End-tidal concentration expressed as a fraction of the age-adjusted MAC."""
    return np.asarray(exp_conc, float) / mac_age_adjusted(age, agent)


def normalized_propofol(ce: np.ndarray, ce_ref: float = CE_REF_PROPOFOL) -> np.ndarray:
    """Propofol effect-site concentration as a fraction of ``ce_ref``."""
    return np.asarray(ce, float) / ce_ref


def unified_exposure(agent: str, drug_track: np.ndarray, age: float) -> np.ndarray:
    """Normalized exposure on a common scale for the two arms."""
    if agent == "propofol":
        return normalized_propofol(drug_track)
    return mac_fraction(drug_track, age, agent)


def estimate_bis_lag(bis: np.ndarray, proxy: np.ndarray, step_s: float, max_lag_s: float = 120.0) -> float:
    """Lag (s, positive = BIS lags the proxy) maximizing cross-correlation between BIS and an EEG proxy (e.g. SEF95)."""
    b, p = np.asarray(bis, float), np.asarray(proxy, float)
    ok = np.isfinite(b) & np.isfinite(p)
    if ok.sum() < 10:
        return float("nan")
    b = b - b[ok].mean()
    p = p - p[ok].mean()
    b[~ok] = 0.0
    p[~ok] = 0.0
    max_k = int(max_lag_s / step_s)
    best_k, best_r = 0, -np.inf
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            x, y = p[: p.size - k] if k else p, b[k:]
        else:
            x, y = p[-k:], b[: b.size + k]
        if x.size < 10:
            continue
        r = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        if r > best_r:
            best_r, best_k = r, k
    return best_k * step_s


def shift_series(x: np.ndarray, lag_s: float, step_s: float) -> np.ndarray:
    """Shift a series earlier by ``lag_s`` (so that a lagged reference aligns with the EEG); pads with NaN."""
    k = int(round(lag_s / step_s))
    out = np.full_like(np.asarray(x, float), np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


PHASES = ("pre", "induction", "maintenance", "emergence", "post")


def phase_labels(t: np.ndarray, anestart: float, opstart: Optional[float], opend: Optional[float],
                 aneend: float, induction_s: float = 600.0, emergence_s: float = 900.0) -> np.ndarray:
    """Coarse anaesthesia phase per time point from clinical timestamps (seconds)."""
    t = np.asarray(t, float)
    lab = np.full(t.shape, "post", dtype=object)
    lab[t < anestart] = "pre"
    lab[(t >= anestart) & (t < anestart + induction_s)] = "induction"
    main_end = (opend if opend is not None and np.isfinite(opend) else aneend - emergence_s)
    lab[(t >= anestart + induction_s) & (t < main_end)] = "maintenance"
    lab[(t >= main_end) & (t < aneend)] = "emergence"
    return lab


def transition_times(t: np.ndarray, depth: np.ndarray, phases: np.ndarray, n_sd: float = 2.0) -> Dict[str, float]:
    """LOC / ROC times from a depth trajectory: first / last entry into the maintenance band (median +/- n_sd * SD)."""
    t, d = np.asarray(t, float), np.asarray(depth, float)
    m = (phases == "maintenance") & np.isfinite(d)
    if m.sum() < 10:
        return {"loc_s": float("nan"), "roc_s": float("nan")}
    med, sd = np.median(d[m]), np.std(d[m]) + 1e-9
    inband = np.isfinite(d) & (np.abs(d - med) <= n_sd * sd)
    after_start = np.where(inband & (phases != "pre"))[0]
    loc = t[after_start[0]] if after_start.size else float("nan")
    before_end = np.where(inband & (phases != "post"))[0]
    roc = t[before_end[-1]] if before_end.size else float("nan")
    return {"loc_s": float(loc), "roc_s": float(roc)}


def bis_threshold_times(t: np.ndarray, bis: np.ndarray, loc_thr: float = 60.0, roc_thr: float = 80.0) -> Dict[str, float]:
    """Conventional BIS-based LOC (first BIS < 60) and ROC (last BIS < 80 before rising above)."""
    t, b = np.asarray(t, float), np.asarray(bis, float)
    below = np.where(np.isfinite(b) & (b < loc_thr))[0]
    loc = t[below[0]] if below.size else float("nan")
    under = np.where(np.isfinite(b) & (b < roc_thr))[0]
    roc = t[under[-1]] if under.size else float("nan")
    return {"loc_s": float(loc), "roc_s": float(roc)}


def responsiveness_labels(hit_rate: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Binary responsive (1) / unresponsive (0) from a behavioural hit rate (Cambridge set)."""
    return (np.asarray(hit_rate, float) >= thr).astype(int)
