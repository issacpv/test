"""Heart-rate correction formulae for the QT interval.

All functions take QT and RR in **seconds** and return the corrected QT (QTc)
in **milliseconds** (the clinical convention), so a normal QTc is ~400 ms.

The population/individual correction (``fit_power_correction``) estimates the
exponent alpha in QTc = QT / RR**alpha by regressing log(QT) on log(RR) over
drug-naive baseline beats, which removes the systematic over-/under-correction
of the fixed Bazett (alpha=1/2) and Fridericia (alpha=1/3) formulae for a given
cohort. This matters because rate-changing drugs (beta-blockers, sotalol) would
otherwise show a spurious QTc slope under Bazett.

References: Bazett (1920); Fridericia (1920); Sagie et al. (1992, Framingham);
Hodges et al. (1983); Vandenberk et al. (2016, JAHA) for formula comparison.
"""
from __future__ import annotations

import numpy as np

_MS = 1000.0


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def bazett(qt_s, rr_s) -> np.ndarray:
    """QTc = QT / sqrt(RR). Over-corrects at high heart rates."""
    qt, rr = _to_array(qt_s), _to_array(rr_s)
    return _MS * qt / np.sqrt(rr)


def fridericia(qt_s, rr_s) -> np.ndarray:
    """QTc = QT / RR**(1/3). The E14 default for most thorough-QT studies."""
    qt, rr = _to_array(qt_s), _to_array(rr_s)
    return _MS * qt / np.cbrt(rr)


def framingham(qt_s, rr_s) -> np.ndarray:
    """Linear (Sagie) correction: QTc = QT + 0.154 * (1 - RR)."""
    qt, rr = _to_array(qt_s), _to_array(rr_s)
    return _MS * (qt + 0.154 * (1.0 - rr))


def hodges(qt_s, rr_s) -> np.ndarray:
    """Hodges: QTc = QT_ms + 1.75 * (HR - 60), HR in bpm."""
    qt, rr = _to_array(qt_s), _to_array(rr_s)
    hr = 60.0 / rr
    return _MS * qt + 1.75 * (hr - 60.0)


def power_correction(qt_s, rr_s, alpha: float) -> np.ndarray:
    """Generic power correction QTc = QT / RR**alpha (alpha estimated per cohort)."""
    qt, rr = _to_array(qt_s), _to_array(rr_s)
    return _MS * qt / np.power(rr, alpha)


def fit_power_correction(qt_s, rr_s) -> float:
    """Estimate alpha by OLS of log(QT) on log(RR): log QT = c + alpha * log RR.

    Fit on drug-naive baseline beats only. Returns the slope alpha (typically
    0.3-0.4 for whole cohorts; ~0.5 recovers Bazett, ~0.33 Fridericia).
    """
    qt, rr = _to_array(qt_s), _to_array(rr_s)
    m = np.isfinite(qt) & np.isfinite(rr) & (qt > 0) & (rr > 0)
    if m.sum() < 10:
        raise ValueError("need >=10 valid (QT, RR) pairs to fit alpha")
    x = np.log(rr[m])
    y = np.log(qt[m])
    alpha = float(np.polyfit(x, y, 1)[0])
    return alpha


FORMULAE = {
    "bazett": bazett,
    "fridericia": fridericia,
    "framingham": framingham,
    "hodges": hodges,
}


def correct(qt_s, rr_s, method: str = "fridericia", alpha: float | None = None) -> np.ndarray:
    """Dispatch to a named correction. ``method='power'`` requires ``alpha``."""
    if method == "power":
        if alpha is None:
            raise ValueError("method='power' requires alpha (see fit_power_correction)")
        return power_correction(qt_s, rr_s, alpha)
    if method not in FORMULAE:
        raise KeyError(f"unknown correction '{method}'; choose {list(FORMULAE) + ['power']}")
    return FORMULAE[method](qt_s, rr_s)
