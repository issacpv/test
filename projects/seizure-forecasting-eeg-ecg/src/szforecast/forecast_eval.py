"""Forecasting evaluation: alarms, sensitivity, time-in-warning, chance level and improvement over chance.

* :func:`firing_power` turns window-level pre-ictal probabilities into alarms
  (Teixeira et al., 2011): moving average of thresholded predictions over the
  SOP length; an alarm fires when the firing power crosses ``alarm_thr`` and is
  followed by a refractory period equal to the SOP.
* :func:`evaluate_alarms` scores alarms against seizure onsets with the SPH/SOP
  rule: a seizure is forecast if an alarm was raised in
  ``[onset - SPH - SOP, onset - SPH]``.
* :func:`chance_sensitivity` is the sensitivity of an unspecific random predictor
  at the same false-alarm rate, ``1 - exp(-FPR * SOP)`` (Winterhalder et al., 2003;
  Schelter et al., 2006), and :func:`binomial_pvalue` tests whether the number of
  forecast seizures exceeds that chance level (Snyder et al., 2008 for the
  time-in-warning formulation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def firing_power(prob: np.ndarray, window_len_s: float, sop_s: float, prob_thr: float = 0.5,
                 alarm_thr: float = 0.5, refractory_s: float | None = None) -> np.ndarray:
    """Boolean alarm array (one entry per window) from window probabilities.

    Firing power = fraction of the last ``sop_s / window_len_s`` windows with
    ``prob > prob_thr``.  After an alarm, no new alarm is raised for ``refractory_s``
    (defaults to the SOP).
    """
    p = np.asarray(prob, float)
    k = max(1, int(round(sop_s / window_len_s)))
    hits = (p > prob_thr).astype(float)
    fp = np.convolve(hits, np.ones(k) / k, mode="full")[: p.size]
    refr = int(round((sop_s if refractory_s is None else refractory_s) / window_len_s))
    alarms = np.zeros(p.size, dtype=bool)
    last = -10 ** 9
    for i in range(p.size):
        if fp[i] >= alarm_thr and i - last > refr:
            alarms[i] = True
            last = i
    return alarms


@dataclass
class AlarmScore:
    n_seizures: int
    n_forecast: int
    n_alarms: int
    n_false_alarms: int
    total_time_s: float
    sop_s: float

    @property
    def sensitivity(self) -> float:
        return self.n_forecast / self.n_seizures if self.n_seizures else float("nan")

    @property
    def fpr_per_h(self) -> float:
        return self.n_false_alarms / (self.total_time_s / 3600.0) if self.total_time_s else float("nan")

    @property
    def time_in_warning(self) -> float:
        """Fraction of the record spent under warning (each alarm covers one SOP)."""
        return min(1.0, self.n_alarms * self.sop_s / self.total_time_s) if self.total_time_s else float("nan")

    def as_dict(self) -> Dict[str, float]:
        return dict(sensitivity=self.sensitivity, fpr_per_h=self.fpr_per_h, time_in_warning=self.time_in_warning,
                    n_seizures=self.n_seizures, n_forecast=self.n_forecast, n_alarms=self.n_alarms)


def evaluate_alarms(alarm_times_s: Sequence[float], onsets_s: Sequence[float], sph_s: float, sop_s: float,
                    total_time_s: float) -> AlarmScore:
    """Score alarm times against seizure onsets with the SPH / SOP rule."""
    alarms = np.sort(np.asarray(alarm_times_s, float))
    onsets = np.sort(np.asarray(onsets_s, float))
    forecast = 0
    used = np.zeros(alarms.size, dtype=bool)
    for o in onsets:
        lo, hi = o - sph_s - sop_s, o - sph_s
        idx = np.where((alarms >= lo) & (alarms <= hi) & ~used)[0]
        if idx.size:
            forecast += 1
            used[idx] = True
    return AlarmScore(n_seizures=onsets.size, n_forecast=forecast, n_alarms=alarms.size,
                      n_false_alarms=int((~used).sum()), total_time_s=float(total_time_s), sop_s=sop_s)


def chance_sensitivity(fpr_per_h: float, sop_s: float) -> float:
    """Sensitivity of an unspecific random predictor with Poisson alarms at ``fpr_per_h`` and SOP ``sop_s``."""
    return float(1.0 - np.exp(-fpr_per_h * sop_s / 3600.0))


def chance_sensitivity_from_tiw(time_in_warning: float) -> float:
    """Chance sensitivity equals the time-in-warning fraction for a random predictor (Snyder et al., 2008)."""
    return float(np.clip(time_in_warning, 0.0, 1.0))


def binomial_pvalue(n_forecast: int, n_seizures: int, p_chance: float) -> float:
    """One-sided binomial p-value that >= ``n_forecast`` of ``n_seizures`` are forecast by chance."""
    if n_seizures == 0:
        return float("nan")
    return float(stats.binom.sf(n_forecast - 1, n_seizures, np.clip(p_chance, 1e-12, 1 - 1e-12)))


def improvement_over_chance(score: AlarmScore, method: str = "tiw") -> float:
    """IoC = sensitivity - chance sensitivity, using time-in-warning (default) or the FPR formula."""
    pc = chance_sensitivity_from_tiw(score.time_in_warning) if method == "tiw" else chance_sensitivity(score.fpr_per_h, score.sop_s)
    return float(score.sensitivity - pc)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def surrogate_pvalue(observed: float, surrogate_values: Sequence[float]) -> float:
    """Fraction of surrogate statistics >= observed, with the +1 correction."""
    s = np.asarray(surrogate_values, float)
    s = s[np.isfinite(s)]
    return float((np.sum(s >= observed) + 1) / (s.size + 1))


def horizon_grid(prob: np.ndarray, window_starts: np.ndarray, window_len_s: float, onsets_s: Sequence[float],
                 total_time_s: float, sph_grid_s: Sequence[float], sop_grid_s: Sequence[float]) -> List[Dict[str, float]]:
    """IoC / sensitivity / FPR over an SPH x SOP grid from one probability trace (no re-fitting)."""
    rows = []
    for sph in sph_grid_s:
        for sop in sop_grid_s:
            al = firing_power(prob, window_len_s, sop)
            sc = evaluate_alarms(np.asarray(window_starts)[al] + window_len_s, onsets_s, sph, sop, total_time_s)
            d = sc.as_dict()
            d.update(sph_s=sph, sop_s=sop, ioc=improvement_over_chance(sc))
            rows.append(d)
    return rows
