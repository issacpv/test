"""Competing-risk evaluation: Aalen-Johansen CIF, cause-specific concordance, IPCW Brier score.

Event coding follows :mod:`oai_prog.cohort`: 0 = censored, k >= 1 = event of cause k.
"""
from __future__ import annotations

import numpy as np


def kaplan_meier(times: np.ndarray, event_indicator: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival evaluated on ``grid`` (right-continuous)."""
    t = np.asarray(times, float)
    d = np.asarray(event_indicator, bool)
    uniq = np.unique(t[d])
    s = 1.0
    steps_t, steps_s = [], []
    for u in uniq:
        at_risk = np.sum(t >= u)
        deaths = np.sum((t == u) & d)
        if at_risk > 0:
            s *= 1 - deaths / at_risk
        steps_t.append(u)
        steps_s.append(s)
    steps_t = np.asarray(steps_t)
    steps_s = np.asarray(steps_s)
    out = np.ones(len(grid))
    for i, g in enumerate(np.asarray(grid, float)):
        idx = np.searchsorted(steps_t, g, side="right") - 1
        out[i] = steps_s[idx] if idx >= 0 else 1.0
    return out


def aalen_johansen(times: np.ndarray, events: np.ndarray, cause: int, grid: np.ndarray) -> np.ndarray:
    """Aalen-Johansen cumulative incidence of ``cause`` on ``grid`` with other causes as competing events."""
    t = np.asarray(times, float)
    e = np.asarray(events, int)
    uniq = np.unique(t[e > 0])
    s_prev = 1.0
    cif = 0.0
    steps_t, steps_c = [], []
    for u in uniq:
        at_risk = np.sum(t >= u)
        d_all = np.sum((t == u) & (e > 0))
        d_cause = np.sum((t == u) & (e == cause))
        if at_risk > 0:
            cif += s_prev * d_cause / at_risk
            s_prev *= 1 - d_all / at_risk
        steps_t.append(u)
        steps_c.append(cif)
    steps_t = np.asarray(steps_t)
    steps_c = np.asarray(steps_c)
    out = np.zeros(len(grid))
    for i, g in enumerate(np.asarray(grid, float)):
        idx = np.searchsorted(steps_t, g, side="right") - 1
        out[i] = steps_c[idx] if idx >= 0 else 0.0
    return out


def cause_specific_cindex(risk: np.ndarray, times: np.ndarray, events: np.ndarray, cause: int) -> float:
    """Cause-specific concordance: pairs (i, j) with event_i = cause and T_i < T_j are comparable.

    Subjects j may be censored or have any event after T_i. Ties in risk count 0.5.
    """
    r = np.asarray(risk, float)
    t = np.asarray(times, float)
    e = np.asarray(events, int)
    idx_cases = np.where(e == cause)[0]
    concordant = 0.0
    comparable = 0
    for i in idx_cases:
        later = t > t[i]
        comparable += int(later.sum())
        concordant += float(np.sum(r[i] > r[later])) + 0.5 * float(np.sum(r[i] == r[later]))
    return float(concordant / comparable) if comparable else float("nan")


def ipcw_brier(pred_cif_t: np.ndarray, times: np.ndarray, events: np.ndarray, cause: int, horizon: float) -> float:
    """IPCW Brier score for the cause-specific CIF at ``horizon`` (Graf et al. style with competing events).

    Status at the horizon: 1 if event of ``cause`` by the horizon, 0 if a
    competing event occurred by the horizon or no event by the horizon; subjects
    censored before the horizon are weighted 0. Weights use the Kaplan-Meier
    estimate of the censoring distribution.
    """
    p = np.asarray(pred_cif_t, float)
    t = np.asarray(times, float)
    e = np.asarray(events, int)
    grid = np.unique(np.concatenate([t, [horizon]]))
    g_cens = kaplan_meier(t, e == 0, grid)

    def g_at(x: float) -> float:
        return float(g_cens[np.searchsorted(grid, x)])

    total = 0.0
    for i in range(len(t)):
        if t[i] <= horizon and e[i] == cause:
            w = 1.0 / max(g_at(t[i]), 1e-8)
            total += w * (1.0 - p[i]) ** 2
        elif t[i] <= horizon and e[i] > 0:
            w = 1.0 / max(g_at(t[i]), 1e-8)
            total += w * (0.0 - p[i]) ** 2
        elif t[i] > horizon:
            w = 1.0 / max(g_at(horizon), 1e-8)
            total += w * (0.0 - p[i]) ** 2
        # censored before horizon: weight 0
    return float(total / len(t))


def simulate_competing_risks(n: int, risk: np.ndarray, base_hazards: tuple[float, float, float] = (0.01, 0.004, 0.002), censor_rate: float = 0.008, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate times/events for three causes where cause 1 hazard scales with ``exp(risk)``."""
    rng = np.random.default_rng(seed)
    h1 = base_hazards[0] * np.exp(np.asarray(risk, float))
    t1 = rng.exponential(1 / h1)
    t2 = rng.exponential(1 / base_hazards[1], n)
    t3 = rng.exponential(1 / base_hazards[2], n)
    tc = rng.exponential(1 / censor_rate, n)
    stack = np.column_stack([tc, t1, t2, t3])
    events = stack.argmin(1)
    times = stack.min(1)
    return times, events
