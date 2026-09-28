"""Single-compartment lung + ventilator simulator that *generates labelled asynchronies*.

Equation of motion (volume ``V`` above FRC, flow ``Q = dV/dt``, all pressures in cmH2O,
volumes in L, flows in L/s):

    Paw = R*Q + V/C + PEEP - Pmus

where ``Pmus >= 0`` is the patient's inspiratory muscle pressure. The ventilator is a
state machine (assist-control volume control, or pressure support with a backup rate)
with flow triggering, a post-cycling trigger lockout and, for PS, an expiratory flow
cycling criterion. Patient efforts follow a neural schedule (independent of the machine)
or are *entrained* to mandatory breaths (reverse triggering). The simulator returns the
waveforms **and** the ground-truth breath / effort tables, so detectors and low-resolution
surrogates can be validated with known asynchrony indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LungParams:
    R: float = 10.0        # cmH2O / (L/s)
    C: float = 0.05        # L / cmH2O  (tau = R*C = 0.5 s)


@dataclass
class VentSettings:
    mode: str = "VC"       # "VC" (assist-control, square flow) or "PS" (pressure support with backup rate)
    vt: float = 0.45       # L (VC)
    ti: float = 1.0        # s (VC inspiratory time)
    rr_set: float = 15.0   # breaths/min (mandatory / backup rate)
    peep: float = 5.0      # cmH2O
    ps: float = 10.0       # cmH2O above PEEP (PS)
    trigger_flow: float = 0.03   # L/s (~2 L/min)
    cycle_pct: float = 0.25      # PS expiratory trigger: cycle when flow < cycle_pct * peak
    max_ti: float = 3.0
    lockout_s: float = 0.2       # trigger lockout (minimum expiratory time) after cycling
    bias_flow: float = 0.08      # L/s: the most a patient can draw from the circuit without a delivered breath


@dataclass
class EffortModel:
    mode: str = "spontaneous"    # "spontaneous", "entrained" (reverse triggering), "none"
    rr_neural: float = 18.0
    pmus_amp: float = 8.0        # cmH2O
    ti_neural: float = 0.9       # s (rise, half-sine)
    decay_s: float = 0.15
    jitter: float = 0.05         # relative jitter of neural period
    entrain_delay_s: float = 0.4 # entrained: effort starts this long after a mandatory breath onset
    entrain_prob: float = 1.0


@dataclass
class SimConfig:
    duration_s: float = 120.0
    fs: float = 50.0
    lung: LungParams = field(default_factory=LungParams)
    vent: VentSettings = field(default_factory=VentSettings)
    effort: EffortModel = field(default_factory=EffortModel)
    cardiac_osc_lps: float = 0.0   # cardiogenic flow oscillation amplitude (auto-triggering source)
    heart_rate: float = 80.0
    flow_noise_lps: float = 0.005
    seed: int = 0


@dataclass
class SimResult:
    t: np.ndarray
    flow: np.ndarray
    paw: np.ndarray
    volume: np.ndarray
    pmus: np.ndarray
    breaths: pd.DataFrame     # start_idx, insp_end_idx, trigger, label, effort_id
    efforts: pd.DataFrame     # effort_id, start_idx, delivered, label
    fs: float

    def asynchrony_index(self) -> float:
        """Thille-style AI (%) = asynchronous events / (delivered breaths + ineffective efforts) * 100."""
        n_ie = int((~self.efforts["delivered"]).sum()) if len(self.efforts) else 0
        n_async_breaths = int(self.breaths["label"].isin(["double_trigger", "auto_trigger", "reverse_trigger"]).sum())
        denom = len(self.breaths) + n_ie
        return 100.0 * (n_ie + n_async_breaths) / denom if denom else float("nan")


def _pmus_value(t: float, starts: np.ndarray, eff: EffortModel) -> float:
    """Sum of active effort pressures at time t (half-sine rise, exponential relaxation)."""
    if starts.size == 0:
        return 0.0
    dtau = t - starts
    active = (dtau >= 0) & (dtau < eff.ti_neural + 5 * eff.decay_s)
    if not active.any():
        return 0.0
    d = dtau[active]
    rise = eff.pmus_amp * np.sin(np.pi * np.clip(d, 0, eff.ti_neural) / eff.ti_neural)
    tail = eff.pmus_amp * 0.05 * np.exp(-(d - eff.ti_neural) / eff.decay_s)
    return float(np.where(d <= eff.ti_neural, rise, tail).sum())


def simulate(cfg: SimConfig) -> SimResult:
    """Run the simulator; see module docstring for the model and ``SimResult`` for the outputs."""
    rng = np.random.default_rng(cfg.seed)
    lung, vent, eff = cfg.lung, cfg.vent, cfg.effort
    fs, dt = cfg.fs, 1.0 / cfg.fs
    n = int(cfg.duration_s * fs)
    flow, paw, vol, pmus = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    # neural effort schedule
    if eff.mode == "spontaneous":
        period = 60.0 / eff.rr_neural
        starts = list(np.cumsum(np.r_[0.3, period * (1 + eff.jitter * rng.normal(size=int(cfg.duration_s / period) + 2))]))
        starts = [s for s in starts if s < cfg.duration_s]
    else:
        starts = []
    effort_starts = np.asarray(starts, float)
    effort_rows = [{"effort_id": k, "start_idx": int(round(s * fs)), "delivered": False, "label": "ineffective_effort"}
                   for k, s in enumerate(effort_starts)]

    state = "EXP"
    V = 0.0
    t_last_mand = -1e9
    t_cycle = -1e9
    t_insp_start = 0.0
    q_peak = 0.0
    breaths: list[dict] = []
    current: dict | None = None
    mand_period = 60.0 / vent.rr_set

    def active_effort(t: float) -> int | None:
        if effort_starts.size == 0:
            return None
        d = t - effort_starts
        m = (d >= 0) & (d < eff.ti_neural + 2 * eff.decay_s)
        return int(np.flatnonzero(m)[-1]) if m.any() else None

    for i in range(n):
        t = i * dt
        P = _pmus_value(t, effort_starts, eff)
        osc = cfg.cardiac_osc_lps * np.sin(2 * np.pi * cfg.heart_rate / 60 * t)
        if state == "EXP":
            Q = (P - V / lung.C) / lung.R + osc + cfg.flow_noise_lps * rng.normal()
            Q = min(Q, vent.bias_flow)  # inspiratory demand during expiration is limited to the bias flow
            Paw = vent.peep
            trig_patient = (Q >= vent.trigger_flow) and (t - t_cycle >= vent.lockout_s)
            trig_time = (t - t_last_mand >= mand_period)
            if trig_patient or trig_time:
                state, t_insp_start, q_peak = "INSP", t, 0.0
                current = {"start_idx": i, "trigger": "patient" if trig_patient else "time", "effort_id": -1, "label": "normal"}
                t_last_mand = t  # assist-control: any breath resets the mandatory timer
                if trig_patient:
                    k = active_effort(t)
                    if k is None:
                        current["label"] = "auto_trigger"
                    else:
                        current["effort_id"] = k
                        if effort_rows[k]["delivered"]:
                            current["label"] = "double_trigger"
                        else:
                            effort_rows[k]["delivered"] = True
                            effort_rows[k]["label"] = "delivered"
                elif eff.mode == "entrained" and rng.random() < eff.entrain_prob:
                    s = t + eff.entrain_delay_s
                    effort_starts = np.append(effort_starts, s)
                    effort_rows.append({"effort_id": len(effort_rows), "start_idx": int(round(s * fs)),
                                        "delivered": True, "label": "reverse_trigger_effort"})
                    current["label"] = "reverse_trigger"
                    current["effort_id"] = len(effort_rows) - 1
        if state == "INSP":
            if vent.mode == "VC":
                Q = vent.vt / vent.ti
                Paw = lung.R * Q + V / lung.C + vent.peep - P
                done = (t - t_insp_start) >= vent.ti
            else:
                Paw = vent.peep + vent.ps
                Q = (vent.ps - V / lung.C + P) / lung.R
                q_peak = max(q_peak, Q)
                done = ((t - t_insp_start) > 0.1 and Q < vent.cycle_pct * q_peak) or (t - t_insp_start) > vent.max_ti
            Q += cfg.flow_noise_lps * rng.normal()
            if done:
                state, t_cycle = "EXP", t
                assert current is not None
                current["insp_end_idx"] = i
                breaths.append(current)
                current = None
        V = max(V + Q * dt, 0.0)
        flow[i], paw[i], vol[i], pmus[i] = Q, Paw, V, P

    if current is not None:  # inspiration cut by the end of the record
        current["insp_end_idx"] = n - 1
        breaths.append(current)
    bdf = pd.DataFrame(breaths, columns=["start_idx", "insp_end_idx", "trigger", "label", "effort_id"])
    edf = pd.DataFrame(effort_rows, columns=["effort_id", "start_idx", "delivered", "label"])
    return SimResult(np.arange(n) / fs, flow, paw, vol, pmus, bdf, edf, fs)


SCENARIOS: dict[str, SimConfig] = {
    "controlled": SimConfig(effort=EffortModel(mode="none")),
    "synchronous_assisted": SimConfig(vent=VentSettings(rr_set=12.0, ti=0.9),
                                      effort=EffortModel(rr_neural=14.0, pmus_amp=8.0, ti_neural=0.9)),
    # obstructive lung (long time constant) + high neural rate + weak efforts: efforts early in expiration fail
    "ineffective_efforts": SimConfig(lung=LungParams(R=20.0, C=0.06), vent=VentSettings(rr_set=12.0, ti=0.9, vt=0.5),
                                     effort=EffortModel(rr_neural=30.0, pmus_amp=3.0, ti_neural=0.6)),
    "double_triggering": SimConfig(vent=VentSettings(rr_set=12.0, ti=0.6, vt=0.4),
                                   effort=EffortModel(rr_neural=14.0, pmus_amp=12.0, ti_neural=1.4)),
    "reverse_triggering": SimConfig(vent=VentSettings(rr_set=15.0, ti=1.0),
                                    effort=EffortModel(mode="entrained", pmus_amp=6.0, ti_neural=0.8, entrain_delay_s=0.5)),
    "auto_triggering": SimConfig(effort=EffortModel(mode="none"), cardiac_osc_lps=0.05),
}
