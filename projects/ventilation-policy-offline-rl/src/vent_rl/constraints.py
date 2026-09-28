"""ARDSNet lung-protective ventilation guideline as machine-checkable constraints.

Sources: ARDS Network, NEJM 2000 (ARMA: VT 6 mL/kg PBW, plateau <= 30 cmH2O, lower PEEP/higher FiO2 table)
and Brower et al., NEJM 2004 (ALVEOLI: higher PEEP/lower FiO2 table). Targets: SpO2 88-95 % or
PaO2 55-80 mmHg; pH 7.30-7.45; RR <= 35.

The constraint set is a data structure (`GuidelineLimits` + tables) so clinicians can edit it; the
`action_mask` function turns it into a per-state boolean mask over the discretised action grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# (FiO2, PEEP) combinations in protocol order
ARDSNET_LOW_PEEP: Tuple[Tuple[float, float], ...] = (
    (0.3, 5), (0.4, 5), (0.4, 8), (0.5, 8), (0.5, 10), (0.6, 10), (0.7, 10), (0.7, 12), (0.7, 14),
    (0.8, 14), (0.9, 14), (0.9, 16), (0.9, 18), (1.0, 18), (1.0, 20), (1.0, 22), (1.0, 24),
)
ARDSNET_HIGH_PEEP: Tuple[Tuple[float, float], ...] = (
    (0.3, 5), (0.3, 8), (0.3, 10), (0.3, 12), (0.3, 14), (0.4, 14), (0.4, 16), (0.5, 16), (0.5, 18),
    (0.5, 20), (0.6, 20), (0.7, 20), (0.8, 20), (0.8, 22), (0.9, 22), (1.0, 22), (1.0, 24),
)
TABLES: Dict[str, Tuple[Tuple[float, float], ...]] = {"low": ARDSNET_LOW_PEEP, "high": ARDSNET_HIGH_PEEP}


@dataclass
class GuidelineLimits:
    vt_min: float = 4.0            # mL/kg PBW
    vt_max: float = 8.0
    vt_target: float = 6.0
    plateau_max: float = 30.0      # cmH2O
    rr_max: float = 35.0
    ph_min: float = 7.30
    ph_max: float = 7.45
    spo2_min: float = 0.88
    spo2_max: float = 0.95
    pao2_min: float = 55.0
    pao2_max: float = 80.0
    peep_tolerance: float = 2.0    # cmH2O slack around the table
    fio2_tolerance: float = 0.05
    table: str = "low"
    extra: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # clinician-added {feature: (lo, hi)}


def peep_allowed(fio2: float, table: str = "low", tolerance: float = 2.0) -> Tuple[float, float]:
    """(min, max) PEEP the ARDSNet table allows at this FiO2 (nearest table FiO2), widened by `tolerance`."""
    tab = TABLES[table]
    fio2s = np.array([f for f, _ in tab])
    nearest = fio2s[np.argmin(np.abs(fio2s - fio2))]
    peeps = [p for f, p in tab if f == nearest]
    return float(min(peeps) - tolerance), float(max(peeps) + tolerance)


def fio2_allowed(peep: float, table: str = "low", tolerance: float = 0.05) -> Tuple[float, float]:
    """(min, max) FiO2 the table allows at this PEEP."""
    tab = TABLES[table]
    peeps = np.array([p for _, p in tab])
    nearest = peeps[np.argmin(np.abs(peeps - peep))]
    f = [ff for ff, p in tab if p == nearest]
    return float(max(0.21, min(f) - tolerance)), float(min(1.0, max(f) + tolerance))


def check_setting(vt_per_kg: Optional[float] = None, peep: Optional[float] = None, fio2: Optional[float] = None,
                  plateau: Optional[float] = None, rr: Optional[float] = None, ph: Optional[float] = None,
                  spo2: Optional[float] = None, pao2: Optional[float] = None,
                  limits: GuidelineLimits = GuidelineLimits(), **extra: float) -> List[str]:
    """List of guideline violations for one (state, setting) combination; empty list = compliant."""
    v: List[str] = []
    if vt_per_kg is not None and not np.isnan(vt_per_kg):
        if vt_per_kg > limits.vt_max:
            v.append(f"VT {vt_per_kg:.1f} > {limits.vt_max} mL/kg PBW")
        if vt_per_kg < limits.vt_min:
            v.append(f"VT {vt_per_kg:.1f} < {limits.vt_min} mL/kg PBW")
    if plateau is not None and not np.isnan(plateau) and plateau > limits.plateau_max:
        v.append(f"Pplat {plateau:.0f} > {limits.plateau_max}")
    if rr is not None and not np.isnan(rr) and rr > limits.rr_max:
        v.append(f"RR {rr:.0f} > {limits.rr_max}")
    if peep is not None and fio2 is not None and not (np.isnan(peep) or np.isnan(fio2)):
        lo, hi = peep_allowed(fio2, limits.table, limits.peep_tolerance)
        if not (lo <= peep <= hi):
            v.append(f"PEEP {peep:.0f} outside ARDSNet-{limits.table} range [{lo:.0f},{hi:.0f}] at FiO2 {fio2:.2f}")
    if ph is not None and not np.isnan(ph) and not (limits.ph_min <= ph <= limits.ph_max):
        v.append(f"pH {ph:.2f} outside [{limits.ph_min},{limits.ph_max}]")
    if spo2 is not None and not np.isnan(spo2) and not (limits.spo2_min <= spo2 <= limits.spo2_max):
        v.append(f"SpO2 {spo2:.2f} outside [{limits.spo2_min},{limits.spo2_max}]")
    if pao2 is not None and not np.isnan(pao2) and not (limits.pao2_min <= pao2 <= limits.pao2_max):
        v.append(f"PaO2 {pao2:.0f} outside [{limits.pao2_min},{limits.pao2_max}]")
    for k, (lo, hi) in limits.extra.items():
        if k in extra and extra[k] is not None and not (lo <= extra[k] <= hi):
            v.append(f"{k} {extra[k]} outside [{lo},{hi}]")
    return v


def action_mask(grid: pd.DataFrame, limits: GuidelineLimits = GuidelineLimits(),
                state: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """Boolean mask over the action grid (from `mdp_builder.action_grid`): True = allowed.

    Static rules: VT bin representative within [vt_min, vt_max]; PEEP/FiO2 pair consistent with the
    ARDSNet table. State-dependent rules (optional `state` dict with 'spo2', 'pao2', 'plateau'):
    if oxygenation is above target, disallow raising FiO2 to the top bin; if plateau > max, disallow the
    top VT bin even if within [vt_min, vt_max].
    """
    ok = np.ones(len(grid), bool)
    for i, (a, row) in enumerate(grid.iterrows()):
        if not (limits.vt_min <= row["vt_per_kg"] <= limits.vt_max):
            ok[i] = False
            continue
        lo, hi = peep_allowed(row["fio2"], limits.table, limits.peep_tolerance)
        if not (lo <= row["peep"] <= hi):
            ok[i] = False
            continue
        if state:
            spo2, pao2, plat = state.get("spo2"), state.get("pao2"), state.get("plateau_pressure")
            high_ox = (spo2 is not None and not np.isnan(spo2) and spo2 > limits.spo2_max) or \
                      (pao2 is not None and not np.isnan(pao2) and pao2 > limits.pao2_max)
            if high_ox and row["fio2_bin"] == grid["fio2_bin"].max():
                ok[i] = False
            if plat is not None and not np.isnan(plat) and plat > limits.plateau_max and row["vt_bin"] == grid["vt_bin"].max():
                ok[i] = False
    if not ok.any():  # never leave the agent without an action: fall back to the static rules only
        return action_mask(grid, limits, None)
    return ok


def state_action_masks(grid: pd.DataFrame, state_features: pd.DataFrame, limits: GuidelineLimits = GuidelineLimits()
                       ) -> np.ndarray:
    """(n_states, n_actions) mask from per-discrete-state representative features (e.g. cluster centroids)."""
    masks = []
    for _, row in state_features.iterrows():
        masks.append(action_mask(grid, limits, row.to_dict()))
    return np.vstack(masks)


def compliance_rate(df: pd.DataFrame, limits: GuidelineLimits = GuidelineLimits(),
                    cols: Mapping[str, str] = None) -> Dict[str, float]:
    """Fraction of rows with no violation, plus per-rule violation rates, over a wide state table."""
    cols = cols or {"vt_per_kg": "vt_per_kg", "peep": "peep", "fio2": "fio2", "plateau": "plateau_pressure",
                    "rr": "resp_rate", "ph": "ph", "spo2": "spo2", "pao2": "po2"}
    n_ok, per_rule = 0, {}
    for _, r in df.iterrows():
        kw = {k: (float(r[c]) if c in r and pd.notna(r[c]) else None) for k, c in cols.items()}
        v = check_setting(limits=limits, **kw)
        n_ok += int(len(v) == 0)
        for msg in v:
            key = msg.split(" ")[0]
            per_rule[key] = per_rule.get(key, 0) + 1
    out = {"compliant": n_ok / max(1, len(df))}
    out.update({f"viol_{k}": v / max(1, len(df)) for k, v in per_rule.items()})
    return out
