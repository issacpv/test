"""Tau-PET labels: composites, binary positivity, ordinal T2 stage, N status.

Composites follow the meta-temporal ROI of Jack et al. (2017): entorhinal,
amygdala, parahippocampal, fusiform, inferior temporal and middle temporal
(bilateral, volume-weighted in the original; here a simple mean unless
weights are supplied). The neocortical composite is a mean over lateral
parietal / precuneus / lateral occipital / frontal regions used in
Braak V/VI-like staging. Thresholds are parameters, never constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

META_TEMPORAL_KEYS = ("entorhinal", "amygdala", "parahippocampal", "fusiform", "inferiortemporal", "middletemporal")
NEOCORTICAL_KEYS = ("precuneus", "inferiorparietal", "lateraloccipital", "superiorfrontal", "posteriorcingulate",
                    "supramarginal", "superiorparietal", "rostralmiddlefrontal")


def _cols_matching(df: pd.DataFrame, keys: tuple[str, ...], prefix: str = "suvr_") -> list[str]:
    cols = []
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        low = c.lower()
        if any(k in low for k in keys):
            cols.append(c)
    return cols


def tau_composites(df: pd.DataFrame, prefix: str = "suvr_", weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Return a DataFrame with ``tau_meta_temporal`` and ``tau_neocortical`` composites."""
    mt = _cols_matching(df, META_TEMPORAL_KEYS, prefix)
    neo = _cols_matching(df, NEOCORTICAL_KEYS, prefix)
    if not mt:
        raise ValueError("no meta-temporal SUVR columns found (expected columns like 'suvr_ctx-lh-entorhinal')")
    out = pd.DataFrame(index=df.index)

    def wmean(cols: list[str]) -> pd.Series:
        if weights:
            w = np.array([weights.get(c, 1.0) for c in cols], dtype=float)
            return (df[cols] * w).sum(axis=1) / w.sum()
        return df[cols].mean(axis=1)

    out["tau_meta_temporal"] = wmean(mt)
    out["tau_neocortical"] = wmean(neo) if neo else np.nan
    return out


@dataclass
class TauThresholds:
    """Positivity thresholds (SUVR, cerebellar-cortex reference, no PVC by default)."""

    meta_temporal: float = 1.25
    neocortical: float = 1.20
    grey_zone: tuple[float, float] | None = None  # e.g. (1.20, 1.30) to exclude ambiguous cases
    sensitivity_grid: tuple[float, ...] = field(default=(1.20, 1.25, 1.30))


def binary_positivity(meta_temporal: pd.Series, thr: TauThresholds) -> pd.Series:
    """T+ / T- (1/0) with optional grey-zone masking (NaN)."""
    pos = (meta_temporal >= thr.meta_temporal).astype(float)
    if thr.grey_zone is not None:
        lo, hi = thr.grey_zone
        pos = pos.mask((meta_temporal > lo) & (meta_temporal < hi))
    return pos


def t2_stage(meta_temporal: pd.Series, neocortical: pd.Series, thr: TauThresholds) -> pd.Series:
    """Ordinal tau stage: 0 = none, 1 = medial-temporal only, 2 = neocortical.

    Mirrors the logic of the 2024 AA revised-criteria T2 sub-stages
    (b = MTL, c/d = moderate/high neocortical). Neocortical positivity with
    negative MTL is rare and is coded as stage 2 (it is usually an atypical
    variant), which the analysis reports separately via ``atypical_flag``.
    """
    mt = meta_temporal >= thr.meta_temporal
    neo = neocortical >= thr.neocortical
    stage = np.where(neo, 2, np.where(mt, 1, 0)).astype(float)
    stage = pd.Series(stage, index=meta_temporal.index)
    stage[meta_temporal.isna()] = np.nan
    return stage


def atypical_flag(meta_temporal: pd.Series, neocortical: pd.Series, thr: TauThresholds) -> pd.Series:
    return ((neocortical >= thr.neocortical) & (meta_temporal < thr.meta_temporal)).astype(int)


def threshold_sensitivity(meta_temporal: pd.Series, thr: TauThresholds) -> pd.DataFrame:
    """Prevalence of T+ under each threshold of the sensitivity grid."""
    rows = []
    for t in thr.sensitivity_grid:
        pos = meta_temporal >= t
        rows.append({"threshold": t, "n": int(meta_temporal.notna().sum()), "n_pos": int(pos.sum()),
                     "prevalence": float(pos.mean())})
    return pd.DataFrame(rows)


def n_status(
    hippocampal_volume_norm: pd.Series,
    signature_thickness: pd.Series,
    reference_mask: pd.Series,
    z_cut: float = -1.0,
) -> pd.DataFrame:
    """Neurodegeneration (N) status from hippocampal volume and AD-signature thickness.

    Z-scores are computed relative to ``reference_mask`` rows (intended: A− CN
    participants). N+ if either z < ``z_cut``. Returns z-scores and the flag.
    """
    ref_h = hippocampal_volume_norm[reference_mask]
    ref_t = signature_thickness[reference_mask]
    z_h = (hippocampal_volume_norm - ref_h.mean()) / ref_h.std(ddof=1)
    z_t = (signature_thickness - ref_t.mean()) / ref_t.std(ddof=1)
    out = pd.DataFrame({"z_hippocampus": z_h, "z_signature": z_t})
    out["N_pos"] = ((z_h < z_cut) | (z_t < z_cut)).astype(int)
    return out


def tn_quadrant(t_pos: pd.Series, n_pos: pd.Series) -> pd.Series:
    """Label each row as one of 'T-N-', 'T+N-', 'T-N+', 'T+N+'."""
    lab = np.where(t_pos == 1, "T+", "T-").astype(object) + np.where(n_pos == 1, "N+", "N-").astype(object)
    out = pd.Series(lab, index=t_pos.index)
    out[t_pos.isna() | n_pos.isna()] = np.nan
    return out
