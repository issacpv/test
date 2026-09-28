"""Order-pattern (observation-process) features and ordering-policy manipulations.

Every function takes the *long* measurement table produced by ``lab_leak.sql``
(or ``lab_leak.simulate``) with at least the columns
``stay_id, t_hours, itemid, valuenum`` and optionally ``priority`` and ``hod``.

The feature builder returns two aligned wide tables: ``values`` (what the lab
said) and ``masks`` (that the lab was ordered, how often, when, and with what
priority). Keeping them separate is the whole point: models are trained on
``values`` only, ``masks`` only, or both, and the attribution code in
``lab_leak.attribution`` measures how much each channel contributes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VALUE_STATS = ("last", "mean", "min", "max")


@dataclass
class FeatureConfig:
    """Window and schedule definitions."""

    window_hours: float = 24.0
    morning_window: tuple[int, int] = (4, 8)
    stat_label: str = "STAT"
    min_stays_per_item: int = 1


def annotate_schedule(long: pd.DataFrame, stays: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Add ``hod`` (if missing), ``is_stat``, ``routine`` and ``off_schedule`` columns.

    A measurement is *routine* when it falls inside the morning draw window and
    is not STAT; everything else is *off-schedule* (ad hoc, clinician-initiated).
    On MIMIC-IV the hour of day is preserved by the date shift, so ``hod`` can be
    taken from ``charttime``; on eICU it is ``unitadmittime24`` plus the offset.
    """
    out = long.copy()
    if "hod" not in out.columns:
        admit = stays.set_index("stay_id")["admit_hour"]
        out["hod"] = (out["stay_id"].map(admit).to_numpy() + np.floor(out["t_hours"].to_numpy())) % 24
    if "priority" in out.columns:
        out["is_stat"] = out["priority"].astype(str).str.upper().eq(cfg.stat_label.upper())
    else:
        out["is_stat"] = False
    lo, hi = cfg.morning_window
    in_window = (out["hod"] >= lo) & (out["hod"] < hi)
    out["routine"] = in_window & ~out["is_stat"]
    out["off_schedule"] = ~out["routine"]
    return out


def build_features(long: pd.DataFrame, stays: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide ``(values, masks)`` tables indexed by ``stay_id`` (all stays, in ``stays`` order).

    values : ``<item>_last/mean/min/max`` (NaN when never measured in the window)
    masks  : ``<item>_n, <item>_any, <item>_hrs_since, <item>_n_stat, <item>_n_off``
             plus stay totals ``n_total, n_stat_total, n_off_total, n_distinct``.
    """
    df = annotate_schedule(long, stays, cfg)
    df = df[(df["t_hours"] >= 0) & (df["t_hours"] <= cfg.window_hours)]
    items = sorted(df["itemid"].unique())
    index = pd.Index(stays["stay_id"].to_numpy(), name="stay_id")

    g = df.sort_values("t_hours").groupby(["stay_id", "itemid"])
    agg = g["valuenum"].agg(last="last", mean="mean", min="min", max="max")
    cnt = g.agg(n=("valuenum", "size"), t_last=("t_hours", "max"), n_stat=("is_stat", "sum"), n_off=("off_schedule", "sum"))

    values = {}
    masks = {}
    for it in items:
        a = agg.xs(it, level="itemid").reindex(index) if it in agg.index.get_level_values(1) else None
        c = cnt.xs(it, level="itemid").reindex(index)
        for s in VALUE_STATS:
            values[f"{it}_{s}"] = a[s].to_numpy() if a is not None else np.full(len(index), np.nan)
        n = c["n"].fillna(0).to_numpy()
        masks[f"{it}_n"] = n
        masks[f"{it}_any"] = (n > 0).astype(float)
        masks[f"{it}_hrs_since"] = np.where(n > 0, cfg.window_hours - c["t_last"].fillna(0).to_numpy(), cfg.window_hours)
        masks[f"{it}_n_stat"] = c["n_stat"].fillna(0).to_numpy()
        masks[f"{it}_n_off"] = c["n_off"].fillna(0).to_numpy()
    values_df = pd.DataFrame(values, index=index)
    masks_df = pd.DataFrame(masks, index=index)
    tot = df.groupby("stay_id").agg(n_total=("valuenum", "size"), n_stat_total=("is_stat", "sum"),
                                    n_off_total=("off_schedule", "sum"), n_distinct=("itemid", "nunique"))
    masks_df = masks_df.join(tot.reindex(index).fillna(0))
    return values_df, masks_df


def thin_orders(long: pd.DataFrame, keep_frac: float, rng: np.random.Generator,
                stays: pd.DataFrame | None = None, cfg: FeatureConfig | None = None,
                subset: str = "all") -> pd.DataFrame:
    """Randomly drop measurements to emulate a lower-intensity ordering policy.

    ``subset='all'`` thins every measurement; ``'off_schedule'`` thins only ad hoc
    orders (routine morning panels are kept), which is the realistic shape of a
    "reduce unnecessary labs" intervention.
    """
    if not 0.0 <= keep_frac <= 1.0:
        raise ValueError("keep_frac must be in [0, 1]")
    if subset == "all":
        keep = rng.random(len(long)) < keep_frac
    elif subset == "off_schedule":
        if stays is None or cfg is None:
            raise ValueError("subset='off_schedule' needs stays and cfg")
        ann = annotate_schedule(long, stays, cfg)
        keep = ann["routine"].to_numpy() | (rng.random(len(long)) < keep_frac)
    else:
        raise ValueError(f"unknown subset {subset!r}")
    return long.loc[keep].reset_index(drop=True)


def schedule_standardize(long: pd.DataFrame, stays: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Keep only routine (in-window, non-STAT) measurements: the 'order-agnostic' input view."""
    ann = annotate_schedule(long, stays, cfg)
    return long.loc[ann["routine"].to_numpy()].reset_index(drop=True)


def ordering_intensity(long: pd.DataFrame, stays: pd.DataFrame, cfg: FeatureConfig) -> pd.Series:
    """Measurements per stay in the window (the quantity that shifts between hospitals)."""
    df = long[(long["t_hours"] >= 0) & (long["t_hours"] <= cfg.window_hours)]
    return df.groupby("stay_id").size().reindex(stays["stay_id"]).fillna(0).rename("n_orders")


def leakage_timing_profile(long: pd.DataFrame, stays: pd.DataFrame, label_col: str = "y",
                           bins_hours: tuple[float, ...] = (0, 6, 12, 18, 24)) -> pd.DataFrame:
    """Order rate per stay-hour by time bin and outcome: where in the window do positives get extra orders?"""
    df = long.merge(stays[["stay_id", label_col]], on="stay_id")
    df["bin"] = pd.cut(df["t_hours"], bins=list(bins_hours), right=False)
    n_stays = stays.groupby(label_col).size()
    counts = df.groupby([label_col, "bin"], observed=False).size().unstack("bin").fillna(0)
    width = np.diff(bins_hours)
    return counts.div(n_stays, axis=0).div(width, axis=1)
