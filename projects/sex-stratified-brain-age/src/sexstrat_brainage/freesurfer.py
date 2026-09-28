"""FreeSurfer feature tables and head-size (TIV) corrections.

Inputs are the wide tables written by ``aparcstats2table`` and
``asegstats2table`` (one row per subject, first column = subject id).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TIVMethod = Literal["none", "covariate", "proportion", "residual", "power"]

ASEG_KEEP = [
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-Thalamus", "Right-Thalamus",
    "Left-Thalamus-Proper", "Right-Thalamus-Proper", "Left-Caudate", "Right-Caudate", "Left-Putamen",
    "Right-Putamen", "Left-Pallidum", "Right-Pallidum", "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala", "Left-Accumbens-area", "Right-Accumbens-area", "Brain-Stem",
    "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior",
    "3rd-Ventricle", "4th-Ventricle", "WM-hypointensities", "CortexVol", "CerebralWhiteMatterVol",
]


def read_stats_table(path: Path | str) -> pd.DataFrame:
    """Read an ``aparcstats2table``/``asegstats2table`` TSV, index = subject id."""
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={df.columns[0]: "subject"}).set_index("subject")
    return df.apply(pd.to_numeric, errors="coerce")


def assemble_features(
    aparc_thickness_lh: pd.DataFrame,
    aparc_thickness_rh: pd.DataFrame,
    aseg_volume: pd.DataFrame,
    aparc_area_lh: pd.DataFrame | None = None,
    aparc_area_rh: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join thickness (and optional area) with selected aseg volumes and eTIV.

    Output columns: ``thk_<hemi>_<region>``, ``area_<hemi>_<region>``, ``vol_<structure>``, ``eTIV``, ``euler``
    (if ``lh_euler``/``rh_euler`` columns exist), ``mean_thickness``.
    """
    parts = []
    for hemi, tab in (("lh", aparc_thickness_lh), ("rh", aparc_thickness_rh)):
        cols = [c for c in tab.columns if c.endswith("_thickness") and not c.startswith(f"{hemi}_MeanThickness")]
        sub = tab[cols].copy()
        sub.columns = ["thk_" + c.replace("_thickness", "") for c in cols]
        parts.append(sub)
    if aparc_area_lh is not None and aparc_area_rh is not None:
        for hemi, tab in (("lh", aparc_area_lh), ("rh", aparc_area_rh)):
            cols = [c for c in tab.columns if c.endswith("_area") and "WhiteSurfArea" not in c]
            sub = tab[cols].copy()
            sub.columns = ["area_" + c.replace("_area", "") for c in cols]
            parts.append(sub)
    keep = [c for c in ASEG_KEEP if c in aseg_volume.columns]
    vol = aseg_volume[keep].copy()
    vol.columns = ["vol_" + c for c in keep]
    parts.append(vol)
    out = pd.concat(parts, axis=1, join="inner")
    tiv_col = "EstimatedTotalIntraCranialVol" if "EstimatedTotalIntraCranialVol" in aseg_volume.columns else "eTIV"
    out["eTIV"] = aseg_volume.loc[out.index, tiv_col]
    thk = [c for c in out.columns if c.startswith("thk_")]
    out["mean_thickness"] = out[thk].mean(axis=1)
    return out


def euler_qc_mask(euler: pd.Series, percentile: float = 5.0) -> pd.Series:
    """Keep subjects whose (mean) Euler number is above the cohort's low-tail cut-off.

    Euler number is negative; more negative = worse surface quality (Rosen et al., 2018).
    """
    cut = np.nanpercentile(euler, percentile)
    return euler >= cut


class TIVCorrector:
    """Head-size correction for *volume* features, fitted on training rows only.

    Methods
    -------
    none        raw volumes; eTIV is dropped so the model has no head-size information.
    covariate   raw volumes with eTIV kept as an input feature.
    proportion  vol / eTIV.
    residual    vol − b·(eTIV − mean eTIV), b fitted by OLS on training rows (Sanchis-Segura et al., 2019).
    power       vol / eTIV**b, b fitted as the log-log slope on training rows (Liu et al., 2014).
    Thickness features are never corrected (they are not head-size dependent to first order).
    """

    def __init__(self, method: TIVMethod = "residual", volume_prefixes: tuple[str, ...] = ("vol_", "area_")):
        self.method = method
        self.volume_prefixes = volume_prefixes

    def _vol_cols(self, X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if c.startswith(self.volume_prefixes)]

    def fit(self, X: pd.DataFrame) -> "TIVCorrector":
        cols = self._vol_cols(X)
        tiv = X["eTIV"].to_numpy(dtype=float)
        self.cols_ = cols
        self.tiv_mean_ = float(tiv.mean())
        if self.method == "residual":
            xc = tiv - self.tiv_mean_
            self.beta_ = {c: float(np.dot(xc, X[c].to_numpy(dtype=float) - X[c].mean()) / np.dot(xc, xc)) for c in cols}
        elif self.method == "power":
            lt = np.log(tiv)
            ltc = lt - lt.mean()
            self.beta_ = {}
            for c in cols:
                lv = np.log(np.clip(X[c].to_numpy(dtype=float), 1e-9, None))
                self.beta_[c] = float(np.dot(ltc, lv - lv.mean()) / np.dot(ltc, ltc))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        tiv = X["eTIV"].to_numpy(dtype=float)
        for c in self.cols_:
            v = X[c].to_numpy(dtype=float)
            if self.method in ("none", "covariate"):
                continue
            if self.method == "proportion":
                out[c] = 1000.0 * v / tiv
            elif self.method == "residual":
                out[c] = v - self.beta_[c] * (tiv - self.tiv_mean_)
            elif self.method == "power":
                out[c] = v / (tiv / self.tiv_mean_) ** self.beta_[c]
        if self.method != "covariate":
            out = out.drop(columns=["eTIV"])
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def feature_set(X: pd.DataFrame, which: Literal["thickness", "volumes", "all"] = "all") -> pd.DataFrame:
    """Select a feature family (thickness only, volumes/areas only, or everything)."""
    if which == "thickness":
        return X[[c for c in X.columns if c.startswith("thk_") or c == "mean_thickness"]]
    if which == "volumes":
        return X[[c for c in X.columns if c.startswith(("vol_", "area_", "eTIV"))]]
    return X
