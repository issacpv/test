"""Structural-MRI feature extraction: FreeSurfer stats parsing, composites, ComBat.

All functions are pure pandas/numpy so they can be unit-tested on synthetic
tables. ``parse_aseg_stats`` / ``parse_aparc_stats`` read the text files that
FreeSurfer writes to ``<subject>/stats/``; OASIS-3 and ADNI both ship them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AD_SIGNATURE_REGIONS = ("entorhinal", "inferiortemporal", "middletemporal", "fusiform")
"""Cortical-signature regions (Jack et al., 2017 meta-ROI thickness; Dickerson et al., 2009 signature)."""


def parse_aseg_stats(path: Path | str) -> dict[str, float]:
    """Parse ``aseg.stats`` into ``{structure: volume_mm3}`` plus ``eTIV``.

    The eTIV line looks like
    ``# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1520000.0, mm^3``.
    """
    out: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure"):].split(",")]
            if len(parts) >= 4 and parts[1] in {"eTIV", "BrainSegVolNotVent", "TotalGrayVol", "CortexVol"}:
                out[parts[1]] = float(parts[3])
            continue
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        # Index SegId NVoxels Volume_mm3 StructName ...
        if len(cols) >= 5:
            try:
                out[cols[4]] = float(cols[3])
            except ValueError:
                continue
    return out


def parse_aparc_stats(path: Path | str, hemi: str) -> dict[str, float]:
    """Parse ``?h.aparc.stats`` into ``{f"{hemi}_{region}_thickness": mm, f"{hemi}_{region}_area": mm2}``."""
    out: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        # StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
        if len(cols) >= 5:
            try:
                out[f"{hemi}_{cols[0]}_thickness"] = float(cols[4])
                out[f"{hemi}_{cols[0]}_area"] = float(cols[2])
            except ValueError:
                continue
    return out


def load_freesurfer_session(stats_dir: Path | str) -> dict[str, float]:
    """Combine aseg + lh/rh aparc from a FreeSurfer ``stats`` directory."""
    d = Path(stats_dir)
    feats = parse_aseg_stats(d / "aseg.stats")
    for hemi in ("lh", "rh"):
        p = d / f"{hemi}.aparc.stats"
        if p.exists():
            feats.update(parse_aparc_stats(p, hemi))
    return feats


def signature_thickness(df: pd.DataFrame, regions: tuple[str, ...] = AD_SIGNATURE_REGIONS) -> pd.Series:
    """Mean bilateral thickness over the AD-signature regions."""
    cols = [c for c in df.columns if c.endswith("_thickness") and any(f"_{r}_" in c for r in regions)]
    if not cols:
        raise ValueError("no signature thickness columns found")
    return df[cols].mean(axis=1)


def hippocampal_volume_norm(df: pd.DataFrame) -> pd.Series:
    """Bilateral hippocampal volume as a fraction of eTIV (×1000 for readability)."""
    return 1000.0 * (df["Left-Hippocampus"] + df["Right-Hippocampus"]) / df["eTIV"]


def asymmetry_index(df: pd.DataFrame, region: str) -> pd.Series:
    """(L − R) / mean(L, R) thickness asymmetry for ``region``."""
    l, r = df[f"lh_{region}_thickness"], df[f"rh_{region}_thickness"]
    return 2.0 * (l - r) / (l + r)


def build_roi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the ROI feature matrix from a wide FreeSurfer table.

    Includes every ``*_thickness`` column, eTIV-normalized subcortical volumes
    (any column in ``SUBCORTICAL`` present), log WMH volume, the AD-signature
    composite, hippocampal occupancy proxy and temporal asymmetry indices.
    """
    feats = pd.DataFrame(index=df.index)
    thick = [c for c in df.columns if c.endswith("_thickness")]
    feats[thick] = df[thick]
    for s in SUBCORTICAL:
        if s in df.columns:
            feats[f"{s}_norm"] = 1000.0 * df[s] / df["eTIV"]
    if "WM-hypointensities" in df.columns:
        feats["log_wmh"] = np.log1p(df["WM-hypointensities"])
    feats["signature_thickness"] = signature_thickness(df)
    feats["hippocampus_norm"] = hippocampal_volume_norm(df)
    for region in ("entorhinal", "inferiortemporal", "middletemporal"):
        if f"lh_{region}_thickness" in df.columns and f"rh_{region}_thickness" in df.columns:
            feats[f"asym_{region}"] = asymmetry_index(df, region)
    return feats


SUBCORTICAL = ("Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala",
               "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-Thalamus", "Right-Thalamus",
               "Left-Putamen", "Right-Putamen", "Left-Caudate", "Right-Caudate")


def residualize(
    X: pd.DataFrame,
    covars: pd.DataFrame,
    fit_mask: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Regress each feature on covariates using only ``fit_mask`` rows; return residuals.

    Intended use: fit on T− controls so that age/sex/eTIV effects estimated
    in unaffected participants are removed from everyone (the proxy test).
    """
    C = np.column_stack([np.ones(len(covars)), covars.to_numpy(dtype=float)])
    Xv = X.to_numpy(dtype=float)
    mask = np.asarray(fit_mask, dtype=bool)
    beta, *_ = np.linalg.lstsq(C[mask], Xv[mask], rcond=None)
    return pd.DataFrame(Xv - C @ beta, index=X.index, columns=X.columns)


class ComBat:
    """Parametric ComBat (Johnson et al., 2007) with covariate preservation.

    ``fit`` estimates location/scale batch effects on the training rows; ``transform``
    applies them to any rows, so it can be used inside a CV loop without leakage.
    Empirical-Bayes shrinkage uses the parametric prior of the original method.
    """

    def __init__(self, eb: bool = True, max_iter: int = 100, tol: float = 1e-4):
        self.eb = eb
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, batch: np.ndarray, covars: np.ndarray | None = None) -> "ComBat":
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch)
        self.batches_ = np.unique(batch)
        n, p = X.shape
        B = np.stack([(batch == b).astype(float) for b in self.batches_], axis=1)
        design = B if covars is None else np.column_stack([B, covars])
        beta, *_ = np.linalg.lstsq(design, X, rcond=None)
        self.n_batch_ = len(self.batches_)
        self.beta_cov_ = beta[self.n_batch_:] if covars is not None else np.zeros((0, p))
        counts = B.sum(axis=0)
        self.grand_mean_ = (counts / n) @ beta[: self.n_batch_]
        stand_mean = np.outer(np.ones(n), self.grand_mean_)
        if covars is not None:
            stand_mean = stand_mean + covars @ self.beta_cov_
        resid = X - (design @ beta)
        self.var_pooled_ = (resid ** 2).sum(axis=0) / n
        s_data = (X - stand_mean) / np.sqrt(self.var_pooled_)
        gamma_hat = np.linalg.lstsq(B, s_data, rcond=None)[0]
        delta_hat = np.stack([s_data[batch == b].var(axis=0, ddof=1) if (batch == b).sum() > 1 else np.ones(p)
                              for b in self.batches_])
        if self.eb:
            gamma_star, delta_star = np.empty_like(gamma_hat), np.empty_like(delta_hat)
            for i, b in enumerate(self.batches_):
                g, d = gamma_hat[i], delta_hat[i]
                g_bar, t2 = g.mean(), g.var(ddof=1) if p > 1 else 1.0
                m, v = d.mean(), d.var(ddof=1) if p > 1 else 1.0
                a_prior = (2 * v + m ** 2) / v if v > 0 else 1.0
                b_prior = (m * v + m ** 3) / v if v > 0 else 1.0
                sd = s_data[batch == b]
                nb = sd.shape[0]
                g_new, d_new = g.copy(), d.copy()
                for _ in range(self.max_iter):
                    g_old, d_old = g_new.copy(), d_new.copy()
                    g_new = (nb * t2 * g + d_new * g_bar) / (nb * t2 + d_new)
                    sum2 = ((sd - g_new) ** 2).sum(axis=0)
                    d_new = (0.5 * sum2 + b_prior) / (nb / 2 + a_prior - 1)
                    if max(np.abs(g_new - g_old).max(), np.abs(d_new - d_old).max()) < self.tol:
                        break
                gamma_star[i], delta_star[i] = g_new, d_new
        else:
            gamma_star, delta_star = gamma_hat, delta_hat
        self.gamma_ = gamma_star
        self.delta_ = delta_star
        return self

    def transform(self, X: np.ndarray, batch: np.ndarray, covars: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch)
        n, p = X.shape
        stand_mean = np.outer(np.ones(n), self.grand_mean_)
        if covars is not None and self.beta_cov_.shape[0]:
            stand_mean = stand_mean + covars @ self.beta_cov_
        s_data = (X - stand_mean) / np.sqrt(self.var_pooled_)
        out = np.empty_like(s_data)
        for i, b in enumerate(self.batches_):
            m = batch == b
            if m.any():
                out[m] = (s_data[m] - self.gamma_[i]) / np.sqrt(self.delta_[i])
        unknown = ~np.isin(batch, self.batches_)
        if unknown.any():
            out[unknown] = s_data[unknown]  # unseen batch: no correction
        return out * np.sqrt(self.var_pooled_) + stand_mean
