"""FreeSurfer ROI features: stats parsing, ICV normalization, residualization and ComBat.

The ComBat implementation follows Johnson, Li & Rabinovic (2007, Biostatistics) with parametric
empirical-Bayes shrinkage, as used for neuroimaging by Fortin et al. (2017/2018, NeuroImage).
It is written as a fit/transform estimator so that it can be fitted inside a cross-validation
training fold and applied to held-out data (avoiding harmonization leakage).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .oasis_tables import parse_oasis_id

# ---------------------------------------------------------------------------
# FreeSurfer stats parsing
# ---------------------------------------------------------------------------
_MEASURE_RE = re.compile(r"^# Measure\s+([^,]+),\s*([^,]+),\s*([^,]+),\s*([-\d.eE+]+),")


def parse_aseg_stats(path: str | Path) -> dict[str, float]:
    """Parse ``aseg.stats`` into ``{structure: volume_mm3}`` plus global measures (eTIV, ...)."""
    out: dict[str, float] = {}
    headers: list[str] = []
    for line in Path(path).read_text().splitlines():
        m = _MEASURE_RE.match(line)
        if m:
            out[m.group(2).strip()] = float(m.group(4))
            continue
        if line.startswith("# ColHeaders"):
            headers = line.replace("# ColHeaders", "").split()
            continue
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if headers and len(parts) >= len(headers):
            row = dict(zip(headers, parts))
            out[row["StructName"]] = float(row["Volume_mm3"])
    return out


def parse_aparc_stats(path: str | Path, measure: str = "ThickAvg", hemi: Optional[str] = None) -> dict[str, float]:
    """Parse ``?h.aparc.stats`` into ``{f"{hemi}_{region}_{measure}": value}``."""
    path = Path(path)
    if hemi is None:
        hemi = path.name.split(".")[0]
    out: dict[str, float] = {}
    headers: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("# ColHeaders"):
            headers = line.replace("# ColHeaders", "").split()
            continue
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if headers and len(parts) >= len(headers):
            row = dict(zip(headers, parts))
            out[f"{hemi}_{row['StructName']}_{measure}"] = float(row[measure])
    return out


def load_freesurfer_dir(root: str | Path, pattern: str = "*Freesurfer*") -> pd.DataFrame:
    """Load every ``<root>/<fs_id>/**/stats/{aseg,lh.aparc,rh.aparc}.stats`` into one wide table.

    The index is the FreeSurfer id (e.g. ``OAS30001_Freesurfer53_d0129``); ``subject`` and ``day``
    columns are parsed from it so the table can be joined to :func:`build_subject_table` output.
    """
    rows = []
    for d in sorted(Path(root).glob(pattern)):
        stats = list(d.rglob("aseg.stats"))
        if not stats:
            continue
        stats_dir = stats[0].parent
        feats = {"fs_id": d.name}
        feats.update(parse_aseg_stats(stats_dir / "aseg.stats"))
        for hemi in ("lh", "rh"):
            f = stats_dir / f"{hemi}.aparc.stats"
            if f.exists():
                feats.update(parse_aparc_stats(f, "ThickAvg", hemi))
        try:
            subj, _, day = parse_oasis_id(d.name)
            feats["subject"], feats["day"] = subj, day
        except ValueError:
            pass
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df.set_index("fs_id") if len(df) else df


SUBCORTICAL_DEFAULT: tuple[str, ...] = (
    "Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala",
    "Left-Thalamus", "Right-Thalamus", "Left-Thalamus-Proper", "Right-Thalamus-Proper",
    "Left-Caudate", "Right-Caudate", "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum", "Left-Accumbens-area", "Right-Accumbens-area",
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent",
    "WM-hypointensities", "3rd-Ventricle", "4th-Ventricle",
)


def default_roi_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (thickness columns, subcortical volume columns) present in ``df``."""
    thick = [c for c in df.columns if c.endswith("_ThickAvg")]
    vols = [c for c in SUBCORTICAL_DEFAULT if c in df.columns]
    return thick, vols


def icv_normalize(df: pd.DataFrame, volume_cols: Sequence[str], icv_col: str = "eTIV",
                  scale: float = 1e3) -> pd.DataFrame:
    """Replace volumes by ``scale * volume / ICV`` (proportion method; ×1e3 keeps numbers readable)."""
    out = df.copy()
    icv = pd.to_numeric(out[icv_col], errors="coerce")
    for c in volume_cols:
        out[c] = scale * pd.to_numeric(out[c], errors="coerce") / icv
    return out


def residualize(df: pd.DataFrame, cols: Sequence[str], covariates: Sequence[str],
                fit_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Regress ``cols`` on ``covariates`` (OLS with intercept), fitted on ``fit_mask`` rows
    (e.g. amyloid-negative controls), and return residuals for all rows.

    This is the age/atrophy-proxy test: after residualizing on age, sex and ICV using A- controls,
    any remaining discrimination cannot be explained by those covariates.
    """
    X = np.column_stack([np.ones(len(df))] + [pd.to_numeric(df[c], errors="coerce").to_numpy(float)
                                              for c in covariates])
    Y = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    mask = np.ones(len(df), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    ok = mask & np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    beta, *_ = np.linalg.lstsq(X[ok], Y[ok], rcond=None)
    resid = Y - X @ beta
    return pd.DataFrame(resid, columns=list(cols), index=df.index)


# ---------------------------------------------------------------------------
# ComBat
# ---------------------------------------------------------------------------
def _aprior(delta_hat: np.ndarray) -> float:
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (m * s2 + m ** 3) / s2


def _it_sol(sdat: np.ndarray, g_hat: np.ndarray, d_hat: np.ndarray, g_bar: float, t2: float,
            a: float, b: float, conv: float = 1e-4, max_iter: int = 500) -> tuple[np.ndarray, np.ndarray]:
    n = sdat.shape[0]
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        sum2 = ((sdat - g_new[None, :]) ** 2).sum(0)
        d_new = (0.5 * sum2 + b) / (n / 2 + a - 1)
        change = max(np.abs(g_new - g_old).max() / (np.abs(g_old).max() + 1e-12),
                     np.abs(d_new - d_old).max() / (np.abs(d_old).max() + 1e-12))
        g_old, d_old = g_new, d_new
        if change < conv:
            break
    return g_old, d_old


class ComBat:
    """Parametric empirical-Bayes ComBat with fit/transform semantics.

    Parameters
    ----------
    eb : bool
        Use empirical-Bayes shrinkage of batch parameters (standard). ``False`` gives per-batch
        location/scale adjustment without shrinkage.

    Notes
    -----
    ``covars`` (biological covariates to preserve, e.g. age, sex, and, if the analysis wants to
    keep it, amyloid status) must be numeric and are used only to estimate the standardization.
    Transforming a batch unseen at fit time raises ``KeyError``; fit on a training fold that
    contains every batch, or use a few held-out controls from the new batch to fit.
    """

    def __init__(self, eb: bool = True):
        self.eb = eb

    def fit(self, X: np.ndarray, batch: Sequence, covars: Optional[np.ndarray] = None) -> "ComBat":
        X = np.asarray(X, float)
        batch = np.asarray(batch)
        self.batches_ = list(pd.unique(batch))
        n, p = X.shape
        B = len(self.batches_)
        onehot = np.column_stack([(batch == b).astype(float) for b in self.batches_])
        C = None if covars is None else np.asarray(covars, float).reshape(n, -1)
        design = onehot if C is None else np.column_stack([onehot, C])
        beta, *_ = np.linalg.lstsq(design, X, rcond=None)
        n_b = onehot.sum(0)
        self.grand_mean_ = (n_b / n) @ beta[:B]
        self.beta_cov_ = None if C is None else beta[B:]
        fitted = design @ beta
        self.var_pooled_ = ((X - fitted) ** 2).sum(0) / n
        stand_mean = self._stand_mean(n, C)
        s_data = (X - stand_mean) / np.sqrt(self.var_pooled_)
        gamma_hat = np.vstack([s_data[batch == b].mean(0) for b in self.batches_])
        delta_hat = np.vstack([s_data[batch == b].var(0, ddof=1) if (batch == b).sum() > 1
                               else np.ones(p) for b in self.batches_])
        if self.eb:
            gamma_star, delta_star = np.empty_like(gamma_hat), np.empty_like(delta_hat)
            for i, b in enumerate(self.batches_):
                g_bar, t2 = gamma_hat[i].mean(), gamma_hat[i].var(ddof=1)
                a, bb = _aprior(delta_hat[i]), _bprior(delta_hat[i])
                gamma_star[i], delta_star[i] = _it_sol(s_data[batch == b], gamma_hat[i], delta_hat[i],
                                                       g_bar, t2, a, bb)
        else:
            gamma_star, delta_star = gamma_hat, delta_hat
        self.gamma_star_, self.delta_star_ = gamma_star, delta_star
        return self

    def _stand_mean(self, n: int, C: Optional[np.ndarray]) -> np.ndarray:
        sm = np.tile(self.grand_mean_, (n, 1))
        if C is not None and self.beta_cov_ is not None:
            sm = sm + C @ self.beta_cov_
        return sm

    def transform(self, X: np.ndarray, batch: Sequence, covars: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, float)
        batch = np.asarray(batch)
        n = X.shape[0]
        C = None if covars is None else np.asarray(covars, float).reshape(n, -1)
        stand_mean = self._stand_mean(n, C)
        s_data = (X - stand_mean) / np.sqrt(self.var_pooled_)
        out = np.empty_like(s_data)
        for b in pd.unique(batch):
            if b not in self.batches_:
                raise KeyError(f"batch {b!r} not seen at fit time")
            i = self.batches_.index(b)
            m = batch == b
            out[m] = (s_data[m] - self.gamma_star_[i]) / np.sqrt(self.delta_star_[i])
        return out * np.sqrt(self.var_pooled_) + stand_mean

    def fit_transform(self, X: np.ndarray, batch: Sequence, covars: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, batch, covars).transform(X, batch, covars)


def combat_harmonize(X: np.ndarray, batch: Sequence, covars: Optional[np.ndarray] = None,
                     eb: bool = True) -> np.ndarray:
    """One-shot ComBat (fit and transform on the same data). Use :class:`ComBat` inside CV folds."""
    return ComBat(eb=eb).fit_transform(X, batch, covars)


def batch_effect_size(X: np.ndarray, batch: Sequence) -> np.ndarray:
    """Per-feature ratio of between-batch to within-batch variance (a quick harmonization check)."""
    X = np.asarray(X, float)
    batch = np.asarray(batch)
    grand = X.mean(0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for b in pd.unique(batch):
        m = batch == b
        between += m.sum() * (X[m].mean(0) - grand) ** 2
        within += ((X[m] - X[m].mean(0)) ** 2).sum(0)
    return between / np.maximum(within, 1e-12)
