"""Pipeline specifications and GLM fitting for the multiverse.

A :class:`PipelineSpec` captures one path through the analytic garden of
forking paths (Carp, 2012; Botvinik-Nezer et al., 2020). The NumPy
implementation (:func:`run_first_level_numpy`) operates on parcel or ROI time
series and is used for the dataset-scale sweep; :func:`run_first_level_nilearn`
wraps :class:`nilearn.glm.first_level.FirstLevelModel` for voxel-wise maps of
the primary pipelines.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineSpec:
    smoothing_fwhm: Optional[float] = 5.0
    confounds: str = "hmp24_phys8"
    hrf_model: str = "spm"  # spm | spm+derivative | glover | glover+derivative | fir
    high_pass: float = 0.01  # Hz (cosine drift cutoff); ignored when confounds include cosines
    parcellation: str = "schaefer200"
    fmriprep_version: str = "shipped"
    noise_model: str = "ols"  # ols | ar1
    drift_model: str = "cosine"  # cosine | none (use fMRIPrep cosines)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        return "|".join(f"{k}={v}" for k, v in self.as_dict().items())


DEFAULT_GRID: Dict[str, Sequence[object]] = {
    "smoothing_fwhm": [None, 4.0, 8.0],
    "confounds": ["hmp6", "hmp24_phys8", "hmp24_phys8_gsr4", "acompcor", "hmp24_phys8_scrub"],
    "hrf_model": ["spm", "spm+derivative", "glover"],
    "high_pass": [0.008, 1 / 128],
    "parcellation": ["schaefer200", "schaefer400"],
    "fmriprep_version": ["shipped"],
    "noise_model": ["ols", "ar1"],
}


def enumerate_multiverse(grid: Optional[Dict[str, Sequence[object]]] = None) -> List[PipelineSpec]:
    """Cartesian product of the grid -> list of :class:`PipelineSpec`."""
    grid = dict(DEFAULT_GRID if grid is None else grid)
    keys = list(grid)
    specs = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        specs.append(PipelineSpec(**dict(zip(keys, combo))))
    return specs


# --------------------------------------------------------------------------- #
# HRF and design
# --------------------------------------------------------------------------- #
def _gamma_pdf(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
    t = np.clip(t, 1e-12, None)
    return np.exp((shape - 1) * np.log(t) - t / scale - gammaln(shape) - shape * np.log(scale))


def spm_hrf(tr: float, oversampling: int = 16, time_length: float = 32.0, onset: float = 0.0) -> np.ndarray:
    """SPM canonical double-gamma HRF sampled at ``tr / oversampling``."""
    dt = tr / oversampling
    t = np.arange(0, time_length, dt) - onset
    peak = _gamma_pdf(t, 6.0, 1.0)
    under = _gamma_pdf(t, 16.0, 1.0)
    h = peak - under / 6.0
    return h / h.sum()


def glover_hrf(tr: float, oversampling: int = 16, time_length: float = 32.0, onset: float = 0.0) -> np.ndarray:
    """Glover (1999) HRF as parameterised in nilearn (delay 6, undershoot 12, ratio 0.35)."""
    dt = tr / oversampling
    t = np.arange(0, time_length, dt) - onset
    peak = _gamma_pdf(t, 6.0, 0.9)
    under = _gamma_pdf(t, 12.0, 0.9)
    h = peak - 0.35 * under
    return h / h.sum()


def hrf_basis(model: str, tr: float, oversampling: int = 16) -> List[np.ndarray]:
    """Return the list of basis functions for an HRF model name."""
    base = model.split("+")[0]
    fn = {"spm": spm_hrf, "glover": glover_hrf}.get(base)
    if fn is None:
        raise ValueError(f"unknown hrf model {model}")
    h = fn(tr, oversampling)
    basis = [h]
    if "derivative" in model:
        dh = np.gradient(h)
        basis.append(dh / np.abs(dh).sum())
    return basis


def cosine_drift(high_pass: float, frame_times: np.ndarray) -> np.ndarray:
    """DCT-basis drift regressors below ``high_pass`` Hz (SPM/nilearn convention)."""
    n = len(frame_times)
    duration = frame_times[-1] - frame_times[0] + (frame_times[1] - frame_times[0])
    n_basis = int(np.floor(2 * duration * high_pass))
    cols = []
    for k in range(1, n_basis + 1):
        cols.append(np.sqrt(2.0 / n) * np.cos(np.pi * (np.arange(n) + 0.5) * k / n))
    return np.column_stack(cols) if cols else np.zeros((n, 0))


def make_design_matrix(
    events: pd.DataFrame,
    frame_times: np.ndarray,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 0.01,
    confounds: Optional[pd.DataFrame] = None,
    oversampling: int = 16,
    conditions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a first-level design matrix.

    ``events`` needs ``onset, duration, trial_type`` (+ optional
    ``modulation``). Boxcars are built at ``tr / oversampling`` resolution,
    convolved with each basis function, and sampled at ``frame_times``.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    tr = float(np.median(np.diff(frame_times)))
    dt = tr / oversampling
    t_end = frame_times[-1] + tr
    hi_res = np.arange(0, t_end + 32.0, dt)
    basis = hrf_basis(hrf_model, tr, oversampling)
    conds = list(conditions) if conditions is not None else sorted(events["trial_type"].astype(str).unique())
    cols: Dict[str, np.ndarray] = {}
    for cond in conds:
        ev = events[events["trial_type"].astype(str) == cond]
        box = np.zeros_like(hi_res)
        for _, row in ev.iterrows():
            amp = float(row["modulation"]) if "modulation" in ev.columns and pd.notna(row.get("modulation")) else 1.0
            start = int(np.round(float(row["onset"]) / dt))
            stop = int(np.round((float(row["onset"]) + max(float(row["duration"]), dt)) / dt))
            box[start:stop] += amp
        for b, h in enumerate(basis):
            conv = np.convolve(box, h)[: len(hi_res)]
            name = cond if b == 0 else f"{cond}_derivative"
            cols[name] = np.interp(frame_times, hi_res, conv)
    X = pd.DataFrame(cols, index=frame_times)
    if drift_model == "cosine":
        D = cosine_drift(high_pass, frame_times)
        for k in range(D.shape[1]):
            X[f"drift_{k + 1}"] = D[:, k]
    if confounds is not None:
        C = confounds.reset_index(drop=True)
        for c in C.columns:
            X[f"conf_{c}"] = C[c].to_numpy()
    X["constant"] = 1.0
    return X


# --------------------------------------------------------------------------- #
# GLM
# --------------------------------------------------------------------------- #
def fit_glm_ols(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """OLS for all columns of ``Y`` ``(T, V)`` on ``X`` ``(T, P)``.

    Returns ``beta (P, V)``, residual variance ``(V,)`` and degrees of freedom.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    pinv = np.linalg.pinv(X)
    beta = pinv @ Y
    resid = Y - X @ beta
    dof = X.shape[0] - np.linalg.matrix_rank(X)
    resvar = (resid**2).sum(axis=0) / max(dof, 1)
    return beta, resvar, dof


def prewhiten_ar1(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simple AR(1) prewhitening with a pooled rho (Cochrane-Orcutt style)."""
    beta, _, _ = fit_glm_ols(Y, X)
    resid = Y - X @ beta
    num = (resid[1:] * resid[:-1]).sum()
    den = (resid[:-1] ** 2).sum()
    rho = float(np.clip(num / den if den > 0 else 0.0, -0.95, 0.95))
    Yw = Y[1:] - rho * Y[:-1]
    Xw = X[1:] - rho * X[:-1]
    return Yw, Xw, rho


def contrast_t(beta: np.ndarray, resvar: np.ndarray, X: np.ndarray, con: np.ndarray, dof: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Effect size, t-statistic and two-sided p for a contrast vector."""
    con = np.asarray(con, dtype=float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_con = float(con @ XtX_inv @ con)
    effect = con @ beta
    se = np.sqrt(resvar * var_con)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, effect / se, 0.0)
    p = 2 * stats.t.sf(np.abs(t), df=max(dof, 1))
    return effect, t, p


def contrast_vector(design_columns: Sequence[str], expression: str) -> np.ndarray:
    """Parse ``"A - B"``, ``"A"``, ``"0.5*A + 0.5*B - C"`` into a vector over columns."""
    cols = list(design_columns)
    vec = np.zeros(len(cols))
    expr = expression.replace(" ", "")
    tokens = [t for t in expr.replace("-", "+-").split("+") if t]
    for tok in tokens:
        sign = -1.0 if tok.startswith("-") else 1.0
        tok = tok.lstrip("-")
        if "*" in tok:
            w, name = tok.split("*")
            weight = float(w) * sign
        else:
            name, weight = tok, sign
        if name not in cols:
            raise KeyError(f"condition {name!r} not in design columns")
        vec[cols.index(name)] += weight
    return vec


def run_first_level_numpy(
    timeseries: np.ndarray,
    events: pd.DataFrame,
    tr: float,
    spec: PipelineSpec,
    confounds: Optional[pd.DataFrame] = None,
    sample_mask: Optional[np.ndarray] = None,
    contrasts: Optional[Dict[str, str]] = None,
    conditions: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """First-level GLM on ``(T, V)`` parcel/ROI time series for one run.

    Scrubbed volumes (``sample_mask == False``) are removed from both data and
    design after convolution. Returns ``{contrast: {"effect", "t", "p"}}``.
    """
    Y = np.asarray(timeseries, dtype=float)
    T = Y.shape[0]
    frame_times = np.arange(T) * tr
    drift = "none" if (confounds is not None and any(str(c).startswith("cosine") for c in confounds.columns)) else spec.drift_model
    X = make_design_matrix(events, frame_times, spec.hrf_model, drift, spec.high_pass, confounds, conditions=conditions)
    if sample_mask is not None:
        keep = np.asarray(sample_mask, dtype=bool)
        Y = Y[keep]
        X = X.iloc[keep]
    Xv = X.to_numpy()
    if spec.noise_model == "ar1":
        Yw, Xw, _ = prewhiten_ar1(Y, Xv)
    else:
        Yw, Xw = Y, Xv
    beta, resvar, dof = fit_glm_ols(Yw, Xw)
    contrasts = contrasts or {c: c for c in (conditions or sorted(events["trial_type"].astype(str).unique()))}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, expr in contrasts.items():
        con = contrast_vector(X.columns, expr)
        effect, t, p = contrast_t(beta, resvar, Xw, con, dof)
        out[name] = {"effect": effect, "t": t, "p": p}
    return out


def run_first_level_nilearn(
    bold_path: str,
    events: pd.DataFrame,
    tr: float,
    spec: PipelineSpec,
    confounds: Optional[pd.DataFrame] = None,
    sample_mask: Optional[np.ndarray] = None,
    mask_img=None,
    contrasts: Optional[Dict[str, str]] = None,
):
    """Voxel-wise first level with nilearn (optional dependency).

    Returns ``{contrast: effect_size_map}``. Uses fMRIPrep cosines when they
    are present in ``confounds`` (``drift_model=None``), otherwise a cosine
    drift model at ``spec.high_pass``.
    """
    from nilearn.glm.first_level import FirstLevelModel  # type: ignore

    drift = None if (confounds is not None and any(str(c).startswith("cosine") for c in confounds.columns)) else spec.drift_model
    model = FirstLevelModel(
        t_r=tr,
        hrf_model=spec.hrf_model,
        drift_model=drift,
        high_pass=spec.high_pass,
        smoothing_fwhm=spec.smoothing_fwhm,
        noise_model=spec.noise_model,
        mask_img=mask_img,
        minimize_memory=True,
    )
    mask_idx = None if sample_mask is None else np.where(np.asarray(sample_mask, dtype=bool))[0]
    model.fit(bold_path, events=events, confounds=confounds, sample_masks=mask_idx)
    contrasts = contrasts or {c: c for c in sorted(events["trial_type"].astype(str).unique())}
    return {name: model.compute_contrast(expr, output_type="effect_size") for name, expr in contrasts.items()}


# --------------------------------------------------------------------------- #
# Second level
# --------------------------------------------------------------------------- #
def second_level_onesample(effects: np.ndarray) -> Dict[str, np.ndarray]:
    """One-sample t-test across subjects for ``(n_subjects, V)`` effects.

    Returns ``mean, t, p, cohen_d``.
    """
    E = np.asarray(effects, dtype=float)
    n = E.shape[0]
    mean = E.mean(axis=0)
    sd = E.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, mean / (sd / np.sqrt(n)), 0.0)
        d = np.where(sd > 0, mean / sd, 0.0)
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    return {"mean": mean, "t": t, "p": p, "cohen_d": d, "n": n}


def sign_flip_max_t(effects: np.ndarray, n_perm: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """FWE-corrected p-values via sign-flipping (max-|t| over V) - within one specification."""
    E = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(seed)
    obs = second_level_onesample(E)["t"]
    max_null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=E.shape[0])[:, None]
        max_null[i] = np.abs(second_level_onesample(E * signs)["t"]).max()
    p_fwe = (np.sum(max_null[None, :] >= np.abs(obs)[:, None], axis=1) + 1) / (n_perm + 1)
    return obs, p_fwe
