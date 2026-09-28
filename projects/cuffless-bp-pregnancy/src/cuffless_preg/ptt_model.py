"""PAT/PTT -> blood-pressure calibration models and a Bramwell-Hill shift simulator.

Two empirical layers:

* per-subject OLS calibration  SBP = a + b * ln(PAT)
* population linear mixed model  SBP ~ ln(PAT) * group + HR, random intercept
  and slope per subject (statsmodels MixedLM)

and one mechanistic layer: an exponential-elastance artery whose pulse wave
velocity follows Moens-Korteweg, used to predict how the PAT-SBP curve moves
under pregnancy-like and preeclampsia-like parameter changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RHO_BLOOD = 1060.0  # kg m^-3


def add_log_pat(df: pd.DataFrame, pat_col: str = "pat_s") -> pd.DataFrame:
    """Add ``pat_ms`` and ``log_pat`` (natural log of PAT in ms) to a beat table.

    ``pat_ms`` is the primary regressor (linear model, slope in mmHg/ms; sensitive
    to baseline stiffness E0); ``log_pat`` is the secondary regressor whose slope
    isolates the pressure-sensitivity alpha of the exponential elastance model.
    """
    out = df.copy()
    out["pat_ms"] = out[pat_col] * 1000.0
    out["log_pat"] = np.log(out["pat_ms"])
    return out


def fit_subject_calibrations(df: pd.DataFrame, subject_col: str = "subject_id", y: str = "sbp", x: str = "log_pat", min_beats: int = 30) -> pd.DataFrame:
    """OLS SBP = a + b*x per subject. Returns intercept, slope, r2, n, residual SD."""
    rows = []
    for sid, g in df.dropna(subset=[x, y]).groupby(subject_col):
        if len(g) < min_beats:
            continue
        X = np.column_stack([np.ones(len(g)), g[x].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)
        resid = g[y].to_numpy() - X @ beta
        ss_tot = np.sum((g[y] - g[y].mean()) ** 2)
        r2 = 1.0 - np.sum(resid**2) / ss_tot if ss_tot > 0 else np.nan
        rows.append({subject_col: sid, "intercept": beta[0], "slope": beta[1], "r2": r2, "n": len(g), "resid_sd": resid.std(ddof=2)})
    return pd.DataFrame(rows)


def fit_mixed_pat_model(
    df: pd.DataFrame,
    group_col: str = "group",
    subject_col: str = "subject_id",
    y: str = "sbp",
    x: str = "log_pat",
    covariates: tuple[str, ...] = ("hr_bpm",),
    reference: str = "control",
    random_slope: bool = True,
):
    """Fit SBP ~ x * group + covariates with random intercept (+ slope) per subject.

    Returns the fitted statsmodels MixedLMResults.  The interaction terms
    ``x:group[T.<level>]`` are the slope shifts relative to ``reference``; the
    ``group[T.<level>]`` terms are the intercept shifts at x = 0, so callers
    should centre ``x`` first (see :func:`centre`) to make intercepts
    interpretable at the population median PAT.
    """
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[x, y, group_col, subject_col] + list(covariates)).copy()
    levels = [reference] + sorted(set(d[group_col]) - {reference})
    d[group_col] = pd.Categorical(d[group_col], categories=levels)
    rhs = f"{x} * {group_col}" + "".join(f" + {c}" for c in covariates)
    re_formula = f"~{x}" if random_slope else "~1"
    model = smf.mixedlm(f"{y} ~ {rhs}", d, groups=d[subject_col], re_formula=re_formula)
    return model.fit(method=["lbfgs"], maxiter=200, disp=False)


def centre(df: pd.DataFrame, col: str = "log_pat") -> tuple[pd.DataFrame, float]:
    """Centre a column on its median; returns (df, median)."""
    m = float(df[col].median())
    out = df.copy()
    out[col] = out[col] - m
    return out, m


def interaction_table(result, x: str = "log_pat", group_col: str = "group") -> pd.DataFrame:
    """Extract group intercept and slope shifts (with 95% CI) from a MixedLM fit."""
    params, ci = result.params, result.conf_int()
    rows = []
    for name in params.index:
        if group_col in name and "[T." in name:
            level = name.split("[T.")[1].rstrip("]")
            kind = "slope_shift" if name.startswith(f"{x}:") or name.endswith(f":{x}") else "intercept_shift"
            rows.append({"group": level, "term": kind, "estimate": params[name], "ci_low": ci.loc[name, 0], "ci_high": ci.loc[name, 1], "p": result.pvalues[name]})
    return pd.DataFrame(rows)


def one_point_calibration(pred: np.ndarray, ref: np.ndarray, t_s: np.ndarray, calib_window_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Shift predictions by the mean bias over the first ``calib_window_s`` seconds.

    Returns (corrected predictions, boolean mask of evaluation beats outside the window).
    """
    calib = t_s <= (t_s.min() + calib_window_s)
    if calib.sum() == 0:
        return pred, np.ones_like(pred, dtype=bool)
    bias = np.nanmean(ref[calib] - pred[calib])
    return pred + bias, ~calib


def error_decomposition(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Split MSE of a population model on one subject into bias, slope and residual parts.

    Fits ref = a + b*pred by OLS; bias part = MSE removed by an intercept-only
    correction, slope part = additional MSE removed by the 2-parameter fit,
    residual = remaining MSE.  All in mmHg^2; fractions sum to 1.
    """
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[ok], ref[ok]
    mse0 = float(np.mean((ref - pred) ** 2))
    mse1 = float(np.mean((ref - pred - np.mean(ref - pred)) ** 2))
    X = np.column_stack([np.ones_like(pred), pred])
    beta, *_ = np.linalg.lstsq(X, ref, rcond=None)
    mse2 = float(np.mean((ref - X @ beta) ** 2))
    return {
        "mse_total": mse0,
        "bias_part": mse0 - mse1,
        "slope_part": mse1 - mse2,
        "residual_part": mse2,
        "bias_frac": (mse0 - mse1) / mse0 if mse0 > 0 else np.nan,
        "slope_frac": (mse1 - mse2) / mse0 if mse0 > 0 else np.nan,
        "fitted_slope": float(beta[1]),
    }


# ----------------------------------------------------------------------------
# Mechanistic layer
# ----------------------------------------------------------------------------
@dataclass
class ArteryParams:
    """Exponential-elastance artery: E(P) = E0 * exp(alpha * P).

    E0 in Pa, alpha in 1/mmHg, wall thickness h and radius r in m, path length L in m.
    Defaults give PWV ~ 6-8 m/s at 100-140 mmHg over a heart-to-finger path.
    """

    E0: float = 1.0e5
    alpha: float = 0.017
    h: float = 0.8e-3
    r: float = 2.5e-3
    L: float = 0.9
    pep_s: float = 0.08

    def pwv(self, p_mmhg: np.ndarray) -> np.ndarray:
        E = self.E0 * np.exp(self.alpha * np.asarray(p_mmhg, float))
        return np.sqrt(E * self.h / (2.0 * RHO_BLOOD * self.r))

    def ptt(self, p_mmhg: np.ndarray) -> np.ndarray:
        return self.L / self.pwv(p_mmhg)

    def pat(self, p_mmhg: np.ndarray) -> np.ndarray:
        return self.pep_s + self.ptt(p_mmhg)


# Under E(P) = E0 exp(alpha P):  ln PTT = c - alpha P / 2, so
#   d SBP / d ln(PTT) = -2 / alpha          (depends on alpha only)
#   d SBP / d PTT     = -2 / (alpha * PTT)  (steeper when the artery is stiffer, i.e. E0 higher)
# Baseline stiffness E0 therefore shifts the intercept (and the linear slope), while
# alpha shifts the log-slope.  Pregnancy: lower E0, larger radius, shorter PEP.
# Preeclampsia: higher E0 (alpha treated as a sensitivity parameter, default unchanged).
PREGNANCY_SCALING = {"E0": 0.70, "r": 1.10, "pep_s": 0.85}
PREECLAMPSIA_SCALING = {"E0": 1.40, "pep_s": 0.90}


def scaled(params: ArteryParams, scaling: dict[str, float]) -> ArteryParams:
    """Return a copy of ``params`` with the given fields multiplied by the factors."""
    kw = {k: getattr(params, k) for k in params.__dataclass_fields__}
    for k, f in scaling.items():
        kw[k] = kw[k] * f
    return ArteryParams(**kw)


def simulated_calibration_shift(base: ArteryParams | None = None, p_range: tuple[float, float] = (90.0, 170.0), n: int = 200) -> pd.DataFrame:
    """Predicted calibration curves of SBP vs PAT for control, pregnancy and preeclampsia arteries.

    Returns one row per group with the OLS intercept at the control median PAT
    (``intercept_at_ref``), the log-slope d SBP / d ln(PAT) (``slope_ln``), the
    linear slope d SBP / d PAT in mmHg per ms (``slope_lin_per_ms``), the PAT at
    130 mmHg and the SBP offset at the control median PAT.  These are the
    mechanistic predictions for RQ1 (offset), RQ2 (linear slope) and the
    exploratory alpha question (log-slope).
    """
    base = base or ArteryParams()
    groups = {
        "control": base,
        "pregnant_normotensive": scaled(base, PREGNANCY_SCALING),
        "hdp": scaled(base, PREECLAMPSIA_SCALING),
    }
    p = np.linspace(*p_range, n)
    ref_pat_ms = 1000.0 * groups["control"].pat(p)
    x0_ln = float(np.median(np.log(ref_pat_ms)))
    x0_lin = float(np.median(ref_pat_ms))
    rows = []
    for name, ap in groups.items():
        pat_ms = 1000.0 * ap.pat(p)
        X_ln = np.column_stack([np.ones(n), np.log(pat_ms) - x0_ln])
        b_ln, *_ = np.linalg.lstsq(X_ln, p, rcond=None)
        X_lin = np.column_stack([np.ones(n), pat_ms - x0_lin])
        b_lin, *_ = np.linalg.lstsq(X_lin, p, rcond=None)
        rows.append(
            {
                "group": name,
                "intercept_at_ref": b_ln[0],
                "slope_ln": b_ln[1],
                "slope_lin_per_ms": b_lin[1],
                "pat_ms_at_130": 1000.0 * float(ap.pat(np.array([130.0]))[0]),
            }
        )
    out = pd.DataFrame(rows)
    ref_int = out.loc[out["group"] == "control", "intercept_at_ref"].iloc[0]
    out["sbp_offset_at_ref_pat"] = out["intercept_at_ref"] - ref_int
    return out
