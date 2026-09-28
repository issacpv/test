"""Low-resolution surrogates of asynchrony burden from charted ventilator parameters.

HiRID charts ventilator variables every 2 minutes, eICU's respiratoryCharting and
MIMIC-IV chartevents roughly hourly; none has waveforms. But asynchronies leave
footprints in those numbers: a measured respiratory rate above the set rate
(patient triggering, double triggering), breath-to-breath variability of tidal
volume and peak pressure (breath stacking, flow starvation), and inconsistency
between minute volume and tidal volume x rate (ineffective efforts counted by
some monitors as "spontaneous rate"). ``bin_breaths`` turns waveform-derived
ground truth into the same bins a monitor would chart, so a mapping from
surrogates to the true asynchrony index can be fitted and validated before it is
applied to the charted databases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

SURROGATE_FEATURES = ("rr_measured", "rr_excess", "cv_vt", "cv_pip", "cv_rr", "mv_inconsistency", "pip_range")


def bin_breaths(features: pd.DataFrame, labels: pd.Series | None, ie_times_s: np.ndarray, bin_s: float = 120.0,
                rr_set: float | None = None, duration_s: float | None = None) -> pd.DataFrame:
    """Per-bin charted-style summaries plus the true asynchrony index.

    ``features`` from ``breaths.breath_features``; ``labels`` (per breath) with values such as
    'double_trigger'; ``ie_times_s`` times of ineffective efforts. Bins with no breaths are kept
    with NaN summaries so exposure time is preserved.
    """
    duration_s = duration_s or float(features["t_start"].max() + 5)
    n_bins = int(np.ceil(duration_s / bin_s))
    b = np.floor(features["t_start"].to_numpy() / bin_s).astype(int)
    ie_bins = np.floor(np.asarray(ie_times_s, float) / bin_s).astype(int) if len(ie_times_s) else np.array([], int)
    async_lab = labels.isin(["double_trigger", "auto_trigger", "reverse_trigger"]).to_numpy() if labels is not None else np.zeros(len(features), bool)
    rows = []
    for k in range(n_bins):
        m = b == k
        f = features[m]
        n_ie = int((ie_bins == k).sum())
        n_b = int(m.sum())
        n_async = int(async_lab[m].sum())
        row = {"bin": k, "t_center": (k + 0.5) * bin_s, "n_breaths": n_b, "n_ie": n_ie, "n_async_breaths": n_async,
               "ai_true": 100.0 * (n_ie + n_async) / (n_b + n_ie) if (n_b + n_ie) else np.nan,
               "rr_measured": 60.0 * n_b / bin_s, "rr_set": rr_set}
        for col in ("vt_insp", "pip", "rr_inst"):
            row[f"mean_{col}"] = float(f[col].mean()) if n_b else np.nan
            row[f"cv_{col}"] = float(f[col].std(ddof=0) / f[col].mean()) if n_b > 1 and f[col].mean() else np.nan
        row["pip_range"] = float(f["pip"].max() - f["pip"].min()) if n_b > 1 else np.nan
        row["mv"] = float(f["vt_insp"].sum() * 60.0 / bin_s) if n_b else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def surrogate_features(bins: pd.DataFrame) -> pd.DataFrame:
    """Charted-style surrogates in ``SURROGATE_FEATURES`` from bin summaries (or from real charted data
    with the same column names: rr_measured, rr_set, mean_vt_insp, cv_vt_insp, cv_pip, cv_rr_inst, mv, pip_range)."""
    out = pd.DataFrame(index=bins.index)
    out["rr_measured"] = bins["rr_measured"]
    out["rr_excess"] = bins["rr_measured"] - bins["rr_set"].fillna(bins["rr_measured"])
    out["cv_vt"] = bins["cv_vt_insp"]
    out["cv_pip"] = bins["cv_pip"]
    out["cv_rr"] = bins["cv_rr_inst"]
    expected_mv = bins["mean_vt_insp"] * bins["rr_measured"]
    out["mv_inconsistency"] = (bins["mv"] - expected_mv).abs() / expected_mv.replace(0, np.nan)
    out["pip_range"] = bins["pip_range"]
    return out


def fit_surrogate_model(X: pd.DataFrame, ai_true: np.ndarray, seed: int = 0) -> HistGradientBoostingRegressor:
    m = np.isfinite(ai_true)
    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    return reg.fit(X.loc[m, list(SURROGATE_FEATURES)], ai_true[m])


def evaluate_surrogate(model, X: pd.DataFrame, ai_true: np.ndarray, threshold: float = 10.0) -> dict[str, float]:
    """Spearman rho of predicted vs true AI, and AUROC for detecting AI >= threshold (Thille's 10%)."""
    m = np.isfinite(ai_true)
    pred = model.predict(X.loc[m, list(SURROGATE_FEATURES)])
    y = ai_true[m] >= threshold
    rho = spearmanr(pred, ai_true[m]).correlation
    auc = roc_auc_score(y, pred) if 0 < y.mean() < 1 else float("nan")
    return {"spearman": float(rho), "auroc_ai10": float(auc), "n": int(m.sum())}


def charted_window_features(chart: pd.DataFrame, window_s: float = 3600.0, t_col: str = "t_s") -> pd.DataFrame:
    """Same surrogate features from a *charted* long-to-wide table with columns
    ``t_s, rr_total, rr_set, vt_obs, pip, minute_volume`` sampled irregularly (HiRID 2 min, eICU/MIMIC ~hourly).
    Windows need >= 3 charted rows for the variability features."""
    w = np.floor(chart[t_col].to_numpy() / window_s).astype(int)
    g = chart.assign(_w=w).groupby("_w")
    out = pd.DataFrame({
        "t_center": (g["_w"].first() + 0.5) * window_s,
        "n_rows": g.size(),
        "rr_measured": g["rr_total"].mean(),
        "rr_set": g["rr_set"].median(),
        "mean_vt_insp": g["vt_obs"].mean(),
        "cv_vt_insp": g["vt_obs"].std(ddof=0) / g["vt_obs"].mean(),
        "cv_pip": g["pip"].std(ddof=0) / g["pip"].mean(),
        "cv_rr_inst": g["rr_total"].std(ddof=0) / g["rr_total"].mean(),
        "pip_range": g["pip"].max() - g["pip"].min(),
        "mv": g["minute_volume"].mean(),
    }).reset_index(drop=True)
    out.loc[out["n_rows"] < 3, ["cv_vt_insp", "cv_pip", "cv_rr_inst", "pip_range"]] = np.nan
    return surrogate_features(out).assign(t_center=out["t_center"], n_rows=out["n_rows"])
