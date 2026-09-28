"""Leave-one-site-out transportability of centiles, adaptation budgets and reference-model agreement.

For each site s and each feature:

1. fit the normative model on controls from all sites ≠ s;
2. score the controls of site s without adaptation (n_adapt = 0) and with shift-scale adaptation
   estimated from n_adapt randomly chosen controls of s (which are then excluded from evaluation);
3. summarize the held-out centile distribution (KS vs uniform, extreme rate, mean centile) and the
   mean absolute difference to the in-sample centiles (model fitted with site s included).

The same machinery gives "clinical detection in- vs out-of-reference": fit with/without the clinical
site's controls, score the patients, compare the AUC of extreme-deviation counts.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .centiles import auc_with_ci, centile_to_z, extreme_deviations, uniformity_check
from .normative import GaussianNormativeModel


def _controls(pheno: pd.DataFrame) -> np.ndarray:
    return (pheno["group"] == "control").to_numpy()


def leave_one_site_out(pheno: pd.DataFrame, features: pd.DataFrame, feature_cols: Sequence[str],
                       adapt_budgets: Sequence[int] = (0, 10, 25, 50), n_reps: int = 3,
                       model_factory: Callable[[], GaussianNormativeModel] = GaussianNormativeModel,
                       min_site_n: int = 30, random_state: int = 0) -> pd.DataFrame:
    """Run the LOSO audit. ``features`` is aligned row-wise with ``pheno``.

    Returns one row per (site, feature, budget, rep) with KS, extreme rate, mean centile, mean |Δcentile|
    vs in-sample scoring, and n evaluated.
    """
    rng = np.random.default_rng(random_state)
    ctrl = _controls(pheno)
    rows = []
    sites = [s for s in pheno.loc[ctrl, "site"].unique() if (ctrl & (pheno["site"] == s).to_numpy()).sum() >= min_site_n]
    for f in feature_cols:
        y = features[f].to_numpy(float)
        # in-sample reference (all control sites) for the Δcentile comparison
        full = model_factory().fit(pheno[ctrl], y[ctrl])
        for s in sites:
            in_site = (pheno["site"] == s).to_numpy()
            train = ctrl & ~in_site
            held = ctrl & in_site
            model = model_factory().fit(pheno[train], y[train])
            c_full = full.centile(pheno[held], y[held])
            held_idx = np.flatnonzero(held)
            for budget in adapt_budgets:
                for rep in range(n_reps if budget > 0 else 1):
                    m = model_factory()
                    m.__dict__.update({k: v for k, v in model.__dict__.items() if k != "site_adapt_"})
                    m.site_adapt_ = {}
                    if budget > 0:
                        if budget >= len(held_idx) - 10:
                            continue
                        calib = rng.choice(held_idx, budget, replace=False)
                        m.adapt_site(pheno.iloc[calib], y[calib], site=s)
                        eval_idx = np.setdiff1d(held_idx, calib)
                    else:
                        eval_idx = held_idx
                    c = m.centile(pheno.iloc[eval_idx], y[eval_idx])
                    u = uniformity_check(c)
                    pos = np.searchsorted(held_idx, eval_idx)
                    rows.append({"site": s, "feature": f, "adapt_n": budget, "rep": rep, **u,
                                 "mean_abs_centile_shift": float(np.nanmean(np.abs(c - c_full[pos])))})
    return pd.DataFrame(rows)


def summarize_loso(results: pd.DataFrame) -> pd.DataFrame:
    """Median (IQR) of KS / extreme rate / centile shift per site × budget across features and reps."""
    g = results.groupby(["site", "adapt_n"])
    out = g.agg(ks_median=("ks", "median"), ks_iqr=("ks", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                extreme_rate_median=("extreme_rate", "median"),
                shift_median=("mean_abs_centile_shift", "median"), n_features=("feature", "nunique"))
    return out.reset_index()


def compare_reference_models(centiles_a: np.ndarray, centiles_b: np.ndarray, threshold_z: float = 1.96) -> dict[str, float]:
    """Agreement between two references scoring the same individuals: Spearman ρ of centiles,
    Cohen's κ for extreme-deviation flags, mean |Δz| and the Bland-Altman limits of agreement for z."""
    a, b = np.asarray(centiles_a, float), np.asarray(centiles_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    za, zb = centile_to_z(a), centile_to_z(b)
    fa, fb = np.abs(za) > threshold_z, np.abs(zb) > threshold_z
    po = float((fa == fb).mean())
    pe = float(fa.mean() * fb.mean() + (1 - fa.mean()) * (1 - fb.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    diff = za - zb
    return {"n": int(len(a)), "spearman": float(stats.spearmanr(a, b).statistic), "kappa_extreme": float(kappa),
            "mean_abs_dz": float(np.abs(diff).mean()), "bland_altman_bias": float(diff.mean()),
            "bland_altman_loa": (float(diff.mean() - 1.96 * diff.std(ddof=1)), float(diff.mean() + 1.96 * diff.std(ddof=1))),
            "flag_rate_a": float(fa.mean()), "flag_rate_b": float(fb.mean())}


def clinical_detection_in_vs_out(pheno: pd.DataFrame, features: pd.DataFrame, feature_cols: Sequence[str],
                                 clinical_site: str, positive_group: str, threshold: float = 1.96,
                                 model_factory: Callable[[], GaussianNormativeModel] = GaussianNormativeModel,
                                 adapt_n: int = 0, random_state: int = 0) -> pd.DataFrame:
    """AUC of extreme-deviation counts (patients vs controls of ``clinical_site``) when the site's
    controls are part of the reference ("in") versus excluded ("out", optionally with ``adapt_n``
    local controls used for shift-scale adaptation and then dropped from evaluation)."""
    rng = np.random.default_rng(random_state)
    ctrl = _controls(pheno)
    in_site = (pheno["site"] == clinical_site).to_numpy()
    patients = in_site & (pheno["group"] == positive_group).to_numpy()
    site_ctrl_idx = np.flatnonzero(ctrl & in_site)
    rows = []
    for setting in ("in", "out"):
        train = ctrl if setting == "in" else (ctrl & ~in_site)
        calib = np.array([], int)
        if setting == "out" and adapt_n > 0:
            calib = rng.choice(site_ctrl_idx, min(adapt_n, len(site_ctrl_idx) // 2), replace=False)
        eval_ctrl = np.setdiff1d(site_ctrl_idx, calib)
        eval_idx = np.concatenate([eval_ctrl, np.flatnonzero(patients)])
        Z = {}
        for f in feature_cols:
            y = features[f].to_numpy(float)
            m = model_factory().fit(pheno[train], y[train])
            if len(calib):
                m.adapt_site(pheno.iloc[calib], y[calib], site=clinical_site)
            Z[f] = m.zscore(pheno.iloc[eval_idx], y[eval_idx])
        Zdf = pd.DataFrame(Z, index=eval_idx)
        counts = extreme_deviations(Zdf, threshold)
        label = np.r_[np.zeros(len(eval_ctrl), int), np.ones(patients.sum(), int)]
        res = auc_with_ci(counts["n_total"].to_numpy(), label)
        rows.append({"setting": setting, "adapt_n": adapt_n if setting == "out" else np.nan,
                     "n_controls": int(len(eval_ctrl)), "n_patients": int(patients.sum()),
                     "control_extreme_rate": float((Zdf.iloc[: len(eval_ctrl)].abs() > threshold).to_numpy().mean()),
                     **res})
    return pd.DataFrame(rows)


def adaptation_curve(pheno: pd.DataFrame, y: np.ndarray, site: str, budgets: Sequence[int] = (5, 10, 25, 50, 100),
                     n_reps: int = 20, model_factory: Callable[[], GaussianNormativeModel] = GaussianNormativeModel,
                     random_state: int = 0) -> pd.DataFrame:
    """Extreme rate and KS of a held-out site as a function of the number of local controls used
    for adaptation (single feature), averaged over random calibration subsets."""
    rng = np.random.default_rng(random_state)
    ctrl = _controls(pheno)
    in_site = (pheno["site"] == site).to_numpy()
    base = model_factory().fit(pheno[ctrl & ~in_site], np.asarray(y, float)[ctrl & ~in_site])
    held = np.flatnonzero(ctrl & in_site)
    rows = []
    c0 = base.centile(pheno.iloc[held], np.asarray(y, float)[held])
    rows.append({"adapt_n": 0, "rep": 0, **uniformity_check(c0)})
    for b in budgets:
        if b >= len(held) - 10:
            continue
        for rep in range(n_reps):
            m = model_factory()
            m.__dict__.update({k: v for k, v in base.__dict__.items() if k != "site_adapt_"})
            m.site_adapt_ = {}
            calib = rng.choice(held, b, replace=False)
            m.adapt_site(pheno.iloc[calib], np.asarray(y, float)[calib], site=site)
            ev = np.setdiff1d(held, calib)
            rows.append({"adapt_n": b, "rep": rep, **uniformity_check(m.centile(pheno.iloc[ev], np.asarray(y, float)[ev]))})
    return pd.DataFrame(rows)
