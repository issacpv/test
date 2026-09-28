"""Quality / pathology stratification, robustness slopes, learning curves and crossing points.

Input is a long results table with one row per (dataset, subject, model, regime, n_train, structure)
and columns such as ``dice``, IQMs (``cjv``, ``cnr``, ``efc``...), ``lesion_ml``, ``age``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit


def quality_tertiles(df: pd.DataFrame, iqm: str = "cjv", higher_is_worse: bool = True, by: str = "dataset") -> pd.Series:
    """Within-dataset tertiles of an IQM: 'good', 'medium', 'poor'."""
    def _cut(s: pd.Series) -> pd.Series:
        ranks = s.rank(pct=True)
        lab = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0001], labels=["good", "medium", "poor"] if higher_is_worse else ["poor", "medium", "good"])
        return lab.astype(str)
    return df.groupby(by, group_keys=False)[iqm].apply(_cut)


def pathology_strata(lesion_ml: pd.Series, edges=(0.0, 1.0, 10.0, np.inf)) -> pd.Series:
    labels = ["none/small", "moderate", "large"]
    return pd.cut(lesion_ml.fillna(0.0), edges, labels=labels, include_lowest=True).astype(str)


def quality_slopes(df: pd.DataFrame, iqm: str = "cjv", value: str = "dice", by_model: str = "model") -> pd.DataFrame:
    """Mixed model value ~ model × z(IQM) + (1|dataset) (+ subject as a second grouping via variance component).

    Returns per-model slopes (change in ``value`` per 1 SD of IQM) with Wald CIs and the slope
    contrast vs the reference model (first alphabetically).
    """
    d = df.dropna(subset=[value, iqm]).copy()
    d["z"] = (d[iqm] - d[iqm].mean()) / d[iqm].std(ddof=1)
    d[by_model] = d[by_model].astype(str)
    if d["dataset"].nunique() > 1:
        vc = {"subject": "0 + C(subject)"} if d["subject"].nunique() > 1 else None
        fit = smf.mixedlm(f"{value} ~ C({by_model}) * z", d, groups=d["dataset"], vc_formula=vc).fit(reml=False)
    else:
        fit = smf.ols(f"{value} ~ C({by_model}) * z", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["subject"])[0]})
    models = sorted(d[by_model].unique())
    ref = models[0]
    ci = fit.conf_int()
    rows = [{"model": ref, "slope": float(fit.params["z"]), "ci_low": float(ci.loc["z", 0]), "ci_high": float(ci.loc["z", 1]),
             "contrast_vs_ref": 0.0, "p_contrast": np.nan}]
    for m in models[1:]:
        term = f"C({by_model})[T.{m}]:z"
        slope = fit.params["z"] + fit.params[term]
        # CI of the sum via the covariance matrix
        cov = fit.cov_params()
        var = cov.loc["z", "z"] + cov.loc[term, term] + 2 * cov.loc["z", term]
        se = float(np.sqrt(var))
        rows.append({"model": m, "slope": float(slope), "ci_low": float(slope - 1.96 * se), "ci_high": float(slope + 1.96 * se),
                     "contrast_vs_ref": float(fit.params[term]), "p_contrast": float(fit.pvalues[term])})
    return pd.DataFrame(rows)


def stratified_table(df: pd.DataFrame, strata: str, value: str = "dice", by=("model", "regime")) -> pd.DataFrame:
    """Mean/median/IQR of ``value`` by model (and regime) within each stratum."""
    g = df.groupby([*by, strata])[value]
    out = g.agg(mean="mean", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="count")
    return out.reset_index()


def _power_law(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_learning_curve(n_train: np.ndarray, value: np.ndarray) -> dict[str, float]:
    """Fit a saturating power law value(N) = a − b·N^(−c); returns parameters and asymptote."""
    n_train = np.asarray(n_train, float)
    value = np.asarray(value, float)
    p0 = [max(value.max(), 0.5), max(value.max() - value.min(), 0.1), 0.5]
    try:
        params, _ = curve_fit(_power_law, n_train, value, p0=p0, bounds=([0, 0, 0.01], [1.0, 2.0, 3.0]), maxfev=10000)
    except RuntimeError:
        params = np.array([np.nan] * 3)
    return {"a": float(params[0]), "b": float(params[1]), "c": float(params[2])}


def crossing_point(curve_scratch: dict[str, float], target_value: float, n_max: float = 10000.0) -> float:
    """Smallest N at which the from-scratch learning curve reaches ``target_value`` (NaN if never below n_max)."""
    a, b, c = curve_scratch["a"], curve_scratch["b"], curve_scratch["c"]
    if not np.isfinite(a) or target_value >= a:
        return float("nan")
    n = (b / (a - target_value)) ** (1.0 / c)
    return float(n) if n <= n_max else float("nan")


def crossing_points(df: pd.DataFrame, scratch_model: str = "nnunet_scratch", value: str = "dice",
                    strata: str | None = None, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """For each foundation-model regime (at its largest N) and stratum: N* where nnU-Net-from-scratch matches it.

    ``df`` needs columns model, regime, n_train, subject, value (and ``strata`` if given).
    """
    rng = np.random.default_rng(seed)
    rows = []
    groups = [(None, df)] if strata is None else list(df.groupby(strata))
    for stratum, d in groups:
        scratch = d[d["model"] == scratch_model]
        if scratch.empty:
            continue
        subjects = scratch["subject"].unique()
        for (model, regime), g in d[d["model"] != scratch_model].groupby(["model", "regime"]):
            n_ref = g["n_train"].max()
            target = g[g["n_train"] == n_ref][value].mean()
            lc = scratch.groupby("n_train")[value].mean()
            fit = fit_learning_curve(lc.index.to_numpy(), lc.to_numpy())
            n_star = crossing_point(fit, target)
            boots = []
            for _ in range(n_boot):
                pick = rng.choice(subjects, len(subjects), replace=True)
                sb = scratch[scratch["subject"].isin(pick)].groupby("n_train")[value].mean()
                tb = g[(g["n_train"] == n_ref) & g["subject"].isin(pick)][value].mean()
                if len(sb) >= 3 and np.isfinite(tb):
                    boots.append(crossing_point(fit_learning_curve(sb.index.to_numpy(), sb.to_numpy()), tb))
            boots = np.array([b for b in boots if np.isfinite(b)])
            rows.append({"stratum": stratum, "model": model, "regime": regime, "target_value": float(target), "n_star": n_star,
                         "n_star_ci_low": float(np.percentile(boots, 2.5)) if boots.size else np.nan,
                         "n_star_ci_high": float(np.percentile(boots, 97.5)) if boots.size else np.nan,
                         "frac_boot_never": float(1 - boots.size / max(1, n_boot))})
    return pd.DataFrame(rows)


def dose_response(df: pd.DataFrame, value: str = "dice", level_col: str = "level", kind_col: str = "kind") -> pd.DataFrame:
    """Mean ± SD of ``value`` per model × degradation × level, plus the linear slope per model × degradation."""
    agg = df.groupby(["model", kind_col, level_col])[value].agg(["mean", "std", "count"]).reset_index()
    slopes = (df.groupby(["model", kind_col]).apply(lambda g: np.polyfit(g[level_col], g[value], 1)[0] if g[level_col].nunique() > 1 else np.nan,
                                                     include_groups=False).rename("slope").reset_index())
    return agg.merge(slopes, on=["model", kind_col])
