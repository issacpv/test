"""Specification grid, multiverse runner, specification curves and variance decomposition."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import models as km
from .tacs import FrameTiming


@dataclass(frozen=True)
class Specification:
    model: str = "srtm"               # srtm | srtm2 | logan_ref | logan_ref_nok2 | mrtm | mrtm2 | suvr
    ref_region: str = "cerebellum"
    t_star: float = 30.0              # minutes; used by logan/mrtm
    weighting: str = "duration"       # uniform | duration | counts
    scan_minutes: float = 90.0        # truncation of the scan
    suvr_window: tuple[float, float] = (60.0, 90.0)
    k2prime_source: str = "population"  # population | fixed
    k2prime_fixed: float = 0.1

    def label(self) -> str:
        return f"{self.model}|{self.ref_region}|t*{self.t_star:g}|{self.weighting}|{self.scan_minutes:g}min|{self.k2prime_source}"


def specification_grid(
    models=("srtm", "srtm2", "logan_ref", "mrtm2", "suvr"),
    ref_regions=("cerebellum",),
    t_stars=(20.0, 30.0, 40.0),
    weightings=("uniform", "duration"),
    scan_minutes=(60.0, 90.0),
    k2prime_sources=("population",),
) -> list[Specification]:
    """Cartesian product with model-irrelevant factors collapsed (e.g., t* only for graphical models)."""
    specs = set()
    for mdl, ref, ts, w, sm, ks in itertools.product(models, ref_regions, t_stars, weightings, scan_minutes, k2prime_sources):
        ts_eff = ts if mdl in ("logan_ref", "logan_ref_nok2", "mrtm", "mrtm2") else 0.0
        w_eff = w if mdl != "suvr" else "uniform"
        ks_eff = ks if mdl in ("srtm2", "mrtm2", "logan_ref") else "population"
        window = (max(sm - 30.0, 0.0), sm)
        specs.add(Specification(mdl, ref, ts_eff, w_eff, sm, window, ks_eff))
    return sorted(specs, key=lambda s: s.label())


def fit_specification(ct: np.ndarray, cr: np.ndarray, timing: FrameTiming, spec: Specification,
                      k2prime_population: float | None = None) -> dict:
    """Fit one region under one specification (handles truncation, weighting, k2′ source)."""
    keep = timing.end <= spec.scan_minutes + 1e-9
    timing_t = FrameTiming(timing.start[keep], timing.duration[keep])
    ct_t, cr_t = np.asarray(ct)[keep], np.asarray(cr)[keep]
    w = km.frame_weights(timing_t, ct_t, spec.weighting)
    k2p = spec.k2prime_fixed if spec.k2prime_source == "fixed" else k2prime_population
    if spec.model == "srtm":
        return km.fit_srtm(ct_t, cr_t, timing_t, w)
    if spec.model == "srtm2":
        return km.fit_srtm2(ct_t, cr_t, timing_t, k2p, w)
    if spec.model == "logan_ref":
        return km.fit_logan_ref(ct_t, cr_t, timing_t, spec.t_star, k2p, w)
    if spec.model == "logan_ref_nok2":
        return km.fit_logan_ref(ct_t, cr_t, timing_t, spec.t_star, None, w)
    if spec.model == "mrtm":
        return km.fit_mrtm(ct_t, cr_t, timing_t, spec.t_star, w)
    if spec.model == "mrtm2":
        return km.fit_mrtm2(ct_t, cr_t, timing_t, spec.t_star, k2p, w)
    if spec.model == "suvr":
        return km.fit_suvr(ct_t, cr_t, timing_t, spec.suvr_window)
    raise ValueError(spec.model)


def run_multiverse(long_tacs: pd.DataFrame, specs: list[Specification], high_binding_regions: list[str],
                   ref_column: str = "cr") -> pd.DataFrame:
    """Run all specifications on a long TAC table (subject, session, region, frame, start_min, duration_min, ct, cr).

    ``cr`` is the reference TAC for the specification's ``ref_region``; if the table has
    several reference columns, name them ``cr_<ref_region>``.
    Population k2′ is estimated per (subject, session) from SRTM in high-binding regions
    with the *same* scan truncation and weighting as the specification.
    """
    out = []
    for (subj, ses), scan in long_tacs.groupby(["subject", "session"]):
        first = scan[scan["region"] == scan["region"].iloc[0]].sort_values("frame")
        timing = FrameTiming(first["start_min"].to_numpy(float), first["duration_min"].to_numpy(float))
        tacs = {r: g.sort_values("frame")["ct"].to_numpy(float) for r, g in scan.groupby("region")}
        for spec in specs:
            col = f"cr_{spec.ref_region}" if f"cr_{spec.ref_region}" in scan.columns else ref_column
            cr = first[col].to_numpy(float)
            k2p = None
            if spec.model in ("srtm2", "mrtm2", "logan_ref") and spec.k2prime_source == "population":
                keep = timing.end <= spec.scan_minutes + 1e-9
                timing_t = FrameTiming(timing.start[keep], timing.duration[keep])
                k2p = km.population_k2prime({r: v[keep] for r, v in tacs.items()}, cr[keep], timing_t, high_binding_regions,
                                            km.frame_weights(timing_t, cr[keep], spec.weighting))
            for region, ct in tacs.items():
                res = fit_specification(ct, cr, timing, spec, k2p)
                out.append({"subject": subj, "session": ses, "region": region, **asdict(spec), "spec": spec.label(),
                            "bp": res.get("bp", np.nan), "rss": res.get("rss", np.nan), "k2prime_used": k2p})
    return pd.DataFrame(out)


def specification_curve(results: pd.DataFrame, value: str = "bp", by: tuple[str, ...] = ("region",)) -> pd.DataFrame:
    """Median (and IQR) of ``value`` per specification, sorted — the classic specification curve."""
    g = results.groupby(["spec", *by])[value]
    curve = g.agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="count").reset_index()
    return curve.sort_values([*by, "median"]).reset_index(drop=True)


def variance_decomposition(results: pd.DataFrame, value: str = "bp",
                           factors: tuple[str, ...] = ("subject", "region", "model", "t_star", "weighting", "scan_minutes")) -> pd.DataFrame:
    """η² per factor from a sequential (type-I) OLS ANOVA with main effects; residual is 'unexplained'."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = results.dropna(subset=[value]).copy()
    d = d[np.isfinite(d[value])]
    # clip absurd fits so a single divergent Logan/MRTM fit does not dominate the ANOVA
    lo, hi = d[value].quantile([0.005, 0.995])
    d[value] = d[value].clip(lo, hi)
    terms = []
    for f in factors:
        if f in d.columns and d[f].nunique() > 1:
            d[f] = d[f].astype(str)
            terms.append(f"C({f})")
    fit = smf.ols(f"{value} ~ " + " + ".join(terms), data=d).fit()
    table = sm.stats.anova_lm(fit, typ=1)
    total = table["sum_sq"].sum()
    out = table[["sum_sq", "df"]].copy()
    out["eta_sq"] = out["sum_sq"] / total
    out.index = [i.replace("C(", "").replace(")", "") for i in out.index]
    return out.reset_index().rename(columns={"index": "factor"})


def contrast_robustness(results: pd.DataFrame, group_col: str, region: str, value: str = "bp") -> pd.DataFrame:
    """For each specification, Welch t-test of ``value`` between two groups in one region."""
    from scipy.stats import ttest_ind

    rows = []
    sub = results[results["region"] == region]
    levels = sorted(sub[group_col].dropna().unique())
    if len(levels) != 2:
        raise ValueError("group_col must have exactly two levels")
    for spec, g in sub.groupby("spec"):
        a = g[g[group_col] == levels[0]].groupby("subject")[value].mean()
        b = g[g[group_col] == levels[1]].groupby("subject")[value].mean()
        t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({"spec": spec, "diff": float(a.mean() - b.mean()), "cohen_d": float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan,
                     "t": float(t), "p": float(p)})
    out = pd.DataFrame(rows)
    out["same_sign_as_median"] = np.sign(out["diff"]) == np.sign(out["diff"].median())
    return out
