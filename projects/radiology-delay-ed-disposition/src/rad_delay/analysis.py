"""Descriptives, adjusted models, instrumental variables and regression discontinuity in time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def describe_tat(exams: pd.DataFrame, by: tuple[str, ...] = ("modality",)) -> pd.DataFrame:
    """TAT quantiles, auto-stored share and quarantine share by grouping columns."""
    d = exams[~exams["is_addendum"]]
    g = d.groupby(list(by))
    out = g["tat_h"].quantile([0.25, 0.5, 0.75, 0.9]).unstack()
    out.columns = [f"tat_q{int(q*100)}" for q in out.columns]
    out["n"] = g.size()
    out["auto_stored_share"] = g["auto_stored"].mean()
    out["quarantine_share"] = g["tat_quarantine"].mean()
    return out.reset_index()


def decision_before_report(linked: pd.DataFrame, by: tuple[str, ...] = ("modality", "disposition")) -> pd.DataFrame:
    """Share of reports whose storetime is after ED outtime, with Wilson CI, per group."""
    from scipy import stats

    d = linked[~linked["is_addendum"] & ~linked["tat_quarantine"]]
    rows = []
    z = stats.norm.ppf(0.975)
    for keys, g in d.groupby(list(by)):
        n, k = len(g), int(g["report_after_exit"].sum())
        p = k / n if n else np.nan
        den = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"n": n, "after_exit": k, "share": p, "ci_low": c - h, "ci_high": c + h, "median_report_to_exit_h": float(g["report_to_exit_h"].median())})
        rows.append(row)
    return pd.DataFrame(rows)


def adjusted_model(df: pd.DataFrame, outcome: str, exposure: str, covariates: list[str], family: str = "logit", cluster: str | None = "subject_id"):
    """Fit outcome ~ exposure + covariates with cluster-robust SEs; family 'logit' or 'ols'.

    Returns the statsmodels results object (use ``.params[exposure]`` etc.).
    """
    formula = f"{outcome} ~ {exposure}" + "".join(f" + {c}" for c in covariates)
    model = smf.logit(formula, data=df) if family == "logit" else smf.ols(formula, data=df)
    if cluster and cluster in df.columns:
        return model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df[cluster].to_numpy()}) if family == "logit" else model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster].to_numpy()})
    return model.fit(disp=False) if family == "logit" else model.fit()


def two_stage_least_squares(df: pd.DataFrame, y: str, endog: str, instruments: list[str], exog: list[str]) -> dict[str, float]:
    """Manual 2SLS of y on an endogenous regressor with instruments and exogenous controls.

    Returns the IV coefficient, its heteroskedasticity-robust SE, the first-stage F
    statistic for the excluded instruments, the OLS coefficient for comparison, and
    the Sargan over-identification statistic / p-value when #instruments > 1.
    """
    from scipy import stats

    d = df.dropna(subset=[y, endog] + instruments + exog)
    X_exog = sm.add_constant(d[exog].astype(float)) if exog else pd.DataFrame({"const": np.ones(len(d))}, index=d.index)
    Z = pd.concat([X_exog, d[instruments].astype(float)], axis=1)
    # first stage
    fs = sm.OLS(d[endog].astype(float), Z).fit()
    fs_r = sm.OLS(d[endog].astype(float), X_exog).fit()
    q = len(instruments)
    f_stat = ((fs_r.ssr - fs.ssr) / q) / (fs.ssr / fs.df_resid)
    xhat = fs.fittedvalues
    X2 = pd.concat([X_exog, xhat.rename(endog)], axis=1)
    ss = sm.OLS(d[y].astype(float), X2).fit()
    # structural residuals with the actual endogenous regressor (not fitted) for correct SEs
    X_true = pd.concat([X_exog, d[endog].astype(float)], axis=1)
    u = d[y].astype(float) - X_true @ ss.params.values
    Zm = Z.to_numpy()
    Xm = X_true.to_numpy()
    PZ = Zm @ np.linalg.pinv(Zm.T @ Zm) @ Zm.T
    A = np.linalg.pinv(Xm.T @ PZ @ Xm)
    meat = (PZ @ Xm).T @ (np.diag(u.to_numpy() ** 2)) @ (PZ @ Xm) if len(d) <= 5000 else ((PZ @ Xm) * (u.to_numpy() ** 2)[:, None]).T @ (PZ @ Xm)
    V = A @ meat @ A
    se_iv = float(np.sqrt(V[-1, -1]))
    ols = sm.OLS(d[y].astype(float), X_true).fit(cov_type="HC1")
    out = {"iv_coef": float(ss.params[endog]), "iv_se": se_iv, "first_stage_F": float(f_stat), "ols_coef": float(ols.params[endog]), "ols_se": float(ols.bse[endog]), "n": int(len(d))}
    if q > 1:
        aux = sm.OLS(u, Z).fit()
        sargan = len(d) * aux.rsquared
        out["sargan"], out["sargan_p"] = float(sargan), float(1 - stats.chi2.cdf(sargan, q - 1))
    return out


def rd_in_time(df: pd.DataFrame, outcome: str, running: str = "hours_from_shift_start", cutoff: float = 0.0, bandwidth: float = 2.0, covariates: list[str] | None = None) -> dict[str, float]:
    """Local-linear regression discontinuity in time-of-day at ``cutoff`` with triangular kernel weights."""
    d = df.dropna(subset=[outcome, running]).copy()
    r = d[running].astype(float) - cutoff
    w = np.clip(1 - np.abs(r) / bandwidth, 0, None)
    d = d[w > 0].assign(_r=r[w > 0], _w=w[w > 0], _t=(r[w > 0] >= 0).astype(int))
    cov = "".join(f" + {c}" for c in (covariates or []))
    res = smf.wls(f"{outcome} ~ _t + _r + _t:_r{cov}", data=d, weights=d["_w"]).fit(cov_type="HC1")
    return {"rd_estimate": float(res.params["_t"]), "se": float(res.bse["_t"]), "p": float(res.pvalues["_t"]), "n_left": int((d["_t"] == 0).sum()), "n_right": int((d["_t"] == 1).sum()), "bandwidth": bandwidth}


def negative_control_check(df: pd.DataFrame, exposure: str, negative_outcomes: list[str], covariates: list[str]) -> pd.DataFrame:
    """OLS of each negative-control outcome on the exposure; coefficients should be ~0."""
    rows = []
    for o in negative_outcomes:
        res = adjusted_model(df.dropna(subset=[o]), o, exposure, covariates, family="ols")
        rows.append({"negative_outcome": o, "coef": float(res.params[exposure]), "se": float(res.bse[exposure]), "p": float(res.pvalues[exposure])})
    return pd.DataFrame(rows)


def permutation_null(df: pd.DataFrame, outcome: str, exposure: str, covariates: list[str], strata: list[str], n_perm: int = 200, seed: int = 0, family: str = "ols") -> dict[str, float]:
    """Permute the exposure within strata (e.g. modality x hour_block) and refit; returns observed coefficient and permutation p-value."""
    rng = np.random.default_rng(seed)
    obs = float(adjusted_model(df, outcome, exposure, covariates, family=family, cluster=None).params[exposure])
    null = []
    for _ in range(n_perm):
        p = df.copy()
        p[exposure] = p.groupby(strata)[exposure].transform(lambda s: s.to_numpy()[rng.permutation(len(s))])
        null.append(float(adjusted_model(p, outcome, exposure, covariates, family=family, cluster=None).params[exposure]))
    null = np.asarray(null)
    return {"observed": obs, "p_value": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)), "null_sd": float(null.std())}
