"""Disparity statistics: adjusted associations, under-triage by group, matching, mediation, model fairness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score


def adjusted_odds(df: pd.DataFrame, outcome: str, exposure: str, covariates: list[str], cluster: str | None = "subject_id") -> pd.DataFrame:
    """Logistic regression outcome ~ exposure + covariates; returns OR table for exposure terms.

    Categorical covariates should be wrapped as C(col) in ``covariates``.  Cluster-robust
    SEs by ``cluster`` when given.
    """
    formula = f"{outcome} ~ {exposure}" + "".join(f" + {c}" for c in covariates)
    model = smf.logit(formula, data=df)
    if cluster is not None and cluster in df.columns:
        res = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df[cluster].to_numpy()})
    else:
        res = model.fit(disp=False)
    ci = res.conf_int()
    rows = []
    for name in res.params.index:
        if name.startswith(exposure) or exposure in name.split("[")[0]:
            rows.append({"term": name, "or": float(np.exp(res.params[name])), "ci_low": float(np.exp(ci.loc[name, 0])), "ci_high": float(np.exp(ci.loc[name, 1])), "p": float(res.pvalues[name])})
    return pd.DataFrame(rows)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def under_triage_by_group(df: pd.DataFrame, group_col: str = "language_group", acuity_col: str = "acuity", outcome: str = "critical", reference: str = "english", min_n: int = 50) -> pd.DataFrame:
    """Critical-outcome rate within each acuity level by group, with Wilson CIs and risk ratio vs reference."""
    rows = []
    d = df.dropna(subset=[acuity_col, outcome, group_col])
    for acu, g in d.groupby(acuity_col):
        ref = g[g[group_col] == reference]
        ref_rate = ref[outcome].mean() if len(ref) else np.nan
        for grp, h in g.groupby(group_col):
            n, k = len(h), int(h[outcome].sum())
            if n < min_n:
                continue
            lo, hi = wilson_ci(k, n)
            rows.append({acuity_col: acu, group_col: grp, "n": n, "events": k, "rate": k / n, "ci_low": lo, "ci_high": hi, "risk_ratio": (k / n) / ref_rate if ref_rate and ref_rate > 0 else np.nan})
    return pd.DataFrame(rows)


def esi_discrimination_by_group(df: pd.DataFrame, group_col: str = "language_group", acuity_col: str = "acuity", outcome: str = "critical", n_boot: int = 500, seed: int = 0, min_n: int = 200) -> pd.DataFrame:
    """AUROC of the (reversed) ESI acuity as a score for the critical outcome, per group, with bootstrap CI."""
    rng = np.random.default_rng(seed)
    rows = []
    d = df.dropna(subset=[acuity_col, outcome, group_col])
    for grp, g in d.groupby(group_col):
        y, s = g[outcome].to_numpy(int), (-g[acuity_col].to_numpy(float))
        if len(g) < min_n or y.min() == y.max():
            continue
        auc = roc_auc_score(y, s)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                boots.append(roc_auc_score(y[i], s[i]))
        rows.append({group_col: grp, "n": len(g), "auroc": auc, "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975))})
    return pd.DataFrame(rows)


def exact_match(df: pd.DataFrame, treat_col: str, match_cols: list[str], ratio: int = 4, seed: int = 0) -> pd.DataFrame:
    """1:k exact matching of treated (treat_col == 1) to controls on ``match_cols``. Returns matched rows with a ``match_id``."""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=match_cols).copy()
    d["_key"] = d[match_cols].astype(str).agg("|".join, axis=1)
    out = []
    mid = 0
    controls_by_key = {k: g.index.to_numpy() for k, g in d[d[treat_col] == 0].groupby("_key")}
    used: set = set()
    for idx, row in d[d[treat_col] == 1].iterrows():
        pool = controls_by_key.get(row["_key"])
        if pool is None:
            continue
        avail = [i for i in pool if i not in used]
        if not avail:
            continue
        take = rng.choice(avail, size=min(ratio, len(avail)), replace=False)
        used.update(take.tolist())
        block = d.loc[[idx, *take]].copy()
        block["match_id"] = mid
        out.append(block)
        mid += 1
    return pd.concat(out).drop(columns="_key") if out else d.iloc[0:0].assign(match_id=[])


def mediation_analysis(df: pd.DataFrame, exposure: str, mediator: str, outcome: str, covariates: list[str], n_boot: int = 500, seed: int = 0, cluster: str | None = "subject_id") -> dict[str, float]:
    """Regression-based natural direct / indirect effects on the risk-difference scale.

    Mediator model: mediator ~ exposure + covariates (OLS).  Outcome model:
    outcome ~ exposure + mediator + covariates (logit).  NIE and NDE are computed by
    standardisation (g-computation) over the observed covariate distribution; CIs by
    (cluster) bootstrap.  Returns total, nde, nie, proportion_mediated with CIs.
    """
    cov = "".join(f" + {c}" for c in covariates)
    f_m = f"{mediator} ~ {exposure}{cov}"
    f_y = f"{outcome} ~ {exposure} + {mediator}{cov}"

    def _effects(d: pd.DataFrame) -> tuple[float, float, float]:
        m_mod = smf.ols(f_m, data=d).fit()
        y_mod = smf.logit(f_y, data=d).fit(disp=False, maxiter=200)
        d1, d0 = d.copy(), d.copy()
        d1[exposure], d0[exposure] = 1, 0
        m1, m0 = m_mod.predict(d1), m_mod.predict(d0)
        # Y(a=1, M(1)), Y(a=1, M(0)), Y(a=0, M(0))
        d11, d10, d00 = d1.copy(), d1.copy(), d0.copy()
        d11[mediator], d10[mediator], d00[mediator] = m1, m0, m0
        y11, y10, y00 = y_mod.predict(d11).mean(), y_mod.predict(d10).mean(), y_mod.predict(d00).mean()
        return float(y11 - y00), float(y10 - y00), float(y11 - y10)

    total, nde, nie = _effects(df)
    rng = np.random.default_rng(seed)
    boots = []
    if cluster is not None and cluster in df.columns:
        groups = {g: h for g, h in df.groupby(cluster)}
        keys = np.array(list(groups))
        for _ in range(n_boot):
            draw = rng.choice(keys, len(keys), replace=True)
            try:
                boots.append(_effects(pd.concat([groups[k] for k in draw], ignore_index=True)))
            except Exception:  # singular bootstrap sample
                continue
    else:
        for _ in range(n_boot):
            i = rng.integers(0, len(df), len(df))
            try:
                boots.append(_effects(df.iloc[i]))
            except Exception:
                continue
    b = np.asarray(boots) if boots else np.full((1, 3), np.nan)
    q = lambda j, a: float(np.nanquantile(b[:, j], a))  # noqa: E731
    return {
        "total": total, "total_ci_low": q(0, 0.025), "total_ci_high": q(0, 0.975),
        "nde": nde, "nde_ci_low": q(1, 0.025), "nde_ci_high": q(1, 0.975),
        "nie": nie, "nie_ci_low": q(2, 0.025), "nie_ci_high": q(2, 0.975),
        "proportion_mediated": nie / total if total != 0 else np.nan, "n_boot_ok": int(len(boots)),
    }


def group_fairness(y: np.ndarray, p: np.ndarray, group: np.ndarray, alert_rate: float = 0.10, min_n: int = 100) -> pd.DataFrame:
    """Per-group AUROC, calibration intercept/slope, sensitivity and FPR at a global alert-rate threshold."""
    y, p, group = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(group)
    thr = np.quantile(p, 1 - alert_rate)
    rows = []
    for g in np.unique(group):
        m = group == g
        if m.sum() < min_n or y[m].min() == y[m].max():
            continue
        z = np.log(np.clip(p[m], 1e-6, 1 - 1e-6) / (1 - np.clip(p[m], 1e-6, 1 - 1e-6)))
        slope = float(sm.Logit(y[m], sm.add_constant(z)).fit(disp=False).params[1])
        intercept = float(sm.GLM(y[m], np.ones((m.sum(), 1)), family=sm.families.Binomial(), offset=z).fit().params[0])
        pred = p[m] >= thr
        rows.append({"group": g, "n": int(m.sum()), "prevalence": float(y[m].mean()), "auroc": float(roc_auc_score(y[m], p[m])), "cal_intercept": intercept, "cal_slope": slope, "sensitivity": float(pred[y[m] == 1].mean()), "fpr": float(pred[y[m] == 0].mean()), "alert_rate": float(pred.mean())})
    out = pd.DataFrame(rows)
    for c in ("auroc", "cal_intercept", "cal_slope", "sensitivity", "fpr"):
        if len(out):
            out.attrs[f"gap_{c}"] = float(out[c].max() - out[c].min())
    return out
