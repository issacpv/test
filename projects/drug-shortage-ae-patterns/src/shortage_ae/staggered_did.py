"""Staggered difference-in-differences for count panels.

Three estimators on the drug x month panel produced by :func:`shortage_ae.shortage_panel.build_panel`:

1. :func:`event_study_poisson` -- two-way fixed-effects Poisson regression with relative-time
   dummies (binned tails), optional exposure offset, cluster-robust SEs by drug. Transparent but
   known to be biased under heterogeneous effects with staggered adoption (Goodman-Bacon, 2021).
2. :func:`callaway_santanna_att` -- group-time ATTs on log rates using *not-yet-treated* drugs as
   controls (Callaway & Sant'Anna, 2021), aggregated to dynamic (event-time) effects with a
   drug-cluster bootstrap. This is the primary estimator.
3. :func:`placebo_permutation` -- randomly reassigns onset months across drugs to obtain a null
   distribution for any scalar summary.

Rates are ``n_outcome / n_exposure`` (e.g. error reports per all reports of the drug); a small
continuity constant avoids log(0) in sparse cells.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# --------------------------------------------------------------------------- event study
def _bin_rel_time(rel: pd.Series, pre: int, post: int) -> pd.Series:
    r = rel.copy()
    r = r.where(r >= -pre, -pre)
    r = r.where(r <= post, post)
    return r


def event_study_poisson(panel: pd.DataFrame, outcome: str = "n_error", exposure: Optional[str] = "n_all",
                        pre: int = 12, post: int = 12, ref: int = -1,
                        cluster: str = "drug") -> Tuple[pd.DataFrame, object]:
    """TWFE Poisson event study.

    Model: ``E[y_it] = exp(alpha_i + gamma_t + sum_k beta_k 1[rel_time_it = k] + log(exposure_it))``
    with never-treated drugs contributing only to the fixed effects. Relative times below ``-pre``
    or above ``post`` are binned into the end categories; ``ref`` is the omitted category.

    Returns
    -------
    (table, result) where ``table`` has ``rel_time, coef, se, irr, ci_lo, ci_hi, p`` and ``result``
    is the fitted statsmodels GLM (cluster-robust covariance).
    """
    d = panel.copy()
    d = d[d[outcome].notna()]
    d["rt"] = _bin_rel_time(d["rel_time"], pre, post)
    dummies: List[str] = []
    for k in range(-pre, post + 1):
        if k == ref:
            continue
        name = f"ev_m{abs(k)}" if k < 0 else f"ev_p{k}"
        d[name] = ((d["rt"] == k) & d["rel_time"].notna()).astype(float)
        dummies.append(name)
    formula = f"{outcome} ~ " + " + ".join(dummies) + " + C(drug) + C(t)"
    if exposure is not None:
        d["_expo"] = np.log(d[exposure].clip(lower=0.5))
        offset = d["_expo"].values
    else:
        offset = None
    model = smf.glm(formula, data=d, family=sm.families.Poisson(), offset=offset)
    groups = pd.factorize(d[cluster])[0]
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    rows = []
    for k in range(-pre, post + 1):
        if k == ref:
            rows.append({"rel_time": k, "coef": 0.0, "se": 0.0, "p": np.nan})
            continue
        name = f"ev_m{abs(k)}" if k < 0 else f"ev_p{k}"
        rows.append({"rel_time": k, "coef": float(res.params[name]), "se": float(res.bse[name]),
                     "p": float(res.pvalues[name])})
    tab = pd.DataFrame(rows)
    tab["irr"] = np.exp(tab["coef"])
    tab["ci_lo"] = np.exp(tab["coef"] - 1.96 * tab["se"])
    tab["ci_hi"] = np.exp(tab["coef"] + 1.96 * tab["se"])
    return tab, res


def pretrend_wald(result: object, pre: int, ref: int = -1, min_lead: int = 2) -> Dict[str, float]:
    """Joint Wald test that all leads from ``-pre`` to ``-min_lead`` are zero."""
    names = [f"ev_m{k}" for k in range(min_lead, pre + 1) if -k != ref]
    R = np.zeros((len(names), len(result.params)))
    idx = {n: i for i, n in enumerate(result.params.index)}
    for r, n in enumerate(names):
        R[r, idx[n]] = 1.0
    w = result.wald_test(R, scalar=True)
    return {"stat": float(w.statistic), "df": len(names), "p": float(w.pvalue)}


# --------------------------------------------------------------------------- Callaway-Sant'Anna
def _log_rate(df: pd.DataFrame, outcome: str, exposure: Optional[str], eps: float) -> pd.Series:
    if exposure is None:
        return np.log(df[outcome] + eps)
    return np.log((df[outcome] + eps) / (df[exposure] + eps))


def callaway_santanna_att(panel: pd.DataFrame, outcome: str = "n_error", exposure: Optional[str] = "n_all",
                          horizons: Sequence[int] = tuple(range(0, 13)), base_lag: int = 1,
                          control: str = "not_yet_treated", eps: float = 0.5,
                          n_boot: int = 200, seed: int = 0,
                          min_cohort_size: int = 1) -> pd.DataFrame:
    """Group-time ATTs on the log rate with not-yet-treated (or never-treated) controls.

    For cohort ``g`` (onset month index) and horizon ``h``:
    ``ATT(g, h) = [mean_{i in g} (y_{i,g+h} - y_{i,g-base_lag})] - [mean_{j in C(g,h)} (y_{j,g+h} - y_{j,g-base_lag})]``
    where ``C(g,h)`` are drugs not yet treated at ``g+h`` (``control="not_yet_treated"``) or never
    treated. Dynamic effects aggregate ATT(g,h) over cohorts weighted by cohort size. Bootstrap
    resamples drugs with replacement (cluster bootstrap) and recomputes the aggregation.

    Returns
    -------
    DataFrame ``horizon, att, se, ci_lo, ci_hi, n_cohorts, n_treated`` (log-rate scale; exp() gives IRR).
    """
    d = panel.copy()
    d["y"] = _log_rate(d, outcome, exposure, eps)
    wide = d.pivot(index="drug", columns="t", values="y")
    cohort = d.groupby("drug")["cohort"].first()
    T = wide.shape[1]
    drugs = wide.index.to_numpy()
    rng = np.random.default_rng(seed)

    def _att_table(sel_drugs: np.ndarray) -> Dict[int, Tuple[float, int, int]]:
        w = wide.loc[sel_drugs]
        c = cohort.loc[sel_drugs]
        out: Dict[int, Tuple[float, int, int]] = {}
        for h in horizons:
            num, den, ncoh, ntreat = 0.0, 0, 0, 0
            for g in sorted(c.dropna().unique()):
                g = int(g)
                t_post, t_pre = g + h, g - base_lag
                if t_pre < 0 or t_post >= T:
                    continue
                tr = c.index[c == g]
                if len(tr) < min_cohort_size:
                    continue
                if control == "never_treated":
                    ctrl = c.index[c.isna()]
                else:
                    ctrl = c.index[c.isna() | (c > t_post)]
                if len(ctrl) == 0:
                    continue
                d_tr = (w.loc[tr, t_post] - w.loc[tr, t_pre]).mean()
                d_ct = (w.loc[ctrl, t_post] - w.loc[ctrl, t_pre]).mean()
                att = float(d_tr - d_ct)
                num += att * len(tr)
                den += len(tr)
                ncoh += 1
                ntreat += len(tr)
            out[h] = (num / den if den else np.nan, ncoh, ntreat)
        return out

    point = _att_table(drugs)
    boots = {h: [] for h in horizons}
    for _ in range(n_boot):
        sel = rng.choice(drugs, size=len(drugs), replace=True)
        # pandas .loc with duplicates keeps duplicates, which is what a cluster bootstrap needs
        bt = _att_table(sel)
        for h in horizons:
            boots[h].append(bt[h][0])
    rows = []
    for h in horizons:
        att, ncoh, ntreat = point[h]
        b = np.array([v for v in boots[h] if np.isfinite(v)])
        se = float(b.std(ddof=1)) if len(b) > 1 else np.nan
        lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) > 1 else (np.nan, np.nan))
        rows.append({"horizon": h, "att": att, "se": se, "ci_lo": lo, "ci_hi": hi,
                     "n_cohorts": ncoh, "n_treated": ntreat})
    return pd.DataFrame(rows)


def aggregate_att(table: pd.DataFrame, horizons: Sequence[int]) -> Dict[str, float]:
    """Simple average of dynamic ATTs over ``horizons`` (e.g. 0..6) with a delta-method SE ignoring covariance."""
    sub = table[table["horizon"].isin(list(horizons))]
    att = float(sub["att"].mean())
    se = float(np.sqrt(np.nansum(sub["se"] ** 2)) / max(len(sub), 1))
    return {"att": att, "se": se, "irr": float(np.exp(att)), "irr_lo": float(np.exp(att - 1.96 * se)),
            "irr_hi": float(np.exp(att + 1.96 * se))}


# --------------------------------------------------------------------------- placebo
def reassign_onsets(panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute cohort (onset) months across *all* drugs, keeping the number of treated drugs and the
    calendar distribution of onsets; recompute ``rel_time`` and ``treated`` (resolution ignored)."""
    d = panel.copy()
    per_drug = d.groupby("drug")["cohort"].first()
    drugs = per_drug.index.to_numpy()
    cohorts = per_drug.to_numpy()
    new = rng.permutation(cohorts)
    mapping = dict(zip(drugs, new))
    d["cohort"] = d["drug"].map(mapping)
    d["rel_time"] = d["t"] - d["cohort"]
    d["treated"] = ((d["rel_time"] >= 0) & d["cohort"].notna()).astype(int)
    d["ever_treated"] = d["cohort"].notna().astype(int)
    return d


def placebo_permutation(panel: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], float], n_perm: int = 200,
                        seed: int = 0) -> Dict[str, object]:
    """Permutation null for a scalar statistic of the panel (e.g. aggregated ATT over horizons 0-6).

    Returns the observed statistic, the null draws and a two-sided p-value ``(1 + #|null| >= |obs|) / (1 + n)``.
    """
    rng = np.random.default_rng(seed)
    obs = float(stat_fn(panel))
    null = np.array([stat_fn(reassign_onsets(panel, rng)) for _ in range(n_perm)], dtype=float)
    null = null[np.isfinite(null)]
    p = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (1.0 + len(null))
    return {"observed": obs, "null": null, "p_value": float(p)}
