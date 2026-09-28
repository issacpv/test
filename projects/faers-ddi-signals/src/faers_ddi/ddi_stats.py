"""Drug-drug interaction (DDI) signal statistics for spontaneous reports.

Implemented measures (all take the eight ``n_ijk`` cells from
:mod:`faers_ddi.ddi_tables`):

* :func:`omega` - shrinkage observed-to-expected ratio for the combination,
  Omega = log2((n111 + 0.5) / (E111 + 0.5)), with the credibility bound
  Omega025 (Noren, Sundberg, Bate & Edwards, 2008, *Stat Med*). E111 =
  n11+ * g11 where g11 is the relative reporting rate expected under a
  no-interaction model. Several no-interaction models are exposed because the
  choice matters and is itself part of this project's methods comparison:
  ``"independence"`` (1 - (1-f10)(1-f01)/(1-f00)), ``"additive"``
  (f10 + f01 - f00), ``"multiplicative"`` (f10 f01 / f00) and ``"max"``
  (max(f10, f01)). Consult Noren et al. (2008) for the model they recommend
  before reporting; the default here is ``"independence"``.
* :func:`interaction_ror` - the interaction reporting odds ratio (IOR):
  exp(beta_AB) from a logistic model event ~ A + B + A:B fitted to the
  aggregated 2x2x2 table (van Puijenbroek et al., 1999, *Br J Clin
  Pharmacol*), with Wald CI and likelihood-ratio p-value.
* :func:`additive_excess` - relative excess reporting due to interaction on
  the odds-ratio scale (RERI-type additive-interaction contrast; cf. Thakrar,
  Grundschober & Doessegger, 2007, *Br J Clin Pharmacol*).
* :func:`sex_specific_interaction` - three-way A x B x sex term from
  event ~ (A + B + A:B) * sex; returns the ratio of IORs (women / men).
* :func:`hierarchical_shrinkage` - empirical-Bayes normal-normal shrinkage of
  log-IORs towards a group mean (e.g. all pairs sharing a CYP3A4-inhibitor
  mechanism), with DerSimonian-Laird between-pair variance. This is the
  "Bayesian hierarchical" component that pools strength across pairs with the
  same pharmacological mechanism.
* :func:`benjamini_hochberg` - FDR control across the (pair, event) screen.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

CELLS = ("n111", "n110", "n101", "n100", "n011", "n010", "n001", "n000")


def _rates(c: Dict[str, float]) -> Dict[str, float]:
    def f(k1: str, k0: str) -> float:
        tot = c[k1] + c[k0]
        return c[k1] / tot if tot > 0 else 0.0

    return {"f11": f("n111", "n110"), "f10": f("n101", "n100"), "f01": f("n011", "n010"), "f00": f("n001", "n000")}


def expected_rate(f10: float, f01: float, f00: float, model: str = "independence") -> float:
    """Relative reporting rate of E under the chosen no-interaction model."""
    if model == "independence":
        denom = max(1.0 - f00, 1e-12)
        g = 1.0 - (1.0 - f10) * (1.0 - f01) / denom
    elif model == "additive":
        g = f10 + f01 - f00
    elif model == "multiplicative":
        g = f10 * f01 / f00 if f00 > 0 else max(f10, f01)
    elif model == "max":
        g = max(f10, f01)
    else:
        raise ValueError(f"unknown no-interaction model: {model}")
    return float(min(max(g, 0.0), 1.0))


def omega(c: Dict[str, float], model: str = "independence") -> Dict[str, float]:
    """Shrinkage observed-to-expected ratio Omega and its lower bound Omega025.

    The Omega025 approximation used here mirrors the IC025 approximation of
    the same authors: Omega - 3.3 (n111 + 0.5)^(-1/2) - 2 (n111 + 0.5)^(-3/2).
    A signal is conventionally Omega025 > 0.
    """
    r = _rates(c)
    n11 = c["n111"] + c["n110"]
    g11 = expected_rate(r["f10"], r["f01"], r["f00"], model)
    e111 = n11 * g11
    o = c["n111"]
    om = np.log2((o + 0.5) / (e111 + 0.5))
    om025 = om - 3.3 * (o + 0.5) ** (-0.5) - 2.0 * (o + 0.5) ** (-1.5)
    return {"omega": float(om), "omega025": float(om025), "expected": float(e111), "observed": float(o), "g11": g11, "model": model}


def _aggregated_rows(c: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for key in CELLS:
        i, j, k = int(key[1]), int(key[2]), int(key[3])
        rows.append({"A": i, "B": j, "event": k, "w": float(c[key])})
    df = pd.DataFrame(rows)
    return df[df["w"] > 0].reset_index(drop=True)


def _fit_logit(df: pd.DataFrame, cols: Sequence[str]) -> sm.GLM:
    X = sm.add_constant(df[list(cols)].astype(float), has_constant="add")
    return sm.GLM(df["event"].astype(float), X, family=sm.families.Binomial(), freq_weights=df["w"]).fit()


def interaction_ror(c: Dict[str, float], cc: float = 0.5) -> Dict[str, float]:
    """Interaction ROR (exp of the A:B logistic coefficient) with CI and LRT p.

    A continuity correction ``cc`` is added to every cell only when one is
    zero, so the model remains identifiable.
    """
    if min(c[k] for k in CELLS) == 0:
        c = {k: c[k] + cc for k in CELLS}
    df = _aggregated_rows(c)
    df["AB"] = df["A"] * df["B"]
    m0 = _fit_logit(df, ["A", "B"])
    m1 = _fit_logit(df, ["A", "B", "AB"])
    beta = float(m1.params["AB"])
    se = float(m1.bse["AB"])
    lr = 2.0 * (m1.llf - m0.llf)
    return {
        "ior": float(np.exp(beta)),
        "ior_lo": float(np.exp(beta - 1.96 * se)),
        "ior_hi": float(np.exp(beta + 1.96 * se)),
        "log_ior": beta,
        "se_log_ior": se,
        "ror_a": float(np.exp(m1.params["A"])),
        "ror_b": float(np.exp(m1.params["B"])),
        "p_lrt": float(stats.chi2.sf(max(lr, 0.0), 1)),
        "n111": float(c["n111"]),
    }


def additive_excess(c: Dict[str, float], cc: float = 0.5) -> Dict[str, float]:
    """RERI-type additive interaction on the reporting-odds scale.

    RERI = OR_11 - OR_10 - OR_01 + 1, where OR_ij is the odds of E in stratum
    (i, j) relative to (0, 0). RERI > 0 indicates more reports with the
    combination than expected if the two drugs' excess odds simply added.
    Delta-method SE (Hosmer & Lemeshow, 1992).
    """
    if min(c[k] for k in CELLS) == 0:
        c = {k: c[k] + cc for k in CELLS}
    odds = {ij: c[f"n{ij}1"] / c[f"n{ij}0"] for ij in ("11", "10", "01", "00")}
    var = {ij: 1.0 / c[f"n{ij}1"] + 1.0 / c[f"n{ij}0"] for ij in ("11", "10", "01", "00")}
    or11, or10, or01 = odds["11"] / odds["00"], odds["10"] / odds["00"], odds["01"] / odds["00"]
    reri = or11 - or10 - or01 + 1.0
    # delta method on log-odds of each stratum vs baseline (independent strata)
    grad = np.array([or11, -or10, -or01, -(or11 - or10 - or01)])
    v = np.array([var["11"], var["10"], var["01"], var["00"]])
    se = float(np.sqrt(np.sum(grad**2 * v)))
    return {"reri": float(reri), "reri_lo": float(reri - 1.96 * se), "reri_hi": float(reri + 1.96 * se), "or11": float(or11), "or10": float(or10), "or01": float(or01)}


def sex_specific_interaction(c_f: Dict[str, float], c_m: Dict[str, float], cc: float = 0.5) -> Dict[str, float]:
    """Does the A x B interaction differ between women and men?

    Fits event ~ (A + B + A:B) * sex on the stacked female/male 2x2x2 tables.
    Returns the female and male IORs, their ratio with Wald CI, and an LRT
    p-value for the three-way A:B:sex term.
    """
    frames = []
    for sex_val, c in ((1, c_f), (0, c_m)):
        if min(c[k] for k in CELLS) == 0:
            c = {k: c[k] + cc for k in CELLS}
        df = _aggregated_rows(c)
        df["sex"] = sex_val
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["AB"] = df["A"] * df["B"]
    df["As"] = df["A"] * df["sex"]
    df["Bs"] = df["B"] * df["sex"]
    df["ABs"] = df["AB"] * df["sex"]
    m0 = _fit_logit(df, ["A", "B", "AB", "sex", "As", "Bs"])
    m1 = _fit_logit(df, ["A", "B", "AB", "sex", "As", "Bs", "ABs"])
    beta3 = float(m1.params["ABs"])
    se3 = float(m1.bse["ABs"])
    lr = 2.0 * (m1.llf - m0.llf)
    ior_m = float(np.exp(m1.params["AB"]))
    ior_f = float(np.exp(m1.params["AB"] + beta3))
    return {
        "ior_female": ior_f,
        "ior_male": ior_m,
        "ratio_of_ior": float(np.exp(beta3)),
        "ratio_lo": float(np.exp(beta3 - 1.96 * se3)),
        "ratio_hi": float(np.exp(beta3 + 1.96 * se3)),
        "p_lrt": float(stats.chi2.sf(max(lr, 0.0), 1)),
    }


def hierarchical_shrinkage(estimates: pd.DataFrame, value: str = "log_ior", se: str = "se_log_ior", group: str = "mechanism") -> pd.DataFrame:
    """Empirical-Bayes shrinkage of per-pair log-IORs towards their group mean.

    Model: theta_p ~ N(mu_g, tau_g^2); y_p | theta_p ~ N(theta_p, s_p^2).
    ``tau_g^2`` is estimated by DerSimonian-Laird (floored at 0) and ``mu_g``
    by inverse-variance weighting. Posterior mean and SD are returned per pair
    together with the shrinkage factor B_p = s_p^2 / (s_p^2 + tau_g^2).
    Groups with a single pair are returned unshrunk (tau undefined).
    """
    out = estimates.copy()
    out["post_mean"] = out[value].astype(float)
    out["post_sd"] = out[se].astype(float)
    out["shrink_b"] = 0.0
    out["group_mu"] = np.nan
    out["group_tau2"] = np.nan
    for g, sub in out.groupby(group):
        y = sub[value].astype(float).values
        s2 = sub[se].astype(float).values ** 2
        k = len(y)
        if k < 2:
            continue
        w = 1.0 / s2
        mu_fe = np.sum(w * y) / np.sum(w)
        q = np.sum(w * (y - mu_fe) ** 2)
        cfac = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2 = max((q - (k - 1)) / cfac, 0.0) if cfac > 0 else 0.0
        w_re = 1.0 / (s2 + tau2)
        mu = np.sum(w_re * y) / np.sum(w_re)
        b = s2 / (s2 + tau2)
        post_mean = (1 - b) * y + b * mu
        post_var = (1 - b) * s2
        idx = sub.index
        out.loc[idx, "post_mean"] = post_mean
        out.loc[idx, "post_sd"] = np.sqrt(post_var)
        out.loc[idx, "shrink_b"] = b
        out.loc[idx, "group_mu"] = mu
        out.loc[idx, "group_tau2"] = tau2
    out["post_lo"] = out["post_mean"] - 1.96 * out["post_sd"]
    out["post_hi"] = out["post_mean"] + 1.96 * out["post_sd"]
    return out


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """BH q-values and rejection flags."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return pd.DataFrame(columns=["p", "q", "reject"])
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.minimum(q, 1.0)
    return pd.DataFrame({"p": p, "q": q_full, "reject": q_full <= alpha})


def screen(tables: pd.DataFrame, omega_model: str = "independence", alpha: float = 0.05) -> pd.DataFrame:
    """Apply Omega, IOR and RERI to every row of a ``pair_event_table`` frame."""
    rows = []
    for _, r in tables.iterrows():
        c = {k: float(r[k]) for k in CELLS}
        d = omega(c, omega_model)
        d.update(interaction_ror(c))
        d.update(additive_excess(c))
        rows.append(d)
    est = pd.DataFrame(rows, index=tables.index).drop(columns=["model", "n111"], errors="ignore")
    out = pd.concat([tables, est], axis=1)
    out["q_ior"] = benjamini_hochberg(out["p_lrt"].fillna(1.0).values, alpha)["q"].values
    out["omega_signal"] = out["omega025"] > 0
    out["ior_signal"] = (out["ior_lo"] > 1) & (out["n111"] >= 3)
    return out
