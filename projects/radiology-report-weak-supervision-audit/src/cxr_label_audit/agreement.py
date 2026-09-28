"""Agreement between report-derived labels and image-level expert labels, overall and by subgroup."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def sensitivity(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) else float("nan")

    @property
    def ppv(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def kappa(self) -> float:
        n = self.n
        if n == 0:
            return float("nan")
        po = (self.tp + self.tn) / n
        p_yes = ((self.tp + self.fp) / n) * ((self.tp + self.fn) / n)
        p_no = ((self.fn + self.tn) / n) * ((self.fp + self.tn) / n)
        pe = p_yes + p_no
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")

    @property
    def pabak(self) -> float:
        """Prevalence- and bias-adjusted kappa = 2·p_o − 1."""
        return 2 * (self.tp + self.tn) / self.n - 1 if self.n else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {"n": self.n, "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "sensitivity": self.sensitivity, "specificity": self.specificity, "ppv": self.ppv,
                "kappa": self.kappa, "pabak": self.pabak}


def resolve_report_labels(report: np.ndarray, unmentioned: str = "negative", uncertain: str = "positive") -> np.ndarray:
    """Map CheXpert-style {1, 0, -1, NaN} to binary; NaN stays NaN if ``unmentioned='missing'``."""
    r = np.asarray(report, float).copy()
    out = np.full(r.shape, np.nan)
    out[r == 1] = 1
    out[r == 0] = 0
    if uncertain == "positive":
        out[r == -1] = 1
    elif uncertain == "negative":
        out[r == -1] = 0
    if unmentioned == "negative":
        out[np.isnan(r)] = 0
    return out


def confusion(report_binary: np.ndarray, expert_binary: np.ndarray) -> Confusion:
    r, e = np.asarray(report_binary, float), np.asarray(expert_binary, float)
    ok = ~(np.isnan(r) | np.isnan(e))
    r, e = r[ok].astype(bool), e[ok].astype(bool)
    return Confusion(int(np.sum(r & e)), int(np.sum(r & ~e)), int(np.sum(~r & e)), int(np.sum(~r & ~e)))


def per_finding_agreement(df: pd.DataFrame, findings: list[str], report_suffix: str = "_report",
                          expert_suffix: str = "_expert", unmentioned: str = "negative",
                          uncertain: str = "positive") -> pd.DataFrame:
    rows = []
    for f in findings:
        rb = resolve_report_labels(df[f + report_suffix].to_numpy(), unmentioned, uncertain)
        c = confusion(rb, df[f + expert_suffix].to_numpy())
        rows.append({"finding": f, **c.as_dict()})
    return pd.DataFrame(rows).set_index("finding")


def cluster_bootstrap_ci(df: pd.DataFrame, stat_fn, cluster_col: str, n_boot: int = 1000, alpha: float = 0.05,
                         rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile CI of ``stat_fn(df) -> float`` resampling clusters (patients) with replacement."""
    rng = np.random.default_rng() if rng is None else rng
    est = float(stat_fn(df))
    groups = {k: v for k, v in df.groupby(cluster_col).indices.items()}
    keys = np.array(list(groups.keys()), dtype=object)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(keys, keys.size, replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        boots[b] = stat_fn(df.iloc[idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)


def disagreement_indicator(report_binary: np.ndarray, expert_binary: np.ndarray, kind: str = "fn") -> np.ndarray:
    """1 where the report label misses an expert-positive (``fn``) or calls an expert-negative (``fp``); NaN elsewhere
    (i.e. among the expert-positives for fn, expert-negatives for fp)."""
    r, e = np.asarray(report_binary, float), np.asarray(expert_binary, float)
    out = np.full(r.shape, np.nan)
    if kind == "fn":
        m = e == 1
        out[m] = (r[m] == 0).astype(float)
    elif kind == "fp":
        m = e == 0
        out[m] = (r[m] == 1).astype(float)
    else:
        raise ValueError(kind)
    out[np.isnan(r)] = np.nan
    return out


def stratified_rates(df: pd.DataFrame, indicator_col: str, by: str) -> pd.DataFrame:
    """Disagreement rate per subgroup with Wilson 95% intervals."""
    from scipy.stats import norm

    z = norm.ppf(0.975)
    rows = []
    for g, sub in df.dropna(subset=[indicator_col]).groupby(by):
        n = len(sub)
        p = sub[indicator_col].mean()
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        rows.append({by: g, "n": n, "rate": p, "ci_low": centre - half, "ci_high": centre + half})
    return pd.DataFrame(rows)


def differential_noise_test(df: pd.DataFrame, indicator_col: str, subgroup_col: str,
                            covariate_cols: list[str] | None = None, cluster_col: str | None = None) -> dict:
    """Logistic regression of a disagreement indicator on subgroup dummies (+ covariates).

    Returns adjusted odds ratios per subgroup level (reference = first level), their CIs and the
    likelihood-ratio p-value for the subgroup factor. Cluster-robust SEs when ``cluster_col`` is given.
    """
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[indicator_col]).copy()
    d[indicator_col] = d[indicator_col].astype(int)
    covs = covariate_cols or []
    rhs = " + ".join([f"C({subgroup_col})"] + covs) if covs else f"C({subgroup_col})"
    kwargs = {}
    if cluster_col is not None:
        kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": d[cluster_col].astype("category").cat.codes}}
    full = smf.logit(f"{indicator_col} ~ {rhs}", d).fit(disp=0, **kwargs)
    rhs0 = " + ".join(covs) if covs else "1"
    red = smf.logit(f"{indicator_col} ~ {rhs0}", d).fit(disp=0)
    from scipy.stats import chi2

    lr = 2 * (full.llf - red.llf)
    dof = full.df_model - red.df_model
    p_lr = float(chi2.sf(lr, dof)) if dof > 0 else float("nan")
    ors = {}
    ci = full.conf_int()
    for name in full.params.index:
        if name.startswith(f"C({subgroup_col})"):
            level = name.split("[T.")[1].rstrip("]")
            ors[level] = {"or": float(np.exp(full.params[name])), "ci_low": float(np.exp(ci.loc[name, 0])),
                          "ci_high": float(np.exp(ci.loc[name, 1])), "p": float(full.pvalues[name])}
    return {"odds_ratios": ors, "lr_stat": float(lr), "lr_dof": int(dof), "lr_p": p_lr, "n": int(len(d))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj
