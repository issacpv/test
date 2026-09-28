"""Allometric mixed-effects models with laboratory (archive) random effects.

The core model is the log-log wiring law

    log L = a + b log n + c log V + covariates + u_archive + e,   u_archive ~ N(0, s_lab^2)

fitted with ``statsmodels.MixedLM``. Helpers test exponents against theoretical values
(2/3, 1/3 for 3-D arbors), partition variance between biology and provenance factors, and
estimate how much of the naive between-species variance is absorbed by lab adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ScalingFit:
    """Container for a fitted mixed model."""

    params: pd.Series
    bse: pd.Series
    conf_int: pd.DataFrame
    lab_variance: float
    resid_variance: float
    n_obs: int
    n_groups: int
    formula: str
    converged: bool
    result: object = field(repr=False, default=None)

    @property
    def icc(self) -> float:
        """Fraction of residual-plus-lab variance attributable to labs."""
        tot = self.lab_variance + self.resid_variance
        return float(self.lab_variance / tot) if tot > 0 else float("nan")


def prepare_allometric_frame(df: pd.DataFrame, length_col: str = "total_length",
                             bp_col: str = "n_branch_points", vol_col: str = "hull_volume",
                             min_branch_points: int = 5) -> pd.DataFrame:
    """Add ``log_L``, ``log_n``, ``log_V`` and drop rows that cannot enter a log-log model."""
    out = df.copy()
    ok = (out[length_col] > 0) & (out[bp_col] >= min_branch_points) & (out[vol_col] > 0)
    out = out.loc[ok].copy()
    out["log_L"] = np.log(out[length_col].astype(float))
    out["log_n"] = np.log(out[bp_col].astype(float))
    out["log_V"] = np.log(out[vol_col].astype(float))
    return out


def _build_formula(response: str, terms: Sequence[str]) -> str:
    return f"{response} ~ " + " + ".join(terms) if terms else f"{response} ~ 1"


def fit_scaling_mixed_model(df: pd.DataFrame, response: str = "log_L",
                            fixed: Sequence[str] = ("log_n", "log_V"),
                            categorical: Sequence[str] = (),
                            group: str = "archive", reml: bool = True,
                            vc_formula: Optional[Dict[str, str]] = None) -> ScalingFit:
    """Fit ``response ~ fixed + C(categorical) + (1 | group)`` with statsmodels MixedLM.

    Parameters
    ----------
    fixed : numeric fixed-effect columns (e.g. ``log_n``, ``log_V``, ``effective_dimension``).
    categorical : columns wrapped in ``C()`` (e.g. ``species``, ``reconstruction_software``).
    group : random-intercept grouping column (the depositing lab / archive).
    vc_formula : optional variance components, e.g. ``{"software": "0 + C(reconstruction_software)"}``
        for software nested within archive.
    """
    import statsmodels.formula.api as smf

    terms = list(fixed) + [f"C({c})" for c in categorical]
    formula = _build_formula(response, terms)
    cols = [response, group] + list(fixed) + list(categorical)
    data = df.dropna(subset=cols).copy()
    data[group] = data[group].astype(str)
    model = smf.mixedlm(formula, data, groups=data[group], re_formula="1", vc_formula=vc_formula)
    try:
        result = model.fit(reml=reml, method=["lbfgs", "powell"], maxiter=500)
    except Exception:  # pragma: no cover - fall back to default optimiser
        result = model.fit(reml=reml)
    lab_var = float(np.asarray(result.cov_re).ravel()[0]) if result.cov_re.size else 0.0
    return ScalingFit(params=result.params, bse=result.bse, conf_int=result.conf_int(),
                      lab_variance=lab_var, resid_variance=float(result.scale), n_obs=int(result.nobs),
                      n_groups=int(data[group].nunique()), formula=formula,
                      converged=bool(getattr(result, "converged", True)), result=result)


def wald_test_exponent(fit: ScalingFit, term: str, h0: float) -> Dict[str, float]:
    """Wald test of a fixed-effect coefficient against a theoretical value ``h0``."""
    est = float(fit.params[term])
    se = float(fit.bse[term])
    z = (est - h0) / se if se > 0 else np.inf
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"estimate": est, "se": se, "h0": h0, "z": float(z), "p": float(p),
            "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se}


def equivalence_test(fit: ScalingFit, term: str, h0: float, margin: float = 0.05) -> Dict[str, float]:
    """Two one-sided tests (TOST): is the coefficient within ``h0 ± margin``?"""
    est = float(fit.params[term])
    se = float(fit.bse[term])
    t_low = (est - (h0 - margin)) / se
    t_high = ((h0 + margin) - est) / se
    p_low = 1 - stats.norm.cdf(t_low)
    p_high = 1 - stats.norm.cdf(t_high)
    return {"estimate": est, "margin": margin, "p_tost": float(max(p_low, p_high))}


# ---------------------------------------------------------------------- variance partition
def variance_partition(df: pd.DataFrame, feature: str, factors: Sequence[str],
                       numeric_covariates: Sequence[str] = ()) -> pd.DataFrame:
    """Semi-partial R² of each factor: R²(full) - R²(full without that factor) via OLS.

    Factors are treated as categorical. Returns a table with ``r2_full``, ``r2_drop`` and
    ``semi_partial_r2`` per factor plus the shared/unique decomposition summary row.
    """
    import statsmodels.formula.api as smf

    cols = [feature] + list(factors) + list(numeric_covariates)
    data = df.dropna(subset=cols).copy()
    for f in factors:
        data[f] = data[f].astype(str)

    def _r2(terms: Sequence[str]) -> float:
        formula = _build_formula(feature, terms)
        return float(smf.ols(formula, data).fit().rsquared)

    all_terms = [f"C({f})" for f in factors] + list(numeric_covariates)
    r2_full = _r2(all_terms)
    rows = []
    for f in factors:
        reduced = [t for t in all_terms if t != f"C({f})"]
        r2_drop = _r2(reduced)
        rows.append({"factor": f, "r2_full": r2_full, "r2_drop": r2_drop, "semi_partial_r2": r2_full - r2_drop,
                     "r2_alone": _r2([f"C({f})"])})
    return pd.DataFrame(rows)


def species_effect_attenuation(df: pd.DataFrame, feature: str, species_col: str = "species",
                               lab_col: str = "archive", covariates: Sequence[str] = (),
                               min_per_species: int = 20) -> Dict[str, object]:
    """How much of the naive between-species variance survives lab adjustment?

    Naive species effects come from ``feature ~ C(species) + covariates`` (OLS). Adjusted
    effects come from the same fixed effects plus a random intercept per lab. The
    attenuation is ``1 - Var(adjusted effects) / Var(naive effects)`` over species.
    Also reports the identifying structure: labs with >1 species and species with >1 lab.
    """
    import statsmodels.formula.api as smf

    cols = [feature, species_col, lab_col] + list(covariates)
    data = df.dropna(subset=cols).copy()
    counts = data[species_col].value_counts()
    keep = counts[counts >= min_per_species].index
    data = data[data[species_col].isin(keep)].copy()
    data[species_col] = data[species_col].astype(str)
    data[lab_col] = data[lab_col].astype(str)
    if data[species_col].nunique() < 2:
        raise ValueError("need at least two species with enough cells")

    terms = [f"C({species_col})"] + list(covariates)
    formula = _build_formula(feature, terms)
    naive = smf.ols(formula, data).fit()
    mixed = smf.mixedlm(formula, data, groups=data[lab_col]).fit(reml=True, method=["lbfgs", "powell"], maxiter=500)

    sp_terms = [k for k in naive.params.index if k.startswith(f"C({species_col})")]
    naive_eff = naive.params[sp_terms].to_numpy()
    adj_eff = mixed.params[sp_terms].to_numpy()
    # include the reference species as 0 so variance is over all species effects
    naive_all = np.concatenate([[0.0], naive_eff])
    adj_all = np.concatenate([[0.0], adj_eff])
    var_naive = float(np.var(naive_all, ddof=1))
    var_adj = float(np.var(adj_all, ddof=1))
    attenuation = 1.0 - var_adj / var_naive if var_naive > 0 else float("nan")

    labs_per_species = data.groupby(species_col)[lab_col].nunique()
    species_per_lab = data.groupby(lab_col)[species_col].nunique()
    return {
        "feature": feature,
        "n_obs": int(len(data)),
        "n_species": int(data[species_col].nunique()),
        "n_labs": int(data[lab_col].nunique()),
        "var_species_effects_naive": var_naive,
        "var_species_effects_adjusted": var_adj,
        "attenuation": float(attenuation),
        "lab_variance": float(np.asarray(mixed.cov_re).ravel()[0]),
        "resid_variance": float(mixed.scale),
        "n_labs_multispecies": int((species_per_lab > 1).sum()),
        "n_species_multilab": int((labs_per_species > 1).sum()),
        "naive_effects": pd.Series(naive_all, index=["<ref>"] + sp_terms),
        "adjusted_effects": pd.Series(adj_all, index=["<ref>"] + sp_terms),
    }


def permutation_null_attenuation(df: pd.DataFrame, feature: str, species_col: str = "species",
                                 lab_col: str = "archive", n_perm: int = 100, seed: int = 0,
                                 **kwargs) -> np.ndarray:
    """Null distribution of the attenuation statistic by permuting species labels across labs.

    Species labels are permuted *at the lab level* (each lab keeps one label but labels are
    shuffled between labs), preserving the nesting structure while destroying the
    biological species signal.
    """
    rng = np.random.default_rng(seed)
    data = df.copy()
    lab_species = data.groupby(lab_col)[species_col].agg(lambda s: s.mode().iloc[0])
    labs = lab_species.index.to_numpy()
    out = []
    for _ in range(n_perm):
        perm = rng.permutation(lab_species.to_numpy())
        mapping = dict(zip(labs, perm))
        data[species_col] = data[lab_col].map(mapping)
        try:
            out.append(species_effect_attenuation(data, feature, species_col, lab_col, **kwargs)["attenuation"])
        except (ValueError, np.linalg.LinAlgError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def cluster_bootstrap(df: pd.DataFrame, stat_fn, cluster_col: str = "archive", n_boot: int = 200,
                      seed: int = 0) -> np.ndarray:
    """Cluster (lab-level) bootstrap of an arbitrary statistic ``stat_fn(df) -> float``."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster_col)}
    keys = np.array(list(groups.keys()), dtype=object)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[k] for k in pick], ignore_index=True)
        try:
            out.append(float(stat_fn(sample)))
        except Exception:  # noqa: BLE001 - a resample can be singular
            out.append(np.nan)
    return np.asarray(out, dtype=float)


__all__ = [
    "ScalingFit", "prepare_allometric_frame", "fit_scaling_mixed_model", "wald_test_exponent", "equivalence_test",
    "variance_partition", "species_effect_attenuation", "permutation_null_attenuation", "cluster_bootstrap",
]
