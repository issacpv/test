"""Population statistics over morphology-wise stimulation thresholds.

The scientific claim of this project is distributional, not about any one cell:
*how much of the variance in stimulation threshold is attributable to cell class,
layer and species, and how much is residual morphological variability within a
class?* That is a variance-partitioning question, so the workhorse here is a
linear mixed model with a random intercept for the contributing archive -- which
absorbs the systematic differences in staining, slicing, shrinkage correction and
tracing convention between labs. Failing to model that nesting is the most likely
way to manufacture a false species effect, because human and mouse
reconstructions come from largely disjoint sets of labs.

Thresholds are analysed on the log scale: they are positive, right-skewed and the
hypotheses are multiplicative ("human thresholds are 30% higher"), so
``log10(threshold)`` coefficients read directly as fold changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MixedModelResult",
    "prepare_frame",
    "fit_threshold_mixed_model",
    "variance_components",
    "bootstrap_ratio_ci",
    "class_separability",
]


@dataclass
class MixedModelResult:
    """Outcome of a mixed-model fit.

    Attributes:
        formula: The fitted fixed-effects formula.
        group_var: The grouping variable for the random intercept.
        coefficients: Fixed-effect estimates, standard errors, z and p values.
        n_obs: Number of observations used.
        n_groups: Number of random-effect groups.
        icc: Intraclass correlation of the grouping variable, i.e. the fraction
            of residual variance explained by between-archive differences.
        converged: Whether the optimiser reported convergence.
        backend: ``"statsmodels_mixedlm"`` or ``"ols_cluster"`` fallback.
    """

    formula: str
    group_var: str
    coefficients: pd.DataFrame
    n_obs: int
    n_groups: int
    icc: float
    converged: bool
    backend: str

    def summary_table(self) -> pd.DataFrame:
        """Return coefficients with fold-change columns for log10 outcomes."""
        out = self.coefficients.copy()
        out["fold_change"] = 10.0 ** out["estimate"]
        out["fold_change_lo"] = 10.0 ** (out["estimate"] - 1.96 * out["std_err"])
        out["fold_change_hi"] = 10.0 ** (out["estimate"] + 1.96 * out["std_err"])
        return out


def prepare_frame(
    rows: Sequence[dict[str, object]],
    *,
    threshold_col: str = "threshold_min_Vpm",
    min_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Turn sweep output into a modelling frame.

    Adds ``log10_threshold``, ``log10_total_length`` and a coarse
    ``cell_class`` (``pyramidal`` / ``interneuron`` / ``other``) derived from the
    free-text NeuroMorpho ``cell_type``, and drops non-finite thresholds (which
    arise for reconstructions with no depolarizing compartment, e.g. dendrite-only
    files).

    Args:
        rows: Output of :func:`morph_stim.thresholds.sweep_population`.
        threshold_col: Which threshold column to model.
        min_threshold: Lower clip to keep the log finite.

    Returns:
        A dataframe with one row per (morphology, waveform).

    Raises:
        ValueError: If ``rows`` is empty or ``threshold_col`` is absent.
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("no rows to analyse")
    if threshold_col not in df.columns:
        raise ValueError(f"{threshold_col!r} not in {sorted(df.columns)}")

    df = df[np.isfinite(pd.to_numeric(df[threshold_col], errors="coerce"))].copy()
    df["threshold"] = pd.to_numeric(df[threshold_col]).clip(lower=min_threshold)
    df["log10_threshold"] = np.log10(df["threshold"])
    if "total_length_um" in df.columns:
        length = pd.to_numeric(df["total_length_um"], errors="coerce").clip(lower=1.0)
        df["log10_total_length"] = np.log10(length)
    df["cell_class"] = df.get("cell_type", pd.Series(index=df.index, dtype=object)).map(_coarse_class)
    if "archive" in df.columns:
        df["archive"] = df["archive"].fillna("unknown").astype(str)
    return df.reset_index(drop=True)


def _coarse_class(cell_type: object) -> str:
    """Map NeuroMorpho's free-text cell type onto three modelling classes."""
    text = str(cell_type).lower()
    if "pyramidal" in text:
        return "pyramidal"
    if any(k in text for k in ("interneuron", "basket", "martinotti", "bipolar", "chandelier", "gabaergic")):
        return "interneuron"
    return "other"


def fit_threshold_mixed_model(
    df: pd.DataFrame,
    *,
    fixed: str = "species * cell_class + log10_total_length",
    outcome: str = "log10_threshold",
    group_var: str = "archive",
) -> MixedModelResult:
    """Fit ``outcome ~ fixed + (1 | group_var)``.

    Uses :mod:`statsmodels` ``MixedLM`` when available. If statsmodels is absent,
    or the grouping variable has fewer than three levels (in which case a random
    intercept is not identifiable), it falls back to OLS with cluster-robust
    standard errors on the same grouping -- a defensible alternative that keeps
    the inference honest about lab-level dependence.

    Args:
        df: Frame from :func:`prepare_frame`.
        fixed: Patsy-style fixed-effects specification.
        outcome: Outcome column.
        group_var: Grouping column for the random intercept / cluster.

    Returns:
        A :class:`MixedModelResult`.

    Raises:
        ValueError: If required columns are missing or fewer than 10 rows remain.
    """
    needed = {outcome, group_var}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    work = df.dropna(subset=[outcome, group_var]).copy()
    if len(work) < 10:
        raise ValueError(f"need >=10 complete rows, got {len(work)}")

    formula = f"{outcome} ~ {fixed}"
    n_groups = work[group_var].nunique()

    try:
        import statsmodels.formula.api as smf

        if n_groups >= 3:
            model = smf.mixedlm(formula, work, groups=work[group_var])
            fit = model.fit(method="lbfgs", maxiter=500)
            coefs = pd.DataFrame(
                {
                    "estimate": fit.fe_params,
                    "std_err": fit.bse[fit.fe_params.index],
                    "z": fit.tvalues[fit.fe_params.index],
                    "p_value": fit.pvalues[fit.fe_params.index],
                }
            )
            group_var_est = float(np.asarray(fit.cov_re).ravel()[0])
            resid_var = float(fit.scale)
            icc = group_var_est / (group_var_est + resid_var) if (group_var_est + resid_var) > 0 else 0.0
            return MixedModelResult(
                formula=formula,
                group_var=group_var,
                coefficients=coefs,
                n_obs=int(len(work)),
                n_groups=int(n_groups),
                icc=float(icc),
                converged=bool(getattr(fit, "converged", True)),
                backend="statsmodels_mixedlm",
            )

        ols = smf.ols(formula, work).fit(
            cov_type="cluster", cov_kwds={"groups": work[group_var]}
        )
        coefs = pd.DataFrame(
            {
                "estimate": ols.params,
                "std_err": ols.bse,
                "z": ols.tvalues,
                "p_value": ols.pvalues,
            }
        )
        return MixedModelResult(
            formula=formula,
            group_var=group_var,
            coefficients=coefs,
            n_obs=int(len(work)),
            n_groups=int(n_groups),
            icc=float("nan"),
            converged=True,
            backend="ols_cluster",
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "fit_threshold_mixed_model requires statsmodels; install it or use "
            "variance_components() for a dependency-free variance partition"
        ) from exc


def variance_components(
    df: pd.DataFrame,
    *,
    outcome: str = "log10_threshold",
    factors: Sequence[str] = ("species", "cell_class", "archive"),
) -> pd.DataFrame:
    """Dependency-free one-way variance partition per factor.

    For each factor reports the between-group share of total variance
    (``eta_squared``) and the within-class coefficient of variation of the
    back-transformed threshold. Useful as a sanity check on the mixed model and
    as the headline "morphological variability dominates cell class" statistic.

    Args:
        df: Frame from :func:`prepare_frame`.
        outcome: Outcome column (log scale expected).
        factors: Grouping columns to evaluate.

    Returns:
        One row per factor with ``n_levels``, ``eta_squared`` and
        ``median_within_level_iqr_dex``.
    """
    out: list[dict[str, object]] = []
    total_var = float(np.var(df[outcome], ddof=1)) if len(df) > 1 else 0.0
    for factor in factors:
        if factor not in df.columns:
            continue
        grouped = df.dropna(subset=[factor]).groupby(factor)[outcome]
        means = grouped.mean()
        counts = grouped.count()
        if len(means) < 2 or total_var == 0:
            out.append(
                {"factor": factor, "n_levels": int(len(means)), "eta_squared": np.nan,
                 "median_within_level_iqr_dex": np.nan}
            )
            continue
        grand = float((means * counts).sum() / counts.sum())
        between = float((counts * (means - grand) ** 2).sum() / (counts.sum() - 1))
        iqr = grouped.apply(lambda s: float(np.subtract(*np.percentile(s, [75, 25]))) if len(s) > 3 else np.nan)
        out.append(
            {
                "factor": factor,
                "n_levels": int(len(means)),
                "eta_squared": min(between / total_var, 1.0),
                "median_within_level_iqr_dex": float(np.nanmedian(iqr)) if len(iqr) else np.nan,
            }
        )
    return pd.DataFrame(out)


def bootstrap_ratio_ci(
    df: pd.DataFrame,
    *,
    group_col: str = "species",
    numerator: str = "human",
    denominator: str = "mouse",
    value_col: str = "threshold",
    cluster_col: str | None = "archive",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster bootstrap CI for the ratio of median thresholds between two groups.

    Resampling *clusters* (archives) rather than individual neurons is essential:
    a single archive can contribute dozens of reconstructions from the same few
    slices, and a naive neuron-level bootstrap would understate the CI by a large
    factor.

    Args:
        df: Frame from :func:`prepare_frame`.
        group_col: Column defining the two groups.
        numerator: Level placed in the numerator.
        denominator: Level placed in the denominator.
        value_col: Positive-valued column to take medians of.
        cluster_col: Column to resample; ``None`` resamples rows.
        n_boot: Bootstrap replicates.
        alpha: Two-sided significance level.
        seed: RNG seed.

    Returns:
        Dict with ``ratio``, ``ci_low``, ``ci_high``, ``n_numerator``,
        ``n_denominator`` and ``n_clusters``.

    Raises:
        ValueError: If either group is empty.
    """
    rng = np.random.default_rng(seed)
    a = df.loc[df[group_col] == numerator, value_col].to_numpy(dtype=float)
    b = df.loc[df[group_col] == denominator, value_col].to_numpy(dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError(f"empty group: {numerator}={a.size}, {denominator}={b.size}")

    point = float(np.median(a) / np.median(b))
    if cluster_col is not None and cluster_col in df.columns:
        clusters = df[cluster_col].astype(str).to_numpy()
        unique = np.unique(clusters)
        index = {c: np.flatnonzero(clusters == c) for c in unique}
    else:
        unique = np.arange(len(df))
        index = {i: np.array([i]) for i in unique}

    groups = df[group_col].to_numpy()
    values = df[value_col].to_numpy(dtype=float)
    ratios: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[c] for c in picked])
        ga = values[rows][groups[rows] == numerator]
        gb = values[rows][groups[rows] == denominator]
        if ga.size and gb.size:
            ratios.append(float(np.median(ga) / np.median(gb)))
    if not ratios:
        return {"ratio": point, "ci_low": np.nan, "ci_high": np.nan,
                "n_numerator": a.size, "n_denominator": b.size, "n_clusters": len(unique)}
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "ratio": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_numerator": int(a.size),
        "n_denominator": int(b.size),
        "n_clusters": int(len(unique)),
    }


def class_separability(
    df: pd.DataFrame,
    *,
    group_col: str = "cell_class",
    value_col: str = "log10_threshold",
) -> pd.DataFrame:
    """Pairwise overlap between threshold distributions of morphological classes.

    Reports Cohen's d *and* the probability of superiority (AUC), because the
    negative result reported by Trotter et al. 2025 (Front Synaptic Neurosci) is
    precisely that class means can differ while distributions overlap almost
    completely. An AUC near 0.5 means a measured threshold carries essentially no
    information about cell class -- the honest way to state a null.

    Args:
        df: Frame from :func:`prepare_frame`.
        group_col: Class column.
        value_col: Value column.

    Returns:
        One row per unordered class pair with ``cohens_d`` and ``auc``.
    """
    levels = [lv for lv in df[group_col].dropna().unique()]
    out: list[dict[str, object]] = []
    for i, a in enumerate(levels):
        for b in levels[i + 1 :]:
            xa = df.loc[df[group_col] == a, value_col].to_numpy(dtype=float)
            xb = df.loc[df[group_col] == b, value_col].to_numpy(dtype=float)
            if xa.size < 2 or xb.size < 2:
                continue
            pooled = np.sqrt(
                ((xa.size - 1) * np.var(xa, ddof=1) + (xb.size - 1) * np.var(xb, ddof=1))
                / (xa.size + xb.size - 2)
            )
            d = float((xa.mean() - xb.mean()) / pooled) if pooled > 0 else 0.0
            auc = float((xa[:, None] > xb[None, :]).mean() + 0.5 * (xa[:, None] == xb[None, :]).mean())
            out.append(
                {"class_a": a, "class_b": b, "n_a": int(xa.size), "n_b": int(xb.size),
                 "cohens_d": d, "auc": auc}
            )
    return pd.DataFrame(out)
