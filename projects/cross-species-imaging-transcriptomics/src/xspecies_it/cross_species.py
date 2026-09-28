"""Cross-species replication statistics for imaging-transcriptomics associations.

An "association" is (brain map, gene set) in one species. Its *profile* is the vector of
per-gene correlations between the map and each gene's regional expression. Replication
across species is assessed on (i) the agreement of profiles over orthologs and (ii) the
gene-set statistic passing both species' spatial and gene-ensemble nulls (conjunction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .nulls import bh_fdr, gene_ensemble_null, spatial_null_corr


def align_orthologs(
    mouse_expr: pd.DataFrame,
    human_expr: pd.DataFrame,
    orthologs: pd.DataFrame,
    mouse_col: str = "Gene name",
    human_col: str = "Human gene name",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict two regions x genes matrices to one-to-one orthologs present in both.

    Returns the aligned mouse and human matrices (same column order, mouse symbols) and the
    ortholog table used.
    """
    o = orthologs[[mouse_col, human_col]].dropna().drop_duplicates()
    o = o[o[mouse_col].isin(mouse_expr.columns) & o[human_col].isin(human_expr.columns)]
    o = o[~o[mouse_col].duplicated(keep=False) & ~o[human_col].duplicated(keep=False)]
    m = mouse_expr[o[mouse_col].to_list()]
    h = human_expr[o[human_col].to_list()]
    h.columns = o[mouse_col].to_list()
    return m, h, o.reset_index(drop=True)


def association_profile(brain_map: pd.Series, region_expr: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """Per-gene correlation between a regional brain map and regional expression."""
    common = brain_map.index.intersection(region_expr.index)
    if len(common) < 5:
        raise ValueError("fewer than 5 shared regions")
    y = brain_map.loc[common].to_numpy(float)
    X = region_expr.loc[common].to_numpy(float)
    if method == "spearman":
        y = stats.rankdata(y)
        X = np.apply_along_axis(stats.rankdata, 0, X)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    den[den == 0] = np.nan
    return pd.Series(num / den, index=region_expr.columns, name="rho")


def profile_agreement(profile_a: pd.Series, profile_b: pd.Series, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Spearman agreement between two per-gene association profiles over shared genes, with bootstrap CI."""
    common = profile_a.index.intersection(profile_b.index)
    a = profile_a.loc[common].to_numpy(float)
    b = profile_b.loc[common].to_numpy(float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    rho = float(stats.spearmanr(a, b)[0])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        boots[i] = stats.spearmanr(a[idx], b[idx])[0]
    sign_conc = float(np.mean(np.sign(a) == np.sign(b)))
    return {"rho": rho, "ci_low": float(np.nanpercentile(boots, 2.5)), "ci_high": float(np.nanpercentile(boots, 97.5)),
            "sign_concordance": sign_conc, "n_genes": int(len(a))}


def attenuation_corrected(rho_xy: float, reliability_x: float, reliability_y: float = 1.0) -> float:
    """Spearman-Brown style disattenuation: rho / sqrt(rel_x * rel_y), clipped to [-1, 1]."""
    den = np.sqrt(max(reliability_x, 1e-9) * max(reliability_y, 1e-9))
    return float(np.clip(rho_xy / den, -1, 1))


@dataclass
class SpeciesAssociation:
    """One species' test of a (map, gene set) association."""

    r_set: float           # gene-set statistic (mean rho of set genes)
    p_spatial: float       # from surrogate maps
    p_ensemble: float      # from matched random gene sets
    z_ensemble: float


def test_association_in_species(
    brain_map: pd.Series,
    region_expr: pd.DataFrame,
    coords: pd.DataFrame,
    gene_set: Sequence[str],
    n_surr: int = 1000,
    n_perm: int = 2000,
    match_on: Optional[pd.Series] = None,
    seed: int = 0,
) -> SpeciesAssociation:
    """Gene-set score of a map in one species with spatial and ensemble p-values.

    The gene-set score is the mean per-gene Spearman rho over the set. The spatial null
    correlates surrogate maps with the set's mean expression profile (the "set map"); the
    ensemble null compares the score with matched random sets.
    """
    common = brain_map.index.intersection(region_expr.index).intersection(coords.index)
    bm = brain_map.loc[common]
    ex = region_expr.loc[common]
    genes_in = [g for g in gene_set if g in ex.columns]
    if len(genes_in) < 3:
        raise ValueError("gene set has fewer than 3 genes in the expression matrix")
    profile = association_profile(bm, ex)
    mask = profile.index.isin(genes_in)
    set_map = ex[genes_in].mean(axis=1)  # regions
    sp = spatial_null_corr(bm.to_numpy(float), set_map.to_numpy(float), coords.loc[common].to_numpy(float),
                           n_surr=n_surr, seed=seed)
    mo = None if match_on is None else match_on.reindex(profile.index).to_numpy(float)
    en = gene_ensemble_null(profile.fillna(0).to_numpy(), mask, match_on=mo, n_perm=n_perm, seed=seed)
    return SpeciesAssociation(r_set=float(profile[mask].mean()), p_spatial=sp["p_spatial"],
                              p_ensemble=en["p_ensemble"], z_ensemble=en["z"])


def conjunction_p(human: SpeciesAssociation, mouse: SpeciesAssociation, require_same_sign: bool = True) -> float:
    """Replication p-value: max over the four nulls (intersection-union test); 1 if signs disagree."""
    if require_same_sign and np.sign(human.r_set) != np.sign(mouse.r_set):
        return 1.0
    return float(max(human.p_spatial, human.p_ensemble, mouse.p_spatial, mouse.p_ensemble))


def replication_table(results: Dict[str, Tuple[SpeciesAssociation, SpeciesAssociation]], q: float = 0.05) -> pd.DataFrame:
    """Summarise a panel of associations: per-species stats, conjunction p, BH-FDR and replication flag."""
    rows = []
    for name, (h, m) in results.items():
        rows.append({"association": name, "r_human": h.r_set, "r_mouse": m.r_set,
                     "p_spatial_h": h.p_spatial, "p_ens_h": h.p_ensemble,
                     "p_spatial_m": m.p_spatial, "p_ens_m": m.p_ensemble,
                     "p_conjunction": conjunction_p(h, m)})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_conjunction"] = bh_fdr(df["p_conjunction"].to_numpy())
        df["replicates"] = df["q_conjunction"] < q
    return df


def replication_rate(table: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Fraction of associations that replicate, with a bootstrap CI over associations."""
    flags = table["replicates"].to_numpy(bool)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(flags, len(flags), replace=True).mean() for _ in range(n_boot)]
    return {"rate": float(flags.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(len(flags))}
