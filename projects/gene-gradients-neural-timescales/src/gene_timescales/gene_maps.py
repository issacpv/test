"""Regional gene-expression matrices and gene-gradient prediction of area-level targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut


# ------------------------------------------------------------------ expression matrices
def ish_area_matrix(ish_long: pd.DataFrame, structure_map: Dict[int, str], plane_preference: str = "coronal",
                    min_experiments: int = 1) -> pd.DataFrame:
    """Area x gene matrix of ISH expression energy.

    ``ish_long`` has columns gene, section_data_set_id, plane, structure_id, expression_energy
    (as written by scripts/download_data.py). ``structure_map`` maps CCF structure ids to
    area acronyms of interest; other structures are ignored. Coronal experiments are used
    when available, otherwise sagittal; experiments are averaged per gene.
    """
    df = ish_long[ish_long["structure_id"].isin(structure_map)].copy()
    df["area"] = df["structure_id"].map(structure_map)
    pref = df[df["plane"].str.lower() == plane_preference]
    genes_pref = set(pref["gene"])
    df = pd.concat([pref, df[~df["gene"].isin(genes_pref)]])
    n_exp = df.groupby("gene")["section_data_set_id"].nunique()
    df = df[df["gene"].isin(n_exp[n_exp >= min_experiments].index)]
    return df.pivot_table(index="area", columns="gene", values="expression_energy", aggfunc="mean")


def merfish_area_matrix(expr: np.ndarray, cell_meta: pd.DataFrame, genes: Sequence[str], area_col: str,
                        min_cells: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Area x gene mean expression and area x subclass proportions from MERFISH cells."""
    df = pd.DataFrame(np.asarray(expr, float), columns=list(genes))
    df["area"] = cell_meta[area_col].astype(str).to_numpy()
    n = df.groupby("area").size()
    keep = n[n >= min_cells].index
    mean_expr = df.groupby("area").mean().loc[keep]
    comp = pd.crosstab(cell_meta[area_col].astype(str), cell_meta["subclass"].astype(str)).loc[keep]
    comp = comp.div(comp.sum(axis=1), axis=0)
    return mean_expr, comp


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std(axis=0, ddof=1).replace(0, np.nan)
    return ((df - df.mean(axis=0)) / sd).dropna(axis=1)


# ------------------------------------------------------------------ prediction
@dataclass
class PredictionResult:
    r2_oos: float                 # out-of-sample R^2 (leave-one-out over areas)
    n_components: int             # chosen PLS components (or 0 for ridge)
    predictions: pd.Series        # out-of-sample predictions per area
    loadings: Optional[pd.Series] # gene weights (first component or ridge coefficients)
    alpha: Optional[float] = None


def _oos_r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pls_predict(X: pd.DataFrame, y: pd.Series, max_components: int = 5, inner_folds: int = 5, seed: int = 0) -> PredictionResult:
    """PLS regression with nested CV: outer leave-one-area-out, inner K-fold to pick components.

    Genes (columns of X) are standardised inside each training fold. The reported R^2 is
    fully out-of-sample; the loadings come from a final fit on all areas with the modal
    number of components chosen across outer folds.
    """
    common = X.index.intersection(y.index)
    Xa = X.loc[common].to_numpy(float)
    ya = y.loc[common].to_numpy(float)
    n = len(common)
    if n < 6:
        raise ValueError("need at least 6 areas")
    max_components = int(min(max_components, n - 2, Xa.shape[1]))
    preds = np.zeros(n)
    chosen = []
    for tr, te in LeaveOneOut().split(Xa):
        Xtr, ytr = Xa[tr], ya[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xtr_s = (Xtr - mu) / sd
        best_k, best_score = 1, -np.inf
        kf = KFold(n_splits=min(inner_folds, len(tr)), shuffle=True, random_state=seed)
        for k in range(1, max_components + 1):
            inner = np.zeros(len(tr))
            for itr, ite in kf.split(Xtr_s):
                m = PLSRegression(n_components=k, scale=False).fit(Xtr_s[itr], ytr[itr])
                inner[ite] = m.predict(Xtr_s[ite]).ravel()
            score = _oos_r2(ytr, inner)
            if score > best_score:
                best_k, best_score = k, score
        chosen.append(best_k)
        m = PLSRegression(n_components=best_k, scale=False).fit(Xtr_s, ytr)
        preds[te] = m.predict((Xa[te] - mu) / sd).ravel()
    k_final = int(np.bincount(chosen).argmax())
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
    m = PLSRegression(n_components=k_final, scale=False).fit((Xa - mu) / sd, ya)
    loadings = pd.Series(m.x_weights_[:, 0], index=X.columns, name="pls_w1")
    return PredictionResult(r2_oos=_oos_r2(ya, preds), n_components=k_final,
                            predictions=pd.Series(preds, index=common), loadings=loadings)


def ridge_predict(X: pd.DataFrame, y: pd.Series, alphas: Sequence[float] = (0.1, 1, 10, 100, 1000),
                  inner_folds: int = 5, seed: int = 0) -> PredictionResult:
    """Ridge regression with nested CV (outer leave-one-area-out, inner K-fold for alpha)."""
    common = X.index.intersection(y.index)
    Xa = X.loc[common].to_numpy(float)
    ya = y.loc[common].to_numpy(float)
    n = len(common)
    preds = np.zeros(n)
    chosen = []
    for tr, te in LeaveOneOut().split(Xa):
        Xtr, ytr = Xa[tr], ya[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xtr_s = (Xtr - mu) / sd
        kf = KFold(n_splits=min(inner_folds, len(tr)), shuffle=True, random_state=seed)
        best_a, best_score = alphas[0], -np.inf
        for a in alphas:
            inner = np.zeros(len(tr))
            for itr, ite in kf.split(Xtr_s):
                inner[ite] = Ridge(alpha=a).fit(Xtr_s[itr], ytr[itr]).predict(Xtr_s[ite])
            score = _oos_r2(ytr, inner)
            if score > best_score:
                best_a, best_score = a, score
        chosen.append(best_a)
        preds[te] = Ridge(alpha=best_a).fit(Xtr_s, ytr).predict((Xa[te] - mu) / sd)
    a_final = float(np.median(chosen))
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
    m = Ridge(alpha=a_final).fit((Xa - mu) / sd, ya)
    return PredictionResult(r2_oos=_oos_r2(ya, preds), n_components=0, predictions=pd.Series(preds, index=common),
                            loadings=pd.Series(m.coef_, index=X.columns, name="ridge_coef"), alpha=a_final)


def composition_partial(
    genes: pd.DataFrame,
    composition: pd.DataFrame,
    y: pd.Series,
    alphas: Sequence[float] = (0.1, 1, 10, 100),
) -> Dict[str, float]:
    """Sequential variance partition: composition first, genes on the residual.

    Returns out-of-sample R^2 of (a) genes alone, (b) composition alone, (c) genes fit to the
    composition residual (the "within-type" increment), and (d) composition + genes together.
    All four use leave-one-area-out ridge.
    """
    common = genes.index.intersection(composition.index).intersection(y.index)
    g = genes.loc[common]
    c = composition.loc[common]
    yy = y.loc[common]
    r_genes = ridge_predict(g, yy, alphas).r2_oos
    comp_res = ridge_predict(c, yy, alphas)
    r_comp = comp_res.r2_oos
    resid = yy - comp_res.predictions.loc[common]
    r_within = ridge_predict(g, resid, alphas).r2_oos
    both = pd.concat([c, g], axis=1)
    r_both = ridge_predict(both, yy, alphas).r2_oos
    return {"r2_genes": r_genes, "r2_composition": r_comp, "r2_genes_given_composition": r_within,
            "r2_both": r_both, "increment_over_composition": r_both - r_comp, "n_areas": int(len(common))}


def gene_family_score(loadings: pd.Series, family_prefixes: Sequence[str]) -> Tuple[float, np.ndarray]:
    """Mean absolute loading of genes whose symbol starts with any prefix, plus the membership mask."""
    mask = np.array([any(g.upper().startswith(p.upper()) for p in family_prefixes) for g in loadings.index])
    if mask.sum() == 0:
        return float("nan"), mask
    return float(np.abs(loadings.to_numpy())[mask].mean()), mask
