"""Regional regression of MRI contrast on cell-type densities, with importance and model comparison."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str = "ridge", alphas: Sequence[float] = (0.01, 0.1, 1, 10, 100), n_components: int = 3
               ) -> Pipeline:
    if kind == "ridge":
        est = RidgeCV(alphas=alphas)
    elif kind == "pls":
        est = PLSRegression(n_components=n_components)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("model", est)])


@dataclass
class FitResult:
    r2_cv: float
    r2_in: float
    coef: pd.Series
    n: int
    kind: str


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_cv(X: pd.DataFrame, y: pd.Series, kind: str = "ridge", cv: str = "loo", groups: Optional[pd.Series] = None,
           seed: int = 0) -> FitResult:
    """Cross-validated R^2 with leave-one-region-out ('loo'), 5-fold, or leave-one-group-out ('group').

    ``groups`` (e.g. major brain division per region) is required for 'group'.
    """
    Xv = X.to_numpy(float)
    yv = y.to_numpy(float)
    n = len(yv)
    pred = np.empty(n)
    if cv == "loo":
        splits = LeaveOneOut().split(Xv)
    elif cv == "kfold":
        splits = KFold(5, shuffle=True, random_state=seed).split(Xv)
    elif cv == "group":
        if groups is None:
            raise ValueError("groups needed for cv='group'")
        g = groups.to_numpy()
        splits = ((np.where(g != u)[0], np.where(g == u)[0]) for u in np.unique(g))
    else:
        raise ValueError(cv)
    for tr, te in splits:
        m = make_model(kind).fit(Xv[tr], yv[tr])
        pred[te] = np.ravel(m.predict(Xv[te]))
    full = make_model(kind).fit(Xv, yv)
    coef = np.ravel(full.named_steps["model"].coef_)
    return FitResult(_r2(yv, pred), _r2(yv, np.ravel(full.predict(Xv))), pd.Series(coef, index=X.columns), n, kind)


def relative_importance(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0, max_predictors: int = 12) -> pd.Series:
    """LMG / Shapley decomposition of in-sample R^2 over predictors (ridge with fixed alpha).

    Exact enumeration over subsets: limited to ``max_predictors`` columns (use class-level
    densities, or group subclasses into classes first).
    """
    cols = list(X.columns)
    if len(cols) > max_predictors:
        raise ValueError(f"too many predictors ({len(cols)}) for exact LMG; aggregate first")
    Xs = StandardScaler().fit_transform(X.to_numpy(float))
    yv = y.to_numpy(float)
    p = len(cols)

    def r2_subset(subset):
        if not subset:
            return 0.0
        m = Ridge(alpha=alpha).fit(Xs[:, list(subset)], yv)
        return _r2(yv, m.predict(Xs[:, list(subset)]))

    cache: Dict[tuple, float] = {}
    from math import factorial

    imp = np.zeros(p)
    for k in range(p + 1):
        for S in combinations(range(p), k):
            cache[S] = r2_subset(S)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for k in range(p):
            w = factorial(k) * factorial(p - k - 1) / factorial(p)
            for S in combinations(others, k):
                imp[j] += w * (cache[tuple(sorted(S + (j,)))] - cache[S])
    return pd.Series(imp, index=cols)


def nested_comparison(X_base: pd.DataFrame, X_extra: pd.DataFrame, y: pd.Series, kind: str = "ridge", cv: str = "loo",
                      **kw) -> Dict[str, float]:
    """CV R^2 of base model vs. base + extra predictors (e.g. densities vs. densities + gene programs)."""
    base = fit_cv(X_base, y, kind, cv, **kw)
    full = fit_cv(pd.concat([X_base, X_extra], axis=1), y, kind, cv, **kw)
    return {"r2_base": base.r2_cv, "r2_full": full.r2_cv, "delta_r2": full.r2_cv - base.r2_cv}


def bootstrap_importance(X: pd.DataFrame, y: pd.Series, n_boot: int = 200, seed: int = 0, **kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        rows.append(relative_importance(X.iloc[idx], y.iloc[idx], **kw))
    B = pd.DataFrame(rows)
    return pd.DataFrame({"mean": B.mean(), "ci_low": B.quantile(0.025), "ci_high": B.quantile(0.975)})


def cross_atlas_concordance(imp_a: pd.Series, imp_b: pd.Series) -> Dict[str, float]:
    """Kendall's tau of importance rankings and Lin's concordance of coefficients across atlases."""
    idx = imp_a.index.intersection(imp_b.index)
    a, b = imp_a.loc[idx].to_numpy(float), imp_b.loc[idx].to_numpy(float)
    tau = stats.kendalltau(a, b)[0] if len(idx) > 2 else float("nan")
    ccc = 2 * np.cov(a, b)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2) if len(idx) > 2 else float("nan")
    return {"kendall_tau": float(tau), "lin_ccc": float(ccc), "n": int(len(idx))}
