"""Landmark dataset construction, pooled-logistic hazard model, dynamic discrimination, lead times.

Landmark (super-)model: for each landmark quarter L and each pair still at risk at L (no prior
label event, >= ``min_reports`` cumulative reports), a row with features from quarters <= L and
outcome ``event within (L, L + horizon]``. Rows from all landmarks are stacked and a logistic
model with landmark-time terms is fitted (van Houwelingen & Putter, 2011). Time-dependent AUC at
horizon h is the AUC among rows with complete follow-up (event observed or L + h <= last quarter).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .faers_trajectories import landmark_features

FEATURE_COLS: List[str] = [
    "ic025", "ic", "ic_slope4", "ic_slope8", "consec_pos", "log_a", "recent_share4", "log_ror_lo",
    "drug_age_q", "serious_share", "hcp_share", "class_warning",
]


def build_landmark_dataset(tables: Dict[str, pd.DataFrame], events: pd.Series, landmarks: Iterable[int],
                           horizon: int, last_quarter: int, min_reports: int = 3,
                           pair_meta: Optional[pd.DataFrame] = None,
                           approval: Optional[pd.Series] = None) -> pd.DataFrame:
    """Stack landmark rows.

    Parameters
    ----------
    tables : ``{pair_id: cumulative_2x2 table}``.
    events : Series ``pair_id -> event quarter`` (NaN / missing = no event).
    landmarks : landmark quarters.
    horizon : prediction horizon in quarters.
    last_quarter : last quarter with FAERS data (for censoring flags).
    pair_meta : optional DataFrame indexed by pair_id with ``drug`` and ``class_warning`` columns.
    approval : optional Series ``pair_id -> approval quarter``.

    Returns
    -------
    DataFrame with feature columns, ``pair_id, drug, landmark, y, complete_followup, time_to_event``.
    """
    ev = pd.Series(events).astype(float)
    rows: List[Dict[str, float]] = []
    for pid, tab in tables.items():
        e = ev.get(pid, np.nan)
        drug = pair_meta.loc[pid, "drug"] if pair_meta is not None and pid in pair_meta.index else pid
        cw = int(pair_meta.loc[pid, "class_warning"]) if pair_meta is not None and "class_warning" in (pair_meta.columns if pair_meta is not None else []) and pid in pair_meta.index else 0
        appr = float(approval.get(pid, np.nan)) if approval is not None else None
        for L in landmarks:
            if not np.isnan(e) and e <= L:
                continue  # already labelled -> not at risk
            t = tab.loc[tab.index <= L]
            if t.empty or float(t["a"].iloc[-1]) < min_reports:
                continue
            f = landmark_features(tab, L, approval_quarter=None if appr is None or np.isnan(appr) else int(appr),
                                  class_warning=cw)
            y = int((not np.isnan(e)) and (L < e <= L + horizon))
            complete = bool(y == 1 or (L + horizon) <= last_quarter)
            f.update({"pair_id": pid, "drug": drug, "landmark": L, "y": y, "complete_followup": complete,
                      "time_to_event": (e - L) if not np.isnan(e) else np.nan})
            rows.append(f)
    df = pd.DataFrame(rows)
    if "drug_age_q" in df and df["drug_age_q"].isna().all():
        df["drug_age_q"] = 0.0
    return df


def _design(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    X = df[list(cols)].astype(float).fillna(0.0).values
    L = df["landmark"].astype(float).values
    Lc = (L - L.mean()) / (L.std() + 1e-9)
    return np.column_stack([X, Lc, Lc ** 2])


def fit_pooled_logistic(df: pd.DataFrame, cols: Sequence[str] = tuple(FEATURE_COLS), C: float = 1.0) -> Pipeline:
    """Penalised pooled logistic (discrete-time hazard) with quadratic landmark-time terms."""
    d = df[df["complete_followup"]]
    model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=C, max_iter=2000))])
    model.fit(_design(d, cols), d["y"].astype(int).values)
    return model


def predict_risk(model: Pipeline, df: pd.DataFrame, cols: Sequence[str] = tuple(FEATURE_COLS)) -> np.ndarray:
    return model.predict_proba(_design(df, cols))[:, 1]


def time_dependent_auc(df: pd.DataFrame, risk: np.ndarray, by_landmark: bool = False) -> Dict[str, float]:
    """AUC restricted to rows with complete follow-up (event observed or horizon fully observed)."""
    m = df["complete_followup"].values
    y = df["y"].values[m]
    r = np.asarray(risk)[m]
    out = {"auc": float(roc_auc_score(y, r)) if len(np.unique(y)) == 2 else np.nan, "n": int(m.sum()), "events": int(y.sum())}
    if by_landmark:
        per = {}
        for L, g in df[m].assign(risk=r).groupby("landmark"):
            per[int(L)] = float(roc_auc_score(g["y"], g["risk"])) if g["y"].nunique() == 2 else np.nan
        out["by_landmark"] = per
    return out


def grouped_cv_auc(df: pd.DataFrame, cols: Sequence[str] = tuple(FEATURE_COLS), n_splits: int = 5,
                   seed: int = 0, fit_fn: Callable[[pd.DataFrame], Pipeline] = None) -> Dict[str, object]:
    """Grouped-by-drug cross-validation; returns pooled out-of-fold AUC and per-fold AUCs."""
    rng = np.random.default_rng(seed)
    drugs = np.array(list(df["drug"].unique()), dtype=object)
    rng.shuffle(drugs)
    folds = np.array_split(drugs, n_splits)
    oof = np.full(len(df), np.nan)
    per_fold = []
    fit = fit_fn or (lambda d: fit_pooled_logistic(d, cols))
    for k in range(n_splits):
        test_mask = df["drug"].isin(folds[k]).values
        train = df[~test_mask]
        if train["y"].nunique() < 2 or test_mask.sum() == 0:
            continue
        model = fit(train)
        oof[test_mask] = predict_risk(model, df[test_mask], cols)
        per_fold.append(time_dependent_auc(df[test_mask], oof[test_mask])["auc"])
    ok = ~np.isnan(oof)
    pooled = time_dependent_auc(df[ok], oof[ok])
    return {"auc_pooled": pooled["auc"], "auc_folds": per_fold, "oof_risk": oof}


def temporal_split_auc(df: pd.DataFrame, cutoff_landmark: int, cols: Sequence[str] = tuple(FEATURE_COLS)) -> Dict[str, float]:
    """Train on landmarks <= cutoff, evaluate on later landmarks (H6)."""
    train, test = df[df["landmark"] <= cutoff_landmark], df[df["landmark"] > cutoff_landmark]
    model = fit_pooled_logistic(train, cols)
    return {"auc_train": time_dependent_auc(train, predict_risk(model, train, cols))["auc"],
            "auc_test": time_dependent_auc(test, predict_risk(model, test, cols))["auc"]}


def lead_times(tables: Dict[str, pd.DataFrame], events: pd.Series, consecutive: int = 2,
               threshold: float = 0.0) -> pd.DataFrame:
    """Quarters between the first *sustained* signal (IC025 > threshold for ``consecutive`` quarters)
    and the label event, per event pair. Negative lead = label change preceded the signal."""
    rows = []
    for pid, e in pd.Series(events).dropna().items():
        tab = tables.get(pid)
        if tab is None:
            continue
        pos = (tab["ic025"].values > threshold).astype(int)
        first = np.nan
        run = 0
        for q, v in zip(tab.index, pos):
            run = run + 1 if v else 0
            if run >= consecutive:
                first = q - consecutive + 1
                break
        rows.append({"pair_id": pid, "event_quarter": int(e), "first_signal_quarter": first,
                     "lead_quarters": (int(e) - first) if not np.isnan(first) else np.nan})
    return pd.DataFrame(rows)


def permutation_null_auc(df: pd.DataFrame, n_perm: int = 50, seed: int = 0,
                         cols: Sequence[str] = tuple(FEATURE_COLS)) -> np.ndarray:
    """Null AUC distribution: permute outcomes within drug (keeps drug base rates), refit, score in-sample."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        d = df.copy()
        d["y"] = d.groupby("drug")["y"].transform(lambda s: rng.permutation(s.values))
        d["complete_followup"] = True
        if d["y"].nunique() < 2:
            continue
        m = fit_pooled_logistic(d, cols)
        out.append(time_dependent_auc(d, predict_risk(m, d, cols))["auc"])
    return np.array(out)
