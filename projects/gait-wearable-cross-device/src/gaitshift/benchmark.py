"""Leave-one-dataset-out benchmark with subject-level bootstrap, in-distribution reference and shift diagnostics."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def default_model() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def _scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def subject_bootstrap(metric: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, s: np.ndarray, subjects: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Metric with percentile CI from resampling subjects with replacement."""
    y, s, subjects = np.asarray(y), np.asarray(s), np.asarray(subjects)
    uniq, inv = np.unique(subjects, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], s[idx]))
    vals = np.asarray(vals)
    point = metric(y, s) if len(np.unique(y)) == 2 else float("nan")
    if len(vals) == 0:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _bacc(y: np.ndarray, s: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, (s >= 0.5).astype(int)))


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def leave_one_dataset_out(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, datasets: np.ndarray, model=None, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Train on all datasets but one, test on the held-out dataset; one row per held-out dataset."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    subjects, datasets = np.asarray(subjects), np.asarray(datasets)
    base = model if model is not None else default_model()
    rows = []
    for d in np.unique(datasets):
        tr, te = datasets != d, datasets == d
        m = clone(base).fit(X[tr], y[tr])
        s = _scores(m, X[te])
        auc, auc_lo, auc_hi = subject_bootstrap(_auroc, y[te], s, subjects[te], n_boot, seed)
        bacc, b_lo, b_hi = subject_bootstrap(_bacc, y[te], s, subjects[te], n_boot, seed)
        rows.append({"held_out": d, "n_windows": int(te.sum()), "n_subjects": int(len(np.unique(subjects[te]))),
                     "auroc": auc, "auroc_lo": auc_lo, "auroc_hi": auc_hi, "bacc": bacc, "bacc_lo": b_lo, "bacc_hi": b_hi})
    return pd.DataFrame(rows)


def in_distribution_reference(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, datasets: np.ndarray, model=None, n_splits: int = 5) -> pd.DataFrame:
    """Within-dataset subject-grouped cross-validation (the in-distribution reference for the deployment gap)."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    subjects, datasets = np.asarray(subjects), np.asarray(datasets)
    base = model if model is not None else default_model()
    rows = []
    for d in np.unique(datasets):
        m = datasets == d
        n_groups = len(np.unique(subjects[m]))
        k = min(n_splits, n_groups)
        if k < 2 or len(np.unique(y[m])) < 2:
            rows.append({"dataset": d, "auroc": np.nan, "bacc": np.nan, "n_subjects": n_groups})
            continue
        cv = GroupKFold(n_splits=k)
        s = cross_val_predict(clone(base), X[m], y[m], groups=subjects[m], cv=cv, method="predict_proba")[:, 1]
        rows.append({"dataset": d, "auroc": _auroc(y[m], s), "bacc": _bacc(y[m], s), "n_subjects": n_groups})
    return pd.DataFrame(rows)


def deployment_gap(id_ref: pd.DataFrame, lodo: pd.DataFrame, metric: str = "bacc") -> pd.DataFrame:
    """Per-dataset in-distribution minus leave-one-dataset-out score."""
    a = id_ref.set_index("dataset")[metric]
    b = lodo.set_index("held_out")[metric]
    out = pd.DataFrame({"in_distribution": a, "lodo": b})
    out["gap"] = out["in_distribution"] - out["lodo"]
    return out.reset_index().rename(columns={"index": "dataset"})


def domain_shift_c2st(Xa: np.ndarray, Xb: np.ndarray, subjects_a: np.ndarray, subjects_b: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    """Classifier two-sample test: subject-grouped CV AUROC of telling dataset A from B (0.5 = no detectable shift)."""
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
    groups = np.concatenate([np.char.add("a_", np.asarray(subjects_a).astype(str)), np.char.add("b_", np.asarray(subjects_b).astype(str))])
    k = min(n_splits, len(np.unique(groups)))
    s = cross_val_predict(default_model(), X, y, groups=groups, cv=GroupKFold(n_splits=k), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, s))


def dataset_identity_null(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, datasets: np.ndarray, model=None, seed: int = 0) -> pd.DataFrame:
    """Shuffle labels within each dataset (preserving dataset identity) and re-run LODO: chance reference."""
    rng = np.random.default_rng(seed)
    y_perm = np.asarray(y).copy()
    subjects, datasets = np.asarray(subjects), np.asarray(datasets)
    for d in np.unique(datasets):
        m = datasets == d
        # permute at subject level so windows of one subject keep one label
        subj = np.unique(subjects[m])
        lab = {s: y_perm[m & (subjects == s)][0] for s in subj}
        vals = rng.permutation([lab[s] for s in subj])
        new = dict(zip(subj, vals))
        y_perm[m] = [new[s] for s in subjects[m]]
    return leave_one_dataset_out(X, y_perm, subjects, datasets, model=model, n_boot=50, seed=seed)
