"""Probes for demographic information in frozen ECG embeddings.

Provides linear and MLP probes plus MDL / online-code probing (Voita & Titov,
2020) and control-task selectivity (Hewitt & Liang, 2019), all with
patient-level cross-validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    attribute: str
    metric_name: str
    metric: float
    chance: float
    selectivity: float
    n: int


def _cv_indices(n: int, groups: np.ndarray | None, k: int, seed: int):
    if groups is None:
        groups = np.arange(n)
    gkf = GroupKFold(n_splits=min(k, len(np.unique(groups))))
    return list(gkf.split(np.zeros(n), groups=groups))


def linear_probe_classification(
    Z: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, k: int = 5, seed: int = 0
) -> float:
    """Cross-validated one-vs-rest macro-AUROC of a logistic probe."""
    Z = np.asarray(Z, float)
    y = np.asarray(y)
    aucs = []
    for tr, te in _cv_indices(len(y), groups, k, seed):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(Z[tr]), y[tr])
        if len(clf.classes_) == 2:
            p = clf.predict_proba(sc.transform(Z[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        else:
            p = clf.predict_proba(sc.transform(Z[te]))
            aucs.append(roc_auc_score(y[te], p, multi_class="ovr", average="macro"))
    return float(np.mean(aucs)) if aucs else float("nan")


def linear_probe_regression(
    Z: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, k: int = 5, seed: int = 0
) -> dict:
    """Cross-validated MAE and R^2 of a ridge probe (e.g., for age)."""
    Z, y = np.asarray(Z, float), np.asarray(y, float)
    maes, r2s = [], []
    for tr, te in _cv_indices(len(y), groups, k, seed):
        sc = StandardScaler().fit(Z[tr])
        reg = Ridge(alpha=1.0).fit(sc.transform(Z[tr]), y[tr])
        pred = reg.predict(sc.transform(Z[te]))
        maes.append(mean_absolute_error(y[te], pred))
        r2s.append(r2_score(y[te], pred))
    return {"mae": float(np.mean(maes)), "r2": float(np.mean(r2s))}


def probe_with_selectivity(
    Z: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, k: int = 5, seed: int = 0
) -> ProbeResult:
    """AUROC of a real probe minus AUROC on a random control task (selectivity)."""
    real = linear_probe_classification(Z, y, groups, k, seed)
    rng = np.random.default_rng(seed)
    y_ctrl = rng.permutation(y)
    ctrl = linear_probe_classification(Z, y_ctrl, groups, k, seed)
    return ProbeResult("attr", "auroc", real, chance=0.5, selectivity=real - ctrl, n=len(y))


def mdl_online_code(
    Z: np.ndarray, y: np.ndarray, portions=(0.1, 0.2, 0.4, 0.8, 1.0), seed: int = 0
) -> float:
    """Online (prequential) codelength in bits for the attribute given embeddings.

    Trains on an increasing prefix and scores the next block's negative
    log-likelihood; lower codelength = more easily/compactly decodable. A small
    self-contained approximation of Voita & Titov (2020).
    """
    Z, y = np.asarray(Z, float), np.asarray(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    Z, y = Z[perm], y[perm]
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    n = len(y)
    # first block coded with uniform prior
    bits = 0.0
    prev_end = int(portions[0] * n)
    bits += prev_end * np.log2(len(classes))
    for frac in portions[1:]:
        end = int(frac * n)
        if end <= prev_end:
            continue
        sc = StandardScaler().fit(Z[:prev_end])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Z[:prev_end]), y[:prev_end])
        proba = clf.predict_proba(sc.transform(Z[prev_end:end]))
        idx = {c: i for i, c in enumerate(clf.classes_)}
        for yi, row in zip(y[prev_end:end], proba):
            p = max(row[idx.get(yi, 0)], 1e-12)
            bits += -np.log2(p)
        prev_end = end
    return float(bits)


def layerwise_probe(
    layer_embeddings: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray | None = None,
    task: str = "classification", seed: int = 0
) -> dict[str, float]:
    """Run the appropriate probe at every layer; returns {layer: metric}."""
    out = {}
    for name, Z in layer_embeddings.items():
        if task == "classification":
            out[name] = linear_probe_classification(Z, y, groups, seed=seed)
        else:
            out[name] = linear_probe_regression(Z, y, groups, seed=seed)["mae"]
    return out
