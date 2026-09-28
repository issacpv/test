"""Domain-shift diagnostics between ECG sources.

Three complementary views of "how different" two sources are:

1. **Discriminability** - AUROC of a classifier that tries to tell source A from
   source B (cross-validated).  0.5 = indistinguishable; the proxy A-distance
   is ``2 * (2*AUC - 1)``.  Applied to handcrafted features it captures
   acquisition + population; applied to demographics only it isolates
   population.
2. **MMD** - kernel maximum mean discrepancy with a median-heuristic RBF
   kernel, with a permutation p-value.
3. **Acquisition fingerprints** - lead-wise noise floor and spectral shape
   (power above 40 Hz, mains-hum ratio at 50/60 Hz), which separate device /
   filtering differences from physiological ones.

Also provides density-ratio weights (target -> source) from a logistic model on
age/sex so that a target set can be reweighted to the source's demographics.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def domain_classifier_auc(Xa: np.ndarray, Xb: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    """Cross-validated AUROC of a gradient-boosting classifier separating A (0) from B (1)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def proxy_a_distance(auc: float) -> float:
    """Proxy A-distance from a domain-classifier AUROC (Ben-David et al. style)."""
    return float(2.0 * (2.0 * max(auc, 1 - auc) - 1.0))


def _rbf_gram(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def mmd_rbf(Xa: np.ndarray, Xb: np.ndarray, gamma: float | None = None, n_perm: int = 200, seed: int = 0) -> tuple[float, float]:
    """Unbiased MMD^2 with an RBF kernel (median heuristic if ``gamma`` is None) and permutation p-value."""
    rng = np.random.default_rng(seed)
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    Z = np.vstack([Xa, Xb])
    if gamma is None:
        sub = Z[rng.choice(len(Z), size=min(500, len(Z)), replace=False)]
        d2 = (sub ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None, :] - 2 * sub @ sub.T
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / med

    def stat(A: np.ndarray, B: np.ndarray) -> float:
        m, n = len(A), len(B)
        Kaa, Kbb, Kab = _rbf_gram(A, A, gamma), _rbf_gram(B, B, gamma), _rbf_gram(A, B, gamma)
        t1 = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1)) if m > 1 else 0.0
        t2 = (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1)) if n > 1 else 0.0
        return float(t1 + t2 - 2 * Kab.mean())

    obs = stat(Xa, Xb)
    m = len(Xa)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(Z))
        null[i] = stat(Z[perm[:m]], Z[perm[m:]])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


def acquisition_fingerprint(X: np.ndarray, fs: int = 500) -> dict[str, np.ndarray]:
    """Per-record device/filter fingerprints for a batch ``X`` of shape (n, 12, samples).

    Returns arrays (n, 12) for: ``noise_floor`` (median absolute first difference),
    ``hf_ratio`` (power > 40 Hz / total), ``mains_ratio`` (power at 49-51 + 59-61 Hz / total),
    ``lf_ratio`` (power < 0.67 Hz / total; baseline-wander / high-pass setting), and ``amp_p99``.
    """
    X = np.asarray(X, dtype=np.float64)
    n, L, _ = X.shape
    out = {k: np.zeros((n, L)) for k in ("noise_floor", "hf_ratio", "mains_ratio", "lf_ratio", "amp_p99")}
    for i in range(n):
        for l in range(L):
            x = X[i, l]
            f, p = sps.welch(x, fs=fs, nperseg=min(1024, x.size))
            tot = np.trapezoid(p, f) + 1e-12
            out["noise_floor"][i, l] = np.median(np.abs(np.diff(x)))
            out["hf_ratio"][i, l] = np.trapezoid(p[f > 40], f[f > 40]) / tot
            mains = ((f >= 49) & (f <= 51)) | ((f >= 59) & (f <= 61))
            out["mains_ratio"][i, l] = np.trapezoid(p[mains], f[mains]) / tot if mains.any() else 0.0
            out["lf_ratio"][i, l] = np.trapezoid(p[f < 0.67], f[f < 0.67]) / tot
            out["amp_p99"][i, l] = np.percentile(np.abs(x), 99)
    return out


def fingerprint_summary(fp: dict[str, np.ndarray]) -> dict[str, float]:
    """Median over records and leads of each fingerprint - one number per source for the data card."""
    return {k: float(np.median(v)) for k, v in fp.items()}


def density_ratio_weights(D_source: np.ndarray, D_target: np.ndarray, clip: float = 20.0, seed: int = 0) -> np.ndarray:
    """Weights w(x) ~ p_source(x)/p_target(x) for target records, from a logistic model on demographics.

    ``D_*`` are (n, d) arrays (e.g. columns age, sex01).  Reweighting the target
    by ``w`` emulates evaluating on a target with the source's demographic mix -
    the "population" counterfactual in the decomposition experiment.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([D_source, D_target]).astype(float)
    y = np.r_[np.ones(len(D_source)), np.zeros(len(D_target))]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000, random_state=seed).fit(sc.transform(X), y)
    p = lr.predict_proba(sc.transform(D_target.astype(float)))[:, 1]
    prior = len(D_source) / len(D_target)
    w = (p / np.clip(1 - p, 1e-6, None)) / prior
    w = np.clip(w, 1.0 / clip, clip)
    return w / w.mean()


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size of a weight vector (sanity check for reweighted evaluation)."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))
