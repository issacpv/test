"""Cross-validated decoders with nested penalty selection, null models (trial shuffle, pseudo-session,
imposter session), latent (factor-analysis) features and unit-yield curves.

Scores: balanced accuracy for binary targets, R² for continuous targets. Cross-validation uses
*contiguous* trial blocks by default because block prior and slow drift make trials non-exchangeable
(Harris 2020).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from .loaders import generate_block_prior


@dataclass
class DecodeResult:
    score: float
    fold_scores: np.ndarray
    alpha: float
    task: str
    n_trials: int
    n_features: int

    def as_dict(self) -> Dict[str, float]:
        return {"score": self.score, "score_sd_folds": float(self.fold_scores.std()), "alpha": self.alpha,
                "n_trials": self.n_trials, "n_features": self.n_features}


def contiguous_folds(n: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous (block) K-fold indices over trial order."""
    idx = np.arange(n)
    parts = np.array_split(idx, n_splits)
    return [(np.setdiff1d(idx, te), te) for te in parts]


def random_folds(n: int, n_splits: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    perm = rng.permutation(n)
    parts = np.array_split(perm, n_splits)
    return [(np.setdiff1d(perm, te), te) for te in parts]


def _fit_predict(Xtr, ytr, Xte, task: str, alpha: float):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if task == "classification":
        clf = LogisticRegression(C=1.0 / alpha, max_iter=2000)
        clf.fit(Xtr, ytr)
        return clf.predict(Xte)
    reg = Ridge(alpha=alpha).fit(Xtr, ytr)
    return reg.predict(Xte)


def _score(y, yhat, task: str) -> float:
    return balanced_accuracy_score(y, yhat) if task == "classification" else r2_score(y, yhat)


def decode_cv(X: np.ndarray, y: np.ndarray, task: str = "classification", n_splits: int = 5,
              alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0), contiguous: bool = True,
              inner_splits: int = 3, seed: int = 0) -> DecodeResult:
    """Nested CV: penalty chosen on inner folds of each training set; score on the outer test fold.

    Binary targets must be coded 0/1; continuous targets are z-scored implicitly by Ridge on standardized X.
    Degenerate folds (one class only) fall back to the majority-class prediction, scoring 0.5.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if task == "classification":
        y = y.astype(int)
    folds = contiguous_folds(len(y), n_splits) if contiguous else random_folds(len(y), n_splits, rng)
    fold_scores, chosen = [], []
    for tr, te in folds:
        if task == "classification" and len(np.unique(y[tr])) < 2:
            fold_scores.append(0.5)
            chosen.append(np.nan)
            continue
        # inner selection
        best_a, best_s = alphas[0], -np.inf
        if len(alphas) > 1:
            inner = contiguous_folds(len(tr), inner_splits) if contiguous else random_folds(len(tr), inner_splits, rng)
            for a in alphas:
                s = []
                for itr, ite in inner:
                    if task == "classification" and len(np.unique(y[tr][itr])) < 2:
                        s.append(0.5)
                        continue
                    yhat = _fit_predict(X[tr][itr], y[tr][itr], X[tr][ite], task, a)
                    s.append(_score(y[tr][ite], yhat, task))
                if np.mean(s) > best_s:
                    best_s, best_a = np.mean(s), a
        yhat = _fit_predict(X[tr], y[tr], X[te], task, best_a)
        fold_scores.append(_score(y[te], yhat, task))
        chosen.append(best_a)
    fs = np.asarray(fold_scores, float)
    return DecodeResult(score=float(fs.mean()), fold_scores=fs, alpha=float(np.nanmedian(chosen)),
                        task=task, n_trials=len(y), n_features=X.shape[1])


# ----------------------------------------------------------------------------- nulls
def shuffle_null(X: np.ndarray, y: np.ndarray, n: int = 200, seed: int = 0, **kw) -> np.ndarray:
    """Trial-label permutation null (exchangeability assumed; fine for stimulus side within block)."""
    rng = np.random.default_rng(seed)
    return np.array([decode_cv(X, rng.permutation(y), seed=seed + i, **kw).score for i in range(n)])


def pseudo_block_labels(n_trials: int, rng: np.random.Generator, binary: bool = True) -> np.ndarray:
    """Pseudo-session block prior drawn from the IBL generative process (Harris 2020 / Findling 2025)."""
    p = generate_block_prior(n_trials, rng)
    if binary:
        return np.where(p == 0.5, np.nan, (p > 0.5).astype(float))
    return p


def pseudo_stimulus_labels(n_trials: int, rng: np.random.Generator, p_left: Optional[np.ndarray] = None) -> np.ndarray:
    """Pseudo stimulus side sampled from the (pseudo) block prior: 1 = left."""
    p = generate_block_prior(n_trials, rng) if p_left is None else p_left
    return (rng.random(n_trials) < p).astype(float)


def pseudo_session_null(X: np.ndarray, label_fn: Callable[[np.random.Generator], np.ndarray], n: int = 200,
                        seed: int = 0, **kw) -> np.ndarray:
    """Decode synthetic labels generated by `label_fn(rng)` (pseudo-session or imposter session).

    Trials where the generated label is NaN are dropped for that pseudo-session.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        yp = label_fn(rng)
        m = ~np.isnan(yp)
        out[i] = decode_cv(X[m], yp[m], seed=seed + i, **kw).score
    return out


def imposter_label_fn(other_sessions_labels: Sequence[np.ndarray], n_trials: int) -> Callable[[np.random.Generator], np.ndarray]:
    """Labels borrowed from other sessions' behaviour (imposter sessions), concatenated to length n_trials."""
    pool = [np.asarray(l, float) for l in other_sessions_labels if len(l) > 0]

    def fn(rng: np.random.Generator) -> np.ndarray:
        parts, total = [], 0
        while total < n_trials:
            l = pool[rng.integers(len(pool))]
            start = rng.integers(0, max(1, len(l) - 10))
            seg = l[start:]
            parts.append(seg)
            total += len(seg)
        return np.concatenate(parts)[:n_trials]
    return fn


def null_corrected(score: float, null: np.ndarray) -> Dict[str, float]:
    """Observed score relative to a null distribution: corrected score, z, one-sided MC p."""
    null = np.asarray(null, float)
    p = (1 + np.sum(null >= score)) / (1 + len(null))
    return {"score": float(score), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "corrected": float(score - null.mean()), "z": float((score - null.mean()) / (null.std() + 1e-12)),
            "p": float(p), "n_null": int(len(null))}


# ----------------------------------------------------------------------------- latents & yield
def latent_features(X: np.ndarray, n_components: int = 10, seed: int = 0) -> np.ndarray:
    """GPFA-lite: factor analysis on (trials x units) features; returns (trials x n_components) scores."""
    n_components = int(min(n_components, X.shape[1] - 1)) if X.shape[1] > 1 else 1
    fa = FactorAnalysis(n_components=n_components, random_state=seed)
    return fa.fit_transform(StandardScaler().fit_transform(X))


def unit_yield_curve(X: np.ndarray, y: np.ndarray, n_units_grid: Iterable[int] = (5, 10, 20, 40, 80),
                     n_rep: int = 5, seed: int = 0, **kw) -> Dict[int, Tuple[float, float]]:
    """Decoding score as a function of the number of (randomly subsampled) units: {n: (mean, sd)}."""
    rng = np.random.default_rng(seed)
    out: Dict[int, Tuple[float, float]] = {}
    for n in n_units_grid:
        if n > X.shape[1]:
            break
        s = [decode_cv(X[:, rng.choice(X.shape[1], n, replace=False)], y, seed=seed + r, **kw).score for r in range(n_rep)]
        out[int(n)] = (float(np.mean(s)), float(np.std(s)))
    return out


def fisher_combine(pvals: Sequence[float]) -> float:
    """Fisher's method to combine per-session one-sided p-values (as in the BWM paper)."""
    p = np.clip(np.asarray(pvals, float), 1e-300, 1.0)
    stat = -2 * np.sum(np.log(p))
    return float(stats.chi2.sf(stat, 2 * len(p)))
