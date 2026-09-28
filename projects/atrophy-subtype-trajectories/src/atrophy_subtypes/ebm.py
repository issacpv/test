"""Compact z-score event-based model with EM subtyping (a SuStaIn-like model).

Model (Young et al., 2018, Nature Communications; Fonteijn et al., 2012, NeuroImage):

* each biomarker ``b`` has events at z-thresholds ``z_1 < z_2 < ...`` (default 1, 2, 3);
* a subtype is an ordering ``S`` of all ``E = B × n_thresholds`` events (events of one biomarker
  must appear in increasing z-order);
* at stage ``k`` (0..E) the expected value of biomarker ``b`` is piecewise linear through the knots
  (0, 0), (position of z_1 event, z_1), ..., (E + 1, z_max);
* ``p(x | k, S) = Π_b N(x_b; μ_b(k), 1)`` and ``p(x | S) = mean_k p(x | k, S)`` (uniform stage prior);
* with ``K`` subtypes, ``p(x) = Σ_k f_k p(x | S_k)``.

Inference: greedy event moves with random restarts for the sequence; EM over responsibilities for
subtypes. This is deliberately small (no MCMC); use ``run_pysustain`` for the reference
implementation with uncertainty estimates once the analysis pipeline is settled.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import kendalltau


class ZScoreEBM:
    """Single- or multi-subtype z-score event-based model.

    Parameters
    ----------
    n_biomarkers : number of biomarkers (columns of X).
    z_thresholds : event thresholds per biomarker.
    z_max : value the trajectory reaches at the virtual end stage E + 1.
    n_subtypes : K.
    n_restarts : random restarts of the optimizer (best log-likelihood kept).
    max_passes : maximum greedy passes over the events per optimization.
    em_iter : EM iterations for K > 1.
    """

    def __init__(self, n_biomarkers: int, z_thresholds: Sequence[float] = (1.0, 2.0, 3.0), z_max: float = 5.0,
                 n_subtypes: int = 1, n_restarts: int = 5, max_passes: int = 20, em_iter: int = 15,
                 random_state: int = 0):
        self.B = int(n_biomarkers)
        self.z_thresholds = np.asarray(z_thresholds, float)
        self.z_max = float(z_max)
        self.K = int(n_subtypes)
        self.n_restarts = n_restarts
        self.max_passes = max_passes
        self.em_iter = em_iter
        self.rng = np.random.default_rng(random_state)
        # event e = (biomarker, threshold index)
        self.events = np.array([(b, j) for b in range(self.B) for j in range(len(self.z_thresholds))])
        self.E = len(self.events)

    # ------------------------------------------------------------------ model
    def expected_values(self, sequence: np.ndarray) -> np.ndarray:
        """(E+1) × B matrix of expected z at each stage for the given event ordering."""
        pos = np.empty(self.E)
        pos[sequence] = np.arange(1, self.E + 1)  # stage at which each event has occurred
        stages = np.arange(self.E + 1)
        mu = np.zeros((self.E + 1, self.B))
        for b in range(self.B):
            ev = [e for e in range(self.E) if self.events[e, 0] == b]
            xs = np.concatenate([[0.0], pos[ev], [self.E + 1.0]])
            ys = np.concatenate([[0.0], self.z_thresholds[self.events[ev, 1]], [self.z_max]])
            order = np.argsort(xs)
            mu[:, b] = np.interp(stages, xs[order], ys[order])
        return mu

    def stage_loglik(self, X: np.ndarray, sequence: np.ndarray) -> np.ndarray:
        """n × (E+1) matrix of log p(x_i | stage k, S); NaN biomarkers are ignored."""
        X = np.asarray(X, float)
        mu = self.expected_values(sequence)
        diff = X[:, None, :] - mu[None, :, :]
        ll = -0.5 * diff ** 2 - 0.5 * np.log(2 * np.pi)
        return np.nansum(ll, axis=2)

    def subject_loglik(self, X: np.ndarray, sequence: np.ndarray) -> np.ndarray:
        """log p(x_i | S) with uniform stage prior."""
        return logsumexp(self.stage_loglik(X, sequence), axis=1) - np.log(self.E + 1)

    def is_valid(self, sequence: np.ndarray) -> bool:
        pos = np.empty(self.E)
        pos[sequence] = np.arange(self.E)
        for b in range(self.B):
            ev = np.flatnonzero(self.events[:, 0] == b)
            if not np.all(np.diff(pos[ev]) > 0):
                return False
        return True

    # ------------------------------------------------------------ optimizer
    def _initial_sequence(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Order events by weighted fraction of subjects above the threshold (descending), then
        repair the within-biomarker order; add a little noise for restarts."""
        frac = np.array([np.average(np.nan_to_num(X[:, b] >= z, nan=0.0), weights=weights)
                         for b, j in self.events for z in [self.z_thresholds[j]]])
        frac = frac + self.rng.normal(0, 0.02, self.E)
        seq = np.argsort(-frac)
        return self._repair(seq)

    def _repair(self, seq: np.ndarray) -> np.ndarray:
        """Enforce increasing z-order within each biomarker by sorting the positions it occupies."""
        seq = seq.copy()
        for b in range(self.B):
            ev = np.flatnonzero(self.events[:, 0] == b)
            slots = np.sort(np.flatnonzero(np.isin(seq, ev)))
            seq[slots] = ev[np.argsort(self.events[ev, 1])]
        return seq

    def optimize_sequence(self, X: np.ndarray, weights: Optional[np.ndarray] = None,
                          init: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
        """Greedy coordinate ascent on the weighted log-likelihood over event positions."""
        X = np.asarray(X, float)
        w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
        seq = self._initial_sequence(X, w) if init is None else np.asarray(init).copy()
        best = float(np.dot(w, self.subject_loglik(X, seq)))
        for _ in range(self.max_passes):
            improved = False
            for e in self.rng.permutation(self.E):
                cur = int(np.flatnonzero(seq == e)[0])
                rest = np.delete(seq, cur)
                for newpos in range(self.E):
                    if newpos == cur:
                        continue
                    cand = np.insert(rest, newpos, e)
                    if not self.is_valid(cand):
                        continue
                    ll = float(np.dot(w, self.subject_loglik(X, cand)))
                    if ll > best + 1e-9:
                        best, seq, cur, rest = ll, cand, newpos, np.delete(cand, newpos)
                        improved = True
            if not improved:
                break
        return seq, best

    def fit(self, X: np.ndarray) -> "ZScoreEBM":
        """Fit K subtype sequences and mixture fractions; keeps the best of ``n_restarts``."""
        X = np.asarray(X, float)
        n = len(X)
        best_ll, best = -np.inf, None
        for r in range(self.n_restarts):
            if self.K == 1:
                seq, ll = self.optimize_sequence(X)
                seqs, f = [seq], np.array([1.0])
            else:
                # random initial hard split, then EM
                resp = self.rng.dirichlet(np.ones(self.K), size=n)
                seqs = [self.optimize_sequence(X, resp[:, k])[0] for k in range(self.K)]
                f = resp.mean(0)
                ll_old = -np.inf
                for _ in range(self.em_iter):
                    L = np.column_stack([self.subject_loglik(X, s) for s in seqs]) + np.log(f + 1e-12)
                    ll = float(logsumexp(L, axis=1).sum())
                    resp = np.exp(L - logsumexp(L, axis=1, keepdims=True))
                    f = resp.mean(0)
                    seqs = [self.optimize_sequence(X, resp[:, k], init=seqs[k])[0] for k in range(self.K)]
                    if ll - ll_old < 1e-4 * max(1.0, abs(ll)):
                        break
                    ll_old = ll
                L = np.column_stack([self.subject_loglik(X, s) for s in seqs]) + np.log(f + 1e-12)
                ll = float(logsumexp(L, axis=1).sum())
            if ll > best_ll:
                best_ll, best = ll, (seqs, f)
        self.sequences_ = np.vstack(best[0])
        self.fractions_ = np.asarray(best[1], float)
        self.loglik_ = best_ll
        return self

    # ------------------------------------------------------------- inference
    def posterior(self, X: np.ndarray) -> np.ndarray:
        """n × K × (E+1) joint posterior over subtype and stage."""
        X = np.asarray(X, float)
        L = np.stack([self.stage_loglik(X, s) for s in self.sequences_], axis=1)  # n × K × (E+1)
        L = L + np.log(self.fractions_ + 1e-12)[None, :, None] - np.log(self.E + 1)
        return np.exp(L - logsumexp(L, axis=(1, 2), keepdims=True))

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Modal subtype, its posterior probability, expected stage and modal stage (given the subtype)."""
        post = self.posterior(X)
        p_sub = post.sum(2)
        subtype = p_sub.argmax(1)
        stage_given = post[np.arange(len(X)), subtype, :]
        stage_given = stage_given / stage_given.sum(1, keepdims=True)
        return {"subtype": subtype, "subtype_prob": p_sub.max(1),
                "stage": stage_given.argmax(1), "stage_expected": stage_given @ np.arange(self.E + 1),
                "stage_entropy": -(stage_given * np.log(stage_given + 1e-12)).sum(1)}

    def loglik(self, X: np.ndarray) -> float:
        L = np.column_stack([self.subject_loglik(X, s) for s in self.sequences_]) + np.log(self.fractions_ + 1e-12)
        return float(logsumexp(L, axis=1).sum())


# ---------------------------------------------------------------------------
# helpers: simulation, sequence comparison, model selection
# ---------------------------------------------------------------------------
def simulate(model: ZScoreEBM, sequences: np.ndarray, fractions: Sequence[float], n: int,
             stages: Optional[np.ndarray] = None, noise_sd: float = 1.0,
             rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n`` subjects from given sequences/fractions. Returns (X, subtype, stage)."""
    rng = np.random.default_rng(0) if rng is None else rng
    sequences = np.atleast_2d(sequences)
    subtype = rng.choice(len(sequences), size=n, p=np.asarray(fractions) / np.sum(fractions))
    stage = rng.integers(0, model.E + 1, size=n) if stages is None else np.asarray(stages)
    X = np.empty((n, model.B))
    for k, s in enumerate(sequences):
        mu = model.expected_values(s)
        m = subtype == k
        X[m] = mu[stage[m]] + rng.normal(0, noise_sd, (m.sum(), model.B))
    return X, subtype, stage


def simulate_longitudinal(model: ZScoreEBM, sequence: np.ndarray, n: int, intervals_years: np.ndarray,
                          events_per_year: float = 0.8, noise_sd: float = 1.0,
                          rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-timepoint simulation: baseline stage uniform, follow-up stage advanced by rate × interval.

    Returns (X0, X1, true_stages[n × 2]). Used to benchmark achievable monotonicity.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    s0 = rng.integers(0, model.E + 1, size=n)
    s1 = np.minimum(model.E, s0 + rng.poisson(events_per_year * np.asarray(intervals_years)))
    mu = model.expected_values(sequence)
    X0 = mu[s0] + rng.normal(0, noise_sd, (n, model.B))
    X1 = mu[s1] + rng.normal(0, noise_sd, (n, model.B))
    return X0, X1, np.column_stack([s0, s1])


def sequence_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Kendall τ between two event orderings (1 = identical order)."""
    pos_a = np.empty(len(seq_a)); pos_a[np.asarray(seq_a)] = np.arange(len(seq_a))
    pos_b = np.empty(len(seq_b)); pos_b[np.asarray(seq_b)] = np.arange(len(seq_b))
    return float(kendalltau(pos_a, pos_b).statistic)


def biomarker_order(model: ZScoreEBM, sequence: np.ndarray, names: Sequence[str],
                    threshold_index: int = 0) -> list[str]:
    """Biomarker names in the order their ``threshold_index``-th event occurs."""
    ev = [e for e in sequence if model.events[e, 1] == threshold_index]
    return [names[model.events[e, 0]] for e in ev]


def cross_validated_loglik(X: np.ndarray, n_subtypes: int, n_folds: int = 5, random_state: int = 0,
                           **kwargs) -> float:
    """Held-out log-likelihood summed over folds (higher is better; ~CVIC for choosing K)."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(random_state)
    folds = rng.permutation(len(X)) % n_folds
    total = 0.0
    for f in range(n_folds):
        m = ZScoreEBM(X.shape[1], n_subtypes=n_subtypes, random_state=random_state + f, **kwargs).fit(X[folds != f])
        total += m.loglik(X[folds == f])
    return total


def run_pysustain(X: np.ndarray, names: Sequence[str], z_vals: Sequence[float] = (1, 2, 3), z_max: float = 5.0,
                  n_subtypes_max: int = 3, n_iter_mcmc: int = 100_000, n_startpoints: int = 25,
                  output_folder: str = "outputs/pysustain", dataset_name: str = "oasis3", seed: int = 0):
    """Reference implementation hook (requires ``pip install git+https://github.com/ucl-pond/pySuStaIn``).

    Returns the fitted ``ZscoreSustain`` object; results are written by pySuStaIn to ``output_folder``.
    """
    try:
        from pySuStaIn import ZscoreSustain
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pySuStaIn not installed") from exc
    X = np.asarray(X, float)
    Z_vals = np.tile(np.asarray(z_vals, float), (X.shape[1], 1))
    Z_max = np.full(X.shape[1], z_max)
    model = ZscoreSustain(X, Z_vals, Z_max, list(names), n_startpoints, n_subtypes_max, n_iter_mcmc,
                          output_folder, dataset_name, False, seed)
    model.run_sustain_algorithm()
    return model
