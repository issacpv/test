"""Off-policy evaluation estimators for tabular target policies on logged trajectories.

Estimators: WIS (weighted importance sampling, per-trajectory), PDIS (per-decision IS, ordinary and
weighted), tabular FQE / model-based Q^pi, linear FQE on continuous features, and the doubly-robust
estimator of Jiang & Li (2016) / Thomas & Brunskill (2016). Diagnostics: effective sample size,
bootstrap confidence intervals.

All estimators take a `Trajectories` with discrete `state` (except `fqe_linear`), a target policy
`pi_e` (S x A) and a behaviour policy `pi_b` (S x A).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mdp_builder import Trajectories
from .policies import empirical_mdp, policy_evaluation


# ----------------------------------------------------------------------------- weights
def step_ratios(traj: Trajectories, pi_e: np.ndarray, pi_b: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
    """Per-step importance ratios pi_e(a|s) / pi_b(a|s) (optionally clipped)."""
    rho = pi_e[traj.state, traj.action] / np.clip(pi_b[traj.state, traj.action], 1e-12, None)
    return np.clip(rho, 0.0, clip) if clip is not None else rho


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.clip((w ** 2).sum(), 1e-300, None))


# ----------------------------------------------------------------------------- IS family
def wis(traj: Trajectories, pi_e: np.ndarray, pi_b: np.ndarray, gamma: float = 1.0, clip: Optional[float] = None,
        weighted: bool = True) -> Dict[str, float]:
    """(Weighted) trajectory-level importance sampling of discounted returns."""
    rho = step_ratios(traj, pi_e, pi_b, clip)
    eps = traj.episodes()
    W = np.array([np.prod(rho[ep]) for ep in eps])
    G = np.array([np.sum(traj.reward[ep] * gamma ** np.arange(len(ep))) for ep in eps])
    if weighted:
        v = float(np.sum(W * G) / np.clip(W.sum(), 1e-300, None))
    else:
        v = float(np.mean(W * G))
    return {"value": v, "ess": effective_sample_size(W), "n_traj": len(eps), "max_weight": float(W.max()) if len(W) else np.nan}


def pdis(traj: Trajectories, pi_e: np.ndarray, pi_b: np.ndarray, gamma: float = 1.0, clip: Optional[float] = None,
         weighted: bool = True) -> Dict[str, float]:
    """Per-decision importance sampling: sum_t gamma^t rho_{1:t} r_t (weighted per time step if `weighted`)."""
    rho = step_ratios(traj, pi_e, pi_b, clip)
    eps = traj.episodes()
    H = max(len(ep) for ep in eps)
    cum = np.zeros((len(eps), H))
    rew = np.zeros((len(eps), H))
    for i, ep in enumerate(eps):
        cum[i, :len(ep)] = np.cumprod(rho[ep])
        rew[i, :len(ep)] = traj.reward[ep]
    disc = gamma ** np.arange(H)
    if weighted:
        norm = np.clip(cum.sum(axis=0), 1e-300, None)  # per-time-step normalisation
        v = float(np.sum(disc * (cum * rew).sum(axis=0) / norm))
    else:
        v = float(np.mean((cum * rew * disc).sum(axis=1)))
    return {"value": v, "ess": effective_sample_size(cum[:, 0]), "n_traj": len(eps)}


# ----------------------------------------------------------------------------- FQE
def fqe_tabular(traj: Trajectories, pi_e: np.ndarray, n_states: int, n_actions: int, gamma: float = 1.0,
                alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Q^pi_e from the empirical MDP fitted to the logged data (tabular FQE / model-based evaluation)."""
    P, R, D, _ = empirical_mdp(traj, n_states, n_actions, alpha)
    Q = policy_evaluation(P, R, D, pi_e, gamma)
    V = (pi_e * Q).sum(axis=1)
    return Q, V


def fqe_value(traj: Trajectories, pi_e: np.ndarray, n_states: int, n_actions: int, gamma: float = 1.0) -> Dict[str, float]:
    """FQE estimate = mean of V^pi_e over the logged initial states."""
    _, V = fqe_tabular(traj, pi_e, n_states, n_actions, gamma)
    s0 = np.array([traj.state[ep[0]] for ep in traj.episodes()])
    return {"value": float(V[s0].mean()), "n_traj": len(s0)}


def fqe_linear(traj: Trajectories, pi_e_fn: Callable[[np.ndarray], np.ndarray], n_actions: int, gamma: float = 1.0,
               n_iter: int = 50, ridge: float = 1e-3) -> Tuple[np.ndarray, float]:
    """Linear FQE on continuous features: Q(s,a) = phi(s,a)^T w with phi = onehot(a) (x) [1, obs].

    `pi_e_fn(obs) -> (n, n_actions)` gives target-policy probabilities. Returns (w, value at initial states).
    """
    X = np.hstack([np.ones((traj.n, 1)), traj.obs])
    nX = np.hstack([np.ones((traj.n, 1)), traj.next_obs])
    d = X.shape[1]

    def phi(F: np.ndarray, a: np.ndarray) -> np.ndarray:
        out = np.zeros((len(F), n_actions * d))
        for k in range(n_actions):
            m = a == k
            out[m, k * d:(k + 1) * d] = F[m]
        return out

    Phi = phi(X, traj.action)
    pi_next = pi_e_fn(traj.next_obs)
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    w = np.zeros(Phi.shape[1])
    for _ in range(n_iter):
        q_next = np.zeros(traj.n)
        for k in range(n_actions):
            q_next += pi_next[:, k] * (nX @ w[k * d:(k + 1) * d])
        y = traj.reward + gamma * (1.0 - traj.done) * q_next
        w = np.linalg.solve(A, Phi.T @ y)
    s0 = np.array([ep[0] for ep in traj.episodes()])
    pi0 = pi_e_fn(traj.obs[s0])
    v0 = sum(pi0[:, k] * (X[s0] @ w[k * d:(k + 1) * d]) for k in range(n_actions))
    return w, float(np.mean(v0))


# ----------------------------------------------------------------------------- doubly robust
def doubly_robust(traj: Trajectories, pi_e: np.ndarray, pi_b: np.ndarray, Q: np.ndarray, gamma: float = 1.0,
                  clip: Optional[float] = None, weighted: bool = False) -> Dict[str, float]:
    """Sequential doubly-robust estimator (Jiang & Li 2016):
    V_DR(t) = V̂(s_t) + rho_t [ r_t + gamma V_DR(t+1) - Q̂(s_t,a_t) ],  V̂(s) = sum_a pi_e(a|s) Q̂(s,a).

    With `weighted=True` the per-step ratios are self-normalised across trajectories at each time step (WDR).
    """
    rho = step_ratios(traj, pi_e, pi_b, clip)
    V = (pi_e * Q).sum(axis=1)
    eps = traj.episodes()
    if weighted:
        H = max(len(ep) for ep in eps)
        cum = np.zeros((len(eps), H))
        for i, ep in enumerate(eps):
            cum[i, :len(ep)] = np.cumprod(rho[ep])
        norm = np.clip(cum.mean(axis=0), 1e-300, None)
    vals = []
    for i, ep in enumerate(eps):
        v_next = 0.0
        for j in range(len(ep) - 1, -1, -1):
            k = ep[j]
            r_j = rho[k]
            if weighted:
                # scale so that self-normalised cumulative weights are used (WDR)
                prev = cum[i, j - 1] / norm[j - 1] if j > 0 else 1.0
                curw = cum[i, j] / norm[j]
                r_j = curw / prev if prev > 0 else 0.0
            v_hat = V[traj.state[k]]
            q_hat = Q[traj.state[k], traj.action[k]]
            v_next = v_hat + r_j * (traj.reward[k] + gamma * (0.0 if traj.done[k] else v_next) - q_hat)
        vals.append(v_next)
    vals = np.asarray(vals)
    return {"value": float(vals.mean()), "se": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
            "n_traj": len(vals)}


# ----------------------------------------------------------------------------- bootstrap & bundle
def bootstrap_ci(est: Callable[[Trajectories], float], traj: Trajectories, n_boot: int = 200, seed: int = 0,
                 level: float = 0.95) -> Tuple[float, float]:
    """Percentile CI by resampling whole trajectories."""
    rng = np.random.default_rng(seed)
    eps = traj.episodes()
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(eps), len(eps))
        idx = np.concatenate([eps[i] for i in pick])
        sub = traj.subset(idx)
        # re-label trajectory ids so duplicates are distinct episodes
        lens = [len(eps[i]) for i in pick]
        sub.traj_id = np.repeat(np.arange(len(pick)), lens)
        vals.append(est(sub))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def evaluate_all(traj: Trajectories, pi_e: np.ndarray, pi_b: np.ndarray, n_states: int, n_actions: int,
                 gamma: float = 1.0, clip: Optional[float] = None, n_boot: int = 0, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Run WIS, PDIS, FQE and DR (+ optional bootstrap CIs) and return a dict of estimator -> results."""
    Q, _ = fqe_tabular(traj, pi_e, n_states, n_actions, gamma)
    out = {
        "WIS": wis(traj, pi_e, pi_b, gamma, clip),
        "IS": wis(traj, pi_e, pi_b, gamma, clip, weighted=False),
        "PDIS": pdis(traj, pi_e, pi_b, gamma, clip),
        "FQE": fqe_value(traj, pi_e, n_states, n_actions, gamma),
        "DR": doubly_robust(traj, pi_e, pi_b, Q, gamma, clip),
        "WDR": doubly_robust(traj, pi_e, pi_b, Q, gamma, clip, weighted=True),
    }
    if n_boot > 0:
        fns = {
            "WIS": lambda t: wis(t, pi_e, pi_b, gamma, clip)["value"],
            "PDIS": lambda t: pdis(t, pi_e, pi_b, gamma, clip)["value"],
            "FQE": lambda t: fqe_value(t, pi_e, n_states, n_actions, gamma)["value"],
            "DR": lambda t: doubly_robust(t, pi_e, pi_b, fqe_tabular(t, pi_e, n_states, n_actions, gamma)[0], gamma, clip)["value"],
        }
        for k, fn in fns.items():
            out[k]["ci_lo"], out[k]["ci_hi"] = bootstrap_ci(fn, traj, n_boot, seed)
    return out
