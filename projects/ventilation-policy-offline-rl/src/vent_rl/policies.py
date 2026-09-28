"""Policies for tabular offline RL: behaviour cloning, empirical MDP, value iteration, CQL-lite.

A tabular policy is a (n_states, n_actions) row-stochastic matrix. Continuous-feature behaviour cloning
(multinomial logistic) is provided for the importance-sampling estimators when states are not clustered.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.special import logsumexp, softmax

from .mdp_builder import Trajectories


# ----------------------------------------------------------------------------- behaviour cloning
def behavior_cloning_tabular(traj: Trajectories, n_states: int, n_actions: int, alpha: float = 0.5) -> np.ndarray:
    """Dirichlet-smoothed empirical action frequencies per discrete state: pi_b[s, a]."""
    if traj.state is None:
        raise ValueError("discrete states required (run mdp_builder.discretize_states)")
    counts = np.zeros((n_states, n_actions))
    np.add.at(counts, (traj.state, traj.action), 1.0)
    return (counts + alpha) / (counts + alpha).sum(axis=1, keepdims=True)


def behavior_cloning_logistic(traj: Trajectories, C: float = 1.0, seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """Multinomial logistic behaviour model on continuous features; returns f(obs) -> (n, n_actions) probs.

    Actions absent from the data get probability ~0 (columns are aligned to range(max_action+1)).
    """
    from sklearn.linear_model import LogisticRegression
    n_actions = int(traj.action.max()) + 1
    clf = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(traj.obs, traj.action)

    def f(obs: np.ndarray) -> np.ndarray:
        p = np.full((len(obs), n_actions), 1e-6)
        p[:, clf.classes_] = clf.predict_proba(obs)
        return p / p.sum(axis=1, keepdims=True)
    return f


# ----------------------------------------------------------------------------- empirical MDP
def empirical_mdp(traj: Trajectories, n_states: int, n_actions: int, alpha: float = 0.0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Counts-based P[s,a,s'], R[s,a], terminal probability D[s,a] and visit counts N[s,a].

    Transitions with done=True are recorded in D (no next state).
    """
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    D = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    s, a, r, ns, d = traj.state, traj.action, traj.reward, traj.next_state, traj.done
    np.add.at(N, (s, a), 1.0)
    np.add.at(R, (s, a), r)
    np.add.at(D, (s, a), d.astype(float))
    cont = ~d
    np.add.at(P, (s[cont], a[cont], ns[cont]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(N > 0, R / np.maximum(N, 1), 0.0)
        D = np.where(N > 0, D / np.maximum(N, 1), 1.0)
        Pn = (P + alpha) / (P + alpha).sum(axis=2, keepdims=True)
        Pn = np.where(np.isnan(Pn), 1.0 / n_states, Pn)
    return Pn, R, D, N


def q_iteration(P: np.ndarray, R: np.ndarray, D: np.ndarray, gamma: float = 0.99, n_iter: int = 500,
                tol: float = 1e-8) -> np.ndarray:
    """Optimal Q by value iteration: Q = R + gamma (1-D) sum_s' P max_a' Q."""
    Q = np.zeros_like(R)
    for _ in range(n_iter):
        V = Q.max(axis=1)
        Qn = R + gamma * (1.0 - D) * (P @ V)
        if np.max(np.abs(Qn - Q)) < tol:
            return Qn
        Q = Qn
    return Q


def policy_evaluation(P: np.ndarray, R: np.ndarray, D: np.ndarray, pi: np.ndarray, gamma: float = 0.99,
                      n_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Q^pi by iterating Q = R + gamma (1-D) P (pi . Q)."""
    Q = np.zeros_like(R)
    for _ in range(n_iter):
        V = (pi * Q).sum(axis=1)
        Qn = R + gamma * (1.0 - D) * (P @ V)
        if np.max(np.abs(Qn - Q)) < tol:
            return Qn
        Q = Qn
    return Q


# ----------------------------------------------------------------------------- CQL-lite
def cql_tabular(traj: Trajectories, n_states: int, n_actions: int, gamma: float = 0.99, alpha: float = 1.0,
                lr: float = 0.1, n_epochs: int = 50, seed: int = 0, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Tabular conservative Q-learning (Kumar et al. 2020) on logged transitions.

    Per transition: TD update towards r + gamma max_a' Q(s',a') plus the CQL regulariser gradient
    alpha * (softmax(Q(s,.)) - onehot(a)), which pushes down Q of actions unsupported by the data.
    `action_mask` (n_states, n_actions) bool restricts the max/argmax to guideline-allowed actions.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))
    mask = np.ones((n_states, n_actions), bool) if action_mask is None else action_mask
    s, a, r, ns, d = traj.state, traj.action, traj.reward, traj.next_state, traj.done
    for _ in range(n_epochs):
        for i in rng.permutation(traj.n):
            qn = np.where(mask[ns[i]], Q[ns[i]], -np.inf)
            target = r[i] + gamma * (0.0 if d[i] else qn.max())
            td = target - Q[s[i], a[i]]
            reg = softmax(np.where(mask[s[i]], Q[s[i]], -1e9))
            reg[a[i]] -= 1.0
            Q[s[i]] -= lr * alpha * reg
            Q[s[i], a[i]] += lr * td
    return Q


def greedy_policy(Q: np.ndarray, action_mask: Optional[np.ndarray] = None, eps: float = 0.0) -> np.ndarray:
    """Deterministic (or eps-greedy over allowed actions) tabular policy from Q."""
    Qm = Q if action_mask is None else np.where(action_mask, Q, -np.inf)
    n_s, n_a = Q.shape
    pi = np.full((n_s, n_a), 0.0)
    best = Qm.argmax(axis=1)
    pi[np.arange(n_s), best] = 1.0
    if eps > 0:
        allowed = np.ones_like(Q, bool) if action_mask is None else action_mask
        uni = allowed / allowed.sum(axis=1, keepdims=True)
        pi = (1 - eps) * pi + eps * uni
    return pi


def softmax_policy(Q: np.ndarray, temperature: float = 1.0, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
    Qm = Q if action_mask is None else np.where(action_mask, Q, -1e9)
    return softmax(Qm / temperature, axis=1)


def mix_with_behavior(pi: np.ndarray, pi_b: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Keep the target inside the behaviour support: (1-eps) pi + eps pi_b."""
    return (1 - eps) * pi + eps * pi_b


def policy_divergence(pi_e: np.ndarray, pi_b: np.ndarray, state_weights: Optional[np.ndarray] = None) -> float:
    """State-weighted mean total-variation distance between two tabular policies."""
    tv = 0.5 * np.abs(pi_e - pi_b).sum(axis=1)
    w = np.ones(len(tv)) / len(tv) if state_weights is None else state_weights / state_weights.sum()
    return float((w * tv).sum())


def perturb_policy(pi: np.ndarray, rng: np.random.Generator, strength: float = 0.2) -> np.ndarray:
    """Random perturbation of a tabular policy (for generating candidate target policies)."""
    noise = rng.dirichlet(np.ones(pi.shape[1]), size=pi.shape[0])
    return (1 - strength) * pi + strength * noise
