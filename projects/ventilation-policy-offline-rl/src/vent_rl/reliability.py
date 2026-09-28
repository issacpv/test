"""OPE-reliability evaluation harness.

* `SyntheticMDP`  tabular MDP with absorbing outcomes, exact policy values (linear solve) and a sampler.
* `make_icu_like_mdp`  severity-ordered synthetic ventilation MDP (states = severity x oxygenation bins).
* `calibrate_from_data`  empirical MDP fitted to real (discretised) trajectories -> SyntheticMDP.
* `site_shift`  perturb transitions and behaviour policy to mimic another site.
* `audit`  run all estimators over candidate policies x replicates -> bias, RMSE, coverage, ESS, rank agreement.
* `cross_site_protocol`  the real-data design: BC on site B -> target; OPE on site A; ground truth on B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .mdp_builder import Trajectories
from .ope import evaluate_all
from .policies import behavior_cloning_tabular, empirical_mdp, greedy_policy, perturb_policy, policy_divergence, q_iteration


@dataclass
class SyntheticMDP:
    P: np.ndarray          # (S, A, S) transition probabilities among non-terminal states
    R: np.ndarray          # (S, A) expected immediate reward (terminal outcome reward folded in)
    D: np.ndarray          # (S, A) probability the episode terminates after (s, a)
    init: np.ndarray       # (S,) initial state distribution
    gamma: float = 1.0
    horizon: int = 42      # 7 days of 4-h bins

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @property
    def n_actions(self) -> int:
        return self.P.shape[1]

    def q_pi(self, pi: np.ndarray, n_iter: int = 2000) -> np.ndarray:
        """Q^pi for a finite horizon by backward induction (exact for the sampled process)."""
        Q = np.zeros_like(self.R)
        for _ in range(self.horizon):
            V = (pi * Q).sum(axis=1)
            Q = self.R + self.gamma * (1.0 - self.D) * (self.P @ V)
        return Q

    def true_value(self, pi: np.ndarray) -> float:
        Q = self.q_pi(pi)
        return float(self.init @ (pi * Q).sum(axis=1))

    def sample(self, pi_b: np.ndarray, n_traj: int, seed: int = 0, reward_noise: float = 0.0) -> Trajectories:
        """Sample trajectories under `pi_b`; rewards are R[s,a] (+ Gaussian noise)."""
        rng = np.random.default_rng(seed)
        S, A = self.n_states, self.n_actions
        s_l, a_l, r_l, ns_l, d_l, id_l, t_l = [], [], [], [], [], [], []
        for i in range(n_traj):
            s = rng.choice(S, p=self.init)
            for t in range(self.horizon):
                a = rng.choice(A, p=pi_b[s])
                r = self.R[s, a] + (rng.normal(0, reward_noise) if reward_noise else 0.0)
                done = rng.random() < self.D[s, a] or t == self.horizon - 1
                ns = s if done else rng.choice(S, p=self.P[s, a])
                s_l.append(s); a_l.append(a); r_l.append(r); ns_l.append(ns); d_l.append(done); id_l.append(i); t_l.append(t)
                if done:
                    break
                s = ns
        s_arr, ns_arr = np.array(s_l), np.array(ns_l)
        obs = s_arr[:, None].astype(float)
        return Trajectories(obs, np.array(a_l), np.array(r_l, float), ns_arr[:, None].astype(float), np.array(d_l),
                            np.array(id_l), np.array(t_l), s_arr, ns_arr, ("state",))


def make_icu_like_mdp(n_severity: int = 6, n_ox: int = 3, n_actions: int = 9, seed: int = 0,
                      base_mortality: float = 0.02, discharge_rate: float = 0.06, horizon: int = 42) -> SyntheticMDP:
    """A small ventilation-like MDP: states = severity (0 best) x oxygenation bin; actions differ in how
    much they improve oxygenation vs. how much lung injury (severity drift) they cause. Terminal outcomes:
    death (reward -1, probability rising with severity/hypoxaemia) and discharge (+1, from low severity).
    """
    rng = np.random.default_rng(seed)
    S = n_severity * n_ox
    A = n_actions
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    D = np.zeros((S, A))
    # action "aggressiveness" in [0,1]: more aggressive -> better oxygenation now, more severity drift later
    aggr = np.linspace(0, 1, A)
    for sev in range(n_severity):
        for ox in range(n_ox):
            s = sev * n_ox + ox
            for a in range(A):
                p_death = base_mortality * (1 + 2 * sev / (n_severity - 1)) * (1 + (n_ox - 1 - ox)) * (1 + 0.5 * aggr[a])
                p_disch = discharge_rate * (1 - sev / (n_severity - 1)) ** 2
                p_death, p_disch = min(p_death, 0.5), min(p_disch, 0.5)
                D[s, a] = p_death + p_disch
                R[s, a] = -1.0 * p_death + 1.0 * p_disch
                # transitions: oxygenation improves with aggr, severity worsens with aggr and improves slowly otherwise
                ox_up = 0.3 + 0.5 * aggr[a]
                sev_up = 0.05 + 0.35 * aggr[a] ** 2
                sev_down = 0.15 * (1 - aggr[a]) + 0.05
                probs = np.zeros(S)
                for dsev, psev in ((1, sev_up), (-1, sev_down), (0, 1 - sev_up - sev_down)):
                    for dox, pox in ((1, ox_up), (-1, 0.2), (0, 1 - ox_up - 0.2)):
                        ns_sev = int(np.clip(sev + dsev, 0, n_severity - 1))
                        ns_ox = int(np.clip(ox + dox, 0, n_ox - 1))
                        probs[ns_sev * n_ox + ns_ox] += max(psev, 0) * max(pox, 0)
                probs += rng.dirichlet(np.ones(S)) * 0.02
                P[s, a] = probs / probs.sum()
    init = np.zeros(S)
    for sev in range(n_severity):
        for ox in range(n_ox):
            init[sev * n_ox + ox] = stats.binom.pmf(sev, n_severity - 1, 0.5) * (1.0 / n_ox)
    init /= init.sum()
    return SyntheticMDP(P, R, D, init, gamma=1.0, horizon=horizon)


def calibrate_from_data(traj: Trajectories, n_states: int, n_actions: int, horizon: int = 42, alpha: float = 0.1) -> SyntheticMDP:
    """Empirical MDP from real discretised trajectories (ICU-Sepsis style) with exact solvable values."""
    P, R, D, _ = empirical_mdp(traj, n_states, n_actions, alpha)
    s0 = np.array([traj.state[ep[0]] for ep in traj.episodes()])
    init = np.bincount(s0, minlength=n_states).astype(float)
    init /= init.sum()
    return SyntheticMDP(P, R, D, init, gamma=1.0, horizon=horizon)


def clinician_like_policy(mdp: SyntheticMDP, temperature: float = 0.5, seed: int = 0) -> np.ndarray:
    """A soft-optimal behaviour policy (clinicians are good but stochastic)."""
    Q = q_iteration(mdp.P, mdp.R, mdp.D, mdp.gamma, n_iter=mdp.horizon)
    z = (Q - Q.max(axis=1, keepdims=True)) / temperature
    pi = np.exp(z)
    return pi / pi.sum(axis=1, keepdims=True)


def site_shift(mdp: SyntheticMDP, pi_b: np.ndarray, strength: float = 0.2, seed: int = 1) -> Tuple[SyntheticMDP, np.ndarray]:
    """Perturb dynamics (Dirichlet mixing) and behaviour policy to emulate a second site."""
    rng = np.random.default_rng(seed)
    P2 = (1 - strength) * mdp.P + strength * rng.dirichlet(np.ones(mdp.n_states), size=(mdp.n_states, mdp.n_actions))
    D2 = np.clip(mdp.D * (1 + strength * rng.normal(0, 0.5, mdp.D.shape)), 0, 0.9)
    R2 = mdp.R * (D2 / np.clip(mdp.D, 1e-9, None))
    pi2 = perturb_policy(pi_b, rng, strength)
    return SyntheticMDP(P2, R2, D2, mdp.init, mdp.gamma, mdp.horizon), pi2


def candidate_policies(mdp: SyntheticMDP, pi_b: np.ndarray, n: int = 10, seed: int = 0,
                       strengths: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.6)) -> List[Tuple[str, np.ndarray]]:
    """Target policies at increasing divergence from behaviour, plus greedy-optimal and behaviour itself."""
    rng = np.random.default_rng(seed)
    out = [("behavior", pi_b.copy())]
    Q = q_iteration(mdp.P, mdp.R, mdp.D, mdp.gamma, n_iter=mdp.horizon)
    out.append(("greedy_opt_mix0.2", 0.8 * greedy_policy(Q) + 0.2 * pi_b))
    k = 0
    while len(out) < n:
        s = strengths[k % len(strengths)]
        out.append((f"perturb{s}_{k}", perturb_policy(pi_b, rng, s)))
        k += 1
    return out


def audit(mdp: SyntheticMDP, pi_b: np.ndarray, targets: Sequence[Tuple[str, np.ndarray]], n_traj: int = 2000,
          n_rep: int = 5, seed: int = 0, n_boot: int = 0, clip: Optional[float] = None,
          bc_alpha: float = 0.5) -> pd.DataFrame:
    """Estimator audit on a ground-truth MDP: bias, RMSE, CI coverage (if n_boot), ESS, divergence, rank agreement.

    The behaviour policy used by the estimators is *estimated* from the sampled data (as in practice), not the
    true `pi_b`.
    """
    rows = []
    S, A = mdp.n_states, mdp.n_actions
    truth = {name: mdp.true_value(pi) for name, pi in targets}
    for rep in range(n_rep):
        traj = mdp.sample(pi_b, n_traj, seed=seed + rep)
        pi_b_hat = behavior_cloning_tabular(traj, S, A, bc_alpha)
        occ = np.bincount(traj.state, minlength=S).astype(float)
        est_by_policy: Dict[str, Dict[str, float]] = {}
        for name, pi in targets:
            res = evaluate_all(traj, pi, pi_b_hat, S, A, mdp.gamma, clip, n_boot, seed + rep)
            est_by_policy[name] = {k: v["value"] for k, v in res.items()}
            for est, v in res.items():
                covered = np.nan
                if "ci_lo" in v:
                    covered = float(v["ci_lo"] <= truth[name] <= v["ci_hi"])
                rows.append({"rep": rep, "policy": name, "estimator": est, "estimate": v["value"], "truth": truth[name],
                             "error": v["value"] - truth[name], "ess": v.get("ess", np.nan), "covered": covered,
                             "divergence": policy_divergence(pi, pi_b, occ)})
        # rank agreement per estimator within replicate
        names = [n for n, _ in targets]
        for est in next(iter(est_by_policy.values())).keys():
            e = [est_by_policy[n][est] for n in names]
            t = [truth[n] for n in names]
            rho = stats.spearmanr(e, t).correlation if len(names) > 2 else np.nan
            rows.append({"rep": rep, "policy": "__rank__", "estimator": est, "estimate": np.nan, "truth": np.nan,
                         "error": np.nan, "ess": np.nan, "covered": np.nan, "divergence": np.nan, "rank_rho": rho})
    return pd.DataFrame(rows)


def summarize_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Per-estimator bias, RMSE, mean |error|, coverage, median ESS and mean rank correlation."""
    d = df[df["policy"] != "__rank__"]
    g = d.groupby("estimator")
    out = pd.DataFrame({
        "bias": g["error"].mean(), "rmse": g["error"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
        "mae": g["error"].apply(lambda e: float(np.mean(np.abs(e)))), "coverage": g["covered"].mean(),
        "median_ess": g["ess"].median(),
    })
    if "rank_rho" in df:
        out["rank_rho"] = df[df["policy"] == "__rank__"].groupby("estimator")["rank_rho"].mean()
    return out.sort_values("rmse")


def cross_site_protocol(traj_A: Trajectories, traj_B: Trajectories, n_states: int, n_actions: int, gamma: float = 1.0,
                        transport_weights: Optional[np.ndarray] = None, clip: Optional[float] = None,
                        n_boot: int = 0, bc_alpha: float = 0.5) -> Dict[str, object]:
    """Cross-site quasi-ground-truth: target = BC(site B); OPE on site A; truth = (weighted) mean return on B.

    `transport_weights` (one per B trajectory) re-weight B's outcomes to A's case-mix (e.g. inverse
    probability of site from baseline covariates). Both sites must share the same state discretisation.
    """
    pi_target = behavior_cloning_tabular(traj_B, n_states, n_actions, bc_alpha)
    pi_A = behavior_cloning_tabular(traj_A, n_states, n_actions, bc_alpha)
    G_B = traj_B.returns(gamma)
    w = np.ones(len(G_B)) if transport_weights is None else np.asarray(transport_weights, float)
    truth = float(np.sum(w * G_B) / w.sum())
    res = evaluate_all(traj_A, pi_target, pi_A, n_states, n_actions, gamma, clip, n_boot)
    occ = np.bincount(traj_A.state, minlength=n_states).astype(float)
    return {"truth_B": truth, "estimates_on_A": res, "divergence": policy_divergence(pi_target, pi_A, occ),
            "errors": {k: v["value"] - truth for k, v in res.items()}}
