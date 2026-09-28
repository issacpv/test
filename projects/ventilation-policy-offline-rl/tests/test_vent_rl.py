"""Synthetic tests for vent_rl (no PhysioNet data needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vent_rl import constraints, mdp_builder, ope, policies, reliability  # noqa: E402


@pytest.fixture(scope="module")
def mdp_and_data():
    mdp = reliability.make_icu_like_mdp(n_severity=4, n_ox=2, n_actions=5, seed=0, horizon=20)
    pi_b = reliability.clinician_like_policy(mdp, temperature=1.0)
    traj = mdp.sample(pi_b, n_traj=1500, seed=1)
    return mdp, pi_b, traj


def test_pbw_and_action_discretisation():
    pbw = mdp_builder.predicted_body_weight([170.0, 170.0], ["M", "F"])
    assert pbw[0] == pytest.approx(50 + 0.91 * (170 - 152.4)) and pbw[1] == pytest.approx(45.5 + 0.91 * (170 - 152.4))
    a = mdp_builder.discretize_actions([5.0, 7.0, 9.0], [5.0, 10.0, 14.0], [0.3, 0.5, 0.8])
    assert list(a) == [0, 13, 26] and mdp_builder.n_actions() == 27
    grid = mdp_builder.action_grid()
    assert len(grid) == 27 and grid.loc[13, "vt_bin"] == 1 and grid.loc[13, "peep_bin"] == 1


def test_bin_events_forward_fill_and_trajectories():
    rows = []
    for sid, died in ((1, 0), (2, 1)):
        for h in np.arange(0, 48, 2.0):
            rows += [{"stay_id": sid, "time_h": h, "variable": "peep", "value": 8.0 + sid},
                     {"stay_id": sid, "time_h": h, "variable": "fio2", "value": 0.5},
                     {"stay_id": sid, "time_h": h, "variable": "tidal_volume_set", "value": 420.0}]
        rows.append({"stay_id": sid, "time_h": 1.0, "variable": "spo2", "value": 0.93})
    events = pd.DataFrame(rows)
    wide = mdp_builder.bin_events(events, bin_hours=4.0)
    assert set(wide["t"].unique()) == set(range(12)) and wide["peep"].notna().all()
    wide = mdp_builder.forward_fill(wide, max_gap_bins=2)
    assert wide.loc[(wide.stay_id == 1) & (wide.t == 2), "spo2"].item() == pytest.approx(0.93)
    assert np.isnan(wide.loc[(wide.stay_id == 1) & (wide.t == 5), "spo2"].item())
    outcomes = pd.DataFrame({"stay_id": [1, 2], "died": [0, 1], "height_cm": [175.0, 160.0], "gender": ["M", "F"], "age": [60, 70]})
    traj = mdp_builder.build_trajectories(wide, outcomes, min_bins=6)
    assert traj.n == 24 and traj.done.sum() == 2
    assert traj.returns().tolist() == [1.0, -1.0]
    assert traj.feature_names == mdp_builder.STATE_FEATURES
    scaled, med, sc = mdp_builder.impute_and_scale(traj)
    assert not np.isnan(scaled.obs).any()
    disc, model = mdp_builder.discretize_states(scaled, k=3)
    assert disc.state.max() < 3 and disc.next_state is not None


def test_on_policy_estimators_recover_truth(mdp_and_data):
    mdp, pi_b, traj = mdp_and_data
    S, A = mdp.n_states, mdp.n_actions
    truth = mdp.true_value(pi_b)
    mc = traj.returns().mean()
    assert abs(mc - truth) < 0.08
    pi_b_hat = policies.behavior_cloning_tabular(traj, S, A, alpha=0.1)
    res = ope.evaluate_all(traj, pi_b, pi_b_hat, S, A, gamma=1.0)
    for k in ("WIS", "PDIS", "FQE", "DR", "WDR"):
        assert abs(res[k]["value"] - truth) < 0.12, (k, res[k]["value"], truth)
    assert res["WIS"]["ess"] > 0.3 * res["WIS"]["n_traj"]


def test_off_policy_fqe_and_dr_track_truth(mdp_and_data):
    mdp, pi_b, traj = mdp_and_data
    S, A = mdp.n_states, mdp.n_actions
    pi_e = policies.mix_with_behavior(policies.perturb_policy(pi_b, np.random.default_rng(3), 0.3), pi_b, 0.2)
    truth = mdp.true_value(pi_e)
    pi_b_hat = policies.behavior_cloning_tabular(traj, S, A, alpha=0.1)
    res = ope.evaluate_all(traj, pi_e, pi_b_hat, S, A)
    assert abs(res["FQE"]["value"] - truth) < 0.15
    assert abs(res["DR"]["value"] - truth) < 0.25
    lo, hi = ope.bootstrap_ci(lambda t: ope.fqe_value(t, pi_e, S, A)["value"], traj, n_boot=10)
    assert lo <= hi


def test_linear_fqe_runs(mdp_and_data):
    mdp, pi_b, traj = mdp_and_data
    A = mdp.n_actions
    onehot = np.eye(mdp.n_states)[traj.state]
    t2 = mdp_builder.Trajectories(onehot, traj.action, traj.reward, np.eye(mdp.n_states)[traj.next_state], traj.done,
                                  traj.traj_id, traj.t, traj.state, traj.next_state)
    fn = lambda obs: pi_b[obs.argmax(axis=1)]  # noqa: E731
    w, v = ope.fqe_linear(t2, fn, A, n_iter=20, ridge=1e-2)
    assert np.isfinite(v) and abs(v - mdp.true_value(pi_b)) < 0.3


def test_cql_lite_respects_mask_and_improves(mdp_and_data):
    mdp, pi_b, traj = mdp_and_data
    S, A = mdp.n_states, mdp.n_actions
    mask = np.ones((S, A), bool)
    mask[:, A - 1] = False  # forbid the most aggressive action everywhere
    Q = policies.cql_tabular(traj, S, A, gamma=1.0, alpha=0.5, lr=0.05, n_epochs=3, action_mask=mask)
    pi = policies.greedy_policy(Q, action_mask=mask)
    assert Q.shape == (S, A) and pi[:, A - 1].sum() == 0
    pi_safe = policies.mix_with_behavior(pi, pi_b, 0.3)
    assert mdp.true_value(pi_safe) > mdp.true_value(pi_b) - 0.2  # conservative policy is not catastrophic


def test_ardsnet_constraints():
    lo, hi = constraints.peep_allowed(0.5, "low", tolerance=0.0)
    assert (lo, hi) == (8.0, 10.0)
    lo_h, hi_h = constraints.peep_allowed(0.3, "high", tolerance=0.0)
    assert (lo_h, hi_h) == (5.0, 14.0)
    assert constraints.check_setting(vt_per_kg=6.0, peep=8.0, fio2=0.5, plateau=25, ph=7.35, spo2=0.92) == []
    v = constraints.check_setting(vt_per_kg=10.0, peep=20.0, fio2=0.4, plateau=35)
    assert any("VT" in s for s in v) and any("Pplat" in s for s in v) and any("PEEP" in s for s in v)
    grid = mdp_builder.action_grid()
    m = constraints.action_mask(grid)
    assert m.sum() > 0 and not m[grid["vt_bin"] == 2].any()          # VT > 8 mL/kg never allowed
    m2 = constraints.action_mask(grid, state={"spo2": 0.99, "pao2": 120.0, "plateau_pressure": 20.0})
    assert m2.sum() <= m.sum() and not m2[(grid["fio2_bin"] == 2)].any()
    df = pd.DataFrame({"vt_per_kg": [6, 9], "peep": [8, 8], "fio2": [0.5, 0.5], "plateau_pressure": [25, 25]})
    c = constraints.compliance_rate(df)
    assert c["compliant"] == pytest.approx(0.5)


def test_audit_and_cross_site(mdp_and_data):
    mdp, pi_b, _ = mdp_and_data
    targets = reliability.candidate_policies(mdp, pi_b, n=4)
    df = reliability.audit(mdp, pi_b, targets, n_traj=300, n_rep=2, n_boot=0)
    summ = reliability.summarize_audit(df)
    assert {"WIS", "PDIS", "FQE", "DR"} <= set(summ.index) and np.isfinite(summ["rmse"]).all()
    assert "rank_rho" in summ
    mdp2, pi_b2 = reliability.site_shift(mdp, pi_b, strength=0.2)
    trajA = mdp.sample(pi_b, 300, seed=5)
    trajB = mdp2.sample(pi_b2, 300, seed=6)
    out = reliability.cross_site_protocol(trajA, trajB, mdp.n_states, mdp.n_actions)
    assert set(out["errors"]) >= {"WIS", "FQE", "DR"} and np.isfinite(out["truth_B"])
    cal = reliability.calibrate_from_data(trajA, mdp.n_states, mdp.n_actions, horizon=20)
    assert np.isfinite(cal.true_value(pi_b))
