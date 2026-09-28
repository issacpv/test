"""Tests on synthetic sessions: criteria detect planted selective units and control false positives."""
import numpy as np
import pandas as pd
import pytest

from concept_cells.nwb_loader import SessionData, binned_counts, rates_matrix, spike_counts
from concept_cells.prevalence import cluster_bootstrap_ci, cochran_q, criterion_agreement, prevalence_table, wilson_ci
from concept_cells.selectivity import (
    anova_criterion,
    binwise_ranksum_criterion,
    holm_reject,
    kruskal_criterion,
    permutation_pvalue,
    poisson_glm_criterion,
    response_strength_criterion,
    run_multiverse,
    split_half_selectivity,
)
from concept_cells.synthetic import make_synthetic_session, make_units_table


def _unit_inputs(session: SessionData, u: int):
    onsets = session.trials["onset"].to_numpy()
    stim = session.trials["stim_id"].to_numpy()
    st = session.spike_times[u]
    counts = spike_counts(st, onsets, 0.0, 1.0)
    rates = counts / 1.0
    binned = binned_counts(st, onsets, 0.0, 1.0, bin_width=0.25, step=0.25)
    baseline = spike_counts(st, onsets, -1.0, -0.75)  # matched 250 ms baseline bins
    return rates, counts, binned, baseline, stim


def test_spike_counts_basic():
    spikes = np.array([0.1, 0.5, 1.2, 2.9, 3.0, 3.4])
    onsets = np.array([0.0, 3.0])
    assert spike_counts(spikes, onsets, 0.0, 1.0).tolist() == [2, 2]
    assert binned_counts(spikes, onsets, 0.0, 1.0, 0.5, 0.5).shape == (2, 2)


def test_criteria_recover_planted_units():
    out = make_synthetic_session(n_units=30, frac_selective=0.4, gain=4.0, trials_per_stim=15, seed=1)
    sess, truth = out["session"], out["truth"]
    hits = {"anova": [], "kruskal": [], "response_strength": [], "poisson_glm": [], "binwise_ranksum": []}
    for u in range(sess.n_units):
        rates, counts, binned, baseline, stim = _unit_inputs(sess, u)
        res = run_multiverse(rates, counts, binned, baseline, stim)
        for k, r in res.items():
            hits[k].append(r.selected)
    sel = truth["selective"].to_numpy()
    for k, flags in hits.items():
        flags = np.array(flags)
        sensitivity = flags[sel].mean()
        fpr = flags[~sel].mean()
        assert sensitivity >= 0.8, f"{k}: sensitivity {sensitivity}"
        assert fpr <= 0.25, f"{k}: false-positive rate {fpr}"
    # preferred stimulus is recovered for the ANOVA-selected planted units
    for u in np.flatnonzero(sel):
        rates, counts, binned, baseline, stim = _unit_inputs(sess, u)
        r = anova_criterion(rates, stim)
        if r.selected:
            assert r.preferred == truth.loc[u, "preferred"]


def test_false_positive_rate_under_null_is_controlled():
    out = make_synthetic_session(n_units=60, frac_selective=0.0, trials_per_stim=12, seed=3)
    sess = out["session"]
    n_sel = 0
    for u in range(sess.n_units):
        rates, _, _, _, stim = _unit_inputs(sess, u)
        n_sel += int(anova_criterion(rates, stim, alpha=0.05).selected)
    assert n_sel <= 9  # 60 units at alpha 0.05 -> expect ~3, allow generous slack


def test_drift_inflates_anova_but_not_glm_in_blocked_design():
    out = make_synthetic_session(n_units=40, frac_selective=0.0, drift_amp=0.6, blocked=True, trials_per_stim=12, seed=7)
    sess = out["session"]
    n_anova = n_glm = 0
    for u in range(sess.n_units):
        rates, counts, _, _, stim = _unit_inputs(sess, u)
        n_anova += int(anova_criterion(rates, stim).selected)
        n_glm += int(poisson_glm_criterion(counts, stim).selected)
    # drift + blocking should push the naive ANOVA far above nominal; the drift-adjusted GLM less so
    assert n_anova > n_glm


def test_circular_null_more_conservative_than_shuffle_under_drift():
    out = make_synthetic_session(n_units=1, frac_selective=0.0, drift_amp=0.8, blocked=True, trials_per_stim=15, seed=11)
    sess = out["session"]
    rates, _, _, _, stim = _unit_inputs(sess, 0)
    p_shuf = permutation_pvalue(anova_criterion, rates, stim, n_perm=200, null="shuffle", seed=0)["p_perm"]
    p_circ = permutation_pvalue(anova_criterion, rates, stim, n_perm=200, null="circular", seed=0)["p_perm"]
    assert 0 < p_shuf <= 1 and 0 < p_circ <= 1
    assert p_circ >= p_shuf - 0.05  # circular shift preserves the drift structure -> not smaller in expectation


def test_split_half_shrinks_selectivity_for_null_units():
    out = make_synthetic_session(n_units=20, frac_selective=0.0, trials_per_stim=10, seed=5)
    sess = out["session"]
    shrink = []
    for u in range(sess.n_units):
        rates, _, _, _, stim = _unit_inputs(sess, u)
        r = split_half_selectivity(rates, stim, n_splits=30, seed=u)
        if np.isfinite(r["shrinkage"]):
            shrink.append(r["shrinkage"])
            assert r["cross_validated"] <= r["in_sample"] + 1e-9
    assert np.mean(shrink) > 0.5  # for non-selective units the in-sample index is pure selection bias


def test_holm_and_binwise():
    assert holm_reject(np.array([0.001, 0.2, 0.04]), 0.05).tolist() == [True, False, False]
    rng = np.random.default_rng(0)
    stim = np.repeat([0, 1, 2], 10)
    binned = rng.poisson(2.0, size=(30, 4))
    binned[stim == 1] += 6
    baseline = rng.poisson(2.0, size=30)
    r = binwise_ranksum_criterion(binned, baseline, stim, alpha=0.05)
    assert r.selected and r.preferred == 1


def test_kruskal_and_response_strength_agree_on_strong_unit():
    rng = np.random.default_rng(2)
    stim = np.repeat(np.arange(5), 12)
    rates = rng.poisson(3.0, size=60).astype(float)
    rates[stim == 3] += 12
    assert kruskal_criterion(rates, stim).selected
    r = response_strength_criterion(rates, stim)
    assert r.selected and r.preferred == 3


def test_prevalence_tools():
    units = make_units_table(n_sessions=6, units_per_session=40, seed=0)
    crits = ["anova", "binwise_ranksum", "response_strength"]
    tab = prevalence_table(units, crits, by=("region",), cluster="session", n_boot=200)
    assert set(tab["criterion"]) == set(crits)
    assert ((tab["boot_low"] <= tab["prevalence"] + 1e-9) & (tab["prevalence"] <= tab["boot_high"] + 1e-9)).all()
    lo, hi = wilson_ci(30, 100)
    assert lo < 0.3 < hi
    cb = cluster_bootstrap_ci(units["anova"].to_numpy(), units["session"].to_numpy(), n_boot=300)
    assert cb["ci_low"] <= cb["estimate"] <= cb["ci_high"]
    kappa, jacc = criterion_agreement(units, crits)
    assert np.allclose(np.diag(kappa), 1.0) and np.allclose(np.diag(jacc), 1.0)
    q = cochran_q([0.3, 0.32, 0.29], [100, 120, 90])
    assert q["p"] > 0.5 and q["I2"] == pytest.approx(0.0, abs=1e-9)
    q2 = cochran_q([0.1, 0.5, 0.3], [200, 200, 200])
    assert q2["p"] < 0.001 and q2["I2"] > 0.8
