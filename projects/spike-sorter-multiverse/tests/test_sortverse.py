"""Synthetic-data tests for sortverse."""
import numpy as np
import pandas as pd
import pytest

from sortverse import agreement as ag
from sortverse import downstream as ds
from sortverse import multiverse as mv
from sortverse import raw_io as rio


def test_byte_range_and_segment_roundtrip(tmp_path):
    b0, b1 = rio.byte_range_for_window(1.0, 2.0)
    assert b0 == 30000 * 384 * 2
    assert (b1 - b0 + 1) == 30000 * 384 * 2
    rng = np.random.default_rng(0)
    x = rio.synthetic_recording(0.5, 1000.0, 8, rng)
    p = rio.write_binary_segment(x, tmp_path / "seg.dat")
    y = rio.read_segment(p, 0.1, 0.3, fs=1000.0, n_channels=8)
    assert y.shape == (200, 8)
    assert np.array_equal(y, x[100:300])
    uv = rio.to_microvolts(y)
    assert uv.dtype == np.float32
    xg, yg = rio.neuropixels1_geometry()
    assert xg.shape == (384,) and yg.max() == 20.0 * 191
    with pytest.raises(ValueError):
        rio.read_segment(p, 0.4, 0.9, fs=1000.0, n_channels=8)


def test_agreement_matching_and_consensus():
    rng = np.random.default_rng(1)
    T = 60.0
    truth = {u: np.sort(rng.uniform(0, T, size=int(rng.integers(300, 600)))) for u in range(5)}
    sorter_b = ag.perturb_sorting(truth, rng, jitter_s=0.05e-3, drop_frac=0.1, add_noise_units=2, duration_s=T)
    M = ag.agreement_matrix(truth, sorter_b)
    pairs = ag.match_units(M, min_agreement=0.5)
    assert len(pairs) == 5
    assert all(pairs["unit_a"] == pairs["unit_b"])
    assert (pairs["agreement"] > 0.8).all()
    # jittered by 5 ms -> agreement collapses
    bad = ag.perturb_sorting(truth, rng, jitter_s=5e-3, drop_frac=0.0)
    assert ag.match_units(ag.agreement_matrix(truth, bad), 0.5).empty
    sorter_c = ag.perturb_sorting(truth, rng, drop_frac=0.2, merge_pairs=[(0, 1)], duration_s=T)
    tab = ag.consensus_table({"A": truth, "B": sorter_b, "C": sorter_c})
    assert tab.loc[(tab.sorter == "B") & (tab.unit.astype(str).str.startswith("noise")), "orphan"].all()
    assert (tab.loc[tab.sorter == "A", "n_sorters"] >= 2).all()


def test_qc_metrics():
    rng = np.random.default_rng(2)
    T = 120.0
    poisson = np.sort(rng.uniform(0, T, 2400))
    refractory = poisson[np.concatenate([[True], np.diff(poisson) > 2e-3])]
    assert ag.isi_violations(refractory, T)["n_violations"] == 0
    assert ag.isi_violations(poisson, T)["isi_violations_ratio"] > 0.5
    assert ag.presence_ratio(poisson, T) == 1.0
    assert ag.presence_ratio(poisson[poisson < T / 2], T) == pytest.approx(0.5, abs=0.02)
    amps = rng.normal(100, 15, 5000)
    assert ag.amplitude_cutoff(amps) < 0.05
    assert ag.amplitude_cutoff(amps[amps > 100]) > 0.3
    qc = ag.qc_table({"a": refractory, "b": poisson}, T, amplitudes={"a": amps, "b": amps[amps > 100]})
    ok = ag.apply_qc(qc, "allen_default")
    assert ok.tolist() == [True, False]
    assert ag.apply_qc(qc, "none").all()


def test_downstream_tuning_and_responsiveness():
    rng = np.random.default_rng(3)
    dirs = np.tile(np.arange(0, 360, 45), 15)
    onsets = np.arange(len(dirs)) * 2.0 + 1.0
    tuned = ds.synthetic_tuned_unit(rng, onsets, dirs, pref_deg=90.0, peak_rate=30.0, base_rate=1.0)
    flat = ds.synthetic_tuned_unit(rng, onsets, dirs, pref_deg=0.0, peak_rate=0.0, base_rate=5.0)
    tc = ds.tuning_curve(tuned, onsets, dirs, (0.0, 0.5))
    sel = ds.orientation_selectivity(tc["mean"].to_numpy(), tc.index.to_numpy())
    assert sel["osi_vec"] > 0.5 and sel["dsi_vec"] > 0.5
    assert abs(sel["pref_dir_deg"] - 90.0) < 30
    tc2 = ds.tuning_curve(flat, onsets, dirs, (0.0, 0.5))
    assert ds.orientation_selectivity(tc2["mean"].to_numpy(), tc2.index.to_numpy())["osi_vec"] < 0.3
    assert ds.responsiveness_permutation(tuned, onsets, n_perm=200)["p"] < 0.01
    assert ds.responsiveness_permutation(flat, onsets, n_perm=200)["p"] > 0.05


def test_drift_and_noise_correlation_and_waveform():
    rng = np.random.default_rng(4)
    A = rng.gamma(2.0, 2.0, size=(30, 20))
    same = ds.representational_drift(A, A + rng.normal(0, 0.1, A.shape))
    shuffled = ds.representational_drift(A, A[rng.permutation(30)])
    assert same["drift_index"] < 0.05 < shuffled["drift_index"]
    half = (A + rng.normal(0, 0.1, A.shape), A + rng.normal(0, 0.1, A.shape))
    ex = ds.representational_drift(A, A[rng.permutation(30)], within_early=half)
    assert ex["excess_drift"] > 0.5
    counts = rng.poisson(5, size=(200, 10)) + np.repeat(rng.poisson(3, size=(200, 1)), 10, axis=1)
    nc = ds.noise_correlations(counts, np.zeros(200, dtype=int))
    assert nc["mean_noise_corr"] > 0.2 and nc["n_pairs"] == 45
    t = np.arange(82) / 30000.0
    wf = -np.exp(-((t - 0.001) ** 2) / (2 * 0.0001 ** 2)) + 0.4 * np.exp(-((t - 0.0016) ** 2) / (2 * 0.0002 ** 2))
    T = np.column_stack([0.2 * wf, wf, 0.5 * wf])
    w = ds.waveform_duration_ms(T, 30000.0)
    assert w["peak_channel"] == 1 and 0.4 < w["trough_to_peak_ms"] < 0.8
    assert ds.narrow_spiking_fraction([0.2, 0.3, 0.6, 0.8]) == 0.5


def test_multiverse_variance_decomposition():
    rng = np.random.default_rng(5)
    grid = mv.specification_grid(sorter=["ks25", "ks4", "sc2"], qc_rule=["allen_default", "lenient"],
                                 mouse=list(range(6)))
    assert len(grid) == 36
    effect = {"ks25": 0.0, "ks4": 0.3, "sc2": 0.6}
    grid["frac_responsive"] = grid["sorter"].map(effect) + rng.normal(0, 0.02, len(grid))
    tab = mv.variance_decomposition(grid, "frac_responsive", ["sorter", "qc_rule", "mouse"])
    assert tab.loc["sorter", "eta2"] > 0.8
    assert tab.loc["qc_rule", "eta2"] < 0.1
    bw = mv.between_vs_within(grid, "frac_responsive", unit_col="mouse", spec_col="sorter")
    assert bw["ratio"] > 1.0
    null = mv.sorter_swap_null(grid, "frac_responsive", n_perm=20, rng=rng)
    assert grid.groupby("sorter")["frac_responsive"].mean().std(ddof=1) > null.max()
    curve = mv.specification_curve(grid, "frac_responsive")
    assert curve["rank"].iloc[-1] == 36
    st = mv.claim_stability(grid, lambda r: r["frac_responsive"] > 0.2)
    assert 0.6 < st["fraction_holds"] < 0.7
