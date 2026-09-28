"""Synthetic-data tests for lamlat."""
import numpy as np
import pandas as pd
import pytest

from lamlat import io_allen, latent, layers, partition


def test_csd_l4_sink_and_layer_assignment():
    rng = np.random.default_rng(0)
    fs, n_ch, spacing = 1250.0, 40, 20.0
    onsets = np.arange(0.5, 20.0, 1.0)
    lfp, depths = layers.synthetic_laminar_lfp(fs, n_ch, spacing, onsets, 21.0, sink_depth_um=400.0, rng=rng)
    ev, t = layers.evoked_average(lfp, fs, onsets)
    csd = layers.compute_csd(ev, spacing, smooth_channels=1.0)
    sink = layers.find_l4_sink(csd, t, depths[1:-1])
    assert abs(sink["depth_um"] - 400.0) < 40.0
    assert 0.02 <= sink["latency_s"] <= 0.08
    lab = layers.assign_layers(np.array([50, 200, 400, 600, 900, 2000]), sink["depth_um"])
    assert lab.tolist() == ["L1", "L2/3", "L4", "L5", "L6", "outside"]
    jit = layers.assign_layers(np.array([50, 200, 400, 600, 900]), 400.0, jitter_um=10.0, rng=rng)
    assert (jit == lab[:5]).mean() >= 0.8
    counts = layers.layer_counts(lab)
    assert counts["L4"] == 1 and sum(counts.values()) == 5


def test_io_allen_depths_and_selection():
    rng = np.random.default_rng(1)
    tabs = io_allen.synthetic_cache_tables(rng)
    surf = io_allen.surface_vertical_position(tabs["channels"], tabs["probes"])
    assert surf.loc[100] == 20 * 35
    u = io_allen.unit_depths_from_surface(tabs["units"], tabs["channels"], tabs["probes"])
    assert (u["depth_um"] >= -100).all() and u["depth_um"].max() <= 20 * 35
    sel = io_allen.select_units(u, session_id=1)
    assert (sel["structure"] == "VISp").all() and (sel["isi_violations"] < 0.5).all()
    order = io_allen.lfp_channel_order(tabs["channels"], 100)
    assert order["probe_vertical_position"].is_monotonic_decreasing


def test_fa_dimensionality_and_overlap_with_nulls():
    rng = np.random.default_rng(2)
    Xa, Xb, Z = latent.synthetic_two_populations(800, 20, 20, k_shared=2, k_private=1, rng=rng)
    cv = latent.fa_cv_dimensionality(Xa, dims=range(1, 6), n_folds=4)
    assert cv.attrs["best_dim"] in (2, 3, 4)
    ov_shared = latent.cross_layer_overlap(Xa, Xb, k=2)
    Xc, _, _ = latent.synthetic_two_populations(800, 20, 20, k_shared=2, k_private=1, rng=rng)  # independent latents
    ov_indep = latent.cross_layer_overlap(Xa, Xc, k=2)
    # populations sharing 2 latents (plus 1 private each) overlap far more than independent populations
    assert ov_shared > 0.45 and ov_indep < 0.2 and (ov_shared - ov_indep) > 0.3
    ceiling = latent.split_unit_ceiling(Xa, k=2, n_rep=2, rng=rng)
    assert ceiling >= ov_shared - 0.2
    A = np.eye(5)[:, :2]
    B = np.eye(5)[:, 2:4]
    assert latent.subspace_overlap(A, A) == pytest.approx(1.0)
    assert latent.subspace_overlap(A, B) == pytest.approx(0.0, abs=1e-9)
    # depth-shuffle null: overlap between a 'layer' and an independent population is lower than null
    X = np.hstack([Xa, Xc])
    labels = np.array(["L4"] * 20 + ["L5"] * 20)
    res = latent.depth_shuffle_null(X, labels, "L4", "L5", lambda P, Q: latent.cross_layer_overlap(P, Q, 2), n_perm=6, rng=rng)
    assert res["observed"] < res["null_mean"]


def test_rrr_rank_recovery():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 15))
    B = rng.normal(size=(15, 2)) @ rng.normal(size=(2, 12))
    Y = X @ B + 0.3 * rng.normal(size=(2000, 12))
    cv = latent.rrr_cv_r2(X, Y, ranks=[0, 1, 2, 3, 4, 6])
    assert cv["cv_r2"].max() > 0.8
    assert latent.communication_dimensionality(cv, 0.95) == 2
    M = latent.spike_count_matrix([rng.uniform(0, 10, 50), rng.uniform(0, 10, 80)], np.linspace(0, 10, 21))
    assert M.shape == (20, 2)


def test_variance_partition_recovers_regressor_identity():
    rng = np.random.default_rng(4)
    Y, groups, labels = partition.synthetic_encoding_data(2000, rng)
    alpha = partition.select_alpha(np.column_stack(list(groups.values())), Y, alphas=(0.1, 1.0, 10.0))
    part = partition.variance_partition(Y, groups, alpha=alpha)
    by = partition.partition_by_layer(part, labels, ["stim", "behav"])
    assert by.loc["stim", "unique_stim"] > 0.5 > by.loc["stim", "unique_behav"]
    assert by.loc["behav", "unique_behav"] > 0.5 > by.loc["behav", "unique_stim"]
    L = partition.lagged_design(np.arange(5.0), [0, 1, -1])
    assert L[1, 1] == 0.0 and L[0, 2] == 1.0
