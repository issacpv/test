"""Synthetic tests for vmb_validation (no downloads)."""
from __future__ import annotations

import numpy as np
import pytest

from vmb_validation.connectome import (Construction, apply, bilateral, edge_density, graph_summary,
                                       homotopic_pairs, multiverse, threshold_density)
from vmb_validation.fc_metrics import fc_similarity, fcd, fcd_ks, fit_gain_over_null, homotopic_similarity
from vmb_validation.models import (HopfParams, apply_kernel, coupling_sweep, gcamp_kernel, linear_fc,
                                   max_stable_coupling, simulate_hopf)
from vmb_validation.nulls import (degree_sequences, distance_preserving_permutation,
                                  rewire_degree_preserving, shuffle_weights)
from vmb_validation.parcellate import parcellate_array, parcellate_frames


def _random_connectome(rng, n=12, density=0.3):
    W = rng.lognormal(0, 1, (n, n)) * (rng.random((n, n)) < density)
    np.fill_diagonal(W, 0)
    return W


def test_construction_and_threshold():
    rng = np.random.default_rng(0)
    W = _random_connectome(rng)
    c = Construction(symmetrise="mean", log=True, density=0.2)
    Wc = apply(W, c)
    assert np.allclose(Wc, Wc.T)
    n_off = W.size - len(W)
    assert edge_density(Wc) <= 0.2 + 2.0 / n_off + 1e-9  # symmetric: rounded up to a whole pair
    assert "sym=mean" in c.name()
    Wt = threshold_density(W, 0.1)  # asymmetric input: exact top-k
    assert (Wt > 0).sum() == min(int(np.ceil(0.1 * n_off)), int((W > 0).sum()))
    B = bilateral(W, 0.5 * W)
    assert B.shape == (24, 24) and np.allclose(B[:12, 12:], 0.5 * W)
    assert homotopic_pairs(12).shape == (12, 2)
    assert len(multiverse()) == 32
    assert graph_summary(W)["n"] == 12


def test_linear_fc_matches_simulation():
    rng = np.random.default_rng(1)
    W = apply(_random_connectome(rng, n=8), Construction(symmetrise="mean"))
    W /= np.abs(np.linalg.eigvals(W)).max()
    G = 0.5 * max_stable_coupling(W)
    fc = linear_fc(W, G)
    assert np.isfinite(fc).all() and np.allclose(np.diag(fc), 1)
    # Euler simulation of the same OU process
    n, dt, T = len(W), 0.01, 200000
    A = -np.eye(n) + G * W
    x = np.zeros(n)
    xs = np.empty((T, n))
    for t in range(T):
        x = x + A @ x * dt + np.sqrt(dt) * rng.normal(size=n)
        xs[t] = x
    fc_sim = np.corrcoef(xs[20000:].T)
    assert fc_similarity(fc, fc_sim) > 0.9
    assert np.isnan(linear_fc(W, 2 * max_stable_coupling(W))).all()


def test_hopf_and_sweep_and_fcd():
    rng = np.random.default_rng(2)
    W = apply(_random_connectome(rng, n=6, density=0.6), Construction(symmetrise="mean"))
    W /= W.max()
    p = HopfParams(duration=60.0, burn_in=10.0, dt=0.1)
    x = simulate_hopf(W, 0.3, p, seed=0)
    assert x.shape == (500, 6) and np.isfinite(x).all()
    y = apply_kernel(x, gcamp_kernel(0.1))
    assert y.shape == x.shape
    fc_emp = np.corrcoef(x.T)
    sw = coupling_sweep(W, fc_emp, [0.05, 0.3], fc_similarity, model="hopf", hopf=p)
    assert np.isfinite(sw["best_score"]) and sw["best_G"] in (0.05, 0.3)
    F = fcd(x, window=100, step=50)
    assert F.shape[0] == F.shape[1] and 0 <= fcd_ks(F, F) < 1e-9
    assert homotopic_similarity(fc_emp, fc_emp, np.array([[0, 3], [1, 4], [2, 5]])) == pytest.approx(1.0)


def test_nulls_preserve_invariants():
    rng = np.random.default_rng(3)
    W = _random_connectome(rng, n=15, density=0.35)
    kin, kout = degree_sequences(W)
    Wr = rewire_degree_preserving(W, rng, n_swaps=200)
    kin2, kout2 = degree_sequences(Wr)
    assert np.array_equal(kin, kin2) and np.array_equal(kout, kout2)
    assert np.isclose(np.sort(W[W > 0]).sum(), np.sort(Wr[Wr > 0]).sum())
    Ws = shuffle_weights(W, rng)
    assert np.array_equal(Ws > 0, W > 0) and np.allclose(np.sort(Ws[Ws > 0]), np.sort(W[W > 0]))
    coords = rng.random((15, 3)) * 1000
    Wd = distance_preserving_permutation(W, coords, rng, n_bins=4)
    assert np.array_equal(Wd > 0, W > 0) and np.allclose(np.sort(Wd[Wd > 0]), np.sort(W[W > 0]))
    res = fit_gain_over_null(0.5, [0.1, 0.2, 0.3])
    assert res["gain"] == pytest.approx(0.3) and res["p"] == pytest.approx(0.25)


def test_parcellation():
    labels = np.zeros((4, 4, 2), dtype=int)
    labels[:2] = 1
    labels[2:] = 2
    T = 10
    data = np.zeros((4, 4, 2, T))
    data[labels == 1] = np.arange(T)
    data[labels == 2] = -np.arange(T)
    ts, ok = parcellate_array(data, labels, [1, 2, 3], min_voxels=1)
    assert ok.tolist() == [True, True, False]
    assert np.allclose(ts[:, 0], np.arange(T)) and np.allclose(ts[:, 1], -np.arange(T)) and np.isnan(ts[:, 2]).all()
    frames = np.random.default_rng(0).random((T, 4, 4))
    ts2, ok2 = parcellate_frames(frames, labels[:, :, 0], [1, 2])
    assert ts2.shape == (T, 2) and ok2.all()
