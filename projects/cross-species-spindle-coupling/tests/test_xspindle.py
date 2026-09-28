"""Synthetic-data tests for xspindle."""
import numpy as np
import pytest

from xspindle import coupling as cp
from xspindle import detect as dt
from xspindle import harmonize as hz
from xspindle import spectrum as sp


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    fs = 200.0
    x = hz.synthetic_nrem(180.0, fs, rng, so_freq=0.8, spindle_freq=13.0, coupling_phase=0.0)
    return x, fs


def test_sigma_peak_is_individualised(synth):
    x, fs = synth
    pk = sp.sigma_peak(x, fs, search_range=(8.0, 18.0))
    assert pk["found"] == 1.0
    assert abs(pk["peak_freq"] - 13.0) < 1.0
    rng = np.random.default_rng(1)
    noise = sp.pink_noise(int(60 * fs), fs, rng)
    assert sp.sigma_peak(noise, fs)["found"] == 0.0


def test_spindle_and_so_detection_counts(synth):
    x, fs = synth
    spin = dt.detect_spindles(x, fs, center_freq=13.0, thresh_percentile=95.0)
    # ~0.25 * 0.8 Hz * 180 s = ~36 spindles inserted; detector should find most of them
    assert 20 <= len(spin) <= 50
    assert (spin["freq_hz"].between(11.0, 15.0)).mean() > 0.8
    assert (spin["n_cycles"] >= 5).all()
    assert spin.attrs["boundary_threshold"] <= spin.attrs["threshold"]
    so = dt.detect_slow_oscillations(x, fs, band=(0.3, 1.5), amp_percentile=50.0)
    assert len(so) > 30
    assert abs(so["duration_s"].median() - 1.25) < 0.25


def test_coupling_phase_recovered(synth):
    x, fs = synth
    spin = dt.detect_spindles(x, fs, center_freq=13.0, thresh_percentile=95.0)
    so = dt.detect_slow_oscillations(x, fs, band=(0.3, 1.5), amp_percentile=50.0)
    phase = cp.so_phase(x, fs, band=(0.3, 1.5))
    coupled = cp.couple_spindles_to_so(spin, so, phase, fs)
    assert len(coupled) > 15
    stats = cp.circular_stats(coupled["phase"].to_numpy())
    assert abs(cp.circular_distance(stats["mean_phase"], 0.0)) < 0.5
    assert stats["mvl"] > 0.7
    assert stats["rayleigh_p"] < 1e-6
    null = cp.surrogate_mvl(phase, stats["n"], n_perm=100, rng=np.random.default_rng(2))
    assert cp.mvl_zscore(stats["mvl"], null) > 5
    # offsets: spindle peaks sit ~half a cycle from the nearest trough (i.e. on the peak)
    assert abs(coupled["offset_cycles"].abs().mean() - 0.5) < 0.15


def test_shifted_coupling_phase_is_detected():
    rng = np.random.default_rng(3)
    fs = 200.0
    x = hz.synthetic_nrem(180.0, fs, rng, coupling_phase=np.pi / 2)
    m = hz.harmonized_metrics(x, fs, hz.HARMONIZED, n_perm=50, rng=rng)
    assert m["sigma_found"] == 1.0
    assert abs(cp.circular_distance(m["mean_phase"], np.pi / 2)) < 0.6
    assert m["mvl"] > 0.6 and m["mvl_z"] > 3


def test_tort_mi_and_surrogates(synth):
    x, fs = synth
    phase = cp.so_phase(x, fs)
    env, _ = dt.spindle_envelope(x, fs, 13.0)
    mi = cp.tort_modulation_index(phase, env)
    rng = np.random.default_rng(4)
    mi_null = cp.tort_modulation_index(phase, rng.permutation(env))
    assert mi > 5 * mi_null
    shifted = cp.shifted_surrogate_phases(np.array([1.0, 2.5, 4.0]), phase, fs, n_perm=10)
    assert shifted.shape == (10,)


def test_multiverse_and_presets(synth):
    x, fs = synth
    grid = hz.multiverse_grid(so_bands=((0.3, 1.5), (0.5, 4.0)), sigma_rules=(None,), percentiles=(95.0,),
                              duration_rules=("cycles", "seconds"))
    assert len(grid) == 4
    df = hz.detector_multiverse(x[: int(90 * fs)], fs, grid, n_perm=20)
    assert len(df) == 4 and df["mvl"].notna().all()
    df["so_band"] = [p.so_band[1] for p in grid]
    df["dur_rule"] = ["cycles" if p.spindle_dur_s is None else "seconds" for p in grid]
    eta = hz.specification_variance(df, "mvl", ["so_band", "dur_rule"])
    assert set(eta.index) == {"so_band", "dur_rule"}
    conv = hz.harmonized_metrics(x[: int(60 * fs)], fs, hz.CONVENTIONAL["human_scalp"], n_perm=10)
    assert conv["preset"] == "human_scalp"


def test_resample_and_crude_nrem(synth):
    x, fs = synth
    y, fs2 = hz.resample_to(x, fs, 100.0)
    assert fs2 == 100.0 and abs(len(y) - len(x) / 2) <= 1
    mask = hz.crude_nrem_mask(x, fs, epoch_s=10.0)
    assert mask.shape == x.shape
