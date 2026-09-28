"""Synthetic tests for layerfmri_repro (PYTHONPATH=src)."""
import numpy as np
import pytest

from layerfmri_repro import layering, profiles, reproducibility


def test_equidistant_and_equivolume_on_annulus_phantom():
    ph = layering.annulus_phantom(size=161, r_wm=25.0, r_pial=45.0)
    gm = ph["gm"]
    eqd = layering.equidistant_depth(ph["d_wm"], ph["d_pial"])
    eqv = layering.equivolume_depth(ph["d_wm"], ph["d_pial"], ph["curvature"])
    # errors are dominated by half-voxel discretisation of the distance transforms at the boundaries
    err_d = np.abs(eqd[gm] - ph["equidist_true"][gm])
    err_v = np.abs(eqv[gm] - ph["equivol_true"][gm])
    assert np.nanmax(err_d) < 0.08 and np.nanmean(err_d) < 0.04
    assert np.nanmax(err_v) < 0.08 and np.nanmean(err_v) < 0.04
    # gyral crown: alpha = beta * (r + r_wm) / (r_pial + r_wm) <= beta, i.e. equivolume depth values are
    # smaller and the bin boundaries move toward the pial surface (deep layers get thicker)
    assert np.nanmean(eqv[gm] - eqd[gm]) < -0.02
    bins_eqv = layering.assign_bins(eqv, 6)
    bins_eqd = layering.assign_bins(eqd, 6)
    fr_v = layering.bin_volume_fractions(bins_eqv, 6)
    fr_d = layering.bin_volume_fractions(bins_eqd, 6)
    assert fr_v.std() < fr_d.std()                      # equivolume bins have (nearly) equal volume
    # interior bins are within 2 % of 1/6; the two edge bins are under-populated because the discretised
    # distance transforms never reach depth 0 / 1 exactly at the boundary voxels
    assert np.all(np.abs(fr_v[1:-1] - 1 / 6) < 0.02)
    assert fr_d[-1] - fr_d[0] > 0.05                    # equidistant: superficial bins hold more volume on a gyrus
    flat = layering.equivolume_depth(np.array([1.0]), np.array([3.0]), np.array([0.0]))
    assert flat[0] == pytest.approx(0.25)
    sulc = layering.equivolume_depth(np.array([1.0]), np.array([3.0]), np.array([-1 / 30.0]))
    assert sulc[0] > 0.25                                # sulcal fundus: the inequality reverses (alpha >= beta)
    gyr = layering.equivolume_depth(np.array([1.0]), np.array([3.0]), np.array([1 / 30.0]))
    assert gyr[0] < 0.25


def test_profile_extraction_and_shape_features():
    n = 8
    bins = np.repeat(np.arange(1, n + 1), 10)
    truth = profiles.synthetic_profile(n, "double_peak")
    values = truth[bins - 1] + 0.01 * np.random.default_rng(0).standard_normal(bins.size)
    prof = profiles.laminar_profile(values, bins, n)
    assert np.allclose(prof, truth, atol=0.02)
    assert profiles.is_double_peak(prof)
    assert not profiles.is_double_peak(profiles.synthetic_profile(n, "middle"))
    assert profiles.superficial_bias_slope(np.linspace(0, 1, n)) > 0
    assert profiles.contrast_superficial_vs_deep(profiles.synthetic_profile(n, "deep")) < 0
    z = profiles.normalize_profile(prof)
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_draining_model_roundtrip_and_bias():
    n = 10
    true = profiles.synthetic_profile(n, "deep")
    meas = profiles.apply_draining(true, leak=0.4, decay=0.9)
    assert profiles.superficial_bias_slope(meas) > profiles.superficial_bias_slope(true)
    rec = profiles.devein(meas, leak=0.4, decay=0.9)
    assert np.allclose(rec, true, atol=1e-9)
    assert profiles.concordance_ccc(true, true) == pytest.approx(1.0)
    assert profiles.concordance_ccc(true, meas) < profiles.pearson(true, meas) + 1e-9


def test_variance_components_recover_nested_structure():
    rng = np.random.default_rng(1)
    labs, subs, reps = 6, 8, 4
    y, lab, subj = [], [], []
    for L in range(labs):
        lab_eff = rng.normal(0, 1.0)
        for S in range(subs):
            sub_eff = rng.normal(0, 0.5)
            for _ in range(reps):
                y.append(lab_eff + sub_eff + rng.normal(0, 0.3))
                lab.append(f"lab{L}")
                subj.append(f"s{S}")
    y = np.array(y)
    vc = reproducibility.variance_components(y, np.array(lab), np.array(subj), method="moments")
    assert vc.lab > vc.subject > vc.residual
    assert abs(vc.residual - 0.09) < 0.04
    vc2 = reproducibility.variance_components(y, np.array(lab), np.array(subj), method="auto")
    assert vc2.lab > vc2.subject > 0 and vc2.residual > 0
    assert abs(sum(vc2.fractions().values()) - 1) < 1e-9


def test_reproducibility_index_and_icc():
    rng = np.random.default_rng(2)
    n_bins = 10
    base = profiles.synthetic_profile(n_bins, "double_peak")
    good = np.stack([base + 0.05 * rng.standard_normal(n_bins) for _ in range(6)])
    bad = rng.standard_normal((6, n_bins))
    r_good = reproducibility.reproducibility_index(good, n_null=100, rng=rng)
    r_bad = reproducibility.reproducibility_index(bad, n_null=100, rng=rng)
    assert r_good["index"] > 0.8 and r_good["p_value"] < 0.05
    assert abs(r_bad["index"]) < 0.4 and r_bad["p_value"] > 0.05
    sess = np.stack([good[:3], good[3:]], axis=1)          # (3 subjects, 2 sessions, n_bins)
    icc = reproducibility.icc_per_bin(sess + rng.normal(0, 0.01, sess.shape))
    assert icc.shape == (n_bins,)


def test_snr_degradation_curve_is_monotone_and_thresholded():
    rng = np.random.default_rng(3)
    true = profiles.synthetic_profile(8, "double_peak")
    levels = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    acc = reproducibility.snr_degradation_curve(true, levels, profiles.is_double_peak, n_rep=60, rng=rng)
    assert acc[-1] > 0.9 and acc[0] < acc[-1]
    smoothed = np.maximum.accumulate(acc)                   # allow small non-monotonic noise
    assert np.all(np.abs(smoothed - acc) < 0.15)
    t80 = reproducibility.tsnr_for_accuracy(levels, acc, 0.8)
    assert levels[0] <= t80 <= levels[-1]
