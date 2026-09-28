"""Synthetic tests for echo_uq: Simpson's EF vs analytic ellipsoid, conformal coverage, shift diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echo_uq import conformal, shift, simpson, video  # noqa: E402


def test_synthetic_video_and_clip_sampling():
    echo = video.synthetic_lv_video(n_frames=40, ef=0.6, seed=1)
    assert echo.frames.shape == (40, 112, 112) and echo.masks.shape == (40, 112, 112)
    assert abs(echo.ef_true - 0.6) < 1e-9
    clip = video.sample_clip(echo.frames, length=32, period=2)
    assert clip.shape == (32, 112, 112)
    clips = video.sample_clips(echo.frames, n_clips=3, length=16, period=2)
    assert clips.shape == (3, 16, 112, 112)
    t = video.to_tensor(clip)
    assert t.shape == (3, 32, 112, 112) and t.dtype == np.float32
    d = video.video_descriptors(echo.frames)
    assert 0 < d["sector_fraction"] <= 1 and d["n_frames"] == 40
    D, keys = video.descriptor_matrix([echo.frames, echo.frames])
    assert D.shape == (2, len(keys))


def test_simpson_matches_analytic_ellipsoid():
    a, b = 40.0, 20.0
    mask = video.ellipse_mask(112, 56, 56, a, b, 0.4)
    vol = simpson.disk_volume_single_plane(mask, n_disks=20).volume
    analytic = 4 / 3 * np.pi * a * b ** 2
    assert abs(vol - analytic) / analytic < 0.03
    bi = simpson.disk_volume_biplane(mask, mask, n_disks=20).volume
    assert abs(bi - vol) / vol < 1e-6
    assert abs(simpson.disk_volume_single_plane(mask).length - 2 * a) < 3


def test_ef_from_masks_and_beats():
    echo = video.synthetic_lv_video(n_frames=120, ef=0.55, heart_rate=75, seed=2)
    res = simpson.ef_from_mask_stack(echo.masks)
    assert abs(res["ef"] - echo.ef_true) < 0.04
    res_bi = simpson.ef_from_mask_stack(echo.masks, masks_a2c=echo.masks)
    assert abs(res_bi["ef"] - echo.ef_true) < 0.04
    beats = simpson.beat_to_beat_ef(echo.masks, fps=echo.fps)
    assert beats["n_beats"] >= 1 and abs(beats["ef_mean"] - echo.ef_true) < 0.06
    irregular = video.synthetic_lv_video(n_frames=240, ef=0.55, heart_rate=75, rr_jitter=0.25, seed=3)
    regular = video.synthetic_lv_video(n_frames=240, ef=0.55, heart_rate=75, rr_jitter=0.0, seed=3)
    b_irr, b_reg = simpson.beat_to_beat_ef(irregular.masks, fps=irregular.fps), simpson.beat_to_beat_ef(regular.masks, fps=regular.fps)
    assert b_reg["n_beats"] >= 2 and b_irr["n_beats"] >= 2
    assert b_irr["rr_cv"] > b_reg["rr_cv"]
    q = simpson.mask_quality(echo.masks[0])
    assert 0 < q["area_fraction"] < 1 and q["solidity"] > 0.9 and q["elongation"] > 1.2


def test_split_conformal_coverage():
    rng = np.random.default_rng(0)
    n_cal, n_test = 1000, 5000
    x_cal, x_test = rng.normal(size=n_cal), rng.normal(size=n_test)
    y_cal, y_test = x_cal + rng.normal(0, 1, n_cal), x_test + rng.normal(0, 1, n_test)
    scores = conformal.conformity_scores(y_cal, x_cal)
    q = conformal.split_conformal_quantile(scores, alpha=0.1)
    lo, hi = conformal.predict_interval(x_test, q)
    m = conformal.evaluate_intervals(y_test, lo, hi)
    assert 0.88 <= m.coverage <= 0.92 and m.coverage_low <= m.coverage <= m.coverage_high
    assert abs(m.mean_width - 2 * 1.645) < 0.3
    cc = conformal.conditional_coverage(y_test, lo, hi, (x_test > 0).astype(int))
    assert set(cc) == {"0", "1"}
    bins = conformal.coverage_by_bins(y_test, lo, hi, x_test, n_bins=4)
    assert len(bins) == 4


def test_weighted_conformal_restores_coverage_under_covariate_shift():
    rng = np.random.default_rng(1)
    # heteroscedastic noise: sd grows with x; test distribution shifted to large x
    def sample(n, mu):
        x = rng.normal(mu, 1, n)
        y = x + rng.normal(0, 0.3 + 0.7 * (x > 0.5), n)
        return x, y
    x_cal, y_cal = sample(2000, 0.0)
    x_test, y_test = sample(2000, 1.5)
    scores = conformal.conformity_scores(y_cal, x_cal)
    q = conformal.split_conformal_quantile(scores, 0.1)
    lo, hi = conformal.predict_interval(x_test, q)
    unweighted = conformal.evaluate_intervals(y_test, lo, hi).coverage
    w_cal, w_test = conformal.density_ratio_weights(x_cal[:, None], x_test[:, None])
    qw = conformal.weighted_conformal_quantile(scores, w_cal, w_test, 0.1)
    lo_w, hi_w = conformal.predict_interval(x_test, qw)
    weighted = conformal.evaluate_intervals(y_test, lo_w, hi_w).coverage
    assert unweighted < 0.87
    assert weighted > unweighted + 0.03 and weighted >= 0.87
    ess = shift.effective_sample_size(w_cal)
    assert 0 < ess <= len(w_cal)


def test_width_flags_and_bland_altman():
    rng = np.random.default_rng(2)
    hard = rng.integers(0, 2, 300)
    widths = 5 + 3 * hard + rng.normal(0, 1, 300)
    assert conformal.width_flags_hard_cases(widths, hard) > 0.85
    assert conformal.abstention_rate(widths + 20, widths) > 0.95
    ba = conformal.bland_altman(rng.normal(55, 10, 100), rng.normal(55, 10, 100) + 2)
    assert {"bias", "loa_low", "loa_high", "mae", "rmse"} <= set(ba)


def test_shift_diagnostics():
    rng = np.random.default_rng(3)
    A, B, C = rng.normal(size=(150, 4)), rng.normal(size=(150, 4)), rng.normal(1.5, 1, size=(150, 4))
    assert shift.domain_classifier_auc(A, B, n_splits=3) < 0.65
    assert shift.domain_classifier_auc(A, C, n_splits=3) > 0.9
    mmd_same, _ = shift.mmd_rbf(A, B, n_perm=30)
    mmd_diff, p = shift.mmd_rbf(A, C, n_perm=30)
    assert mmd_diff > mmd_same and p < 0.05
    card = shift.shift_card(A, C, ["a", "b", "c", "d"])
    assert (card["cohens_d"] > 0.8).all()
    summ = shift.shift_summary(A, C, ["a", "b", "c", "d"], w_cal=np.ones(150))
    assert summ["ess_fraction"] == 1.0 and summ["proxy_a_distance"] > 1.0
