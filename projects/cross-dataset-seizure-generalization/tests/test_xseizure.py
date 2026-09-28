"""Synthetic-data tests for the xseizure starter modules (no EEG files needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xseizure import domain_shift, features, io_edf, scoring, splits  # noqa: E402


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_ovlp_basic_counts():
    ref = [(10, 20), (100, 130)]
    hyp = [(12, 18), (300, 310)]
    r = scoring.ovlp_score(ref, hyp, duration_s=3600)
    assert (r.tp, r.fp, r.fn) == (1, 1, 1)
    assert r.sensitivity == pytest.approx(0.5)
    assert r.precision == pytest.approx(0.5)
    assert r.fp_per_24h == pytest.approx(24.0)


def test_taes_fractional_credit():
    ref = [(10, 20)]
    hyp = [(12, 18)]  # covers 60% of the reference, nothing outside
    r = scoring.taes_score(ref, hyp, duration_s=1000)
    assert r.tp == pytest.approx(0.6)
    assert r.fn == pytest.approx(0.4)
    assert r.fp == pytest.approx(0.0)
    # over-extended detection: 10 s outside a 10 s event -> 1.0 fractional FP
    r2 = scoring.taes_score(ref, [(5, 25)], duration_s=1000)
    assert r2.tp == pytest.approx(1.0)
    assert r2.fp == pytest.approx(1.0)
    # unmatched hypothesis counts as one FP
    r3 = scoring.taes_score(ref, [(500, 510)], duration_s=1000)
    assert r3.fp == pytest.approx(1.0) and r3.fn == pytest.approx(1.0)


def test_szcore_tolerance_and_merging():
    ref = [(100, 130)]
    early = [(75, 85)]  # ends 15 s before onset: inside the 30 s pre-onset tolerance
    r = scoring.szcore_event_score(ref, early, duration_s=3600)
    assert r.tp == 1 and r.fp == 0
    too_early = [(40, 60)]
    r = scoring.szcore_event_score(ref, too_early, duration_s=3600)
    assert r.tp == 0 and r.fp == 1 and r.fn == 1
    # two hypotheses 30 s apart are merged into one event -> a single FP
    r = scoring.szcore_event_score([], [(500, 510), (540, 550)], duration_s=3600)
    assert r.fp == 1
    # a 12-minute hypothesis is split into 5-min chunks -> 3 events, 1 TP + 2 FP
    r = scoring.szcore_event_score([(0, 30)], [(0, 720)], duration_s=3600)
    assert r.tp == 1 and r.fp == 2


def test_probabilities_to_events_and_masks():
    p = np.zeros(100)
    p[10:20] = 0.9   # windows 10..19 -> 20 s .. 42 s with 2 s step / 4 s window
    p[22] = 0.9      # isolated short blip -> removed by min duration
    ev = scoring.probabilities_to_events(p, step_s=2.0, win_s=4.0, threshold=0.5,
                                         min_duration_s=5.0, merge_gap_s=1.0)
    assert ev == [(20.0, 42.0)]
    m = scoring.mask_from_events(ev, duration_s=200, fs=1.0)
    assert scoring.events_from_mask(m, fs=1.0) == [(20.0, 42.0)]
    s = scoring.sample_score(m, m, fs=1.0)
    assert s["sensitivity"] == 1.0 and s["specificity"] == 1.0


def test_score_all_keys():
    out = scoring.score_all([(10, 20)], [(12, 18)], duration_s=100)
    assert set(out) == {"ovlp", "taes", "szcore"}
    assert all("f1" in v for v in out.values())


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def test_patient_wise_split_has_no_leakage_and_record_wise_does():
    rng = np.random.default_rng(0)
    subjects = np.repeat([f"chb{i:02d}" for i in range(10)], 40)
    records = np.array([f"{s}_{k // 10}" for k, s in zip(range(400), subjects)])
    y = (rng.random(400) < 0.1).astype(int)
    pw = splits.patient_wise_split(subjects, n_folds=5, seed=1, weights=y)
    assert splits.subject_overlap_fraction(subjects, pw) == 0.0
    assert sum(len(te) for _, te in pw) == 400
    rw = splits.record_wise_split(records, n_folds=5, seed=1)
    assert splits.subject_overlap_fraction(subjects, rw) > 0.5
    ww = splits.window_wise_split(400, n_folds=5)
    assert splits.subject_overlap_fraction(subjects, ww) > 0.9
    lopo = splits.leave_one_patient_out(subjects)
    assert len(lopo) == 10
    with pytest.raises(AssertionError):
        splits.assert_no_subject_leakage(subjects, np.arange(0, 20), np.arange(10, 30))


def test_leakage_inflation_bootstrap():
    out = splits.leakage_inflation([0.9, 0.92, 0.88], [0.7, 0.72, 0.68], n_boot=200)
    assert out["abs"] == pytest.approx(0.2, abs=1e-9)
    assert out["ci_low"] <= out["abs"] <= out["ci_high"]


# --------------------------------------------------------------------------- #
# io / harmonization
# --------------------------------------------------------------------------- #
def test_channel_name_normalization_and_harmonization():
    assert io_edf.normalize_channel_name("EEG FP1-REF") == "FP1"
    assert io_edf.normalize_channel_name("EEG Fp1") == "FP1"
    assert io_edf.normalize_channel_name("T7-P7") == "T3-T5"
    assert io_edf.normalize_channel_name("EEG T3-LE") == "T3"
    rng = np.random.default_rng(0)
    # referential (Siena/TUSZ-style) recording
    ref_names = ["EEG Fp1", "EEG F7", "EEG T3", "EEG Fz", "EEG Cz", "EEG Pz"]
    data = rng.standard_normal((6, 100))
    out, mask = io_edf.harmonize_to_bipolar(data, ref_names)
    pairs = list(io_edf.CANONICAL_BIPOLAR_18)
    assert mask[pairs.index("FP1-F7")] and mask[pairs.index("FZ-CZ")]
    assert not mask[pairs.index("T3-T5")]
    np.testing.assert_allclose(out[pairs.index("FP1-F7")], data[0] - data[1])
    assert np.isnan(out[pairs.index("T3-T5")]).all()
    # CHB-MIT style bipolar with new names and a reversed pair
    bip_names = ["FP1-F7", "P7-T7", "FZ-CZ"]
    data2 = rng.standard_normal((3, 100))
    out2, mask2 = io_edf.harmonize_to_bipolar(data2, bip_names)
    np.testing.assert_allclose(out2[pairs.index("T3-T5")], -data2[1])
    assert mask2.sum() == 3


def test_subject_inference_and_windows():
    assert io_edf.infer_subject_id("data/chbmit/chb21/chb21_03.edf") == "chb01"
    assert io_edf.infer_subject_id("data/siena/PN06/PN06-2.edf") == "PN06"
    assert io_edf.infer_subject_id("data/tusz/edf/train/aaaaaaac/s001_2002/01_tcp_ar/aaaaaaac_s001_t000.edf") == "aaaaaaac"
    assert io_edf.infer_subject_id("data/helsinki/eeg12.edf") == "eeg12"
    b = io_edf.window_bounds(1000, fs=100, win_s=4, step_s=2)
    assert b[0] == (0, 400) and b[-1][1] <= 1000
    x = np.random.default_rng(0).standard_normal((2, 512))
    x[1] = np.nan
    y = io_edf.resample_signal(x, 512, 256)
    assert y.shape == (2, 256) and np.isnan(y[1]).all()
    z = io_edf.bandpass(x, 512, 0.5, 70, notch=50)
    assert z.shape == x.shape and np.isnan(z[1]).all()


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_spectral_and_riemannian_features():
    rng = np.random.default_rng(1)
    fs, n_ch = 256, 4
    W = rng.standard_normal((30, n_ch, 4 * fs))
    X = features.spectral_features(W, fs)
    assert X.shape == (30, len(features.spectral_feature_names(n_ch)))
    assert np.isfinite(X).all()
    T, ref = features.riemannian_features(W)
    assert T.shape == (30, n_ch * (n_ch + 1) // 2)
    # re-centering by the Riemannian mean gives an identity mean (tangent vectors ~ 0)
    covs = features.covariances(W)
    rc = features.recenter(covs, ref)
    np.testing.assert_allclose(features.riemannian_mean(rc), np.eye(n_ch), atol=1e-6)
    assert features.riemannian_distance(ref, ref) == pytest.approx(0.0, abs=1e-8)
    y = np.r_[np.zeros(15), np.ones(15)].astype(int)
    W[15:] *= 3.0  # "seizure" windows with larger amplitude
    Xs = features.spectral_features(W, fs)
    p = features.LogisticBaseline().fit(Xs, y).predict_proba(Xs)
    assert ((p > 0.5).astype(int) == y).mean() > 0.9


def test_window_labels():
    y = features.window_labels(10, win_s=4, step_s=2, events=[(5, 9)])
    # windows: [0,4) [2,6) [4,8) [6,10) [8,12) ...; overlap>=2 s for [4,8) and [6,10)
    assert y.tolist()[:5] == [0, 0, 1, 1, 0]


# --------------------------------------------------------------------------- #
# domain shift
# --------------------------------------------------------------------------- #
def test_mmd_detects_shift():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 5))
    Y = rng.standard_normal((80, 5))
    Z = rng.standard_normal((80, 5)) + 2.0
    same = domain_shift.mmd_permutation_test(X, Y, n_perm=100)
    diff = domain_shift.mmd_permutation_test(X, Z, n_perm=100)
    assert diff["mmd2"] > same["mmd2"]
    assert diff["p_value"] < 0.05 < same["p_value"] + 0.5  # same-dist p not tiny


def test_psd_and_covariance_shift():
    rng = np.random.default_rng(3)
    fs = 256
    A = rng.standard_normal((20, 3, 2 * fs))
    B = rng.standard_normal((20, 3, 2 * fs))
    t = np.arange(2 * fs) / fs
    B += 3 * np.sin(2 * np.pi * 50 * t)  # add 50 Hz line noise to cohort B
    f, pa = domain_shift.cohort_psd(A, fs)
    _, pb = domain_shift.cohort_psd(B, fs)
    d = domain_shift.psd_divergence(pa, pb)
    assert d["js_divergence_bits"].shape == (3,) and (d["js_divergence_bits"] > 0).all()
    cs = domain_shift.covariance_shift(A, B)
    assert cs["between"] > 0
    rep = domain_shift.shift_report(A.reshape(20, -1)[:, :50], B.reshape(20, -1)[:, :50], A, B, fs, n_perm=20)
    assert "mmd_mmd2" in rep and "cov_between" in rep
