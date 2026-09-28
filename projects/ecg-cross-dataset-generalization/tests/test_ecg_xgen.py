"""Synthetic-data tests for ecg_xgen core functions (no dataset downloads needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_xgen import labels, loaders, metrics, models, shift  # noqa: E402


def synthetic_ecg(n: int, fs: int = 500, seconds: float = 10.0, hr: float = 70.0, noise: float = 0.02,
                  seed: int = 0, wide_qrs: bool = False) -> np.ndarray:
    """Crude 12-lead ECG: Gaussian QRS pulses at a given heart rate plus noise. Shape (n, 12, samples)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    X = np.zeros((n, 12, t.size), dtype=np.float32)
    for i in range(n):
        rr = 60.0 / (hr + rng.normal(0, 3))
        width = 0.05 if wide_qrs else 0.02
        beats = np.arange(0.2, seconds, rr)
        sig = sum(np.exp(-0.5 * ((t - b) / width) ** 2) for b in beats)
        for lead in range(12):
            X[i, lead] = (0.5 + 0.1 * lead / 12) * sig + noise * rng.normal(size=t.size)
    return X


def test_harmonize_leads_reconstructs_limb_leads():
    rng = np.random.default_rng(1)
    I, II = rng.normal(size=100), rng.normal(size=100)
    x = np.stack([I, II] + [rng.normal(size=100) for _ in range(6)])
    names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    out = loaders.harmonize_leads(x, names)
    assert out.shape == (12, 100)
    np.testing.assert_allclose(out[2], II - I, atol=1e-5)          # III
    np.testing.assert_allclose(out[3], -(I + II) / 2, atol=1e-5)   # aVR
    assert loaders.normalize_lead_name("DII") == "II"


def test_standardize_record_resamples_400_to_500():
    x = np.random.default_rng(0).normal(size=(12, 4000)).astype(np.float32)  # 10 s at 400 Hz
    y = loaders.standardize_record(x, fs=400, lead_names=loaders.STANDARD_LEADS)
    assert y.shape == (12, 5000)
    z = loaders.crop_or_pad(x, 6000)
    assert z.shape[-1] == 6000 and np.all(z[:, 4000:] == 0)


def test_parse_wfdb_header(tmp_path):
    hea = tmp_path / "A0001.hea"
    hea.write_text("A0001 12 500 5000\n" + "\n".join(f"A0001.mat 16 1000 0 0 0 0 0 {ld}" for ld in loaders.STANDARD_LEADS)
                   + "\n#Age: 64\n#Sex: Female\n#Dx: 164889003,59118001\n")
    meta = loaders.parse_wfdb_header(hea)
    assert meta.fs == 500 and meta.n_samples == 5000 and meta.age == 64 and meta.sex == "F"
    assert meta.dx == ["164889003", "59118001"]
    assert meta.lead_names[:3] == ["I", "II", "III"]


def test_label_policies_differ_on_incomplete_rbbb():
    strict, lenient, sup = labels.get_policy("strict"), labels.get_policy("lenient"), labels.get_policy("superclass")
    codes = ["713426002", "426783006"]  # incomplete RBBB + sinus rhythm
    ys, yl = labels.harmonize(codes, strict), labels.harmonize(codes, lenient)
    assert ys[strict.index("NORM")] == 1 and ys[strict.index("RBBB")] == 0
    assert yl[lenient.index("RBBB")] == 1 and yl[lenient.index("NORM")] == 0
    ysup = labels.harmonize(codes, sup)
    assert ysup[sup.index("CONDUCTION")] == 1 and ysup[sup.index("NORMAL")] == 0
    # PTB-XL and CODE-15 vocabularies
    assert labels.harmonize(["AFIB"], strict, source="ptbxl")[strict.index("AF")] == 1
    assert labels.harmonize(labels.code15_row_to_codes({"AF": 1, "RBBB": 0, "normal_ecg": 0}), strict, "code15")[strict.index("AF")] == 1
    kappas = labels.policy_agreement([codes, ["59118001"], ["426783006"]], "snomed", strict, lenient)
    assert kappas["RBBB"] < 1.0 and kappas["AF"] == 1.0 or np.isnan(kappas["AF"])
    mask = labels.agreement_mask([codes, ["59118001"]], "snomed", strict, lenient)
    assert mask.tolist() == [False, True]


def test_handcrafted_features_and_logistic_baseline():
    X_slow = synthetic_ecg(12, hr=55, seed=1)
    X_fast = synthetic_ecg(12, hr=120, seed=2)
    fe = models.HandcraftedFeatures()
    F = fe.transform(np.concatenate([X_slow, X_fast]))
    assert F.shape == (24, len(fe.names))
    hr = F[:, 0]
    assert np.nanmean(hr[:12]) < np.nanmean(hr[12:])
    Y = np.zeros((24, 3), dtype=int)
    Y[12:, 1] = 1  # class 1 = tachycardia; class 0 and 2 have no positives
    pipe = models.fit_logistic_baseline(F, Y)
    P = models.predict_proba_full(pipe, F, 3)
    assert P.shape == (24, 3) and P[:, 0].max() == 0.0
    assert P[12:, 1].mean() > P[:12, 1].mean()


def test_shift_diagnostics_detect_shift():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 5))
    B = rng.normal(size=(200, 5))
    C = rng.normal(loc=1.5, size=(200, 5))
    auc_same = shift.domain_classifier_auc(A, B, n_splits=3)
    auc_diff = shift.domain_classifier_auc(A, C, n_splits=3)
    assert auc_same < 0.65 and auc_diff > 0.9
    assert shift.proxy_a_distance(auc_diff) > shift.proxy_a_distance(auc_same)
    mmd_same, p_same = shift.mmd_rbf(A, B, n_perm=50)
    mmd_diff, p_diff = shift.mmd_rbf(A, C, n_perm=50)
    assert mmd_diff > mmd_same and p_diff < 0.05
    fp = shift.acquisition_fingerprint(synthetic_ecg(3, noise=0.2), fs=500)
    fp2 = shift.acquisition_fingerprint(synthetic_ecg(3, noise=0.0), fs=500)
    assert shift.fingerprint_summary(fp)["noise_floor"] > shift.fingerprint_summary(fp2)["noise_floor"]
    D_s = np.c_[rng.normal(60, 10, 300), rng.integers(0, 2, 300)]
    D_t = np.c_[rng.normal(45, 10, 300), rng.integers(0, 2, 300)]
    w = shift.density_ratio_weights(D_s, D_t)
    assert w.shape == (300,) and abs(w.mean() - 1) < 1e-6
    assert np.average(D_t[:, 0], weights=w) > D_t[:, 0].mean()  # reweighting pulls target age towards source
    assert 0 < shift.effective_sample_size(w) <= 300


def test_metrics_bootstrap_and_noise():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    s = y + rng.normal(0, 1.0, n)
    sex = rng.integers(0, 2, n)
    pid = rng.integers(0, 150, n)
    ci = metrics.bootstrap_metric(y, s, "auroc", n_boot=200, groups=pid)
    assert ci.low <= ci.value <= ci.high and 0.6 < ci.value < 0.95
    sg = metrics.subgroup_auroc(y, s, sex, n_boot=100)
    assert set(sg) == {"0", "1"}
    gap = metrics.parity_gap(y, s, sex, n_boot=100)
    assert 0 <= gap.value < 0.2
    eo = metrics.equalized_odds_gap(y, s, sex, threshold=0.5)
    assert 0 <= eo["tpr_gap"] <= 1
    ns = metrics.noise_sensitivity(y, s, flip_rates=(0.1,), n_sim=20)
    assert ns[0.1][0] < ci.value  # flipping positives lowers observed AUROC
    full, sub = metrics.agreement_restricted_auroc(y, s, rng.random(n) > 0.3, n_boot=50)
    assert np.isfinite(sub.value)
    stab = metrics.rank_stability({"a": s, "b": s + rng.normal(0, 0.1, n), "c": rng.normal(size=n)}, y, n_sim=10)
    assert -1 <= stab <= 1
    assert 0 <= metrics.expected_calibration_error(y, 1 / (1 + np.exp(-s))) <= 1


def test_resnet_forward_shape():
    torch = pytest.importorskip("torch")  # skipped when PyTorch is not installed
    m = models.build_resnet1d(n_classes=4, widths=(8, 16), blocks_per_stage=1)
    out = m(torch.zeros(2, 12, 5000))
    assert out.shape == (2, 4)
