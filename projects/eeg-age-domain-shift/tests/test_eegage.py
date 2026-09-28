"""Synthetic-data tests for eegage (no downloads, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegage import ages, drift, features, transfer  # noqa: E402


def test_age_parsing(tmp_path):
    assert ages.parse_tuh_patient_field("X X X Age:47") == 47.0
    assert ages.parse_tuh_patient_field("no age here") is None
    hdr = bytearray(b" " * 256)
    hdr[0:8] = b"0       "
    hdr[8:88] = b"00000123 M 01-JAN-2000 Age:12".ljust(80)
    p = tmp_path / "x.edf"
    p.write_bytes(bytes(hdr))
    assert ages.parse_edf_header_age(p) == 12.0
    info = ages.parse_chbmit_subject_info("Case\tGender\tAge\nchb01\tF\t11\nchb02\tM\t11\n")
    assert info["chb01"]["age"] == 11.0 and info["chb02"]["sex"] == "M"
    si = ages.parse_siena_subject_info("patient_id,age,gender\nPN00,55,M\nPN01,46,M\n")
    assert si["PN01"] == 46.0
    assert ages.age_bin(0.5) == "0-1" and ages.age_bin(30) == "18-40" and ages.age_bin(None) == ""
    assert ages.age_distance(10, 10) == 0 and ages.age_distance(1, 10) > ages.age_distance(40, 50)
    D = ages.bin_distance_matrix(["1-6", "18-40", "65+"])
    assert D.loc["1-6", "65+"] > D.loc["18-40", "65+"]


def synthetic_windows(n: int, fs: float, age: float, seizure: np.ndarray, rng, n_ch: int = 2):
    """EEG-like windows: background 1/f noise + age-dependent dominant rhythm; seizure adds rhythmic 3 Hz power."""
    L = int(4 * fs)
    t = np.arange(L) / fs
    pdr = 4.0 + 6.0 * (1 - np.exp(-age / 5.0))  # 4 Hz infant -> ~10 Hz adult
    X = np.zeros((n, n_ch, L))
    for i in range(n):
        for c in range(n_ch):
            w = np.cumsum(rng.normal(size=L))
            w = (w - w.mean()) / (w.std() + 1e-9)
            x = w + 0.8 * np.sin(2 * np.pi * pdr * t + rng.uniform(0, 2 * np.pi))
            if seizure[i]:
                x += 2.0 * np.sin(2 * np.pi * 3.0 * t) * (0.5 + 0.5 * age / 40)
            X[i, c] = x
    return X


def test_spectral_features_and_normative_scaler():
    rng = np.random.default_rng(0)
    fs = 128.0
    rows, agev = [], []
    for age in [1, 3, 8, 15, 25, 45, 70]:
        X = synthetic_windows(20, fs, age, np.zeros(20, bool), rng)
        rows.append(features.spectral_features(X, fs))
        agev += [age] * 20
    F = np.vstack(rows)
    agev = np.asarray(agev, float)
    assert F.shape == (140, 2 * len(features.PER_CHANNEL_FEATURES))
    names = features.feature_names(2)
    j = names.index("ch0:peak_freq_3_13")
    assert np.corrcoef(F[:, j], np.log1p(agev))[0, 1] > 0.6  # PDR proxy matures with age
    Z = features.AgeNormativeScaler().fit_transform(F, agev)
    assert abs(np.corrcoef(Z[:, j], np.log1p(agev))[0, 1]) < 0.3
    assert np.isfinite(Z).all()
    S = features.per_record_standardize(F, np.repeat(np.arange(7), 20))
    assert np.allclose(S.mean(0), 0, atol=1e-6)


def test_transfer_matrix_and_decay():
    rng = np.random.default_rng(1)
    fs = 128.0
    F, y, g, b = [], [], [], []
    for age, lab in [(3, "1-6"), (25, "18-40"), (70, "65+")]:
        for s in range(6):
            sz = rng.random(24) < 0.4
            X = synthetic_windows(24, fs, age + rng.normal(0, 0.5), sz, rng)
            F.append(features.spectral_features(X, fs))
            y.append(sz.astype(int))
            g += [f"{lab}_s{s}"] * 24
            b += [lab] * 24
    F, y, g, b = np.vstack(F), np.concatenate(y), np.asarray(g), np.asarray(b)
    M = transfer.transfer_matrix(F, y, g, b, n_splits=3)
    assert M.shape == (3, 3) and np.isfinite(M.values).all()
    long = transfer.matrix_to_long(M)
    fit = transfer.fit_decay(long, n_boot=100)
    assert set(["slope_dist", "r2", "slope_dist_ci_lo"]) <= set(fit)
    L = drift.transfer_loss(M)
    assert np.allclose(np.diag(L.values), 0)
    curve = transfer.sample_value_curve(F, y, g, b == "65+", ks=(1, 2), far_multiplier=2, n_rep=2)
    assert {"age_matched", "far_age", "base"} <= set(curve["added"])


def test_drift_utilities():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(150, 5))
    B = rng.normal(size=(150, 5))
    C = rng.normal(size=(150, 5)) + 1.5
    m_same, p_same = drift.mmd_rbf(A, B, n_perm=100)
    m_diff, p_diff = drift.mmd_rbf(A, C, n_perm=100)
    assert m_diff > m_same and p_diff < 0.05 and p_same > 0.05
    assert drift.centroid_distance(A, C) > drift.centroid_distance(A, B) - 0.5
    ages_ = rng.uniform(1, 80, 200)
    groups = np.arange(200) // 4
    E = np.column_stack([np.log1p(ages_) + 0.1 * rng.normal(size=200), rng.normal(size=(200, 4))])
    assert drift.age_probe_r2(E, ages_, groups) > 0.8
    w = drift.age_importance_weights(rng.uniform(1, 80, 500), rng.uniform(1, 10, 200))
    assert w.mean() == pytest.approx(1.0) and w.size == 500
    res = drift.drift_vs_loss([0.1, 0.2, 0.3, 0.4, 0.5], [0.01, 0.03, 0.02, 0.05, 0.06])
    assert res["n"] == 5 and res["rho"] > 0.5
