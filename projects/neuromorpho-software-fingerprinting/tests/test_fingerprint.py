"""Synthetic-tree tests: SWC round trip, fingerprint features, detectability, harmonisation."""
import numpy as np
import pandas as pd
import pytest

from nm_fingerprint.fingerprint import archive_level_permutation_null, detectability, feature_subset_contrast, provenance_risk
from nm_fingerprint.harmonize import combat, effect_size_preservation, kbet_rejection_rate
from nm_fingerprint.swc_features import feature_vector, morphometrics, read_swc, resample_swc, sampling_fingerprint, write_swc
from nm_fingerprint.synthetic import SoftwareProfile, make_tree

MANUAL = SoftwareProfile(spacing_mean=3.0, spacing_cv=0.6, precision=2, radius_quantum=0.0, z_quantum=0.5)
AUTO = SoftwareProfile(spacing_mean=1.0, spacing_cv=0.05, precision=4, radius_quantum=0.25, z_quantum=0.0)


def _table(profiles, n_per_profile, n_archives_per_profile, seed=0):
    rows, labels, groups = [], [], []
    k = 0
    for name, prof in profiles.items():
        for a in range(n_archives_per_profile):
            for _ in range(n_per_profile // n_archives_per_profile):
                rows.append(feature_vector(make_tree(prof, seed=seed + k)))
                labels.append(name)
                groups.append(f"{name}_archive{a}")
                k += 1
    return pd.DataFrame(rows), np.array(labels, dtype=object), np.array(groups, dtype=object)


def test_swc_roundtrip(tmp_path):
    tree = make_tree(MANUAL, seed=1)
    path = tmp_path / "t.swc"
    write_swc(tree, path, header="synthetic")
    back = read_swc(path)
    assert back["xyz"].shape == tree["xyz"].shape
    assert np.allclose(back["xyz"], tree["xyz"], atol=1e-5)
    assert np.array_equal(back["parent"], tree["parent"])
    m = morphometrics(back)
    assert m["n_bifurcations"] > 5 and m["total_length"] > 100 and m["n_tips"] == m["n_bifurcations"] + m["n_stems"]


def test_fingerprint_features_reflect_profile():
    f_manual = sampling_fingerprint(make_tree(MANUAL, seed=2))
    f_auto = sampling_fingerprint(make_tree(AUTO, seed=2))
    assert f_auto["spacing_cv"] < f_manual["spacing_cv"]
    assert f_auto["spacing_grid_fraction"] > f_manual["spacing_grid_fraction"]
    assert f_auto["radius_unique_fraction"] < f_manual["radius_unique_fraction"]
    assert f_manual["z_unique_fraction"] < f_auto["z_unique_fraction"]
    assert f_manual["decimals_x"] <= 2 and f_auto["decimals_x"] >= 3


def test_resampling_removes_sampling_signature():
    a = resample_swc(make_tree(MANUAL, seed=3), spacing=1.0, precision=2, drop_radius=True)
    b = resample_swc(make_tree(AUTO, seed=3), spacing=1.0, precision=2, drop_radius=True)
    fa, fb = sampling_fingerprint(a), sampling_fingerprint(b)
    assert abs(fa["spacing_q50"] - fb["spacing_q50"]) < 0.3
    # radii dropped -> a single radius value in both trees (unique fraction ~ 1 / n_nodes)
    assert fa["radius_unique_fraction"] < 0.01 and fb["radius_unique_fraction"] < 0.01
    assert abs(fa["spacing_grid_fraction"] - fb["spacing_grid_fraction"]) < 0.2
    assert morphometrics(a)["n_bifurcations"] == morphometrics(make_tree(MANUAL, seed=3))["n_bifurcations"]


def test_detectability_with_and_without_software_effect():
    X, y, g = _table({"manual": MANUAL, "auto": AUTO}, n_per_profile=32, n_archives_per_profile=4, seed=10)
    res = detectability(X, y, g, n_splits=4)
    assert res["balanced_accuracy"] > 0.9 and res["detectability"] > 0.8
    # same profile under two labels -> near chance with archives held out
    X0, y0, g0 = _table({"labA": MANUAL, "labB": MANUAL}, n_per_profile=32, n_archives_per_profile=4, seed=200)
    res0 = detectability(X0, y0, g0, n_splits=4)
    assert res0["balanced_accuracy"] < 0.7
    sub = feature_subset_contrast(X, y, g, {"sampling": [c for c in X.columns if c.startswith("s_")],
                                            "morphometric": [c for c in X.columns if c.startswith("m_")]}, n_splits=4)
    assert sub.set_index("subset").loc["sampling", "detectability"] > 0.8
    risk = provenance_risk(res["proba"], sorted(set(y)))
    assert (risk["risk_max_proba"] >= 0.5).all()


def test_archive_level_null_is_calibrated():
    X0, y0, g0 = _table({"labA": MANUAL, "labB": MANUAL}, n_per_profile=24, n_archives_per_profile=4, seed=300)
    null = archive_level_permutation_null(X0, y0, g0, n_perm=6, n_splits=4)
    assert null["p_perm"] > 0.05


def test_combat_removes_batch_and_preserves_biology():
    rng = np.random.default_rng(0)
    n, p = 200, 6
    batch = np.repeat(["A", "B"], n // 2)
    bio = np.tile([0, 1], n // 2)
    X = rng.normal(size=(n, p))
    X[batch == "B"] += 2.0          # batch shift
    X[batch == "B"] *= 1.8          # batch scale
    X[bio == 1, 0] += 1.0           # biological contrast on feature 0
    kb_before = kbet_rejection_rate(X, batch, k=20, n_samples=150)["rejection_rate"]
    out = combat(X, batch, covars=bio[:, None])
    Xa = out["X_adj"]
    kb_after = kbet_rejection_rate(Xa, batch, k=20, n_samples=150)["rejection_rate"]
    assert kb_before > 0.8 and kb_after < 0.3
    assert abs(Xa[batch == "A"].mean() - Xa[batch == "B"].mean()) < 0.2
    pres = effect_size_preservation(X, Xa, bio)
    # d = mean(bio==0) - mean(bio==1) is negative by construction; its magnitude must survive harmonisation
    assert abs(pres.loc[0, "d_after"]) > 0.5 and pres.loc[0, "abs_change"] < 0.4
    assert np.sign(pres.loc[0, "d_after"]) == np.sign(pres.loc[0, "d_before"])
