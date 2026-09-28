"""Synthetic-data tests for spatial_null_audit (no network, no real maps)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_null_audit import audit, claims, nulls  # noqa: E402


@pytest.fixture(scope="module")
def sphere():
    coords = nulls.fibonacci_sphere(120)
    dist = nulls.sphere_distance(coords)
    hemi = np.where(coords[:, 0] < 0, "L", "R")
    return coords, dist, hemi


def test_geometry_helpers(sphere):
    coords, dist, _ = sphere
    assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)
    assert dist.shape == (120, 120) and np.allclose(np.diag(dist), 0.0) and dist.max() <= np.pi + 1e-9
    proj = nulls.project_to_sphere(np.random.default_rng(0).normal(size=(40, 3)) * 30 + 50)
    assert np.allclose(np.linalg.norm(proj, axis=1), 1.0)


def test_grf_is_smooth(sphere):
    _, dist, _ = sphere
    rng = np.random.default_rng(0)
    smooth = nulls.gaussian_random_field(dist, 1.0, rng)
    rough = nulls.gaussian_random_field(dist, 0.02, rng)
    assert nulls.morans_i(smooth, dist) > nulls.morans_i(rough, dist)
    lam = nulls.autocorr_length(smooth, dist)["lambda"]
    assert np.isfinite(lam) and lam > 0.1


def test_spin_surrogates_are_permutations_and_preserve_sa(sphere):
    coords, dist, hemi = sphere
    rng = np.random.default_rng(1)
    x = nulls.gaussian_random_field(dist, 0.8, rng)
    surr, perms = nulls.spin_surrogates(x, coords, 30, hemi=hemi, rng=rng, return_perms=True)
    assert surr.shape == (30, 120)
    for p in perms:
        assert sorted(p.tolist()) == list(range(120))  # one-to-one assignment
    i_x = nulls.morans_i(x, dist)
    i_s = np.mean([nulls.morans_i(s, dist) for s in surr])
    i_naive = np.mean([nulls.morans_i(s, dist) for s in nulls.naive_permutation(x, 30, rng)])
    assert abs(i_s - i_x) < abs(i_naive - i_x)
    nearest = nulls.spin_surrogates(x, coords, 5, hemi=hemi, rng=rng, assignment="nearest")
    assert nearest.shape == (5, 120)
    corrected = nulls.spin_surrogates(x, coords, 10, hemi=hemi, rng=rng, sa_tolerance=0.2, dist=dist)
    assert all(abs(nulls.morans_i(s, dist) - i_x) <= 0.2 for s in corrected)


def test_moran_surrogates_preserve_spectrum(sphere):
    _, dist, _ = sphere
    rng = np.random.default_rng(2)
    x = nulls.gaussian_random_field(dist, 0.6, rng)
    surr = nulls.moran_surrogates(x, dist, 50, rng=rng)
    assert surr.shape == (50, 120)
    assert np.allclose(surr.mean(axis=1), x.mean(), atol=1e-8)
    assert np.allclose(surr.std(axis=1), x.std(), atol=1e-6)
    i_x = nulls.morans_i(x, dist)
    i_s = np.array([nulls.morans_i(s, dist) for s in surr])
    assert abs(i_s.mean() - i_x) < 0.05


def test_variogram_surrogates_match_variogram(sphere):
    _, dist, _ = sphere
    rng = np.random.default_rng(3)
    x = nulls.gaussian_random_field(dist, 0.6, rng)
    surr = nulls.variogram_surrogates(x, dist, 8, rng=rng, n_bandwidths=8)
    assert surr.shape == (8, 120)
    assert np.allclose(np.sort(surr[0]), np.sort(x))  # rank-resampled to the original values
    _, g_x = nulls.variogram(x, dist)
    _, g_s = nulls.variogram(surr[0], dist)
    _, g_n = nulls.variogram(x[rng.permutation(120)], dist)
    m = min(g_x.size, g_s.size, g_n.size)
    assert np.abs(g_s[:m] - g_x[:m]).mean() < np.abs(g_n[:m] - g_x[:m]).mean()


def test_naive_null_is_anticonservative_on_smooth_maps(sphere):
    coords, dist, hemi = sphere
    rng = np.random.default_rng(4)
    fields = nulls.gaussian_random_field(dist, 1.0, rng, n_maps=60)
    p_naive, p_moran = [], []
    for i in range(30):
        res = audit.audit_pair(fields[2 * i], fields[2 * i + 1], coords=coords, dist=dist, hemi=hemi, null_names=("naive", "moran"), n_perm=150, seed=i)
        p_naive.append(res.p_values["naive"])
        p_moran.append(res.p_values["moran"])
    fpr_naive = np.mean(np.array(p_naive) < 0.05)
    fpr_moran = np.mean(np.array(p_moran) < 0.05)
    assert fpr_naive > fpr_moran
    assert fpr_naive >= 0.2  # independent smooth maps look "significant" without a spatial null


def test_audit_pair_detects_true_association_and_records_diagnostics(sphere):
    coords, dist, hemi = sphere
    rng = np.random.default_rng(5)
    x = nulls.gaussian_random_field(dist, 0.5, rng)
    y = x + 0.3 * rng.standard_normal(x.size)
    res = audit.audit_pair(x, y, coords=coords, dist=dist, hemi=hemi, n_perm=200, seed=1)
    assert set(res.p_values) == set(audit.DEFAULT_NULLS)
    assert all(p < 0.05 for p in res.p_values.values())
    assert res.robustness_index() == 1.0 and res.classify(0.01) == "confirmed"
    assert np.isfinite(res.lambda_a) and "sadev_spin" in res.as_dict()
    with pytest.raises(ValueError):
        audit.audit_pair(x, y[:-1])


def test_calibrated_fpr_runs_and_has_cis(sphere):
    coords, dist, hemi = sphere
    df = audit.calibrated_fpr(dist, coords, length_scales=[0.3], n_sims=4, n_perm=30, null_names=("naive", "moran"), hemi=hemi)
    assert set(df["null"]) == {"naive", "moran"} and (df["ci_low"] <= df["fpr"]).all() and (df["fpr"] <= df["ci_high"]).all()


def test_claims_registry_roundtrip(tmp_path):
    df = pd.DataFrame(
        [
            {"claim_id": "a2020_x", "doi": "10.1/x", "year": 2020, "map_a": "neuromaps:a", "map_b": "neuromaps:b", "space": "fsaverage", "parcellation": "schaefer", "n_parcels": 400, "hemispheres": "both", "reported_r": 0.4, "reported_p": 0.01, "reported_stat": "pearson", "null_method": "spin_parcel", "n_perm": 10000, "sa_check_reported": 0, "notes": ""},
            {"claim_id": "b2021_y", "doi": "10.1/y", "year": 2021, "map_a": "neurovault:1", "map_b": "neuromaps:c", "space": "MNI152", "parcellation": "desikan", "n_parcels": 68, "hemispheres": "both", "reported_r": -0.3, "reported_p": 0.2, "reported_stat": "spearman", "null_method": "brainsmash", "n_perm": None, "sa_check_reported": 1, "notes": ""},
        ]
    )
    path = tmp_path / "claims.csv"
    df.to_csv(path, index=False)
    loaded = claims.load_claims(path)
    typed = claims.claims_from_frame(loaded)
    assert typed[0].seed == claims.Claim.from_row(loaded.iloc[0]).seed and typed[1].n_perm is None
    summary = claims.summarize_claims(loaded)
    assert summary["n_claims"] == 2 and summary["by_null_method"]["spin_parcel"] == 1
    bad = df.copy()
    bad.loc[0, "null_method"] = "magic"
    assert claims.validate_claims(bad)
    assert claims.null_robustness_index({"naive": 0.001, "spin": 0.03, "moran": 0.2}) == 0.5
    assert claims.classify_claim(0.01, {"spin": 0.2, "moran": 0.3}) == "flipped"
    assert claims.classify_claim(0.2, {"spin": 0.01}) == "reported_null"
    results = {"a2020_x": audit.AuditResult(0.4, 0.38, 400, p_values={"spin": 0.01, "moran": 0.04})}
    table = audit.flip_table(loaded, results)
    assert len(table) == 1 and table.loc[0, "class"] == "confirmed"
