"""Synthetic-data tests for tms_target.

The properties checked are the ones the scientific claims depend on:

* Fisher-z averaging and seed-map extraction are correct and order-independent.
* A field-weighted seed reduces to a point seed when all the field is in one parcel.
* Volume-weighted dosing metrics respect element volumes (an unweighted mean would
  be biased toward small mesh elements).
* Dose normalization is exact, because field solutions are linear in output.
* Spin and variogram surrogates preserve the marginal distribution and produce
  calibrated p-values on *independent* smooth maps -- the property that makes them
  worth the cost. A naive parametric test is shown to fail the same check.
* Network dose reduces to plain connectivity under a point field, and the
  cross-validated comparison respects subject grouping.

No network access, no HCP data, and neither nibabel nor meshio required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tms_target.connectome import (  # noqa: E402
    Connectome,
    average_connectomes,
    fisher_z,
    inverse_fisher_z,
    load_parcellated_connectome,
    seed_connectivity,
    structural_to_weights,
)
from tms_target.efield import (  # noqa: E402
    EFieldMap,
    field_weighted_centroid,
    focality,
    normalize_to_motor_threshold,
    parcellate_field,
    roi_dose_metrics,
)
from tms_target.nulls import (  # noqa: E402
    effective_dof,
    random_rotation,
    spatial_correlation_test,
    spin_surrogates,
    variogram,
    variogram_surrogates,
)
from tms_target.scoring import (  # noqa: E402
    compare_normative_vs_individual,
    fit_score_model,
    grouped_cv_predictions,
    linear_weighting,
    network_dose,
    score_targets,
    threshold_weighting,
)


# --------------------------------------------------------------------- fixtures
def sphere_coords(p: int = 120, seed: int = 0) -> np.ndarray:
    """Parcel centroids spread over a sphere of radius 70 mm, both hemispheres."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(p, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * 70.0


def smooth_map(coords: np.ndarray, seed: int = 0, length_scale: float = 40.0) -> np.ndarray:
    """A spatially autocorrelated random map: Gaussian-process draw over coords."""
    rng = np.random.default_rng(seed)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    cov = np.exp(-(d**2) / (2 * length_scale**2)) + 1e-6 * np.eye(len(coords))
    chol = np.linalg.cholesky(cov)
    return chol @ rng.normal(size=len(coords))


def toy_connectome(p: int = 120, seed: int = 0) -> Connectome:
    """A symmetric connectome whose strength decays with distance."""
    coords = sphere_coords(p, seed)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    m = np.exp(-d / 50.0)
    np.fill_diagonal(m, 0.0)
    hemi = np.where(coords[:, 0] >= 0, "L", "R")
    labels = np.array([f"{h}_p{i:03d}" for i, h in enumerate(hemi)])
    return Connectome(m, labels, coords, "functional", hemi)


def toy_field(n: int = 500, seed: int = 0) -> EFieldMap:
    """A focal field blob with non-uniform element volumes."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-70, 70, size=(n, 3))
    centre = np.array([40.0, 20.0, 50.0])
    r = np.linalg.norm(coords - centre, axis=1)
    mag = 120.0 * np.exp(-(r**2) / (2 * 25.0**2))
    volumes = rng.uniform(0.5, 8.0, size=n)
    return EFieldMap(magnitude=mag, coords=coords, volumes=volumes, space="MNI152")


# ------------------------------------------------------------------- connectome
def test_fisher_z_roundtrip_and_clipping() -> None:
    r = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    z = fisher_z(r)
    assert np.all(np.isfinite(z))
    assert np.allclose(inverse_fisher_z(z)[1:4], r[1:4], atol=1e-9)


def test_connectome_validates_shapes() -> None:
    with pytest.raises(ValueError, match="square"):
        Connectome(np.zeros((3, 4)), np.arange(3))
    with pytest.raises(ValueError, match="labels must have length"):
        Connectome(np.zeros((3, 3)), np.arange(2))
    with pytest.raises(ValueError, match="coords must be"):
        Connectome(np.zeros((3, 3)), np.arange(3), coords=np.zeros((3, 2)))


def test_index_of_and_distance_matrix() -> None:
    conn = toy_connectome(20)
    label = str(conn.labels[7])
    assert conn.index_of(label) == 7
    assert conn.index_of(label.lower()) == 7
    with pytest.raises(KeyError, match="not found"):
        conn.index_of("nonexistent_parcel")
    d = conn.distance_matrix()
    assert d.shape == (20, 20)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)


def test_distance_matrix_requires_coords() -> None:
    conn = Connectome(np.zeros((4, 4)), np.arange(4))
    with pytest.raises(ValueError, match="coords are required"):
        conn.distance_matrix()


def test_seed_connectivity_single_seed_is_the_row() -> None:
    conn = toy_connectome(30)
    got = seed_connectivity(conn, 5, exclude_seed=False)
    assert np.allclose(got, conn.matrix[5])
    masked = seed_connectivity(conn, 5, exclude_seed=True)
    assert np.isnan(masked[5])


def test_seed_connectivity_by_label_matches_by_index() -> None:
    conn = toy_connectome(30)
    by_index = seed_connectivity(conn, 11)
    by_label = seed_connectivity(conn, str(conn.labels[11]))
    assert np.allclose(by_index, by_label, equal_nan=True)


def test_weighted_seed_reduces_to_point_seed() -> None:
    """All weight on one parcel must reproduce that parcel's row exactly."""
    conn = toy_connectome(40)
    w = np.zeros(40)
    w[9] = 1.0
    weighted = seed_connectivity(conn, range(40), weights=w, exclude_seed=False)
    assert np.allclose(weighted, conn.matrix[9])


def test_seed_connectivity_validation() -> None:
    conn = toy_connectome(10)
    with pytest.raises(ValueError, match="out of range"):
        seed_connectivity(conn, 99)
    with pytest.raises(ValueError, match="no seed parcels"):
        seed_connectivity(conn, [])
    with pytest.raises(ValueError, match="weights must be"):
        seed_connectivity(conn, [1, 2], weights=np.ones(3))
    with pytest.raises(ValueError, match="must be positive"):
        seed_connectivity(conn, [1, 2], weights=np.zeros(2))


def test_average_connectomes() -> None:
    a = toy_connectome(15, seed=1)
    b = Connectome(a.matrix + 2.0, a.labels, a.coords, a.kind, a.hemisphere)
    avg = average_connectomes([a, b])
    assert np.allclose(avg.matrix, a.matrix + 1.0)
    assert avg.metadata["n_inputs"] == 2
    with pytest.raises(ValueError, match="no connectomes"):
        average_connectomes([])
    mismatched = Connectome(np.zeros((15, 15)), np.arange(15))
    with pytest.raises(ValueError, match="same parcel labels"):
        average_connectomes([a, mismatched])


def test_load_parcellated_connectome_fisher_transforms(tmp_path: Path) -> None:
    r = np.array([[0.0, 0.5], [0.5, 0.0]])
    path = tmp_path / "fc.npy"
    np.save(path, r)
    conn = load_parcellated_connectome(path, hcp_release="S1200")
    assert conn.matrix[0, 1] == pytest.approx(np.arctanh(0.5))
    assert conn.metadata["hcp_release"] == "S1200"
    already = load_parcellated_connectome(path, already_fisher_z=True)
    assert already.matrix[0, 1] == pytest.approx(0.5)
    with pytest.raises(FileNotFoundError):
        load_parcellated_connectome(tmp_path / "missing.npy")


def test_structural_to_weights() -> None:
    counts = np.array([[0.0, 100.0], [0.0, 0.0]])
    w = structural_to_weights(counts)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 0.0)
    assert w.max() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        structural_to_weights(np.array([[0.0, -1.0], [-1.0, 0.0]]))
    with pytest.raises(ValueError, match="square"):
        structural_to_weights(np.zeros((2, 3)))


# ---------------------------------------------------------------------- efield
def test_efield_validation() -> None:
    with pytest.raises(ValueError, match="magnitude is empty"):
        EFieldMap(magnitude=np.array([]))
    with pytest.raises(ValueError, match="coords must be"):
        EFieldMap(magnitude=np.ones(3), coords=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="volumes must have length"):
        EFieldMap(magnitude=np.ones(3), volumes=np.ones(2))
    with pytest.raises(ValueError, match="non-negative"):
        EFieldMap(magnitude=np.ones(2), volumes=np.array([-1.0, 1.0]))


def test_roi_dose_metrics_are_volume_weighted() -> None:
    """Two samples, unequal volumes: the mean must follow the volumes."""
    fm = EFieldMap(magnitude=np.array([10.0, 20.0]), volumes=np.array([9.0, 1.0]))
    m = roi_dose_metrics(fm)
    assert m["mean_Vpm"] == pytest.approx(11.0)  # (10*9 + 20*1)/10
    assert m["volume_mm3"] == pytest.approx(10.0)
    assert m["max_Vpm"] == pytest.approx(20.0)
    unweighted = EFieldMap(magnitude=np.array([10.0, 20.0]))
    assert roi_dose_metrics(unweighted)["mean_Vpm"] == pytest.approx(15.0)


def test_roi_dose_metrics_mask_and_errors() -> None:
    fm = toy_field(200)
    mask = fm.magnitude > np.median(fm.magnitude)
    inside = roi_dose_metrics(fm, mask)
    outside = roi_dose_metrics(fm, ~mask)
    assert inside["mean_Vpm"] > outside["mean_Vpm"]
    assert 0.0 <= inside["fraction_above_half_max"] <= 1.0
    with pytest.raises(ValueError, match="selects no samples"):
        roi_dose_metrics(fm, np.zeros(fm.n_samples, dtype=bool))
    with pytest.raises(ValueError, match="must have length"):
        roi_dose_metrics(fm, np.ones(3, dtype=bool))


def test_focality_shrinks_with_higher_threshold() -> None:
    fm = toy_field(400)
    low = focality(fm, threshold_fraction=0.25)
    high = focality(fm, threshold_fraction=0.75)
    assert high["volume_mm3"] < low["volume_mm3"]
    assert high["threshold_Vpm"] > low["threshold_Vpm"]
    with pytest.raises(ValueError, match="threshold_fraction"):
        focality(fm, threshold_fraction=0.0)


def test_field_weighted_centroid_finds_the_blob() -> None:
    fm = toy_field(2000, seed=2)
    centroid = field_weighted_centroid(fm, power=4.0)
    assert np.linalg.norm(centroid - np.array([40.0, 20.0, 50.0])) < 15.0
    no_coords = EFieldMap(magnitude=np.ones(3))
    with pytest.raises(ValueError, match="requires coords"):
        field_weighted_centroid(no_coords)


def test_scaling_is_exact_and_normalization_hits_the_target() -> None:
    fm = toy_field(300)
    doubled = fm.scaled(2.0)
    assert np.allclose(doubled.magnitude, 2.0 * fm.magnitude)
    m1 = fm.magnitude > np.percentile(fm.magnitude, 90)
    normed = normalize_to_motor_threshold(fm, m1, target_Vpm=100.0, statistic="p99")
    got = roi_dose_metrics(normed, m1, percentiles=(99.0,))["p99_Vpm"]
    assert got == pytest.approx(100.0, rel=1e-8)


def test_normalization_rejects_dead_roi() -> None:
    fm = EFieldMap(magnitude=np.zeros(5))
    with pytest.raises(ValueError, match="cannot normalize"):
        normalize_to_motor_threshold(fm, np.ones(5, dtype=bool))


def test_parcellate_field_statistics() -> None:
    fm = EFieldMap(magnitude=np.array([1.0, 3.0, 10.0]), volumes=np.array([1.0, 1.0, 1.0]))
    labels = np.array([0, 0, 1])
    assert np.allclose(parcellate_field(fm, labels, 2, statistic="mean"), [2.0, 10.0])
    assert np.allclose(parcellate_field(fm, labels, 2, statistic="max"), [3.0, 10.0])
    empty = parcellate_field(fm, labels, 3)
    assert np.isnan(empty[2])
    with pytest.raises(ValueError, match="must have length"):
        parcellate_field(fm, np.array([0]), 2)
    with pytest.raises(ValueError, match="unknown statistic"):
        parcellate_field(fm, labels, 2, statistic="mode")  # type: ignore[arg-type]


def test_unassigned_parcel_labels_are_ignored() -> None:
    fm = EFieldMap(magnitude=np.array([5.0, 50.0]))
    out = parcellate_field(fm, np.array([-1, 0]), 1)
    assert out[0] == pytest.approx(50.0)


# ----------------------------------------------------------------------- nulls
def test_random_rotation_is_orthogonal() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = random_rotation(rng)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
    proper = random_rotation(rng, reflect=False)
    assert np.linalg.det(proper) == pytest.approx(1.0)


def test_spin_surrogates_preserve_the_value_multiset_per_hemisphere() -> None:
    coords = sphere_coords(60, seed=3)
    hemi = np.where(coords[:, 0] >= 0, "L", "R")
    values = smooth_map(coords, seed=4)
    sur = spin_surrogates(values, coords, n_surrogates=8, hemisphere=hemi, seed=1)
    assert sur.shape == (8, 60)
    for s in sur:
        for side in ("L", "R"):
            sel = hemi == side
            # nearest-neighbour matching resamples with replacement, so values
            # must come from the same hemisphere's value set
            assert np.all(np.isin(np.round(s[sel], 10), np.round(values[sel], 10)))


def test_spin_surrogates_validation() -> None:
    coords = sphere_coords(10)
    with pytest.raises(ValueError, match="coords must be"):
        spin_surrogates(np.ones(10), coords[:, :2])
    with pytest.raises(ValueError, match="n_surrogates must be"):
        spin_surrogates(np.ones(10), coords, n_surrogates=0)


def test_variogram_increases_with_distance_for_a_smooth_map() -> None:
    coords = sphere_coords(150, seed=5)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    values = smooth_map(coords, seed=6, length_scale=30.0)
    centres, semi = variogram(values, d, n_bins=10)
    ok = np.isfinite(semi)
    assert ok.sum() >= 5
    # a smooth map's semivariance rises with lag
    assert semi[ok][-1] > semi[ok][0]
    assert np.all(np.diff(centres) > 0)


def test_variogram_shape_validation() -> None:
    with pytest.raises(ValueError, match="distances must be"):
        variogram(np.ones(5), np.ones((4, 4)))


def test_variogram_surrogates_match_moments_and_structure() -> None:
    coords = sphere_coords(120, seed=7)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    values = smooth_map(coords, seed=8, length_scale=35.0)
    sur = variogram_surrogates(values, d, n_surrogates=12, seed=2)
    assert sur.shape == (12, 120)
    # surrogates are rescaled to the target's mean and sd
    assert np.allclose(sur.mean(axis=1), values.mean(), atol=1e-6)
    assert np.allclose(sur.std(axis=1), values.std(), rtol=1e-6)
    # and they retain spatial structure: neighbour correlation well above zero
    _, target_semi = variogram(values, d, n_bins=10)
    _, sur_semi = variogram(sur[0], d, n_bins=10)
    ok = np.isfinite(target_semi) & np.isfinite(sur_semi)
    assert np.corrcoef(target_semi[ok], sur_semi[ok])[0, 1] > 0.8


def test_variogram_surrogates_validation() -> None:
    with pytest.raises(ValueError, match="distances must be"):
        variogram_surrogates(np.ones(5), np.ones((4, 4)))
    with pytest.raises(ValueError, match="at least 3 finite"):
        variogram_surrogates(np.array([np.nan, np.nan, 1.0]), np.zeros((3, 3)))


def test_perfectly_correlated_maps_beat_the_null() -> None:
    coords = sphere_coords(100, seed=9)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    x = smooth_map(coords, seed=10)
    res = spatial_correlation_test(x, 2.0 * x + 1.0, distances=d, n_surrogates=199, seed=3)
    assert res.statistic == pytest.approx(1.0, abs=1e-9)
    assert res.p_value < 0.05
    assert res.n_surrogates == 199


def test_spatial_null_is_more_conservative_than_parametric() -> None:
    """Two independent smooth maps: the spatial null must not reject as readily.

    This is the whole justification for the module. We compare the *median*
    p-value across independent map pairs, which is a stable calibration check at
    modest surrogate counts.
    """
    coords = sphere_coords(120, seed=11)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    spatial_p: list[float] = []
    parametric_p: list[float] = []
    for trial in range(12):
        x = smooth_map(coords, seed=100 + trial, length_scale=45.0)
        y = smooth_map(coords, seed=500 + trial, length_scale=45.0)
        res = spatial_correlation_test(x, y, distances=d, n_surrogates=99, seed=trial)
        spatial_p.append(res.p_value)
        parametric_p.append(res.parametric_p)
    assert np.median(spatial_p) > np.median(parametric_p)
    # and the parametric test is badly miscalibrated on smooth independent maps
    assert np.mean(np.array(parametric_p) < 0.05) > np.mean(np.array(spatial_p) < 0.05)


def test_spin_method_runs_and_reports_inflation() -> None:
    coords = sphere_coords(80, seed=12)
    hemi = np.where(coords[:, 0] >= 0, "L", "R")
    x = smooth_map(coords, seed=13)
    y = smooth_map(coords, seed=14)
    res = spatial_correlation_test(
        x, y, coords=coords, method="spin", hemisphere=hemi, n_surrogates=99, seed=4
    )
    assert res.method == "spin"
    assert 0.0 < res.p_value <= 1.0
    assert np.isfinite(res.inflation_factor())


def test_spatial_correlation_test_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        spatial_correlation_test(np.ones(4), np.ones(5), distances=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="requires coords"):
        spatial_correlation_test(np.ones(4), np.ones(4), method="spin")
    with pytest.raises(ValueError, match="requires distances"):
        spatial_correlation_test(np.ones(4), np.ones(4), method="variogram")


def test_effective_dof_below_parcel_count() -> None:
    coords = sphere_coords(150, seed=15)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    x = smooth_map(coords, seed=16, length_scale=50.0)
    y = smooth_map(coords, seed=17, length_scale=50.0)
    sur = variogram_surrogates(x, d, n_surrogates=60, seed=5)
    dof = effective_dof(x, sur, y)
    assert np.isfinite(dof)
    assert dof < 150, "a smooth 150-parcel map cannot carry 150 independent observations"


# --------------------------------------------------------------------- scoring
def test_weightings() -> None:
    e = np.array([0.0, 30.0, 90.0, np.nan])
    assert np.allclose(linear_weighting(e), [0.0, 30.0, 90.0, 0.0])
    hard = threshold_weighting(60.0)(e)
    assert np.allclose(hard, [0.0, 0.0, 30.0, 0.0])
    soft = threshold_weighting(60.0, steepness=10.0)(e)
    assert soft[1] < soft[2]
    with pytest.raises(ValueError, match="non-negative"):
        threshold_weighting(-1.0)
    with pytest.raises(ValueError, match="steepness"):
        threshold_weighting(60.0, steepness=0.0)


def test_network_dose_reduces_to_a_point_seed() -> None:
    """With all the field in one parcel, network dose is that parcel's row."""
    conn = toy_connectome(50).matrix
    field = np.zeros(50)
    field[7] = 100.0
    nd = network_dose(conn, field, normalize="sum")
    assert np.allclose(nd, conn[7])


def test_network_dose_normalization_semantics() -> None:
    conn = toy_connectome(40, seed=20).matrix
    field = np.abs(smooth_map(sphere_coords(40, 20), seed=21)) * 50.0
    normed = network_dose(conn, field, normalize="sum")
    raw = network_dose(conn, field, normalize="none")
    # 'sum' is scale-invariant under a global rescale; 'none' is not.
    assert np.allclose(normed, network_dose(conn, 3.0 * field, normalize="sum"))
    assert not np.allclose(raw, network_dose(conn, 3.0 * field, normalize="none"))


def test_network_dose_exclude_and_validation() -> None:
    conn = toy_connectome(20, seed=22).matrix
    field = np.ones(20) * 10.0
    excl = np.zeros(20, dtype=bool)
    excl[:10] = True
    out = network_dose(conn, field, exclude=excl)
    assert np.allclose(out, conn[10:].mean(axis=0))
    with pytest.raises(ValueError, match="connectivity must be"):
        network_dose(np.zeros((3, 4)), np.ones(3))
    with pytest.raises(ValueError, match="exclude must have length"):
        network_dose(conn, field, exclude=np.zeros(3, dtype=bool))
    dead = network_dose(conn, np.zeros(20))
    assert np.all(np.isnan(dead))


def test_score_targets_shapes_and_default_site() -> None:
    conn = toy_connectome(60, seed=23)
    field = np.zeros(60)
    field[33] = 95.0
    field[34] = 40.0
    scores = score_targets(conn.matrix, field, labels=conn.labels)
    assert scores.metadata["site_index"] == 33
    x, names = scores.as_feature_matrix()
    assert x.shape == (60, 4)
    assert names[0] == "connectivity_only"
    # the thresholded score drops the subthreshold parcel entirely
    assert np.allclose(scores.network_dose_threshold, conn.matrix[33])


def test_fit_score_model_recovers_a_planted_weighting() -> None:
    rng = np.random.default_rng(31)
    n = 200
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + 0.05 * rng.normal(size=n)
    model = fit_score_model(x, y, ["a", "b", "c"], alpha=1e-6)
    assert model.coefficients[0] > 1.5
    assert model.coefficients[1] < -0.5
    assert abs(model.coefficients[2]) < 0.3
    assert model.predict(x[:5]).shape == (5,)


def test_fit_score_model_validation() -> None:
    x = np.zeros((5, 2))
    with pytest.raises(ValueError, match="rows but"):
        fit_score_model(x, np.zeros(4), ["a", "b"])
    with pytest.raises(ValueError, match="columns but"):
        fit_score_model(x, np.zeros(5), ["a"])
    with pytest.raises(ValueError, match="alpha must be"):
        fit_score_model(x, np.zeros(5), ["a", "b"], alpha=-1.0)
    with pytest.raises(ValueError, match=">=3 complete rows"):
        fit_score_model(np.zeros((2, 2)), np.zeros(2), ["a", "b"])
    with pytest.raises(ValueError, match="expected 2 features"):
        fit_score_model(x, np.arange(5.0), ["a", "b"]).predict(np.zeros((1, 3)))


def test_grouped_cv_generalizes_across_subjects() -> None:
    rng = np.random.default_rng(41)
    n_sub, per = 10, 8
    groups = np.repeat(np.arange(n_sub), per)
    x = rng.normal(size=(n_sub * per, 2))
    subject_offset = rng.normal(size=n_sub)[groups]
    y = 1.5 * x[:, 0] + subject_offset + 0.3 * rng.normal(size=n_sub * per)
    res = grouped_cv_predictions(x, y, groups, ["a", "b"], alpha=1.0)
    assert res["n_groups"] == n_sub
    assert res["n_scored"] == n_sub * per
    assert res["r_spearman"] > 0.4
    assert len(res["fold_coefficients"]) == n_sub


def test_grouped_cv_validation() -> None:
    with pytest.raises(ValueError, match="row mismatch"):
        grouped_cv_predictions(np.zeros((4, 2)), np.zeros(3), np.zeros(4), ["a", "b"])
    with pytest.raises(ValueError, match="at least 2 groups"):
        grouped_cv_predictions(np.zeros((4, 2)), np.zeros(4), np.zeros(4), ["a", "b"])


def test_compare_normative_vs_individual_prefers_the_informative_features() -> None:
    rng = np.random.default_rng(51)
    n_sub, per = 12, 6
    groups = np.repeat(np.arange(n_sub), per)
    n = n_sub * per
    informative = rng.normal(size=(n, 2))
    y = 2.0 * informative[:, 0] + 0.3 * rng.normal(size=n)
    noise = rng.normal(size=(n, 2))
    out = compare_normative_vs_individual(noise, informative, y, groups, ["a", "b"], n_boot=200)
    assert out["delta_r_spearman"] > 0
    assert out["delta_ci_low"] < out["delta_r_spearman"] < out["delta_ci_high"]
    assert out["n_boot_valid"] > 100
    with pytest.raises(ValueError, match="design matrices must match"):
        compare_normative_vs_individual(noise, informative[:, :1], y, groups, ["a", "b"])
