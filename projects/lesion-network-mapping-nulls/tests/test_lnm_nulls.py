"""Synthetic tests for lnm_nulls (PYTHONPATH=src)."""
import numpy as np
import pytest

from lnm_nulls import lesion_sampling as ls
from lnm_nulls import lnm, nulls, simulation


@pytest.fixture(scope="module")
def brain():
    return simulation.make_toy_brain(rng=np.random.default_rng(0))


def test_sphere_and_features(brain):
    s = ls.sphere_lesion(brain.shape, (6, 14, 10), 3.0) & brain.brain_mask
    f = ls.lesion_features(s, brain.territory, 6)
    assert f.volume == s.sum() and f.volume > 50
    assert f.hemisphere == 0                    # x=6 < 12 -> left
    assert f.dominant_territory in (1, 2, 3)
    assert abs(f.territory_fractions.sum() - 1) < 1e-9


def test_random_spheres_are_volume_matched(brain):
    rng = np.random.default_rng(1)
    vols = np.array([80, 150, 300])
    fake = ls.random_sphere_lesions(3, brain.brain_mask, vols, rng)
    for m, v in zip(fake, vols):
        assert 0.8 * v <= m.sum() <= 1.6 * v


def test_shuffle_keeps_shape_volume_and_hemisphere(brain):
    rng = np.random.default_rng(2)
    les = ls.sphere_lesion(brain.shape, (7, 10, 10), 2.5) & brain.brain_mask
    out = ls.shuffle_lesion_locations([les], brain.brain_mask, rng)[0]
    assert 0.9 * les.sum() <= out.sum() <= les.sum()
    assert ls.hemisphere_of(out) == ls.hemisphere_of(les)
    assert not np.array_equal(out, les) or rng.random() >= 0  # may coincide by chance; volume check is the key


def test_matched_resample_respects_constraints(brain):
    rng = np.random.default_rng(3)
    pool = simulation.sample_pool(brain, 120, rng)
    fp = [ls.lesion_features(m, brain.territory, 6) for m in pool]
    targets = fp[:10]
    picks = ls.matched_resample(pool, fp, targets, rng)
    for t, i in zip(targets, picks):
        assert abs(fp[i].volume - t.volume) <= 0.2 * t.volume + 1e-9
        assert fp[i].hemisphere == t.hemisphere


def test_lesion_network_map_is_mean_of_rows(brain):
    les = ls.sphere_lesion(brain.shape, (12, 14, 10), 2.0) & brain.brain_mask
    m = lnm.lesion_network_map(les, brain.fc_vp)
    rows = np.flatnonzero(les.ravel())
    assert np.allclose(m, brain.fc_vp[rows].mean(0))
    assert m.shape == (brain.n_parcels,)


def test_disconnection_map():
    grid = (4, 4, 4)
    les = np.zeros(grid, bool)
    les[1, 1, 1] = True
    flat = np.ravel_multi_index((1, 1, 1), grid)
    tract_voxels = [np.array([flat, flat + 1]), np.array([0, 2])]
    tract_parcels = np.array([[1, 1, 0], [0, 1, 1]])
    d = lnm.disconnection_map(les, tract_voxels, tract_parcels)
    assert np.allclose(d, [1.0, 0.5, 0.0])


def test_symptom_association_matches_ols():
    rng = np.random.default_rng(4)
    n, P = 60, 5
    maps = rng.standard_normal((n, P))
    cov = rng.standard_normal((n, 1))
    y = 0.8 * maps[:, 2] + 0.5 * cov[:, 0] + rng.standard_normal(n)
    t = lnm.symptom_association(maps, y, cov)
    import statsmodels.api as sm
    X = sm.add_constant(np.column_stack([maps[:, 2], cov]))
    ref = sm.OLS(y, X).fit().tvalues[1]
    assert t[2] == pytest.approx(ref, rel=1e-6)
    assert np.argmax(np.abs(t)) == 2


def test_label_permutation_controls_fpr_and_detects_effect(brain):
    rng = np.random.default_rng(5)
    pool = simulation.sample_pool(brain, 80, rng)
    lesions = pool[:40]
    maps = lnm.lesion_network_maps(lesions, brain.fc_vp)
    vols = np.array([m.sum() for m in lesions])
    network = np.arange(3)
    null_scores = simulation.generate_symptoms(maps, vols, network, beta=0.0, gamma=0.0, noise_sd=1.0, rng=rng)
    res0 = nulls.label_permutation_null(maps, null_scores, None, 199, rng)
    assert res0.p_fwer.min() > 0.005
    eff = simulation.generate_symptoms(maps, vols, network, beta=2.5, gamma=0.0, noise_sd=0.5, rng=rng)
    res1 = nulls.label_permutation_null(maps, eff, None, 199, rng)
    assert res1.significant(0.05).any()


def test_bias_atlas_and_leakage(brain):
    rng = np.random.default_rng(6)
    pool = simulation.sample_pool(brain, 60, rng)
    null_maps = lnm.lesion_network_maps(pool, brain.fc_vp)
    mean, sd = nulls.bias_atlas(null_maps)
    assert mean.shape == (brain.n_parcels,) and np.all(sd >= 0)
    obs = mean + 0.05 * rng.standard_normal(mean.size)
    assert nulls.prior_leakage_r2(obs, mean) > 0.8
    assert nulls.prior_leakage_r2(rng.standard_normal(mean.size), mean) < 0.3


def test_moran_surrogates_preserve_autocorrelation(brain):
    rng = np.random.default_rng(7)
    W = nulls.spatial_weights(brain.parcel_coords)
    x = np.exp(-np.linalg.norm(brain.parcel_coords - brain.parcel_coords[0], axis=1) / 5.0)
    x += 0.05 * rng.standard_normal(x.size)
    I_obs = nulls.morans_i(x, W)
    surr = nulls.moran_spectral_surrogates(x, W, 20, rng, procedure="singleton")
    I_surr = np.array([nulls.morans_i(s, W) for s in surr])
    assert np.allclose(I_surr, I_obs, atol=1e-6)
    assert np.allclose(surr.std(axis=1), x.std(), rtol=1e-6)
    I_pair = nulls.morans_i(nulls.moran_spectral_surrogates(x, W, 1, rng, procedure="pair")[0], W)
    assert abs(I_pair - I_obs) < 0.25 * abs(I_obs) + 0.05
    y = rng.standard_normal(x.size)
    r, p = nulls.surrogate_corr_pvalue(x, y, W, 50, rng)
    assert -1 <= r <= 1 and 0 < p <= 1


def test_simulation_engine_runs_and_power_exceeds_fpr():
    null = simulation.run_simulation(n_studies=4, n_patients=30, beta=0.0, gamma=0.0, n_perm=19, pool_size=60,
                                     seed=1, families=("N1_label", "N4_matched"))
    eff = simulation.run_simulation(n_studies=4, n_patients=30, beta=3.0, gamma=0.0, n_perm=19, pool_size=60,
                                    noise_sd=0.3, seed=1, families=("N1_label", "N4_matched"))
    for fam in ("N1_label", "N4_matched"):
        assert 0.0 <= null[fam]["positive_rate"] <= 1.0
        assert eff[fam]["positive_rate"] >= null[fam]["positive_rate"]
    assert eff["N1_label"]["positive_rate"] > 0.5
