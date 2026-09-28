"""Synthetic-data tests for tes_dose."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tes_dose import dose_response, efield, openneuro, spatial_nulls


def test_openneuro_parse_and_filter():
    payload = {
        "data": {
            "datasets": {
                "pageInfo": {"hasNextPage": False, "endCursor": "abc"},
                "edges": [
                    {"node": {"id": "ds000001", "name": "Resting state", "latestSnapshot": {
                        "tag": "1.0.0", "readme": "nothing here",
                        "summary": {"modalities": ["MRI"], "subjects": ["01"], "sessions": [], "tasks": ["rest"]},
                        "description": {"Name": "Resting state"}}}},
                    {"node": {"id": "ds000002", "name": "tDCS working memory", "latestSnapshot": {
                        "tag": "1.0.1", "readme": "Anodal tDCS over F3 at 2 mA.",
                        "summary": {"modalities": ["T1w", "EEG"], "subjects": ["01", "02"], "sessions": ["1", "2"], "tasks": []},
                        "description": {"Name": "tDCS working memory"}}}},
                    {"node": {"id": "ds000003", "name": "EEG only", "latestSnapshot": {
                        "tag": "1.0.0", "readme": "Transcranial alternating current stimulation dataset",
                        "summary": {"modalities": ["EEG"], "subjects": ["01"], "sessions": [], "tasks": []},
                        "description": {}}}},
                ],
            }
        }
    }
    recs = openneuro.parse_dataset_nodes(payload)
    assert len(recs) == 3
    hits = openneuro.filter_tes_datasets(recs)
    assert [h.accession for h in hits] == ["ds000002", "ds000003"]
    assert hits[0].has_t1w and not hits[1].has_t1w
    frame = openneuro.registry_frame(hits)
    assert list(frame["n_subjects"]) == [2, 1]
    assert openneuro.page_info(payload) == (False, "abc")


def test_sphere_efield_scaling_and_metrics():
    rng = np.random.default_rng(0)
    pts = efield.shell_points(0.06, 0.075, 2000, rng)
    src = np.array([0.0, 0.0, 1.0])
    snk = np.array([0.0, 1.0, 0.0])
    e1 = efield.sphere_efield(pts, src, snk, current_a=1e-3)
    e2 = efield.sphere_efield(pts, src, snk, current_a=2e-3)
    assert np.allclose(e2, 2 * e1, rtol=1e-6)  # linear in current
    mag = np.linalg.norm(e1, axis=1)
    assert 0.01 < mag.mean() < 2.0  # V/m order of magnitude at 1 mA on a 9 cm sphere
    # field should be strongest near the electrodes
    near = pts @ src / np.linalg.norm(pts, axis=1) > 0.95
    far = np.abs(pts @ src / np.linalg.norm(pts, axis=1)) < 0.2
    assert mag[near].mean() > mag[far].mean()
    gm = np.ones(len(pts), bool)
    roi = near
    m = efield.dose_metrics(e1, gm, roi, current_ma=1.0)
    assert m.roi_mean > m.gm_mean
    assert 0 < m.focality_fraction <= 1
    assert efield.current_for_target(m.roi_mean, 2 * m.roi_mean) == pytest.approx(2.0)
    assert efield.current_for_target(0.01, 1.0) == 4.0  # capped


def test_spin_test_detects_identity_and_rejects_noise():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(400, 3))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    # smooth map: function of latitude
    x = coords[:, 2] + 0.1 * rng.normal(size=400)
    r, p, null = spatial_nulls.spin_test_correlation(x, x, coords, n_perm=200, seed=0)
    assert r == pytest.approx(1.0)
    assert p < 0.05
    assert null.shape == (200,)
    y = rng.normal(size=400)
    _, p2, _ = spatial_nulls.spin_test_correlation(x, y, coords, n_perm=200, seed=0)
    assert p2 > 0.01


def test_montage_null_recovers_true_vertex():
    rng = np.random.default_rng(2)
    n_sub, n_vert = 60, 50
    common = np.abs(rng.normal(size=n_vert))  # montage shape shared by everyone
    gain = rng.lognormal(0, 0.3, size=n_sub)
    maps = gain[:, None] * common[None, :] + 0.05 * rng.normal(size=(n_sub, n_vert))
    effect = 2.0 * maps[:, 7] + 0.5 * rng.normal(size=n_sub)
    r_map, p_fwe, thr = spatial_nulls.montage_null(maps, effect, n_perm=300, seed=0)
    assert r_map.shape == (n_vert,)
    assert p_fwe[7] < 0.05
    assert 0 < thr < 1
    frac = spatial_nulls.survival_fraction(np.full(n_vert, 0.01), p_fwe)
    assert 0 <= frac <= 1


def test_dose_response_and_meta_analysis():
    rng = np.random.default_rng(3)
    n = 80
    dose = rng.lognormal(-1.5, 0.3, size=n)
    age = rng.uniform(20, 70, size=n)
    effect = 3.0 * (dose - dose.mean()) / dose.std() + 0.01 * age + rng.normal(size=n)
    est = dose_response.fit_dose_response(effect, dose, covariates=pd.DataFrame({"age": age}))
    assert est.slope > 2.0 and est.p_value < 1e-6 and est.n == n
    gain = dose * rng.lognormal(0, 0.05, size=n)
    out = dose_response.gain_shape_test(effect, dose, gain)
    assert 0 <= out["p_shape"] <= 1 and out["r2_gain_shape"] >= out["r2_gain"]
    pooled = dose_response.dersimonian_laird(np.array([0.5, 0.7, 0.4]), np.array([0.04, 0.05, 0.03]))
    assert 0.4 < pooled["pooled"] < 0.7 and pooled["k"] == 3 and 0 <= pooled["i2"] <= 1
    assert dose_response.attenuation_corrected_slope(1.0, 0.25) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        dose_response.dersimonian_laird(np.array([1.0]), np.array([0.1]))
