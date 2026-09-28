"""Synthetic tests for morph_ttype (no downloads)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morph_ttype.features import density_map, drop_position, featurize, morphometrics, persistence_summary
from morph_ttype.swc import AXON, BASAL, from_array, synthetic_interneuron
from morph_ttype.taxonomy import harmonise_subclass, marker_tokens, species_of_ttype
from morph_ttype.transfer import coral, leave_dataset_out, permutation_null, transfer, within_dataset_cv


def test_taxonomy_mapping():
    assert harmonise_subclass("Pvalb Tpbg") == "Pvalb"
    assert harmonise_subclass("Sst Calb2 Pdlim5") == "Sst"
    assert harmonise_subclass("Inh L2-4 PVALB WFDC2") == "Pvalb"
    assert harmonise_subclass("Inh L1-2 PAX6 CDH12") == "Sncg"
    assert harmonise_subclass("Inh L1-2 PAX6 CDH12", pax6_target="Lamp5") == "Lamp5"
    assert harmonise_subclass("Sst Chodl") == "other"
    assert harmonise_subclass("Sst Chodl", keep_chodl=True) == "Sst Chodl"
    assert harmonise_subclass("Inh L1 LAMP5 NMBR") == "Lamp5"
    assert harmonise_subclass(None) == "other"
    assert marker_tokens("Inh L1-3 SST CALB1") == ["sst", "calb1"]
    assert species_of_ttype("Inh L1-3 SST CALB1") == "human" and species_of_ttype("Vip Chat") == "mouse"


def test_features_on_known_tree():
    rows = [[1, 1, 0, 0, 0, 5, -1], [2, BASAL, 0, 10, 0, 0.5, 1], [3, BASAL, 0, 20, 0, 0.5, 2],
            [4, BASAL, 10, 20, 0, 0.5, 3], [5, BASAL, -10, 20, 0, 0.5, 3], [6, AXON, 0, -30, 0, 0.3, 1]]
    n = from_array(np.asarray(rows, float))
    f = morphometrics(n, (BASAL,), "dend")
    assert f["dend_total_length"] == pytest.approx(40.0)
    assert f["dend_n_bifurcations"] == 1 and f["dend_n_tips"] == 2 and f["dend_n_stems"] == 1
    assert f["dend_frac_above_soma"] == 0.0  # all dendrite is at positive y (deeper)
    dm = density_map(n, (BASAL,), bins=4)
    assert sum(dm.values()) == pytest.approx(1.0)
    ps = persistence_summary(n, (BASAL,), n_bins=4, max_r=100, prefix="pers_dend")
    assert ps["pers_dend_n_sections"] == 3  # stem section + two daughter sections
    full = featurize(n, soma_depth_norm=0.3, layer_index=2, cortical_thickness_um=1000.0)
    assert full["has_axon"] == 1.0 and "axon_total_length" in full
    assert "soma_depth_norm" not in drop_position(pd.DataFrame([full])).columns


def _synthetic_dataset(rng, n_per_class, size_scale, depth_offsets, name):
    rows, ys, ds = [], [], []
    for cls, (n_stems, depth_scale, off) in {"A": (7, 0.4, depth_offsets[0]), "B": (4, 1.6, depth_offsets[1])}.items():
        for _ in range(n_per_class):
            nrn = synthetic_interneuron(rng, n_stems=n_stems, depth_scale=depth_scale, size_scale=size_scale)
            f = featurize(nrn, soma_depth_norm=off + rng.normal(0, 0.05), normalise=True)
            rows.append(f)
            ys.append(cls)
            ds.append(name)
    return pd.DataFrame(rows), pd.Series(ys), pd.Series(ds)


def test_within_and_transfer_pipeline():
    rng = np.random.default_rng(0)
    Xm, ym, dm = _synthetic_dataset(rng, 30, 1.0, (0.2, 0.6), "mouse")
    Xh, yh, dh = _synthetic_dataset(rng, 20, 1.8, (0.2, 0.6), "human")
    Xh = Xh.reindex(columns=Xm.columns)
    res_m = within_dataset_cv(Xm, ym, n_splits=3)
    assert res_m.balanced_accuracy > 0.7
    tr = transfer(Xm, ym, Xh, yh, align="coral")
    assert tr.n_test == 40 and 0 <= tr.balanced_accuracy <= 1
    tr0 = transfer(Xm, ym, Xh, yh, align="none")
    assert tr0.design == "transfer:none"
    X = pd.concat([Xm, Xh], ignore_index=True)
    y = pd.concat([ym, yh], ignore_index=True)
    d = pd.concat([dm, dh], ignore_index=True)
    ldo = leave_dataset_out(X, y, d)
    assert set(ldo) == {"mouse", "human"}
    obs, null, p = permutation_null(lambda yy, **kw: within_dataset_cv(Xm, yy, n_splits=3), ym, n_perm=5)
    assert 0 < p <= 1 and null.shape == (5,) and obs == pytest.approx(res_m.balanced_accuracy)


def test_coral_matches_target_covariance():
    rng = np.random.default_rng(1)
    Xs = rng.normal(0, 1, (300, 4)) @ np.array([[1, 0.8, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    Xt = rng.normal(0, 1, (300, 4)) @ np.diag([2.0, 0.5, 1.0, 1.0])
    Xa = coral(Xs - Xs.mean(0), Xt - Xt.mean(0), eps=1e-3)
    Ca, Ct = np.cov(Xa, rowvar=False), np.cov(Xt, rowvar=False)
    assert np.abs(Ca - Ct).max() < 0.25
