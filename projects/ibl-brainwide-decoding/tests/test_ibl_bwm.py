"""Synthetic tests for ibl_bwm (no network, no IBL/Allen dependencies)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ibl_bwm import binning, decoding, hierarchical, loaders  # noqa: E402


@pytest.fixture(scope="module")
def session():
    return loaders.make_synthetic_session(n_units=24, n_trials=240, encode={"choice": 6.0, "stimulus": 6.0}, seed=0)


def test_synthetic_session_shapes(session):
    assert session.spike_times.ndim == 1 and np.all(np.diff(session.spike_times) >= 0)
    assert set(session.regions(min_units=5)) == {"VISp", "CA1"}
    assert len(session.trials) == 240


def test_bin_spikes_counts_match_manual(session):
    units = session.units_in_region("VISp")
    t = session.trials["stimOn_times"].to_numpy()
    tensor = binning.bin_spikes(session.spike_times, session.spike_clusters, units, t, -0.1, 0.1, 0.02)
    assert tensor.shape == (240, len(units), 10)
    # manual count for trial 0, unit 0
    u = units[0]
    m = (session.spike_clusters == u) & (session.spike_times >= t[0] - 0.1) & (session.spike_times < t[0] + 0.1)
    assert tensor[0, 0].sum() == m.sum()
    # NaN alignment gives zeros
    t2 = t.copy()
    t2[3] = np.nan
    assert binning.bin_spikes(session.spike_times, session.spike_clusters, units, t2, -0.1, 0.1)[3].sum() == 0


def test_targets_from_trials(session):
    tg = binning.targets_from_trials(session.trials)
    assert set(tg) >= {"choice", "stimulus", "contrast", "block", "block_bin", "reward"}
    assert np.nanmax(tg["choice"]) == 1 and np.nanmin(tg["choice"]) == 0
    assert np.isnan(tg["stimulus"][session.trials["contrastLeft"].fillna(session.trials["contrastRight"]) == 0]).all()


def test_decode_choice_above_null(session):
    X, valid = binning.region_features(session, "VISp", "choice")
    y = binning.targets_from_trials(session.trials)["choice"]
    X, y = binning.usable(X, y, valid, min_trials=100)
    res = decoding.decode_cv(X, y, task="classification", n_splits=4, alphas=(1.0, 10.0))
    assert res.score > 0.7
    null = decoding.shuffle_null(X, y, n=15, task="classification", n_splits=4, alphas=(1.0,))
    nc = decoding.null_corrected(res.score, null)
    assert nc["p"] < 0.1 and nc["corrected"] > 0.1
    assert abs(null.mean() - 0.5) < 0.1


def test_pseudo_session_and_imposter():
    rng = np.random.default_rng(0)
    p = loaders.generate_block_prior(400, rng)
    assert set(np.unique(p)) <= {0.2, 0.5, 0.8} and np.all(p[:90] == 0.5)
    lab = decoding.pseudo_block_labels(400, rng)
    assert np.isnan(lab[:90]).all() and set(np.unique(lab[~np.isnan(lab)])) == {0.0, 1.0}
    fn = decoding.imposter_label_fn([np.arange(50.0), np.arange(30.0)], 120)
    y = fn(rng)
    assert y.shape == (120,)
    X = rng.normal(size=(120, 5))
    null = decoding.pseudo_session_null(X, lambda r: (r.random(120) > 0.5).astype(float), n=3, n_splits=3, alphas=(1.0,))
    assert null.shape == (3,)


def test_latents_and_yield_curve(session):
    X, valid = binning.region_features(session, "CA1", "stimulus")
    y = binning.targets_from_trials(session.trials)["stimulus"]
    X, y = binning.usable(X, y, valid, min_trials=50)
    Z = decoding.latent_features(X, n_components=4)
    assert Z.shape == (X.shape[0], 4)
    curve = decoding.unit_yield_curve(X, y, n_units_grid=(2, 4), n_rep=2, n_splits=3, alphas=(1.0,))
    assert set(curve) == {2, 4}


def test_dersimonian_laird_and_fisher():
    dl = hierarchical.dersimonian_laird([0.1, 0.12, 0.09], [0.001, 0.001, 0.001])
    assert 0.09 < dl["effect"] < 0.12 and dl["p"] < 1e-6 and dl["I2"] == 0.0
    dl2 = hierarchical.dersimonian_laird([0.3, -0.3], [0.001, 0.001])
    assert dl2["tau2"] > 0 and dl2["I2"] > 0.9
    assert decoding.fisher_combine([0.01, 0.01]) < 0.01


def test_icc_detects_lab_effect():
    rng = np.random.default_rng(1)
    labs = np.repeat(["a", "b", "c", "d"], 10)
    y_no = rng.normal(0, 1, 40)
    y_lab = y_no + np.repeat([0, 2, 4, 6], 10)
    assert hierarchical.icc_oneway(y_no, labs)["icc"] < 0.3
    assert hierarchical.icc_oneway(y_lab, labs)["icc"] > 0.7
    ci = hierarchical.bootstrap_icc(y_lab, labs, n_boot=50)
    assert ci["icc_lo"] <= ci["icc_hi"]


def test_mixed_model_and_region_table():
    rng = np.random.default_rng(2)
    rows = []
    for region, lab_shift in (("VISp", {"a": 0.0, "b": 0.0, "c": 0.0}), ("CA1", {"a": 0.3, "b": 0.0, "c": 0.0})):
        for lab, shift in lab_shift.items():
            for s in range(4):
                for rep in range(2):
                    base = 0.15 if region == "VISp" else 0.0
                    rows.append({"region": region, "lab": lab, "subject": f"{lab}{s}", "n_units": int(rng.integers(8, 60)),
                                 "corrected": base + shift + rng.normal(0, 0.03), "null_sd": 0.03})
    df = pd.DataFrame(rows)
    mm = hierarchical.mixed_model_icc(df[df.region == "VISp"])
    assert "icc_lab" in mm
    tab = hierarchical.region_table(df, min_sessions=3, min_labs=2)
    assert tab.loc["VISp", "p_pooled"] < 1e-4 and not tab.loc["VISp", "fragile_lolo"]
    assert tab.loc["CA1", "icc_lab"] > tab.loc["VISp", "icc_lab"]
    lolo = hierarchical.leave_one_lab_out(df[df.region == "CA1"].assign(null_var=0.03 ** 2))
    assert lolo["n_labs"] == 3


def test_unit_yield_table(session):
    tab = binning.unit_yield_table(session)
    assert set(tab.index) == {"VISp", "CA1"} and (tab["n_good"] == tab["n_units"]).all()
