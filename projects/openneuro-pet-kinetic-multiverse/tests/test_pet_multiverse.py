"""Model-recovery and multiverse tests on simulated TACs."""
from __future__ import annotations

import numpy as np
import pytest

from pet_multiverse import models as km
from pet_multiverse import multiverse, reliability, tacs


@pytest.fixture(scope="module")
def timing():
    return tacs.FrameTiming.standard_90min()


@pytest.fixture(scope="module")
def clean_tacs(timing):
    cr = km.simulate_reference_tac(timing)
    ct = km.srtm_forward(cr, timing, r1=1.0, k2=0.12, bp=2.0)
    return cr, ct


def test_frame_timing_from_sidecar():
    ft = tacs.FrameTiming.from_sidecar({"FrameTimesStart": [0, 60, 120], "FrameDuration": [60, 60, 120]})
    assert ft.total_minutes == 4.0 and list(ft.mid) == [0.5, 1.5, 3.0]
    with pytest.raises(ValueError):
        tacs.FrameTiming.from_sidecar({"FrameTimesStart": [0, 0], "FrameDuration": [60, 60]})
    assert tacs.classify_tracer("[11C]raclopride") == "reference_cerebellum"
    assert tacs.classify_tracer("[11C]PBR28") == "arterial_input"
    assert tacs.classify_tracer("mystery") == "unknown"


def test_extract_tacs():
    rng = np.random.default_rng(0)
    pet = rng.random((4, 4, 4, 5))
    lab = np.zeros((4, 4, 4), int)
    lab[:2] = 1
    lab[2:] = 2
    out = tacs.extract_tacs(pet, lab, {"a": [1], "b": [2]})
    assert out.shape == (5, 2)
    assert np.allclose(out["a"].to_numpy(), pet[:2].reshape(-1, 5).mean(0))


def test_srtm_family_recovers_parameters(timing, clean_tacs):
    cr, ct = clean_tacs
    w = km.frame_weights(timing, ct, "duration")
    s = km.fit_srtm(ct, cr, timing, w)
    assert abs(s["bp"] - 2.0) < 0.1 and abs(s["r1"] - 1.0) < 0.05
    k2p = s["k2prime"]
    s2 = km.fit_srtm2(ct, cr, timing, k2p, w)
    assert abs(s2["bp"] - 2.0) < 0.1
    m2 = km.fit_mrtm2(ct, cr, timing, 30.0, k2p, w)
    assert abs(m2["bp"] - 2.0) < 0.15
    lg = km.fit_logan_ref(ct, cr, timing, 30.0, k2p, w)
    assert abs(lg["bp"] - 2.0) < 0.15
    lg_nok2 = km.fit_logan_ref(ct, cr, timing, 40.0, None, w)
    assert np.isfinite(lg_nok2["bp"])
    sv = km.fit_suvr(ct, cr, timing, (60.0, 90.0))
    assert sv["bp"] > 1.0  # ratio methods overestimate BP for non-equilibrium tracers


def test_noise_and_population_k2prime(timing, clean_tacs):
    cr, ct = clean_tacs
    rng = np.random.default_rng(1)
    bps = [km.fit_srtm(km.add_noise(ct, timing, 0.05, rng), km.add_noise(cr, timing, 0.05, rng), timing)["bp"] for _ in range(10)]
    assert abs(np.mean(bps) - 2.0) < 0.2
    k2p = km.population_k2prime({"a": ct, "b": ct}, cr, timing, ["a", "b"])
    assert 0.05 < k2p < 0.3


def test_multiverse_runner_and_summaries():
    long, truth = km.simulate_test_retest_dataset(n_subjects=6, seed=2, noise=0.04)
    specs = multiverse.specification_grid(models=("srtm", "srtm2", "logan_ref", "mrtm2", "suvr"),
                                          t_stars=(30.0,), weightings=("duration",), scan_minutes=(60.0, 90.0))
    assert len(specs) >= 6
    res = multiverse.run_multiverse(long, specs, high_binding_regions=["putamen", "caudate"])
    assert res["bp"].notna().mean() > 0.95
    # SRTM at 90 min should recover the true regional ordering
    srtm90 = res[(res["model"] == "srtm") & (res["scan_minutes"] == 90.0)].groupby("region")["bp"].median()
    assert srtm90["putamen"] > srtm90["caudate"] > srtm90["thalamus"] > srtm90["occipital"]
    curve = multiverse.specification_curve(res)
    assert {"spec", "region", "median"} <= set(curve.columns)
    vd = multiverse.variance_decomposition(res, factors=("subject", "region", "model", "scan_minutes"))
    assert abs(vd["eta_sq"].sum() - 1.0) < 1e-6
    assert vd.set_index("factor").loc["region", "eta_sq"] > vd.set_index("factor").loc["model", "eta_sq"]
    rel = reliability.reliability_by_specification(res)
    assert (rel["icc21"] <= 1.0 + 1e-9).all() and len(rel) > 0
    best = rel[rel["region"] == "putamen"].iloc[0]
    assert best["icc21"] > 0.5
    a, b = rel["spec"].unique()[:2]
    d = reliability.bootstrap_icc_difference(res, a, b, "putamen", n_boot=50)
    assert d["ci_low"] <= d["delta_icc"] + 1e-9 and d["delta_icc"] <= d["ci_high"] + 1e-9


def test_contrast_robustness():
    long, _ = km.simulate_test_retest_dataset(n_subjects=8, seed=3, noise=0.04)
    specs = multiverse.specification_grid(models=("srtm", "suvr"), t_stars=(30.0,), weightings=("duration",), scan_minutes=(90.0,))
    res = multiverse.run_multiverse(long, specs, ["putamen"])
    res["group"] = np.where(res["subject"].str[-2:].astype(int) <= 4, "A", "B")
    tab = multiverse.contrast_robustness(res, "group", "putamen")
    assert set(tab.columns) >= {"spec", "cohen_d", "p", "same_sign_as_median"}
