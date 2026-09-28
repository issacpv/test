"""Synthetic-data tests for wmh_amyloid (run with PYTHONPATH=src)."""

import numpy as np
import pandas as pd
import pytest

from wmh_amyloid import (
    additive_interaction,
    amyloid_positive,
    bootstrap_additive_interaction,
    build_longitudinal_table,
    dice,
    discrete_time_hazard,
    fit_interaction_mixed_model,
    lobar_volumes,
    match_nearest,
    normalize_wmh,
    parse_oasis_id,
    periventricular_deep_split,
    person_period,
    progression_events,
    simulate_wmh_cohort,
    volume_agreement,
    wmh_volume_ml,
)


def test_ids_and_amyloid_status():
    assert parse_oasis_id("OAS30001_MR_d0129") == ("OAS30001", "MR", 129)
    assert parse_oasis_id("OAS30001_PIB_PUPTIMECOURSE_d2430") == ("OAS30001", "PIB_PUPTIMECOURSE", 2430)
    with pytest.raises(ValueError):
        parse_oasis_id("sub-01")
    st = amyloid_positive([10.0, 18.0, 18.0, 25.0, np.nan], ["PIB", "PIB", "AV45", "AV45", "PIB"])
    assert list(st[:4]) == [0.0, 1.0, 0.0, 1.0] and np.isnan(st[4])


def test_matching_and_table():
    wmh = pd.DataFrame({"subject": ["A", "A", "B"], "day": [0, 400, 10], "wmh_ml": [2.0, 3.0, 10.0], "icv_ml": [1500, 1500, 1400]})
    pet = pd.DataFrame({"subject": ["A", "B"], "day": [30, 900], "tracer": ["PIB", "AV45"], "centiloid": [30.0, 5.0]})
    clin = pd.DataFrame({"subject": ["A", "A", "B"], "day": [5, 380, 0], "cdr": [0.0, 0.5, 0.0], "sumbox": [0, 1.0, 0], "mmse": [30, 28, 29]})
    demo = pd.DataFrame({"subject": ["A", "B"], "age_at_entry": [70.0, 65.0], "sex": [1, 0], "education": [16, 12], "apoe4": [1, 0]})
    m = match_nearest(wmh, pet, window_days=365, suffix="_pet")
    assert m.loc[0, "centiloid_pet"] == 30.0 and np.isnan(m.loc[2, "centiloid_pet"])  # B's PET is 890 days away
    df = build_longitudinal_table(wmh, pet, clin, demo)
    assert df.loc[2, "centiloid_pet"] == 5.0  # baseline PET carried
    assert df.loc[0, "amyloid_pos"] == 1.0 and df.loc[2, "amyloid_pos"] == 0.0
    assert df.loc[1, "cdr_clin"] == 0.5 and df.loc[1, "time"] == pytest.approx(400 / 365.25)
    assert df.loc[0, "age"] == pytest.approx(70.0)
    ev = progression_events(clin)
    assert ev.set_index("subject").loc["A", "event"] == 1 and ev.set_index("subject").loc["B", "event"] == 0


def test_wmh_features():
    shape = (20, 20, 20)
    vent = np.zeros(shape, bool)
    vent[8:12, 8:12, 8:12] = True
    wmh = np.zeros(shape, bool)
    wmh[12:14, 8:12, 8:12] = True   # adjacent to ventricles -> periventricular
    wmh[0:2, 0:2, 0:2] = True       # far corner -> deep
    pv, deep, vols = periventricular_deep_split(wmh, vent, voxel_size_mm=(1, 1, 1), pv_distance_mm=5)
    assert vols["periventricular_ml"] == pytest.approx(32 / 1000)
    assert vols["deep_ml"] == pytest.approx(8 / 1000)
    assert vols["total_ml"] == pytest.approx(wmh_volume_ml(wmh, (1, 1, 1)))
    lobes = np.ones(shape, int)
    lobes[:, :, :10] = 2
    lv = lobar_volumes(wmh, lobes, (1, 1, 1), {1: "frontal", 2: "parietal"}, posterior_labels=(2,))
    assert lv["parietal_ml"] == pytest.approx(0.008 + 0.016) and 0 < lv["posterior_fraction"] < 1
    assert dice(wmh, wmh) == 1.0 and dice(wmh, ~wmh) == 0.0
    v = normalize_wmh(np.array([1.0, 9.0]), icv_ml=np.array([1000.0, 1000.0]))
    assert np.allclose(v, np.log1p([0.1, 0.9]))
    rng = np.random.default_rng(0)
    a = rng.lognormal(1, 0.5, 100)
    agree = volume_agreement(a, a * 1.1 + rng.normal(0, 0.1, 100))
    assert agree["icc_2_1"] > 0.9 and agree["pearson_r"] > 0.95


def test_mixed_model_recovers_three_way_interaction():
    df = simulate_wmh_cohort(n_subjects=350, b_aw=-0.08, noise_sd=0.1, seed=1)
    result, table = fit_interaction_mixed_model(df, "cognition", covariates=("age", "sex"), random_slope=True)
    est = table.set_index("term").loc["time:amyloid_pos:wmh_log"]
    assert est["estimate"] < 0
    assert abs(est["estimate"] - (-0.08)) < 3 * est["se"] + 0.02
    # null cohort: interaction not detectable
    df0 = simulate_wmh_cohort(n_subjects=300, b_aw=0.0, noise_sd=0.1, seed=2)
    _, t0 = fit_interaction_mixed_model(df0, "cognition", covariates=("age",), random_slope=False)
    e0 = t0.set_index("term").loc["time:amyloid_pos:wmh_log"]
    assert abs(e0["estimate"]) < 3 * e0["se"] + 0.02


def test_additive_interaction_and_hazard():
    ai = additive_interaction(np.log(2.0), np.log(1.5), np.log(1.6))
    assert ai["RR11"] == pytest.approx(4.8)
    assert ai["RERI"] == pytest.approx(4.8 - 2.0 - 1.5 + 1)
    assert 0 < ai["AP"] < 1 and ai["S"] > 1
    df = simulate_wmh_cohort(n_subjects=600, h_aw=1.0, seed=3)
    pp = person_period(df)
    assert pp["y"].sum() == df.groupby("subject")["event"].first().sum()
    res = discrete_time_hazard(pp, "amyloid_pos * wmh_bin + age")
    b_a, b_w, b_aw = res.params["amyloid_pos"], res.params["wmh_bin"], res.params["amyloid_pos:wmh_bin"]
    assert b_a > 0 and additive_interaction(b_a, b_w, b_aw)["RERI"] > 0

    def fit_fn(d):
        r = discrete_time_hazard(person_period(d), "amyloid_pos * wmh_bin")
        return r.params["amyloid_pos"], r.params["wmh_bin"], r.params["amyloid_pos:wmh_bin"]

    boot = bootstrap_additive_interaction(df, fit_fn, n_boot=8, seed=0)
    assert "RERI" in boot.index and boot.loc["RERI", "n_success"] >= 5
