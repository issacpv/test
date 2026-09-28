"""Synthetic-data tests for atrophy_subtypes."""
import numpy as np
import pandas as pd
import pytest

from atrophy_subtypes import ebm, longitudinal as lg, longitudinal_metrics as lm, zscoring as zs


# ---------------------------------------------------------------------------
# longitudinal
# ---------------------------------------------------------------------------
def _tables():
    mr = pd.DataFrame({"mr_id": ["OAS30001_MR_d0000", "OAS30001_MR_d0400", "OAS30001_MR_d1200",
                                 "OAS30002_MR_d0010", "OAS30003_MR_d0000"],
                       "scanner": ["3T"] * 5})
    clinical = pd.DataFrame({"subject": ["OAS30001", "OAS30001", "OAS30001", "OAS30002", "OAS30003"],
                             "day": [0, 380, 1210, 0, 0], "cdr": [0.0, 0.5, 0.5, 0.5, 0.0],
                             "cdr_sb": [0, 1.5, 3.0, 2.0, 0], "mmse": [30, 28, 25, 26, 29],
                             "age_at_entry": [70, 70, 70, 75, 65], "sex_male": [1, 1, 1, 0, 0]})
    pet = pd.DataFrame({"subject": ["OAS30001", "OAS30001", "OAS30002", "OAS30003"],
                        "day": [0, 1100, 5, 20], "tracer": ["PIB", "AV45", "AV45", "PIB"],
                        "centiloid": [5.0, 30.0, 10.0, 50.0]})
    return mr, clinical, pet


def test_amyloid_status_once_positive_rule():
    mr, clinical, pet = _tables()
    t = lg.build_timeline(mr, clinical, pet)
    s = t.set_index("mr_id")
    assert s.loc["OAS30001_MR_d0000", "amyloid_pos"] == 0.0        # negative PiB at d0
    assert s.loc["OAS30001_MR_d0400", "amyloid_pos"] == 1.0        # positive PET at d1100 within 730 d
    assert s.loc["OAS30001_MR_d1200", "amyloid_pos"] == 1.0
    assert s.loc["OAS30002_MR_d0010", "group"] == "A-CI"
    assert s.loc["OAS30003_MR_d0000", "group"] == "A+CN"
    assert s.loc["OAS30001_MR_d0000", "group"] == "A-CN"
    assert list(t[t.subject == "OAS30001"].visit) == [0, 1, 2]
    assert abs(s.loc["OAS30001_MR_d1200", "years"] - 1200 / 365.25) < 1e-9
    pairs = lg.consecutive_pairs(t, min_interval_years=0.5)
    assert len(pairs) == 2 and pairs.interval_years.min() > 1.0
    counts = lg.group_counts(t)
    assert counts.loc["A-CN", "subjects_2plus"] == 1


# ---------------------------------------------------------------------------
# zscoring
# ---------------------------------------------------------------------------
def test_control_zscorer_orientation_and_calibration():
    rng = np.random.default_rng(0)
    n = 500
    ctrl = pd.DataFrame({"age": rng.uniform(60, 90, n), "sex_male": rng.integers(0, 2, n),
                         "icv": rng.normal(1.5e6, 1e5, n)})
    ctrl["hippocampus"] = 4000 - 25 * (ctrl.age - 60) + 0.001 * ctrl.icv + rng.normal(0, 150, n)
    ctrl["lateral_ventricle"] = 10000 + 400 * (ctrl.age - 60) + rng.normal(0, 2000, n)
    scorer = zs.ControlZScorer(["hippocampus", "lateral_ventricle"]).fit(ctrl)
    z = scorer.transform(ctrl)
    assert abs(z.mean()).max() < 0.1 and abs(z.std() - 1).max() < 0.1
    patient = ctrl.iloc[[0]].copy()
    patient["hippocampus"] -= 600     # atrophy -> z should go UP
    patient["lateral_ventricle"] += 8000  # enlargement -> z should go UP
    zp = scorer.transform(patient)
    assert zp["z_hippocampus"].iloc[0] > z["z_hippocampus"].iloc[0] + 3
    assert zp["z_lateral_ventricle"].iloc[0] > z["z_lateral_ventricle"].iloc[0] + 3
    frac = zs.control_calibration_check(z, 2.0)
    assert (frac < 0.06).all()


# ---------------------------------------------------------------------------
# ebm
# ---------------------------------------------------------------------------
def test_ebm_recovers_sequence_and_stages():
    B = 5
    model = ZScoreEBM = ebm.ZScoreEBM(B, z_thresholds=(1.0, 2.0), n_restarts=2, max_passes=10, random_state=1)
    true_seq = model._repair(np.random.default_rng(3).permutation(model.E))
    X, _, stage = ebm.simulate(model, true_seq, [1.0], n=400, noise_sd=0.8, rng=np.random.default_rng(4))
    model.fit(X)
    assert ebm.sequence_similarity(model.sequences_[0], true_seq) > 0.7
    pred = model.predict(X)
    rho = np.corrcoef(pred["stage_expected"], stage)[0, 1]
    assert rho > 0.85
    assert model.is_valid(model.sequences_[0])


def test_ebm_two_subtypes_em_separates_orders():
    B = 4
    model = ebm.ZScoreEBM(B, z_thresholds=(1.0, 2.0), n_subtypes=2, n_restarts=2, max_passes=6, em_iter=6,
                          random_state=0)
    s1 = model._repair(np.arange(model.E))          # biomarker order 0,1,2,3
    s2 = model._repair(np.arange(model.E)[::-1])    # order 3,2,1,0
    X, subtype, _ = ebm.simulate(model, np.vstack([s1, s2]), [0.5, 0.5], n=400, noise_sd=0.7,
                                 rng=np.random.default_rng(5))
    model.fit(X)
    pred = model.predict(X)
    # subtype labels are arbitrary: agreement should be far from 0.5 either way
    agree = (pred["subtype"] == subtype).mean()
    assert max(agree, 1 - agree) > 0.8
    sims = [ebm.sequence_similarity(model.sequences_[k], s) for k in range(2) for s in (s1, s2)]
    assert max(sims) > 0.8


def test_simulate_longitudinal_and_monotonicity_benchmark():
    model = ebm.ZScoreEBM(4, z_thresholds=(1.0, 2.0), n_restarts=1, max_passes=5, random_state=0)
    seq = model._repair(np.arange(model.E))
    rng = np.random.default_rng(6)
    X0, X1, true = ebm.simulate_longitudinal(model, seq, n=150, intervals_years=rng.uniform(1, 4, 150),
                                             events_per_year=1.0, noise_sd=0.5, rng=rng)
    model.fit(X0)  # baseline only
    p0, p1 = model.predict(X0), model.predict(X1)
    df = pd.DataFrame({"subject": np.r_[np.arange(150), np.arange(150)],
                       "years": np.r_[np.zeros(150), np.ones(150)],
                       "stage": np.r_[p0["stage"], p1["stage"]]})
    res = lm.stage_monotonicity(df, n_perm=200)
    assert res["frac_nondecreasing"] > 0.8
    assert res["excess_over_null"] > 0.15 and res["p_value"] < 0.05


# ---------------------------------------------------------------------------
# longitudinal_metrics
# ---------------------------------------------------------------------------
def test_monotonicity_null_for_random_stages():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({"subject": np.repeat(np.arange(n), 3), "years": np.tile([0, 1, 2], n),
                       "stage": rng.integers(0, 10, 3 * n)})
    res = lm.stage_monotonicity(df, n_perm=200)
    assert abs(res["excess_over_null"]) < 0.08 and res["p_value"] > 0.01


def test_subtype_stability_and_holm():
    df = pd.DataFrame({"subject": np.repeat(np.arange(50), 2), "years": np.tile([0, 2], 50),
                       "subtype": np.repeat(np.r_[np.zeros(25), np.ones(25)], 2).astype(int),
                       "subtype_prob": 0.9, "stage": np.tile([2, 5], 50)})
    df.loc[df.index[1::20], "subtype"] = 1 - df.loc[df.index[1::20], "subtype"]  # flip 5 follow-ups
    res = lm.subtype_stability(df)
    assert res["n_pairs"] == 50 and 0.85 <= res["agreement"] <= 0.95
    assert abs(res["expected_agreement"] - 0.5) < 0.05 and res["kappa"] > 0.7
    assert "by_stage_bin" in res
    adj = lm.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_stage_vs_clinical_change_and_mixed_model():
    rng = np.random.default_rng(8)
    n = 80
    stage0 = rng.integers(0, 6, n)
    d_stage = rng.integers(0, 4, n)
    df = pd.DataFrame({"subject": np.repeat(np.arange(n), 2), "years": np.tile([0, 2], n),
                       "stage": np.r_[stage0, stage0 + d_stage],
                       "age": np.repeat(rng.uniform(65, 85, n), 2), "sex_male": np.repeat(rng.integers(0, 2, n), 2)})
    df["cdr_sb"] = 0.5 * df["stage"] + rng.normal(0, 0.5, len(df))
    df["z_hipp"] = 0.3 * df["stage"] + rng.normal(0, 1.0, len(df))
    r = lm.stage_vs_clinical_change(df, "cdr_sb")
    assert r["spearman_rho"] > 0.5 and r["p_value"] < 0.01
    mm = lm.mixed_model_comparison(df, "cdr_sb", ["stage", "z_hipp"])
    assert set(mm.predictor) == {"stage", "z_hipp"}
    assert mm.set_index("predictor").loc["stage", "aic"] < mm.set_index("predictor").loc["z_hipp", "aic"]
