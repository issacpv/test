"""Synthetic-data tests for fmri_multiverse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_multiverse import confounds, fetch, glm, speccurve  # noqa: E402


def _fake_confounds(T=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.standard_normal(T) * 0.1 for c in confounds.MOTION6})
    df["white_matter"] = rng.standard_normal(T)
    df["csf"] = rng.standard_normal(T)
    df["global_signal"] = rng.standard_normal(T)
    for k in range(10):
        df[f"a_comp_cor_{k:02d}"] = rng.standard_normal(T)
    for k in range(3):
        df[f"cosine{k:02d}"] = np.cos(np.pi * (np.arange(T) + 0.5) * (k + 1) / T)
    fd = np.abs(rng.standard_normal(T)) * 0.2
    fd[50:53] = 1.0  # spikes
    fd[120] = 0.9
    df["framewise_displacement"] = fd
    df.loc[0, "framewise_displacement"] = np.nan
    return df


def test_confound_strategies_select_expected_columns():
    df = _fake_confounds()
    reg, mask = confounds.select_confounds(df, "hmp6")
    assert reg.shape[1] == 6 + 3 and mask.all()
    reg, _ = confounds.select_confounds(df, "hmp24_phys8_gsr4")
    assert reg.shape[1] == 24 + 8 + 4 + 3
    assert not reg.isna().any().any()
    reg, _ = confounds.select_confounds(df, "acompcor")
    assert sum(c.startswith("a_comp_cor") for c in reg.columns) == 5
    reg, mask = confounds.select_confounds(df, "hmp24_phys8_scrub")
    assert (~mask).sum() >= 4 and not mask[50:53].any() and not mask[120]
    # short surviving segments (< 5 volumes) are dropped
    m = confounds._drop_short_segments(np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1], dtype=bool), 5)
    assert m.tolist() == [False, False, False, True, True, True, True, True, False, False]
    kw = confounds.STRATEGIES["hmp24_phys8_scrub"].nilearn_kwargs()
    assert "scrub" in kw["strategy"] and kw["fd_threshold"] == 0.5
    ms = confounds.motion_summary(df)
    assert ms["max_fd"] == 1.0 and 0 < ms["frac_fd_above"] < 0.1


def test_hrf_and_design_matrix():
    h = glm.spm_hrf(2.0)
    t = np.arange(len(h)) * (2.0 / 16)
    assert 4 < t[np.argmax(h)] < 7
    assert h.min() < 0  # undershoot
    events = pd.DataFrame({"onset": [10, 40, 70], "duration": [10, 10, 10], "trial_type": ["A", "B", "A"]})
    ft = np.arange(60) * 2.0
    X = glm.make_design_matrix(events, ft, hrf_model="spm+derivative", high_pass=0.01)
    assert {"A", "B", "A_derivative", "B_derivative", "constant"} <= set(X.columns)
    assert X["A"].iloc[0] == 0 and X["A"].max() > 0
    assert any(c.startswith("drift_") for c in X.columns)
    con = glm.contrast_vector(X.columns, "A - B")
    assert con[list(X.columns).index("A")] == 1 and con[list(X.columns).index("B")] == -1
    con2 = glm.contrast_vector(X.columns, "0.5*A + 0.5*B")
    assert con2.sum() == pytest.approx(1.0)


def test_first_level_recovers_planted_effect():
    rng = np.random.default_rng(0)
    tr, T, V = 2.0, 240, 6
    events = pd.DataFrame({
        "onset": np.arange(8, 470, 40.0),
        "duration": 12.0,
        "trial_type": ["A", "B"] * 6,
    })
    ft = np.arange(T) * tr
    X = glm.make_design_matrix(events, ft, hrf_model="spm", drift_model="none")
    Y = rng.standard_normal((T, V))
    Y[:, 0] += 3.0 * X["A"].to_numpy()  # region 0 responds to A only
    Y[:, 1] += 3.0 * X["B"].to_numpy()  # region 1 to B only
    conf = _fake_confounds(T=T)
    reg, mask = confounds.select_confounds(conf, "hmp24_phys8_scrub")
    for noise in ("ols", "ar1"):
        spec = glm.PipelineSpec(confounds="hmp24_phys8_scrub", hrf_model="spm", noise_model=noise)
        res = glm.run_first_level_numpy(Y, events, tr, spec, confounds=reg, sample_mask=mask, contrasts={"A-B": "A - B"})
        eff, t = res["A-B"]["effect"], res["A-B"]["t"]
        assert eff[0] > 1.5 and eff[1] < -1.5
        assert abs(t[0]) > 4 and abs(t[1]) > 4 and np.all(np.abs(t[2:]) < 4)


def test_enumerate_multiverse_and_second_level():
    specs = glm.enumerate_multiverse({"smoothing_fwhm": [None, 4.0], "confounds": ["hmp6", "acompcor"], "hrf_model": ["spm"]})
    assert len(specs) == 4 and len({s.label for s in specs}) == 4
    rng = np.random.default_rng(1)
    eff = rng.standard_normal((30, 5)) + np.array([1.0, 0, 0, 0, 0])
    sl = glm.second_level_onesample(eff)
    assert sl["p"][0] < 0.001 and sl["cohen_d"][0] > 0.5
    obs, p_fwe = glm.sign_flip_max_t(eff, n_perm=200)
    assert p_fwe[0] < 0.05 and p_fwe[1:].min() > 0.05


def test_specification_curve_and_scores():
    rng = np.random.default_rng(2)
    n_specs = 60
    df = pd.DataFrame({
        "effect": rng.normal(0.5, 0.2, n_specs),
        "p": rng.uniform(0, 0.1, n_specs),
        "smoothing": rng.choice(["0", "4", "8"], n_specs),
        "confounds": rng.choice(["a", "b"], n_specs),
    })
    df.loc[df["smoothing"] == "8", "effect"] += 0.5
    curve = speccurve.specification_curve(df)
    assert curve["rank"].tolist() == list(range(1, n_specs + 1))
    sc = speccurve.robustness_score(df)
    assert 0 <= sc["prs"] <= 1 and sc["prs"] <= sc["share_significant"]
    assert sc["sign_consistency"] > 0.9
    vd = speccurve.variance_decomposition(df, "effect", ["smoothing", "confounds"])
    assert vd["smoothing"] > vd["confounds"] and abs(vd.sum() - 1) < 1e-9


def test_multiverse_inference_null_and_signal():
    rng = np.random.default_rng(3)
    S, n = 20, 25
    null = rng.standard_normal((S, n))
    res0 = speccurve.multiverse_inference(null, n_perm=200, n_boot=100)
    assert res0["p_median"] > 0.01
    signal = null + 0.8
    res1 = speccurve.multiverse_inference(signal, n_perm=200, n_boot=100)
    assert res1["p_median"] < 0.05 and res1["prs"] > 0.8
    assert res1["prs_ci_low"] <= res1["prs"] <= res1["prs_ci_high"]


def test_fragility_model_and_registry():
    rng = np.random.default_rng(4)
    n = 24
    feats = pd.DataFrame({
        "n_subjects": rng.integers(15, 200, n),
        "mean_fd": rng.uniform(0.1, 0.5, n),
        "tr": rng.choice([1.0, 2.0], n),
        "design_block": rng.integers(0, 2, n),
    })
    prs = 0.5 + 0.002 * feats["n_subjects"] - 0.5 * feats["mean_fd"] + 0.05 * rng.standard_normal(n)
    out = speccurve.fit_fragility_model(feats, prs.clip(0, 1))
    assert out["coef_standardized"]["n_subjects"] > 0 and out["coef_standardized"]["mean_fd"] < 0
    assert out["loo_r"] > 0.5
    assert "ds001734" in fetch.DATASETS and fetch.DATASETS["ds001734"].findings
    cmd = fetch.datalad_install("ds000117", Path("/tmp/x"), get=["sub-01"], dry_run=True)
    assert cmd[0][0] == "datalad"
