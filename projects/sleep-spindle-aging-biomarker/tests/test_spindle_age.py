"""Synthetic-data tests for spindle_age (no EDF files, no YASA/MNE required)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spindle_age import coupling, detect, io_edf, normative, transport  # noqa: E402
from spindle_age.io_edf import N2, N3, REM, W  # noqa: E402

FS = 100.0


def synth_night(coupled: bool = True, n_epochs: int = 20, seed: int = 0, so_freq: float = 0.8):
    """N3 epochs with SOs at ``so_freq`` Hz; spindles at the SO up-state (coupled) or random."""
    rng = np.random.default_rng(seed)
    epoch = 30.0
    n = int(n_epochs * epoch * FS)
    t = np.arange(n) / FS
    hyp = np.full(n_epochs, N3)
    hyp[:2] = W
    hyp[-2:] = REM
    x = rng.standard_normal(n) * 8.0                       # background noise (uV)
    so = 60.0 * np.cos(2 * np.pi * so_freq * t)            # SO: peak = up-state at phase 0
    stage_s = io_edf.hypno_to_samples(hyp, epoch, FS, n)
    nrem = np.isin(stage_s, [N2, N3])
    x += so * nrem
    peaks_true = []
    cycle = 1.0 / so_freq
    k = 0
    while (k + 1) * cycle < n / FS:
        centre = k * cycle  # cosine peak (up-state)
        k += 1
        if k % 6:  # one spindle every sixth SO (~8 per minute)
            continue
        if not coupled:
            centre += rng.uniform(0, cycle)
        c = int(centre * FS)
        if c < int(2 * epoch * FS) or c > n - int(2.5 * epoch * FS):
            continue
        dur = 1.2
        idx = np.arange(int(-dur / 2 * FS), int(dur / 2 * FS))
        win = np.hanning(len(idx))
        x[c + idx] += 40.0 * win * np.sin(2 * np.pi * 13.0 * idx / FS)
        peaks_true.append(c / FS)
    return x, hyp, epoch, np.array(peaks_true)


def test_hypnogram_parsing_and_upsampling():
    xml = """<PSGAnnotation><EpochLength>30</EpochLength><ScoredEvents>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0</Start><Duration>60</Duration></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 2 sleep|2</EventConcept><Start>60</Start><Duration>90</Duration></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 4 sleep|4</EventConcept><Start>150</Start><Duration>30</Duration></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>REM sleep|5</EventConcept><Start>180</Start><Duration>30</Duration></ScoredEvent>
    <ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Hypopnea|Hypopnea</EventConcept><Start>100</Start><Duration>12</Duration></ScoredEvent>
    </ScoredEvents></PSGAnnotation>"""
    hyp, ep = io_edf.parse_nsrr_xml(xml)
    assert ep == 30 and hyp.tolist() == [W, W, N2, N2, N2, N3, REM]
    s = io_edf.hypno_to_samples(hyp, 30, 1, 250)   # 1 Hz -> 30 samples per epoch, 210 scored samples
    assert len(s) == 250 and s[0] == W and s[60] == N2 and s[150] == N3 and s[-1] == io_edf.UNK
    assert io_edf.stage_minutes(hyp, 30) == pytest.approx(2.0)
    h2 = io_edf.annotations_to_hypno([0, 30, 60], [30, 30, 30], ["Sleep stage W", "Sleep stage 2", "Sleep stage R"])
    assert h2.tolist() == [W, N2, REM]


def test_channel_resolution():
    assert io_edf.resolve_channel(["EEG", "EEG(sec)", "EOG(L)"], "C4-M1", "shhs") == "EEG"
    assert io_edf.resolve_channel(["EEG1", "EEG2", "EEG3"], "C4-M1", "mesa") == "EEG3"
    assert io_edf.resolve_channel(["C3", "C4", "A1", "A2", "ROC"], "C4-M1", "mros") == ("C4", "A1")
    assert io_edf.resolve_channel(["EEG Fpz-Cz", "EEG Pz-Oz"], "Fpz-Cz", "sleep-edf") == "EEG Fpz-Cz"
    assert io_edf.resolve_channel(["EEG C4-A1", "EEG C3-A2"], "C4-M1") == "EEG C4-A1"   # generic alias
    assert io_edf.resolve_channel(["F3", "F4"], "C4-M1", "cfs") is None
    d, how = io_edf.pick_derivation(["EEG Fpz-Cz", "EEG Pz-Oz"], "sleep-edf")
    assert d == "Fpz-Cz"
    y, fs = io_edf.preprocess(np.random.default_rng(0).standard_normal(2560), 256, 100)
    assert fs == 100 and len(y) == 1000


def test_detectors_and_coupling_on_synthetic_night():
    x, hyp, ep, peaks_true = synth_night(coupled=True)
    hs = io_edf.hypno_to_samples(hyp, ep, FS, len(x))
    sp = detect.detect_spindles_simple(x, FS, hs, thresh_sd=2.0)
    so = detect.detect_sos_simple(x, FS, hs, relative_sd=1.0, dur_range=(0.8, 2.0))
    assert len(sp) >= 0.7 * len(peaks_true)
    # detected spindle peaks should mostly be near injected ones
    d = np.min(np.abs(sp["Peak"].to_numpy()[:, None] - peaks_true[None, :]), axis=1)
    assert np.median(d) < 0.15
    assert len(so) > 0.5 * (io_edf.stage_minutes(hyp, ep) * 60 * 0.8)
    assert list(sp.columns) == detect.SPINDLE_COLS and list(so.columns) == detect.SO_COLS
    dens = detect.spindle_density(sp, hyp, ep)
    assert 0.5 < dens < 60
    m = coupling.coupling_metrics(x, FS, hs, sp, so, n_surr=50)
    assert m["n_coupled"] > 0.5 * len(sp)
    assert m["mrl"] > 0.7                       # tightly coupled
    assert abs(m["preferred_phase"]) < 0.5      # at the up-state (phase 0)
    assert m["mrl_z"] > 3 and m["rayleigh_p"] < 1e-3
    assert np.isfinite(m["mi_pac"]) and m["mi_pac"] > 0
    # uncoupled control: random phases
    xu, hypu, _, _ = synth_night(coupled=False, seed=1)
    hsu = io_edf.hypno_to_samples(hypu, ep, FS, len(xu))
    spu = detect.detect_spindles_simple(xu, FS, hsu, thresh_sd=2.0)
    sou = detect.detect_sos_simple(xu, FS, hsu, relative_sd=1.0)
    mu = coupling.coupling_metrics(xu, FS, hsu, spu, sou, n_surr=50)
    assert mu["mrl"] < m["mrl"] and mu["mrl"] < 0.4
    summ = detect.spindle_summary(sp, hyp, ep)
    assert summ["sp_count"] == len(sp)


def test_circular_stats():
    rng = np.random.default_rng(0)
    tight = rng.normal(0.3, 0.2, 200)
    unif = rng.uniform(-np.pi, np.pi, 200)
    assert coupling.mean_resultant_length(tight) > 0.9 > coupling.mean_resultant_length(unif) + 0.5
    assert abs(coupling.circular_mean(tight) - 0.3) < 0.1
    z, p = coupling.rayleigh_test(tight)
    assert p < 1e-6
    _, p2 = coupling.rayleigh_test(unif)
    assert p2 > 0.01
    ph = rng.uniform(-np.pi, np.pi, 5000)
    amp = 1 + 0.8 * np.cos(ph)
    assert coupling.modulation_index(ph, amp) > coupling.modulation_index(ph, rng.random(5000)) + 0.02
    with pytest.raises(Exception):
        coupling.coupled_spindles(pd.DataFrame({"Peak": [1.0]}), pd.DataFrame({"NegPeak": [1.0]}), np.zeros(10), FS)["nonexistent"]


def test_normative_model_recovers_trend_and_centiles():
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 90, 1500)
    sex = np.where(rng.random(1500) < 0.5, "F", "M")
    y = 0.8 - 0.005 * age + 0.03 * (sex == "F") + rng.normal(0, 0.05, 1500)
    model = normative.fit_normative(age, y, sex, df=5)
    Q = model.predict(np.array([30.0, 50.0, 80.0]))
    assert Q.shape == (3, 5) and np.all(np.diff(Q, axis=1) >= 0)
    assert Q[1, 2] == pytest.approx(0.8 - 0.25 + 0.015, abs=0.03)     # pooled median at 50 y
    assert Q[0, 2] > Q[2, 2]                                          # declines with age
    QF, QM = model.predict(np.array([50.0]), "F"), model.predict(np.array([50.0]), "M")
    assert QF[0, 2] > QM[0, 2]
    c = model.centile(np.array([50.0, 50.0]), np.array([Q[1, 2], Q[1, 4] + 0.2]))
    assert abs(c[0] - 50) < 3 and c[1] > 97
    z = model.zscore(np.array([50.0]), np.array([Q[1, 2]]))
    assert abs(z[0]) < 0.1
    cal = normative.centile_calibration(model, age, y)
    assert abs(cal[0.5] - 0.5) < 0.05 and abs(cal[0.05] - 0.05) < 0.03
    s = normative.standardized_age_slope(age, y, n_boot=200)
    assert s["ci_high"] < 0 and s["slope_sd_per_decade"] < -0.3
    assert normative.pinball_loss(y, model.predict(age)[:, 2], 0.5) < 0.03


def test_transport_harmonization_and_loco():
    rng = np.random.default_rng(0)
    n = 900
    cohort = np.repeat(["A", "B", "C"], n // 3)
    age = rng.uniform(30, 80, n)
    offsets = {"A": 0.0, "B": 1.5, "C": -1.0}
    scale = {"A": 1.0, "B": 2.0, "C": 0.7}
    f1 = -0.04 * age + np.array([offsets[c] for c in cohort]) + rng.normal(0, 0.15, n) * np.array([scale[c] for c in cohort])
    f2 = 0.02 * age + rng.normal(0, 0.5, n)
    X = np.column_stack([f1, f2])
    h = transport.LocationScaleHarmonizer().fit(X, cohort, age)
    Xh = h.transform(X, cohort, age)
    resid = Xh[:, 0] + 0.04 * age
    means = [resid[cohort == c].mean() for c in "ABC"]
    assert np.ptp(means) < 0.15                       # site offsets removed
    raw = transport.leave_one_cohort_out(X, age, cohort, harmonize=False)
    harm = transport.leave_one_cohort_out(X, age, cohort, harmonize=True, covariates=age)
    assert set(raw.columns) >= {"cohort", "mae_loco", "mae_within", "transport_gap"}
    assert harm["mae_loco"].mean() < raw["mae_loco"].mean()
    assert harm["transport_gap"].mean() < raw["transport_gap"].mean()
    cmp = transport.compare_feature_sets({"coupling": X[:, :1], "density": X[:, 1:]}, age, cohort, harmonize=True,
                                         covariates=age, reference="density")
    assert set(cmp["feature_set"]) == {"coupling", "density"}
    row = cmp[cmp["feature_set"] == "coupling"].iloc[0]
    assert row["mae_loco"] < cmp[cmp["feature_set"] == "density"]["mae_loco"].iloc[0]
    assert row["vs_density_ci_high"] < 0
    pb = transport.paired_bootstrap_difference(np.ones(50), np.ones(50) * 2, groups=np.arange(50) // 5, n_boot=100)
    assert pb["diff"] == pytest.approx(-1.0)
