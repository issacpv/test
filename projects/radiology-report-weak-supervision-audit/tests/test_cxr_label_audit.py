"""Tests for cxr_label_audit on hand-written synthetic reports (PYTHONPATH=src). No real data are used."""
import numpy as np
import pandas as pd
import pytest

from cxr_label_audit import agreement, local_llm_labeler, noise_simulation, report_labeler

REPORT_ABNORMAL = """FINDINGS: The heart is mildly enlarged. There is a small left pleural effusion.
Bibasilar atelectasis is again noted, unchanged from the prior study. No pneumothorax.
Possible early right lower lobe pneumonia. An endotracheal tube terminates 4 cm above the carina.
IMPRESSION: Cardiomegaly with small left effusion; questionable RLL pneumonia."""

REPORT_NORMAL = """FINDINGS: Lungs are clear. Heart size is normal. No effusion or pneumothorax.
IMPRESSION: No acute cardiopulmonary process."""


def test_rule_labeler_on_abnormal_report():
    lab = report_labeler.label_report(REPORT_ABNORMAL)
    assert lab["Cardiomegaly"] == 1
    assert lab["Pleural Effusion"] == 1
    assert lab["Atelectasis"] == 1
    assert lab["Pneumothorax"] == 0
    assert lab["Pneumonia"] == -1
    assert lab["Support Devices"] == 1
    assert lab["Fracture"] is None
    assert lab["No Finding"] is None


def test_rule_labeler_on_normal_report_and_features():
    lab = report_labeler.label_report(REPORT_NORMAL)
    assert lab["No Finding"] == 1
    assert lab["Pleural Effusion"] == 0 and lab["Pneumothorax"] == 0
    assert lab["Cardiomegaly"] is None
    f = report_labeler.report_features(REPORT_NORMAL)
    assert f.templated_normal and not f.comparison_language and f.has_impression_section
    g = report_labeler.report_features(REPORT_ABNORMAL)
    assert g.comparison_language and g.hedging_density > 0 and g.n_words > 30
    row = report_labeler.labels_to_row(lab, unmentioned_as=0)
    assert row["Cardiomegaly"] == 0.0 and row["No Finding"] == 1.0


def test_confusion_and_kappa():
    rep = np.array([1, 1, 0, 0, np.nan, -1, 0, 1])
    exp = np.array([1, 0, 0, 1, 1, 1, 0, 1])
    rb = agreement.resolve_report_labels(rep, unmentioned="negative", uncertain="positive")
    c = agreement.confusion(rb, exp)
    # resolved report = [1,1,0,0,0,1,0,1] vs expert = [1,0,0,1,1,1,0,1]
    assert (c.tp, c.fp, c.fn, c.tn) == (3, 1, 2, 2)
    assert c.sensitivity == pytest.approx(3 / 5) and c.specificity == pytest.approx(2 / 3)
    assert -1 <= c.kappa <= 1 and c.pabak == pytest.approx(2 * 5 / 8 - 1)
    rb_missing = agreement.resolve_report_labels(rep, unmentioned="missing")
    assert np.isnan(rb_missing[4]) and agreement.confusion(rb_missing, exp).n == 7
    df = pd.DataFrame({"Cardiomegaly_report": rep, "Cardiomegaly_expert": exp, "patient": [1, 1, 2, 2, 3, 3, 4, 4]})
    tab = agreement.per_finding_agreement(df, ["Cardiomegaly"])
    assert tab.loc["Cardiomegaly", "tp"] == 4
    est, lo, hi = agreement.cluster_bootstrap_ci(
        df, lambda d: agreement.confusion(agreement.resolve_report_labels(d["Cardiomegaly_report"].to_numpy()),
                                          d["Cardiomegaly_expert"].to_numpy()).sensitivity,
        "patient", n_boot=50, rng=np.random.default_rng(0))
    assert lo <= est <= hi


def test_differential_noise_detects_subgroup_effect():
    rng = np.random.default_rng(1)
    n = 3000
    setting = rng.choice(["icu", "ed"], n)
    portable = (setting == "icu") & (rng.random(n) < 0.9) | (setting == "ed") & (rng.random(n) < 0.3)
    expert = np.ones(n, int)
    p_fn = np.where(setting == "icu", 0.45, 0.15)
    report = (rng.random(n) > p_fn).astype(float)
    df = pd.DataFrame({"setting": setting, "portable": portable.astype(int), "expert": expert, "report": report,
                       "patient": rng.integers(0, 800, n)})
    df["fn"] = agreement.disagreement_indicator(df["report"].to_numpy(), df["expert"].to_numpy(), "fn")
    rates = agreement.stratified_rates(df, "fn", "setting").set_index("setting")
    assert rates.loc["icu", "rate"] > rates.loc["ed", "rate"]
    res = agreement.differential_noise_test(df, "fn", "setting", covariate_cols=["portable"], cluster_col="patient")
    assert res["odds_ratios"]["icu"]["or"] > 2.0 and res["lr_p"] < 1e-6
    adj = agreement.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.5)


def test_noise_injection_estimation_and_fairness_simulation():
    rng = np.random.default_rng(2)
    n = 20000
    g = rng.choice(["A", "B"], n)
    y = (rng.random(n) < 0.3).astype(int)
    fnr = {"A": 0.1, "B": 0.4}
    y_noisy = noise_simulation.inject_noise(y, g, fnr, {"A": 0.02, "B": 0.02}, rng)
    obs_fnr_B = 1 - y_noisy[(g == "B") & (y == 1)].mean()
    assert abs(obs_fnr_B - 0.4) < 0.03
    probs = np.clip(0.15 + 0.7 * y + 0.1 * rng.standard_normal(n), 0.01, 0.99)
    est = noise_simulation.estimate_noise_rates(probs, y_noisy, g)
    assert est["by_group"]["B"][1, 0] > est["by_group"]["A"][1, 0]      # P(noisy=0 | true=1) larger in B
    sim = noise_simulation.simulate_fairness_gap(4000, {"A": 0.3, "B": 0.3}, fnr, {"A": 0.0, "B": 0.0},
                                                 classifier_auc=0.9, n_rep=30, rng=rng)
    assert sim["gap_noisy_labels"] > sim["gap_true_labels"] + 0.05
    assert sim["auc_noisy"] < sim["auc_true"]
    tau = noise_simulation.rank_agreement({"m1": 0.9, "m2": 0.8, "m3": 0.7}, {"m1": 0.7, "m2": 0.8, "m3": 0.9})
    assert tau == pytest.approx(-1.0)


def test_local_llm_guard_prompt_and_parsing():
    local_llm_labeler.assert_local_endpoint("http://127.0.0.1:11434")
    local_llm_labeler.assert_local_endpoint("http://localhost:8000")
    local_llm_labeler.assert_local_endpoint("http://10.0.0.5:8000")
    with pytest.raises(ValueError):
        local_llm_labeler.assert_local_endpoint("https://api.openai.com/v1")
    with pytest.raises(ValueError):
        local_llm_labeler.assert_local_endpoint("http://8.8.8.8:80")
    prompt = local_llm_labeler.build_prompt(REPORT_NORMAL)
    assert "Cardiomegaly" in prompt and REPORT_NORMAL.strip() in prompt
    resp = 'Sure: {"Cardiomegaly": "negative", "Pleural Effusion": "uncertain", "Support Devices": "positive"}'
    lab = local_llm_labeler.parse_response(resp)
    assert lab["Cardiomegaly"] == 0 and lab["Pleural Effusion"] == -1 and lab["Support Devices"] == 1
    assert lab["Fracture"] is None and lab["No Finding"] is None
    labeler = local_llm_labeler.LocalLLMLabeler(base_url="http://127.0.0.1:11434")
    url, payload = labeler._payload("x")
    assert url.startswith("http://127.0.0.1") and payload["options"]["temperature"] == 0.0
    with pytest.raises(ValueError):
        local_llm_labeler.LocalLLMLabeler(base_url="https://example.com")
