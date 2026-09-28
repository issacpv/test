"""Synthetic-data tests for recall_lag (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recall_lag import linkage as lk  # noqa: E402
from recall_lag import signals as sg  # noqa: E402
from recall_lag import survival as sv  # noqa: E402
from recall_lag.openfda_client import OpenFDAClient, flatten_enforcement, flatten_event_min, flatten_recall  # noqa: E402


class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def get(self, url, params=None, timeout=None):
        if "page2" in url:
            return _Resp(200, {"results": [{"res_event_number": "3"}]})
        if params and "nomatch" in str(params.get("search")):
            return _Resp(404, {"error": {"code": "NOT_FOUND"}})
        return _Resp(200, {"results": [{"res_event_number": "1"}, {"res_event_number": "2"}]}, {"Link": '<https://api.fda.gov/device/recall.json?page2=1>; rel="next"'})


def test_client_and_flatteners():
    c = OpenFDAClient(session=_FakeSession(), api_key="k", max_per_minute=10000)
    assert [r["res_event_number"] for r in c.fetch_records("device/recall", None)] == ["1", "2", "3"]
    assert c.fetch_records("device/recall", "nomatch") == []
    ev = flatten_event_min({"report_number": "R", "date_received": "20230105", "date_of_event": "20221220", "event_type": "Injury", "device": [{"device_report_product_code": "abc", "manufacturer_d_name": "Acme Inc."}]})
    assert ev["serious"] and ev["product_code"] == "ABC" and (ev["date_received"] - ev["date_of_event"]).days == 16
    rc = flatten_recall({"res_event_number": "1", "event_date_initiated": "2023-06-01", "product_code": "abc", "recalling_firm": "Acme, Inc.", "root_cause_description": "Software design", "k_numbers": ["K1"]})
    assert rc["initiated"] == pd.Timestamp("2023-06-01") and rc["k_numbers"] == "K1"
    en = flatten_enforcement({"event_id": "1", "classification": "Class II", "recall_initiation_date": "20230601", "center_classification_date": "20230715"})
    assert en["classification"] == "Class II" and (en["center_classification"] - en["initiation"]).days == 44


def test_keys_recall_table_and_time_to_event():
    assert lk.normalize_firm("ACME Medical Systems, Inc.") == "acme"
    assert lk.device_key("abc", "Acme Inc") == "ABC|acme" and lk.device_key(None, "Acme") is None
    reports = pd.DataFrame({
        "report_number": list("abcdef"),
        "date_received": pd.to_datetime(["2015-01-10", "2015-03-01", "2016-01-01", "2018-05-01", "2019-01-01", "2010-01-01"]),
        "date_of_event": pd.to_datetime(["2014-12-20", "2015-02-01", "2015-12-01", "2018-04-01", "2018-12-01", "2009-12-01"]),
        "serious": [False, True, True, False, True, True],
        "product_code": ["ABC", "ABC", "ABC", "XYZ", "XYZ", "QQQ"],
        "manufacturer": ["Acme Inc", "ACME, Inc.", "Acme", "Beta LLC", "Beta", "Gamma"],
    })
    reports = lk.add_keys(reports, "product_code", "manufacturer")
    assert reports["key"].nunique() == 3
    fe = sg.first_event_dates(reports)
    assert fe.set_index("key").loc["ABC|acme", "first_serious"] == pd.Timestamp("2015-03-01")
    recalls = pd.DataFrame({"res_event_number": ["1", "2"], "initiated": pd.to_datetime(["2017-03-01", "2008-01-01"]), "posted": pd.to_datetime(["2017-03-20", "2008-02-01"]),
                            "product_code": ["ABC", "QQQ"], "recalling_firm": ["Acme Inc.", "Gamma Corp"], "root_cause": ["Software design", "Labeling"]})
    enforcement = pd.DataFrame({"res_event_number": ["1"], "classification": ["Class II"], "center_classification": pd.to_datetime(["2017-04-15"]), "initiation": pd.to_datetime(["2017-03-01"])})
    rt = lk.recall_table(recalls, enforcement)
    assert rt.loc[0, "software_root_cause"] and not rt.loc[1, "software_root_cause"] and rt.loc[0, "classification"] == "Class II"
    tte = lk.build_time_to_event(fe, rt, data_end=pd.Timestamp("2024-01-01"), reliable_start=pd.Timestamp("2012-01-01"))
    t = tte.set_index("key")
    assert t.loc["ABC|acme", "event"] == 1 and t.loc["ABC|acme", "time_years"] == pytest.approx(2.0, abs=0.01)
    assert t.loc["XYZ|beta", "event"] == 0 and t.loc["XYZ|beta", "status"] == "censored"
    assert t.loc["QQQ|gamma", "status"] == "recall_before_origin" and not t.loc["QQQ|gamma", "analysis"]
    assert t.loc["QQQ|gamma", "entry_years"] == pytest.approx(2.0, abs=0.01)
    comp = lk.lag_components(reports[reports["key"] == "ABC|acme"], rt.iloc[0], signal_month=pd.Period("2016-01", "M"))
    assert comp["n_reports_pre"] == 3 and comp["event_to_receipt_median_days"] == pytest.approx(28.0) and comp["initiation_to_classification_days"] == 45


def test_cusum_detects_step_and_false_alarms_are_rare():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2015-01", periods=48, freq="M")
    quiet = pd.Series(rng.poisson(2.0, 48), index=idx)
    step = quiet.copy()
    step.iloc[24:] = rng.poisson(8.0, 24)
    sd = sg.signal_date(step, baseline_months=12, multiplier=2.0, h=6.0)
    assert sd is not None and pd.Period("2017-01", "M") <= sd <= pd.Period("2017-06", "M")
    series = {f"k{i}": pd.Series(rng.poisson(2.0, 48), index=idx) for i in range(60)}
    far = sg.false_alarm_rate(series, h=8.0)
    assert far["per_100_key_years"] < 10.0
    lt = sg.lead_times({"a": step}, {"a": pd.Period("2017-10", "M")}, h=6.0)
    assert bool(lt.loc[0, "flagged"]) and lt.loc[0, "lead_months"] >= 4
    res = sg.poisson_cusum([0, 0, 10, 10], lambda0=1.0, h=5.0)
    assert res["first_signal_index"] == 2
    mc = sg.monthly_counts(pd.Series(pd.to_datetime(["2020-01-05", "2020-01-20", "2020-04-01"])))
    assert mc.tolist() == [2, 0, 0, 1]


def test_survival_estimators():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.binomial(1, 0.5, n)
    t_true = rng.exponential(1.0 / np.exp(0.8 * x))
    c = rng.exponential(3.0, n)
    time = np.minimum(t_true, c)
    event = (t_true <= c).astype(int)
    km = sv.kaplan_meier(time, event)
    assert (km["survival"].diff().dropna() <= 0).all() and (km["ci_lo"] <= km["survival"]).all() and (km["survival"] <= km["ci_hi"]).all()
    # no censoring: KM equals the empirical survival at the median
    km_full = sv.kaplan_meier(t_true, np.ones(n, int))
    assert sv.median_survival(km_full) == pytest.approx(np.median(t_true), rel=0.05)
    assert 0 < sv.restricted_mean(km_full, 2.0) < 2.0
    lr = sv.logrank_test(time, event, x)
    assert lr["p"] < 0.01
    lr0 = sv.logrank_test(time, event, rng.binomial(1, 0.5, n))
    assert lr0["p"] > 0.01
    df = pd.DataFrame({"time": time, "event": event, "x": x, "firm": rng.integers(0, 30, n), "entry": 0.0})
    cox = sv.cox_ph(df, "time", "event", ["x"], entry_col="entry", cluster_col="firm")
    assert cox.loc["x", "ci_lo"] < np.exp(0.8) < cox.loc["x", "ci_hi"] and cox.loc["x", "hr"] > 1.5
    its = sv.interrupted_time_series(np.r_[np.linspace(3, 2.5, 20), np.linspace(1.5, 1.2, 20)], np.arange(40), break_at=20)
    assert its.loc["level_change", "coef"] < 0 and its.loc["level_change", "p"] < 0.05
