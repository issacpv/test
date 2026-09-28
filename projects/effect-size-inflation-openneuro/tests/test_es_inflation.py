"""Synthetic-data tests for es_inflation (no network access)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from es_inflation import effects, winners_curse as wc  # noqa: E402
from es_inflation.openneuro_client import (  # noqa: E402
    OpenNeuroClient,
    count_subjects_from_participants_tsv,
    extract_dois,
    fetch_dataset_from_github,
    flatten_dataset_node,
)


# ---------------------------------------------------------- fake network
class _Resp:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Serves two GraphQL pages, then a page with hasNextPage=False."""

    def __init__(self):
        self.calls = []
        self.pages = {
            None: {"edges": [_node("ds000001", 10), _node("ds000002", 25)], "pageInfo": {"hasNextPage": True, "endCursor": "c1"}},
            "c1": {"edges": [_node("ds000003", 4)], "pageInfo": {"hasNextPage": False, "endCursor": None}},
        }

    def post(self, url, json=None, timeout=None):
        after = json["variables"].get("after")
        self.calls.append(after)
        return _Resp({"data": {"datasets": self.pages[after]}})


def _node(ds_id, n):
    return {
        "node": {
            "id": ds_id,
            "name": f"Dataset {ds_id}",
            "created": "2020-01-01",
            "public": True,
            "latestSnapshot": {
                "tag": "1.0.0",
                "created": "2020-02-01",
                "summary": {
                    "subjects": [f"{i:02d}" for i in range(n)],
                    "sessions": [],
                    "modalities": ["MRI"],
                    "tasks": ["rest"],
                    "totalFiles": 100,
                    "size": 12345,
                },
                "description": {
                    "Name": f"Dataset {ds_id}",
                    "DatasetDOI": "10.18112/openneuro.ds000001.v1.0.0",
                    "ReferencesAndLinks": ["Author A (2012). Title. Journal. doi: 10.3389/fnins.2012.00080."],
                    "Authors": ["A", "B"],
                },
            },
        }
    }


def test_client_paginates_and_flattens():
    sess = FakeSession()
    client = OpenNeuroClient(session=sess, page_size=2, sleep_between_pages=0)
    df = client.fetch_datasets_table()
    assert sess.calls == [None, "c1"]
    assert list(df["dataset_id"]) == ["ds000001", "ds000002", "ds000003"]
    assert list(df["n_subjects"]) == [10, 25, 4]
    assert df.loc[0, "reference_dois"] == "10.3389/fnins.2012.00080"
    assert df.loc[0, "modalities"] == "MRI"


def test_client_max_datasets_stops_early():
    sess = FakeSession()
    client = OpenNeuroClient(session=sess, page_size=2, sleep_between_pages=0)
    df = client.fetch_datasets_table(max_datasets=1)
    assert len(df) == 1 and sess.calls == [None]


def test_flatten_handles_missing_fields():
    rec = flatten_dataset_node({"id": "ds999999", "latestSnapshot": None})
    assert rec["n_subjects"] == 0 and rec["reference_dois"] == ""


def test_extract_dois_and_participants_tsv():
    dois = extract_dois("see https://doi.org/10.1016/j.neuroimage.2021.118000, and 10.1038/S41586-022-04492-9.")
    assert dois == ["10.1016/j.neuroimage.2021.118000", "10.1038/s41586-022-04492-9"]
    tsv = "participant_id\tsex\tage\nsub-01\tF\t26\nsub-02\tM\t24\n\nsub-02\tM\t24\n"
    assert count_subjects_from_participants_tsv(tsv) == 2


def test_github_fallback_with_fake_session():
    class S:
        def get(self, url, timeout=None):
            if url.endswith("/master/dataset_description.json"):
                return _Resp({"Name": "X", "ReferencesAndLinks": ["doi:10.1000/abc.1"], "Authors": ["Q"]}, 200)
            if url.endswith("/master/participants.tsv"):
                return _Resp(None, 200, text="participant_id\tage\nsub-01\t3\nsub-02\t4\nsub-03\t5\n")
            return _Resp(None, 404)

    rec = fetch_dataset_from_github("ds000001", session=S())
    assert rec["n_subjects"] == 3 and rec["reference_dois"] == "10.1000/abc.1"

    class Missing:
        def get(self, url, timeout=None):
            return _Resp(None, 404)

    assert fetch_dataset_from_github("ds999999", session=Missing()) is None


# ------------------------------------------------------- winner's curse
def test_expected_reported_effect_matches_simulation():
    rng = np.random.default_rng(1)
    d, n = 0.3, 20
    se = float(wc.se_cohens_d(n, None, d))
    c = wc.z_from_alpha(0.05)
    sims = rng.normal(d, se, size=400_000)
    sel = sims[np.abs(sims / se) > c]
    assert float(wc.expected_reported_effect(d, se, c)) == pytest.approx(sel.mean(), abs=0.01)
    assert float(wc.expected_abs_reported_effect(d, se, c)) == pytest.approx(np.abs(sel).mean(), abs=0.01)
    assert float(wc.selection_probability(d, se, c)) == pytest.approx(sel.size / sims.size, abs=0.005)


def test_type_m_decreases_with_n_and_type_s_small_for_large_n():
    d = 0.3
    small = float(wc.type_m_error(d, wc.se_cohens_d(15, None, d)))
    large = float(wc.type_m_error(d, wc.se_cohens_d(500, None, d)))
    assert small > 1.5 and large == pytest.approx(1.0, abs=0.02)
    assert float(wc.type_s_error(d, wc.se_cohens_d(500, None, d))) < 1e-6


def test_conditional_mle_reduces_bias_on_simulated_literature():
    rng = np.random.default_rng(7)
    lit = wc.simulate_literature(4000, true_effects=[0.25], sample_sizes=[20], alpha=0.05, rng=rng)
    naive_bias = lit["d_hat"].abs().mean() - 0.25
    corrected = np.array([wc.conditional_mle(r.d_hat, r.se, wc.z_from_alpha(0.05)).d_cond for r in lit.itertuples()])
    corr_bias = np.abs(corrected).mean() - 0.25
    assert naive_bias > 0.15  # heavy winner's curse at n=20
    assert abs(corr_bias) < naive_bias / 2  # conditional MLE removes most of it
    res = wc.conditional_mle(0.9, 0.22, wc.z_from_alpha(0.05))
    assert res.ci_low < res.d_cond < res.ci_high and res.d_cond < res.d_hat


def test_conditional_mle_passthrough_when_not_significant():
    res = wc.conditional_mle(0.1, 0.2, wc.z_from_alpha(0.05))
    assert res.d_cond == 0.1 and res.shrinkage == 1.0


def test_empirical_bayes_and_required_n():
    rng = np.random.default_rng(3)
    ses = np.full(30, 0.2)
    y = rng.normal(0.4, 0.05, 30) + rng.normal(0, ses)
    post, mu, tau2 = wc.empirical_bayes_shrink(y, ses)
    assert np.var(post) < np.var(y) and abs(mu - 0.4) < 0.1
    n80 = wc.required_n(0.3, power=0.8)
    assert 80 <= n80 <= 100  # classic ~ 90 for one-sample d=0.3


# ---------------------------------------------------------------- effects
def test_conversions_and_annotation():
    assert float(effects.d_from_t(4.0, 16)) == pytest.approx(1.0)
    assert float(effects.d_from_t(2.0, 20, 20)) == pytest.approx(2.0 * np.sqrt(0.1))
    assert float(effects.r_from_t(3.0, 27)) == pytest.approx(np.sqrt(9 / 36))
    e = effects.annotation_to_effect({"stat_type": "t", "stat_value": 4.0, "n1": 16, "design": "one_sample", "threshold_p": 0.001})
    assert e["d"] == pytest.approx(1.0) and e["threshold_z"] == pytest.approx(3.29, abs=0.01)


def test_funnel_asymmetry_detects_selection():
    rng = np.random.default_rng(11)
    lit = wc.simulate_literature(600, true_effects=[0.2], sample_sizes=np.arange(10, 80), alpha=0.05, rng=rng)
    out = effects.funnel_asymmetry(lit["d_hat"], lit["se"])
    assert out["intercept"] > 0 and out["p"] < 0.01
    pooled = effects.pooled_effect(lit["d_hat"], lit["se"])
    assert pooled["k"] == len(lit)


def test_circular_vs_cross_validated_peak_effect_under_null():
    rng = np.random.default_rng(5)
    betas = rng.normal(0, 1, size=(24, 500))  # no true effect anywhere
    d_circ, _ = effects.circular_peak_effect(betas)
    d_cv = effects.cross_validated_peak_effect(betas, n_folds=4, rng=rng)
    assert d_circ > 0.5  # max over 500 null features is large at n=24
    assert abs(d_cv) < 0.3  # held-out estimate is close to zero
    res = effects.split_half_peak_inflation(betas, rng=rng, n_splits=20)
    assert res.d_circular > 0.5 and abs(res.d_cv) < 0.3
    assert res.ratio > 2 or res.ratio == float("inf")  # unbounded when the held-out effect is <= 0

    # planted effect: the true feature should be found and the ratio should be modest but > 1
    betas[:, :5] += 0.9
    res2 = effects.split_half_peak_inflation(betas, rng=rng, n_splits=20)
    assert res2.d_cv > 0.4 and res2.ratio > 1.0
    assert effects.cross_validated_peak_effect(betas, n_folds=4, rng=rng) > 0.4


def test_json_roundtrip_of_mle_result():
    res = wc.conditional_mle(0.8, 0.25, 1.96)
    assert json.loads(json.dumps(res.as_dict()))["d_cond"] == pytest.approx(res.d_cond)
