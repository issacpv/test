"""Synthetic-data tests for mriqc_audit (no network access)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mriqc_audit.openneuro_client import (  # noqa: E402
    parse_annex_key, annex_md5_index, bids_entities, flatten_dataset, link_iqms_to_openneuro,
)
from mriqc_audit.mriqc_client import records_to_frame, dedupe_records, summarise_by_scanner, major_version  # noqa: E402
from mriqc_audit.normative import (  # noqa: E402
    QuantileNormativeModel, LocationScaleModel, coverage_check, simulate_iqm_cohort,
)
from mriqc_audit.qc_sensitivity import (  # noqa: E402
    QCRule, apply_rules, qc_multiverse, effect_size, specification_summary, exclusion_risk_ratio,
)

MD5 = "0123456789abcdef0123456789abcdef"


def _record(md5: str, cjv: float, version="23.1.0", created="2023-05-01T10:00:00", fs=3.0):
    return {
        "_id": md5[:8], "_created": created, "_updated": created, "_etag": "x",
        "_links": {"self": {"href": "T1w/1"}},
        "bids_meta": {"MagneticFieldStrength": fs, "Manufacturer": "Siemens", "modality": "T1w",
                      "subject_id": "hash", "session_id": "hash"},
        "provenance": {"md5sum": md5, "version": version, "software": "mriqc", "settings": {"fd_thres": 0.2}},
        "cjv": cjv, "cnr": 3.1, "snr_total": 9.0, "efc": 0.5,
    }


def test_records_to_frame_and_dedupe():
    recs = [_record(MD5, 0.4, created="2023-01-01T00:00:00"), _record(MD5, 0.41, created="2023-06-01T00:00:00"),
            _record("f" * 32, 0.6, version="22.0.6", fs=1.5)]
    df = records_to_frame(recs, "T1w")
    assert "provenance.settings" not in df.columns and "_links" not in df.columns
    assert df["cjv"].dtype.kind == "f" and df["bids_meta.MagneticFieldStrength"].dtype.kind == "f"
    dd = dedupe_records(df)
    assert len(dd) == 2
    assert dd.loc[dd["provenance.md5sum"] == MD5, "cjv"].iloc[0] == pytest.approx(0.41)  # most recent kept
    summ = summarise_by_scanner(dd)
    assert {"bids_meta.Manufacturer", "bids_meta.MagneticFieldStrength", "mriqc_major", "n"} <= set(summ.columns)
    assert major_version("23.1.0") == "23" and major_version(None) is None


def test_annex_key_parsing_and_index(tmp_path):
    target = f".git/annex/objects/Xk/9Q/MD5E-s1234--{MD5}.nii.gz/MD5E-s1234--{MD5}.nii.gz"
    assert parse_annex_key(target) == (MD5, 1234)
    assert parse_annex_key("not-a-key") is None
    ds = tmp_path / "ds000001"
    anat = ds / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (ds / ".git").mkdir()
    os.symlink(target, anat / "sub-01_T1w.nii.gz")
    (anat / "sub-01_T1w.json").write_text("{}")
    idx = annex_md5_index(ds)
    assert len(idx) == 1 and idx.iloc[0]["md5"] == MD5 and idx.iloc[0]["size"] == 1234
    assert bids_entities(idx.iloc[0]["filename"]) == {"sub": "01", "suffix": "T1w"}
    iqms = records_to_frame([_record(MD5, 0.4)], "T1w")
    linked = link_iqms_to_openneuro(iqms, idx, "ds000001")
    assert len(linked) == 1 and linked.iloc[0]["bids_sub"] == "01" and linked.iloc[0]["dataset_id"] == "ds000001"


def test_flatten_dataset():
    node = {"id": "ds000001", "name": "Test", "created": "2020", "public": True,
            "metadata": {"studyDomain": "Cognitive", "species": "Human"},
            "latestSnapshot": {"tag": "1.0.0", "created": "2020", "description": {"Name": "Test", "DatasetDOI": "doi"},
                               "summary": {"modalities": ["MRI"], "subjects": ["01", "02"], "sessions": [], "tasks": ["rest"],
                                           "size": 10, "totalFiles": 3, "dataProcessed": False,
                                           "subjectMetadata": [{"participantId": "01", "age": 8, "sex": "F", "group": "ASD"},
                                                               {"participantId": "02", "age": 30, "sex": "M", "group": "TD"}]}}}
    row = flatten_dataset(node)
    assert row["n_subjects"] == 2 and row["frac_pediatric"] == 0.5 and row["n_groups"] == 2 and row["study_domain"] == "Cognitive"


def test_normative_models():
    df = simulate_iqm_cohort(n=1500, seed=0)
    qm = QuantileNormativeModel("cjv", cat_cols=("MagneticFieldStrength", "Manufacturer"), log_transform=True).fit(df)
    Q = qm.predict_quantiles(df)
    # quantile curves ordered
    assert (np.diff(Q[sorted(Q.columns)].to_numpy(), axis=1) >= -1e-6).mean() > 0.99
    cent = qm.centile(df)
    cov = coverage_check(cent)
    assert np.all(np.abs(cov["empirical"] - cov["nominal"]) < 0.06)
    curves = qm.curves(np.linspace(5, 85, 9), MagneticFieldStrength="3.0")
    assert curves.shape[0] == 9 and 0.5 in curves.columns
    # children should have higher median cjv than young adults in this simulation
    assert curves.iloc[0][0.5] > curves.iloc[3][0.5]
    ls = LocationScaleModel("cjv", cat_cols=("MagneticFieldStrength",), log_transform=True).fit(df)
    z = ls.zscore(df)
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.15
    # heteroscedasticity captured: predicted sigma larger in children
    p = ls.predict(pd.DataFrame({"age": [6.0, 40.0], "MagneticFieldStrength": [3.0, 3.0]}))
    assert p["sigma"].iloc[0] > p["sigma"].iloc[1]


def test_qc_multiverse():
    df = simulate_iqm_cohort(n=1200, seed=1)
    rng = np.random.default_rng(0)
    # outcome partially driven by quality (motion shrinks apparent volume) and by group
    df["gm_vol"] = 600 - 40 * np.log(df["cjv"]) - 10 * (df["group"] == "patient") + rng.normal(0, 15, len(df))
    df["dataset_id"] = rng.choice(["dsA", "dsB", "dsC"], len(df))
    keep = apply_rules(df, [QCRule("cjv", 0.45)])
    assert keep.sum() < len(df) and keep.dtype == bool
    rel = apply_rules(df, [QCRule("cjv", 0.9, relative=True, by="dataset_id")])
    assert 0.85 < rel.mean() < 0.95
    rr = exclusion_risk_ratio(keep, df["group"], "patient", "control")
    assert rr["rr"] > 1.0  # patients excluded more often
    grid = {"cjv": [None, QCRule("cjv", 0.45), QCRule("cjv", 0.35), QCRule("cjv", 0.9, relative=True, by="dataset_id")]}
    res = qc_multiverse(df, grid, "group", "gm_vol", covariates=["age"])
    assert len(res) == 4 and res.loc[0, "n_retained"] == len(df)
    assert (res["n_retained"].iloc[1:] < len(df)).all()
    assert res["g"].notna().all() and (res["power_at_full_g"].between(0, 1)).all()
    summ = specification_summary(res)
    assert summ["n_specs"] == 4 and summ["g_range"] >= 0
    es = effect_size(rng.normal(1, 1, 100), rng.normal(0, 1, 100))
    assert 0.5 < es["g"] < 1.5 and es["g_lo"] < es["g"] < es["g_hi"]
