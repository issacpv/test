"""Synthetic phantom tests for fm_clinical_val."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fm_clinical_val import degradations, metrics, registry, stratify

LABELS = {"brain": 1, "lesion": 2}


@pytest.fixture(scope="module")
def phantom():
    return degradations.make_phantom(size=40, seed=0)


def test_degradations_preserve_shape_and_are_monotone(phantom):
    img, _ = phantom
    for kind in degradations.DEGRADATIONS:
        out = degradations.apply(img, kind, 0.5, seed=1)
        assert out.shape == img.shape and out.dtype == np.float32
    err = [np.abs(degradations.apply(img, "noise", l, seed=2) - img).mean() for l in (0.0, 0.3, 0.8)]
    assert err[0] < err[1] < err[2]
    with pytest.raises(KeyError):
        degradations.apply(img, "nope", 0.1)


def test_metrics_basic(phantom):
    _, lab = phantom
    ref = lab == 1
    assert metrics.dice(ref, ref) == 1.0
    shifted = np.roll(ref, 2, axis=0)
    d = metrics.dice(ref, shifted)
    assert 0.8 < d < 1.0
    assert metrics.hd95(ref, shifted) >= 1.0
    assert metrics.assd(ref, ref) == 0.0
    tab = metrics.per_structure(lab, lab, LABELS)
    assert (tab["dice"] == 1.0).all() and set(tab["structure"]) == set(LABELS)
    cmp = metrics.paired_comparison(np.full(8, 0.8), np.full(8, 0.8) + np.linspace(-0.02, 0.05, 8), n_boot=100)
    assert cmp["ci_low"] <= cmp["mean_diff"] <= cmp["ci_high"]
    ni = metrics.tost_noninferiority(np.full(10, 0.8), np.full(10, 0.79), margin=0.02)
    assert ni["p_noninferior"] < 0.05


def test_baseline_model_and_benchmark(phantom):
    img, lab = phantom
    model = registry.get_model("threshold_baseline")
    seg = model.segment(img)
    assert metrics.dice(seg >= 1, lab >= 1) > 0.9
    assert metrics.dice(seg == 2, lab == 2) > 0.5
    cases = [(f"c{i}", *degradations.make_phantom(40, seed=i)) for i in range(2)]
    res = registry.run_benchmark([model], cases, LABELS, degradations={"noise": [0.0, 0.8]})
    assert {"case", "model", "kind", "level", "structure", "dice"} <= set(res.columns)
    clean = res[(res["level"] == 0.0) & (res["structure"] == "brain")]["dice"].mean()
    noisy = res[(res["level"] == 0.8) & (res["structure"] == "brain")]["dice"].mean()
    assert noisy <= clean + 1e-9
    dr = stratify.dose_response(res)
    assert "slope" in dr.columns


def _fake_results(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ("A", "B"):
        for s in range(30):
            cjv = rng.normal(0.4, 0.05)
            for model, slope in (("fm_frozen", -0.9), ("synthseg", -0.3)):
                rows.append({"dataset": ds, "subject": f"{ds}{s}", "model": model, "regime": "frozen", "n_train": 50,
                             "cjv": cjv, "lesion_ml": rng.gamma(1.5, 3), "dice": 0.85 + slope * (cjv - 0.4) + rng.normal(0, 0.02)})
    return pd.DataFrame(rows)


def test_strata_and_quality_slopes():
    df = _fake_results()
    df["qtile"] = stratify.quality_tertiles(df, "cjv")
    assert set(df["qtile"]) == {"good", "medium", "poor"}
    df["ptile"] = stratify.pathology_strata(df["lesion_ml"])
    tab = stratify.stratified_table(df, "qtile")
    assert len(tab) == 6
    slopes = stratify.quality_slopes(df, "cjv")
    s = slopes.set_index("model")["slope"]
    assert s["fm_frozen"] < s["synthseg"] < 0


def test_learning_curve_and_crossing():
    n = np.array([5, 10, 25, 50, 100])
    truth = 0.9 - 0.4 * n ** -0.6
    fit = stratify.fit_learning_curve(n, truth)
    assert abs(fit["a"] - 0.9) < 0.02
    n_star = stratify.crossing_point(fit, 0.8)
    assert 10 < n_star < 30
    assert np.isnan(stratify.crossing_point(fit, 0.95))
    rng = np.random.default_rng(0)
    rows = []
    for s in range(12):
        for nn in n:
            rows.append({"model": "nnunet_scratch", "regime": "scratch", "n_train": nn, "subject": f"s{s}",
                         "dice": 0.9 - 0.4 * nn ** -0.6 + rng.normal(0, 0.01)})
        rows.append({"model": "fm", "regime": "fewshot", "n_train": 50, "subject": f"s{s}", "dice": 0.8 + rng.normal(0, 0.01)})
    cp = stratify.crossing_points(pd.DataFrame(rows), n_boot=30)
    assert len(cp) == 1 and 5 < cp.loc[0, "n_star"] < 40
