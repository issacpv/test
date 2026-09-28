"""Synthetic-data tests for echo_view."""
from __future__ import annotations

import numpy as np

from echo_view import views, acquisition, shift, metrics


def test_canonical_crosswalk():
    assert views.canonical("camus", "4CH") == "A4C"
    assert views.canonical("tmed2", "PLAX") == "PLAX"
    assert views.canonical("echonet-dynamic", "a4c") == "A4C"
    assert views.canonical("tmed2", "nonsense") == "OTHER"


def test_danger_weights():
    assert views.danger_weight("A4C", "A4C") == 0.0
    assert views.danger_weight("A4C", "A2C") > views.danger_weight("PLAX", "PSAX")
    assert views.danger_weight("A4C", "OTHER") > 0


def test_histogram_match_moves_distribution():
    rng = np.random.default_rng(0)
    src = np.clip(rng.normal(0.3, 0.1, (32, 32)), 0, 1)
    ref = np.clip(rng.normal(0.7, 0.1, (40, 40)), 0, 1)
    matched = acquisition.histogram_match(src, ref)
    assert matched.shape == src.shape
    # matched mean should move toward the reference mean
    assert abs(matched.mean() - ref.mean()) < abs(src.mean() - ref.mean())


def test_sector_mask_zeros_corners():
    f = np.ones((32, 32))
    masked = acquisition.sector_mask(f, half_angle_deg=30.0)
    assert masked[0, 0] == 0.0 and masked[0, -1] == 0.0  # top corners outside fan
    assert masked[20, 16] == 1.0                          # near center-bottom inside fan


def test_resample_frame_rate_changes_length():
    clip = np.random.default_rng(0).normal(size=(30, 8, 8))
    out = acquisition.resample_frame_rate(clip, src_fps=30, dst_fps=15)
    assert out.shape[0] == 15
    assert out.shape[1:] == clip.shape[1:]


def test_vendor_discriminability_high_when_separable():
    rng = np.random.default_rng(0)
    Xa = rng.normal(0, 1, (100, 6))
    Xb = rng.normal(2.5, 1, (100, 6))  # clearly different vendor
    auc = shift.vendor_discriminability(Xa, Xb)
    assert auc > 0.9


def test_per_view_drop_and_macro_f1():
    yt = np.array(["A4C", "A2C", "PLAX", "A4C", "A2C", "PLAX"])
    yp_src = yt.copy()
    yp_tgt = np.array(["A2C", "A2C", "PLAX", "A4C", "A4C", "PSAX"])  # target errors
    drops = shift.per_view_drop(yt, yp_src, yt, yp_tgt, ["A4C", "A2C", "PLAX"])
    assert drops["A4C"] > 0  # A4C recall drops on target
    assert shift.macro_f1(yt, yp_src) == 1.0


def test_decompose_drop_shapley_sums():
    increments = {"intensity": 0.10, "geometry": 0.05}

    def ev(active):
        return 0.6 + sum(increments[n] for n in active)

    contrib = shift.decompose_drop(ev, list(increments))
    total = ev(frozenset(increments)) - ev(frozenset())
    assert abs(sum(contrib.values()) - total) < 1e-9


def test_clinically_weighted_error_and_null():
    yt = np.array(["A4C"] * 50 + ["A2C"] * 50)
    # dangerous confusions: many A4C predicted as A2C
    yp = np.array(["A2C"] * 25 + ["A4C"] * 25 + ["A2C"] * 50)
    cwe = metrics.clinically_weighted_error(yt, yp)
    assert cwe > 0
    out = metrics.confusion_null(yt, yp, n_perm=300)
    assert "p" in out and 0 <= out["p"] <= 1


def test_ef_impact_inflation():
    rng = np.random.default_rng(0)
    tv = np.array(["A4C"] * 100)
    pv = tv.copy()
    pv[:20] = "A2C"  # 20% misrouted
    ef = rng.uniform(30, 65, 100)
    out = metrics.ef_impact(tv, pv, ef)
    assert out["fraction_misrouted"] == 0.2
    assert out["ef_mae_inflation"] > 0
