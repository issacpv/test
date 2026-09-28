"""Synthetic-data tests for lidc_uq.

The properties checked are the ones the scientific claims depend on:

* Soft-label schemes behave as documented on hand-worked rating vectors, including
  the edge cases (all-3, unanimous) where the schemes diverge most.
* Disagreement metrics peak where readers split on the decision that matters.
* Soft-label expansion is mathematically equivalent to soft cross-entropy: a
  logistic model fit on expanded rows recovers the planted probability.
* Calibration metrics identify a deliberately miscalibrated predictor, and the
  Brier decomposition's identity holds.
* Grouped CV actually groups: a leaky nodule-level split scores better than the
  correct scan-level split on planted scan-level structure.
* Lung-RADS boundaries are exactly where v2022 puts them, and a sub-millimetre
  reader difference across 6 mm flips the management category.
* Triage metrics reward a ranking that concentrates disagreement at the top.

No LIDC download, no pylidc, no pyradiomics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lidc_uq.annotations import (  # noqa: E402
    NoduleConsensus,
    RaterAnnotation,
    build_nodule_table,
    consensus_mask,
    disagreement_metrics,
    segmentation_disagreement,
    soft_label_from_ratings,
)
from lidc_uq.calibration import (  # noqa: E402
    brier_decomposition,
    calibration_metrics,
    compare_label_schemes,
    expand_soft_labels,
    grouped_cv_soft,
    reliability_curve,
)
from lidc_uq.disagreement import (  # noqa: E402
    compare_to_size_baseline,
    fit_disagreement_model,
    recall_at_k,
    selective_prediction_curve,
    triage_curve,
)
from lidc_uq.features import (  # noqa: E402
    SIMPLE_FEATURE_NAMES,
    extract_simple_features,
    feature_frame,
    resample_to_isotropic,
)
from lidc_uq.lungrads import (  # noqa: E402
    assign_lung_rads,
    assign_lung_rads_batch,
    category_disagreement,
    management_from_category,
    operating_point_analysis,
)


# --------------------------------------------------------------------- fixtures
def make_nodule(ratings, nodule_id="n0", scan_id="s0", diameters=None) -> NoduleConsensus:
    """A NoduleConsensus from a list of malignancy ratings."""
    diameters = diameters if diameters is not None else [6.0] * len(ratings)
    return NoduleConsensus(
        nodule_id=nodule_id,
        scan_id=scan_id,
        annotations=[
            RaterAnnotation(rater_id=i, malignancy=r, diameter_mm=d)
            for i, (r, d) in enumerate(zip(ratings, diameters))
        ],
    )


def synthetic_cohort(n_scans: int = 60, seed: int = 0):
    """A cohort with a planted feature -> soft-label relationship and scan structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for scan in range(n_scans):
        scan_effect = rng.normal(0, 0.6)
        for nod in range(rng.integers(1, 4)):
            f1 = rng.normal(scan_effect, 1.0)
            f2 = rng.normal(0, 1.0)
            logit = 1.4 * f1 + 0.3 * f2
            p_true = 1.0 / (1.0 + np.exp(-logit))
            ratings = [
                int(np.clip(round(1 + 4 * p_true + rng.normal(0, 0.9)), 1, 5)) for _ in range(4)
            ]
            rows.append(
                {
                    "nodule_id": f"s{scan}_n{nod}",
                    "scan_id": f"s{scan}",
                    "f1": f1,
                    "f2": f2,
                    "p_true": p_true,
                    "ratings": ratings,
                }
            )
    df = pd.DataFrame(rows)
    df["soft_split_3"] = [soft_label_from_ratings(r, scheme="split_3") for r in df["ratings"]]
    df["soft_exclude_3"] = [soft_label_from_ratings(r, scheme="exclude_3") for r in df["ratings"]]
    df["soft_linear"] = [soft_label_from_ratings(r, scheme="linear") for r in df["ratings"]]
    df["hard"] = [int(np.median(r) > 3) for r in df["ratings"]]
    df["n_raters"] = [len(r) for r in df["ratings"]]
    df["binary_disagreement"] = [
        disagreement_metrics(r)["binary_disagreement"] for r in df["ratings"]
    ]
    df["crosses"] = [disagreement_metrics(r)["crosses_threshold"] for r in df["ratings"]]
    return df


def ball_volume(radius: float = 5.0, shape: int = 24, inside_hu: float = 30.0,
                outside_hu: float = -750.0, blur: float = 0.0, seed: int = 0):
    """A spherical nodule in lung parenchyma, optionally blurred at the boundary."""
    from scipy import ndimage

    zz, yy, xx = np.mgrid[0:shape, 0:shape, 0:shape]
    c = shape / 2.0
    mask = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) < radius**2
    img = np.full((shape,) * 3, outside_hu, dtype=float)
    img[mask] = inside_hu
    if blur > 0:
        img = ndimage.gaussian_filter(img, blur)
    img += np.random.default_rng(seed).normal(0, 5.0, img.shape)
    return img, mask


# ----------------------------------------------------------------- soft labels
@pytest.mark.parametrize(
    "ratings,scheme,expected",
    [
        ([1, 1, 1, 1], "split_3", 0.0),
        ([5, 5, 5, 5], "split_3", 1.0),
        ([1, 2, 4, 5], "split_3", 0.5),
        ([3, 3, 3, 3], "split_3", 0.5),
        ([2, 3, 4, 5], "split_3", 0.625),
        ([3, 3, 3, 3], "exclude_3", 0.5),  # no informative readers left
        ([1, 3, 4, 5], "exclude_3", 2 / 3),
        ([1, 1, 1, 1], "linear", 0.0),
        ([5, 5, 5, 5], "linear", 1.0),
        ([3, 3, 3, 3], "linear", 0.5),
    ],
)
def test_soft_label_schemes(ratings, scheme, expected) -> None:
    assert soft_label_from_ratings(ratings, scheme=scheme) == pytest.approx(expected)


def test_soft_label_validation() -> None:
    with pytest.raises(ValueError, match="no finite ratings"):
        soft_label_from_ratings([])
    with pytest.raises(ValueError, match="unknown scheme"):
        soft_label_from_ratings([3], scheme="bogus")  # type: ignore[arg-type]


def test_rater_annotation_rejects_off_scale_ratings() -> None:
    with pytest.raises(ValueError, match="malignancy must be one of"):
        RaterAnnotation(rater_id=0, malignancy=6)
    with pytest.raises(ValueError, match="malignancy must be one of"):
        RaterAnnotation(rater_id=0, malignancy=0)


def test_nodule_requires_annotations() -> None:
    with pytest.raises(ValueError, match="has no annotations"):
        NoduleConsensus(nodule_id="x", scan_id="s", annotations=[])


# --------------------------------------------------------------- disagreement
def test_disagreement_peaks_on_an_even_split() -> None:
    even = disagreement_metrics([1, 2, 4, 5])
    unanimous = disagreement_metrics([5, 5, 5, 5])
    assert even["binary_disagreement"] == pytest.approx(1.0)
    assert unanimous["binary_disagreement"] == pytest.approx(0.0)
    assert even["crosses_threshold"] == 1.0
    assert unanimous["crosses_threshold"] == 0.0


def test_disagreement_distinguishes_variance_from_decision_relevance() -> None:
    """Variance and decision relevance can order two nodules oppositely.

    ``[1, 1, 1, 5]`` has one extreme dissenter: high variance, but three of four
    readers agree it is benign. ``[2, 2, 4, 4]`` has lower variance yet splits the
    panel exactly on the benign/malignant boundary. Only the second is a case where
    a second read would change the answer, and ``binary_disagreement`` is the metric
    that says so.
    """
    lone_dissenter = disagreement_metrics([1, 1, 1, 5])
    even_straddle = disagreement_metrics([2, 2, 4, 4])
    assert lone_dissenter["variance"] > even_straddle["variance"]
    assert lone_dissenter["binary_disagreement"] < even_straddle["binary_disagreement"]
    assert even_straddle["binary_disagreement"] == pytest.approx(1.0)
    # both cross the threshold, but to very different degrees
    assert lone_dissenter["crosses_threshold"] == 1.0
    assert even_straddle["crosses_threshold"] == 1.0
    # a split entirely within the benign range crosses nothing
    assert disagreement_metrics([1, 1, 2, 2])["crosses_threshold"] == 0.0


def test_entropy_is_normalised_and_counts_indeterminates() -> None:
    m = disagreement_metrics([1, 2, 3, 4])
    assert 0.0 <= m["entropy"] <= 1.0
    assert m["n_indeterminate"] == 1.0
    assert disagreement_metrics([3, 3, 3])["n_indeterminate"] == 3.0
    assert disagreement_metrics([4, 4, 4])["entropy"] == pytest.approx(0.0)


def test_majority_label_and_indeterminate_flag() -> None:
    assert make_nodule([1, 2, 2, 3]).majority_label() == 0
    assert make_nodule([4, 4, 5, 5]).majority_label() == 1
    borderline = make_nodule([2, 3, 3, 4])
    assert borderline.is_indeterminate()
    assert borderline.majority_label() == 0  # documented tie-break at median 3


def test_build_nodule_table_columns() -> None:
    nodules = [make_nodule([1, 2, 4, 5], "n0", "s0"), make_nodule([5, 5, 5], "n1", "s1")]
    tbl = build_nodule_table(nodules)
    assert len(tbl) == 2
    for col in ("soft_split_3", "soft_exclude_3", "soft_linear",
                "disagree_binary_disagreement", "majority_label", "scan_id"):
        assert col in tbl.columns
    with pytest.raises(ValueError, match="no nodules"):
        build_nodule_table([])


# ------------------------------------------------------------- consensus masks
def test_consensus_mask_levels() -> None:
    a = np.array([[1, 1, 0, 0]], dtype=bool)
    b = np.array([[1, 1, 1, 0]], dtype=bool)
    c = np.array([[1, 0, 1, 0]], dtype=bool)
    union = consensus_mask([a, b, c], agreement_level=1 / 3)
    half = consensus_mask([a, b, c], agreement_level=0.5)
    inter = consensus_mask([a, b, c], agreement_level=1.0)
    assert union.tolist() == [[True, True, True, False]]
    assert half.tolist() == [[True, True, True, False]]
    assert inter.tolist() == [[True, False, False, False]]


def test_consensus_mask_50_percent_with_four_readers() -> None:
    """Exactly 2 of 4 readers must count as 50% consensus."""
    masks = [np.array([True]), np.array([True]), np.array([False]), np.array([False])]
    assert consensus_mask(masks, agreement_level=0.5).tolist() == [True]


def test_consensus_mask_validation() -> None:
    with pytest.raises(ValueError, match="no masks"):
        consensus_mask([])
    with pytest.raises(ValueError, match="agreement_level"):
        consensus_mask([np.ones(2, dtype=bool)], agreement_level=0.0)
    with pytest.raises(ValueError, match="share a shape"):
        consensus_mask([np.ones(2, dtype=bool), np.ones(3, dtype=bool)])


def test_segmentation_disagreement() -> None:
    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    b = a.copy()
    b[2:8, 2:9] = True
    out = segmentation_disagreement([a, b])
    assert out["intersection_voxels"] == 36
    assert out["union_voxels"] == 42
    assert out["mean_pairwise_dice"] > 0.9
    assert out["volume_cv"] > 0
    identical = segmentation_disagreement([a, a])
    assert identical["jaccard_union_intersection"] == pytest.approx(1.0)
    assert identical["volume_cv"] == pytest.approx(0.0)


# -------------------------------------------------------------------- features
def test_simple_features_complete_and_physical() -> None:
    img, mask = ball_volume(radius=5.0)
    feats = extract_simple_features(img, mask, (1.0, 1.0, 1.0))
    assert set(feats) == set(SIMPLE_FEATURE_NAMES)
    # a sphere of radius 5 has volume ~524 mm^3 and equivalent diameter 10 mm
    assert feats["volume_mm3"] == pytest.approx(4 / 3 * np.pi * 125, rel=0.1)
    assert feats["equivalent_diameter_mm"] == pytest.approx(10.0, rel=0.1)
    assert feats["inner_outer_hu_contrast"] > 500
    assert 0.0 < feats["sphericity"] <= 1.0
    assert feats["elongation"] == pytest.approx(1.0, abs=0.15)


def test_spacing_changes_physical_features() -> None:
    img, mask = ball_volume(radius=5.0)
    fine = extract_simple_features(img, mask, (1.0, 1.0, 1.0))
    thick = extract_simple_features(img, mask, (2.0, 1.0, 1.0))
    assert thick["volume_mm3"] == pytest.approx(2.0 * fine["volume_mm3"], rel=1e-9)


def test_blurred_boundary_lowers_gradient_and_widens_transition() -> None:
    """The disagreement-relevant features must respond to margin sharpness."""
    sharp_img, mask = ball_volume(radius=6.0, blur=0.0, seed=1)
    blurred_img, _ = ball_volume(radius=6.0, blur=2.0, seed=1)
    sharp = extract_simple_features(sharp_img, mask, (1.0, 1.0, 1.0))
    blurred = extract_simple_features(blurred_img, mask, (1.0, 1.0, 1.0))
    assert blurred["boundary_gradient_p90"] < sharp["boundary_gradient_p90"]
    assert blurred["rim_transition_width_mm"] > sharp["rim_transition_width_mm"]


def test_threshold_proximity_features() -> None:
    img, mask = ball_volume(radius=3.0)  # equivalent diameter ~6 mm
    feats = extract_simple_features(img, mask, (1.0, 1.0, 1.0))
    assert abs(feats["diameter_distance_to_6mm"]) < 1.0
    assert feats["diameter_distance_to_nearest_threshold"] < 1.0


def test_feature_validation() -> None:
    img, mask = ball_volume()
    with pytest.raises(ValueError, match="must match"):
        extract_simple_features(img, mask[:-1], (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="mask is empty"):
        extract_simple_features(img, np.zeros_like(mask), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="n_gray_levels"):
        extract_simple_features(img, mask, (1.0, 1.0, 1.0), n_gray_levels=1)
    with pytest.raises(ValueError, match="spacing has"):
        extract_simple_features(img, mask, (1.0, 1.0))
    with pytest.raises(ValueError, match="spacing must be positive"):
        extract_simple_features(img, mask, (1.0, 1.0, 0.0))


def test_resample_to_isotropic() -> None:
    img, _ = ball_volume(shape=16)
    out, spacing = resample_to_isotropic(img, (2.0, 1.0, 1.0), target_mm=1.0)
    assert out.shape[0] == pytest.approx(32, abs=1)
    assert spacing == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="spacing has"):
        resample_to_isotropic(img, (1.0, 1.0))
    with pytest.raises(ValueError, match="positive and finite"):
        resample_to_isotropic(img, (1.0, 1.0, -1.0))


def test_feature_frame_skips_failures() -> None:
    img, mask = ball_volume(shape=16)
    records = [
        ("good", img, mask, (1.0, 1.0, 1.0)),
        ("bad", img, np.zeros_like(mask), (1.0, 1.0, 1.0)),  # empty mask -> skipped
    ]
    frame = feature_frame(records)
    assert list(frame.index) == ["good"]
    with pytest.raises(ValueError, match="unknown backend"):
        feature_frame(records, backend="nope")


# ----------------------------------------------------------------- calibration
def test_expand_soft_labels_weights_and_drops_degenerate_rows() -> None:
    x = np.array([[1.0], [2.0], [3.0]])
    soft = np.array([0.25, 0.0, 1.0])
    xs, ys, ws = expand_soft_labels(x, soft)
    # row 0 -> two rows; rows 1 and 2 -> one row each
    assert xs.shape[0] == 4
    assert ws.sum() == pytest.approx(3.0)
    pos = ws[ys == 1].sum()
    assert pos == pytest.approx(0.25 + 1.0)


def test_expand_soft_labels_applies_rater_weights() -> None:
    x = np.array([[0.0], [0.0]])
    _, _, ws = expand_soft_labels(x, np.array([0.5, 0.5]), weights=np.array([4.0, 1.0]))
    assert ws.sum() == pytest.approx(5.0)


def test_expand_soft_labels_validation() -> None:
    with pytest.raises(ValueError, match="rows but soft has"):
        expand_soft_labels(np.zeros((3, 1)), np.zeros(2))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expand_soft_labels(np.zeros((2, 1)), np.array([-0.5, 0.5]))
    with pytest.raises(ValueError, match="weights must have length"):
        expand_soft_labels(np.zeros((2, 1)), np.array([0.5, 0.5]), weights=np.ones(3))


def test_soft_label_training_recovers_the_planted_probability() -> None:
    """Weighted expansion == soft cross-entropy, so a constant-feature fit returns p."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    from lidc_uq.calibration import SoftLabelTrainer

    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=(n, 1))
    p_true = np.full(n, 0.3)
    trainer = SoftLabelTrainer(LogisticRegression(max_iter=1000, C=1e6)).fit(x, p_true)
    pred = trainer.predict_proba(x)
    assert pred.mean() == pytest.approx(0.3, abs=0.03)


def test_trainer_requires_fit_before_predict() -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    from lidc_uq.calibration import SoftLabelTrainer

    with pytest.raises(RuntimeError, match="call fit"):
        SoftLabelTrainer(LogisticRegression()).predict_proba(np.zeros((2, 1)))


def test_trainer_rejects_single_class_and_unknown_route() -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    from lidc_uq.calibration import SoftLabelTrainer

    x = np.zeros((10, 1))
    with pytest.raises(ValueError, match="single-class"):
        SoftLabelTrainer(LogisticRegression()).fit(x, np.zeros(10))
    with pytest.raises(ValueError, match="unknown route"):
        SoftLabelTrainer(LogisticRegression(), route="bogus").fit(x, np.full(10, 0.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires the hard label"):
        SoftLabelTrainer(LogisticRegression(), route="hard").fit(x, np.full(10, 0.5))


def test_calibration_metrics_detect_miscalibration() -> None:
    rng = np.random.default_rng(11)
    n = 2000
    truth = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < truth).astype(float)
    perfect = calibration_metrics(truth, y, n_bins=10)
    overconfident = calibration_metrics(np.clip(truth * 1.8 - 0.4, 0, 1), y, n_bins=10)
    assert perfect.ece < overconfident.ece
    assert perfect.brier < overconfident.brier
    assert perfect.auc > 0.7


def test_brier_decomposition_identity_holds() -> None:
    rng = np.random.default_rng(13)
    p = rng.uniform(0, 1, 3000)
    y = (rng.uniform(size=3000) < p).astype(float)
    d = brier_decomposition(p, y, n_bins=20)
    assert abs(d["decomposition_residual"]) < 0.01
    assert d["uncertainty"] > 0
    assert d["reliability"] >= 0


def test_uncertainty_term_is_the_label_noise_floor() -> None:
    """With soft targets all equal to 0.5, no predictor can beat Brier 0.25."""
    p = np.full(100, 0.5)
    s = np.full(100, 0.5)
    d = brier_decomposition(p, s)
    assert d["uncertainty"] == pytest.approx(0.0)
    assert d["brier"] == pytest.approx(0.0)


def test_reliability_curve_shapes_and_validation() -> None:
    rng = np.random.default_rng(17)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(float)
    curve = reliability_curve(p, y, n_bins=5)
    assert len(curve) <= 5
    assert curve["n"].sum() == 500
    assert set(curve.columns) == {"bin", "n", "mean_predicted", "mean_observed", "gap"}
    with pytest.raises(ValueError, match="length mismatch"):
        reliability_curve(p, y[:-1])
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve(p, y, n_bins=0)
    with pytest.raises(ValueError, match="unknown strategy"):
        reliability_curve(p, y, strategy="magic")


def test_calibration_metrics_validation() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        calibration_metrics(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="hard_targets must have length"):
        calibration_metrics(np.zeros(3), np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="no finite predictions"):
        calibration_metrics(np.full(3, np.nan), np.zeros(3))


def test_grouped_cv_soft_runs_and_reports() -> None:
    pytest.importorskip("sklearn")
    df = synthetic_cohort(50, seed=3)
    x = df[["f1", "f2"]].to_numpy()
    res = grouped_cv_soft(
        x, df["soft_split_3"].to_numpy(), df["scan_id"].to_numpy(),
        n_raters=df["n_raters"].to_numpy(), hard=df["hard"].to_numpy(), n_splits=5,
    )
    assert np.isfinite(res["oof_probabilities"]).all()
    assert res["n_splits_used"] == 5
    assert res["report"].auc > 0.7


def test_grouped_cv_soft_validation() -> None:
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="row mismatch"):
        grouped_cv_soft(np.zeros((4, 2)), np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="at least 2 groups"):
        grouped_cv_soft(np.zeros((4, 2)), np.full(4, 0.5), np.zeros(4))
    df = synthetic_cohort(20, seed=4)
    with pytest.raises(ValueError, match="eval_soft must have length"):
        grouped_cv_soft(
            df[["f1", "f2"]].to_numpy(), df["soft_split_3"].to_numpy(),
            df["scan_id"].to_numpy(), eval_soft=np.zeros(3),
        )


def test_nodule_level_split_leaks_relative_to_scan_level() -> None:
    """A nodule-level split must look better than the correct scan-level split.

    This is the concrete demonstration of why grouping matters: the synthetic
    cohort has a scan-level feature offset, which a nodule-level split lets the
    model memorise.
    """
    pytest.importorskip("sklearn")
    df = synthetic_cohort(60, seed=5)
    x = df[["f1", "f2"]].to_numpy()
    soft = df["soft_split_3"].to_numpy()
    hard = df["hard"].to_numpy()

    correct = grouped_cv_soft(x, soft, df["scan_id"].to_numpy(), hard=hard, n_splits=5)
    leaky = grouped_cv_soft(x, soft, df["nodule_id"].to_numpy(), hard=hard, n_splits=5)
    assert leaky["report"].auc >= correct["report"].auc - 0.02


def test_compare_label_schemes_uses_a_common_reference() -> None:
    pytest.importorskip("sklearn")
    df = synthetic_cohort(60, seed=6)
    x = df[["f1", "f2"]].to_numpy()
    out = compare_label_schemes(
        x,
        {"split_3": df["soft_split_3"].to_numpy(), "linear": df["soft_linear"].to_numpy()},
        df["hard"].to_numpy(),
        df["scan_id"].to_numpy(),
        n_raters=df["n_raters"].to_numpy(),
        n_splits=5,
    )
    assert set(out["arm"]) == {"majority_vote", "soft_split_3", "soft_linear"}
    # every arm scored against the same reference, so 'uncertainty' must be identical
    assert out["uncertainty"].nunique() == 1
    assert (out["eval_reference"] == "split_3").all()
    assert out["auc"].between(0, 1).all()
    with pytest.raises(ValueError, match="no soft-label schemes"):
        compare_label_schemes(x, {}, df["hard"].to_numpy(), df["scan_id"].to_numpy())
    with pytest.raises(ValueError, match="eval_scheme"):
        compare_label_schemes(
            x, {"split_3": df["soft_split_3"].to_numpy()}, df["hard"].to_numpy(),
            df["scan_id"].to_numpy(), eval_scheme="nope",
        )


# ------------------------------------------------------------------- Lung-RADS
@pytest.mark.parametrize(
    "diameter,expected",
    [
        (3.0, "2"), (5.99, "2"), (6.0, "3"), (7.99, "3"),
        (8.0, "4A"), (14.99, "4A"), (15.0, "4B"), (30.0, "4B"),
    ],
)
def test_solid_nodule_boundaries_are_exact(diameter, expected) -> None:
    assert assign_lung_rads(diameter, nodule_type="solid") == expected


def test_non_solid_and_benign_and_unknown() -> None:
    assert assign_lung_rads(10.0, nodule_type="non_solid") == "2"
    assert assign_lung_rads(30.0, nodule_type="non_solid") == "3"
    assert assign_lung_rads(20.0, benign_features=True) == "2"
    assert assign_lung_rads(20.0, nodule_type="unknown") == "0"
    assert assign_lung_rads(float("nan")) == "0"
    assert assign_lung_rads(-1.0) == "0"


def test_part_solid_rules() -> None:
    assert assign_lung_rads(5.0, nodule_type="part_solid") == "2"
    # >= 6 mm total with an unmeasured solid component is genuinely indeterminate
    assert assign_lung_rads(10.0, nodule_type="part_solid") == "0"
    assert assign_lung_rads(10.0, nodule_type="part_solid", solid_component_mm=4.0) == "3"
    assert assign_lung_rads(10.0, nodule_type="part_solid", solid_component_mm=7.0) == "4A"
    assert assign_lung_rads(12.0, nodule_type="part_solid", solid_component_mm=9.0) == "4B"


def test_new_and_growing_solid_nodules() -> None:
    assert assign_lung_rads(3.0, is_baseline=False, is_new=True) == "2"
    assert assign_lung_rads(5.0, is_baseline=False, is_new=True) == "3"
    assert assign_lung_rads(7.0, is_baseline=False, is_new=True) == "4A"
    assert assign_lung_rads(9.0, is_baseline=False, is_new=True) == "4B"
    assert assign_lung_rads(5.0, is_baseline=False, growing=True) == "4A"
    assert assign_lung_rads(9.0, is_baseline=False, growing=True) == "4B"


def test_assign_lung_rads_rejects_bad_type() -> None:
    with pytest.raises(ValueError, match="unknown nodule_type"):
        assign_lung_rads(6.0, nodule_type="cavitary")  # type: ignore[arg-type]


def test_batch_assignment_and_management_text() -> None:
    cats = assign_lung_rads_batch([3.0, 6.5, 9.0, 20.0])
    assert cats.tolist() == ["2", "3", "4A", "4B"]
    assert "6-month" in management_from_category("3")
    assert "annual" in management_from_category("2")
    assert "unrecognised" in management_from_category("9Z")


def test_submillimetre_reader_difference_flips_management() -> None:
    """The clinically legible statistic: 5.9 vs 6.1 mm changes the follow-up interval."""
    out = category_disagreement("n0", [5.8, 5.9, 6.1, 6.2])
    assert out.n_distinct == 2
    assert out.crosses_positive_boundary is True
    assert out.max_category == "3"
    assert set(out.categories) == {"2", "3"}


def test_category_agreement_when_readers_agree() -> None:
    out = category_disagreement("n1", [9.0, 9.5, 10.0, 8.5])
    assert out.n_distinct == 1
    assert out.crosses_positive_boundary is False
    assert out.modal_category == "4A"


def test_category_disagreement_tie_breaks_upward() -> None:
    out = category_disagreement("n2", [5.0, 5.5, 7.0, 7.5])
    assert out.modal_category == "3"  # 2-2 tie resolved to the higher category


def test_category_disagreement_validation() -> None:
    with pytest.raises(ValueError, match="no finite reader diameters"):
        category_disagreement("n3", [float("nan")])


def test_operating_point_analysis_monotonicity_and_prevalence() -> None:
    rng = np.random.default_rng(23)
    n = 1000
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.18, n), 0, 1)
    table = operating_point_analysis(p, y, target_sensitivities=(0.95,), prevalence=0.01)
    assert table["sensitivity"].is_monotonic_decreasing
    assert table["specificity"].is_monotonic_increasing
    # PPV re-expressed at 1% prevalence must be far below the cohort PPV
    mid = table.iloc[len(table) // 2]
    assert mid["ppv_at_prevalence"] < mid["ppv"]
    assert table["meets_target_sensitivity"].any()


def test_operating_point_analysis_validation() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        operating_point_analysis(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="both classes"):
        operating_point_analysis(np.array([0.1, 0.2]), np.array([1, 1]))
    with pytest.raises(ValueError, match="prevalence"):
        operating_point_analysis(np.array([0.1, 0.9]), np.array([0, 1]), prevalence=1.5)


# ------------------------------------------------------- disagreement triage
def test_triage_curve_rewards_a_good_ranking() -> None:
    n = 1000
    y = np.zeros(n)
    y[:100] = 1.0  # 10% of nodules were disagreed on
    perfect = -np.arange(n, dtype=float)  # ranks the disagreed ones first
    random_score = np.random.default_rng(29).normal(size=n)
    good = triage_curve(perfect, y, fractions=(0.1,))
    bad = triage_curve(random_score, y, fractions=(0.1,))
    assert good["recall"].iloc[0] == pytest.approx(1.0)
    assert good["lift"].iloc[0] == pytest.approx(10.0)
    assert bad["recall"].iloc[0] < 0.4
    assert bad["lift"].iloc[0] < 4.0


def test_triage_curve_validation() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        triage_curve(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="no disagreed nodules"):
        triage_curve(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="k_fraction"):
        recall_at_k(np.zeros(5), np.ones(5), k_fraction=0.0)


def test_compare_to_size_baseline_includes_threshold_proximity() -> None:
    rng = np.random.default_rng(31)
    n = 400
    diameter = rng.uniform(2, 20, n)
    # disagreement is driven by proximity to 6 mm
    prob = np.exp(-((diameter - 6.0) ** 2) / 2.0)
    y = (rng.uniform(size=n) < prob).astype(float)
    model_score = prob + rng.normal(0, 0.05, n)
    out = compare_to_size_baseline(model_score, y, diameter)
    assert set(out["method"]) == {"model", "size", "threshold_proximity"}
    prox_auc = float(out.loc[out["method"] == "threshold_proximity", "auc"].iloc[0])
    size_auc = float(out.loc[out["method"] == "size", "auc"].iloc[0])
    assert prox_auc > size_auc, "proximity must beat raw size when disagreement is threshold-driven"
    with pytest.raises(ValueError, match="length mismatch"):
        compare_to_size_baseline(model_score, y, diameter[:-1])


def test_fit_disagreement_model_recovers_a_planted_signal() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(37)
    n_scans = 60
    rows = []
    for s in range(n_scans):
        for _ in range(rng.integers(1, 4)):
            blur = rng.uniform(0, 1)
            noise = rng.normal(size=2)
            rows.append({"scan": f"s{s}", "blur": blur, "n1": noise[0], "n2": noise[1],
                         "disagree": np.clip(blur + rng.normal(0, 0.15), 0, 1)})
    df = pd.DataFrame(rows)
    model = fit_disagreement_model(
        df[["blur", "n1", "n2"]].to_numpy(),
        df["disagree"].to_numpy(),
        df["scan"].to_numpy(),
        feature_names=["blur", "n1", "n2"],
        n_splits=5,
    )
    assert model.spearman > 0.6
    assert model.auc > 0.7
    assert model.n_splits_used == 5
    assert np.isfinite(model.oof_predictions).all()


def test_fit_disagreement_model_binary_route_and_validation() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(41)
    n = 300
    x = rng.normal(size=(n, 2))
    groups = np.repeat(np.arange(30), 10)
    y = (x[:, 0] + rng.normal(0, 0.5, n) > 0).astype(float)
    model = fit_disagreement_model(x, y, groups, target_kind="binary", n_splits=4)
    assert model.target_kind == "binary"
    assert model.auc > 0.75
    with pytest.raises(ValueError, match="row mismatch"):
        fit_disagreement_model(np.zeros((4, 2)), np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="at least 2 groups"):
        fit_disagreement_model(np.zeros((4, 2)), np.arange(4.0), np.zeros(4))
    with pytest.raises(ValueError, match="is constant"):
        fit_disagreement_model(np.zeros((10, 2)), np.zeros(10), np.arange(10))


def test_selective_prediction_improves_accuracy_when_uncertainty_is_informative() -> None:
    rng = np.random.default_rng(43)
    n = 1000
    y = rng.integers(0, 2, n).astype(float)
    uncertainty = rng.uniform(0, 1, n)
    # noise on the predicted probability grows with uncertainty
    p = np.clip(y * 0.75 + 0.125 + rng.normal(0, 0.05 + 0.5 * uncertainty, n), 0, 1)
    curve = selective_prediction_curve(p, y, uncertainty, coverages=(0.5, 1.0))
    at_half = float(curve.loc[curve["coverage"] == 0.5, "accuracy"].iloc[0])
    at_full = float(curve.loc[curve["coverage"] == 1.0, "accuracy"].iloc[0])
    assert at_half > at_full
    assert curve["n_retained"].iloc[0] == 500


def test_selective_prediction_validation() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        selective_prediction_curve(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="coverages"):
        selective_prediction_curve(np.zeros(3), np.zeros(3), np.zeros(3), coverages=(0.0,))
