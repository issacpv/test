"""Multi-rater LIDC-IDRI annotation aggregation and disagreement quantification.

LIDC-IDRI (Armato et al. 2011, *Medical Physics*) had up to four thoracic
radiologists independently annotate each scan in a two-phase read. For every
nodule >= 3 mm each reader supplied a contour and nine semantic ratings, including
**malignancy on a 1-5 scale** where 3 means "indeterminate". Readers were never
forced to agree, and they frequently did not: complete agreement on malignancy
across all nodules in a case holds for only about 163 of the ~1000 patients.

Nearly all published LIDC malignancy classifiers begin by destroying that
information -- taking a median or a majority vote and training on a single hard
label. This module keeps it. The central objects are the **soft label** (the
fraction of raters who called a nodule malignant) and the **disagreement score**
(how much they disagreed), and this project treats the second as a prediction
target in its own right rather than as noise.

Three soft-label conventions are implemented because the choice is itself a
research variable, not an implementation detail:

``exclude_3``
    Drop ratings of 3, then take the fraction of the remainder above 3. The
    common convention; it discards the readers who were explicitly uncertain,
    which is exactly the signal we care about.
``split_3``
    A rating of 3 contributes 0.5. Keeps every reader and treats indeterminacy as
    maximal uncertainty.
``linear``
    ``(mean_rating - 1) / 4``, using the full ordinal scale. Retains gradation
    between 4 and 5 but assumes the scale is interval-spaced, which it is not.

``pylidc`` is an optional dependency: everything below works on plain rating
tables, so the pipeline and its tests run without the 125 GB DICOM download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SoftLabelScheme",
    "RaterAnnotation",
    "NoduleConsensus",
    "soft_label_from_ratings",
    "disagreement_metrics",
    "consensus_mask",
    "build_nodule_table",
    "segmentation_disagreement",
    "load_pylidc_nodules",
    "MALIGNANCY_SCALE",
    "SEMANTIC_FEATURES",
]

LOG = logging.getLogger(__name__)

SoftLabelScheme = Literal["exclude_3", "split_3", "linear"]

#: Valid malignancy ratings. 1 = highly unlikely, 3 = indeterminate, 5 = highly suspicious.
MALIGNANCY_SCALE = (1, 2, 3, 4, 5)

#: The nine LIDC semantic characteristics, all rated per reader per nodule.
SEMANTIC_FEATURES = (
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
)


@dataclass
class RaterAnnotation:
    """One radiologist's annotation of one nodule.

    Attributes:
        rater_id: Reader identifier. LIDC readers are anonymous and **not
            consistently identified across cases**, so this is a within-scan index
            only; it must never be used as a random effect across scans.
        malignancy: Malignancy rating, 1-5.
        diameter_mm: Reader's nodule diameter estimate in mm.
        volume_mm3: Reader's segmented volume in mm^3, if available.
        semantic: Other semantic ratings keyed by name.
        mask: Optional boolean segmentation mask for this reader.
        bbox: Optional bounding box as slices into the parent volume.
    """

    rater_id: int | str
    malignancy: int
    diameter_mm: float | None = None
    volume_mm3: float | None = None
    semantic: dict[str, float] = field(default_factory=dict)
    mask: np.ndarray | None = None
    bbox: tuple[slice, ...] | None = None

    def __post_init__(self) -> None:
        if int(self.malignancy) not in MALIGNANCY_SCALE:
            raise ValueError(
                f"malignancy must be one of {MALIGNANCY_SCALE}, got {self.malignancy!r}"
            )
        self.malignancy = int(self.malignancy)


@dataclass
class NoduleConsensus:
    """All readers' annotations of a single physical nodule.

    A "nodule" here is a cluster of annotations that pylidc (or an equivalent
    distance-based clustering) judged to refer to the same lesion. Clustering is
    itself uncertain -- readers sometimes disagree about whether one lesion or two
    are present -- so ``n_raters`` below 4 can mean either "a reader saw nothing
    here" or "the clustering split the annotations".

    Attributes:
        nodule_id: Identifier unique within the scan.
        scan_id: Scan / patient identifier. **This is the grouping unit for every
            data split**: multiple nodules from one patient must never straddle a
            train/test boundary.
        annotations: One entry per reader.
        spacing_mm: Voxel spacing (z, y, x) in mm.
        metadata: Free-form provenance (dataset, slice thickness, manufacturer).
    """

    nodule_id: str
    scan_id: str
    annotations: list[RaterAnnotation]
    spacing_mm: tuple[float, float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.annotations:
            raise ValueError(f"nodule {self.nodule_id} has no annotations")

    @property
    def n_raters(self) -> int:
        return len(self.annotations)

    @property
    def ratings(self) -> np.ndarray:
        """(n_raters,) integer malignancy ratings."""
        return np.array([a.malignancy for a in self.annotations], dtype=int)

    @property
    def diameters_mm(self) -> np.ndarray:
        """(n_raters,) diameter estimates; ``nan`` where a reader gave none."""
        return np.array(
            [np.nan if a.diameter_mm is None else float(a.diameter_mm) for a in self.annotations]
        )

    def soft_label(self, scheme: SoftLabelScheme = "split_3") -> float:
        """Probability-of-malignancy soft label under the chosen scheme."""
        return soft_label_from_ratings(self.ratings, scheme=scheme)

    def majority_label(self) -> int:
        """Hard label from the reader median, the usual LIDC convention.

        Returns:
            1 if the median rating exceeds 3, 0 if below, and 0 at exactly 3 --
            the tie-breaking choice that most published pipelines make implicitly
            and that biases the operating point. Compare against
            :meth:`soft_label` to see the cost.
        """
        return int(np.median(self.ratings) > 3)

    def is_indeterminate(self) -> bool:
        """True if the median rating is exactly 3 (readers collectively unsure)."""
        return bool(np.median(self.ratings) == 3)

    def disagreement(self) -> dict[str, float]:
        """Disagreement metrics for this nodule; see :func:`disagreement_metrics`."""
        return disagreement_metrics(self.ratings)


def soft_label_from_ratings(
    ratings: Sequence[int] | np.ndarray, *, scheme: SoftLabelScheme = "split_3"
) -> float:
    """Convert per-reader malignancy ratings to a probability in [0, 1].

    Args:
        ratings: Malignancy ratings on the 1-5 scale.
        scheme: ``"exclude_3"``, ``"split_3"`` or ``"linear"`` (see module docstring).

    Returns:
        Soft label in [0, 1]. Under ``"exclude_3"``, a nodule whose every reader
        said 3 has no informative ratings left and returns 0.5.

    Raises:
        ValueError: If ``ratings`` is empty or the scheme is unknown.
    """
    r = np.asarray(ratings, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("no finite ratings")

    if scheme == "linear":
        return float(np.clip((r.mean() - 1.0) / 4.0, 0.0, 1.0))
    if scheme == "split_3":
        return float(np.mean(np.where(r > 3, 1.0, np.where(r < 3, 0.0, 0.5))))
    if scheme == "exclude_3":
        kept = r[r != 3]
        if kept.size == 0:
            return 0.5
        return float(np.mean(kept > 3))
    raise ValueError(f"unknown scheme {scheme!r}")


def disagreement_metrics(ratings: Sequence[int] | np.ndarray) -> dict[str, float]:
    """Quantify how much readers disagreed about one nodule.

    Several measures are returned because they capture different failure modes,
    and a triage tool would care about different ones:

    ``variance`` / ``sd``
        Spread on the raw 1-5 scale. Simple, but treats a 1-versus-2 split (both
        benign) the same as a 2-versus-3 split (benign versus indeterminate).
    ``range``
        Max minus min. Sensitive to a single dissenting reader, which is arguably
        the clinically important case.
    ``binary_disagreement``
        ``1 - |2p - 1|`` where ``p`` is the ``split_3`` soft label. Peaks at 1 when
        readers split evenly on the decision that actually matters. **This is the
        recommended primary target**: it is the disagreement that changes management.
    ``entropy``
        Shannon entropy of the rating histogram in bits, normalised by
        ``log2(5)``. Captures multi-modal disagreement that variance misses.
    ``crosses_threshold``
        1.0 if any reader was above 3 and any below, else 0.0. The binary
        "would a second read change the answer" flag.
    ``n_indeterminate``
        Count of readers who rated exactly 3.

    Args:
        ratings: Malignancy ratings, 1-5.

    Returns:
        Dict of the metrics above plus ``n_raters`` and ``mean_rating``.

    Raises:
        ValueError: If ``ratings`` is empty.
    """
    r = np.asarray(ratings, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("no finite ratings")

    p = soft_label_from_ratings(r, scheme="split_3")
    counts = np.array([np.sum(r == level) for level in MALIGNANCY_SCALE], dtype=float)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log2(nz)).sum() / np.log2(len(MALIGNANCY_SCALE)))

    return {
        "n_raters": float(r.size),
        "mean_rating": float(r.mean()),
        "variance": float(np.var(r, ddof=1)) if r.size > 1 else 0.0,
        "sd": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
        "range": float(r.max() - r.min()),
        "binary_disagreement": float(1.0 - abs(2.0 * p - 1.0)),
        "entropy": entropy,
        "crosses_threshold": float(bool(np.any(r > 3) and np.any(r < 3))),
        "n_indeterminate": float(np.sum(r == 3)),
    }


def consensus_mask(
    masks: Sequence[np.ndarray], *, agreement_level: float = 0.5, return_counts: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Voxel-wise consensus segmentation from several readers' masks.

    A voxel is included when at least ``agreement_level`` of the readers included
    it. ``0.5`` is the usual 50% consensus; ``1.0`` gives the intersection and
    ``1/n`` the union. The spread between the union and intersection volumes is
    itself a useful segmentation-disagreement feature and is why ``return_counts``
    exists.

    Args:
        masks: Boolean or 0/1 arrays of identical shape, one per reader.
        agreement_level: Required fraction of readers, in (0, 1].
        return_counts: Also return the per-voxel reader count.

    Returns:
        The consensus boolean mask, or ``(mask, counts)`` if ``return_counts``.

    Raises:
        ValueError: If no masks are given, shapes differ, or the level is invalid.
    """
    if not masks:
        raise ValueError("no masks given")
    if not 0.0 < agreement_level <= 1.0:
        raise ValueError("agreement_level must be in (0, 1]")
    shapes = {np.asarray(m).shape for m in masks}
    if len(shapes) != 1:
        raise ValueError(f"masks must share a shape, got {sorted(shapes)}")

    counts = np.zeros(next(iter(shapes)), dtype=np.int16)
    for m in masks:
        counts += np.asarray(m).astype(bool).astype(np.int16)
    needed = agreement_level * len(masks)
    # ">=" with a small tolerance so agreement_level=0.5 with 4 readers needs 2.
    out = counts >= (needed - 1e-9)
    return (out, counts) if return_counts else out


def segmentation_disagreement(masks: Sequence[np.ndarray]) -> dict[str, float]:
    """Volume-based disagreement between readers' segmentations.

    Args:
        masks: One boolean mask per reader, identical shapes.

    Returns:
        Dict with ``union_voxels``, ``intersection_voxels``,
        ``jaccard_union_intersection`` (intersection over union across all readers,
        i.e. a generalised Jaccard), ``mean_pairwise_dice`` and
        ``volume_cv`` (coefficient of variation of per-reader volumes).

    Raises:
        ValueError: If fewer than one mask is given or shapes differ.
    """
    if not masks:
        raise ValueError("no masks given")
    arrays = [np.asarray(m).astype(bool) for m in masks]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"masks must share a shape, got {sorted(shapes)}")

    union = np.zeros_like(arrays[0])
    inter = np.ones_like(arrays[0])
    for a in arrays:
        union |= a
        inter &= a
    volumes = np.array([a.sum() for a in arrays], dtype=float)

    dices: list[float] = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            denom = volumes[i] + volumes[j]
            dices.append(2.0 * float((arrays[i] & arrays[j]).sum()) / denom if denom > 0 else np.nan)

    return {
        "n_raters": float(len(arrays)),
        "union_voxels": float(union.sum()),
        "intersection_voxels": float(inter.sum()),
        "jaccard_union_intersection": float(inter.sum() / union.sum()) if union.any() else np.nan,
        "mean_pairwise_dice": float(np.nanmean(dices)) if dices else np.nan,
        "volume_cv": float(volumes.std(ddof=1) / volumes.mean())
        if len(volumes) > 1 and volumes.mean() > 0
        else 0.0,
    }


def build_nodule_table(
    nodules: Iterable[NoduleConsensus],
    *,
    schemes: Sequence[SoftLabelScheme] = ("exclude_3", "split_3", "linear"),
) -> pd.DataFrame:
    """Flatten nodules into one row each, with labels and disagreement metrics.

    Args:
        nodules: Nodule consensus objects.
        schemes: Soft-label schemes to compute; each becomes a ``soft_<scheme>``
            column, so downstream code can compare them without re-aggregating.

    Returns:
        A dataframe with one row per nodule.

    Raises:
        ValueError: If no nodules are supplied.
    """
    rows: list[dict[str, object]] = []
    for nod in nodules:
        row: dict[str, object] = {
            "nodule_id": nod.nodule_id,
            "scan_id": nod.scan_id,
            "n_raters": nod.n_raters,
            "median_rating": float(np.median(nod.ratings)),
            "majority_label": nod.majority_label(),
            "is_indeterminate": nod.is_indeterminate(),
            "mean_diameter_mm": float(np.nanmean(nod.diameters_mm))
            if np.isfinite(nod.diameters_mm).any()
            else np.nan,
            "diameter_sd_mm": float(np.nanstd(nod.diameters_mm, ddof=1))
            if np.isfinite(nod.diameters_mm).sum() > 1
            else np.nan,
        }
        for scheme in schemes:
            row[f"soft_{scheme}"] = nod.soft_label(scheme)
        row.update({f"disagree_{k}": v for k, v in nod.disagreement().items()})
        for key, value in nod.metadata.items():
            row.setdefault(str(key), value)
        rows.append(row)
    if not rows:
        raise ValueError("no nodules to tabulate")
    return pd.DataFrame(rows)


def load_pylidc_nodules(
    *,
    limit: int | None = None,
    min_raters: int = 3,
    clustering_tol: float = 0.5,
    load_masks: bool = False,
) -> list[NoduleConsensus]:
    """Build :class:`NoduleConsensus` objects from a local LIDC-IDRI via ``pylidc``.

    ``pylidc`` clusters each scan's annotations into nodules using an inter-
    annotation distance criterion, then exposes per-reader ratings and contours.
    It requires the DICOM data on disk and a ``~/.pylidcrc`` pointing at it.

    Args:
        limit: Stop after this many scans. Use a small value first -- iterating
            every scan with ``load_masks=True`` is slow and memory-hungry.
        min_raters: Skip nodule clusters annotated by fewer readers than this.
            Three is the usual floor (and matches LUNA16's acceptance criterion),
            but **record how many nodules this excludes**: nodules that only one
            reader saw are the most uncertain of all, and dropping them silently
            biases any disagreement analysis toward agreement.
        clustering_tol: Passed to ``pylidc``'s ``cluster_annotations``.
        load_masks: Also load per-reader boolean masks. Expensive.

    Returns:
        One :class:`NoduleConsensus` per accepted cluster.

    Raises:
        ImportError: If ``pylidc`` is not installed.
    """
    try:
        import pylidc as pl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_pylidc_nodules requires pylidc: pip install pylidc, and configure "
            "~/.pylidcrc with the path to the LIDC-IDRI DICOM directory "
            "(see data/README.md)"
        ) from exc

    out: list[NoduleConsensus] = []
    scans = pl.query(pl.Scan)
    for scan_index, scan in enumerate(scans):
        if limit is not None and scan_index >= limit:
            break
        try:
            clusters = scan.cluster_annotations(tol=clustering_tol)
        except Exception as exc:  # noqa: BLE001 - one bad scan must not stop the crawl
            LOG.warning("clustering failed for scan %s: %s", scan.patient_id, exc)
            continue

        for cluster_index, cluster in enumerate(clusters):
            if len(cluster) < min_raters:
                continue
            annotations: list[RaterAnnotation] = []
            for rater_index, ann in enumerate(cluster):
                semantic = {}
                for name in SEMANTIC_FEATURES:
                    value = getattr(ann, name, None)
                    if value is not None:
                        semantic[name] = float(value)
                annotations.append(
                    RaterAnnotation(
                        rater_id=rater_index,
                        malignancy=int(ann.malignancy),
                        diameter_mm=float(getattr(ann, "diameter", np.nan)),
                        volume_mm3=float(getattr(ann, "volume", np.nan)),
                        semantic=semantic,
                        mask=ann.boolean_mask() if load_masks else None,
                        bbox=ann.bbox() if load_masks else None,
                    )
                )
            spacing = (
                float(scan.slice_thickness),
                float(scan.pixel_spacing),
                float(scan.pixel_spacing),
            )
            out.append(
                NoduleConsensus(
                    nodule_id=f"{scan.patient_id}_n{cluster_index}",
                    scan_id=str(scan.patient_id),
                    annotations=annotations,
                    spacing_mm=spacing,
                    metadata={
                        "dataset": "LIDC-IDRI",
                        "slice_thickness_mm": float(scan.slice_thickness),
                        "pixel_spacing_mm": float(scan.pixel_spacing),
                        "manufacturer": getattr(scan, "manufacturer", None),
                    },
                )
            )
    return out
