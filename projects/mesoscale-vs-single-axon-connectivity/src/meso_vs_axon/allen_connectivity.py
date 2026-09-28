"""Allen Mouse Brain Connectivity Atlas access: structure-level projection matrices and vectors.

Wraps ``allensdk.core.mouse_connectivity_cache.MouseConnectivityCache``. Everything that needs
allensdk is inside ``ConnectivityFetcher``; the pure-pandas helpers at the bottom work on the CSV
files written by ``scripts/download_data.py`` so analyses can run without allensdk installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache  # type: ignore

    HAVE_ALLENSDK = True
except Exception:  # noqa: BLE001
    MouseConnectivityCache = None  # type: ignore
    HAVE_ALLENSDK = False

#: "Mouse Connectivity - Summary" structure set (the ~300 summary structures of Oh et al. 2014).
SUMMARY_STRUCTURE_SET_ID = 167587189
HEMI_LEFT, HEMI_RIGHT, HEMI_BOTH = 1, 2, 3
#: Allen injections are in the right hemisphere: ipsi = right, contra = left.
HEMI_LABEL = {HEMI_LEFT: "contra", HEMI_RIGHT: "ipsi", HEMI_BOTH: "both"}


class ConnectivityFetcher:
    """Fetch and cache Allen connectivity data at structure level."""

    def __init__(self, cache_dir: Path, resolution: int = 25) -> None:
        if not HAVE_ALLENSDK:
            raise ImportError("allensdk is required: pip install allensdk")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mcc = MouseConnectivityCache(manifest_file=str(self.cache_dir / "manifest.json"), resolution=resolution)
        self._tree = None
        self._summary: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ ontology
    def structure_tree(self):
        if self._tree is None:
            self._tree = self.mcc.get_structure_tree()
        return self._tree

    def summary_structures(self) -> pd.DataFrame:
        if self._summary is None:
            tree = self.structure_tree()
            rows = tree.get_structures_by_set_id([SUMMARY_STRUCTURE_SET_ID])
            self._summary = pd.DataFrame([{"id": r["id"], "acronym": r["acronym"], "name": r["name"],
                                           "graph_order": r.get("graph_order")} for r in rows])
        return self._summary

    def ancestor_map(self) -> Dict[int, List[int]]:
        """structure id -> list of ancestor ids (self first), from the ontology."""
        return self.structure_tree().get_ancestor_id_map()

    def annotation(self) -> Tuple[np.ndarray, dict]:
        """CCF annotation volume (structure ids) and its metadata (spacing in µm)."""
        return self.mcc.get_annotation_volume()

    # ------------------------------------------------------------------ experiments
    def experiments(self, cre: Optional[bool] = False,
                    injection_structure_acronyms: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Experiments table; ``cre=False`` wild-type only, ``None`` all, ``True`` Cre lines only."""
        ids = None
        if injection_structure_acronyms:
            tree = self.structure_tree()
            ids = [s["id"] for s in tree.get_structures_by_acronym(list(injection_structure_acronyms))]
        df = self.mcc.get_experiments(dataframe=True, cre=cre, injection_structure_ids=ids)
        return df.reset_index(drop=True)

    def projection_matrix(self, experiment_ids: Sequence[int], parameter: str = "normalized_projection_volume",
                          hemisphere_ids: Sequence[int] = (HEMI_LEFT, HEMI_RIGHT),
                          target_structure_ids: Optional[Sequence[int]] = None) -> pd.DataFrame:
        """Experiments x (target structure, hemisphere) matrix of the chosen unionize parameter."""
        if target_structure_ids is None:
            target_structure_ids = self.summary_structures()["id"].tolist()
        pm = self.mcc.get_projection_matrix(experiment_ids=list(experiment_ids),
                                            projection_structure_ids=list(target_structure_ids),
                                            hemisphere_ids=list(hemisphere_ids), parameter=parameter)
        acr = dict(zip(self.summary_structures()["id"], self.summary_structures()["acronym"]))
        cols = [f"{acr.get(c['structure_id'], c['structure_id'])}_{HEMI_LABEL.get(c['hemisphere_id'], c['hemisphere_id'])}"
                for c in pm["columns"]]
        return pd.DataFrame(pm["matrix"], index=pd.Index(pm["rows"], name="experiment_id"), columns=cols)

    def injection_fractions(self, experiment_ids: Sequence[int]) -> pd.DataFrame:
        """Fraction of the injection volume inside each summary structure (for injection QC)."""
        u = self.mcc.get_structure_unionizes(list(experiment_ids), is_injection=True,
                                             structure_ids=self.summary_structures()["id"].tolist(),
                                             hemisphere_ids=[HEMI_BOTH], include_descendants=True)
        piv = u.pivot_table(index="experiment_id", columns="structure_id", values="projection_volume", aggfunc="sum")
        return piv.div(piv.sum(axis=1), axis=0)


# ---------------------------------------------------------------------- pure pandas helpers
def region_projection_vector(matrix: pd.DataFrame, experiments: pd.DataFrame, source_acronym: str,
                             min_injection_fraction: float = 0.5,
                             injection_fractions: Optional[pd.DataFrame] = None,
                             agg: str = "mean") -> pd.Series:
    """Average projection vector of experiments injected in ``source_acronym``.

    ``experiments`` needs columns ``id`` and ``structure_abbrev`` (primary injection structure, as
    returned by allensdk). If ``injection_fractions`` (experiments x structure acronym) is given,
    experiments with fraction < ``min_injection_fraction`` in the source are dropped.
    """
    col = "structure_abbrev" if "structure_abbrev" in experiments.columns else "primary_injection_structure"
    ids = experiments.loc[experiments[col] == source_acronym, "id"].tolist()
    if injection_fractions is not None and source_acronym in injection_fractions.columns:
        ok = injection_fractions.loc[injection_fractions.index.isin(ids), source_acronym] >= min_injection_fraction
        ids = ok[ok].index.tolist()
    sub = matrix.loc[matrix.index.isin(ids)]
    if sub.empty:
        raise ValueError(f"no experiments for source {source_acronym}")
    vec = sub.mean(axis=0) if agg == "mean" else sub.median(axis=0)
    vec.name = source_acronym
    return vec


def normalize_vector(v: pd.Series, drop_self: Optional[str] = None, log1p: bool = False) -> pd.Series:
    """Non-negative, optionally self-excluded and log-compressed vector normalised to sum 1."""
    x = v.clip(lower=0).astype(float)
    if drop_self is not None:
        x = x[~x.index.str.startswith(drop_self + "_")]
    if log1p:
        x = np.log1p(x / max(x[x > 0].min(), 1e-12))
    s = x.sum()
    return x / s if s > 0 else x


def load_projection_table(allen_dir: Path, parameter: str = "normalized_projection_volume") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the matrix and experiment table written by ``scripts/download_data.py``."""
    allen_dir = Path(allen_dir)
    mat = pd.read_csv(allen_dir / f"projection_matrix_{parameter}.csv", index_col=0)
    exps = pd.read_csv(allen_dir / "experiments.csv")
    return mat, exps


def voxel_model_row(soma_xyz_um: Sequence[float], cache_dir: Path):  # pragma: no cover - optional dependency
    """Predicted projection (per target voxel) from the Knox et al. 2019 voxel model at a soma location.

    Requires ``mcmodels`` (github.com/AllenInstitute/mouse_connectivity_models). Returns the voxel
    row and the target mask so it can be aggregated per structure with the annotation volume.
    """
    from mcmodels.core import VoxelModelCache  # type: ignore

    cache = VoxelModelCache(manifest_file=str(Path(cache_dir) / "voxel_model_manifest.json"))
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
    coords = np.asarray(source_mask.coordinates)  # in 100 µm voxel units
    soma_vox = np.asarray(soma_xyz_um, float) / 100.0
    k = int(np.argmin(np.linalg.norm(coords - soma_vox, axis=1)))
    return np.asarray(voxel_array[k]), target_mask


__all__ = ["HAVE_ALLENSDK", "SUMMARY_STRUCTURE_SET_ID", "HEMI_LABEL", "ConnectivityFetcher",
           "region_projection_vector", "normalize_vector", "load_projection_table", "voxel_model_row"]
