"""Loaders for FreeSurfer ``*.stats`` tables.

FreeSurfer writes two kinds of information into a stats file:

* ``# Measure`` header lines, e.g.
  ``# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1512345.0, mm^3``
* a whitespace-delimited table whose column names are given by the
  ``# ColHeaders`` line.

Both HCP (``<subject>/T1w/<subject>/stats/``) and OASIS-3 FreeSurfer
assessors (``<fs_id>/DATA/<subject>/stats/``) follow this layout, so a single
parser serves both cohorts.  The feature-table builder produces the standard
"FreeSurfer brain-age feature vector" used by e.g. More et al. (2023):
subcortical volumes (aseg), regional thickness / surface area / grey-matter
volume (Desikan-Killiany aparc), and global measures (eTIV, total grey, WM).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Subcortical structures conventionally used as brain-age features (aseg).
ASEG_STRUCTURES: List[str] = [
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
    "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter", "Right-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
    "Left-Thalamus", "Right-Thalamus",  # "Left-Thalamus-Proper" in FS<7
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area",
    "Left-VentralDC", "Right-VentralDC",
    "3rd-Ventricle", "4th-Ventricle", "Brain-Stem", "CSF",
    "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior",
]

# Global "# Measure" short-names that are kept as features / covariates.
ASEG_GLOBAL_MEASURES: List[str] = [
    "eTIV", "BrainSegVolNotVent", "TotalGrayVol", "CortexVol",
    "CerebralWhiteMatterVol", "SubCortGrayVol", "VentricleChoroidVol",
]

APARC_METRICS: List[str] = ["ThickAvg", "SurfArea", "GrayVol"]


@dataclass
class StatsTable:
    """Parsed content of one FreeSurfer stats file."""

    measures: Dict[str, float] = field(default_factory=dict)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    header: Dict[str, str] = field(default_factory=dict)


_MEASURE_RE = re.compile(r"^#\s*Measure\s+(.*)$")


def parse_stats_file(path: PathLike) -> StatsTable:
    """Parse a FreeSurfer ``.stats`` file into measures + a table.

    Parameters
    ----------
    path
        Path to ``aseg.stats``, ``lh.aparc.stats``, ``wmparc.stats`` ...

    Returns
    -------
    StatsTable
        ``measures`` maps the *short* name (e.g. ``eTIV``) to its float value;
        ``table`` has one row per structure with columns from ``# ColHeaders``.

    Notes
    -----
    Numeric columns are coerced with ``pd.to_numeric``; ``StructName`` stays a
    string.  Lines that are neither comments nor complete rows are skipped.
    """
    path = Path(path)
    measures: Dict[str, float] = {}
    header: Dict[str, str] = {}
    col_headers: Optional[List[str]] = None
    rows: List[List[str]] = []

    with path.open("r", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                m = _MEASURE_RE.match(line)
                if m:
                    parts = [p.strip() for p in m.group(1).split(",")]
                    # Measure <name>, <short>, <description>, <value>, <unit>
                    if len(parts) >= 4:
                        short = parts[1]
                        try:
                            measures[short] = float(parts[3])
                        except ValueError:
                            pass
                    continue
                if line.startswith("# ColHeaders"):
                    col_headers = line.replace("# ColHeaders", "").split()
                    continue
                kv = line[1:].strip().split(" ", 1)
                if len(kv) == 2:
                    header.setdefault(kv[0], kv[1].strip())
                continue
            rows.append(line.split())

    if col_headers is None:
        raise ValueError(f"No '# ColHeaders' line found in {path}")

    rows = [r for r in rows if len(r) == len(col_headers)]
    table = pd.DataFrame(rows, columns=col_headers)
    for col in table.columns:
        if col != "StructName":
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return StatsTable(measures=measures, table=table, header=header)


def _thalamus_alias(names: Iterable[str]) -> Dict[str, str]:
    """Map FreeSurfer<7 'Thalamus-Proper' labels onto the FS7 names."""
    out = {}
    for n in names:
        if n.endswith("Thalamus-Proper"):
            out[n] = n.replace("Thalamus-Proper", "Thalamus")
    return out


def load_aseg_features(
    aseg_path: PathLike,
    structures: Sequence[str] = ASEG_STRUCTURES,
    global_measures: Sequence[str] = ASEG_GLOBAL_MEASURES,
) -> pd.Series:
    """Extract subcortical volumes and global measures from ``aseg.stats``.

    Returns a Series indexed like ``aseg_<StructName>`` and ``global_<short>``.
    Missing structures are ``NaN`` (they are reported, not silently dropped).
    """
    st = parse_stats_file(aseg_path)
    tab = st.table.copy()
    tab["StructName"] = tab["StructName"].replace(_thalamus_alias(tab["StructName"]))
    vol = tab.set_index("StructName")["Volume_mm3"]
    out = {f"aseg_{s}": float(vol.get(s, np.nan)) for s in structures}
    for g in global_measures:
        out[f"global_{g}"] = float(st.measures.get(g, np.nan))
    return pd.Series(out, dtype=float)


def load_aparc_features(
    aparc_path: PathLike,
    hemi: str,
    metrics: Sequence[str] = APARC_METRICS,
) -> pd.Series:
    """Extract regional cortical metrics from ``?h.aparc.stats``.

    Series index is ``aparc_<hemi>_<region>_<metric>``, e.g.
    ``aparc_lh_precuneus_ThickAvg``.  Also includes ``aparc_<hemi>_MeanThickness``
    from the ``# Measure Cortex, MeanThickness`` line when present.
    """
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be 'lh' or 'rh'")
    st = parse_stats_file(aparc_path)
    tab = st.table.set_index("StructName")
    out: Dict[str, float] = {}
    for region, row in tab.iterrows():
        for m in metrics:
            if m in tab.columns:
                out[f"aparc_{hemi}_{region}_{m}"] = float(row[m])
    if "MeanThickness" in st.measures:
        out[f"aparc_{hemi}_MeanThickness"] = float(st.measures["MeanThickness"])
    if "WhiteSurfArea" in st.measures:
        out[f"aparc_{hemi}_WhiteSurfArea"] = float(st.measures["WhiteSurfArea"])
    return pd.Series(out, dtype=float)


def load_subject_features(stats_dir: PathLike, aparc_name: str = "aparc") -> pd.Series:
    """Combine aseg + lh/rh aparc features from one ``stats/`` directory."""
    stats_dir = Path(stats_dir)
    parts = [load_aseg_features(stats_dir / "aseg.stats")]
    for hemi in ("lh", "rh"):
        p = stats_dir / f"{hemi}.{aparc_name}.stats"
        if p.exists():
            parts.append(load_aparc_features(p, hemi))
    return pd.concat(parts)


def build_feature_table(
    stats_dirs: Dict[str, PathLike],
    aparc_name: str = "aparc",
    icv_normalise: bool = False,
) -> pd.DataFrame:
    """Build a subjects x features table from many ``stats/`` directories.

    Parameters
    ----------
    stats_dirs
        Mapping ``session_id -> path/to/stats``.  Use a *session* (scan) id as
        key so that longitudinal OASIS-3 visits stay distinct rows.
    aparc_name
        ``"aparc"`` (Desikan-Killiany, 34 regions/hemi) or ``"aparc.a2009s"``.
    icv_normalise
        If True, divide every volumetric feature (aseg volumes, GrayVol) by
        ``global_eTIV`` (proportional ICV correction).  Thickness is untouched.

    Returns
    -------
    pd.DataFrame indexed by session id.  Rows with unreadable files are kept
    as all-NaN so failures are visible.
    """
    records = {}
    for sid, d in stats_dirs.items():
        try:
            records[sid] = load_subject_features(d, aparc_name=aparc_name)
        except (FileNotFoundError, ValueError):
            records[sid] = pd.Series(dtype=float)
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "session_id"
    if icv_normalise and "global_eTIV" in df.columns:
        vol_cols = [c for c in df.columns if c.startswith("aseg_") or c.endswith("_GrayVol")]
        df[vol_cols] = df[vol_cols].div(df["global_eTIV"], axis=0)
    return df


def read_stats2table(path: PathLike, index_col: int = 0) -> pd.DataFrame:
    """Read the TSV emitted by ``asegstats2table`` / ``aparcstats2table``.

    These commands (FreeSurfer) write one row per subject with the first
    column named e.g. ``Measure:volume`` or ``lh.aparc.thickness``.  The
    result is indexed by subject id with float columns.
    """
    df = pd.read_csv(path, sep="\t", index_col=index_col)
    df.index.name = "session_id"
    return df.apply(pd.to_numeric, errors="coerce")


def feature_groups(columns: Iterable[str]) -> Dict[str, List[str]]:
    """Split feature names into families (useful for ablations / plots)."""
    groups: Dict[str, List[str]] = {"aseg": [], "thickness": [], "area": [], "volume": [], "global": []}
    for c in columns:
        if c.startswith("aseg_"):
            groups["aseg"].append(c)
        elif c.endswith("_ThickAvg") or c.endswith("MeanThickness"):
            groups["thickness"].append(c)
        elif c.endswith("_SurfArea") or c.endswith("WhiteSurfArea"):
            groups["area"].append(c)
        elif c.endswith("_GrayVol"):
            groups["volume"].append(c)
        elif c.startswith("global_"):
            groups["global"].append(c)
    return groups
