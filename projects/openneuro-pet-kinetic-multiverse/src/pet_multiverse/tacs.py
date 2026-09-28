"""BIDS-PET frame timing, tracer classification and TAC handling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_TRACERS: dict[str, tuple[str, ...]] = {
    # tracers routinely quantified with a reference region (keyword match on TracerName, lowercase)
    "reference_cerebellum": ("raclopride", "dasb", "sb207145", "cimbi", "az10419369", "way100635", "way-100635",
                             "flb457", "fallypride", "pib", "pittsburgh", "florbetapir", "av45", "av1451", "flortaucipir",
                             "mk6240", "pk11195", "altanserin", "carfentanil", "ucb-j", "ucbj", "mdl100907",
                             "yohimbine", "ps13", "mc1", "abp688", "dtbz"),
    "arterial_input": ("pbr28", "dpa714", "fdg", "fluorodeoxyglucose", "ro948", "flumazenil", "fmz", "sch23390",
                       "l-deprenyl", "deprenyl", "martinostat", "er176"),
}


def classify_tracer(tracer_name: str) -> str:
    """Rough tracer class from TracerName: 'reference_cerebellum', 'arterial_input' or 'unknown'."""
    s = str(tracer_name).lower().replace("[", "").replace("]", "").replace(" ", "")
    for cls, kws in REFERENCE_TRACERS.items():
        if any(k.replace(" ", "") in s for k in kws):
            return cls
    return "unknown"


@dataclass
class FrameTiming:
    """Frame schedule in minutes."""

    start: np.ndarray
    duration: np.ndarray

    @property
    def end(self) -> np.ndarray:
        return self.start + self.duration

    @property
    def mid(self) -> np.ndarray:
        return self.start + self.duration / 2.0

    @property
    def total_minutes(self) -> float:
        return float(self.end[-1])

    def truncate(self, max_minutes: float) -> "FrameTiming":
        keep = self.end <= max_minutes + 1e-9
        return FrameTiming(self.start[keep], self.duration[keep])

    @classmethod
    def from_sidecar(cls, sidecar: dict) -> "FrameTiming":
        """Build from a BIDS ``*_pet.json`` (FrameTimesStart/FrameDuration in seconds)."""
        start = np.asarray(sidecar["FrameTimesStart"], dtype=float) / 60.0
        dur = np.asarray(sidecar["FrameDuration"], dtype=float) / 60.0
        if len(start) != len(dur):
            raise ValueError("FrameTimesStart and FrameDuration lengths differ")
        if np.any(np.diff(start) <= 0):
            raise ValueError("FrameTimesStart must be strictly increasing")
        return cls(start, dur)

    @classmethod
    def standard_90min(cls) -> "FrameTiming":
        """A common [11C] schedule: 6×10 s, 4×30 s, 2×60 s, 2×120 s, 5×300 s, 12×600 s ... ≈ 90 min."""
        dur = np.array([10] * 6 + [30] * 4 + [60] * 2 + [120] * 2 + [300] * 5 + [600] * 6, dtype=float) / 60.0
        start = np.concatenate([[0.0], np.cumsum(dur)[:-1]])
        return cls(start, dur)


def load_sidecar(path: Path | str) -> dict:
    js = json.loads(Path(path).read_text())
    if js.get("ImageDecayCorrected") is False:
        raise ValueError(f"{path}: images are not decay corrected; correct before modelling")
    return js


def extract_tacs(pet_4d, label_img, labels: dict[str, list[int]]) -> pd.DataFrame:
    """Mean TAC per region from a 4D NIfTI (or array) and an integer label volume in the same grid.

    ``labels`` maps region name → list of label integers (e.g., FreeSurfer aparc+aseg codes).
    Accepts nibabel images or numpy arrays.
    """
    data = np.asarray(pet_4d.get_fdata() if hasattr(pet_4d, "get_fdata") else pet_4d, dtype=float)
    lab = np.asarray(label_img.get_fdata() if hasattr(label_img, "get_fdata") else label_img).astype(int)
    if data.shape[:3] != lab.shape:
        raise ValueError("PET and label volumes have different grids")
    flat = data.reshape(-1, data.shape[-1])
    lab_flat = lab.reshape(-1)
    out = {}
    for name, codes in labels.items():
        m = np.isin(lab_flat, codes)
        if m.sum() == 0:
            raise ValueError(f"region {name!r} has no voxels")
        out[name] = flat[m].mean(axis=0)
    return pd.DataFrame(out)


def load_tacs_tsv(path: Path | str) -> tuple[FrameTiming, pd.DataFrame]:
    """Read a ``*_tacs.tsv`` with ``frame_start_min``, ``frame_end_min`` + region columns."""
    df = pd.read_csv(path, sep="\t" if str(path).endswith(".tsv") else ",")
    start = df["frame_start_min"].to_numpy(float)
    end = df["frame_end_min"].to_numpy(float)
    regions = df.drop(columns=["frame_start_min", "frame_end_min"])
    return FrameTiming(start, end - start), regions


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary erosion (6-connectivity) for eroded reference regions."""
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask.astype(bool), iterations=iterations)
