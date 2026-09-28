"""Model-adapter registry.

Every model is wrapped in an adapter with ``segment(img, spacing) -> label volume``
so the benchmark loop is model-agnostic. Included:

* ``ThresholdBaseline``  — intensity-threshold + largest-component brain mask with an
  intensity-based 'lesion' class; a real, runnable floor used in tests.
* ``SynthSegAdapter``    — calls FreeSurfer's ``mri_synthseg`` CLI when available.
* ``NnUNetAdapter``      — calls ``nnUNetv2_predict`` when available.
* ``CallableAdapter``    — wraps any Python callable (e.g., a torch model's predict function).

Adapters for BrainSegFounder / SAM-Med3D are provided as ``CallableAdapter`` factories
that import the model code lazily (they need the upstream repositories and weights).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from scipy import ndimage


class SegmentationModel(Protocol):
    name: str

    def segment(self, img: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray: ...


@dataclass
class ThresholdBaseline:
    """Otsu-style two-class threshold + largest connected component; bright outliers → lesion (label 2)."""

    name: str = "threshold_baseline"
    lesion_z: float = 2.5
    smooth_sigma: float = 1.0

    @staticmethod
    def _otsu(x: np.ndarray, bins: int = 256) -> float:
        hist, edges = np.histogram(x, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        w0 = np.cumsum(hist)
        w1 = hist.sum() - w0
        m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
        m1 = (np.cumsum((hist * centers)[::-1])[::-1] / np.maximum(w1, 1))
        between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[1:]) ** 2
        return float(centers[np.argmax(between)])

    def segment(self, img: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
        x = ndimage.gaussian_filter(np.asarray(img, np.float32), self.smooth_sigma)
        thr = self._otsu(x.ravel())
        fg = x > thr
        lab, n = ndimage.label(fg)
        if n > 1:
            sizes = ndimage.sum(fg, lab, range(1, n + 1))
            fg = lab == (1 + int(np.argmax(sizes)))
        fg = ndimage.binary_fill_holes(fg)
        out = fg.astype(np.int16)
        vals = x[fg]
        if vals.size > 10:
            z = (x - np.median(vals)) / (1.4826 * np.median(np.abs(vals - np.median(vals))) + 1e-6)
            lesion = fg & (z > self.lesion_z)
            lesion = ndimage.binary_opening(lesion, iterations=1)
            out[lesion] = 2
        return out


@dataclass
class CallableAdapter:
    name: str
    fn: Callable[[np.ndarray, tuple[float, float, float]], np.ndarray]

    def segment(self, img: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
        return np.asarray(self.fn(img, spacing)).astype(np.int16)


@dataclass
class SynthSegAdapter:
    """FreeSurfer ``mri_synthseg`` (Billot et al., 2023); requires FreeSurfer ≥ 7.3 on PATH."""

    name: str = "synthseg"
    extra_args: list[str] = field(default_factory=lambda: ["--robust"])

    def available(self) -> bool:
        return shutil.which("mri_synthseg") is not None

    def segment(self, img: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
        import nibabel as nib

        if not self.available():
            raise RuntimeError("mri_synthseg not found on PATH")
        with tempfile.TemporaryDirectory() as td:
            aff = np.diag([*spacing, 1.0])
            inp, out = Path(td) / "in.nii.gz", Path(td) / "out.nii.gz"
            nib.save(nib.Nifti1Image(np.asarray(img, np.float32), aff), str(inp))
            subprocess.run(["mri_synthseg", "--i", str(inp), "--o", str(out), *self.extra_args], check=True)
            seg = nib.load(str(out))
            data = np.asarray(seg.dataobj).astype(np.int16)
            if data.shape != np.asarray(img).shape:  # synthseg resamples to 1 mm; bring back to input grid
                data = ndimage.zoom(data, [s1 / s2 for s1, s2 in zip(img.shape, data.shape)], order=0)
            return data


@dataclass
class NnUNetAdapter:
    """nnU-Net v2 inference via ``nnUNetv2_predict`` (dataset/config chosen at construction)."""

    dataset_id: int
    configuration: str = "3d_fullres"
    folds: str = "all"
    name: str = "nnunet"

    def available(self) -> bool:
        return shutil.which("nnUNetv2_predict") is not None

    def segment(self, img: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
        import nibabel as nib

        if not self.available():
            raise RuntimeError("nnUNetv2_predict not found on PATH")
        with tempfile.TemporaryDirectory() as td:
            aff = np.diag([*spacing, 1.0])
            idir, odir = Path(td) / "in", Path(td) / "out"
            idir.mkdir()
            odir.mkdir()
            nib.save(nib.Nifti1Image(np.asarray(img, np.float32), aff), str(idir / "case_0000.nii.gz"))
            subprocess.run(["nnUNetv2_predict", "-i", str(idir), "-o", str(odir), "-d", str(self.dataset_id),
                            "-c", self.configuration, "-f", self.folds], check=True)
            return np.asarray(nib.load(str(odir / "case.nii.gz")).dataobj).astype(np.int16)


def brainsegfounder_adapter(checkpoint: Path | str, task_head: Callable | None = None) -> CallableAdapter:
    """Lazy factory for BrainSegFounder (Swin UNETR); needs the upstream repo, MONAI and torch."""

    def _fn(img: np.ndarray, spacing):
        import torch  # noqa: F401
        from monai.networks.nets import SwinUNETR

        model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=task_head.out_channels if task_head else 2, feature_size=48)
        state = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.eval()
        x = torch.from_numpy(np.asarray(img, np.float32))[None, None]
        x = (x - x.mean()) / (x.std() + 1e-6)
        with torch.no_grad():
            from monai.inferers import sliding_window_inference

            logits = sliding_window_inference(x, (128, 128, 128), 2, model, overlap=0.5)
        return logits.argmax(1)[0].numpy()

    return CallableAdapter("brainsegfounder", _fn)


def sam_med3d_adapter(checkpoint: Path | str, prompt_fn: Callable[[np.ndarray], np.ndarray]) -> CallableAdapter:
    """Lazy factory for SAM-Med3D with a prompt strategy (oracle / automatic) supplied by the caller.

    ``prompt_fn(img) -> (k, 3) array of point prompts``. The upstream ``segment_anything`` (medical 3D
    fork) package must be importable; see https://github.com/uni-medical/SAM-Med3D.
    """

    def _fn(img: np.ndarray, spacing):
        raise NotImplementedError("Install SAM-Med3D (uni-medical/SAM-Med3D) and implement prompting per its inference script; "
                                  "this factory records the prompt strategy so oracle vs automatic regimes are kept separate.")

    adapter = CallableAdapter("sam_med3d", _fn)
    adapter.prompt_strategy = getattr(prompt_fn, "__name__", "custom")  # type: ignore[attr-defined]
    adapter.checkpoint = str(checkpoint)  # type: ignore[attr-defined]
    return adapter


REGISTRY: dict[str, Callable[[], SegmentationModel]] = {
    "threshold_baseline": ThresholdBaseline,
    "synthseg": SynthSegAdapter,
}


def get_model(name: str, **kwargs) -> SegmentationModel:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def run_benchmark(models: list[SegmentationModel], cases: list[tuple[str, np.ndarray, np.ndarray]],
                  labels: dict[str, int], degradations: dict[str, list[float]] | None = None) -> "pd.DataFrame":
    """Segment each case (optionally under degradations) with each model and return per-structure metrics."""
    import pandas as pd

    from .degradations import apply
    from .metrics import per_structure

    rows = []
    degradations = degradations or {"none": [0.0]}
    for cid, img, ref in cases:
        for kind, levels in degradations.items():
            for lvl in levels:
                x = img if kind == "none" else apply(img, kind, lvl, seed=hash(cid) % 2**32)
                for m in models:
                    seg = m.segment(x)
                    tab = per_structure(seg, ref, labels, distances=False)
                    tab["case"], tab["model"], tab["kind"], tab["level"] = cid, m.name, kind, lvl
                    rows.append(tab)
    return pd.concat(rows, ignore_index=True)
