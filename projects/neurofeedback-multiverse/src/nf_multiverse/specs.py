"""The specification space of an EEG neurofeedback signal.

Every online neurofeedback system makes the same seven decisions when turning a
raw EEG stream into the number shown to the participant.  Papers report at most
two or three of them.  We make all of them explicit so that the *same* recording
can be re-scored under every combination (a multiverse; Steegen et al., 2016)
and the resulting learning estimates can be summarised as a specification curve
(Simonsohn, Simmons & Nelson, 2020).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeedbackSpec:
    """One way of computing a feedback signal.

    Attributes
    ----------
    reference:
        ``'none'`` (recording reference), ``'car'`` (common average),
        ``'laplacian'`` (target minus mean of its neighbours), ``'mastoid'``
        (linked mastoids M1/M2 if present).
    band:
        ``'fixed'`` uses ``band_low``-``band_high``; ``'individual'`` centres a
        band of the same width on the participant's individual alpha frequency
        estimated from the baseline (Klimesch, 1999).
    estimator:
        ``'welch'`` (mean PSD in band), ``'fft'`` (single periodogram),
        ``'hilbert'`` (band-pass + analytic envelope, mean squared),
        ``'bandpass_rms'`` (band-pass + RMS).
    window_s, step_s:
        Sliding-window length and hop, in seconds (the online update rate).
    normalization:
        ``'none'``, ``'baseline_z'``, ``'baseline_ratio'``, ``'log_ratio'``.
    artifact:
        ``'none'``, ``'threshold'`` (freeze feedback when peak-to-peak exceeds
        ``artifact_threshold_uv``), ``'regress_eog'`` (regress the low-pass
        frontal EOG proxy out of the target signal before power estimation).
    smoothing_windows:
        Moving-average length (in windows) applied to the feedback values.
    """

    reference: str = "car"
    band: str = "fixed"
    band_low: float = 8.0
    band_high: float = 12.0
    estimator: str = "welch"
    window_s: float = 1.0
    step_s: float = 0.25
    normalization: str = "baseline_z"
    artifact: str = "none"
    artifact_threshold_uv: float = 100.0
    smoothing_windows: int = 1

    def name(self) -> str:
        return (
            f"{self.reference}|{self.band}[{self.band_low:g}-{self.band_high:g}]|{self.estimator}|"
            f"w{self.window_s:g}s{self.step_s:g}|{self.normalization}|{self.artifact}|sm{self.smoothing_windows}"
        )

    def with_(self, **kw) -> "FeedbackSpec":
        return replace(self, **kw)


#: default factor levels (5 x 2 x 4 x 3 x 3 x 3 x 2 = 2,160 specifications before pruning)
DEFAULT_LEVELS: Dict[str, Sequence] = {
    "reference": ("none", "car", "laplacian", "mastoid"),
    "band": ("fixed", "individual"),
    "estimator": ("welch", "fft", "hilbert", "bandpass_rms"),
    "window": ((0.5, 0.125), (1.0, 0.25), (2.0, 0.5)),
    "normalization": ("baseline_z", "baseline_ratio", "log_ratio"),
    "artifact": ("none", "threshold", "regress_eog"),
    "smoothing_windows": (1, 4),
}


def build_specification_space(levels: Optional[Dict[str, Sequence]] = None, base: Optional[FeedbackSpec] = None, max_specs: Optional[int] = None) -> List[FeedbackSpec]:
    """Cartesian product of factor levels (optionally truncated deterministically)."""
    lv = dict(DEFAULT_LEVELS)
    if levels:
        lv.update(levels)
    base = base or FeedbackSpec()
    out: List[FeedbackSpec] = []
    for ref, band, est, (w, s), norm, art, sm in product(lv["reference"], lv["band"], lv["estimator"], lv["window"], lv["normalization"], lv["artifact"], lv["smoothing_windows"]):
        out.append(base.with_(reference=ref, band=band, estimator=est, window_s=w, step_s=s, normalization=norm, artifact=art, smoothing_windows=sm))
    if max_specs is not None and len(out) > max_specs:
        stride = max(1, len(out) // max_specs)
        out = out[::stride][:max_specs]
    return out


def spec_table(specs: Iterable[FeedbackSpec]) -> pd.DataFrame:
    """One row per specification with all factor columns and the name."""
    rows = []
    for i, s in enumerate(specs):
        d = asdict(s)
        d["spec_id"] = i
        d["name"] = s.name()
        rows.append(d)
    return pd.DataFrame(rows)
