"""Label CGM segments by insulin-delivery regimen.

Regimen is the distribution-shift axis of the study. Labels are assigned from
device/algorithm metadata (pump model, AID algorithm, device mode). Datasets
map on as follows (defaults, overridable):
- OhioT1DM   -> open-loop
- AZT1D      -> hybrid-closed-loop (Tandem Control-IQ), device_mode available
- DiaTrend   -> per-stream (open-loop or hybrid-closed-loop) from pump model
- OpenAPS    -> DIY-loop
"""
from __future__ import annotations

import numpy as np

REGIMENS = ["open-loop", "hybrid-closed-loop", "DIY-loop"]

DATASET_DEFAULT = {
    "ohiot1dm": "open-loop",
    "azt1d": "hybrid-closed-loop",
    "diatrend": "open-loop",     # override per-subject where AID is documented
    "openaps": "DIY-loop",
    "t1dexi": "open-loop",
}

# Pump/algorithm strings that indicate automated insulin delivery.
AID_MARKERS = ["control-iq", "control iq", "780g", "smartguard", "openaps",
               "androidaps", "loop", "fx2", "camaps"]


def label_dataset(dataset: str) -> str:
    """Default regimen for a dataset."""
    return DATASET_DEFAULT.get(dataset.lower(), "open-loop")


def label_from_metadata(pump_model: str | None, algorithm: str | None) -> str:
    """Infer regimen from free-text pump/algorithm metadata."""
    text = f"{pump_model or ''} {algorithm or ''}".lower()
    if any(m in text for m in ("openaps", "androidaps", "loop")):
        return "DIY-loop"
    if any(m in text for m in AID_MARKERS):
        return "hybrid-closed-loop"
    return "open-loop"


def aid_active(device_mode) -> np.ndarray:
    """Boolean per-sample indicator that AID was actively modulating insulin.

    Uses the device_mode field (AZT1D: regular/sleep/exercise are all AID-active;
    'unknown'/'manual'/'open' are not).
    """
    dm = np.asarray(device_mode).astype(str)
    inactive = np.isin(np.char.lower(dm), ["unknown", "manual", "open", "suspend"])
    return ~inactive
