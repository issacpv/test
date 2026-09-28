"""EDF loading, channel harmonization across cohorts and hypnogram parsing.

Stage codes follow YASA: W=0, N1=1, N2=2, N3=3, REM=4, unknown/artifact=-1.
MNE is imported lazily; the channel-resolution and XML parsing logic is pure python.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

W, N1, N2, N3, REM, UNK = 0, 1, 2, 3, 4, -1

#: Preferred derivations in order; the first resolvable one is used per night.
DERIVATION_PRIORITY: Tuple[str, ...] = ("C4-M1", "C3-M2", "Fz-Cz", "Fpz-Cz", "F4-M1", "Cz-Oz", "Pz-Oz")

#: Per-cohort raw channel labels for canonical derivations or electrodes.
COHORT_CHANNEL_MAPS: Dict[str, Dict[str, Sequence[str]]] = {
    "sleep-edf": {"Fpz-Cz": ("EEG Fpz-Cz",), "Pz-Oz": ("EEG Pz-Oz",)},
    "shhs": {"C4-M1": ("EEG", "EEG C4-A1", "C4-A1"), "C3-M2": ("EEG(sec)", "EEG(SEC)", "EEG 2", "EEG2", "EEG sec", "C3-A2")},
    "mesa": {"Fz-Cz": ("EEG1", "EEG Fz-Cz"), "Cz-Oz": ("EEG2", "EEG Cz-Oz"), "C4-M1": ("EEG3", "EEG C4-M1")},
    "mros": {"C3": ("C3",), "C4": ("C4",), "M1": ("A1", "M1"), "M2": ("A2", "M2")},
    "cfs": {"C3": ("C3",), "C4": ("C4",), "M1": ("M1", "A1"), "M2": ("M2", "A2")},
    "hmc": {"C4-M1": ("EEG C4-M1",), "C3-M2": ("EEG C3-M2",), "F4-M1": ("EEG F4-M1",), "O2-M1": ("EEG O2-M1",)},
}

_ALIAS = {"A1": "M1", "A2": "M2", "FP1": "FP1", "FPZ": "FPZ"}


def _norm(label: str) -> str:
    s = re.sub(r"^(EEG|EOG|EMG|ECG)\s*", "", label.strip(), flags=re.IGNORECASE).strip().upper().replace(" ", "")
    if "-" in s:
        a, b = s.split("-", 1)
        return f"{_ALIAS.get(a, a)}-{_ALIAS.get(b, b)}"
    return _ALIAS.get(s, s)


def resolve_channel(ch_names: Sequence[str], derivation: str, cohort: Optional[str] = None
                    ) -> Optional[Union[str, Tuple[str, str]]]:
    """Find how to obtain ``derivation`` (e.g. ``'C4-M1'``) from the raw channel list.

    Returns the raw channel name if the derivation exists directly, a
    ``(anode_raw, cathode_raw)`` tuple if it must be computed by re-referencing,
    or ``None`` if unavailable. Cohort-specific label maps are tried first, then
    generic label normalization (``'EEG C4-A1'`` == ``'C4-M1'``).
    """
    cmap = COHORT_CHANNEL_MAPS.get(cohort or "", {})
    lookup = {n: n for n in ch_names}
    for raw in cmap.get(derivation, ()):
        if raw in lookup:
            return raw
    norm = {}
    for n in ch_names:
        norm.setdefault(_norm(n), n)
    d = _norm(derivation)
    if d in norm:
        return norm[d]
    a, b = d.split("-") if "-" in d else (d, None)
    if b is None:
        return None
    ra = None
    for raw in cmap.get(a, ()):
        if raw in lookup:
            ra = raw
    ra = ra or norm.get(a)
    rb = None
    for raw in cmap.get(b, ()):
        if raw in lookup:
            rb = raw
    rb = rb or norm.get(b)
    if ra is not None and rb is not None:
        return (ra, rb)
    return None


def pick_derivation(ch_names: Sequence[str], cohort: Optional[str] = None,
                    priority: Sequence[str] = DERIVATION_PRIORITY) -> Tuple[Optional[str], Optional[Union[str, Tuple[str, str]]]]:
    """First derivation in ``priority`` that can be resolved; returns ``(derivation, how)``."""
    for d in priority:
        how = resolve_channel(ch_names, d, cohort)
        if how is not None:
            return d, how
    return None, None


def preprocess(x: np.ndarray, fs: float, target_fs: float = 100.0, l_freq: float = 0.3, h_freq: float = 35.0) -> Tuple[np.ndarray, float]:
    """Band-pass (zero-phase Butterworth, order 4) then polyphase-resample a 1-D signal."""
    x = np.asarray(x, float)
    nyq = fs / 2
    sos = butter(4, [l_freq / nyq, min(h_freq, 0.95 * nyq) / nyq], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    if not np.isclose(fs, target_fs):
        from fractions import Fraction

        fr = Fraction(target_fs / fs).limit_denominator(1000)
        y = resample_poly(y, fr.numerator, fr.denominator)
    return y, float(target_fs)


@dataclass
class Night:
    signal: np.ndarray     # 1-D, microvolts
    fs: float
    derivation: str
    hypno: np.ndarray      # per-epoch stage codes
    epoch_len: float
    cohort: str
    night_id: str
    meta: dict

    @property
    def hypno_samples(self) -> np.ndarray:
        return hypno_to_samples(self.hypno, self.epoch_len, self.fs, len(self.signal))


def load_eeg(path: Union[str, Path], cohort: str, derivation: Optional[str] = None,
             target_fs: float = 100.0) -> Tuple[np.ndarray, float, str]:
    """Load one EEG derivation from an EDF with MNE (microvolts), harmonized to ``target_fs``.

    Returns ``(signal, fs, derivation_used)``.
    """
    import mne  # lazy

    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="error")
    if derivation is None:
        derivation, how = pick_derivation(raw.ch_names, cohort)
    else:
        how = resolve_channel(raw.ch_names, derivation, cohort)
    if how is None:
        raise ValueError(f"no usable EEG derivation in {path}: {raw.ch_names}")
    if isinstance(how, tuple):
        a = raw.get_data(picks=[how[0]])[0]
        b = raw.get_data(picks=[how[1]])[0]
        x = (a - b) * 1e6
    else:
        x = raw.get_data(picks=[how])[0] * 1e6
    y, fs = preprocess(x, float(raw.info["sfreq"]), target_fs)
    return y, fs, derivation


# --------------------------------------------------------------------------- #
# hypnograms
# --------------------------------------------------------------------------- #
_NSRR_STAGE = {"0": W, "1": N1, "2": N2, "3": N3, "4": N3, "5": REM, "9": UNK}


def parse_nsrr_xml(text: str, epoch_len: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Parse an NSRR ``*-nsrr.xml`` (Profusion-style) annotation into per-epoch stages.

    Stage events have ``EventType`` ``'Stages|Stages'`` and ``EventConcept`` like
    ``'Stage 2 sleep|2'``; ``Start``/``Duration`` are seconds. Stage 4 is merged into N3.
    Returns ``(hypno, epoch_len)``.
    """
    root = ET.fromstring(text)
    el = root.find("EpochLength")
    ep = float(el.text) if (el is not None and el.text) else (epoch_len or 30.0)
    events = []
    for ev in root.iter("ScoredEvent"):
        etype = (ev.findtext("EventType") or "")
        if not etype.lower().startswith("stages"):
            continue
        concept = ev.findtext("EventConcept") or ""
        code = concept.split("|")[-1].strip()
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)
        events.append((start, dur, _NSRR_STAGE.get(code, UNK)))
    if not events:
        return np.empty(0, int), ep
    total = max(s + d for s, d, _ in events)
    n_ep = int(round(total / ep))
    hyp = np.full(n_ep, UNK, dtype=int)
    for s, d, st in events:
        i0, i1 = int(round(s / ep)), int(round((s + d) / ep))
        hyp[i0:i1] = st
    return hyp, ep


def load_hypnogram_nsrr(path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    return parse_nsrr_xml(Path(path).read_text(), None)


_SLEEPEDF_STAGE = {"Sleep stage W": W, "Sleep stage 1": N1, "Sleep stage 2": N2, "Sleep stage 3": N3,
                   "Sleep stage 4": N3, "Sleep stage R": REM, "Sleep stage ?": UNK, "Movement time": UNK}


def annotations_to_hypno(onsets: Sequence[float], durations: Sequence[float], descriptions: Sequence[str],
                         epoch_len: float = 30.0, total_dur: Optional[float] = None) -> np.ndarray:
    """Convert (onset, duration, description) annotations (Sleep-EDF style) into per-epoch stages."""
    total = total_dur if total_dur is not None else max((o + d for o, d in zip(onsets, durations)), default=0.0)
    hyp = np.full(int(round(total / epoch_len)), UNK, dtype=int)
    for o, d, desc in zip(onsets, durations, descriptions):
        st = _SLEEPEDF_STAGE.get(str(desc).strip(), UNK)
        hyp[int(round(o / epoch_len)): int(round((o + d) / epoch_len))] = st
    return hyp


def load_hypnogram_sleepedf(path: Union[str, Path], epoch_len: float = 30.0) -> np.ndarray:
    """Read a Sleep-EDF ``*-Hypnogram.edf`` (EDF+ annotations) with MNE."""
    import mne  # lazy

    ann = mne.read_annotations(str(path))
    return annotations_to_hypno(ann.onset, ann.duration, ann.description, epoch_len)


def hypno_to_samples(hypno: np.ndarray, epoch_len: float, fs: float, n_samples: int) -> np.ndarray:
    """Upsample per-epoch stages to per-sample codes of length ``n_samples`` (pad with UNK)."""
    per = int(round(epoch_len * fs))
    out = np.repeat(np.asarray(hypno, int), per)
    if len(out) >= n_samples:
        return out[:n_samples]
    return np.concatenate([out, np.full(n_samples - len(out), UNK, int)])


def stage_minutes(hypno: np.ndarray, epoch_len: float, stages: Sequence[int] = (N2, N3)) -> float:
    return float(np.isin(hypno, list(stages)).sum() * epoch_len / 60.0)


def load_night(edf_path: Union[str, Path], hypno_path: Union[str, Path], cohort: str,
               derivation: Optional[str] = None, target_fs: float = 100.0, epoch_len: float = 30.0) -> Night:
    """Convenience loader combining EEG and hypnogram for one night (requires MNE)."""
    sig, fs, der = load_eeg(edf_path, cohort, derivation, target_fs)
    if str(hypno_path).lower().endswith(".xml"):
        hyp, ep = load_hypnogram_nsrr(hypno_path)
    else:
        hyp, ep = load_hypnogram_sleepedf(hypno_path, epoch_len), epoch_len
    return Night(sig, fs, der, hyp, ep, cohort, Path(edf_path).stem, {"edf": str(edf_path)})
