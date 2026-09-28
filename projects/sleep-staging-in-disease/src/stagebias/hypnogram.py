"""Hypnogram and event ingestion for NSRR (profusion XML) and Sleep-EDF (EDF+ annotations).

All stages are mapped to the 5-class AASM convention used throughout the package::

    W = 0, N1 = 1, N2 = 2, N3 = 3, R = 4, UNSCORED = -1

R&K stages 3 and 4 are merged into N3. Epochs are 30 s unless the XML says otherwise.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

W, N1, N2, N3, R, UNSCORED = 0, 1, 2, 3, 4, -1
STAGE_NAMES: Dict[int, str] = {W: "W", N1: "N1", N2: "N2", N3: "N3", R: "R", UNSCORED: "?"}

# Compumedics profusion codes used by NSRR (SHHS, MESA, MrOS, CFS ...)
PROFUSION_CODES: Dict[int, int] = {0: W, 1: N1, 2: N2, 3: N3, 4: N3, 5: R, 9: UNSCORED}

# EDF+ annotation labels used by Sleep-EDF Expanded
EDFX_LABELS: Dict[str, int] = {
    "Sleep stage W": W, "Sleep stage 1": N1, "Sleep stage 2": N2, "Sleep stage 3": N3,
    "Sleep stage 4": N3, "Sleep stage R": R, "Sleep stage ?": UNSCORED, "Movement time": UNSCORED,
}

RESPIRATORY_KEYWORDS = ("apnea", "apnoea", "hypopnea", "hypopnoea")
AROUSAL_KEYWORDS = ("arousal",)


@dataclass
class ScoredEvents:
    """Table of scored events with columns: type, concept, start_s, duration_s, signal."""

    table: pd.DataFrame

    def respiratory(self) -> pd.DataFrame:
        m = self.table["concept"].str.lower().str.contains("|".join(RESPIRATORY_KEYWORDS), regex=True)
        return self.table[m]

    def arousals(self) -> pd.DataFrame:
        m = self.table["concept"].str.lower().str.contains("|".join(AROUSAL_KEYWORDS), regex=True)
        return self.table[m]


def parse_profusion_xml(xml_text: str) -> Tuple[np.ndarray, ScoredEvents]:
    """Parse a Compumedics *profusion* XML (NSRR annotations-events-profusion).

    Parameters
    ----------
    xml_text : str
        File contents. The document has ``<EpochLength>``, ``<SleepStages><SleepStage>k</SleepStage>...``
        and ``<ScoredEvents><ScoredEvent><EventType/><EventConcept/><Start/><Duration/>...``.

    Returns
    -------
    stages : (n_epochs,) int array in the 5-class convention (-1 = unscored)
    events : ScoredEvents
    """
    root = ET.fromstring(xml_text)
    epoch_len_el = root.find(".//EpochLength")
    epoch_len = float(epoch_len_el.text) if epoch_len_el is not None and epoch_len_el.text else 30.0
    codes = [int(el.text) for el in root.findall(".//SleepStages/SleepStage") if el.text is not None]
    stages = np.array([PROFUSION_CODES.get(c, UNSCORED) for c in codes], dtype=int)

    rows: List[dict] = []
    for ev in root.findall(".//ScoredEvents/ScoredEvent"):
        def _txt(tag: str) -> str:
            el = ev.find(tag)
            return (el.text or "").strip() if el is not None else ""
        try:
            start = float(_txt("Start"))
            dur = float(_txt("Duration"))
        except ValueError:
            continue
        rows.append({"type": _txt("EventType"), "concept": _txt("EventConcept"), "start_s": start,
                     "duration_s": dur, "signal": _txt("SignalLocation")})
    table = pd.DataFrame(rows, columns=["type", "concept", "start_s", "duration_s", "signal"])
    table.attrs["epoch_length_s"] = epoch_len
    return stages, ScoredEvents(table)


def annotations_to_epochs(onsets_s: Sequence[float], durations_s: Sequence[float], labels: Sequence[str],
                          epoch_len: float = 30.0, n_epochs: Optional[int] = None,
                          label_map: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Convert EDF+ annotation triplets (Sleep-EDF hypnogram files) to per-epoch stages.

    Each annotation covers ``duration_s`` seconds starting at ``onset_s`` (relative to the PSG start).
    Epochs not covered by any annotation are UNSCORED.
    """
    label_map = EDFX_LABELS if label_map is None else label_map
    onsets = np.asarray(onsets_s, dtype=float)
    durs = np.asarray(durations_s, dtype=float)
    if n_epochs is None:
        n_epochs = int(np.ceil((onsets + durs).max() / epoch_len)) if len(onsets) else 0
    stages = np.full(n_epochs, UNSCORED, dtype=int)
    for on, du, lab in zip(onsets, durs, labels):
        code = label_map.get(str(lab).strip(), UNSCORED)
        i0 = int(np.floor(on / epoch_len + 1e-9))
        i1 = int(np.ceil((on + du) / epoch_len - 1e-9))
        stages[max(i0, 0):min(i1, n_epochs)] = code
    return stages


def events_to_epoch_mask(events: pd.DataFrame, n_epochs: int, epoch_len: float = 30.0,
                         pad_epochs: int = 0) -> np.ndarray:
    """Boolean mask of epochs overlapping any event in ``events`` (+/- ``pad_epochs``)."""
    mask = np.zeros(n_epochs, dtype=bool)
    for start, dur in zip(events["start_s"].to_numpy(), events["duration_s"].to_numpy()):
        i0 = int(np.floor(start / epoch_len)) - pad_epochs
        i1 = int(np.floor((start + max(dur, 0.0)) / epoch_len)) + pad_epochs
        mask[max(i0, 0):min(i1 + 1, n_epochs)] = True
    return mask


def stage_fractions(stages: np.ndarray) -> Dict[str, float]:
    """Fraction of total sleep time (TST) per sleep stage plus TST in minutes and sleep efficiency."""
    s = np.asarray(stages)
    scored = s[s != UNSCORED]
    sleep = scored[scored != W]
    tst = float(len(sleep))
    out = {STAGE_NAMES[k]: (float((sleep == k).sum()) / tst if tst else np.nan) for k in (N1, N2, N3, R)}
    out["tst_min"] = tst * 0.5
    out["sleep_efficiency"] = tst / len(scored) if len(scored) else np.nan
    return out


def n3_percent(stages: np.ndarray) -> float:
    """N3 as a percentage of total sleep time."""
    return 100.0 * stage_fractions(stages)["N3"]


def expected_stage_fractions(posteriors: np.ndarray) -> Dict[str, float]:
    """Posterior-expected stage fractions from an (n_epochs, 5) probability matrix (columns W,N1,N2,N3,R).

    This is the 'expected N3%' exposure: sum over epochs of P(N3) divided by the expected TST.
    """
    p = np.asarray(posteriors, dtype=float)
    p_sleep = 1.0 - p[:, W]
    tst = float(p_sleep.sum())
    return {STAGE_NAMES[k]: float(p[:, k].sum()) / tst if tst else np.nan for k in (N1, N2, N3, R)}


def epoch_times(n_epochs: int, epoch_len: float = 30.0) -> np.ndarray:
    """Start time (s) of every epoch."""
    return np.arange(n_epochs) * epoch_len


def synthetic_hypnogram(n_epochs: int, rng: np.random.Generator, n3_fraction: float = 0.2) -> np.ndarray:
    """A crude Markov hypnogram for tests/simulations (not a physiological model)."""
    states = [W, N1, N2, N3, R]
    # transition matrix favouring stage persistence; N3 prevalence tuned by n3_fraction
    stay = 0.9
    P = np.full((5, 5), (1 - stay) / 4)
    np.fill_diagonal(P, stay)
    P[N2, N3] += n3_fraction * 0.2
    P[N2] /= P[N2].sum()
    s = np.empty(n_epochs, dtype=int)
    s[0] = W
    for i in range(1, n_epochs):
        s[i] = rng.choice(states, p=P[s[i - 1]])
    return s
