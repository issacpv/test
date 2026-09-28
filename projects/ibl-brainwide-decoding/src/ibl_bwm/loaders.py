"""Session containers and loaders for IBL (ONE API) and Allen Visual Coding (NWB) Neuropixels data.

Heavy dependencies (`one.api`, `brainbox`, `iblatlas`, `pynwb`, `allensdk`) are imported lazily inside
the functions that need them, so the module (and the synthetic generator used by the tests) works with
numpy/pandas only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"
ONE_PASSWORD = "international"

# IBL trials fields used downstream
TRIAL_COLUMNS = ("choice", "contrastLeft", "contrastRight", "probabilityLeft", "feedbackType",
                 "stimOn_times", "firstMovement_times", "feedback_times", "response_times")


@dataclass
class SessionData:
    """One insertion (probe) of one session, with trials and per-cluster region labels."""

    eid: str
    pid: str
    lab: str
    subject: str
    spike_times: np.ndarray          # (n_spikes,) seconds, sorted
    spike_clusters: np.ndarray       # (n_spikes,) cluster id per spike
    cluster_ids: np.ndarray          # (n_clusters,)
    cluster_regions: np.ndarray      # (n_clusters,) Beryl acronym
    cluster_good: np.ndarray         # (n_clusters,) bool QC flag
    trials: pd.DataFrame             # columns as in TRIAL_COLUMNS
    dataset: str = "ibl"
    meta: Dict[str, object] = field(default_factory=dict)

    def units_in_region(self, region: str, good_only: bool = True) -> np.ndarray:
        m = self.cluster_regions == region
        if good_only:
            m &= self.cluster_good
        return self.cluster_ids[m]

    def regions(self, min_units: int = 10, good_only: bool = True) -> List[str]:
        regs, counts = np.unique(self.cluster_regions[self.cluster_good] if good_only else self.cluster_regions,
                                 return_counts=True)
        return [r for r, c in zip(regs, counts) if c >= min_units and r not in ("void", "root")]


# ----------------------------------------------------------------------------- region mapping
def assign_beryl(acronyms: Sequence[str]) -> np.ndarray:
    """Map Allen CCF acronyms to the IBL 'Beryl' parcellation via iblatlas; identity fallback.

    The fallback keeps the analysis runnable without iblatlas (e.g. tests), but real analyses must
    use the Beryl mapping so that layer-specific cortical acronyms (VISp2/3, VISp5 ...) are merged.
    """
    acr = np.asarray(acronyms, dtype=object)
    try:
        from iblatlas.regions import BrainRegions
        br = BrainRegions()
        return np.asarray(br.acronym2acronym(acr, mapping="Beryl"), dtype=object)
    except Exception:  # noqa: BLE001 - iblatlas missing or acronym unknown
        return acr


def qc_good(clusters: Mapping[str, np.ndarray], min_fr: float = 0.1, min_presence: float = 0.5,
            max_rp_violation: float = 0.1) -> np.ndarray:
    """Good-unit flag: IBL `label == 1` if present, else a firing-rate / presence / RP-violation rule."""
    if "label" in clusters:
        return np.asarray(clusters["label"]) >= 1
    fr = np.asarray(clusters.get("firing_rate", np.ones(len(clusters["cluster_id"]))))
    pr = np.asarray(clusters.get("presence_ratio", np.ones_like(fr)))
    rp = np.asarray(clusters.get("slidingRP_viol", np.zeros_like(fr)))
    return (fr >= min_fr) & (pr >= min_presence) & (rp <= max_rp_violation)


# ----------------------------------------------------------------------------- IBL via ONE
def get_one(cache_dir: Optional[str] = None):
    from one.api import ONE
    return ONE(base_url=ONE_BASE_URL, password=ONE_PASSWORD, silent=True, cache_dir=cache_dir)


def load_ibl_session(pid: str, one=None, cache_dir: Optional[str] = None) -> SessionData:
    """Load one BWM insertion: spikes, clusters (with Beryl regions + QC), and trials."""
    from brainbox.io.one import SpikeSortingLoader
    from iblatlas.atlas import AllenAtlas

    one = one or get_one(cache_dir)
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    eid = sl.eid
    sess = one.alyx.rest("sessions", "read", id=eid)
    trials = one.load_object(eid, "trials")
    tdf = pd.DataFrame({k: np.asarray(trials[k]) for k in TRIAL_COLUMNS if k in trials})
    cluster_ids = np.asarray(clusters["cluster_id"])
    regions = assign_beryl(np.asarray(clusters["acronym"], dtype=object))
    order = np.argsort(spikes["times"], kind="stable")
    return SessionData(eid=eid, pid=pid, lab=sess["lab"], subject=sess["subject"],
                       spike_times=np.asarray(spikes["times"])[order],
                       spike_clusters=np.asarray(spikes["clusters"])[order],
                       cluster_ids=cluster_ids, cluster_regions=regions, cluster_good=qc_good(clusters),
                       trials=tdf, dataset="ibl", meta={"probe_name": sl.pname, "n_channels": len(channels["x"])})


# ----------------------------------------------------------------------------- Allen via allensdk
def load_allen_session(session, stimulus_name: str = "flashes", lab: str = "allen") -> SessionData:
    """Wrap an `allensdk` EcephysSession into SessionData with a stimulus-onset 'trials' table.

    Stimulus epochs from `session.get_stimulus_table(stimulus_name)` become trials with
    `stimOn_times` = start_time and a `contrast`-like column when available; there is no choice.
    """
    units = session.units
    spike_times_dict = session.spike_times
    st, sc = [], []
    for uid in units.index:
        t = np.asarray(spike_times_dict[uid])
        st.append(t)
        sc.append(np.full(len(t), uid))
    st = np.concatenate(st)
    sc = np.concatenate(sc)
    order = np.argsort(st, kind="stable")
    stim = session.get_stimulus_table(stimulus_name)
    trials = pd.DataFrame({"stimOn_times": stim["start_time"].to_numpy()})
    for col in ("contrast", "orientation", "spatial_frequency", "color"):
        if col in stim.columns:
            trials[col] = pd.to_numeric(stim[col], errors="coerce").to_numpy()
    good = np.ones(len(units), dtype=bool)
    for col, thr, op in (("isi_violations", 0.5, "le"), ("amplitude_cutoff", 0.1, "le"), ("presence_ratio", 0.9, "ge")):
        if col in units.columns:
            v = units[col].to_numpy()
            good &= (v <= thr) if op == "le" else (v >= thr)
    return SessionData(eid=str(session.ecephys_session_id), pid=str(session.ecephys_session_id), lab=lab,
                       subject=str(session.metadata.get("specimen_name", "")) if hasattr(session, "metadata") else "",
                       spike_times=st[order], spike_clusters=sc[order], cluster_ids=units.index.to_numpy(),
                       cluster_regions=assign_beryl(units["ecephys_structure_acronym"].fillna("root").to_numpy()),
                       cluster_good=good, trials=trials, dataset="allen")


# ----------------------------------------------------------------------------- synthetic
def make_synthetic_session(n_units: int = 40, n_trials: int = 300, regions: Sequence[str] = ("VISp", "CA1"),
                           encode: Optional[Mapping[str, float]] = None, base_rate: float = 5.0,
                           lab: str = "labA", subject: str = "m1", seed: int = 0, eid: str = "syn") -> SessionData:
    """Simulate a Poisson session with IBL-like trials where a subset of units encodes task variables.

    `encode` maps variable -> rate modulation (Hz) applied in the variable's window for the first half
    of units of each region; variables: 'choice', 'stimulus', 'block', 'reward'.
    """
    rng = np.random.default_rng(seed)
    encode = dict(encode or {"choice": 4.0, "stimulus": 4.0})
    iti = 2.0
    stim_on = np.cumsum(rng.uniform(1.5, 2.5, n_trials)) + 5.0
    p_left = generate_block_prior(n_trials, rng)
    side_left = rng.random(n_trials) < p_left
    contrast = rng.choice([0.0, 0.0625, 0.125, 0.25, 1.0], n_trials)
    correct = rng.random(n_trials) < 0.8
    # choice: -1 = right, 1 = left (IBL sign convention is arbitrary here; stored as ±1)
    choice = np.where(side_left == correct, 1, -1)
    first_move = stim_on + rng.uniform(0.15, 0.5, n_trials)
    feedback = first_move + rng.uniform(0.05, 0.3, n_trials)
    trials = pd.DataFrame({
        "choice": choice,
        "contrastLeft": np.where(side_left, contrast, np.nan),
        "contrastRight": np.where(~side_left, contrast, np.nan),
        "probabilityLeft": p_left,
        "feedbackType": np.where(correct, 1, -1),
        "stimOn_times": stim_on, "firstMovement_times": first_move, "feedback_times": feedback,
        "response_times": feedback,
    })
    T = feedback[-1] + iti
    cluster_ids = np.arange(n_units)
    cluster_regions = np.asarray([regions[i % len(regions)] for i in range(n_units)], dtype=object)
    st, sc = [], []
    for u in cluster_ids:
        encoding = (u // len(regions)) % 2 == 0  # half of the units per region encode
        rate = np.full(int(np.ceil(T * 100)), base_rate)  # 10 ms resolution
        if encoding:
            for k in range(n_trials):
                if "stimulus" in encode:
                    s, e = stim_on[k], stim_on[k] + 0.2
                    rate[int(s * 100):int(e * 100)] += encode["stimulus"] * (1 if side_left[k] else -1) * (contrast[k] > 0)
                if "choice" in encode:
                    s, e = first_move[k] - 0.1, first_move[k] + 0.1
                    rate[int(s * 100):int(e * 100)] += encode["choice"] * choice[k]
                if "block" in encode:
                    s, e = stim_on[k] - 0.4, stim_on[k]
                    rate[int(s * 100):int(e * 100)] += encode["block"] * (p_left[k] - 0.5) * 2
                if "reward" in encode:
                    s, e = feedback[k], feedback[k] + 0.2
                    rate[int(s * 100):int(e * 100)] += encode["reward"] * (1 if correct[k] else -1)
        rate = np.clip(rate, 0.2, None)
        counts = rng.poisson(rate * 0.01)
        idx = np.repeat(np.arange(len(rate)), counts)
        t = idx * 0.01 + rng.random(len(idx)) * 0.01
        st.append(t)
        sc.append(np.full(len(t), u))
    st = np.concatenate(st)
    sc = np.concatenate(sc)
    order = np.argsort(st, kind="stable")
    return SessionData(eid=eid, pid=eid + "_p0", lab=lab, subject=subject, spike_times=st[order],
                       spike_clusters=sc[order], cluster_ids=cluster_ids, cluster_regions=cluster_regions,
                       cluster_good=np.ones(n_units, bool), trials=trials, dataset="synthetic")


def generate_block_prior(n_trials: int, rng: np.random.Generator, first_block: int = 90, mean_len: float = 60.0,
                         min_len: int = 20, max_len: int = 100) -> np.ndarray:
    """IBL block structure: 90 unbiased trials (p=0.5), then alternating 0.2/0.8 blocks whose lengths are
    exponential(mean 60) truncated to [20, 100]. Used both for simulation and pseudo-session nulls."""
    p = np.empty(n_trials)
    p[:first_block] = 0.5
    i = first_block
    cur = rng.choice([0.2, 0.8])
    while i < n_trials:
        L = int(np.clip(rng.exponential(mean_len), min_len, max_len))
        p[i:i + L] = cur
        i += L
        cur = 0.8 if cur == 0.2 else 0.2  # exact literals (1.0 - 0.8 != 0.2 in floating point)
    return p
