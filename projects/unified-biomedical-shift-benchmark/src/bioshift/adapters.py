"""Data adapters: the single interface every modality must satisfy.

``Adapter.load(task_id, domain_id, split)`` returns a :class:`DomainData` with

* ``X`` - features or an embedding (n, d).  For raw waveforms/videos the
  sibling projects export a fixed feature or embedding cache; the harness never
  touches raw signals, which keeps the evaluation identical across modalities.
* ``y`` - labels: (n,) for binary/regression/event windows, (n, k) for multilabel.
* ``groups`` - the leakage unit (subject / patient / stay); every split and
  bootstrap resamples over these.
* ``meta`` - subgroup columns (``sex``, ``age_band``) and, for event detection,
  ``record_id`` and ``t_start`` (s) of each window.

Two adapters are provided:

* :class:`SyntheticAdapter` generates multi-domain data with controllable
  covariate, label and concept shift per domain (deterministic from the domain
  id) so the whole harness can be exercised without any data.
* :class:`ManifestAdapter` reads ``data/cache/<task>/<domain>.npz`` plus a
  ``manifests/<task>/<domain>_splits.csv`` (group -> train/val/test) produced by
  the sibling projects' export scripts (see data/README.md).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol

import numpy as np
import pandas as pd

from .spec import Benchmark, TaskType, default_benchmark


@dataclass
class DomainData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame
    domain: str
    task: str
    split: str = "all"

    def __len__(self) -> int:
        return int(len(self.y))

    def subset(self, mask: np.ndarray, split: Optional[str] = None) -> "DomainData":
        return DomainData(self.X[mask], self.y[mask], self.groups[mask], self.meta.iloc[np.flatnonzero(mask)].reset_index(drop=True), self.domain, self.task, split or self.split)


class Adapter(Protocol):
    def load(self, task_id: str, domain_id: str, split: str = "all") -> DomainData: ...


# --------------------------------------------------------------------------- #
def _seed_from(*parts: str) -> int:
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def group_split(groups: np.ndarray, test_frac: float = 0.3, seed: int = 0) -> np.ndarray:
    """Boolean mask selecting a group-disjoint test portion of about ``test_frac``."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_groups = set(uniq[:n_test].tolist())
    return np.array([g in test_groups for g in groups])


@dataclass
class ShiftParams:
    """Per-domain generative parameters for the synthetic adapter."""

    mean_shift: np.ndarray  # covariate shift: added to feature means
    prior: float  # label prior P(y=1) (binary) or fraction of positive windows
    concept_flip: float  # concept shift: rotation of the true coefficient vector (0..1)
    noise: float


class SyntheticAdapter:
    """Deterministic synthetic multi-domain generator for all task types.

    Parameters
    ----------
    benchmark:
        Registry used to look up the task type.
    n_per_domain, n_groups, d:
        Samples, groups (subjects) and feature dimension per domain.
    shift_scale:
        Global multiplier for the amount of shift between domains (0 -> i.i.d.).
    """

    def __init__(self, benchmark: Optional[Benchmark] = None, n_per_domain: int = 600, n_groups: int = 60, d: int = 12, shift_scale: float = 1.0, seed: int = 0) -> None:
        self.b = benchmark or default_benchmark()
        self.n = n_per_domain
        self.n_groups = n_groups
        self.d = d
        self.shift_scale = shift_scale
        self.seed = seed
        self._cache: Dict[tuple, DomainData] = {}

    def params(self, domain_id: str) -> ShiftParams:
        rng = np.random.default_rng(_seed_from("params", domain_id, str(self.seed)))
        return ShiftParams(
            mean_shift=self.shift_scale * rng.normal(0, 0.6, self.d),
            prior=float(np.clip(0.3 + self.shift_scale * rng.normal(0, 0.12), 0.05, 0.7)),
            concept_flip=float(np.clip(self.shift_scale * abs(rng.normal(0, 0.25)), 0, 0.9)),
            noise=1.0,
        )

    def _beta(self, task_id: str, concept_flip: float) -> np.ndarray:
        rng = np.random.default_rng(_seed_from("beta", task_id, str(self.seed)))
        beta = rng.normal(0, 1, self.d)
        alt = rng.normal(0, 1, self.d)
        beta = (1 - concept_flip) * beta + concept_flip * alt
        return beta / np.linalg.norm(beta) * 2.0

    def load(self, task_id: str, domain_id: str, split: str = "all") -> DomainData:
        key = (task_id, domain_id)
        if key not in self._cache:
            self._cache[key] = self._generate(task_id, domain_id)
        data = self._cache[key]
        if split == "all":
            return data
        mask = group_split(data.groups, 0.3, seed=_seed_from("split", task_id, domain_id))
        return data.subset(mask if split == "test" else ~mask, split)

    def _generate(self, task_id: str, domain_id: str) -> DomainData:
        task = self.b.tasks[task_id]
        p = self.params(domain_id)
        rng = np.random.default_rng(_seed_from("data", task_id, domain_id, str(self.seed)))
        groups = np.repeat(np.arange(self.n_groups), int(np.ceil(self.n / self.n_groups)))[: self.n]
        group_eff = rng.normal(0, 0.5, (self.n_groups, self.d))[groups]
        X = rng.normal(0, 1, (self.n, self.d)) + p.mean_shift + group_eff
        beta = self._beta(task_id, p.concept_flip)
        logit = X @ beta
        sex = rng.integers(0, 2, self.n)
        age_band = rng.choice(["<40", "40-64", ">=65"], self.n, p=[0.3, 0.4, 0.3])
        meta = pd.DataFrame({"sex": sex, "age_band": age_band})
        if task.task_type in (TaskType.BINARY, TaskType.EVENT_DETECTION):
            # calibrate intercept so that mean P(y=1) ~ prior
            b0 = np.log(p.prior / (1 - p.prior)) - float(np.mean(logit))
            prob = 1 / (1 + np.exp(-(logit + b0)))
            y = (rng.uniform(size=self.n) < prob).astype(int)
            if task.task_type == TaskType.EVENT_DETECTION:
                # windows ordered in time within each group (record); make positives contiguous-ish
                order = np.argsort(groups, kind="stable")
                X, y, groups, meta = X[order], y[order], groups[order], meta.iloc[order].reset_index(drop=True)
                y = _make_contiguous(y, groups, rng)
                meta["record_id"] = groups
                meta["t_start"] = _window_times(groups, step_s=2.0)
        elif task.task_type == TaskType.MULTILABEL:
            k = len(task.label_names)
            rng2 = np.random.default_rng(_seed_from("ml", task_id, str(self.seed)))
            B = rng2.normal(0, 1, (self.d, k)) / np.sqrt(self.d) * 2.0
            alt = rng2.normal(0, 1, (self.d, k)) / np.sqrt(self.d) * 2.0
            B = (1 - p.concept_flip) * B + p.concept_flip * alt
            L = X @ B + np.log(p.prior / (1 - p.prior))
            y = (rng.uniform(size=L.shape) < 1 / (1 + np.exp(-L))).astype(int)
        else:  # regression (EF-like, 20..80)
            y = 55.0 + 8.0 * logit / np.std(logit) + rng.normal(0, 4.0 * p.noise, self.n) - 10.0 * (p.prior - 0.3)
            y = np.clip(y, 10, 85)
        return DomainData(X.astype(np.float32), y, groups, meta, domain_id, task_id)


def _make_contiguous(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn i.i.d. positive windows into runs of 2-6 windows (events) within each record."""
    out = np.zeros_like(y)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        n_pos = int(y[idx].sum())
        n_events = max(0, n_pos // 4)
        for _ in range(n_events):
            start = int(rng.integers(0, max(1, len(idx) - 6)))
            length = int(rng.integers(2, 7))
            out[idx[start : start + length]] = 1
    return out


def _window_times(groups: np.ndarray, step_s: float) -> np.ndarray:
    t = np.zeros(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        t[idx] = np.arange(len(idx)) * step_s
    return t


# --------------------------------------------------------------------------- #
class ManifestAdapter:
    """Reads exported caches and fixed group-level split manifests.

    Layout (see data/README.md)::

        data/cache/<task>/<domain>.npz          keys: X, y, groups  (+ optional 'record_id', 't_start')
        data/cache/<task>/<domain>_meta.csv     columns: sex, age_band, ... (row-aligned with X)
        manifests/<task>/<domain>_splits.csv    columns: group, split  (train / val / test)
    """

    def __init__(self, root: str | Path = ".", benchmark: Optional[Benchmark] = None) -> None:
        self.root = Path(root)
        self.b = benchmark or default_benchmark()

    def _paths(self, task_id: str, domain_id: str) -> Dict[str, Path]:
        return {
            "npz": self.root / "data" / "cache" / task_id / f"{domain_id}.npz",
            "meta": self.root / "data" / "cache" / task_id / f"{domain_id}_meta.csv",
            "splits": self.root / "manifests" / task_id / f"{domain_id}_splits.csv",
        }

    def load(self, task_id: str, domain_id: str, split: str = "all") -> DomainData:
        if task_id not in self.b.tasks or domain_id not in self.b.domains:
            raise KeyError(f"unknown task/domain {task_id}/{domain_id}")
        p = self._paths(task_id, domain_id)
        if not p["npz"].exists():
            raise FileNotFoundError(
                f"{p['npz']} missing. Export it from the sibling project for modality "
                f"'{self.b.tasks[task_id].modality}' (see data/README.md, section 'Cache export')."
            )
        z = np.load(p["npz"], allow_pickle=False)
        X, y, groups = z["X"], z["y"], z["groups"]
        meta = pd.read_csv(p["meta"]) if p["meta"].exists() else pd.DataFrame(index=range(len(y)))
        for k in ("record_id", "t_start"):
            if k in z.files:
                meta[k] = z[k]
        data = DomainData(X, y, groups, meta, domain_id, task_id)
        if split == "all":
            return data
        if not p["splits"].exists():
            raise FileNotFoundError(f"{p['splits']} missing; run scripts/download_data.py --make-manifests")
        sp = pd.read_csv(p["splits"])
        wanted = set(sp.loc[sp["split"] == split, "group"].astype(str))
        mask = np.array([str(g) in wanted for g in groups])
        return data.subset(mask, split)


def write_split_manifest(groups: np.ndarray, path: str | Path, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Create a fixed group-level train/val/test manifest (record ids only; safe to commit)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups).astype(str)
    rng.shuffle(uniq)
    n_test = int(round(test_frac * len(uniq)))
    n_val = int(round(val_frac * len(uniq)))
    split = np.array(["train"] * len(uniq), dtype=object)
    split[:n_test] = "test"
    split[n_test : n_test + n_val] = "val"
    df = pd.DataFrame({"group": uniq, "split": split})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
