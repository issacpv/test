"""Typed benchmark specification: tasks, domains, shift cells, and the default registry.

Design (WILDS-style, extended):

* A **task** fixes the prediction target, its type and the primary metric.
* A **domain** is a dataset (or a sub-population of one) with provenance
  metadata: country, site, population descriptor, acquisition device.
* A **shift cell** is an ordered (source -> target) pair for one task, tagged
  with the *axes* along which the two domains differ.  The axes are the
  first-class experimental variable of the benchmark: every cross-modality
  claim ("population shift hurts more than device shift") is a comparison of
  cells grouped by axis.
* Cells also carry **contamination flags**: model families whose pre-training
  corpus overlaps the target (e.g. EEG foundation models trained on TUEG for the
  TUSZ target), so leaderboards can grey those entries out.

Everything is a frozen dataclass and serialises to JSON so that fixed cell
lists can be released with the manifests.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import permutations
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple


class ShiftAxis(str, Enum):
    SITE = "site"  # different institution / database, same country or not
    POPULATION = "population"  # age structure, case mix, country
    DEVICE = "device"  # acquisition hardware, sampling rate, montage, charting resolution
    LABEL_PROTOCOL = "label_protocol"  # annotation vocabulary / operational definition differs
    TEMPORAL = "temporal"  # different calendar period


class TaskType(str, Enum):
    BINARY = "binary"
    MULTILABEL = "multilabel"
    REGRESSION = "regression"
    EVENT_DETECTION = "event_detection"


@dataclass(frozen=True)
class TaskSpec:
    id: str
    modality: str
    task_type: TaskType
    label_names: Tuple[str, ...]
    primary_metric: str
    description: str = ""
    group_column: str = "group"
    subgroup_columns: Tuple[str, ...] = ("sex", "age_band")
    horizon: str = ""  # free text, e.g. "48 h after 24 h observation"


@dataclass(frozen=True)
class DomainSpec:
    id: str
    dataset: str
    modality: str
    country: str
    site: str
    population: str
    device: str
    access: str  # 'open' | 'registration' | 'dua' | 'credentialed'
    url: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ShiftCell:
    id: str
    task: str
    source: Tuple[str, ...]
    target: Tuple[str, ...]
    axes: FrozenSet[ShiftAxis]
    contaminated_models: Tuple[str, ...] = ()
    note: str = ""

    def has(self, axis: ShiftAxis) -> bool:
        return axis in self.axes


@dataclass
class Benchmark:
    tasks: Dict[str, TaskSpec] = field(default_factory=dict)
    domains: Dict[str, DomainSpec] = field(default_factory=dict)
    cells: List[ShiftCell] = field(default_factory=list)

    # ................................................................ #
    def validate(self) -> None:
        ids = [c.id for c in self.cells]
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate cell ids: {dup}")
        for c in self.cells:
            if c.task not in self.tasks:
                raise ValueError(f"cell {c.id}: unknown task {c.task}")
            for d in (*c.source, *c.target):
                if d not in self.domains:
                    raise ValueError(f"cell {c.id}: unknown domain {d}")
                if self.domains[d].modality != self.tasks[c.task].modality:
                    raise ValueError(f"cell {c.id}: domain {d} modality != task modality")
            if set(c.source) & set(c.target):
                raise ValueError(f"cell {c.id}: source and target overlap")
            if not c.axes:
                raise ValueError(f"cell {c.id}: no shift axis")

    def cells_for(self, task: Optional[str] = None, modality: Optional[str] = None, axis: Optional[ShiftAxis] = None) -> List[ShiftCell]:
        out = []
        for c in self.cells:
            if task and c.task != task:
                continue
            if modality and self.tasks[c.task].modality != modality:
                continue
            if axis and axis not in c.axes:
                continue
            out.append(c)
        return out

    def to_json(self) -> str:
        payload = {
            "tasks": [dict(asdict(t), task_type=t.task_type.value) for t in self.tasks.values()],
            "domains": [asdict(d) for d in self.domains.values()],
            "cells": [dict(asdict(c), axes=sorted(a.value for a in c.axes)) for c in self.cells],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Benchmark":
        raw = json.loads(text)
        b = cls()
        for t in raw["tasks"]:
            t = dict(t)
            t["task_type"] = TaskType(t["task_type"])
            t["label_names"] = tuple(t["label_names"])
            t["subgroup_columns"] = tuple(t["subgroup_columns"])
            b.tasks[t["id"]] = TaskSpec(**t)
        for d in raw["domains"]:
            b.domains[d["id"]] = DomainSpec(**d)
        for c in raw["cells"]:
            c = dict(c)
            c["axes"] = frozenset(ShiftAxis(a) for a in c["axes"])
            c["source"] = tuple(c["source"])
            c["target"] = tuple(c["target"])
            c["contaminated_models"] = tuple(c.get("contaminated_models", ()))
            b.cells.append(ShiftCell(**c))
        b.validate()
        return b


# --------------------------------------------------------------------------- #
# default registry composing the four sibling projects
# --------------------------------------------------------------------------- #
def _axes(*a: ShiftAxis) -> FrozenSet[ShiftAxis]:
    return frozenset(a)


def _icu(b: Benchmark) -> None:
    for tid, desc, horizon in (
        ("icu_mortality_48h", "Death within 48 h of prediction time (24 h after ICU admission).", "24 h obs -> 48 h"),
        ("icu_aki_48h", "KDIGO stage >= 1 AKI within 48 h (creatinine + urine output).", "24 h obs -> 48 h"),
        ("icu_sepsis_12h", "Sepsis-3 onset within 12 h (suspected infection + SOFA >= 2).", "24 h obs -> 12 h"),
    ):
        b.tasks[tid] = TaskSpec(tid, "icu", TaskType.BINARY, ("event",), "auroc", desc, group_column="subject_id", subgroup_columns=("sex", "age_band"), horizon=horizon)
    b.domains["mimic_iv"] = DomainSpec("mimic_iv", "MIMIC-IV v3.1", "icu", "US", "BIDMC Boston", "adult ICU, single tertiary centre", "hourly charting", "credentialed", "https://physionet.org/content/mimiciv/3.1/")
    b.domains["eicu"] = DomainSpec("eicu", "eICU-CRD v2.0", "icu", "US", "208 hospitals", "adult ICU, multi-centre community + academic", "5-min charting", "credentialed", "https://physionet.org/content/eicu-crd/2.0/")
    b.domains["hirid"] = DomainSpec("hirid", "HiRID v1.1.1", "icu", "CH", "Bern Inselspital", "adult ICU, single tertiary centre", "2-min resolution", "credentialed", "https://physionet.org/content/hirid/1.1.1/")
    b.domains["aumcdb"] = DomainSpec("aumcdb", "AmsterdamUMCdb v1.0.2", "icu", "NL", "Amsterdam UMC", "adult ICU, banded age", "high-resolution monitoring", "dua", "https://amsterdammedicaldatascience.nl/amsterdamumcdb/")
    country = {d: b.domains[d].country for d in ("mimic_iv", "eicu", "hirid", "aumcdb")}
    for tid in ("icu_mortality_48h", "icu_aki_48h", "icu_sepsis_12h"):
        for s, t in permutations(("mimic_iv", "eicu", "hirid", "aumcdb"), 2):
            axes = {ShiftAxis.SITE}
            if country[s] != country[t]:
                axes.add(ShiftAxis.POPULATION)
            if "hirid" in (s, t) or "eicu" in (s, t):
                axes.add(ShiftAxis.DEVICE)  # charting resolution / monitoring infrastructure
            if tid == "icu_sepsis_12h" and "eicu" in (s, t):
                axes.add(ShiftAxis.LABEL_PROTOCOL)  # sparse culture timing -> susp_inf fallback
            b.cells.append(ShiftCell(f"{tid}:{s}->{t}", tid, (s,), (t,), frozenset(axes)))


def _ecg(b: Benchmark) -> None:
    classes = ("NORM", "AF", "AFL", "1dAVB", "RBBB", "LBBB", "PVC", "PAC", "SB", "ST", "LQT", "LVH")
    b.tasks["ecg_dx12"] = TaskSpec("ecg_dx12", "ecg", TaskType.MULTILABEL, classes, "macro_auroc", "12 harmonised diagnostic classes (lenient SNOMED/SCP mapping).", group_column="patient_id", subgroup_columns=("sex", "age_band"))
    b.domains["ptbxl"] = DomainSpec("ptbxl", "PTB-XL v1.0.3", "ecg", "DE", "PTB / Schiller", "median age ~61 y, clinical", "500 Hz diagnostic cart", "open", "https://physionet.org/content/ptb-xl/1.0.3/")
    b.domains["chapman"] = DomainSpec("chapman", "Chapman-Shaoxing", "ecg", "CN", "Shaoxing People's Hospital", "median age ~51 y", "500 Hz, same vendor family as Ningbo", "open", "https://physionet.org/content/ecg-arrhythmia/1.0.0/")
    b.domains["ningbo"] = DomainSpec("ningbo", "Ningbo First Hospital", "ecg", "CN", "Ningbo", "median age ~51 y", "500 Hz, same vendor family as Chapman", "open", "https://physionet.org/content/ecg-arrhythmia/1.0.0/")
    b.domains["georgia"] = DomainSpec("georgia", "Georgia 12-lead (Challenge 2021)", "ecg", "US", "Emory", "US clinical", "500 Hz", "open", "https://physionet.org/content/challenge-2021/1.0.3/")
    b.domains["code15"] = DomainSpec("code15", "CODE-15%", "ecg", "BR", "Telehealth Network of Minas Gerais", "primary-care telehealth, younger", "400 Hz, zero-padded 4096 samples", "open", "https://zenodo.org/records/4916206")
    b.domains["cpsc"] = DomainSpec("cpsc", "CPSC-2018 + Extra", "ecg", "CN", "CPSC", "Chinese clinical", "500 Hz, 6-60 s", "open", "https://physionet.org/content/challenge-2021/1.0.3/")
    sources = ("ptbxl", "chapman", "ningbo", "georgia", "code15")
    targets = sources + ("cpsc",)
    country = {d: b.domains[d].country for d in targets}
    for s in sources:
        for t in targets:
            if s == t:
                continue
            axes = {ShiftAxis.SITE}
            same_vendor = {s, t} == {"chapman", "ningbo"}
            if not same_vendor:
                axes.add(ShiftAxis.DEVICE)
                axes.add(ShiftAxis.LABEL_PROTOCOL)
            if country[s] != country[t]:
                axes.add(ShiftAxis.POPULATION)
            if "code15" in (s, t):
                axes.add(ShiftAxis.POPULATION)
            contaminated = ("ecg-fm", "hubert-ecg") if t in ("ptbxl", "chapman", "ningbo", "georgia", "cpsc") else ()
            b.cells.append(ShiftCell(f"ecg_dx12:{s}->{t}", "ecg_dx12", (s,), (t,), frozenset(axes), contaminated, "FM pre-training overlap with public sets must be verified per checkpoint."))


def _eeg(b: Benchmark) -> None:
    b.tasks["eeg_seizure_event"] = TaskSpec("eeg_seizure_event", "eeg", TaskType.EVENT_DETECTION, ("seizure",), "event_f1", "Seizure event detection from 18-pair bipolar scalp EEG at 256 Hz; 4-s windows, event scoring with 30 s / 60 s tolerances.", group_column="subject_id", subgroup_columns=("sex", "age_band"))
    b.domains["chbmit"] = DomainSpec("chbmit", "CHB-MIT", "eeg", "US", "Boston Children's", "pediatric 1.5-22 y", "bipolar 256 Hz", "open", "https://physionet.org/content/chbmit/1.0.0/")
    b.domains["siena"] = DomainSpec("siena", "Siena Scalp EEG", "eeg", "IT", "Siena", "adult 20-71 y", "referential 512 Hz", "open", "https://physionet.org/content/siena-scalp-eeg/1.0.0/")
    b.domains["tusz"] = DomainSpec("tusz", "TUH EEG Seizure Corpus v2.0.3", "eeg", "US", "Temple", "mixed ages", "referential AR/LE 250-512 Hz", "registration", "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/")
    b.domains["helsinki"] = DomainSpec("helsinki", "Helsinki neonatal EEG", "eeg", "FI", "Helsinki", "neonatal, 3 annotators", "19-ch referential 256 Hz", "open", "https://zenodo.org/ (Stevenson et al., 2019)")
    b.cells.append(ShiftCell("eeg_seizure_event:chbmit->siena", "eeg_seizure_event", ("chbmit",), ("siena",), _axes(ShiftAxis.SITE, ShiftAxis.POPULATION, ShiftAxis.DEVICE)))
    b.cells.append(ShiftCell("eeg_seizure_event:chbmit->tusz", "eeg_seizure_event", ("chbmit",), ("tusz",), _axes(ShiftAxis.SITE, ShiftAxis.POPULATION, ShiftAxis.DEVICE, ShiftAxis.LABEL_PROTOCOL), ("biot", "labram", "eegpt", "cbramod"), "FMs pre-trained on TUEG; TUSZ is not a clean zero-shot target for them."))
    b.cells.append(ShiftCell("eeg_seizure_event:chbmit->helsinki", "eeg_seizure_event", ("chbmit",), ("helsinki",), _axes(ShiftAxis.SITE, ShiftAxis.POPULATION, ShiftAxis.DEVICE, ShiftAxis.LABEL_PROTOCOL)))
    b.cells.append(ShiftCell("eeg_seizure_event:tusz->siena", "eeg_seizure_event", ("tusz",), ("siena",), _axes(ShiftAxis.SITE, ShiftAxis.DEVICE)))
    b.cells.append(ShiftCell("eeg_seizure_event:tusz->chbmit", "eeg_seizure_event", ("tusz",), ("chbmit",), _axes(ShiftAxis.SITE, ShiftAxis.POPULATION, ShiftAxis.DEVICE)))


def _echo(b: Benchmark) -> None:
    b.tasks["echo_ef"] = TaskSpec("echo_ef", "echo", TaskType.REGRESSION, ("lvef",), "mae", "Left-ventricular ejection fraction (%) from apical-4-chamber video; secondary binary EF < 40%.", group_column="patient_id", subgroup_columns=("sex", "age_band"))
    b.domains["echonet_dynamic"] = DomainSpec("echonet_dynamic", "EchoNet-Dynamic", "echo", "US", "Stanford", "adult, 10,030 videos", "A4C, 112x112 de-identified", "registration", "https://echonet.github.io/dynamic/")
    b.domains["echonet_pediatric"] = DomainSpec("echonet_pediatric", "EchoNet-Pediatric", "echo", "US", "Stanford Children's", "pediatric", "A4C + PSAX", "registration", "https://echonet.github.io/pediatric/")
    b.domains["camus"] = DomainSpec("camus", "CAMUS", "echo", "FR", "CHU Saint-Etienne", "adult, 500 patients, half with EF < 45%", "GE Vivid E95, A4C + A2C, expert contours", "registration", "https://www.creatis.insa-lyon.fr/Challenge/camus/")
    b.cells.append(ShiftCell("echo_ef:echonet_dynamic->echonet_pediatric", "echo_ef", ("echonet_dynamic",), ("echonet_pediatric",), _axes(ShiftAxis.POPULATION)))
    b.cells.append(ShiftCell("echo_ef:echonet_dynamic->camus", "echo_ef", ("echonet_dynamic",), ("camus",), _axes(ShiftAxis.SITE, ShiftAxis.DEVICE, ShiftAxis.LABEL_PROTOCOL, ShiftAxis.POPULATION), (), "CAMUS EF from expert biplane contours; EchoNet EF from clinical report."))
    b.cells.append(ShiftCell("echo_ef:echonet_pediatric->echonet_dynamic", "echo_ef", ("echonet_pediatric",), ("echonet_dynamic",), _axes(ShiftAxis.POPULATION)))


def default_benchmark() -> Benchmark:
    """The concrete BioShift v0 registry: 4 modalities, 6 tasks, 17 domains."""
    b = Benchmark()
    _icu(b)
    _ecg(b)
    _eeg(b)
    _echo(b)
    b.validate()
    return b


def cell_table(b: Benchmark):
    """pandas DataFrame with one row per cell and one boolean column per axis."""
    import pandas as pd

    rows = []
    for c in b.cells:
        row = {"cell": c.id, "task": c.task, "modality": b.tasks[c.task].modality, "source": "+".join(c.source), "target": "+".join(c.target), "n_axes": len(c.axes), "contaminated": ";".join(c.contaminated_models)}
        for a in ShiftAxis:
            row[a.value] = a in c.axes
        rows.append(row)
    return pd.DataFrame(rows)
