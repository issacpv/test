"""Typed cohort-definition DSL, extraction prompt, parser and multiverse expansion.

The DSL is deliberately small and closed: every criterion has a ``kind`` from ``CRITERION_KINDS`` with
typed parameters, a quoted ``source`` sentence, an ``ambiguous`` flag and optional ``alternatives``
(parameter dicts that would also be consistent with the paper). The compiler (``compiler.py``)
implements every kind for MIMIC-IV/III; anything a paper says that the DSL cannot express goes into
``unsupported`` so that coverage is measured instead of silently dropped.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "CRITERION_KINDS",
    "OUTCOME_KINDS",
    "UNITS",
    "DATABASES",
    "Criterion",
    "Outcome",
    "CohortDefinition",
    "validate_definition_dict",
    "parse_extraction",
    "EXTRACTION_SYSTEM_PROMPT",
    "build_extraction_prompt",
    "enumerate_multiverse",
]

DATABASES = ("mimic-iii", "mimic-iv")
UNITS = ("icustay", "hadm", "subject")

# kind -> (required params, optional params)
CRITERION_KINDS: Dict[str, Dict[str, List[str]]] = {
    "age_min": {"required": ["years"], "optional": ["at"]},  # at: icu_intime | hosp_admittime
    "age_max": {"required": ["years"], "optional": ["at"]},
    "first_icu_stay_only": {"required": [], "optional": ["per"]},  # per: subject | hadm
    "first_admission_only": {"required": [], "optional": []},
    "min_icu_los_hours": {"required": ["hours"], "optional": []},
    "max_icu_los_hours": {"required": ["hours"], "optional": []},
    "min_hospital_los_hours": {"required": ["hours"], "optional": []},
    "care_unit_in": {"required": ["units"], "optional": []},  # list of first_careunit strings
    "care_unit_not_in": {"required": ["units"], "optional": []},
    "admission_type_in": {"required": ["types"], "optional": []},
    "require_icd_prefix": {"required": ["prefixes"], "optional": ["icd_version"]},  # any code starts with a prefix
    "exclude_icd_prefix": {"required": ["prefixes"], "optional": ["icd_version"]},
    "require_lab_measured": {"required": ["itemids"], "optional": ["window_hours"]},
    "exclude_death_within_hours": {"required": ["hours"], "optional": ["from"]},  # from: icu_intime | hosp_admittime
    "exclude_missing_outcome": {"required": [], "optional": []},
}

OUTCOME_KINDS = ("in_hospital_mortality", "icu_mortality", "mortality_30d", "mortality_90d", "mortality_1y", "icu_los_gt_hours", "readmission_30d")


@dataclass
class Criterion:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    ambiguous: bool = False
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def canonical(self) -> str:
        return json.dumps({"kind": self.kind, "params": _normalise_params(self.params)}, sort_keys=True)


@dataclass
class Outcome:
    kind: str
    hours: Optional[float] = None  # for icu_los_gt_hours
    source: str = ""


@dataclass
class CohortDefinition:
    paper_id: str
    database: str
    unit: str
    criteria: List[Criterion]
    outcome: Optional[Outcome] = None
    prediction_time: str = "icu_intime"
    mimic_version: str = ""
    reported_n: Optional[int] = None
    reported_prevalence: Optional[float] = None
    unsupported: List[str] = field(default_factory=list)
    notes: str = ""
    extraction_model: str = ""

    # ------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CohortDefinition":
        crits = [Criterion(**{k: v for k, v in c.items() if k in Criterion.__dataclass_fields__}) for c in d.get("criteria", [])]
        out = d.get("outcome")
        outcome = Outcome(**{k: v for k, v in out.items() if k in Outcome.__dataclass_fields__}) if out else None
        return cls(
            paper_id=str(d.get("paper_id", "")),
            database=str(d.get("database", "mimic-iv")),
            unit=str(d.get("unit", "icustay")),
            criteria=crits,
            outcome=outcome,
            prediction_time=str(d.get("prediction_time", "icu_intime")),
            mimic_version=str(d.get("mimic_version", "") or ""),
            reported_n=int(d["reported_n"]) if d.get("reported_n") not in (None, "") else None,
            reported_prevalence=float(d["reported_prevalence"]) if d.get("reported_prevalence") not in (None, "") else None,
            unsupported=list(d.get("unsupported", []) or []),
            notes=str(d.get("notes", "") or ""),
            extraction_model=str(d.get("extraction_model", "") or ""),
        )

    def criterion_set(self) -> set:
        return {c.canonical() for c in self.criteria}

    def canonical_hash(self) -> str:
        payload = {"database": self.database, "unit": self.unit, "criteria": sorted(self.criterion_set()), "outcome": asdict(self.outcome) if self.outcome else None}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _normalise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, list):
            out[k] = sorted(str(x).strip().upper() if isinstance(x, str) else x for x in v)
        elif isinstance(v, str):
            out[k] = v.strip().lower()
        else:
            out[k] = v
    return out


def validate_definition_dict(d: Dict[str, Any]) -> List[str]:
    """Problems in a raw (model-produced) definition dict; empty list means valid."""
    problems: List[str] = []
    if d.get("database") not in DATABASES:
        problems.append(f"database must be one of {DATABASES}")
    if d.get("unit") not in UNITS:
        problems.append(f"unit must be one of {UNITS}")
    crits = d.get("criteria")
    if not isinstance(crits, list):
        problems.append("criteria must be a list")
        crits = []
    for i, c in enumerate(crits):
        if not isinstance(c, dict) or "kind" not in c:
            problems.append(f"criteria[{i}] must be an object with a 'kind'")
            continue
        kind = c["kind"]
        if kind not in CRITERION_KINDS:
            problems.append(f"criteria[{i}]: unknown kind {kind!r}; allowed: {sorted(CRITERION_KINDS)}")
            continue
        params = c.get("params", {}) or {}
        for req in CRITERION_KINDS[kind]["required"]:
            if req not in params:
                problems.append(f"criteria[{i}] ({kind}): missing param {req!r}")
        allowed = set(CRITERION_KINDS[kind]["required"]) | set(CRITERION_KINDS[kind]["optional"])
        for k in params:
            if k not in allowed:
                problems.append(f"criteria[{i}] ({kind}): unexpected param {k!r}")
        for j, alt in enumerate(c.get("alternatives", []) or []):
            if not isinstance(alt, dict):
                problems.append(f"criteria[{i}].alternatives[{j}] must be a params object (optionally with its own 'kind')")
            elif "kind" in alt and alt["kind"] not in CRITERION_KINDS:
                problems.append(f"criteria[{i}].alternatives[{j}]: unknown kind {alt['kind']!r}")
    out = d.get("outcome")
    if out is not None:
        if not isinstance(out, dict) or out.get("kind") not in OUTCOME_KINDS:
            problems.append(f"outcome.kind must be one of {OUTCOME_KINDS}")
        elif out.get("kind") == "icu_los_gt_hours" and out.get("hours") is None:
            problems.append("outcome icu_los_gt_hours needs 'hours'")
    return problems


def parse_extraction(obj: Dict[str, Any], paper_id: str, model: str = "") -> CohortDefinition:
    """Validate and convert a model-produced dict into a ``CohortDefinition``."""
    problems = validate_definition_dict(obj)
    if problems:
        raise ValueError("; ".join(problems))
    obj = dict(obj)
    obj["paper_id"] = paper_id
    obj["extraction_model"] = model
    return CohortDefinition.from_dict(obj)


EXTRACTION_SYSTEM_PROMPT = """You are an expert in the MIMIC-III and MIMIC-IV critical-care databases.
You convert the cohort-selection description of a published study into a strict, structured JSON
cohort definition. Use ONLY the criterion kinds listed. Quote the sentence each criterion comes from.
When the paper is ambiguous (e.g. "first ICU stay" without saying per patient or per hospital
admission; a length-of-stay threshold without saying ICU or hospital), set "ambiguous": true and list
the plausible alternative parameter sets in "alternatives". Put anything you cannot express in the
"unsupported" list verbatim. Do not invent criteria that are not in the text. Reply with one JSON
object and nothing else."""


def build_extraction_prompt(paper_text: str, database_hint: Optional[str] = None) -> str:
    """User prompt: DSL vocabulary + JSON shape + the paper's methods text (public content only)."""
    kinds = "\n".join(f'- "{k}": required {v["required"]}, optional {v["optional"]}' for k, v in CRITERION_KINDS.items())
    shape = json.dumps(
        {
            "database": "mimic-iv | mimic-iii",
            "mimic_version": "e.g. 2.2 (empty if not stated)",
            "unit": "icustay | hadm | subject",
            "prediction_time": "icu_intime | icu_intime+24h | hosp_admittime | ...",
            "criteria": [{"kind": "age_min", "params": {"years": 18}, "source": "quoted sentence", "ambiguous": False, "alternatives": []}],
            "outcome": {"kind": "in_hospital_mortality", "hours": None, "source": "quoted sentence"},
            "reported_n": "integer cohort size stated in the paper or null",
            "reported_prevalence": "outcome prevalence stated in the paper (0-1) or null",
            "unsupported": ["verbatim criteria the DSL cannot express"],
            "notes": "",
        },
        indent=2,
    )
    hint = f"The study uses {database_hint}.\n" if database_hint else ""
    return (
        f"{hint}Allowed criterion kinds:\n{kinds}\n\nAllowed outcome kinds: {list(OUTCOME_KINDS)}\n\n"
        f"JSON shape:\n{shape}\n\nPaper text (methods / cohort section):\n\"\"\"\n{paper_text}\n\"\"\"\n"
    )


def enumerate_multiverse(defn: CohortDefinition, max_variants: int = 64) -> List[CohortDefinition]:
    """All definitions obtained by choosing, for each ambiguous criterion, the original or one alternative.

    An alternative is a parameter dict; it may also carry a ``kind`` key to express a kind-level
    ambiguity (e.g. ``min_icu_los_hours`` vs ``min_hospital_los_hours`` for "stayed at least 24 h").
    The first element is always the original definition. The product is capped at ``max_variants``.
    """
    choices: List[List[tuple]] = []  # per criterion: list of (kind, params)
    for c in defn.criteria:
        opts = [(c.kind, dict(c.params))]
        if c.ambiguous:
            for a in c.alternatives:
                a = dict(a)
                kind = str(a.pop("kind", c.kind))
                if (kind, a) not in opts:
                    opts.append((kind, a))
        choices.append(opts)
    variants: List[CohortDefinition] = []
    for combo in itertools.product(*choices):
        crits = [Criterion(kind=kind, params=dict(p), source=c.source, ambiguous=c.ambiguous, alternatives=list(c.alternatives)) for c, (kind, p) in zip(defn.criteria, combo)]
        d = CohortDefinition.from_dict({**defn.to_dict(), "criteria": [asdict(x) for x in crits]})
        variants.append(d)
        if len(variants) >= max_variants:
            break
    return variants
