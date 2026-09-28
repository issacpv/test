"""Extract spine-density statements from text and normalise them to spines per micrometre.

The grammar targets patterns such as::

    "spine density was 1.8 ± 0.3 spines/µm"
    "12.4 spines per 10 µm of dendrite"
    "density of 0.9 spines per micron (n = 12 cells)"
    "spine densities ranged from 8 to 14 per 100 μm"
    "1.2 spines/μm2"   (surface density, flagged and kept separate)

Each match yields a record with the raw number(s), the unit as written, the normalised linear
density (spines/um) or surface density, an uncertainty if given, and context tags for species,
method, imaging, region, compartment and condition drawn from keyword dictionaries applied to a
window around the match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

MICRO = r"(?:µ|μ|u|micro)"
UM = rf"(?:{MICRO}m|micrometers?|micrometres?|microns?)"
NUM = r"(\d+(?:[.,]\d+)?)"
PM = r"(?:±|\+/-|\+/−|±|plus or minus)"

# number [± err] [spines] (per|/) [k] um[^2]
_DENSITY = re.compile(
    rf"{NUM}\s*(?:{PM}\s*{NUM})?\s*(?:spines?|protrusions?)?\s*(?:per|/)\s*(\d+(?:\.\d+)?)?\s*{UM}\s*(?:(\^?2|²|squared)|(?:of\s+)?(?:dendrit\w*|length)?)",
    re.IGNORECASE,
)
# "from A to B per 10 um" ranges
_RANGE = re.compile(
    rf"(?:from|between)\s*{NUM}\s*(?:to|and|-|–)\s*{NUM}\s*(?:spines?)?\s*(?:per|/)\s*(\d+(?:\.\d+)?)?\s*{UM}(?!\s*(?:\^?2|²))",
    re.IGNORECASE,
)
_N_CELLS = re.compile(r"n\s*=\s*(\d+)\s*(cells?|neurons?|dendrites?|segments?|animals?|mice|rats)", re.IGNORECASE)

KEYWORDS: Dict[str, Dict[str, Sequence[str]]] = {
    "species": {
        "mouse": ("mouse", "mice", "murine"), "rat": ("rat", "rats"), "human": ("human", "patient", "patients", "postmortem", "post-mortem"),
        "macaque": ("macaque", "monkey", "rhesus"), "marmoset": ("marmoset",), "ferret": ("ferret",), "cat": ("feline", " cat "),
    },
    "method": {
        "golgi": ("golgi", "golgi-cox"), "dii": ("dii", "diolistic", "gene gun"), "biocytin": ("biocytin", "neurobiotin"),
        "lucifer_yellow": ("lucifer yellow",), "gfp": ("gfp", "yfp", "thy1", "egfp", "tdtomato", "fluorescent protein", "viral"),
        "em": ("electron microscop", "serial section", "ssem", "fib-sem", "sbem", "sbf-sem", "ultrastructur", "connectom"),
    },
    "imaging": {
        "confocal": ("confocal",), "two_photon": ("two-photon", "2-photon", "two photon", "multiphoton"),
        "sted": ("sted", "super-resolution", "superresolution", "storm", "expansion microscopy"), "brightfield": ("bright-field", "brightfield", "light microscop"),
    },
    "region": {
        "hippocampus_ca1": ("ca1",), "hippocampus_ca3": ("ca3",), "dentate": ("dentate", "granule cell"), "prefrontal": ("prefrontal", "pfc", "mpfc"),
        "visual": ("visual cortex", "v1", "striate"), "somatosensory": ("somatosensory", "barrel", "s1"), "motor": ("motor cortex", "m1"),
        "striatum": ("striat", "medium spiny", "msn", "nucleus accumbens"), "amygdala": ("amygdal",), "cerebellum": ("purkinje", "cerebell"),
        "temporal": ("temporal cortex", "temporal lobe"),
    },
    "compartment": {
        "basal": ("basal",), "apical_trunk": ("apical trunk", "main apical", "apical shaft"), "apical_oblique": ("oblique",),
        "apical_tuft": ("tuft",), "apical": ("apical",), "proximal": ("proximal",), "distal": ("distal",),
    },
    "layer": {"L1": ("layer 1", "layer i "), "L2/3": ("layer 2/3", "layer ii/iii", "layer 2", "layer 3", "l2/3"), "L4": ("layer 4", "layer iv"),
              "L5": ("layer 5", "layer v ", "l5"), "L6": ("layer 6", "layer vi")},
    "cell_type": {"pyramidal": ("pyramidal",), "granule": ("granule",), "msn": ("medium spiny", "msn"), "purkinje": ("purkinje",),
                  "interneuron": ("interneuron",)},
    "condition": {"control": ("control", "wild-type", "wild type", "wt ", "sham", "vehicle", "naive", "baseline"),
                  "treated": ("knockout", "ko ", "transgenic", "mutant", "treated", "stress", "lesion", "injected", "deprived", "enriched", "model")},
    "shrinkage": {"corrected": ("shrinkage", "corrected for shrinkage", "shrinkage correction")},
    "hidden_spines": {"corrected": ("hidden spines", "feldman and peters", "feldman & peters", "obscured spines")},
}


@dataclass
class Statement:
    value: float
    value_high: Optional[float]
    uncertainty: Optional[float]
    per_length: Optional[float]
    unit_text: str
    is_surface: bool
    density_per_um: Optional[float]
    density_high_per_um: Optional[float]
    n_reported: Optional[int]
    n_unit: Optional[str]
    start: int
    end: int
    sentence: str
    tags: Dict[str, List[str]]

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["tags"] = {k: "|".join(v) for k, v in self.tags.items()}
        return d


def _num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    return float(s.replace(",", "."))


def normalise_density(value: float, per_length: Optional[float], is_surface: bool) -> Optional[float]:
    """Convert ``value spines per (per_length) um`` to spines per um; None for surface densities."""
    if is_surface:
        return None
    k = per_length if per_length and per_length > 0 else 1.0
    return value / k


def tag_context(text: str, keywords: Dict[str, Dict[str, Sequence[str]]] = KEYWORDS) -> Dict[str, List[str]]:
    """Keyword-dictionary tags found in a text window (lower-cased substring matching)."""
    low = f" {text.lower()} "
    tags: Dict[str, List[str]] = {}
    for field, table in keywords.items():
        found = [label for label, kws in table.items() if any(k in low for k in kws)]
        if field == "compartment" and "apical" in found and any(f.startswith("apical_") for f in found):
            found.remove("apical")
        tags[field] = found
    return tags


def _sentence_bounds(text: str, pos: int) -> tuple:
    starts = [m.end() for m in re.finditer(r"[.!?]\s+(?=[A-Z(])", text[:pos])]
    s = starts[-1] if starts else 0
    m = re.search(r"[.!?]\s+(?=[A-Z(])", text[pos:])
    e = pos + m.end() if m else len(text)
    return s, e


def extract_statements(text: str, window: int = 400, require_spine_context: bool = True) -> List[Statement]:
    """All density statements in ``text`` with normalised values and context tags."""
    out: List[Statement] = []
    taken: List[tuple] = []

    def _overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in taken)

    for m in _RANGE.finditer(text):
        lo, hi, k = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        s, e = _sentence_bounds(text, m.start())
        ctx = text[max(0, m.start() - window): m.end() + window]
        if require_spine_context and "spine" not in ctx.lower():
            continue
        out.append(Statement(lo, hi, None, k, m.group(0), False, normalise_density(lo, k, False), normalise_density(hi, k, False),
                             *_n(ctx), m.start(), m.end(), text[s:e].strip(), tag_context(ctx)))
        taken.append((m.start(), m.end()))
    for m in _DENSITY.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        val, err, k = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        is_surface = m.group(4) is not None
        s, e = _sentence_bounds(text, m.start())
        ctx = text[max(0, m.start() - window): m.end() + window]
        if require_spine_context and "spine" not in ctx.lower():
            continue
        out.append(Statement(val, None, err, k, m.group(0), is_surface, normalise_density(val, k, is_surface), None,
                             *_n(ctx), m.start(), m.end(), text[s:e].strip(), tag_context(ctx)))
        taken.append((m.start(), m.end()))
    out.sort(key=lambda st: st.start)
    return out


def _n(ctx: str) -> tuple:
    m = _N_CELLS.search(ctx)
    return (int(m.group(1)), m.group(2).lower()) if m else (None, None)


def statements_to_records(statements: Sequence[Statement], pmcid: str = "") -> List[Dict[str, object]]:
    recs = []
    for i, st in enumerate(statements):
        d = st.as_dict()
        d["pmcid"], d["statement_id"] = pmcid, f"{pmcid}:{i}"
        recs.append(d)
    return recs


__all__ = ["Statement", "extract_statements", "normalise_density", "tag_context", "statements_to_records", "KEYWORDS"]
