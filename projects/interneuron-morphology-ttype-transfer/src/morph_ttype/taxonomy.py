"""Harmonise mouse and human t-type names to a cross-species interneuron subclass.

Mouse t-types (Allen V1 / mini-atlas) look like "Pvalb Tpbg", "Sst Calb2 Pdlim5",
"Lamp5 Lsp1", "Sncg Vip Nptx2", "Vip Chat Htr1f", "Sst Chodl". Human MTG t-types look
like "Inh L1-2 PAX6 CDH12", "Inh L2-4 PVALB WFDC2", "Inh L1-3 SST CALB1",
"Inh L1-2 LAMP5 DBP", "Inh L1 SST CHRNA4" (Hodge et al. 2019 naming).

The harmonised subclasses follow the consensus cross-species taxonomy
(Bakken et al. 2021): Pvalb, Sst, Vip, Lamp5, Sncg (human PAX6 aligned to mouse
Sncg by default; see ``HUMAN_PAX6_TARGET``), plus Sst Chodl and Meis2 kept
separate as ``other`` unless requested.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

SUBCLASSES = ("Pvalb", "Sst", "Vip", "Lamp5", "Sncg")
HUMAN_PAX6_TARGET = "Sncg"  # sensitivity parameter: alternatives "Lamp5" or "other"

_MARKERS: Dict[str, str] = {
    "pvalb": "Pvalb", "sst": "Sst", "vip": "Vip", "lamp5": "Lamp5", "sncg": "Sncg",
    "pax6": HUMAN_PAX6_TARGET, "adarb2": "Sncg",
}
_HUMAN_PREFIX = re.compile(r"^inh\s+l\d+(?:-\d+)?\s+", re.IGNORECASE)


def marker_tokens(t_type: str) -> List[str]:
    """Lowercase tokens of a t-type after stripping the human 'Inh L1-2 ' prefix."""
    s = _HUMAN_PREFIX.sub("", str(t_type).strip())
    return [t.lower() for t in re.split(r"[\s_/]+", s) if t]


def harmonise_subclass(t_type: object, keep_chodl: bool = False, pax6_target: Optional[str] = None) -> str:
    """Map a t-type name to a harmonised subclass ('other' when no marker is recognised)."""
    if t_type is None or (isinstance(t_type, float) and t_type != t_type):
        return "other"
    toks = marker_tokens(str(t_type))
    if not toks:
        return "other"
    if "chodl" in toks and "sst" in toks:
        return "Sst Chodl" if keep_chodl else "other"
    if "meis2" in toks:
        return "other"
    first = toks[0]
    if first == "pax6":
        return pax6_target or HUMAN_PAX6_TARGET
    if first in _MARKERS:
        return _MARKERS[first]
    # some names lead with a layer/other token; fall back to the first recognised marker anywhere
    for t in toks:
        if t in _MARKERS:
            return _MARKERS[t] if t != "pax6" else (pax6_target or HUMAN_PAX6_TARGET)
    return "other"


def is_interneuron_ttype(t_type: object) -> bool:
    s = str(t_type).lower()
    return s.startswith("inh") or any(s.startswith(m) for m in _MARKERS)


def species_of_ttype(t_type: object) -> str:
    """'human' for Hodge-style names ('Inh L...'), else 'mouse'."""
    return "human" if _HUMAN_PREFIX.match(str(t_type).strip()) else "mouse"


def audit(t_types: Iterable[object]) -> Dict[str, Dict[str, int]]:
    """Count t-types per harmonised subclass to check the mapping by eye."""
    out: Dict[str, Dict[str, int]] = {}
    for t in t_types:
        sc = harmonise_subclass(t)
        out.setdefault(sc, {})
        out[sc][str(t)] = out[sc].get(str(t), 0) + 1
    return out
