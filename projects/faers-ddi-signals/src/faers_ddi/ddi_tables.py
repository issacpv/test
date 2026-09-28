"""Build 2 x 2 x 2 (drug A x drug B x event) contingency counts from FAERS reports.

Notation (report counts) follows Noren et al. (2008):

    n_ijk : i = 1 if drug A on the report, j = 1 if drug B, k = 1 if event E
    n_ij+ : reports with drug status (i, j) regardless of event

so ``n111`` is the number of reports listing A, B and E together, ``n11+``
the number listing A and B, ``f_ij = n_ij1 / n_ij+`` the relative reporting
rate of E in stratum (i, j).

Every report counts once in exactly one (i, j) stratum, whatever the number
of drugs or reactions it lists (report-level, not drug-event-pair-level,
counting). Stratifying by sex is done by simply restricting the report set
before counting, so the background (n_00k) is sex-specific.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

KEYS = ("n111", "n110", "n101", "n100", "n011", "n010", "n001", "n000")


def _as_set(x: Iterable[str]) -> Set[str]:
    return {str(v).upper() for v in x}


def index_reports(reports: Sequence[Dict], drug_field: str = "drugs") -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]], int]:
    """Invert flattened reports into ``drug -> {row ids}`` and ``event -> {row ids}``.

    ``drug_field`` selects which role list to use (``"drugs"`` for any role,
    ``"suspect"`` for suspect-only cohorts, etc.).
    """
    drug_idx: Dict[str, Set[int]] = {}
    event_idx: Dict[str, Set[int]] = {}
    for i, r in enumerate(reports):
        for d in _as_set(r.get(drug_field, [])):
            drug_idx.setdefault(d, set()).add(i)
        for e in _as_set(r.get("reactions", [])):
            event_idx.setdefault(e, set()).add(i)
    return drug_idx, event_idx, len(reports)


def three_way_counts(
    a_reports: Set[int],
    b_reports: Set[int],
    e_reports: Set[int],
    n_total: int,
    universe: Optional[Set[int]] = None,
) -> Dict[str, int]:
    """Return the eight ``n_ijk`` cells for one (A, B, E) triple.

    ``universe`` optionally restricts all cells to a subset of report ids
    (e.g. one sex), in which case ``n_total`` is ignored.
    """
    if universe is not None:
        a_reports, b_reports, e_reports = a_reports & universe, b_reports & universe, e_reports & universe
        n_total = len(universe)
    ab = a_reports & b_reports
    n111 = len(ab & e_reports)
    n110 = len(ab) - n111
    a_only = a_reports - b_reports
    n101 = len(a_only & e_reports)
    n100 = len(a_only) - n101
    b_only = b_reports - a_reports
    n011 = len(b_only & e_reports)
    n010 = len(b_only) - n011
    neither_e = len(e_reports) - n111 - n101 - n011
    n001 = neither_e
    n000 = n_total - n111 - n110 - n101 - n100 - n011 - n010 - n001
    return dict(zip(KEYS, (n111, n110, n101, n100, n011, n010, n001, n000)))


def candidate_pairs(drug_idx: Dict[str, Set[int]], min_coreports: int = 20, drugs: Optional[Sequence[str]] = None) -> List[Tuple[str, str, int]]:
    """Enumerate unordered drug pairs co-reported at least ``min_coreports`` times.

    Restricting ``drugs`` to a curated list (e.g. the 100 most-reported
    substances plus a mechanism panel) keeps the pair space tractable: FAERS
    has > 10 000 distinct generic names, i.e. > 5e7 unordered pairs.
    """
    names = [d for d in (drugs or drug_idx.keys()) if d in drug_idx and len(drug_idx[d]) >= min_coreports]
    out = []
    for a, b in combinations(sorted(names), 2):
        n_ab = len(drug_idx[a] & drug_idx[b])
        if n_ab >= min_coreports:
            out.append((a, b, n_ab))
    return sorted(out, key=lambda t: -t[2])


def pair_event_table(
    reports: Sequence[Dict],
    pairs: Sequence[Tuple[str, str]],
    min_n111: int = 3,
    drug_field: str = "drugs",
    sex: Optional[str] = None,
    top_events: Optional[int] = None,
) -> pd.DataFrame:
    """Compute ``n_ijk`` for every (pair, event) with ``n111 >= min_n111``.

    Returns a frame with columns ``drug_a, drug_b, event, n111 ... n000``.
    If ``sex`` is given ("female"/"male"), all cells are restricted to reports
    of that sex.
    """
    drug_idx, event_idx, n_total = index_reports(reports, drug_field)
    universe: Optional[Set[int]] = None
    if sex is not None:
        universe = {i for i, r in enumerate(reports) if r.get("sex") == sex}
    if top_events is not None:
        keep = [e for e, _ in Counter({e: len(v) for e, v in event_idx.items()}).most_common(top_events)]
        event_idx = {e: event_idx[e] for e in keep}
    rows = []
    for a, b in pairs:
        if a not in drug_idx or b not in drug_idx:
            continue
        ab = drug_idx[a] & drug_idx[b]
        if universe is not None:
            ab = ab & universe
        # only events that actually occur on A+B reports can reach n111 >= min_n111
        ev_counts = Counter()
        for i in ab:
            ev_counts.update(_as_set(reports[i].get("reactions", [])))
        for e, c in ev_counts.items():
            if c < min_n111 or e not in event_idx:
                continue
            cells = three_way_counts(drug_idx[a], drug_idx[b], event_idx[e], n_total, universe)
            rows.append({"drug_a": a, "drug_b": b, "event": e, **cells})
    cols = ["drug_a", "drug_b", "event", *KEYS]
    return pd.DataFrame(rows, columns=cols)


def sex_stratified_tables(reports: Sequence[Dict], pairs: Sequence[Tuple[str, str]], **kw) -> pd.DataFrame:
    """Female and male ``n_ijk`` tables side by side (suffixes ``_f`` / ``_m``)."""
    f = pair_event_table(reports, pairs, sex="female", **kw)
    m = pair_event_table(reports, pairs, sex="male", **kw)
    return f.merge(m, on=["drug_a", "drug_b", "event"], how="outer", suffixes=("_f", "_m")).fillna(0)
