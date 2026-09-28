"""Predicate-chain graph from 510(k) summaries.

openFDA's ``device/510k`` endpoint does not expose predicate devices. They are
named in the public 510(k) summary/statement PDFs
(``https://www.accessdata.fda.gov/cdrh_docs/pdf<YY>/K<number>.pdf``), so this
module (1) extracts candidate K/DEN/P numbers with contextual cues from summary
text, (2) builds a directed graph child -> predicate with ``networkx``, and
(3) computes predicate-chain features: depth (generations back to a root
device), number of ancestors, ancestor recalls before the child's clearance,
generations to the nearest recalled ancestor, and network centrality.

Edges pointing to a device cleared *after* the child are impossible (or OCR
errors) and are removed by :func:`remove_invalid_edges`, which also guarantees
a DAG so that depth is well defined.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

K_RE = re.compile(r"\b(K\s?\d{6})\b", re.IGNORECASE)
DEN_RE = re.compile(r"\b(DEN\s?\d{6})\b", re.IGNORECASE)
PMA_RE = re.compile(r"\b(P\s?\d{6})(?:/S\d{3})?\b", re.IGNORECASE)
_CUES_PRED = ("predicate", "substantially equivalent", "substantial equivalence", "se to", "equivalent to")
_CUES_REF = ("reference device", "reference devices")


def _norm_num(s: str) -> str:
    return re.sub(r"\s+", "", s.upper())


def extract_predicates(text: str, self_number: Optional[str] = None, window: int = 160) -> List[Dict[str, object]]:
    """Find candidate predicate / reference submission numbers in summary text.

    Returns one dict per distinct number with ``number``, ``kind`` (K/DEN/P),
    ``n_mentions``, ``predicate_cue`` (a predicate/SE cue within ``window``
    characters of any mention) and ``reference_cue``. The submission's own
    number is excluded.
    """
    if not text:
        return []
    found: Dict[str, Dict[str, object]] = {}
    low = text.lower()
    self_n = _norm_num(self_number) if self_number else None
    for kind, rx in (("K", K_RE), ("DEN", DEN_RE), ("P", PMA_RE)):
        for m in rx.finditer(text):
            num = _norm_num(m.group(1))
            if self_n and num == self_n:
                continue
            ctx = low[max(0, m.start() - window) : m.end() + window]
            d = found.setdefault(num, {"number": num, "kind": kind, "n_mentions": 0, "predicate_cue": False, "reference_cue": False})
            d["n_mentions"] = int(d["n_mentions"]) + 1
            d["predicate_cue"] = bool(d["predicate_cue"]) or any(c in ctx for c in _CUES_PRED)
            d["reference_cue"] = bool(d["reference_cue"]) or any(c in ctx for c in _CUES_REF)
    return sorted(found.values(), key=lambda d: (-int(d["n_mentions"]), str(d["number"])))


def summary_pdf_url(k_number: str) -> str:
    """Heuristic URL of the public 510(k) summary PDF on accessdata.fda.gov.

    Folder naming: ``pdf/`` for 1990s numbers (K9xxxxx), ``pdf<Y>/`` for
    2000-2009 (e.g. K053456 -> pdf5), ``pdf<YY>/`` from 2010 (K213456 -> pdf21).
    Fall back to the 510(k) database page
    ``https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=<K>``
    which links the summary if the heuristic 404s.
    """
    k = _norm_num(k_number)
    yy = k[1:3]
    if yy.startswith("9"):
        folder = "pdf"
    elif yy.startswith("0"):
        folder = f"pdf{int(yy)}"
    else:
        folder = f"pdf{yy}"
    return f"https://www.accessdata.fda.gov/cdrh_docs/{folder}/{k}.pdf"


def pdf_to_text(path: str) -> str:
    """Extract text from a PDF with pypdf or pdfminer.six (whichever is installed)."""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except ImportError:
        pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(path)
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install pypdf or pdfminer.six to read 510(k) summary PDFs") from exc


def build_predicate_graph(edges: pd.DataFrame, child_col: str = "k_number", pred_col: str = "predicate", attrs: Optional[pd.DataFrame] = None) -> nx.DiGraph:
    """Directed graph with an edge child -> predicate for each row of ``edges``.

    ``attrs`` (indexed by submission number) may carry ``decision_date``,
    ``product_code``, ``applicant`` etc.; they are copied onto nodes.
    """
    G = nx.DiGraph()
    for c, p in zip(edges[child_col].astype(str).str.upper(), edges[pred_col].astype(str).str.upper()):
        if c and p and c != p:
            G.add_edge(c, p)
    if attrs is not None:
        for node in G.nodes:
            if node in attrs.index:
                G.nodes[node].update({k: v for k, v in attrs.loc[node].to_dict().items() if not (isinstance(v, float) and np.isnan(v))})
    return G


def remove_invalid_edges(G: nx.DiGraph, decision_dates: Dict[str, pd.Timestamp]) -> nx.DiGraph:
    """Drop edges whose predicate was cleared after the child, then break any
    remaining cycles (undated nodes) by removing the edge into the later node in
    each cycle until the graph is a DAG."""
    H = G.copy()
    bad = []
    for c, p in H.edges:
        dc, dp = decision_dates.get(c), decision_dates.get(p)
        if dc is not None and dp is not None and pd.notna(dc) and pd.notna(dp) and dp > dc:
            bad.append((c, p))
    H.remove_edges_from(bad)
    while not nx.is_directed_acyclic_graph(H):
        cycle = nx.find_cycle(H)
        H.remove_edge(*cycle[-1][:2])
    return H


def predicate_depth(G: nx.DiGraph) -> Dict[str, int]:
    """Longest predicate chain below each node (0 = root: cites nothing in graph).

    Requires a DAG (apply :func:`remove_invalid_edges` first).
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("graph has cycles; call remove_invalid_edges first")
    depth: Dict[str, int] = {}
    for node in reversed(list(nx.topological_sort(G))):  # predecessors computed after successors
        succ = list(G.successors(node))
        depth[node] = 0 if not succ else 1 + max(depth[s] for s in succ)
    return depth


def ancestor_recall_exposure(
    G: nx.DiGraph,
    recalls: pd.DataFrame,
    decision_dates: Optional[Dict[str, pd.Timestamp]] = None,
    class_i_only: bool = False,
) -> pd.DataFrame:
    """Per node: recalled ancestors, Class I ancestors, generations to nearest
    recalled ancestor, and whether any *direct* predicate was recalled before the
    node's own clearance (the Kadakia et al. 2023 exposure, generalised).

    ``recalls`` needs ``submission_number, recall_date, recall_class``.
    """
    rec = recalls.dropna(subset=["submission_number"]).copy()
    rec["submission_number"] = rec["submission_number"].astype(str).str.upper()
    if class_i_only:
        rec = rec[rec["recall_class"] == "Class I"]
    first_recall = rec.groupby("submission_number")["recall_date"].min()
    class1 = set(rec.loc[rec["recall_class"] == "Class I", "submission_number"])
    recalled = set(first_recall.index)
    rows = []
    for node in G.nodes:
        anc = nx.descendants(G, node)  # edges point child -> predicate, so "descendants" are ancestors
        anc_recalled = anc & recalled
        anc_class1 = anc & class1
        # generations to nearest recalled ancestor (BFS along predicate edges)
        gen = np.nan
        if anc_recalled:
            lengths = nx.single_source_shortest_path_length(G, node)
            gen = min(lengths[a] for a in anc_recalled)
        direct = set(G.successors(node))
        direct_recalled_before = False
        if decision_dates is not None and node in decision_dates and pd.notna(decision_dates[node]):
            d0 = decision_dates[node]
            direct_recalled_before = any((p in first_recall.index) and pd.notna(first_recall[p]) and first_recall[p] <= d0 for p in direct)
        any_anc_recalled_before = False
        if decision_dates is not None and node in decision_dates and pd.notna(decision_dates[node]):
            d0 = decision_dates[node]
            any_anc_recalled_before = any(pd.notna(first_recall[a]) and first_recall[a] <= d0 for a in anc_recalled)
        rows.append(
            {
                "k_number": node,
                "n_ancestors": len(anc),
                "n_recalled_ancestors": len(anc_recalled),
                "n_class1_ancestors": len(anc_class1),
                "generations_to_recalled_ancestor": gen,
                "direct_predicate_recalled_before_clearance": direct_recalled_before,
                "any_ancestor_recalled_before_clearance": any_anc_recalled_before,
            }
        )
    return pd.DataFrame(rows).set_index("k_number")


def graph_features(G: nx.DiGraph) -> pd.DataFrame:
    """Depth, degree, ancestor/descendant counts, PageRank on the reversed graph
    (influence of a device as a predicate) and weakly-connected component size."""
    depth = predicate_depth(G)
    pr = nx.pagerank(G.reverse(copy=True)) if G.number_of_edges() else {n: 0.0 for n in G.nodes}
    comp_size: Dict[str, int] = {}
    for comp in nx.weakly_connected_components(G):
        for n in comp:
            comp_size[n] = len(comp)
    rows = []
    for n in G.nodes:
        rows.append(
            {
                "k_number": n,
                "predicate_depth": depth[n],
                "n_predicates": G.out_degree(n),
                "n_children": G.in_degree(n),
                "n_ancestors": len(nx.descendants(G, n)),
                "n_descendants": len(nx.ancestors(G, n)),
                "pagerank_as_predicate": pr.get(n, 0.0),
                "component_size": comp_size.get(n, 1),
            }
        )
    return pd.DataFrame(rows).set_index("k_number")


def edges_from_summaries(summaries: Dict[str, str], require_cue: bool = True) -> pd.DataFrame:
    """``{k_number: summary_text}`` -> edge table (child, predicate, cue flags)."""
    rows = []
    for k, text in summaries.items():
        for d in extract_predicates(text, self_number=k):
            if require_cue and not (d["predicate_cue"] or d["reference_cue"]):
                continue
            rows.append({"k_number": _norm_num(k), "predicate": d["number"], "kind": d["kind"], "predicate_cue": d["predicate_cue"], "reference_cue": d["reference_cue"], "n_mentions": d["n_mentions"]})
    return pd.DataFrame(rows, columns=["k_number", "predicate", "kind", "predicate_cue", "reference_cue", "n_mentions"])
