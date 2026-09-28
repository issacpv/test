"""Evaluation: cohort-size / prevalence reproduction, criterion agreement, consistency, multiverse."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cohort_spec import CohortDefinition

__all__ = [
    "count_error",
    "prevalence_error",
    "criterion_agreement",
    "self_consistency",
    "multiverse_summary",
    "summarize_benchmark",
    "wilson_ci",
]


def count_error(n_repro: int, n_reported: int, tolerance: float = 0.25) -> Dict[str, float]:
    """Cohort-size reproduction error; ``within_tol`` mirrors Johnson et al. (2017)'s 25% criterion."""
    if n_reported <= 0:
        raise ValueError("n_reported must be positive")
    rel = (n_repro - n_reported) / n_reported
    return {
        "n_repro": int(n_repro),
        "n_reported": int(n_reported),
        "rel_error": float(rel),
        "abs_rel_error": float(abs(rel)),
        "log_ratio": float(np.log(max(n_repro, 1) / n_reported)),
        "within_tol": bool(abs(rel) <= tolerance),
    }


def prevalence_error(p_repro: float, p_reported: Optional[float]) -> Dict[str, float]:
    """Absolute difference in outcome prevalence (percentage points), NaN if not reported."""
    if p_reported is None or not np.isfinite(p_reported):
        return {"prev_repro": float(p_repro), "prev_reported": float("nan"), "prev_abs_diff_pp": float("nan")}
    return {"prev_repro": float(p_repro), "prev_reported": float(p_reported), "prev_abs_diff_pp": float(abs(p_repro - p_reported) * 100)}


def criterion_agreement(pred: CohortDefinition, gold: CohortDefinition) -> Dict[str, float]:
    """Precision / recall / F1 of criteria, exact (kind + params) and kind-only."""
    ps, gs = pred.criterion_set(), gold.criterion_set()
    pk = {c.kind for c in pred.criteria}
    gk = {c.kind for c in gold.criteria}

    def prf(a: set, b: set) -> Dict[str, float]:
        tp = len(a & b)
        p = tp / len(a) if a else (1.0 if not b else 0.0)
        r = tp / len(b) if b else (1.0 if not a else 0.0)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f}

    exact = prf(ps, gs)
    kind = prf(pk, gk)
    out = {f"exact_{k}": v for k, v in exact.items()}
    out.update({f"kind_{k}": v for k, v in kind.items()})
    out["outcome_match"] = float((pred.outcome is None and gold.outcome is None) or (pred.outcome is not None and gold.outcome is not None and pred.outcome.kind == gold.outcome.kind))
    out["unit_match"] = float(pred.unit == gold.unit)
    return out


def self_consistency(defs: Sequence[CohortDefinition]) -> Dict[str, float]:
    """Mean pairwise Jaccard of criterion sets and fraction of identical canonical hashes."""
    if len(defs) < 2:
        return {"mean_jaccard": float("nan"), "identical_frac": float("nan"), "n": len(defs)}
    sets = [d.criterion_set() for d in defs]
    jac = []
    for a, b in combinations(sets, 2):
        u = a | b
        jac.append(len(a & b) / len(u) if u else 1.0)
    hashes = [d.canonical_hash() for d in defs]
    mode_count = max(hashes.count(h) for h in set(hashes))
    return {"mean_jaccard": float(np.mean(jac)), "identical_frac": mode_count / len(defs), "n": len(defs)}


def multiverse_summary(counts: Iterable[int], n_reported: Optional[int], tolerance: float = 0.25) -> Dict[str, float]:
    """Spread of cohort sizes across ambiguity alternatives and whether the reported n is covered."""
    c = np.asarray(list(counts), dtype=float)
    out = {"n_variants": int(c.size), "min": float(c.min()), "max": float(c.max()), "spread_log": float(np.log(c.max() / max(c.min(), 1)))}
    if n_reported:
        lo, hi = c.min() * (1 - tolerance), c.max() * (1 + tolerance)
        out["reported_in_range"] = bool(lo <= n_reported <= hi)
        out["frac_variants_within_tol"] = float(np.mean(np.abs(c - n_reported) / n_reported <= tolerance))
    return out


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_benchmark(rows: pd.DataFrame, by: Sequence[str] = ("model",)) -> pd.DataFrame:
    """Aggregate per-paper rows (columns: model, paper_id, within_tol, abs_rel_error, exact_f1?, executed).

    Returns per-group: n papers, execution-failure rate, within-25% rate with Wilson CI, median
    absolute relative error, and mean criterion F1 where available.
    """
    out: List[Dict[str, object]] = []
    for key, g in rows.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        executed = g["executed"].astype(bool) if "executed" in g else pd.Series(True, index=g.index)
        ok = g.loc[executed]
        k = int(ok["within_tol"].astype(bool).sum())
        n = int(len(g))
        lo, hi = wilson_ci(k, n)
        rec: Dict[str, object] = dict(zip(by, key))
        rec.update({"n_papers": n, "exec_failure_rate": float(1 - executed.mean()), "within_tol_rate": k / n if n else float("nan"), "within_tol_ci_low": lo, "within_tol_ci_high": hi, "median_abs_rel_error": float(ok["abs_rel_error"].median()) if len(ok) else float("nan")})
        if "exact_f1" in g:
            rec["mean_exact_f1"] = float(g["exact_f1"].mean())
        out.append(rec)
    return pd.DataFrame(out)
