"""Aggregation of scan results into tables, prevalence estimates and Markdown.

The audit unit is the *repository* (one paper -> one repo); a repository is
"flagged" for a rule if any file in it carries that rule at severity >= the
chosen threshold.  Prevalence is reported with Wilson score intervals, which
behave well for the small-to-moderate counts a manual audit yields.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .detector import SEVERITY_ORDER, Finding, ScanResult

LEAKAGE_RULES: Sequence[str] = (
    "split-without-groups",
    "window-level-random-split",
    "preprocess-before-split",
    "feature-selection-before-split",
    "resample-before-split",
    "test-set-in-training",
    "no-holdout-evaluation",
)
POSITIVE_RULES: Sequence[str] = ("group-aware-split-present", "pipeline-present")


def findings_to_frame(results: Iterable[ScanResult], repo: Optional[str] = None) -> pd.DataFrame:
    """Flatten scan results into one row per finding."""
    rows: List[Dict[str, object]] = []
    for res in results:
        for f in res.findings:
            rows.append(
                {
                    "repo": repo,
                    "file": res.file,
                    "rule": f.rule,
                    "severity": f.severity,
                    "severity_rank": SEVERITY_ORDER[f.severity],
                    "lineno": f.lineno,
                    "cell": f.cell,
                    "message": f.message,
                    "n_subject_tokens": len(res.subject_evidence),
                    "subject_tokens": ";".join(res.subject_evidence),
                    "window_tokens": ";".join(res.window_evidence),
                }
            )
    cols = ["repo", "file", "rule", "severity", "severity_rank", "lineno", "cell", "message", "n_subject_tokens", "subject_tokens", "window_tokens"]
    return pd.DataFrame(rows, columns=cols)


def summarise_repo(df: pd.DataFrame, min_severity: str = "medium") -> pd.DataFrame:
    """One row per repo with a boolean column per rule (flagged at >= ``min_severity``)."""
    thr = SEVERITY_ORDER[min_severity]
    if df.empty:
        return pd.DataFrame(columns=["repo", *LEAKAGE_RULES, *POSITIVE_RULES, "any_leakage", "n_files"])
    out = []
    for repo, g in df.groupby("repo", dropna=False):
        row: Dict[str, object] = {"repo": repo, "n_files": g["file"].nunique()}
        for r in LEAKAGE_RULES:
            row[r] = bool(((g["rule"] == r) & (g["severity_rank"] >= thr)).any())
        for r in POSITIVE_RULES:
            row[r] = bool((g["rule"] == r).any())
        row["any_leakage"] = any(row[r] for r in LEAKAGE_RULES)
        out.append(row)
    return pd.DataFrame(out)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def prevalence_table(repo_df: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    """Prevalence of each rule across repos (optionally stratified by a column, e.g. dataset)."""
    rules = [r for r in (*LEAKAGE_RULES, *POSITIVE_RULES, "any_leakage") if r in repo_df.columns]
    groups = [(None, repo_df)] if by is None else list(repo_df.groupby(by))
    rows = []
    for key, g in groups:
        n = len(g)
        for r in rules:
            k = int(g[r].sum())
            lo, hi = wilson_interval(k, n)
            rows.append({"stratum": key, "rule": r, "k": k, "n": n, "prevalence": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def to_markdown(results: Iterable[ScanResult], min_severity: str = "low") -> str:
    """Human-readable report for a set of scan results."""
    thr = SEVERITY_ORDER[min_severity]
    lines = ["# leakscan report", ""]
    n_files = 0
    n_flag = 0
    for res in results:
        n_files += 1
        shown = [f for f in res.findings if SEVERITY_ORDER[f.severity] >= thr]
        if not shown:
            continue
        n_flag += 1
        lines.append(f"## {res.file}")
        lines.append(f"subject evidence: {', '.join(res.subject_evidence) or 'none'}; window evidence: {', '.join(res.window_evidence) or 'none'}")
        lines.append("")
        for f in shown:
            loc = f"line {f.lineno}" + (f" (cell {f.cell})" if f.cell is not None and f.cell >= 0 else "")
            lines.append(f"- **{f.rule}** [{f.severity}] {loc}: {f.message}")
        lines.append("")
    lines.insert(2, f"{n_files} files scanned, {n_flag} with findings at severity >= {min_severity}.")
    lines.insert(3, "")
    return "\n".join(lines)
