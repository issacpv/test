"""Chief-complaint informativeness features and interpreter-mention detection (local regex only)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LANGUAGE_BARRIER_PATTERNS: tuple[str, ...] = (
    r"language barrier",
    r"\binterpreter\b",
    r"\bper (family|daughter|son|wife|husband|friend|ems)\b",
    r"unable to (obtain|provide) (hx|history)",
    r"hx (obtained )?(from|per) ",
    r"non[- ]?english",
    r"(spanish|portuguese|chinese|russian|creole|vietnamese|arabic)[- ]speaking",
    r"\bno english\b",
    r"\blimited english\b",
)
_BARRIER_RE = re.compile("|".join(LANGUAGE_BARRIER_PATTERNS), flags=re.IGNORECASE)

INTERPRETER_PATTERNS: tuple[str, ...] = (
    r"\binterpreter\b",
    r"\binterpretor\b",
    r"language line",
    r"(phone|video|in[- ]person) interpret",
    r"with (the )?(assistance|help) of (an? )?(\w+ )?interpret",
    r"interpreted by",
)
_INTERP_RE = re.compile("|".join(INTERPRETER_PATTERNS), flags=re.IGNORECASE)

GENERIC_TOKENS: frozenset[str] = frozenset(
    {"pain", "eval", "evaluation", "abnormal", "other", "n/a", "unknown", "weakness", "ill", "sick", "transfer", "general", "unspecified", "concern", "complaint", "issue", "problem", "wound", "check"}
)
_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\band\b|\+|&)\s*", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def chief_complaint_features(cc: pd.Series) -> pd.DataFrame:
    """Per-visit informativeness features from the triage chief complaint.

    Columns: cc_n_tokens, cc_n_items, cc_vagueness (share of generic tokens),
    cc_barrier (language-barrier marker present), cc_missing.
    """
    s = cc.fillna("").astype(str).str.strip().str.lower()
    rows = []
    for text in s:
        tokens = _TOKEN_RE.findall(text)
        items = [i for i in _SPLIT_RE.split(text) if i.strip()]
        n_tok = len(tokens)
        vague = float(np.mean([t in GENERIC_TOKENS for t in tokens])) if n_tok else np.nan
        rows.append({"cc_n_tokens": n_tok, "cc_n_items": len(items), "cc_vagueness": vague, "cc_barrier": int(bool(_BARRIER_RE.search(text))), "cc_missing": int(n_tok == 0)})
    return pd.DataFrame(rows, index=cc.index)


def interpreter_mentions(text: str) -> int:
    """Number of interpreter-related mentions in one clinical note (local regex; no external APIs)."""
    if not isinstance(text, str):
        return 0
    return len(_INTERP_RE.findall(text))


def interpreter_mentions_by_hadm(notes: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Aggregate interpreter mentions per hadm_id: n_notes, n_mentions, any_interpreter."""
    n = notes[["hadm_id", text_col]].copy()
    n["m"] = n[text_col].map(interpreter_mentions)
    g = n.groupby("hadm_id").agg(n_notes=("m", "size"), n_mentions=("m", "sum"))
    g["any_interpreter"] = (g["n_mentions"] > 0).astype(int)
    return g.reset_index()


def chief_complaint_clusters(cc: pd.Series, n_clusters: int = 30, seed: int = 0, min_df: int = 5) -> tuple[pd.Series, object]:
    """TF-IDF + k-means clusters of the chief complaint for use as a covariate. Returns (labels, fitted pipeline)."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    s = cc.fillna("").astype(str).str.lower()
    pipe = make_pipeline(TfidfVectorizer(min_df=min_df, ngram_range=(1, 2), token_pattern=r"[a-z0-9']+"), KMeans(n_clusters=n_clusters, random_state=seed, n_init=5))
    labels = pipe.fit_predict(s)
    return pd.Series(labels, index=cc.index, name="cc_cluster"), pipe
