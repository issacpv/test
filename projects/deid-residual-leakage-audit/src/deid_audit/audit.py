"""Corpus-level audit driver: per-note counts (regex + optional local NER), aggregate tables, CLI.

The only things that ever leave this module are *counts*: per-note category
counts, placeholder statistics and note length. Text, matched strings and
offsets are discarded before anything is returned or written.

Local NER is optional and strictly offline (spaCy models installed on the
machine). No hosted API is ever called; see the README's ethics section.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from . import estimate
from .patterns import RESIDUAL_CATEGORIES, count_by_category, note_flags, scan
from .sections import DISCHARGE_HEADERS, RADIOLOGY_HEADERS, placeholder_stats, split_sections

NER_LABELS: tuple[str, ...] = ("PERSON", "GPE", "LOC", "FAC", "ORG", "DATE")


def load_local_ner(model: str = "en_core_web_sm"):
    """Return a spaCy pipeline if the model is installed locally, else ``None`` (regex-only audit)."""
    try:
        import spacy  # type: ignore

        return spacy.load(model, disable=["lemmatizer", "textcat"])
    except Exception:  # ImportError or OSError (model missing)
        return None


def ner_counts(text: str, nlp, labels: Iterable[str] = NER_LABELS) -> Counter:
    """Entity-label histogram from a local NER model; entity strings are not retained."""
    wanted = set(labels)
    c: Counter = Counter()
    if nlp is None:
        return c
    for ent in nlp(text).ents:
        if ent.label_ in wanted:
            c[f"ner_{ent.label_}"] += 1
    return c


def audit_note(text: str, note_type: str = "discharge", nlp=None) -> dict[str, int | str]:
    """Counts for one note: regex categories (total + per section), placeholders, NER labels."""
    headers = DISCHARGE_HEADERS if note_type == "discharge" else RADIOLOGY_HEADERS
    secs = split_sections(text, headers)
    findings = scan(text, secs)
    row: dict[str, int | str] = {"note_type": note_type}
    row.update({f"cat_{k}": v for k, v in count_by_category(findings).items()})
    row.update({f"catsec_{k}__{s}": v for (k, s), v in count_by_category(findings, by_section=True).items()})
    row.update({f"flag_{k}": v for k, v in note_flags(findings).items()})
    row.update(placeholder_stats(text, secs))
    row.update(ner_counts(text, nlp))
    return row


def aggregate_notes(records: Iterable[tuple[str, str, str]], nlp=None) -> pd.DataFrame:
    """``records`` yields ``(note_id, note_type, text)``; returns one count row per note (no text)."""
    rows = []
    for note_id, note_type, text in records:
        r = audit_note(text, note_type, nlp)
        r["note_id"] = note_id
        rows.append(r)
    df = pd.DataFrame(rows).fillna(0)
    for c in df.columns:
        if c not in ("note_id", "note_type"):
            df[c] = df[c].astype(int)
    return df


def summarise(df: pd.DataFrame, by: str = "note_type", min_cell: int = 10) -> pd.DataFrame:
    """Note-level residual rates per 10k notes by group and category, with Wilson CIs and cell suppression."""
    rows = []
    for g, sub in df.groupby(by):
        n = len(sub)
        for cat in RESIDUAL_CATEGORIES:
            col = f"flag_{cat}"
            k = int(sub[col].sum()) if col in sub else 0
            r = estimate.rate_per_10k(k, n)
            rows.append({by: g, "category": cat, "n_notes": n, "notes_with_candidate": k,
                         "rate_per_10k": r.rate_per_10k, "ci_low": r.low_per_10k, "ci_high": r.high_per_10k})
    out = pd.DataFrame(rows)
    return estimate.suppress_small_cells(out, ["notes_with_candidate"], min_cell)


def regex_vs_ner_capture(df: pd.DataFrame, regex_flag: str = "flag_name_after_title",
                         ner_col: str = "ner_PERSON") -> estimate.CaptureRecapture:
    """Chapman estimate of notes containing a name-like residual from regex and NER flags."""
    if ner_col not in df:
        raise ValueError("NER counts absent; run with a local spaCy model")
    return estimate.capture_recapture_from_flags(df[regex_flag].to_numpy(), (df[ner_col] > 0).to_numpy())


def iter_csv_notes(path: Path, text_col: str = "text", id_col: str = "note_id",
                   note_type: str = "discharge", chunksize: int = 2000) -> Iterator[tuple[str, str, str]]:
    for chunk in pd.read_csv(path, usecols=[id_col, text_col], chunksize=chunksize):
        for nid, txt in zip(chunk[id_col], chunk[text_col]):
            yield str(nid), note_type, "" if pd.isna(txt) else str(txt)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate residual-identifier audit (counts only).")
    ap.add_argument("--input", type=Path, required=True, help="discharge.csv.gz or radiology.csv.gz")
    ap.add_argument("--note-type", choices=["discharge", "radiology"], default="discharge")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col", default="note_id")
    ap.add_argument("--ner-model", default=None, help="local spaCy model name (optional)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/audit_counts.csv"))
    args = ap.parse_args(argv)
    nlp = load_local_ner(args.ner_model) if args.ner_model else None
    if args.ner_model and nlp is None:
        print(f"local NER model {args.ner_model!r} not available; running regex-only", file=sys.stderr)
    it = iter_csv_notes(args.input, args.text_col, args.id_col, args.note_type)
    if args.limit:
        it = (r for i, r in enumerate(it) if i < args.limit)
    df = aggregate_notes(it, nlp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = summarise(df)
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
