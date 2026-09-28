"""Link the FDA AI-enabled device list to MAUDE reports.

Tiers (recorded per match so that downstream analyses can restrict to the strictest):

* tier 1: same primary product code and brand-name similarity >= ``brand_threshold`` to the listed device name;
* tier 2: same product code and manufacturer similarity >= ``company_threshold`` to the listed company;
* tier 3: same product code only (used for building comparator sets, never as an AI-device match).

Similarity is ``rapidfuzz.fuzz.token_set_ratio`` when available, otherwise a token-set Jaccard blended with
``difflib.SequenceMatcher`` — both on normalised names.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:  # optional dependency
    from rapidfuzz import fuzz as _fuzz  # type: ignore
except Exception:  # noqa: BLE001
    _fuzz = None

LEGAL_SUFFIXES = r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|gmbh|ag|sa|sas|srl|bv|nv|plc|pty|kk|s\.?p\.?a)\b\.?"
GENERIC_TOKENS = {"the", "of", "and", "system", "systems", "software", "medical", "technologies", "technology", "solutions", "healthcare", "health"}

COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "decision_date": ("date of final decision", "decision date", "date"),
    "submission_number": ("submission number", "submission", "k number", "510(k) number", "number"),
    "device": ("device", "device name", "product"),
    "company": ("company", "applicant", "manufacturer"),
    "panel": ("panel (lead)", "panel", "lead panel"),
    "product_code": ("primary product code", "product code"),
}


def normalize_name(s: object) -> str:
    """Lower-case, strip punctuation/legal suffixes/generic tokens, collapse whitespace."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).lower()
    t = re.sub(r"[®™©]", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(LEGAL_SUFFIXES, " ", t)
    tokens = [w for w in t.split() if w not in GENERIC_TOKENS]
    return " ".join(tokens)


def similarity(a: str, b: str) -> float:
    """Name similarity in [0, 1] on normalised strings."""
    a, b = normalize_name(a), normalize_name(b)
    if not a or not b:
        return 0.0
    if _fuzz is not None:
        return float(_fuzz.token_set_ratio(a, b)) / 100.0
    ta, tb = set(a.split()), set(b.split())
    jacc = len(ta & tb) / len(ta | tb)
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    contain = 1.0 if (ta <= tb or tb <= ta) else 0.0
    return float(max(0.5 * jacc + 0.5 * seq, 0.85 * contain))


def load_ai_device_list(path: Path | str) -> pd.DataFrame:
    """Read the FDA AI-enabled device list (xlsx/csv) into the canonical columns.

    Columns are matched by lower-cased aliases (``COLUMN_ALIASES``); unmatched canonical columns are NaN.
    """
    path = Path(path)
    raw = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    return canonicalise_ai_list(raw)


def canonicalise_ai_list(raw: pd.DataFrame) -> pd.DataFrame:
    lower = {c: str(c).strip().lower() for c in raw.columns}
    out = pd.DataFrame(index=raw.index)
    for canon, aliases in COLUMN_ALIASES.items():
        col = next((c for c, lc in lower.items() if lc in aliases), None)
        if col is None:
            col = next((c for c, lc in lower.items() if any(a in lc for a in aliases)), None)
        out[canon] = raw[col] if col is not None else np.nan
    out["product_code"] = out["product_code"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    out["submission_number"] = out["submission_number"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    out["decision_date"] = pd.to_datetime(out["decision_date"], errors="coerce")
    out["device_norm"] = out["device"].map(normalize_name)
    out["company_norm"] = out["company"].map(normalize_name)
    return out.reset_index(drop=True)


def candidate_matches(
    ai_devices: pd.DataFrame,
    maude: pd.DataFrame,
    brand_threshold: float = 0.85,
    company_threshold: float = 0.80,
) -> pd.DataFrame:
    """Match flattened MAUDE rows to AI devices within product code; returns one row per (report, device) pair.

    ``maude`` needs ``report_number, product_code, brand_name, manufacturer_d_name``. Output columns:
    ``report_number, ai_index, submission_number, brand_score, company_score, tier`` (1, 2 or 3).
    """
    rows = []
    by_code = {c: g for c, g in ai_devices.dropna(subset=["product_code"]).groupby("product_code")}
    for _, r in maude.iterrows():
        code = str(r.get("product_code") or "").upper()
        if code not in by_code:
            continue
        for ai_idx, d in by_code[code].iterrows():
            bs = similarity(str(r.get("brand_name") or ""), str(d["device"]))
            cs = similarity(str(r.get("manufacturer_d_name") or ""), str(d["company"]))
            tier = 1 if bs >= brand_threshold else (2 if cs >= company_threshold else 3)
            rows.append({"report_number": r["report_number"], "ai_index": ai_idx, "submission_number": d["submission_number"], "brand_score": bs, "company_score": cs, "tier": tier})
    cols = ["report_number", "ai_index", "submission_number", "brand_score", "company_score", "tier"]
    return pd.DataFrame(rows, columns=cols)


def best_match_per_report(matches: pd.DataFrame, max_tier: int = 2) -> pd.DataFrame:
    """Keep, per report, the best-scoring AI device with tier <= ``max_tier``."""
    m = matches[matches["tier"] <= max_tier].copy()
    if m.empty:
        return m
    m["score"] = m[["brand_score", "company_score"]].max(axis=1)
    m = m.sort_values(["report_number", "tier", "score"], ascending=[True, True, False])
    return m.drop_duplicates("report_number").drop(columns="score").reset_index(drop=True)


def linkage_report(matches: pd.DataFrame, ai_devices: pd.DataFrame) -> Dict[str, object]:
    """Summary counts: reports per tier, AI devices with >= 1 tier-1/2 report, score distributions."""
    best = best_match_per_report(matches, max_tier=2)
    per_tier = best["tier"].value_counts().to_dict() if len(best) else {}
    n_dev = best["ai_index"].nunique() if len(best) else 0
    return {
        "n_reports_tier1": int(per_tier.get(1, 0)),
        "n_reports_tier2": int(per_tier.get(2, 0)),
        "n_reports_tier3_only": int(matches.loc[~matches["report_number"].isin(best["report_number"]), "report_number"].nunique()) if len(matches) else 0,
        "n_ai_devices_with_reports": int(n_dev),
        "n_ai_devices_total": int(len(ai_devices)),
        "brand_score_quantiles": matches["brand_score"].quantile([0.1, 0.5, 0.9]).round(3).to_dict() if len(matches) else {},
    }


def comparator_devices(k510: pd.DataFrame, ai_devices: pd.DataFrame, years_window: int = 2) -> pd.DataFrame:
    """Non-AI 510(k) devices in the same product code cleared within +/- ``years_window`` years of an AI device.

    ``k510`` needs ``k_number, product_code, applicant, device_name, decision_date``. Returns the comparator
    rows with the matched AI device index.
    """
    k = k510.copy()
    k["decision_date"] = pd.to_datetime(k["decision_date"], errors="coerce")
    ai_subs = set(ai_devices["submission_number"].dropna().astype(str).str.upper())
    k = k[~k["k_number"].astype(str).str.upper().isin(ai_subs)]
    rows = []
    for ai_idx, d in ai_devices.dropna(subset=["product_code", "decision_date"]).iterrows():
        c = k[(k["product_code"].astype(str).str.upper() == d["product_code"]) & (np.abs((k["decision_date"] - d["decision_date"]).dt.days) <= 365.25 * years_window)]
        for _, r in c.iterrows():
            rows.append({"ai_index": ai_idx, **r.to_dict()})
    return pd.DataFrame(rows)
