"""Entity normalisation and blocked fuzzy linking across openFDA device tables.

MAUDE reports carry free-text ``manufacturer_d_name`` / ``brand_name`` and a
``device_report_product_code`` but rarely a K/PMA number; recalls carry
``k_numbers`` only sometimes; 510(k)s carry ``applicant`` and ``device_name``.
The linking strategy is:

1. **Exact keys first**: UDI-DI (GUDID ``premarket_submissions``), recall
   ``k_numbers`` / ``pma_numbers``.
2. **Blocked fuzzy match** within a product code: normalised firm names
   (legal suffixes removed) compared with a token-set / sequence blend; ties
   broken by device/brand-name similarity and by the clearance closest before
   the report date.
3. Everything else stays at the (product code, firm) "device family" level,
   which is the unit used for survival analysis when submission-level linkage
   is impossible.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional accelerator
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except Exception:  # pragma: no cover
    _rf_fuzz = None

_LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l l c", "ltd", "limited", "corp", "corporation", "co", "company",
    "gmbh", "ag", "sa", "s a", "srl", "s r l", "plc", "bv", "b v", "nv", "n v", "kk", "k k", "pty",
    "pte", "ab", "as", "oy", "spa", "s p a", "the", "and", "&", "div", "division", "of",
}
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_TM = re.compile(r"[®™©]")  # ®, ™, ©


def normalize_firm(name: Optional[str]) -> str:
    """Lower-case, strip punctuation and legal suffixes (``"Medtronic, Inc."`` -> ``"medtronic"``)."""
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = _TM.sub("", str(name).lower())
    s = s.replace(".", " ").replace(",", " ").replace("/", " ")
    s = _PUNCT.sub(" ", s)
    toks = [t for t in _WS.split(s.strip()) if t and t not in _LEGAL_SUFFIXES]
    return " ".join(toks)


def normalize_device_name(name: Optional[str]) -> str:
    """Lower-case device/brand name with punctuation and trademark symbols removed."""
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = _TM.sub("", str(name).lower())
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _token_set_ratio(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def similarity(a: str, b: str) -> float:
    """Blend of token-set Jaccard and character sequence ratio in [0, 1]."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if _rf_fuzz is not None:
        seq = _rf_fuzz.ratio(a, b) / 100.0
        tok = _rf_fuzz.token_set_ratio(a, b) / 100.0
    else:
        seq = SequenceMatcher(None, a, b).ratio()
        tok = _token_set_ratio(a, b)
    # a firm name that is a prefix of a longer one ("abbott" vs "abbott vascular")
    prefix = 1.0 if (a.startswith(b) or b.startswith(a)) and min(len(a), len(b)) >= 5 else 0.0
    return float(max(0.5 * seq + 0.5 * tok, 0.9 * prefix))


def device_key(product_code: Optional[str], firm_norm: str, brand_norm: str = "") -> str:
    """Stable device-family key ``<product_code>|<firm>|<brand>``."""
    return f"{(product_code or '').upper()}|{firm_norm}|{brand_norm}"


def blocked_fuzzy_match(
    left: pd.DataFrame,
    right: pd.DataFrame,
    block_col: str,
    left_name: str,
    right_name: str,
    threshold: float = 0.85,
    left_id: str = "left_id",
    right_id: str = "right_id",
    secondary: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Best fuzzy match of each left row to right rows sharing ``block_col``.

    Returns ``left_id, right_id, score[, score2]`` for matches with
    ``score >= threshold``. ``secondary`` = (left_col, right_col) adds a second
    similarity used to break ties (e.g. brand vs device name).
    """
    rows = []
    right_groups = {k: g for k, g in right.groupby(block_col)}
    for blk, lg in left.groupby(block_col):
        rg = right_groups.get(blk)
        if rg is None or rg.empty:
            continue
        r_names = rg[right_name].fillna("").tolist()
        r_ids = rg[right_id].tolist()
        r_sec = rg[secondary[1]].fillna("").tolist() if secondary else None
        # cache per unique left name inside the block
        cache: Dict[str, Tuple[int, float, float]] = {}
        for _, lrow in lg.iterrows():
            lname = lrow[left_name] or ""
            lsec = (lrow[secondary[0]] or "") if secondary else ""
            key = lname + "\x00" + lsec
            if key not in cache:
                best = (-1, 0.0, 0.0)
                for j, rn in enumerate(r_names):
                    s = similarity(lname, rn)
                    if s < threshold:
                        continue
                    s2 = similarity(lsec, r_sec[j]) if secondary else 0.0
                    if (s, s2) > (best[1], best[2]):
                        best = (j, s, s2)
                cache[key] = best
            j, s, s2 = cache[key]
            if j >= 0:
                rows.append({left_id: lrow[left_id], right_id: r_ids[j], "score": s, "score2": s2, block_col: blk})
    return pd.DataFrame(rows, columns=[left_id, right_id, "score", "score2", block_col])


def link_maude_to_510k(
    maude: pd.DataFrame,
    k510: pd.DataFrame,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """Link MAUDE rows (``mdr_report_key, product_code, manufacturer_d_name,
    brand_name, date_received``) to the most plausible 510(k) (same product code,
    similar applicant, cleared before the report). Returns one row per MDR with
    ``k_number`` or NaN and the match score."""
    m = maude.copy()
    m["firm_norm"] = m["manufacturer_d_name"].map(normalize_firm)
    m["brand_norm"] = m["brand_name"].map(normalize_device_name)
    k = k510.copy()
    k["firm_norm"] = k["applicant"].map(normalize_firm)
    k["name_norm"] = k["device_name"].map(normalize_device_name)
    k = k.rename(columns={"k_number": "right_id"})
    m = m.rename(columns={"mdr_report_key": "left_id"})
    matches = blocked_fuzzy_match(
        m, k, "product_code", "firm_norm", "firm_norm", threshold, secondary=("brand_norm", "name_norm")
    )
    if matches.empty:
        out = m.rename(columns={"left_id": "mdr_report_key"})
        out["k_number"] = np.nan
        out["link_score"] = np.nan
        return out[["mdr_report_key", "k_number", "link_score"]]
    matches = matches.rename(columns={"left_id": "mdr_report_key", "right_id": "k_number", "score": "link_score"})
    # enforce clearance before report date when both dates known
    dates = k.set_index("right_id")["decision_date"]
    matches["decision_date"] = matches["k_number"].map(dates)
    rep_dates = m.set_index("left_id")["date_received"]
    matches["date_received"] = matches["mdr_report_key"].map(rep_dates)
    bad = matches["decision_date"].notna() & matches["date_received"].notna() & (matches["decision_date"] > matches["date_received"])
    matches.loc[bad, ["k_number", "link_score"]] = [np.nan, np.nan]
    out = m[["left_id"]].rename(columns={"left_id": "mdr_report_key"}).merge(
        matches[["mdr_report_key", "k_number", "link_score"]], on="mdr_report_key", how="left"
    )
    return out


def family_table(maude: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MAUDE rows to (product_code, firm) device families with counts."""
    m = maude.copy()
    m["firm_norm"] = m["manufacturer_d_name"].map(normalize_firm)
    m["family_key"] = [device_key(pc, f) for pc, f in zip(m["product_code"], m["firm_norm"])]
    g = m.groupby("family_key").agg(
        product_code=("product_code", "first"),
        firm_norm=("firm_norm", "first"),
        n_reports=("mdr_report_key", "nunique"),
        first_report=("date_received", "min"),
        last_report=("date_received", "max"),
    )
    return g.reset_index()
