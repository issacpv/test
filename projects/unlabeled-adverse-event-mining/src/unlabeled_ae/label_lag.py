"""Label-lag analysis and evaluation against FDA's potential-signals list.

* ``first_labeled_dates``: from per-version term tables (built by applying
  ``term_extractor.extract_from_label`` to each archived SPL version, or from
  the FDA Safety-related Labeling Changes (SrLC) database summaries), the first
  date each PT appeared in a drug's safety sections.
* ``compute_label_lag``: join with ``signal_scan.time_scan`` output → lag in
  months from first FAERS signal to first labelling (negative = labelled before
  the signal emerged; censored = still unlabeled at the data cut).
* ``kaplan_meier``: KM curve of time-to-labelling after signal emergence
  (dependency-free implementation).
* ``load_potential_signals`` / ``evaluate_against_fda``: parse the quarterly FDA
  "Potential Signals of Serious Risks/New Safety Information Identified from
  FAERS" tables (saved HTML or a curated CSV with columns quarter, drug, signal
  text) into (drug, PT) pairs and measure how many are detected by the scan,
  and with what lead time.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_QUARTER_RE = re.compile(r"(January|April|July|October)\s*[-–]\s*(March|June|September|December)\s+(\d{4})", re.IGNORECASE)
_Q_MAP = {"january": 1, "april": 2, "july": 3, "october": 4}


def first_labeled_dates(version_terms: pd.DataFrame, drug_col: str = "drug", date_col: str = "effective_time", pt_col: str = "pt") -> pd.DataFrame:
    """Earliest ``date_col`` at which each (drug, pt) appears in a label version.

    ``version_terms`` has one row per (drug, set_id, version, effective_time, pt)
    from affirmed mentions; multiple set ids (generics) per drug are pooled, so
    the date is the first *any* label of that drug carried the term.
    """
    v = version_terms.dropna(subset=[date_col]).copy()
    v[date_col] = pd.to_datetime(v[date_col])
    out = v.groupby([drug_col, pt_col], as_index=False)[date_col].min().rename(columns={date_col: "first_labeled_date"})
    return out


def srlc_terms(srlc: pd.DataFrame, dictionary, drug_col: str = "drug", date_col: str = "approval_date", text_col: str = "summary") -> pd.DataFrame:
    """Extract PTs from SrLC change summaries -> (drug, pt, approval_date, section)."""
    from .term_extractor import extract_terms

    rows = []
    for _, r in srlc.iterrows():
        for m in extract_terms(str(r[text_col]), dictionary):
            if not m.negated:
                rows.append({drug_col: r[drug_col], "pt": m.pt, "effective_time": pd.to_datetime(r[date_col]), "section": r.get("section", "")})
    return pd.DataFrame(rows, columns=[drug_col, "pt", "effective_time", "section"])


def compute_label_lag(
    signals: pd.DataFrame,
    labeled_dates: pd.DataFrame,
    data_cut: str,
    drug_col: str = "drug",
    pt_col: str = "pt",
) -> pd.DataFrame:
    """Per (drug, pt): months from first signal quarter start to first labelling.

    Columns added: ``first_labeled_date, lag_months, labeled (0/1), category`` in
    {"label_first", "signal_first", "unlabeled_censored", "never_signal"}.
    ``time_to_label_months`` is the KM duration (lag if labelled, else time from
    signal to ``data_cut``).
    """
    s = signals.merge(labeled_dates, on=[drug_col, pt_col], how="left")
    sig_date = s["first_signal_quarter"].map(lambda q: q.start_time if pd.notna(q) else pd.NaT)
    s["first_signal_date"] = sig_date
    cut = pd.Timestamp(data_cut)
    lag = (s["first_labeled_date"] - s["first_signal_date"]).dt.days / 30.4375
    s["lag_months"] = lag
    s["labeled"] = s["first_labeled_date"].notna().astype(int)
    cat = np.where(
        s["first_signal_date"].isna(),
        "never_signal",
        np.where(s["first_labeled_date"].isna(), "unlabeled_censored", np.where(lag < 0, "label_first", "signal_first")),
    )
    s["category"] = cat
    ttl = np.where(s["labeled"] == 1, lag.clip(lower=0), (cut - s["first_signal_date"]).dt.days / 30.4375)
    s["time_to_label_months"] = ttl
    # a term labelled before the signal is not "at risk" of being labelled after it
    s.loc[s["category"] == "label_first", "time_to_label_months"] = np.nan
    return s


def kaplan_meier(durations: Sequence[float], events: Sequence[int]) -> pd.DataFrame:
    """Kaplan-Meier survival table (time, n_at_risk, n_events, survival, se)."""
    t = np.asarray(durations, float)
    e = np.asarray(events, int)
    ok = ~np.isnan(t)
    t, e = t[ok], e[ok]
    order = np.argsort(t)
    t, e = t[order], e[order]
    times = np.unique(t[e == 1])
    surv, rows, var_sum = 1.0, [], 0.0
    for ti in times:
        at_risk = int((t >= ti).sum())
        d = int(((t == ti) & (e == 1)).sum())
        if at_risk == 0:
            continue
        surv *= 1 - d / at_risk
        if at_risk - d > 0:
            var_sum += d / (at_risk * (at_risk - d))
        rows.append({"time": ti, "n_at_risk": at_risk, "n_events": d, "survival": surv, "se": surv * np.sqrt(var_sum)})
    return pd.DataFrame(rows, columns=["time", "n_at_risk", "n_events", "survival", "se"])


def median_survival(km: pd.DataFrame) -> float:
    below = km[km["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if not below.empty else float("inf")


# ------------------------------------------------------- FDA potential signals
def _quarter_from_text(s: str) -> Optional[pd.Period]:
    m = _QUARTER_RE.search(str(s))
    if not m:
        m2 = re.search(r"(\d{4})\s*Q([1-4])", str(s))
        return pd.Period(f"{m2.group(1)}Q{m2.group(2)}", freq="Q") if m2 else None
    return pd.Period(f"{m.group(3)}Q{_Q_MAP[m.group(1).lower()]}", freq="Q")


def parse_potential_signals_html(html: str, quarter_label: str) -> pd.DataFrame:
    """Parse one saved FDA quarterly page with ``pandas.read_html``.

    Tables have columns like 'Product Name: Active Ingredient (Trade Name)',
    'Potential Signal of a Serious Risk / New Safety Information',
    'Additional Information'. Returns ``quarter, product, signal_text, additional``.
    """
    tables = pd.read_html(html)
    frames = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("product" in c for c in cols) and any("signal" in c or "safety" in c for c in cols):
            t = t.copy()
            t.columns = ["product", "signal_text", "additional"][: len(t.columns)] + list(t.columns[3:])
            frames.append(t[["product", "signal_text"] + (["additional"] if "additional" in t.columns else [])])
    if not frames:
        return pd.DataFrame(columns=["quarter", "product", "signal_text", "additional"])
    out = pd.concat(frames, ignore_index=True)
    out["quarter"] = _quarter_from_text(quarter_label)
    return out


def load_potential_signals(path_or_df, dictionary=None, drug_normalizer=None) -> pd.DataFrame:
    """Curated CSV (quarter, product, signal_text[, additional]) -> (quarter, drug, pt) rows.

    Products like 'Atorvastatin (Lipitor)' are split on separators and the
    ingredient(s) before parentheses kept; ``drug_normalizer`` (e.g. upper-case
    salt stripping) is applied. PTs are extracted from ``signal_text`` with the
    dictionary; if none match, the raw text is kept in ``pt`` so recall can be
    audited manually.
    """
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    if not isinstance(df["quarter"].dtype, pd.PeriodDtype):
        df["quarter"] = df["quarter"].map(lambda q: q if isinstance(q, pd.Period) else _quarter_from_text(q))
    rows = []
    for _, r in df.iterrows():
        products = re.split(r";|/|\band\b|,", re.sub(r"\([^)]*\)", "", str(r["product"])))
        products = [p.strip() for p in products if p.strip()]
        pts: List[str] = []
        if dictionary is not None:
            from .term_extractor import extract_terms

            pts = sorted({m.pt for m in extract_terms(str(r["signal_text"]), dictionary)})
        if not pts:
            pts = [str(r["signal_text"]).strip()]
        for p in products:
            d = drug_normalizer(p) if drug_normalizer else p.upper()
            for pt in pts:
                rows.append({"quarter": r["quarter"], "drug": d, "pt": pt, "signal_text": r["signal_text"]})
    return pd.DataFrame(rows, columns=["quarter", "drug", "pt", "signal_text"])


def evaluate_against_fda(
    signals: pd.DataFrame,
    fda: pd.DataFrame,
    max_lead_quarters: int = 40,
) -> Dict[str, object]:
    """Compare scan output with FDA's list.

    For each FDA (drug, pt, quarter): detected = our first_signal_quarter is not
    NaT and <= FDA quarter + 0 (i.e. we had a sustained signal by the time FDA
    listed it); ``lead_quarters`` = FDA quarter - our first signal quarter.
    Also returns a precision proxy: share of our *unlabeled current* signals for
    FDA-listed drugs that appear in the list within ``max_lead_quarters``.
    """
    s = signals.copy()
    s["first_signal_quarter"] = s["first_signal_quarter"].map(lambda q: q if pd.isna(q) else pd.Period(q, freq="Q"))
    f = fda.copy()
    f["quarter"] = f["quarter"].map(lambda q: pd.Period(q, freq="Q") if not isinstance(q, pd.Period) else q)
    m = f.merge(s[["drug", "pt", "first_signal_quarter", "n_current", "ic025_current"]], on=["drug", "pt"], how="left")
    m["detected"] = m["first_signal_quarter"].notna() & (m["first_signal_quarter"] <= m["quarter"])
    m["lead_quarters"] = np.where(
        m["first_signal_quarter"].notna(),
        [((q - fq).n if pd.notna(fq) else np.nan) for q, fq in zip(m["quarter"], m["first_signal_quarter"])],
        np.nan,
    )
    detected_rate = float(m["detected"].mean()) if len(m) else float("nan")
    lead = m.loc[m["detected"], "lead_quarters"]
    # precision proxy among our signals for drugs FDA has ever listed
    fda_pairs = set(zip(f["drug"], f["pt"]))
    ours = s[s["drug"].isin(set(f["drug"])) & s["first_signal_quarter"].notna()]
    if "on_label" in ours:
        ours = ours[~ours["on_label"]]
    hits = [((d, p) in fda_pairs) for d, p in zip(ours["drug"], ours["pt"])]
    precision_proxy = float(np.mean(hits)) if hits else float("nan")
    return {
        "n_fda_signals": int(len(m)),
        "detected_rate": detected_rate,
        "median_lead_quarters": float(lead.median()) if len(lead) else float("nan"),
        "lead_quarters": lead.reset_index(drop=True),
        "precision_proxy": precision_proxy,
        "table": m,
    }
