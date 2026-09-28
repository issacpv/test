"""Reference sets and performance evaluation for DDI signal detection.

Three complementary "truth" sources are supported:

1. **Curated clinical DDI lists** loaded from local CSV files that the user
   downloads (see ``data/README.md``): the ONC high-priority DDI list
   (Phansalkar et al., 2012, *JAMIA*), CredibleMeds QTdrugs categories (free
   registration), and the FDA "Drug Development and Drug Interactions" tables
   of CYP substrates / inhibitors / inducers. Each is reduced to a set of
   unordered (drug_a, drug_b) pairs plus an optional expected event class
   (e.g. "QT prolongation", "bleeding", "serotonin syndrome").
2. **Mechanism-derived positives**: every (strong CYP3A4 inhibitor, sensitive
   CYP3A4 substrate) pair is a pharmacokinetic positive whose expected events
   are the substrate's own dose-dependent toxicities.
3. **Reporter-flagged interactions** from FAERS itself: drugs coded
   ``drugcharacterization = 3`` ("interacting"). A pair (suspect, interacting)
   observed on >= ``min_reports`` reports is a *partial positive label*
   (reporter judgement, not adjudicated truth). Negatives cannot be inferred
   from this source; use it for recall-type evaluation only.

Evaluation utilities compute AUROC / average precision / precision-at-k of a
signal score (Omega, log-IOR, posterior mean ...) against any of these sets,
with pair-level bootstrap CIs.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

Pair = FrozenSet[str]


def make_pair(a: str, b: str) -> Pair:
    return frozenset({str(a).upper().strip(), str(b).upper().strip()})


def load_pair_csv(path: str, col_a: str = "drug_a", col_b: str = "drug_b", event_col: Optional[str] = None) -> Dict[Pair, Optional[str]]:
    """Load an unordered pair list from CSV -> ``{pair: expected_event_class}``."""
    df = pd.read_csv(path)
    out: Dict[Pair, Optional[str]] = {}
    for _, r in df.iterrows():
        out[make_pair(r[col_a], r[col_b])] = (str(r[event_col]) if event_col and pd.notna(r.get(event_col)) else None)
    return out


def mechanism_pairs(inhibitors: Iterable[str], substrates: Iterable[str], label: str) -> Dict[Pair, str]:
    """Cross product of a perpetrator list and a victim list (e.g. CYP3A4)."""
    out: Dict[Pair, str] = {}
    for i in inhibitors:
        for s in substrates:
            if i.upper() != s.upper():
                out[make_pair(i, s)] = label
    return out


def interacting_role_pairs(reports: Sequence[Dict], min_reports: int = 3) -> Dict[Pair, int]:
    """Pairs (suspect, interacting) explicitly flagged by reporters.

    Returns ``{pair: n_reports}`` for pairs seen on at least ``min_reports``
    reports. Reports with an interacting drug but no suspect drug pair it with
    every other drug on the report.
    """
    counts: Counter = Counter()
    for r in reports:
        inter = [d for d in r.get("interacting", [])]
        if not inter:
            continue
        partners = r.get("suspect", []) or [d for d in r.get("drugs", []) if d not in inter]
        for i in inter:
            for p in partners:
                if p != i:
                    counts[make_pair(i, p)] += 1
    return {p: n for p, n in counts.items() if n >= min_reports}


def label_pairs(scores: pd.DataFrame, positives: Iterable[Pair], negatives: Optional[Iterable[Pair]] = None) -> pd.DataFrame:
    """Attach a 0/1 ``label`` column (NaN for unlabeled pairs) to a score frame."""
    pos = set(positives)
    neg = set(negatives) if negatives is not None else None
    out = scores.copy()
    pairs = [make_pair(a, b) for a, b in zip(out["drug_a"], out["drug_b"])]
    lab: List[float] = []
    for p in pairs:
        if p in pos:
            lab.append(1.0)
        elif neg is None or p in neg:
            lab.append(0.0)
        else:
            lab.append(np.nan)
    out["label"] = lab
    return out


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC (Mann-Whitney), ties count 0.5; NaN if one class only."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-s, kind="stable")
    y = y[order]
    if y.sum() == 0:
        return float("nan")
    hits = np.cumsum(y)
    prec = hits / (np.arange(len(y)) + 1)
    return float(np.sum(prec * y) / y.sum())


def precision_at_k(scores: Sequence[float], labels: Sequence[int], k: int) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-s, kind="stable")[:k]
    return float(y[order].mean()) if len(order) else float("nan")


def evaluate_scores(df: pd.DataFrame, score_cols: Sequence[str], label_col: str = "label", n_boot: int = 500, seed: int = 0, k: int = 50) -> pd.DataFrame:
    """AUROC / AP / P@k per score column with pair-level bootstrap 95% CIs."""
    d = df.dropna(subset=[label_col]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for col in score_cols:
        s = d[col].fillna(d[col].min()).values
        y = d[label_col].astype(int).values
        auc, ap, pk = roc_auc(s, y), average_precision(s, y), precision_at_k(s, y, k)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(d), len(d))
            boots.append(roc_auc(s[idx], y[idx]))
        boots = np.array([b for b in boots if not np.isnan(b)])
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
        rows.append({"score": col, "auroc": auc, "auroc_lo": lo, "auroc_hi": hi, "ap": ap, f"p_at_{k}": pk, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())})
    return pd.DataFrame(rows)


#: A small, well-established CYP3A4 panel to seed the mechanism reference set
#: (FDA Drug Development and Drug Interactions tables; verify against the
#: current FDA page before use - lists change).
CYP3A4_STRONG_INHIBITORS: Tuple[str, ...] = ("CLARITHROMYCIN", "ITRACONAZOLE", "KETOCONAZOLE", "RITONAVIR", "COBICISTAT", "POSACONAZOLE", "VORICONAZOLE")
CYP3A4_SENSITIVE_SUBSTRATES: Tuple[str, ...] = ("SIMVASTATIN", "LOVASTATIN", "MIDAZOLAM", "TACROLIMUS", "SIROLIMUS", "EVEROLIMUS", "APIXABAN", "RIVAROXABAN", "TICAGRELOR", "IBRUTINIB")
#: Expected victim toxicities to pair with the mechanism positives.
CYP3A4_EXPECTED_EVENTS: Dict[str, Tuple[str, ...]] = {
    "SIMVASTATIN": ("RHABDOMYOLYSIS", "MYOPATHY", "MYALGIA", "BLOOD CREATINE PHOSPHOKINASE INCREASED"),
    "LOVASTATIN": ("RHABDOMYOLYSIS", "MYOPATHY", "MYALGIA"),
    "MIDAZOLAM": ("SEDATION", "RESPIRATORY DEPRESSION", "SOMNOLENCE"),
    "TACROLIMUS": ("NEPHROTOXICITY", "BLOOD CREATININE INCREASED", "TREMOR", "DRUG LEVEL INCREASED"),
    "APIXABAN": ("HAEMORRHAGE", "GASTROINTESTINAL HAEMORRHAGE", "EPISTAXIS"),
    "RIVAROXABAN": ("HAEMORRHAGE", "GASTROINTESTINAL HAEMORRHAGE", "EPISTAXIS"),
}
