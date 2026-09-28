"""Label-based expectedness for biosimilar vs originator reports.

A biosimilar's prescribing information is, by regulation, essentially a copy
of the reference product's label (with the biosimilar's own immunogenicity
and device data). That makes labels a *control*: if the two products have the
same labelled adverse reactions, differences in the fraction of reports that
carry *unlabelled* events cannot be explained by "expectedness" and point to
product-, device- or reporter-related effects.

Functions here fetch SPL sections through openFDA ``drug/label``
(``openfda.brand_name`` search), normalise the free text, match FAERS PTs
against it with a small synonym table, and measure label-to-label similarity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .openfda import OpenFDAClient

LABEL_SECTIONS: Sequence[str] = ("boxed_warning", "warnings_and_cautions", "warnings", "adverse_reactions", "adverse_reactions_table")

#: FAERS PT -> alternative phrasings found in label prose.
SYNONYMS: Dict[str, Sequence[str]] = {
    "HYPERSENSITIVITY": ("hypersensitivity", "allergic reaction"),
    "ANAPHYLACTIC REACTION": ("anaphylaxis", "anaphylactic"),
    "INFUSION RELATED REACTION": ("infusion reaction", "infusion-related reaction"),
    "INJECTION SITE REACTION": ("injection site reaction", "injection-site reaction"),
    "INJECTION SITE PAIN": ("injection site pain",),
    "INJECTION SITE ERYTHEMA": ("injection site erythema", "injection site redness"),
    "UPPER RESPIRATORY TRACT INFECTION": ("upper respiratory tract infection", "upper respiratory infection"),
    "PNEUMONIA": ("pneumonia",),
    "SEPSIS": ("sepsis",),
    "TUBERCULOSIS": ("tuberculosis",),
    "NEUTROPENIA": ("neutropenia",),
    "FEBRILE NEUTROPENIA": ("febrile neutropenia",),
    "BONE PAIN": ("bone pain",),
    "SPLENIC RUPTURE": ("splenic rupture",),
    "PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY": ("progressive multifocal leukoencephalopathy", "pml"),
    "HEPATITIS B REACTIVATION": ("hepatitis b virus reactivation", "hbv reactivation"),
    "DRUG INEFFECTIVE": ("lack of efficacy", "loss of response", "loss of efficacy"),
    "DEVICE MALFUNCTION": ("device malfunction", "pen malfunction"),
    "MALIGNANCY": ("malignancy", "malignancies", "lymphoma"),
    "HEART FAILURE": ("heart failure", "cardiac failure"),
    "OSTEONECROSIS OF JAW": ("osteonecrosis of the jaw",),
    "HYPOCALCAEMIA": ("hypocalcemia", "hypocalcaemia"),
    "HYPOGLYCAEMIA": ("hypoglycemia", "hypoglycaemia"),
    "ENDOPHTHALMITIS": ("endophthalmitis",),
    "INTRAOCULAR PRESSURE INCREASED": ("increase in intraocular pressure", "intraocular pressure"),
}

_WS = re.compile(r"\s+")


def normalise_text(text: Any) -> str:
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text)
    t = str(text or "").lower()
    t = re.sub(r"[^a-z0-9%\-\s]", " ", t)
    return _WS.sub(" ", t).strip()


def fetch_label(client: OpenFDAClient, brand: str, sections: Sequence[str] = LABEL_SECTIONS) -> Optional[Dict[str, Any]]:
    """Latest SPL for ``brand`` via ``drug/label``; returns selected sections."""
    search = f"openfda.brand_name:{client.quote(brand)}"
    recs = list(client.iter_records("drug/label", search, limit=10, max_records=10, sort="effective_time:desc"))
    if not recs:
        return None
    rec = recs[0]
    out = {"brand": brand, "set_id": rec.get("set_id"), "effective_time": rec.get("effective_time"), "version": rec.get("version")}
    for s in sections:
        out[s] = normalise_text(rec.get(s, ""))
    return out


def label_text(label: Dict[str, Any], sections: Sequence[str] = LABEL_SECTIONS) -> str:
    return " ".join(label.get(s, "") for s in sections if label.get(s))


def pt_in_label(pt: str, text: str) -> bool:
    """True if the PT (or a synonym) occurs in the normalised label text."""
    cands = [pt.lower()] + [s.lower() for s in SYNONYMS.get(pt.upper(), ())]
    # MedDRA uses British spellings; add American variants
    extra = []
    for c in cands:
        extra.append(c.replace("aemia", "emia").replace("oedema", "edema").replace("haem", "hem").replace("diarrhoea", "diarrhea"))
    for c in set(cands + extra):
        if re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", text):
            return True
    return False


def labelled_flags(pts: Iterable[str], text: str) -> Dict[str, bool]:
    return {pt: pt_in_label(pt, text) for pt in pts}


def unlabelled_fraction(reports: Sequence[Dict], text: str) -> Dict[str, float]:
    """Fraction of reports whose PTs are all unlabelled / any unlabelled."""
    cache: Dict[str, bool] = {}
    n = len(reports)
    any_unl = all_unl = 0
    for r in reports:
        pts = list(r.get("reactions", []))
        if not pts:
            continue
        flags = []
        for pt in pts:
            if pt not in cache:
                cache[pt] = pt_in_label(pt, text)
            flags.append(cache[pt])
        if not all(flags):
            any_unl += 1
        if not any(flags):
            all_unl += 1
    return {"n": n, "frac_any_unlabelled": any_unl / n if n else float("nan"), "frac_all_unlabelled": all_unl / n if n else float("nan")}


def _shingles(text: str, k: int = 5) -> Set[str]:
    toks = text.split()
    return {" ".join(toks[i : i + k]) for i in range(max(len(toks) - k + 1, 0))}


def label_similarity(text_a: str, text_b: str, k: int = 5) -> float:
    """Jaccard similarity of k-word shingles between two label texts.

    Values near 1 confirm the "biosimilar label == originator label"
    assumption; lower values flag sections that differ (device instructions,
    immunogenicity paragraphs, indication carve-outs).
    """
    a, b = _shingles(normalise_text(text_a), k), _shingles(normalise_text(text_b), k)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
