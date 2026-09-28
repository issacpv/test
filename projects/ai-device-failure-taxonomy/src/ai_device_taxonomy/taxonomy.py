"""AI-specific failure-mode taxonomy for MAUDE narratives.

The rules give *weak* labels with evidence spans; the annotation guideline (``data/annotation_guideline.md``)
and human double annotation are the ground truth.  A TF-IDF classifier trained on the weak labels smooths over
vocabulary the rules miss and is evaluated against the annotated set.  Categories are multi-label.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

TAXONOMY: Dict[str, Dict[str, object]] = {
    "incorrect_output": {
        "description": "The algorithm produced a wrong, missed or spurious result (false positive/negative, wrong measurement, misclassification, wrong laterality/segmentation).",
        "patterns": [r"false[\s-]*(positive|negative)", r"\bmiss(ed|ing)\s+(the\s+)?(lesion|finding|nodule|bleed|fracture|detection|stroke|pe\b)", r"incorrect(ly)?\s+(result|measurement|calculation|identif|classif|segment|flag|triage)", r"wrong\s+(result|measurement|value|laterality|patient|side|score)", r"(inaccurate|erroneous|spurious)\s+(result|reading|measurement|value|output|alert)", r"did not (detect|identify|flag|alert)", r"over[\s-]*estimat|under[\s-]*estimat", r"misclassif|misidentif|mislabel|mis-?read"],
    },
    "no_output_or_delay": {
        "description": "No result, delayed result, crash, freeze, timeout, failure to process or display.",
        "patterns": [r"(no|failed to (produce|generate|provide|display|return))\s+(output|result|report|analysis|alert)", r"\b(crash(ed|es)?|froze|frozen|freez(e|ing)|hang(s|ing)?|hung|timed?\s*out|time-?out|unresponsive|locked up)\b", r"delay(ed)?\s+(in\s+)?(the\s+)?(result|output|processing|analysis|notification|alert|report)", r"(did|could|would) not (load|process|open|complete|run|start)", r"error message|software error|application error|system error"],
    },
    "input_data_quality": {
        "description": "Problem attributable to the input data: image/signal quality, artefact, wrong protocol or acquisition, incompatible or corrupted input.",
        "patterns": [r"(poor|low|inadequate|suboptimal)\s+(image|signal|scan|data)\s+quality", r"\bartifact|artefact\b", r"motion\s+(artifact|artefact|blur)", r"(wrong|incorrect|unsupported|incompatible)\s+(protocol|series|sequence|modality|format|input|acquisition|view)", r"corrupt(ed)?\s+(image|data|file)", r"(missing|incomplete)\s+(series|images|data|slices)"],
    },
    "integration_interface": {
        "description": "Interfaces and integration: PACS/RIS/EHR, DICOM/HL7, worklists, network transfer, wrong-patient association, routing.",
        "patterns": [r"\b(pacs|ris|emr|ehr|his|dicom|hl7|fhir|worklist|modality worklist)\b", r"(network|connectivity|connection|transfer|transmission|routing|interface)\s+(issue|problem|failure|error|loss|lost)", r"(sent|routed|assigned|associated|attached)\s+to\s+(the\s+)?wrong\s+patient", r"(mismatch|mis-?match)(ed)?\s+(patient|study|accession|record)", r"integration\s+(issue|problem|error|failure)", r"server\s+(issue|problem|error|failure|down)", r"cloud\s+(outage|issue|problem|error)"],
    },
    "software_update_version": {
        "description": "Failure associated with a software update, upgrade, version change, patch, configuration change or release regression.",
        "patterns": [r"(software|firmware|system|application)\s+(update|upgrade|version|release|patch)", r"(after|following|since)\s+(the\s+)?(update|upgrade|installation|patch|new version|release)", r"\bversion\s+\d", r"regression\s+(bug|issue|defect)", r"configuration\s+(change|error|issue)"],
    },
    "hardware_of_device": {
        "description": "Hardware component of the AI-enabled device: sensor, probe, battery, cable, display, pump, motor, detector.",
        "patterns": [r"\b(battery|batteries|cable|connector|sensor|probe|transducer|detector|display|screen|monitor|motor|pump|valve|lead|electrode|wire|housing|casing|hardware)\b\s*(failure|fail(ed|s)?|broke(n)?|malfunction|defect|damage|cracked|loose|disconnect)", r"(failure|failed|broken|malfunction|defect|damaged)\s+(battery|cable|connector|sensor|probe|transducer|detector|display|screen|motor|pump|hardware)", r"overheat|smoke|spark|burn(ed|ing)?\b"],
    },
    "user_workflow_human_factors": {
        "description": "Use-related: user error, misinterpretation of output, alert fatigue, over-reliance, training, workflow bypass.",
        "patterns": [r"\buser\s+error\b", r"(operator|user|clinician|technologist|nurse|physician)\s+(did not|failed to|forgot|misinterpret|misread|ignored|overrode|dismissed)", r"(over|excessive)[\s-]*(reliance|trust)", r"alert\s+fatigue", r"(insufficient|lack of|inadequate)\s+training", r"(misinterpret|misread|misunderst)", r"not\s+(reviewed|read|confirmed)\s+by\s+(a\s+)?(radiologist|physician|clinician)"],
    },
    "cybersecurity_access": {
        "description": "Security, authentication, unauthorised access, malware, ransomware.",
        "patterns": [r"\b(cyber|security|ransomware|malware|virus|unauthori[sz]ed|breach|hack(ed|ing)?|credential|password|login|log-in|authentication)\b"],
    },
}

OUTCOME_PATTERNS: Dict[str, str] = {
    "harm_death": r"\b(death|died|expired|fatal)\b",
    "harm_injury": r"\b(injur|harm|adverse (reaction|outcome)|complication)\w*",
    "harm_delay_in_care": r"delay(ed)?\s+(in\s+)?(treatment|care|diagnosis|intervention|therapy|surgery)",
    "harm_unnecessary_procedure": r"(unnecessary|additional)\s+(procedure|biopsy|surgery|imaging|scan|test)",
}

CATEGORIES: List[str] = list(TAXONOMY)
_COMPILED = {c: [re.compile(p, re.I) for p in spec["patterns"]] for c, spec in TAXONOMY.items()}  # type: ignore[index]
_OUTCOMES = {k: re.compile(v, re.I) for k, v in OUTCOME_PATTERNS.items()}


def label_narrative(text: object, max_evidence: int = 3) -> Dict[str, List[str]]:
    """Weak multi-label assignment with up to ``max_evidence`` matched spans per category (empty list = absent)."""
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    out: Dict[str, List[str]] = {}
    for cat, pats in _COMPILED.items():
        spans: List[str] = []
        for p in pats:
            for m in p.finditer(t):
                spans.append(m.group(0))
                if len(spans) >= max_evidence:
                    break
            if len(spans) >= max_evidence:
                break
        out[cat] = spans
    return out


def label_frame(df: pd.DataFrame, text_col: str = "text_event", extra_text_col: Optional[str] = "text_manufacturer") -> pd.DataFrame:
    """0/1 columns per category (``tax_<category>``), outcome flags (``out_<name>``) and ``n_categories``.

    Manufacturer narratives are appended when ``extra_text_col`` exists (templated manufacturer text can inflate
    rule hits; compare with ``extra_text_col=None`` as a sensitivity analysis).
    """
    texts = df[text_col].fillna("").astype(str)
    if extra_text_col and extra_text_col in df:
        texts = texts + " " + df[extra_text_col].fillna("").astype(str)
    lab = pd.DataFrame(index=df.index)
    ev = []
    for i, t in texts.items():
        res = label_narrative(t)
        ev.append({c: "|".join(v) for c, v in res.items() if v})
        for c in CATEGORIES:
            lab.loc[i, f"tax_{c}"] = int(bool(res[c]))
        for name, pat in _OUTCOMES.items():
            lab.loc[i, f"out_{name}"] = int(bool(pat.search(t)))
    lab = lab.astype(int)
    lab["n_categories"] = lab[[f"tax_{c}" for c in CATEGORIES]].sum(axis=1)
    lab["algorithm_implicated"] = ((lab["tax_incorrect_output"] == 1) | (lab["tax_no_output_or_delay"] == 1)).astype(int)
    lab["evidence"] = ev
    return lab


def strip_brand_tokens(text: str, brand: Optional[str], manufacturer: Optional[str]) -> str:
    """Remove brand/manufacturer tokens from a narrative so a classifier cannot learn device identity."""
    t = text or ""
    for name in (brand, manufacturer):
        if not name:
            continue
        for tok in re.findall(r"[A-Za-z0-9]{3,}", str(name)):
            t = re.sub(rf"\b{re.escape(tok)}\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def train_weak_classifier(texts: Sequence[str], labels: pd.DataFrame, categories: Sequence[str] = CATEGORIES, min_positive: int = 5) -> Pipeline:
    """TF-IDF (1-2 grams) + one-vs-rest logistic regression on weak labels ``tax_<category>``.

    Categories with fewer than ``min_positive`` positives are dropped from the target (they cannot be learned);
    the fitted pipeline stores ``categories_`` for prediction.
    """
    Y = labels[[f"tax_{c}" for c in categories]].to_numpy()
    keep = Y.sum(axis=0) >= min_positive
    cats = [c for c, k in zip(categories, keep) if k]
    if not cats:
        raise ValueError("no category has enough positives")
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"))
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), clf)
    pipe.fit(list(texts), Y[:, keep])
    pipe.categories_ = cats  # type: ignore[attr-defined]
    return pipe


def predict_categories(pipe: Pipeline, texts: Sequence[str], threshold: float = 0.5) -> pd.DataFrame:
    """Predicted probabilities and binary labels per learned category."""
    P = pipe.predict_proba(list(texts))
    cats = pipe.categories_  # type: ignore[attr-defined]
    if P.ndim == 1 or (len(cats) == 1 and P.shape[1] == 2):
        P = P[:, [1]]
    out = pd.DataFrame(P, columns=[f"p_{c}" for c in cats])
    for c in cats:
        out[f"pred_{c}"] = (out[f"p_{c}"] >= threshold).astype(int)
    return out


def agreement(a: Sequence[int], b: Sequence[int]) -> Dict[str, float]:
    """Cohen's kappa and raw agreement between two binary annotations."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    return {"kappa": float(cohen_kappa_score(a, b)) if len(np.unique(np.concatenate([a, b]))) > 1 else 1.0, "agreement": float(np.mean(a == b)), "n": int(a.size)}
