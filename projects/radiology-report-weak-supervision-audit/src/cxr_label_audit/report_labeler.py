"""A small, transparent CheXpert-style report labeler and report-structure features.

Label semantics follow the CheXpert convention: ``1`` positive, ``0`` negated, ``-1`` uncertain,
``None`` not mentioned. This is deliberately simple (regex mention + cue-scoped negation/uncertainty)
so that its behaviour is fully inspectable; production runs additionally use the official CheXpert
labeler, NegBio and CheXbert, and the audit compares all of them against image-level expert labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FINDINGS: dict[str, list[str]] = {
    "No Finding": [],
    "Enlarged Cardiomediastinum": [r"enlarged cardiomediastin\w*", r"widened mediastin\w*", r"mediastinal (widening|enlargement)",
                                   r"cardiomediastinal (silhouette|contour)s? (is|are) (enlarged|widened)"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged (cardiac|heart) (silhouette|size)", r"heart (size )?is enlarged",
                     r"cardiac (silhouette|enlargement) (is )?(mildly |moderately |severely )?enlarged", r"heart is (mildly |moderately )?enlarged"],
    "Lung Opacity": [r"opacit(y|ies)", r"opacification", r"infiltrate", r"airspace disease", r"air-space disease", r"haziness", r"hazy"],
    "Lung Lesion": [r"nodule", r"mass\b", r"lesion", r"nodular (density|opacity)"],
    "Edema": [r"edema", r"vascular congestion", r"pulmonary congestion", r"fluid overload", r"kerley"],
    "Consolidation": [r"consolidat\w*"],
    "Pneumonia": [r"pneumonia", r"infection\b", r"infectious process"],
    "Atelectasis": [r"atelecta\w*", r"collapse"],
    "Pneumothorax": [r"pneumothora\w*"],
    "Pleural Effusion": [r"effusion", r"pleural fluid"],
    "Pleural Other": [r"pleural thickening", r"fibrothorax", r"pleural scarring", r"pleural plaque"],
    "Fracture": [r"fracture"],
    "Support Devices": [r"\btube\b", r"catheter", r"pacemaker", r"\bline\b", r"\bpicc\b", r"\bport-?a-?cath", r"endotracheal", r"\bng tube", r"\bicd\b",
                        r"sternotomy wires", r"drain\b", r"stent"],
}

NEGATION_CUES = [r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\bfree of\b", r"\bclear of\b",
                 r"\bresolved\b", r"\bresolution of\b", r"\babsence of\b", r"\bno evidence of\b", r"\bno longer\b",
                 r"\bruled out\b", r"\bexcluded\b", r"\bclear\b"]
UNCERTAINTY_CUES = [r"\bmay\b", r"\bmight\b", r"\bpossibl\w*", r"\bquestionable\b", r"\bcannot (be )?exclude\w*",
                    r"\bcan ?not (be )?rule\w* out", r"\bsuggest\w*", r"\bsuspicious\b", r"\bconcern\w* for\b",
                    r"\bversus\b", r"\bvs\.?\b", r"\blikely\b", r"\bprobabl\w*", r"\bequivocal\b", r"\bdifferential\b",
                    r"\bcould (be|represent)\b", r"\bworrisome\b", r"\bborderline\b"]
NORMAL_TEMPLATES = [r"no acute cardiopulmonary (process|abnormality|disease)", r"no acute intrathoracic process",
                    r"normal chest (radiograph|x-?ray)", r"lungs are clear", r"no active disease", r"unremarkable"]
COMPARISON_CUES = [r"\bunchanged\b", r"\bstable\b", r"\bagain\b", r"\bpersist\w*", r"\bsimilar to\b", r"\bprior\b",
                   r"\bprevious\b", r"\binterval\b", r"\bcompared? (to|with)\b", r"\bno (significant )?change\b", r"\bre-?demonstrat\w*"]
LIMITED_CUES = [r"\bportable\b", r"\blimited\b", r"\blow lung volumes?\b", r"\bsuboptimal\b", r"\brotat\w*", r"\bunderpenetrat\w*", r"\bmotion\b"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=[A-Z])")


@dataclass
class ReportFeatures:
    n_words: int
    n_sentences: int
    has_findings_section: bool
    has_impression_section: bool
    templated_normal: bool
    comparison_language: bool
    limited_study: bool
    hedging_density: float   # uncertainty cues per 100 words


def split_sections(report: str) -> dict[str, str]:
    """Return ``{'findings': ..., 'impression': ..., 'other': ...}`` (case-insensitive headers)."""
    text = report.replace("\r", "")
    sec = {"findings": "", "impression": "", "other": ""}
    pattern = re.compile(r"(FINDINGS?|IMPRESSION|CONCLUSION)\s*:", re.I)
    parts = pattern.split(text)
    if len(parts) == 1:
        sec["other"] = text
        return sec
    sec["other"] = parts[0]
    for head, body in zip(parts[1::2], parts[2::2]):
        key = "impression" if head.lower().startswith(("impression", "conclusion")) else "findings"
        sec[key] += body + "\n"
    return sec


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _has(patterns: list[str], s: str) -> bool:
    return any(re.search(p, s, re.I) for p in patterns)


def label_sentence(s: str) -> dict[str, int]:
    """Labels for one sentence: positive / negated / uncertain for each finding mentioned."""
    out: dict[str, int] = {}
    low = s.lower()
    neg = _has(NEGATION_CUES, low)
    unc = _has(UNCERTAINTY_CUES, low)
    for finding, pats in FINDINGS.items():
        if not pats or not _has(pats, low):
            continue
        # scope: negation cue must appear before the mention in the sentence
        m = min(re.search(p, low).start() for p in pats if re.search(p, low))
        neg_before = any(re.search(c, low[:m]) for c in NEGATION_CUES)
        if unc and not neg_before:
            out[finding] = -1
        elif neg_before or (neg and m < 40):
            out[finding] = 0
        else:
            out[finding] = 1
    return out


def label_report(report: str, use_sections: tuple[str, ...] = ("findings", "impression", "other")) -> dict[str, int | None]:
    """Report-level labels with CheXpert-style aggregation: positive > uncertain > negative > unmentioned."""
    sec = split_sections(report)
    labels: dict[str, int | None] = {f: None for f in FINDINGS}
    rank = {1: 3, -1: 2, 0: 1}
    for key in use_sections:
        for s in sentences(sec[key]):
            for f, v in label_sentence(s).items():
                cur = labels[f]
                if cur is None or rank[v] > rank[cur]:
                    labels[f] = v
    any_abnormal = any(v in (1, -1) for f, v in labels.items() if f not in ("No Finding", "Support Devices"))
    labels["No Finding"] = None if any_abnormal else 1
    return labels


def report_features(report: str) -> ReportFeatures:
    sec = split_sections(report)
    low = report.lower()
    words = re.findall(r"[a-z0-9]+", low)
    n_words = len(words)
    n_unc = sum(len(re.findall(c, low)) for c in UNCERTAINTY_CUES)
    return ReportFeatures(
        n_words=n_words,
        n_sentences=len(sentences(report)),
        has_findings_section=bool(sec["findings"].strip()),
        has_impression_section=bool(sec["impression"].strip()),
        templated_normal=_has(NORMAL_TEMPLATES, low),
        comparison_language=_has(COMPARISON_CUES, low),
        limited_study=_has(LIMITED_CUES, low),
        hedging_density=100.0 * n_unc / max(n_words, 1),
    )


def labels_to_row(labels: dict[str, int | None], unmentioned_as: int | None = None) -> dict[str, float]:
    """Flatten labels for a DataFrame row; ``unmentioned_as`` maps None to 0 (training convention) or NaN."""
    return {f: (float("nan") if v is None and unmentioned_as is None else float(unmentioned_as if v is None else v))
            for f, v in labels.items()}
