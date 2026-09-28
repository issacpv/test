"""Chief-complaint text features.

All processing is local.  No function in this module performs network calls;
transformer checkpoints must already be on disk (``local_files_only=True``),
in line with PhysioNet's rule that MIMIC text may not be sent to external
services.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

ABBREVIATIONS: dict[str, str] = {
    "sob": "shortness of breath", "doe": "dyspnea on exertion", "cp": "chest pain", "abd": "abdominal",
    "abd pain": "abdominal pain", "n/v": "nausea vomiting", "n/v/d": "nausea vomiting diarrhea",
    "loc": "loss of consciousness", "ams": "altered mental status", "etoh": "alcohol", "sz": "seizure",
    "h/a": "headache", "ha": "headache", "uti": "urinary tract infection", "mvc": "motor vehicle collision",
    "mva": "motor vehicle accident", "s/p": "status post", "fx": "fracture", "lac": "laceration",
    "bp": "blood pressure", "htn": "hypertension", "dm": "diabetes", "gi": "gastrointestinal",
    "gib": "gastrointestinal bleed", "brbpr": "bright red blood per rectum", "le": "lower extremity",
    "ue": "upper extremity", "rle": "right lower extremity", "lle": "left lower extremity",
    "rue": "right upper extremity", "lue": "left upper extremity", "r": "right", "l": "left",
    "si": "suicidal ideation", "hi": "homicidal ideation", "od": "overdose", "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis", "afib": "atrial fibrillation", "chf": "heart failure", "copd": "copd",
    "ili": "influenza like illness", "uri": "upper respiratory infection", "tx": "treatment", "eval": "evaluation",
}
_SPLIT = re.compile(r"\s*[,;]\s*|\s+/\s+|\s+and\s+|\s+&\s+")  # a slash inside a token (n/v) is kept
_NONWORD = re.compile(r"[^a-z0-9/ ]+")
_WS = re.compile(r"\s+")


def normalize_complaint(text: object) -> str:
    """Lower-case, strip punctuation, expand common ED abbreviations (multi-word first)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower().strip()
    s = s.replace("w/", " with ").replace("s/p", " status post ")
    for k in sorted(ABBREVIATIONS, key=len, reverse=True):
        if " " in k or "/" in k:
            s = re.sub(rf"(?<![a-z]){re.escape(k)}(?![a-z])", ABBREVIATIONS[k], s)
    s = _NONWORD.sub(" ", s)
    tokens = [ABBREVIATIONS.get(tok, tok) for tok in s.split()]
    return _WS.sub(" ", " ".join(tokens)).strip()


def split_complaints(text: object) -> list[str]:
    """Split multi-complaint strings ('chest pain, sob / dizziness') into normalised parts."""
    raw = "" if text is None else str(text)
    parts = [normalize_complaint(p) for p in _SPLIT.split(raw) if p and p.strip()]
    return [p for p in parts if p]


@dataclass
class TfidfComplaintFeatures:
    """Word (1-2 gram) + character (3-5 gram) TF-IDF on normalised chief complaints.

    Fit on the *training era only* (see ``ed_triage.validation``) to avoid vocabulary leakage.
    """

    max_word_features: int = 5000
    max_char_features: int = 10000
    min_df: int = 5
    word_vec: object = field(default=None, repr=False)
    char_vec: object = field(default=None, repr=False)

    def fit(self, texts: Iterable[object]) -> "TfidfComplaintFeatures":
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [normalize_complaint(t) for t in texts]
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_word_features, sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=self.min_df, max_features=self.max_char_features, sublinear_tf=True)
        self.word_vec.fit(docs)
        self.char_vec.fit(docs)
        return self

    def transform(self, texts: Iterable[object]):
        """Sparse matrix (n, n_word + n_char)."""
        from scipy import sparse

        if self.word_vec is None or self.char_vec is None:
            raise RuntimeError("call fit() first")
        docs = [normalize_complaint(t) for t in texts]
        return sparse.hstack([self.word_vec.transform(docs), self.char_vec.transform(docs)]).tocsr()

    def vocabulary_size(self) -> int:
        return len(self.word_vec.vocabulary_) + len(self.char_vec.vocabulary_)  # type: ignore[attr-defined]


def complaint_category(text: object, keywords: dict[str, Sequence[str]] | None = None) -> str:
    """Coarse chief-complaint category used as an adjustment covariate in the mis-triage models."""
    keywords = keywords or {
        "cardiac": ("chest pain", "palpitation", "syncope", "atrial fibrillation"),
        "respiratory": ("shortness of breath", "dyspnea", "cough", "asthma", "copd", "wheez"),
        "neuro": ("headache", "seizure", "stroke", "weakness", "numbness", "altered mental status", "dizz"),
        "abdominal": ("abdominal pain", "nausea", "vomiting", "diarrhea", "gastrointestinal bleed"),
        "trauma": ("fall", "motor vehicle", "laceration", "fracture", "injury", "assault", "trauma"),
        "infection": ("fever", "sepsis", "urinary tract infection", "cellulitis", "influenza"),
        "psych": ("suicidal", "homicidal", "anxiety", "depress", "psych", "overdose", "alcohol"),
        "pain_msk": ("back pain", "neck pain", "knee", "shoulder", "hip pain", "joint"),
    }
    s = normalize_complaint(text)
    for cat, keys in keywords.items():
        if any(k in s for k in keys):
            return cat
    return "other"


def local_transformer_embeddings(texts: Sequence[object], model_dir: str, batch_size: int = 64, max_length: int = 32,
                                 device: str | None = None) -> np.ndarray:
    """Mean-pooled [CLS]-free embeddings from a *local* HF checkpoint (e.g. Bio_ClinicalBERT).

    ``model_dir`` must be a directory on disk; ``local_files_only=True`` guarantees no download.
    Returns (n, hidden) float32.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install torch transformers (local inference only)") from exc
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    docs = [normalize_complaint(t) or "[EMPTY]" for t in texts]
    out = []
    with torch.no_grad():
        for s in range(0, len(docs), batch_size):
            enc = tok(docs[s:s + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.vstack(out)
