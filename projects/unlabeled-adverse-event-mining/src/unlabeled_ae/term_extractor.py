"""Adverse-reaction term extraction from label text.

The extractor is dictionary based: a MedDRA-like map from Preferred Term (PT)
to lower-level synonyms (LLT-like strings) compiled into word-boundary regular
expressions, matched longest-first, with a small negation window ("no cases
of", "not associated with" ...). The bundled ``SEED_TERMS`` dictionary covers
~130 common / regulatory-relevant PTs and is meant for development and tests;
for research use load a licensed MedDRA export (``MedDRADictionary.from_meddra_ascii``
reads ``pt.asc`` + ``llt.asc``) or a CSV of ``pt,synonym`` pairs (e.g. derived
from SIDER/UMLS under their licences).

An optional scispaCy hook (``scispacy_candidates``) uses the ``en_ner_bc5cdr_md``
DISEASE entity recogniser to propose candidate strings, which are then mapped
to dictionary PTs; it raises ``ImportError`` when spaCy/scispaCy are absent.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

# PT -> synonyms (US and UK spellings, lay terms). Keys follow MedDRA PT spelling.
SEED_TERMS: Dict[str, List[str]] = {
    "Headache": ["headache", "headaches", "cephalalgia"],
    "Nausea": ["nausea"],
    "Vomiting": ["vomiting", "emesis"],
    "Diarrhoea": ["diarrhea", "diarrhoea", "loose stools"],
    "Constipation": ["constipation"],
    "Abdominal pain": ["abdominal pain", "stomach pain", "abdominal discomfort"],
    "Dyspepsia": ["dyspepsia", "indigestion", "heartburn"],
    "Gastrointestinal haemorrhage": ["gastrointestinal bleeding", "gastrointestinal hemorrhage", "gastrointestinal haemorrhage", "gi bleeding", "gi bleed"],
    "Pancreatitis": ["pancreatitis"],
    "Gastroparesis": ["gastroparesis", "delayed gastric emptying"],
    "Ileus": ["ileus", "intestinal obstruction", "bowel obstruction"],
    "Cholelithiasis": ["cholelithiasis", "gallstones", "gallbladder disease"],
    "Hepatotoxicity": ["hepatotoxicity", "liver injury", "drug-induced liver injury", "hepatic injury"],
    "Hepatic failure": ["hepatic failure", "liver failure", "acute liver failure"],
    "Hepatitis": ["hepatitis"],
    "Jaundice": ["jaundice", "icterus"],
    "Cholestasis": ["cholestasis", "cholestatic"],
    "Transaminases increased": ["transaminases increased", "elevated transaminases", "transaminase elevations", "alt increased", "ast increased", "elevated liver enzymes", "liver enzyme elevations"],
    "Rash": ["rash", "rashes", "skin rash"],
    "Pruritus": ["pruritus", "itching", "itch"],
    "Urticaria": ["urticaria", "hives"],
    "Alopecia": ["alopecia", "hair loss"],
    "Photosensitivity reaction": ["photosensitivity", "photosensitivity reaction"],
    "Stevens-Johnson syndrome": ["stevens-johnson syndrome", "stevens johnson syndrome", "sjs"],
    "Toxic epidermal necrolysis": ["toxic epidermal necrolysis", "ten"],
    "Drug reaction with eosinophilia and systemic symptoms": ["drug reaction with eosinophilia and systemic symptoms", "dress syndrome", "dress"],
    "Anaphylactic reaction": ["anaphylaxis", "anaphylactic reaction", "anaphylactic reactions", "anaphylactoid reaction"],
    "Angioedema": ["angioedema", "angioneurotic edema"],
    "Hypersensitivity": ["hypersensitivity", "hypersensitivity reactions", "allergic reaction", "allergic reactions"],
    "Dizziness": ["dizziness", "lightheadedness"],
    "Vertigo": ["vertigo"],
    "Somnolence": ["somnolence", "drowsiness", "sleepiness", "sedation"],
    "Insomnia": ["insomnia", "sleeplessness", "difficulty sleeping"],
    "Fatigue": ["fatigue", "tiredness"],
    "Asthenia": ["asthenia", "weakness"],
    "Tremor": ["tremor", "tremors"],
    "Seizure": ["seizure", "seizures", "convulsion", "convulsions"],
    "Syncope": ["syncope", "fainting"],
    "Paraesthesia": ["paresthesia", "paraesthesia", "tingling"],
    "Peripheral neuropathy": ["peripheral neuropathy", "neuropathy peripheral", "polyneuropathy"],
    "Serotonin syndrome": ["serotonin syndrome"],
    "Neuroleptic malignant syndrome": ["neuroleptic malignant syndrome", "nms"],
    "Tardive dyskinesia": ["tardive dyskinesia"],
    "Extrapyramidal disorder": ["extrapyramidal symptoms", "extrapyramidal disorder", "extrapyramidal reactions"],
    "Progressive multifocal leukoencephalopathy": ["progressive multifocal leukoencephalopathy", "pml"],
    "Depression": ["depression", "depressed mood"],
    "Anxiety": ["anxiety", "nervousness"],
    "Suicidal ideation": ["suicidal ideation", "suicidal thoughts", "suicidal thinking", "suicidality"],
    "Completed suicide": ["completed suicide", "suicide"],
    "Hallucination": ["hallucination", "hallucinations"],
    "Confusional state": ["confusion", "confusional state"],
    "Agitation": ["agitation"],
    "Aggression": ["aggression", "aggressive behavior", "aggressive behaviour"],
    "Somnambulism": ["sleepwalking", "sleep-walking", "somnambulism", "sleep driving"],
    "Abnormal dreams": ["abnormal dreams", "nightmares", "vivid dreams"],
    "Myalgia": ["myalgia", "muscle pain", "muscle aches"],
    "Arthralgia": ["arthralgia", "joint pain"],
    "Back pain": ["back pain"],
    "Rhabdomyolysis": ["rhabdomyolysis"],
    "Myopathy": ["myopathy"],
    "Tendon rupture": ["tendon rupture", "tendon ruptures", "ruptured tendon"],
    "Tendonitis": ["tendonitis", "tendinitis"],
    "Osteonecrosis of jaw": ["osteonecrosis of the jaw", "jaw osteonecrosis", "onj"],
    "Bone fracture": ["fracture", "fractures", "bone fracture"],
    "Myocardial infarction": ["myocardial infarction", "heart attack"],
    "Cardiac failure": ["heart failure", "cardiac failure", "congestive heart failure"],
    "Atrial fibrillation": ["atrial fibrillation"],
    "Electrocardiogram QT prolonged": ["qt prolongation", "qtc prolongation", "prolonged qt", "prolongation of the qt interval", "qt interval prolongation"],
    "Torsade de pointes": ["torsade de pointes", "torsades de pointes", "torsades"],
    "Bradycardia": ["bradycardia"],
    "Tachycardia": ["tachycardia", "palpitations"],
    "Hypertension": ["hypertension", "elevated blood pressure", "increased blood pressure"],
    "Hypotension": ["hypotension", "low blood pressure", "orthostatic hypotension"],
    "Aortic aneurysm": ["aortic aneurysm"],
    "Aortic dissection": ["aortic dissection"],
    "Cerebrovascular accident": ["stroke", "cerebrovascular accident"],
    "Pulmonary embolism": ["pulmonary embolism"],
    "Deep vein thrombosis": ["deep vein thrombosis", "dvt"],
    "Thrombosis": ["thrombosis", "thromboembolic events", "thromboembolism"],
    "Dyspnoea": ["dyspnea", "dyspnoea", "shortness of breath"],
    "Cough": ["cough"],
    "Bronchospasm": ["bronchospasm"],
    "Interstitial lung disease": ["interstitial lung disease", "ild"],
    "Pneumonitis": ["pneumonitis"],
    "Pneumonia": ["pneumonia"],
    "Upper respiratory tract infection": ["upper respiratory tract infection", "upper respiratory infection"],
    "Nasopharyngitis": ["nasopharyngitis", "common cold"],
    "Urinary tract infection": ["urinary tract infection", "uti"],
    "Herpes zoster": ["herpes zoster", "shingles"],
    "Tuberculosis": ["tuberculosis"],
    "Sepsis": ["sepsis", "septicemia"],
    "Infection": ["serious infections", "infections", "infection"],
    "Neutropenia": ["neutropenia"],
    "Agranulocytosis": ["agranulocytosis"],
    "Thrombocytopenia": ["thrombocytopenia", "low platelet count"],
    "Anaemia": ["anemia", "anaemia"],
    "Leukopenia": ["leukopenia", "leucopenia"],
    "Pancytopenia": ["pancytopenia"],
    "Lymphoma": ["lymphoma"],
    "Malignancy": ["malignancy", "malignancies", "cancer"],
    "Thyroid cancer": ["thyroid c-cell tumors", "thyroid cancer", "medullary thyroid carcinoma"],
    "Diabetes mellitus": ["diabetes mellitus", "new-onset diabetes", "new onset diabetes"],
    "Hyperglycaemia": ["hyperglycemia", "hyperglycaemia", "increases in blood glucose", "elevated blood glucose", "increased blood sugar"],
    "Hypoglycaemia": ["hypoglycemia", "hypoglycaemia", "low blood sugar"],
    "Diabetic ketoacidosis": ["diabetic ketoacidosis", "dka"],
    "Ketoacidosis": ["ketoacidosis"],
    "Lactic acidosis": ["lactic acidosis"],
    "Metabolic acidosis": ["metabolic acidosis"],
    "Hyponatraemia": ["hyponatremia", "hyponatraemia"],
    "Hyperkalaemia": ["hyperkalemia", "hyperkalaemia"],
    "Hypokalaemia": ["hypokalemia", "hypokalaemia"],
    "Weight increased": ["weight gain", "weight increased", "increased weight"],
    "Weight decreased": ["weight loss", "weight decreased", "decreased weight"],
    "Decreased appetite": ["decreased appetite", "anorexia", "loss of appetite"],
    "Vitamin B12 deficiency": ["vitamin b12 deficiency"],
    "Acute kidney injury": ["acute kidney injury", "acute renal failure", "renal failure", "kidney failure"],
    "Renal impairment": ["renal impairment", "decreased renal function", "impaired renal function"],
    "Urinary retention": ["urinary retention"],
    "Erectile dysfunction": ["erectile dysfunction", "impotence"],
    "Gynaecomastia": ["gynecomastia", "gynaecomastia"],
    "Fournier's gangrene": ["fournier's gangrene", "fournier gangrene", "necrotizing fasciitis of the perineum"],
    "Oedema peripheral": ["peripheral edema", "peripheral oedema", "edema", "oedema", "swelling of the extremities"],
    "Pyrexia": ["fever", "pyrexia"],
    "Chills": ["chills"],
    "Hyperhidrosis": ["sweating", "hyperhidrosis", "excessive sweating"],
    "Dry mouth": ["dry mouth", "xerostomia"],
    "Dysgeusia": ["dysgeusia", "taste disturbance", "altered taste"],
    "Vision blurred": ["blurred vision", "vision blurred"],
    "Visual impairment": ["visual impairment", "vision loss", "visual disturbances"],
    "Tinnitus": ["tinnitus"],
    "Deafness": ["hearing loss", "deafness"],
    "Memory impairment": ["memory impairment", "memory loss", "forgetfulness", "amnesia"],
    "Cognitive disorder": ["cognitive impairment", "cognitive disorder"],
    "Pancreatic carcinoma": ["pancreatic cancer", "pancreatic carcinoma"],
    "Vasculitis": ["vasculitis"],
    "Lupus-like syndrome": ["lupus-like syndrome", "drug-induced lupus"],
    "Osteoporosis": ["osteoporosis", "bone loss"],
}

_WORD = r"(?<![A-Za-z0-9])"
_WORD_END = r"(?![A-Za-z0-9])"
_NEGATION_CUES = (
    "no cases of", "no reports of", "no evidence of", "not associated with", "did not", "no increase in",
    "without", "absence of", "no ", "not ", "neither", "nor ", "unrelated to", "rather than",
)


@dataclass(frozen=True)
class Mention:
    pt: str
    matched: str
    start: int
    end: int
    negated: bool


class MedDRADictionary:
    """PT -> synonym map with compiled longest-first matching."""

    def __init__(self, terms: Dict[str, Iterable[str]]):
        self.terms: Dict[str, List[str]] = {}
        for pt, syns in terms.items():
            s = {pt.lower()} | {x.lower().strip() for x in syns if x and x.strip()}
            self.terms[pt] = sorted(s, key=len, reverse=True)
        self._syn_to_pt: Dict[str, str] = {}
        for pt, syns in self.terms.items():
            for s in syns:
                self._syn_to_pt.setdefault(s, pt)
        self._compile()

    def _compile(self) -> None:
        # single alternation regex, longest alternatives first so that
        # "sleep driving" wins over "sleep" if both were present.
        alts = sorted(self._syn_to_pt.keys(), key=len, reverse=True)
        # allow optional plural 's' and flexible whitespace/hyphen
        pats = []
        for a in alts:
            p = re.escape(a).replace(r"\ ", r"[\s\-]+").replace(r"\-", r"[\s\-]+")
            if len(a) > 3 and not a.endswith("s"):
                p += "s?"
            pats.append(p)
        self._regex = re.compile(_WORD + "(" + "|".join(pats) + ")" + _WORD_END, re.IGNORECASE)

    # ---------------------------------------------------------- loaders
    @classmethod
    def from_seed(cls) -> "MedDRADictionary":
        return cls(SEED_TERMS)

    @classmethod
    def from_csv(cls, path: str, pt_col: str = "pt", syn_col: str = "synonym") -> "MedDRADictionary":
        terms: Dict[str, List[str]] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                terms.setdefault(row[pt_col].strip(), []).append(row[syn_col].strip())
        return cls(terms)

    @classmethod
    def from_meddra_ascii(cls, pt_asc: str, llt_asc: str, current_only: bool = True) -> "MedDRADictionary":
        """Load a licensed MedDRA distribution (``MedAscii/pt.asc`` and ``llt.asc``).

        Format: ``$``-separated; ``pt.asc`` = pt_code$pt_name$...; ``llt.asc`` =
        llt_code$llt_name$pt_code$...$llt_currency (Y/N).
        """
        pts: Dict[str, str] = {}
        with open(pt_asc, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("$")
                if len(parts) >= 2:
                    pts[parts[0]] = parts[1]
        terms: Dict[str, List[str]] = {name: [] for name in pts.values()}
        with open(llt_asc, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("$")
                if len(parts) < 3:
                    continue
                llt_name, pt_code = parts[1], parts[2]
                currency = parts[9] if len(parts) > 9 else "Y"
                if current_only and currency == "N":
                    continue
                if pt_code in pts:
                    terms[pts[pt_code]].append(llt_name)
        return cls(terms)

    # ---------------------------------------------------------- lookups
    def pt_for(self, synonym: str) -> Optional[str]:
        return self._syn_to_pt.get(synonym.lower().strip())

    def match_string(self, s: str, fuzzy_threshold: float = 0.9) -> Optional[str]:
        """Map an arbitrary string (e.g. a NER span or a FAERS PT) to a PT."""
        key = s.lower().strip()
        if key in self._syn_to_pt:
            return self._syn_to_pt[key]
        if key.endswith("s") and key[:-1] in self._syn_to_pt:
            return self._syn_to_pt[key[:-1]]
        best, best_s = None, 0.0
        for syn, pt in self._syn_to_pt.items():
            r = SequenceMatcher(None, key, syn).ratio()
            if r > best_s:
                best, best_s = pt, r
        return best if best_s >= fuzzy_threshold else None

    def pts(self) -> Set[str]:
        return set(self.terms)

    def finditer(self, text: str):
        return self._regex.finditer(text)


def _is_negated(text: str, start: int, window_chars: int = 40) -> bool:
    ctx = text[max(0, start - window_chars) : start].lower()
    return any(ctx.rstrip().endswith(c.strip()) or (" " + c) in (" " + ctx) for c in _NEGATION_CUES)


def extract_terms(text: str, dictionary: MedDRADictionary, negation: bool = True) -> List[Mention]:
    """All dictionary mentions in ``text`` with negation flags."""
    if not text:
        return []
    out: List[Mention] = []
    for m in dictionary.finditer(text):
        matched = m.group(1)
        pt = dictionary.pt_for(matched) or dictionary.pt_for(matched.rstrip("sS")) or dictionary.match_string(matched)
        if pt is None:
            continue
        out.append(Mention(pt=pt, matched=matched, start=m.start(1), end=m.end(1), negated=negation and _is_negated(text, m.start(1))))
    return out


def extract_from_label(doc, dictionary: MedDRADictionary, sections: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Term table for a :class:`~unlabeled_ae.label_client.LabelDoc`.

    Columns: ``set_id, section, pt, n_mentions, n_affirmed, postmarketing_only``.
    ``postmarketing_only`` marks PTs that appear only in the '6.2 Postmarketing
    Experience' subsection of Adverse Reactions.
    """
    from .label_client import SAFETY_SECTIONS

    secs = list(sections) if sections else list(SAFETY_SECTIONS)
    rows: List[Dict[str, object]] = []
    pm_pts = {m.pt for m in extract_terms(getattr(doc, "postmarketing_text", ""), dictionary) if not m.negated}
    for sec in secs:
        text = doc.sections.get(sec, "")
        ments = extract_terms(text, dictionary)
        if not ments:
            continue
        df = pd.DataFrame([{"pt": m.pt, "negated": m.negated} for m in ments])
        g = df.groupby("pt").agg(n_mentions=("negated", "size"), n_affirmed=("negated", lambda s: int((~s).sum())))
        for pt, r in g.iterrows():
            rows.append({"set_id": doc.set_id, "section": sec, "pt": pt, "n_mentions": int(r["n_mentions"]), "n_affirmed": int(r["n_affirmed"])})
    out = pd.DataFrame(rows, columns=["set_id", "section", "pt", "n_mentions", "n_affirmed"])
    if out.empty:
        out["postmarketing_only"] = pd.Series(dtype=bool)
        return out
    affirmed_elsewhere = set()
    ar_text = doc.sections.get("adverse_reactions", "")
    pm_text = getattr(doc, "postmarketing_text", "")
    non_pm = ar_text.replace(pm_text, " ") if pm_text else ar_text
    for sec in secs:
        t = non_pm if sec == "adverse_reactions" else doc.sections.get(sec, "")
        affirmed_elsewhere |= {m.pt for m in extract_terms(t, dictionary) if not m.negated}
    out["postmarketing_only"] = out["pt"].map(lambda p: (p in pm_pts) and (p not in affirmed_elsewhere))
    return out


def labeled_terms(term_table: pd.DataFrame, require_affirmed: bool = True) -> Set[str]:
    """Set of PTs considered 'on label' from :func:`extract_from_label` output."""
    t = term_table
    if require_affirmed and "n_affirmed" in t:
        t = t[t["n_affirmed"] > 0]
    return set(t["pt"].unique())


def scispacy_candidates(text: str, model: str = "en_ner_bc5cdr_md") -> List[str]:  # pragma: no cover
    """DISEASE entity strings from scispaCy (optional dependency)."""
    try:
        import spacy  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install spacy scispacy and the en_ner_bc5cdr_md model") from exc
    nlp = spacy.load(model)
    doc = nlp(text)
    return [e.text for e in doc.ents if e.label_ == "DISEASE"]


def map_candidates(cands: Iterable[str], dictionary: MedDRADictionary, fuzzy_threshold: float = 0.9) -> Dict[str, Optional[str]]:
    """Map NER strings to PTs (exact/synonym/fuzzy); unmapped -> None."""
    return {c: dictionary.match_string(c, fuzzy_threshold) for c in set(cands)}
