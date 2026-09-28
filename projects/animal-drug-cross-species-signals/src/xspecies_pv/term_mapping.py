"""Ingredient normalisation and organ-system harmonisation of VeDDRA and MedDRA terms.

Why a rule-based layer? MedDRA is licensed and VeDDRA's hierarchy is only partly aligned with
it. A dictionary-independent classifier that assigns *both* vocabularies to the same ~22
harmonised organ-system buckets gives a reproducible primary analysis; the licensed SOC
hierarchies then refine it (``refine_with_soc``). PT-level candidates for manual review come from
:func:`suggest_pt_matches` (difflib; ``rapidfuzz`` if installed).
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional accelerator
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except ImportError:  # pragma: no cover
    _rf_fuzz = None

# --------------------------------------------------------------------------- ingredients
_SALTS = (
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "sulfate", "sulphate", "acetate", "citrate",
    "tartrate", "maleate", "mesylate", "mesilate", "besylate", "succinate", "fumarate", "phosphate",
    "bromide", "chloride", "monohydrate", "dihydrate", "trihydrate", "anhydrous", "hydrobromide",
    "lactate", "gluconate", "tosylate", "pamoate", "valerate", "propionate", "decanoate", "benzoate",
    "trans", "cis", "dl", "usp", "bp", "micronized", "micronised", "hemihydrate", "sesquihydrate",
)
_STRENGTH_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|ug|g|ml|%|iu|units?)(/\s*(ml|kg|l))?\b")


def normalize_ingredient(name: Optional[str]) -> str:
    """Lower-case ingredient key without salts, strengths or parentheticals.

    >>> normalize_ingredient("Meloxicam 1.5 mg/mL")
    'meloxicam'
    >>> normalize_ingredient("Fluoxetine Hydrochloride")
    'fluoxetine'
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = _STRENGTH_RE.sub(" ", s)
    s = re.sub(r"[^a-z\s\-]", " ", s)
    toks = [t for t in re.split(r"[\s\-]+", s) if t and t not in _SALTS]
    return " ".join(toks).strip()


# --------------------------------------------------------------------------- organ systems
#: Harmonised organ-system buckets (order matters: earlier rules win on ties).
ORGAN_SYSTEMS: Tuple[str, ...] = (
    "death", "lack_of_efficacy", "gastrointestinal", "hepatic", "renal_urinary", "nervous", "behavioural_psychiatric",
    "cardiovascular", "respiratory", "skin_hypersensitivity", "immune_infection", "haematological", "musculoskeletal",
    "endocrine_metabolic", "reproductive", "eye", "ear", "injection_application_site", "neoplasm", "general_systemic",
    "investigations", "injury_poisoning",
)

_RULES: Dict[str, Tuple[str, ...]] = {
    "death": ("death", "died", "euthan", "fatal", "sudden death", "found dead"),
    "lack_of_efficacy": ("lack of efficacy", "ineffective", "lack of expected effect", "drug ineffective",
                         "treatment failure", "no effect", "inefficacy"),
    "gastrointestinal": ("vomit", "emesis", "diarrh", "gastro", "intestin", "colitis", "constipat", "haematemesis",
                         "hematemesis", "melaena", "melena", "ulcer", "nausea", "abdominal", "flatulence",
                         "regurgitat", "salivat", "hypersalivat", "drool", "anorexia", "inappetence", "decreased appetite",
                         "appetite", "pancreatitis", "dysphag", "stomach", "bowel", "faeces", "feces", "stool"),
    "hepatic": ("hepat", "liver", "jaundice", "icter", "cholesta", "bilirubin", "alt increased", "alanine aminotransferase",
                "aspartate aminotransferase", "alkaline phosphatase"),
    "renal_urinary": ("renal", "kidney", "nephr", "urin", "azotaemia", "azotemia", "creatinine", "polyuria", "polydipsia",
                      "haematuria", "hematuria", "cystitis", "incontinence", "dysuria", "stranguria"),
    "nervous": ("seizure", "convuls", "epilep", "ataxi", "tremor", "neurolog", "neuropath", "paresis", "paralys",
                "twitch", "nystagmus", "head tilt", "dizz", "vertigo", "syncope", "somnolen", "sedat", "lethargy",
                "letharg", "stupor", "coma", "encephal", "myoclon", "hyperaesthesia", "hyperesthesia", "headache",
                "neuro", "muscle fascicul", "collapse"),
    "behavioural_psychiatric": ("behavio", "aggress", "anxiety", "anxious", "agitat", "hyperactiv", "depress",
                                "restless", "vocali", "disorient", "confus", "hallucin", "insomnia", "psych",
                                "suicid", "irritab", "hiding", "excitab"),
    "cardiovascular": ("cardi", "heart", "arrhythm", "tachycard", "bradycard", "hypotens", "hypertens", "myocard",
                       "thromb", "embol", "vascul", "haemorrhage", "hemorrhage", "bleeding", "oedema", "edema",
                       "pallor", "cyanosis", "shock", "qt "),
    "respiratory": ("respir", "dyspn", "tachypn", "cough", "pneumon", "pulmon", "bronch", "lung", "apnoea", "apnea",
                    "wheez", "sneez", "nasal", "rhinitis", "epistaxis", "breath"),
    "skin_hypersensitivity": ("pruritus", "prurit", "itch", "alopecia", "hair loss", "dermat", "skin", "rash", "urticaria",
                              "hives", "erythema", "hypersensitiv", "allerg", "anaphyla", "angioedema", "facial swelling",
                              "pyoderma", "wound", "lesion", "scab", "crust"),
    "immune_infection": ("infect", "sepsis", "abscess", "immune", "autoimmun", "lupus", "fever", "pyrexia", "hyperthermia",
                         "fungal", "bacterial", "viral"),
    "haematological": ("anaemia", "anemia", "thrombocytop", "neutropen", "leukopen", "pancytopen", "haemolytic", "hemolytic",
                       "methaemoglobin", "methemoglobin", "coagul", "platelet", "haematolog", "hematolog", "bone marrow",
                       "petechia", "ecchymos", "lymphaden"),
    "musculoskeletal": ("lame", "musculoskelet", "arthr", "joint", "muscle", "myalg", "myopath", "bone", "fracture",
                        "tendon", "ligament", "stiff", "weakness", "paw", "limb", "gait", "reluctan"),
    "endocrine_metabolic": ("hypoglyc", "hyperglyc", "diabet", "thyroid", "cushing", "addison", "adrenal", "cortisol",
                            "hypokal", "hyperkal", "hyponatr", "hypernatr", "hypocalc", "hypercalc", "acidosis", "alkalosis",
                            "weight", "obes", "metabol", "polyphagia", "endocrin", "electrolyte", "dehydrat", "glucose"),
    "reproductive": ("abort", "pregnan", "reproduct", "infertil", "oestrus", "estrus", "testic", "uterine", "vagin",
                     "mammary", "lactation", "libido", "erectile", "menstru"),
    "eye": ("ocular", "eye", "blind", "conjunctiv", "corneal", "cataract", "uveitis", "mydriasis", "miosis", "vision",
            "visual", "retin", "glaucoma", "lacrimation", "epiphora", "keratitis"),
    "ear": ("ear", "otitis", "deaf", "hearing", "tinnitus", "otic"),
    "injection_application_site": ("injection site", "application site", "administration site", "infusion site",
                                   "implant site", "site reaction", "site swelling", "site pain"),
    "neoplasm": ("neoplas", "tumour", "tumor", "cancer", "carcinoma", "sarcoma", "lymphoma", "malignan", "mast cell"),
    "investigations": ("increased", "decreased", "abnormal", "elevated", "test", "laboratory", "investigation", "blood ",
                       "serum", "level", "count", "ecg", "electrocardiogram"),
    "injury_poisoning": ("overdose", "poison", "toxic", "intoxic", "accidental", "exposure", "ingestion", "medication error",
                         "wrong", "injur", "trauma", "laceration", "bite"),
    "general_systemic": ("malaise", "asthenia", "fatigue", "pain", "swelling", "systemic", "general", "discomfort",
                         "hyperthermia", "hypothermia", "shaking", "shiver", "panting", "weak", "condition aggravated",
                         "unresponsive", "recumben", "ill", "unwell"),
}


def _norm(s: str) -> str:
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def assign_organ_system(term: Optional[str], vocab_soc: Optional[str] = None) -> str:
    """Rule-based organ-system bucket for a VeDDRA or MedDRA term.

    If an official SOC is supplied (``vocab_soc``), it is tried first via :func:`soc_to_bucket`.
    Falls back to keyword rules in priority order, then ``"unclassified"``.
    """
    if vocab_soc:
        b = soc_to_bucket(vocab_soc)
        if b != "unclassified":
            return b
    if not term or not isinstance(term, str):
        return "unclassified"
    t = " " + _norm(term) + " "
    for bucket in ORGAN_SYSTEMS:
        for kw in _RULES.get(bucket, ()):
            if kw in t:
                return bucket
    return "unclassified"


_SOC_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("gastro", "gastrointestinal"), ("digest", "gastrointestinal"), ("hepat", "hepatic"), ("renal", "renal_urinary"),
    ("urinary", "renal_urinary"), ("nervous", "nervous"), ("neurolog", "nervous"), ("psychiatr", "behavioural_psychiatric"),
    ("behavio", "behavioural_psychiatric"), ("cardiac", "cardiovascular"), ("vascular", "cardiovascular"),
    ("respiratory", "respiratory"), ("skin", "skin_hypersensitivity"), ("immune", "immune_infection"),
    ("infection", "immune_infection"), ("blood", "haematological"), ("lymphatic", "haematological"),
    ("musculoskeletal", "musculoskeletal"), ("endocrine", "endocrine_metabolic"), ("metabol", "endocrine_metabolic"),
    ("nutrition", "endocrine_metabolic"), ("reproduct", "reproductive"), ("pregnan", "reproductive"), ("eye", "eye"),
    ("ear ", "ear"), ("labyrinth", "ear"), ("injection", "injection_application_site"), ("application site", "injection_application_site"),
    ("neoplasm", "neoplasm"), ("investigation", "investigations"), ("injury", "injury_poisoning"), ("poison", "injury_poisoning"),
    ("product issue", "injury_poisoning"), ("general", "general_systemic"), ("systemic", "general_systemic"),
    ("death", "death"), ("efficacy", "lack_of_efficacy"), ("effectiveness", "lack_of_efficacy"),
)


def soc_to_bucket(soc: str) -> str:
    """Map an official MedDRA or VeDDRA SOC name to a harmonised bucket."""
    s = " " + _norm(soc) + " "
    for kw, bucket in _SOC_KEYWORDS:
        if kw in s:
            return bucket
    return "unclassified"


def refine_with_soc(terms: Iterable[str], soc_lookup: Dict[str, str]) -> Dict[str, str]:
    """Assign buckets using the official SOC where known, keyword rules otherwise."""
    return {t: assign_organ_system(t, soc_lookup.get(t)) for t in terms}


# --------------------------------------------------------------------------- PT matching
_SYNONYMS: Dict[str, str] = {
    "emesis": "vomiting", "vomited": "vomiting", "anorexia": "decreased appetite", "inappetence": "decreased appetite",
    "diarrhoea": "diarrhoea", "diarrhea": "diarrhoea", "lethargy": "lethargy", "hypersalivation": "salivary hypersecretion",
    "ptyalism": "salivary hypersecretion", "polydipsia": "polydipsia", "pruritus": "pruritus", "itching": "pruritus",
    "convulsion": "seizure", "seizures": "seizure", "fit": "seizure", "ataxia": "ataxia", "death": "death",
    "elevated alt": "alanine aminotransferase increased", "alt increased": "alanine aminotransferase increased",
    "elevated alp": "blood alkaline phosphatase increased", "ineffective": "drug ineffective",
    "lack of efficacy": "drug ineffective",
}


def _similarity(a: str, b: str) -> float:
    if _rf_fuzz is not None:
        return float(_rf_fuzz.token_set_ratio(a, b)) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def suggest_pt_matches(veddra_term: str, meddra_pts: Sequence[str], k: int = 3,
                       min_score: float = 0.6) -> List[Tuple[str, float]]:
    """Top-``k`` MedDRA PT candidates for a VeDDRA term (synonym table first, then fuzzy).

    Returns ``[(pt, score)]`` sorted by score; a synonym hit scores 1.0. Intended output for a
    human review sheet, not for unsupervised use.
    """
    v = _norm(veddra_term)
    out: List[Tuple[str, float]] = []
    syn = _SYNONYMS.get(v)
    pts_norm = {p: _norm(p) for p in meddra_pts}
    if syn:
        for p, pn in pts_norm.items():
            if pn == _norm(syn):
                out.append((p, 1.0))
    scored = [(p, _similarity(v, pn)) for p, pn in pts_norm.items()]
    scored.sort(key=lambda x: -x[1])
    for p, s in scored:
        if s < min_score:
            break
        if all(p != q for q, _ in out):
            out.append((p, float(s)))
        if len(out) >= k:
            break
    return out[:k]
