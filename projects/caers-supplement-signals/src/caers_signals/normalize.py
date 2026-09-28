"""CAERS (openFDA ``food/event``) record flattening and product normalisation.

CAERS record shape (openFDA):

    report_number, date_created (YYYYMMDD), date_started, outcomes [str],
    reactions [MedDRA PT str], consumer {age, age_unit, gender},
    products [{role: Suspect|Concomitant, name_brand, industry_code, industry_name}]

Dietary supplements are ``industry_code == "54"`` ("Vit/Min/Prot/Unconv
Diet(Human/Animal)"). Unlike FAERS, CAERS has **no ingredient coding**: the
only product identifier is the verbatim ``name_brand`` string ("AG1",
"HYDROXYCUT HARDCORE", "NATURE MADE VITAMIN D3 2000 IU"). This module

* normalises product strings (case, punctuation, dose tokens, size tokens),
* maps them to canonical ingredients with a curated regex lexicon (fast,
  transparent, covers the high-volume botanicals and nutrients), and
* looks brand names up in the NIH ODS Dietary Supplement Label Database
  (DSLD) API to retrieve the full ingredient list (slower, comprehensive).

Both mappings are kept so the analysis can report how signals depend on
the mapping route (lexicon-only vs DSLD-augmented).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .openfda import OpenFDAClient

logger = logging.getLogger(__name__)

SUPPLEMENT_INDUSTRY_CODE = "54"
#: Selected CAERS industry codes (FDA product-code industry groups); verify
#: against the CAERS data dictionary before use in publications.
INDUSTRY_CODES: Dict[str, str] = {
    "54": "Vit/Min/Prot/Unconv Diet(Human/Animal)",
    "53": "Cosmetics",
    "41": "Dietary Conv Food/Meal Replacements",
    "40": "Baked Goods/Doughs/Mixes/Icings",
    "31": "Coffee/Tea",
    "29": "Soft Drink/Water",
    "23": "Nuts/Edible Seed",
    "20": "Fruit/Fruit Prod",
    "24": "Vegetables/Vegetable Products",
    "16": "Fishery/Seafood Prod",
    "9": "Milk/Butter/Dried Milk Prod",
    "33": "Candy W/O Choc/Special/Chew Gum",
    "34": "Chocolate/Cocoa Prod",
    "45": "Food Additives (Human Use)",
}

#: canonical ingredient -> regex over the normalised product string
INGREDIENT_LEXICON: Dict[str, str] = {
    "GREEN TEA EXTRACT": r"green\s*tea|\begcg\b|camellia",
    "GARCINIA CAMBOGIA": r"garcinia|hydroxycitric|\bhca\b",
    "KAVA": r"\bkava",
    "KRATOM": r"kratom|mitragyn",
    "ASHWAGANDHA": r"ashwagandha|withania",
    "TURMERIC/CURCUMIN": r"turmeric|curcumin",
    "YOHIMBE": r"yohimb",
    "EPHEDRA": r"ephedra|ma\s*huang|ephedrine",
    "BITTER ORANGE/SYNEPHRINE": r"bitter\s*orange|synephrine|aurantium",
    "RED YEAST RICE": r"red\s*yeast\s*rice|monacolin",
    "NIACIN": r"\bniacin|nicotinic\s*acid|vitamin\s*b\s*3\b",
    "VITAMIN D": r"vitamin\s*d\s*3?\b|cholecalciferol|ergocalciferol",
    "VITAMIN A": r"vitamin\s*a\b|retinol|retinyl",
    "VITAMIN C": r"vitamin\s*c\b|ascorbic",
    "VITAMIN E": r"vitamin\s*e\b|tocopherol",
    "VITAMIN B12": r"vitamin\s*b\s*12\b|cobalamin",
    "IRON": r"\biron\b|ferrous|ferric",
    "MAGNESIUM": r"magnesium",
    "ZINC": r"\bzinc\b",
    "CALCIUM": r"calcium",
    "POTASSIUM": r"potassium",
    "SELENIUM": r"selenium",
    "CHROMIUM": r"chromium",
    "MELATONIN": r"melatonin",
    "CREATINE": r"creatine",
    "WHEY/PROTEIN POWDER": r"\bwhey\b|protein\s*(powder|shake|isolate)",
    "BIOTIN": r"biotin",
    "COLLAGEN": r"collagen",
    "ELDERBERRY": r"elderberry|sambucus",
    "BLACK COHOSH": r"black\s*cohosh|cimicifuga",
    "SAW PALMETTO": r"saw\s*palmetto|serenoa",
    "ST JOHN'S WORT": r"st\.?\s*john|hypericum",
    "VALERIAN": r"valerian",
    "BERBERINE": r"berberine",
    "5-HTP": r"5-?\s*htp|hydroxytryptophan",
    "DMAA": r"\bdmaa\b|methylhexan|geranium",
    "CAFFEINE/STIMULANT": r"caffeine|guarana|energy\s*(shot|drink|pill)|pre-?\s*workout",
    "FISH OIL/OMEGA-3": r"fish\s*oil|omega-?\s*3|krill|\bdha\b|\bepa\b",
    "PROBIOTIC": r"probiotic|lactobacillus|bifidobacter|acidophilus",
    "MULTIVITAMIN": r"multi-?\s*vit|multiple\s*vitamin|centrum|one\s*a\s*day|gummy\s*vitamin|gummies",
    "CBD/HEMP": r"\bcbd\b|cannabidiol|\bhemp\b",
    "APPLE CIDER VINEGAR": r"apple\s*cider\s*vinegar|\bacv\b",
    "ALPHA-LIPOIC ACID": r"lipoic",
    "GLUCOSAMINE/CHONDROITIN": r"glucosamine|chondroitin",
    "MILK THISTLE": r"milk\s*thistle|silymarin",
    "GINKGO": r"ginkgo",
    "GINSENG": r"ginseng",
    "ECHINACEA": r"echinacea",
    "TESTOSTERONE BOOSTER": r"testosterone|tribulus|fenugreek|\bt-?booster",
    "WEIGHT-LOSS BLEND": r"weight\s*loss|fat\s*burn|diet\s*pill|slim|lipo|thermogenic|hydroxycut|oxyelite",
    "SEXUAL ENHANCEMENT": r"sexual|male\s*enhance|rhino\b|libido|viril",
    "BODYBUILDING/SARM": r"sarm|ostarine|ligandrol|anabolic|prohormone|bodybuild",
    "FIBER/PSYLLIUM": r"psyllium|metamucil|fiber|fibre",
    "GLP-1 MIMIC (SUPPLEMENT)": r"ozempic|semaglutide|glp-?1|berberine.*glp",
}
_LEXICON_RE: Dict[str, re.Pattern] = {k: re.compile(v, re.I) for k, v in INGREDIENT_LEXICON.items()}

_DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|iu|g|ml|oz|ct|caps?|tabs?|tablets?|capsules?|softgels?|gummies|servings?|count|pack)\b", re.I)
_PUNCT_RE = re.compile(r"[^A-Z0-9 ]+")
_WS_RE = re.compile(r"\s+")
_AGE_UNIT_TO_YEARS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52.18, "day": 1 / 365.25, "decade": 10.0}

SERIOUS_OUTCOMES: Set[str] = {
    "DEATH",
    "LIFE THREATENING",
    "HOSPITALIZATION",
    "DISABILITY",
    "CONGENITAL ANOMALY",
    "REQUIRED INTERVENTION",
    "OTHER SERIOUS OR IMPORTANT MEDICAL EVENT",
}
MEDICALLY_ATTENDED: Set[str] = {"VISITED AN EMERGENCY ROOM", "VISITED A HEALTH CARE PROVIDER"}


def normalise_product(name: Any) -> str:
    """Upper-case, strip dose/size tokens and punctuation, collapse spaces."""
    s = str(name or "").upper()
    s = _DOSE_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def map_ingredients(product_name: str) -> List[str]:
    """Canonical ingredients matched by the lexicon (possibly several)."""
    s = normalise_product(product_name)
    return [k for k, pat in _LEXICON_RE.items() if pat.search(s)]


def age_years(age: Any, unit: Any) -> Optional[float]:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    u = str(unit or "").lower().replace("(s)", "").strip()
    f = _AGE_UNIT_TO_YEARS.get(u)
    if f is None:
        return None
    yrs = a * f
    return yrs if 0 <= yrs <= 120 else None


def outcome_flags(outcomes: Iterable[str]) -> Dict[str, bool]:
    o = {str(x).upper().strip() for x in outcomes or []}
    return {"serious": bool(o & SERIOUS_OUTCOMES), "death": "DEATH" in o, "hospitalization": "HOSPITALIZATION" in o, "medically_attended": bool(o & MEDICALLY_ATTENDED)}


def flatten_caers(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one CAERS report; products keep role, industry code and mapped ingredients."""
    consumer = rec.get("consumer", {}) or {}
    products = []
    for p in rec.get("products", []) or []:
        name = p.get("name_brand") or ""
        products.append(
            {
                "name": normalise_product(name),
                "raw_name": name,
                "role": str(p.get("role") or "").title(),
                "industry_code": str(p.get("industry_code") or ""),
                "industry_name": p.get("industry_name"),
                "ingredients": map_ingredients(name),
            }
        )
    supp = [p for p in products if p["industry_code"] == SUPPLEMENT_INDUSTRY_CODE]
    gender = str(consumer.get("gender") or "").lower()
    flags = outcome_flags(rec.get("outcomes", []))
    return {
        "report_number": rec.get("report_number"),
        "date_created": rec.get("date_created"),
        "date_started": rec.get("date_started"),
        "sex": "female" if gender.startswith("f") else ("male" if gender.startswith("m") else "unknown"),
        "age_years": age_years(consumer.get("age"), consumer.get("age_unit")),
        "outcomes": [str(x).upper() for x in rec.get("outcomes", []) or []],
        **flags,
        "reactions": sorted({str(r).upper() for r in rec.get("reactions", []) or [] if r}),
        "products": products,
        "supplement": bool(supp),
        "suspect_supplements": sorted({p["name"] for p in supp if p["role"] == "Suspect"}),
        "ingredients": sorted({i for p in supp for i in p["ingredients"]}),
    }


def flatten_faers_for_supplements(rec: Dict[str, Any]) -> Dict[str, Any]:
    """FAERS report -> comparable row using verbatim ``medicinalproduct`` names.

    Supplements reach FAERS as drugs (often without ``openfda`` mapping);
    the same lexicon is applied to the verbatim names so CAERS and FAERS
    ingredient-level tables are directly comparable.
    """
    patient = rec.get("patient", {}) or {}
    names = []
    for d in patient.get("drug", []) or []:
        names.append(str(d.get("medicinalproduct") or ""))
        names.extend((d.get("openfda", {}) or {}).get("generic_name") or [])
    ingredients = sorted({i for n in names for i in map_ingredients(n)})
    sex_code = str(patient.get("patientsex"))
    return {
        "report_number": rec.get("safetyreportid"),
        "date_created": rec.get("receivedate"),
        "sex": {"1": "male", "2": "female"}.get(sex_code, "unknown"),
        "serious": str(rec.get("serious")) == "1",
        "reactions": sorted({str(r.get("reactionmeddrapt", "")).upper() for r in (patient.get("reaction", []) or []) if r.get("reactionmeddrapt")}),
        "product_names": [normalise_product(n) for n in names if n],
        "ingredients": ingredients,
        "supplement": bool(ingredients),
    }


# --------------------------------------------------------------------- DSLD
DSLD_BASE_URL = "https://api.ods.od.nih.gov/dsld/v9"


def dsld_search(query: str, session: Any, base_url: str = DSLD_BASE_URL, size: int = 5, timeout: float = 30.0) -> List[Dict[str, Any]]:
    """Look a product name up in the DSLD API and return candidate labels.

    Returns ``[{"dsld_id", "full_name", "brand", "ingredients": [...]}]``.
    The response schema of the DSLD API has changed between versions; this
    parser accepts both ``{"hits": [{"_source": {...}}]}`` and a plain list
    and extracts ingredient names defensively. Verify the endpoint and field
    names against https://dsld.od.nih.gov/api-guide before large runs.
    """
    resp = session.get(f"{base_url}/search-filter", params={"q": query, "size": size}, timeout=timeout)
    if getattr(resp, "status_code", 500) != 200:
        return []
    payload = resp.json()
    hits = payload.get("hits", payload) if isinstance(payload, dict) else payload
    if isinstance(hits, dict) and "hits" in hits:
        hits = hits["hits"]
    out = []
    for h in hits or []:
        src = h.get("_source", h) if isinstance(h, dict) else {}
        ingredients: List[str] = []
        for row in src.get("ingredientRows", []) or src.get("ingredients", []) or []:
            if isinstance(row, dict):
                nm = row.get("name") or row.get("ingredientName")
                if nm:
                    ingredients.append(str(nm).upper())
            elif isinstance(row, str):
                ingredients.append(row.upper())
        out.append({"dsld_id": src.get("id") or h.get("_id"), "full_name": src.get("fullName") or src.get("productName"), "brand": src.get("brandName"), "ingredients": sorted(set(ingredients))})
    return out


def dsld_ingredients_for_products(names: Iterable[str], session: Any, cache: Optional[Dict[str, List[str]]] = None, **kw: Any) -> Dict[str, List[str]]:
    """Batch DSLD lookup with an in-memory cache; unmatched names map to []."""
    cache = cache if cache is not None else {}
    for n in names:
        if n in cache:
            continue
        try:
            hits = dsld_search(n, session, **kw)
        except Exception as exc:  # network / schema problems are logged, not fatal
            logger.warning("DSLD lookup failed for %r: %s", n, exc)
            hits = []
        cache[n] = hits[0]["ingredients"] if hits else []
    return cache
