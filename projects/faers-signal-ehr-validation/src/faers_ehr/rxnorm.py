"""RxNorm / RxNav helpers for mapping MIMIC-IV drugs and openFDA drugs to ingredients.

MIMIC-IV side
-------------
``hosp/prescriptions`` carries ``ndc`` (11-digit string, '0' or empty when unknown), ``gsn``
(Generic Sequence Number, First Databank), ``drug`` (free-text order name), ``formulary_drug_cd``
and ``pharmacy_id``. ``hosp/emar`` holds administrations (``event_txt = 'Administered'``,
``charttime``) and links to prescriptions through ``pharmacy_id``; ``emar_detail`` carries the
dose actually given. Exposure timing should come from emar, the product identity from
prescriptions.ndc -> RxCUI -> ingredient (RxNav ``/rxcui.json?idtype=NDC``, then
``/rxcui/{rxcui}/related.json?tty=IN``), with a name-based fallback (``/approximateTerm.json``).

openFDA side
------------
``patient.drug.openfda.generic_name`` and ``patient.drug.openfda.rxcui`` are populated by FDA
from the product NDC/SPL; ingredient names are upper-case and salts are usually already stripped
(e.g. 'METOPROLOL TARTRATE' -> generic_name 'METOPROLOL TARTRATE' but rxcui includes the IN).
:func:`ingredient_key` normalises both sides to a comparable key.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

RXNAV = "https://rxnav.nlm.nih.gov/REST"

SALT_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "sulfate", "sulphate", "tartrate", "succinate", "maleate",
    "mesylate", "besylate", "acetate", "citrate", "phosphate", "bromide", "chloride", "fumarate", "calcium",
    "magnesium", "disodium", "trihydrate", "monohydrate", "dihydrate", "hydrobromide", "tosylate", "lactate",
    "gluconate", "bitartrate", "nitrate", "carbonate", "bicarbonate", "valerate", "propionate", "dipropionate",
    "palmitate", "decanoate", "enanthate", "pamoate", "hyclate", "estolate", "stearate", "benzoate",
}


UNIT_FORM_WORDS = {
    "mg", "mcg", "g", "ml", "meq", "iu", "unit", "units", "tablet", "tablets", "tab", "tabs", "capsule", "capsules",
    "cap", "caps", "injection", "inj", "solution", "soln", "suspension", "oral", "iv", "po", "er", "sr", "xl", "cr",
    "extended", "release", "delayed", "vial", "bag", "premix", "syringe", "patch", "cream", "ointment", "drops",
}


def normalize_ndc(ndc: str | int | float | None) -> str | None:
    """MIMIC-IV NDCs are 11-digit (5-4-2) strings; return zero-padded 11 digits or None."""
    if ndc is None or (isinstance(ndc, float) and pd.isna(ndc)):
        return None
    s = re.sub(r"\D", "", str(ndc))
    if not s or int(s) == 0:
        return None
    return s.zfill(11)[-11:]


def ingredient_key(name: str | None) -> str | None:
    """Lower-case, strip salts / dosage-form words / punctuation for cross-source joining."""
    if not name or (isinstance(name, float) and pd.isna(name)):
        return None
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    toks = [t for t in s.split()
            if t not in SALT_WORDS and t not in UNIT_FORM_WORDS and not re.fullmatch(r"\d+(\.\d+)?(mg|mcg|g|ml|meq|iu|units?)?", t)]
    return " ".join(toks) or None


@dataclass
class RxNavClient:
    """Thin cached client for the RxNav REST API (no key; be polite: <= 20 requests/s)."""

    cache_path: str | os.PathLike | None = "data/rxnorm_cache.json"
    min_interval_s: float = 0.06
    timeout: float = 30.0
    session: requests.Session = field(default_factory=requests.Session)
    _cache: dict = field(default_factory=dict, repr=False)
    _last: float = 0.0

    def __post_init__(self) -> None:
        if self.cache_path and Path(self.cache_path).exists():
            with open(self.cache_path) as f:
                self._cache = json.load(f)

    def save(self) -> None:
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f)

    def _get(self, path: str, params: dict | None = None) -> dict:
        key = path + "?" + json.dumps(params or {}, sort_keys=True)
        if key in self._cache:
            return self._cache[key]
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.session.get(RXNAV + path, params=params, timeout=self.timeout)
        self._last = time.time()
        r.raise_for_status()
        out = r.json()
        self._cache[key] = out
        return out

    def ndc_to_rxcui(self, ndc: str) -> str | None:
        n = normalize_ndc(ndc)
        if not n:
            return None
        out = self._get("/rxcui.json", {"idtype": "NDC", "id": n})
        ids = out.get("idGroup", {}).get("rxnormId", [])
        if ids:
            return ids[0]
        # obsolete NDCs: ndcstatus gives the historical RxCUI
        st = self._get("/ndcstatus.json", {"ndc": n}).get("ndcStatus", {})
        return st.get("rxcui") or None

    def name_to_rxcui(self, name: str) -> str | None:
        out = self._get("/rxcui.json", {"name": name, "search": 2})
        ids = out.get("idGroup", {}).get("rxnormId", [])
        if ids:
            return ids[0]
        approx = self._get("/approximateTerm.json", {"term": name, "maxEntries": 1}).get("approximateGroup", {})
        cands = approx.get("candidate", [])
        return cands[0].get("rxcui") if cands else None

    def rxcui_to_ingredients(self, rxcui: str) -> list[dict[str, str]]:
        """Ingredient (TTY=IN) concepts related to any RxCUI (SCD, SBD, BN, PIN, ...)."""
        out = self._get(f"/rxcui/{rxcui}/related.json", {"tty": "IN"})
        groups = out.get("relatedGroup", {}).get("conceptGroup", [])
        res = []
        for g in groups:
            for c in g.get("conceptProperties", []) or []:
                res.append({"rxcui": c.get("rxcui"), "name": c.get("name")})
        if not res:  # already an ingredient?
            props = self._get(f"/rxcui/{rxcui}/properties.json").get("properties", {})
            if props.get("tty") == "IN":
                res.append({"rxcui": props.get("rxcui"), "name": props.get("name")})
        return res

    def map_prescriptions(self, prescriptions: pd.DataFrame, ndc_col: str = "ndc", name_col: str = "drug",
                          max_unique: int | None = None) -> pd.DataFrame:
        """Map unique (ndc, drug) pairs to ingredient RxCUIs / names; returns a lookup frame.

        Columns: ndc, drug, rxcui, ingredient_rxcui, ingredient, method (ndc|name|none).
        Multi-ingredient products give one row per ingredient.
        """
        pairs = prescriptions[[ndc_col, name_col]].drop_duplicates()
        if max_unique:
            pairs = pairs.head(max_unique)
        rows = []
        for ndc, name in pairs.itertuples(index=False):
            rx, method = None, "none"
            n = normalize_ndc(ndc)
            if n:
                rx = self.ndc_to_rxcui(n)
                method = "ndc" if rx else method
            if not rx and isinstance(name, str) and name.strip():
                rx = self.name_to_rxcui(name.strip())
                method = "name" if rx else method
            ings = self.rxcui_to_ingredients(rx) if rx else []
            if not ings:
                rows.append({"ndc": ndc, "drug": name, "rxcui": rx, "ingredient_rxcui": None, "ingredient": None, "method": method})
            for ing in ings:
                rows.append({"ndc": ndc, "drug": name, "rxcui": rx, "ingredient_rxcui": ing["rxcui"],
                             "ingredient": ingredient_key(ing["name"]), "method": method})
        self.save()
        return pd.DataFrame(rows)


def mapping_coverage(lookup: pd.DataFrame, prescriptions: pd.DataFrame, ndc_col: str = "ndc", name_col: str = "drug") -> dict[str, float]:
    """Share of prescription rows (not unique products) that received an ingredient."""
    ok = lookup.dropna(subset=["ingredient"])[[ndc_col, name_col]].drop_duplicates()
    merged = prescriptions[[ndc_col, name_col]].merge(ok.assign(_ok=1), how="left", on=[ndc_col, name_col])
    return {"rows": int(len(merged)), "rows_mapped": int(merged["_ok"].fillna(0).sum()),
            "coverage": float(merged["_ok"].fillna(0).mean())}


OPENFDA_MAPPING_NOTES = """
openFDA <-> RxNorm notes
- patient.drug.openfda.rxcui lists all RxCUIs attached to the product SPL (IN, PIN, SCD, SBD, BN).
  Join on the ingredient RxCUI (TTY=IN) when possible; generic_name is a fallback.
- Combination products appear once per product but map to several ingredients; count the report
  for every ingredient (standard practice) and flag combination-only exposures in sensitivity analyses.
- ~15-20% of FAERS drug entries have no openfda block (foreign or unrecognised products); they are
  invisible to generic_name queries. Query patient.drug.medicinalproduct with the brand/generic
  synonyms from RxNav (/rxcui/{rxcui}/allrelated.json) to estimate the loss.
"""
