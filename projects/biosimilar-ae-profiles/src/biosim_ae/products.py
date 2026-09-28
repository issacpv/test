"""US biosimilar catalogue and product attribution for FAERS drug entries.

The central measurement problem of biosimilar pharmacovigilance is *product
attribution*: a FAERS drug entry may name a brand ("HYRIMOZ"), a suffixed
nonproprietary name ("ADALIMUMAB-ADAZ"), the originator brand ("HUMIRA"), or
only the shared INN ("ADALIMUMAB"), which cannot be attributed to any single
product (Vermeer et al., 2013, *Drug Saf*). FDA's four-letter suffix naming
convention (guidance "Nonproprietary Naming of Biological Products", 2017)
exists precisely to make such reports attributable.

:func:`attribute_drug_entry` implements the hierarchy

    biosimilar brand  >  suffixed INN  >  originator brand  >  INN only (unattributable)

and :func:`attribute_report` summarises a whole report. The catalogue below is
curated from the FDA Purple Book (https://purplebooksearch.fda.gov/); launch
dates are the first US commercial availability where that is well
established and ``None`` where it is uncertain or the product has not
launched. **Re-verify against the Purple Book before publication**; entries
change every quarter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .openfda import SEX_LABELS, age_in_years


@dataclass(frozen=True)
class Biosimilar:
    brand: str
    nonproprietary: str  # INN with suffix, e.g. "adalimumab-atto"
    sponsor: str
    approval_year: int
    us_launch: Optional[str]  # "YYYY-MM" or None
    interchangeable: bool = False

    @property
    def suffix(self) -> str:
        return self.nonproprietary.split("-")[-1].upper()


@dataclass(frozen=True)
class Family:
    inn: str
    originator_brand: str
    originator_sponsor: str
    originator_bla: Optional[str]
    route: str
    biosimilars: Sequence[Biosimilar]


FAMILIES: Dict[str, Family] = {
    "ADALIMUMAB": Family(
        "ADALIMUMAB", "HUMIRA", "AbbVie", "BLA125057", "subcutaneous",
        (
            Biosimilar("AMJEVITA", "adalimumab-atto", "Amgen", 2016, "2023-01"),
            Biosimilar("CYLTEZO", "adalimumab-adbm", "Boehringer Ingelheim", 2017, "2023-07", True),
            Biosimilar("HYRIMOZ", "adalimumab-adaz", "Sandoz", 2018, "2023-07"),
            Biosimilar("HADLIMA", "adalimumab-bwwd", "Samsung Bioepis/Organon", 2019, "2023-07"),
            Biosimilar("ABRILADA", "adalimumab-afzb", "Pfizer", 2019, "2023-11", True),
            Biosimilar("HULIO", "adalimumab-fkjp", "Biocon/Viatris", 2020, "2023-07"),
            Biosimilar("YUSIMRY", "adalimumab-aqvh", "Coherus", 2021, "2023-07"),
            Biosimilar("IDACIO", "adalimumab-aacf", "Fresenius Kabi", 2022, "2023-07"),
            Biosimilar("YUFLYMA", "adalimumab-aaty", "Celltrion", 2023, "2023-07"),
            Biosimilar("SIMLANDI", "adalimumab-ryvk", "Alvotech/Teva", 2024, "2024-05", True),
        ),
    ),
    "INFLIXIMAB": Family(
        "INFLIXIMAB", "REMICADE", "Janssen", "BLA103772", "intravenous",
        (
            Biosimilar("INFLECTRA", "infliximab-dyyb", "Celltrion/Pfizer", 2016, "2016-11"),
            Biosimilar("RENFLEXIS", "infliximab-abda", "Samsung Bioepis/Organon", 2017, "2017-07"),
            Biosimilar("IXIFI", "infliximab-qbtx", "Pfizer", 2017, None),
            Biosimilar("AVSOLA", "infliximab-axxq", "Amgen", 2019, "2020-07"),
        ),
    ),
    "RITUXIMAB": Family(
        "RITUXIMAB", "RITUXAN", "Genentech/Biogen", "BLA103705", "intravenous",
        (
            Biosimilar("TRUXIMA", "rituximab-abbs", "Celltrion/Teva", 2018, "2019-11"),
            Biosimilar("RUXIENCE", "rituximab-pvvr", "Pfizer", 2019, "2020-01"),
            Biosimilar("RIABNI", "rituximab-arrx", "Amgen", 2020, "2021-01"),
        ),
    ),
    "TRASTUZUMAB": Family(
        "TRASTUZUMAB", "HERCEPTIN", "Genentech", "BLA103792", "intravenous",
        (
            Biosimilar("OGIVRI", "trastuzumab-dkst", "Biocon/Viatris", 2017, "2019-12"),
            Biosimilar("HERZUMA", "trastuzumab-pkrb", "Celltrion/Teva", 2018, "2020-03"),
            Biosimilar("ONTRUZANT", "trastuzumab-dttb", "Samsung Bioepis/Organon", 2019, "2020-04"),
            Biosimilar("TRAZIMERA", "trastuzumab-qyyp", "Pfizer", 2019, "2020-02"),
            Biosimilar("KANJINTI", "trastuzumab-anns", "Amgen", 2019, "2019-07"),
            Biosimilar("HERCESSI", "trastuzumab-strf", "Accord", 2024, None),
        ),
    ),
    "BEVACIZUMAB": Family(
        "BEVACIZUMAB", "AVASTIN", "Genentech", "BLA125085", "intravenous",
        (
            Biosimilar("MVASI", "bevacizumab-awwb", "Amgen", 2017, "2019-07"),
            Biosimilar("ZIRABEV", "bevacizumab-bvzr", "Pfizer", 2019, "2019-12"),
            Biosimilar("ALYMSYS", "bevacizumab-maly", "Amneal", 2022, "2022-10"),
            Biosimilar("VEGZELMA", "bevacizumab-adcd", "Celltrion", 2022, "2023-04"),
            Biosimilar("AVZIVI", "bevacizumab-tnjn", "Bio-Thera/Sandoz", 2023, None),
        ),
    ),
    "FILGRASTIM": Family(
        "FILGRASTIM", "NEUPOGEN", "Amgen", "BLA103353", "subcutaneous",
        (
            Biosimilar("ZARXIO", "filgrastim-sndz", "Sandoz", 2015, "2015-09"),
            Biosimilar("NIVESTYM", "filgrastim-aafi", "Pfizer", 2018, "2018-10"),
            Biosimilar("RELEUKO", "filgrastim-ayow", "Kashiv/Amneal", 2022, "2022-10"),
            Biosimilar("NYPOZI", "filgrastim-txid", "Tanvex", 2024, None),
        ),
    ),
    "PEGFILGRASTIM": Family(
        "PEGFILGRASTIM", "NEULASTA", "Amgen", "BLA125031", "subcutaneous",
        (
            Biosimilar("FULPHILA", "pegfilgrastim-jmdb", "Biocon/Viatris", 2018, "2018-07"),
            Biosimilar("UDENYCA", "pegfilgrastim-cbqv", "Coherus", 2018, "2019-01"),
            Biosimilar("ZIEXTENZO", "pegfilgrastim-bmez", "Sandoz", 2019, "2019-11"),
            Biosimilar("NYVEPRIA", "pegfilgrastim-apgf", "Pfizer", 2020, "2020-12"),
            Biosimilar("FYLNETRA", "pegfilgrastim-pbbk", "Kashiv/Amneal", 2022, None),
            Biosimilar("STIMUFEND", "pegfilgrastim-fpgk", "Fresenius Kabi", 2022, "2023-01"),
        ),
    ),
    "EPOETIN ALFA": Family(
        "EPOETIN ALFA", "EPOGEN", "Amgen", "BLA103234", "intravenous/subcutaneous",
        (Biosimilar("RETACRIT", "epoetin alfa-epbx", "Pfizer/Hospira", 2018, "2018-11"),),
    ),
    "INSULIN GLARGINE": Family(
        "INSULIN GLARGINE", "LANTUS", "Sanofi", None, "subcutaneous",
        (
            Biosimilar("SEMGLEE", "insulin glargine-yfgn", "Biocon/Viatris", 2021, "2021-11", True),
            Biosimilar("REZVOGLAR", "insulin glargine-aglr", "Eli Lilly", 2021, "2023-04", True),
        ),
    ),
    "RANIBIZUMAB": Family(
        "RANIBIZUMAB", "LUCENTIS", "Genentech", None, "intravitreal",
        (
            Biosimilar("BYOOVIZ", "ranibizumab-nuna", "Samsung Bioepis/Biogen", 2021, "2022-06"),
            Biosimilar("CIMERLI", "ranibizumab-eqrn", "Coherus/Sandoz", 2022, "2022-10", True),
        ),
    ),
    "AFLIBERCEPT": Family(
        "AFLIBERCEPT", "EYLEA", "Regeneron", None, "intravitreal",
        (
            Biosimilar("YESAFILI", "aflibercept-jbvf", "Biocon", 2024, None),
            Biosimilar("OPUVIZ", "aflibercept-yszy", "Samsung Bioepis/Biogen", 2024, None),
            Biosimilar("AHZANTIVE", "aflibercept-mrbb", "Formycon", 2024, None),
            Biosimilar("ENZEEVU", "aflibercept-abzv", "Sandoz", 2024, None),
            Biosimilar("PAVBLU", "aflibercept-ayyh", "Amgen", 2024, "2024-10"),
        ),
    ),
    "TOCILIZUMAB": Family(
        "TOCILIZUMAB", "ACTEMRA", "Genentech", None, "intravenous/subcutaneous",
        (
            Biosimilar("TOFIDENCE", "tocilizumab-bavi", "Biogen/Bio-Thera", 2023, "2024-05"),
            Biosimilar("TYENNE", "tocilizumab-aazg", "Fresenius Kabi", 2024, "2024-04"),
            Biosimilar("AVTOZMA", "tocilizumab-anoh", "Celltrion", 2025, None),
        ),
    ),
    "USTEKINUMAB": Family(
        "USTEKINUMAB", "STELARA", "Janssen", None, "subcutaneous/intravenous",
        (
            Biosimilar("WEZLANA", "ustekinumab-auub", "Amgen", 2023, "2025-01", True),
            Biosimilar("SELARSDI", "ustekinumab-aekn", "Alvotech/Teva", 2024, "2025-02"),
            Biosimilar("PYZCHIVA", "ustekinumab-ttwe", "Samsung Bioepis/Sandoz", 2024, "2025-02"),
            Biosimilar("OTULFI", "ustekinumab-aauz", "Fresenius Kabi", 2024, None),
            Biosimilar("IMULDOSA", "ustekinumab-srlf", "Accord", 2024, None),
            Biosimilar("YESINTEK", "ustekinumab-kfce", "Biocon", 2024, "2025-02"),
            Biosimilar("STEQEYMA", "ustekinumab-stba", "Celltrion", 2024, "2025-03"),
        ),
    ),
    "DENOSUMAB": Family(
        "DENOSUMAB", "PROLIA", "Amgen", None, "subcutaneous",
        (
            Biosimilar("JUBBONTI", "denosumab-bbdz", "Sandoz", 2024, "2025-06", True),
            Biosimilar("OSPOMYV", "denosumab-dssb", "Samsung Bioepis", 2025, None),
            Biosimilar("STOBOCLO", "denosumab-bmwo", "Celltrion", 2025, None),
            Biosimilar("CONEXXENCE", "denosumab-bnht", "Fresenius Kabi", 2025, None),
        ),
    ),
    "ECULIZUMAB": Family(
        "ECULIZUMAB", "SOLIRIS", "Alexion/AstraZeneca", None, "intravenous",
        (
            Biosimilar("BKEMV", "eculizumab-aeeb", "Amgen", 2024, None),
            Biosimilar("EPYSQLI", "eculizumab-aagh", "Samsung Bioepis", 2024, None),
        ),
    ),
    "NATALIZUMAB": Family(
        "NATALIZUMAB", "TYSABRI", "Biogen", None, "intravenous",
        (Biosimilar("TYRUKO", "natalizumab-sztn", "Sandoz/Polpharma", 2023, None),),
    ),
}

#: Originator brands that have a second, closely related originator product
#: (not biosimilars) and must not be mis-attributed.
ORIGINATOR_ALIASES: Dict[str, Sequence[str]] = {
    "DENOSUMAB": ("PROLIA", "XGEVA"),
    "RITUXIMAB": ("RITUXAN", "RITUXAN HYCELA"),
    "TRASTUZUMAB": ("HERCEPTIN", "HERCEPTIN HYLECTA"),
    "EPOETIN ALFA": ("EPOGEN", "PROCRIT"),
    "INSULIN GLARGINE": ("LANTUS", "TOUJEO"),
    "ECULIZUMAB": ("SOLIRIS",),
}

_NON_ALNUM = re.compile(r"[^A-Z0-9\- ]+")


def normalise(name: Any) -> str:
    return _NON_ALNUM.sub(" ", str(name or "").upper()).strip()


def brand_index() -> Dict[str, Dict[str, str]]:
    """``{brand_or_suffixed_inn: {"inn", "product", "kind"}}`` for fast lookup."""
    idx: Dict[str, Dict[str, str]] = {}
    for inn, fam in FAMILIES.items():
        for alias in ORIGINATOR_ALIASES.get(inn, (fam.originator_brand,)):
            idx[normalise(alias)] = {"inn": inn, "product": fam.originator_brand, "kind": "originator"}
        for b in fam.biosimilars:
            idx[normalise(b.brand)] = {"inn": inn, "product": b.brand, "kind": "biosimilar"}
            idx[normalise(b.nonproprietary)] = {"inn": inn, "product": b.brand, "kind": "biosimilar"}
    return idx


_BRAND_INDEX = brand_index()


def attribute_drug_entry(entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Attribute one FAERS ``patient.drug[]`` entry to a product.

    Returns ``None`` when the entry is not in any catalogued family, else
    ``{"inn", "product", "kind"}`` with ``kind`` in
    {"originator", "biosimilar", "inn_only"}. Attribution order: biosimilar
    brand or suffixed INN anywhere in the brand/medicinal-product/generic
    fields, then originator brand, then bare INN (unattributable).
    """
    ofda = entry.get("openfda", {}) or {}
    candidates: List[str] = []
    for key in ("brand_name", "generic_name", "substance_name"):
        candidates.extend(ofda.get(key) or [])
    for key in ("medicinalproduct", "activesubstance"):
        v = entry.get(key)
        if isinstance(v, dict):
            v = v.get("activesubstancename")
        if v:
            candidates.append(str(v))
    norm = [normalise(c) for c in candidates if c]
    # 1) biosimilar brand / suffixed INN (token-level match inside longer strings)
    for kind_wanted in ("biosimilar", "originator"):
        for c in norm:
            for key, meta in _BRAND_INDEX.items():
                if meta["kind"] != kind_wanted:
                    continue
                if re.search(rf"(?<![A-Z0-9]){re.escape(key)}(?![A-Z0-9])", c):
                    return dict(meta)
    # 2) bare INN
    for c in norm:
        for inn in FAMILIES:
            if re.search(rf"(?<![A-Z0-9]){re.escape(inn)}(?![A-Z0-9\-])", c):
                return {"inn": inn, "product": inn, "kind": "inn_only"}
    return None


def attribute_report(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a FAERS report and attribute every catalogued biologic on it.

    Output keys: ``products`` (list of ``{"inn","product","kind","role"}``),
    ``inns`` (set of families present), plus the usual demographics and the
    PT list. ``manufacturer_names`` collects ``openfda.manufacturer_name`` for
    a secondary, sponsor-based attribution check.
    """
    patient = rec.get("patient", {}) or {}
    src = rec.get("primarysource", {}) or {}
    products: List[Dict[str, Any]] = []
    manufacturers: set = set()
    for d in patient.get("drug", []) or []:
        a = attribute_drug_entry(d)
        if a:
            a["role"] = str(d.get("drugcharacterization") or "")
            products.append(a)
        for m in (d.get("openfda", {}) or {}).get("manufacturer_name") or []:
            manufacturers.add(str(m).upper())
    reactions = sorted({r.get("reactionmeddrapt", "").upper() for r in (patient.get("reaction", []) or []) if r.get("reactionmeddrapt")})
    return {
        "safetyreportid": rec.get("safetyreportid"),
        "receivedate": rec.get("receivedate"),
        "reporttype": rec.get("reporttype"),
        "serious": rec.get("serious"),
        "seriousnessdeath": rec.get("seriousnessdeath"),
        "occurcountry": rec.get("occurcountry"),
        "qualification": str(src.get("qualification")) if src.get("qualification") is not None else None,
        "sex": SEX_LABELS.get(str(patient.get("patientsex")), "unknown"),
        "age_years": age_in_years(patient.get("patientonsetage"), patient.get("patientonsetageunit")),
        "products": products,
        "inns": sorted({p["inn"] for p in products}),
        "manufacturer_names": sorted(manufacturers),
        "reactions": reactions,
    }


def attributability_summary(reports: Iterable[Dict[str, Any]], inn: str) -> Dict[str, int]:
    """Count reports in a family by attribution kind (traceability outcome)."""
    out = {"originator": 0, "biosimilar": 0, "inn_only": 0, "mixed": 0}
    for r in reports:
        kinds = {p["kind"] for p in r.get("products", []) if p["inn"] == inn}
        if not kinds:
            continue
        if len(kinds) > 1:
            out["mixed"] += 1
        else:
            out[kinds.pop()] += 1
    return out


def launch_dates(inn: str) -> Dict[str, Optional[str]]:
    fam = FAMILIES[inn]
    return {b.brand: b.us_launch for b in fam.biosimilars}
