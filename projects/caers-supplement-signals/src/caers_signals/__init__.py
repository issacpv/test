"""caers_signals: ingredient-level dietary-supplement signals from CAERS (openFDA food/event).

Modules
-------
openfda    rate-limited openFDA client with cursor pagination (food/event, drug/event)
normalize  CAERS flattening, product-name normalisation, ingredient lexicon,
           seriousness coding, DSLD API lookup, FAERS supplement flattening
signals    ROR / PRR / IC screen with BH, seriousness model, monthly series,
           segmented Poisson ITS, Poisson CUSUM, CAERS-FAERS concordance,
           reference-set AUROC
"""

from .normalize import INDUSTRY_CODES, INGREDIENT_LEXICON, SUPPLEMENT_INDUSTRY_CODE, dsld_ingredients_for_products, dsld_search, flatten_caers, flatten_faers_for_supplements, map_ingredients, normalise_product, outcome_flags
from .openfda import OpenFDAClient, read_jsonl, write_jsonl
from .signals import HEPATIC_PTS, HEPATOTOXIC_INGREDIENTS, cross_system_concordance, information_component, monthly_series, poisson_cusum, prr, reference_evaluation, ror, screen, segmented_poisson_its, serious_fraction_model

__all__ = [
    "OpenFDAClient",
    "read_jsonl",
    "write_jsonl",
    "INDUSTRY_CODES",
    "INGREDIENT_LEXICON",
    "SUPPLEMENT_INDUSTRY_CODE",
    "normalise_product",
    "map_ingredients",
    "outcome_flags",
    "flatten_caers",
    "flatten_faers_for_supplements",
    "dsld_search",
    "dsld_ingredients_for_products",
    "ror",
    "prr",
    "information_component",
    "screen",
    "serious_fraction_model",
    "monthly_series",
    "segmented_poisson_its",
    "poisson_cusum",
    "cross_system_concordance",
    "reference_evaluation",
    "HEPATOTOXIC_INGREDIENTS",
    "HEPATIC_PTS",
]

__version__ = "0.1.0"
