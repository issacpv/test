"""xspecies_pv: cross-species pharmacovigilance signals (openFDA CVM animal reports vs FAERS).

Modules
-------
openfda_animal         client + flatteners for animalandveterinary/event and drug/event
term_mapping           ingredient normalisation, organ-system harmonisation of VeDDRA/MedDRA terms
cross_species_signals  BCPNN signal tables per species, concordance, lead-lag, dose scaling
"""

from .cross_species_signals import (
    bcpnn_ic,
    concordance,
    first_sustained_signal,
    lead_lag,
    mg_per_kg,
    permutation_null_rho,
    signal_table,
)
from .openfda_animal import OpenFDAAnimalClient, flatten_animal_record, flatten_faers_record
from .term_mapping import (
    ORGAN_SYSTEMS,
    assign_organ_system,
    normalize_ingredient,
    suggest_pt_matches,
)

__all__ = [
    "OpenFDAAnimalClient", "flatten_animal_record", "flatten_faers_record",
    "ORGAN_SYSTEMS", "assign_organ_system", "normalize_ingredient", "suggest_pt_matches",
    "bcpnn_ic", "signal_table", "concordance", "permutation_null_rho", "first_sustained_signal",
    "lead_lag", "mg_per_kg",
]

__version__ = "0.1.0"
