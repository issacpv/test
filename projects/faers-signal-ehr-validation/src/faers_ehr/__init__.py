"""faers_ehr: validating FAERS disproportionality signals against lab-defined EHR outcomes.

Modules
-------
openfda_client       rate-limit-aware, paginated openFDA drug/event client; 2x2 tables per drug x outcome
disproportionality   PRR, ROR, BCPNN IC/IC025, MGPS EBGM/EB05 with CIs; signal flags; PPV scoring
rxnorm               RxNav helpers: NDC -> RxCUI -> ingredient; MIMIC-IV prescriptions mapping notes
outcomes             MIMIC-IV lab-/ECG-defined adverse outcome SQL templates and incident-outcome logic
cohort               active-comparator new-user cohorts, propensity-score matching, empirical calibration
"""

from . import cohort, disproportionality, openfda_client, outcomes, rxnorm

__all__ = ["cohort", "disproportionality", "openfda_client", "outcomes", "rxnorm"]
__version__ = "0.1.0"
