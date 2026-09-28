"""bayes_pk: Bayesian forecasting of vancomycin / aminoglycoside exposure from real-world MIMIC-IV dosing records.

Modules
-------
dosing_records   Build dose / level / course tables from MIMIC-IV ``inputevents``, ``emar`` and ``labevents``;
                 synthetic course simulator.
pk_models        Analytic 1- and 2-compartment infusion models, ``ModelSpec`` container, AUC integration.
bayes_forecast   MAP-Bayesian individual estimation, next-level forecasting, OFV-weighted model averaging.
evaluation       rBias / rRMSE / MPE with cluster bootstrap, target attainment, timing-perturbation experiment,
                 renal-function equations.
"""

from . import bayes_forecast, dosing_records, evaluation, pk_models  # noqa: F401

__all__ = ["bayes_forecast", "dosing_records", "evaluation", "pk_models"]
__version__ = "0.1.0"
