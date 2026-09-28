"""lifespan_norm: lifespan normative models for morphometry and diffusion metrics, and a centile
transportability audit.

Modules
-------
harmonize         Multi-dataset phenotype harmonizer (HCP-YA, NDA/HCP Lifespan, OASIS-3, BIDS) to one schema;
                  FreeSurfer / tract table readers.
normative         Quantile-regression (GAMLSS-like) and heteroscedastic-Gaussian normative models with a
                  natural-spline age basis, sex and site effects; site adaptation; pcntoolkit hook.
centiles          Centile/z conversion, extreme-deviation counts, uniformity checks, group contrasts, ICC.
site_sensitivity  Leave-one-site-out transport, adaptation budgets, reference-model agreement,
                  clinical detection in- vs out-of-reference.
"""
from . import centiles, harmonize, normative, site_sensitivity  # noqa: F401

__all__ = ["harmonize", "normative", "centiles", "site_sensitivity"]
__version__ = "0.1.0"
