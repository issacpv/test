"""tau_proxy: PET-free tau-positivity proxies from structural MRI.

Modules
-------
tables       OASIS-3 ID parsing, PET-MR-clinical matching, synthetic sample tables
labels       tau composites, binary positivity, ordinal T2 stage, N status
features     FreeSurfer stats parsing, composites, residualization, ComBat
models       covariate baseline vs ROI models with nested grouped CV, bootstrap deltas
discordance  T/N quadrant analysis
decision     decision curves and scans-avoided policy analysis
"""

__all__ = ["tables", "labels", "features", "models", "discordance", "decision"]
__version__ = "0.1.0"
