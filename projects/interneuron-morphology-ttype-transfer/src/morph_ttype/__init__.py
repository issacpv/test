"""morph_ttype: morphology-only transcriptomic subclass prediction and cross-species transfer.

Modules
-------
swc       : SWC parsing.
features  : hand-crafted morphometrics, depth-normalised density maps, persistence-style
            branch summaries, and scale-normalised variants.
taxonomy  : t-type name -> harmonised cross-species subclass.
transfer  : per-dataset standardisation, CORAL alignment, within/leave-dataset-out evaluation,
            permutation nulls and ablations.
"""

__all__ = ["swc", "features", "taxonomy", "transfer"]
__version__ = "0.1.0"
