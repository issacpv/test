"""fm_clinical_val: stratified external validation of brain-MRI foundation models.

Modules
-------
degradations  synthetic image corruptions (noise, k-space motion, bias field, downsampling, ghosting) + phantom
metrics       Dice / HD95 / ASSD / volume error with paired statistics and the silver-label ceiling
stratify      quality & pathology strata, robustness slopes, learning curves and crossing points
registry      model-adapter protocol, baseline adapters and CLI hooks for SynthSeg / nnU-Net
"""

__all__ = ["degradations", "metrics", "stratify", "registry"]
__version__ = "0.1.0"
