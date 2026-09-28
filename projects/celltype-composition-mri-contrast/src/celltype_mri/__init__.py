"""celltype_mri: which cell types explain regional MRI contrast in the mouse brain.

Modules
-------
abc_atlas      : MERFISH cell table -> per-structure counts / densities (class, subclass); voxel density maps.
mri            : regional contrast extraction from CCF-registered volumes; T1w:T2w; z-scoring.
regress        : ridge / PLS with leave-region-out CV, relative importance, nested model comparison,
                 cross-atlas concordance.
spatial_nulls  : Moran's I, Moran spectral randomisation surrogates, permutation p-values.
"""

__all__ = ["abc_atlas", "mri", "regress", "spatial_nulls"]
__version__ = "0.1.0"
