"""imgtx_nulls: null-model benchmark for imaging transcriptomics.

Modules
-------
expression
    abagen wrapper for parcellated Allen Human Brain Atlas expression, gene
    filtering by differential stability, and a synthetic generator.
maps
    Loaders for HCP parcellated cortical maps (Glasser MMP1.0 / Schaefer) from
    CIFTI, neuromaps annotations, or plain CSV, plus parcel centroid helpers.
nulls
    Spatial nulls: parcel-level spin tests (nearest-neighbour and Vasa
    non-duplicating variants), BrainSMASH-style variogram-matched surrogates,
    Moran spectral randomization, and naive permutation; wrappers for
    neuromaps / brainsmash when installed.
enrichment
    Gene-set enrichment with random-gene nulls and ensemble (spatial) nulls
    after Fulcher et al. (2021), with BH-FDR.
"""

from .expression import (  # noqa: F401
    fetch_parcellated_expression, differential_stability, simulate_expression,
)
from .maps import (  # noqa: F401
    parcellate_vector, load_parcellated_csv, fibonacci_sphere, split_hemispheres,
)
from .nulls import (  # noqa: F401
    spin_permutations, variogram_surrogates, moran_spectral_randomization, moran_i,
    null_pvalue, compare_nulls,
)
from .enrichment import (  # noqa: F401
    gene_scores, enrichment_with_nulls, bh_fdr,
)

__version__ = "0.1.0"
