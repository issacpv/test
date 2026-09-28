# Imaging-transcriptomics null-model benchmark

**One sentence:** A reproducibility benchmark that re-tests the classic Allen Human Brain Atlas vs cortical-map associations (and their gene-set enrichments) under a factorial of spatial nulls (spin variants, variogram surrogates, Moran spectral randomization), gene-set nulls (random-gene vs ensemble) and processing choices, then applies the resulting protocol to under-studied HCP maps (T1w/T2w myelin, HCP-Aging age-effect maps).

| | |
|---|---|
| Status | proposal + starter code (abagen wrapper, map loaders, four null families, ensemble-null enrichment) |
| Difficulty / timeline | MSc-level for the benchmark on open maps (6 months); PhD-paper scope with HCP-Aging maps and the protocol paper (9-12 months) |
| Compute | laptop/workstation; AHBA download 4 GB; nulls are cheap (10⁴ surrogates x 360 parcels in minutes); enrichment across ~10⁴ GO categories x 10⁴ surrogates is a few CPU-hours with the matrix formulation in `enrichment.py` |
| Package | `src/imgtx_nulls` |

## Background

Imaging transcriptomics correlates a cortical map (myelin, thickness change, connectivity gradient, MEG power ...) with regional gene expression from the six-donor Allen Human Brain Atlas (Hawrylycz et al., 2012, *Nature*) and interprets the correlated genes through category enrichment. The field has converged on best-practice processing (Arnatkeviciute et al., 2019, *NeuroImage*; Markello et al., 2021, *eLife*, abagen; Arnatkeviciute et al., 2023, *Biol Psychiatry*) and on two statistical corrections:

1. **Spatial nulls** for map-map or map-gene correlations, because both are spatially autocorrelated: spherical rotations (Alexander-Bloch et al., 2018, *NeuroImage*; Váša et al., 2018, *Cereb Cortex*), variogram-matched surrogates (Burt et al., 2020, *NeuroImage*, BrainSMASH), Moran spectral randomization (Wagner & Dray, 2015) and others, compared by Markello & Misic (2021, *NeuroImage*) and reviewed by Váša & Mišić (2022, *Nat Rev Neurosci*); implemented in neuromaps (Markello et al., 2022, *Nat Methods*).
2. **Ensemble-based gene-set nulls**, because some GO categories are enriched for almost any smooth map when tested against random genes (Fulcher, Arnatkeviciute & Fornito, 2021, *Nat Commun*); Wei et al. (2022, *Hum Brain Mapp*) evaluated spatial vs gene specificity tests, and a 2024 comparative analysis of null models and test statistics extended this.

Most of the classic findings that motivate the field (Table below) predate both corrections or used only one of them.

## The research gap

**What exists (2023-2026 check, Sept 2026).** Markello & Misic (2021) benchmarked spatial nulls on map-map correlations, not on gene enrichment. Fulcher et al. (2021) introduced ensemble nulls and showed inflated category false-positive rates, mainly in mouse and with one human example. Wei et al. (2022) and the 2024 comparative study evaluated test statistics and null families for gene specificity, on generic maps. A 2025 *Nat Commun* toolbox for surface-based transcriptomic decoding and a 2025 systematic review of 152 studies (84.9 % did not report background genes) document that practice is still heterogeneous. A 2026 *bioRxiv* preprint ("A significant enrichment that is not: spatial nulls, co-expression, and the imaging transcriptomics of EEG alpha-power genetics") shows in a single case that a spin-test-guarded enrichment dissolves under complementary nulls, and recommends reporting a gene-set null alongside the spatial null. A 2025 *Imaging Neuroscience* paper shows that the spherical projection itself changes spin-test outcomes, and eigenmode-based surrogates ("eigenstrapping", Koussis et al., 2024, *bioRxiv*) add another null family.

**What is missing.**
1. No study has systematically re-tested the *canonical* human findings (Table) under a full factorial of {spin-nearest, spin-Váša, variogram, Moran, eigenmode} x {random-gene, ensemble} x {parcellation: Glasser 360 / Schaefer 100-400 / Desikan} x {gene filter: none / DS > 0.1 / top-50 %} x {statistic: Pearson / Spearman, mean / mean|r|} and reported which survive. Individual replications exist but with one pipeline each.
2. Under-studied maps: the HCP T1w/T2w myelin map has been related to a transcriptomic hierarchy gradient (Burt et al., 2018, *Nat Neurosci*), but the enrichment side was tested with a random-gene null; HCP-Aging age-effect maps (thickness and myelin slopes across 36-100 y) have not been decoded transcriptomically with any null-controlled protocol.
3. There is no calibrated recommendation for *which pair* of spatial + gene nulls to report; the required decision rule (agreement across nulls? most conservative?) has not been evaluated by false-positive rate and power on planted signals.

## Benchmark of classic findings to be re-tested

| # | Finding (map ↔ transcriptome) | Paper | Original null |
|---|---|---|---|
| 1 | T1w/T2w myelin and FC gradient ↔ transcriptomic hierarchy PC1 (brain-specific genes) | Burt et al., 2018, *Nat Neurosci* | spatial permutation on surface (early spin) |
| 2 | Adolescent thickness change ↔ oligodendroglial/synaptic genes, schizophrenia risk genes | Whitaker et al., 2016, *PNAS* | random-gene GSEA |
| 3 | Structural covariance ↔ supragranular-enriched genes | Romero-Garcia et al., 2018, *NeuroImage* | random-gene + partial spin |
| 4 | HSE (supragranular) genes ↔ cortico-cortical network architecture | Krienen et al., 2016, *PNAS* | random gene sets |
| 5 | Correlated gene expression ↔ synchronous fMRI networks | Richiardi et al., 2015, *Science* | permutation of network labels (critiqued for distance confounds) |
| 6 | Functional network expression signatures across cortex & striatum | Anderson et al., 2018, *Nat Commun* | gene-permutation |
| 7 | Morphometric similarity ↔ cell-type marker genes | Seidlitz et al., 2018, *Neuron* | random-gene GSEA |
| 8 | MEG spectral power ↔ gene expression (dominant genes) | Hansen et al., 2021, *Nat Neurosci* | spin + cross-validation |
| 9 | Hub connectivity ↔ genes (human structural connectome) | Arnatkeviciute et al., 2021, *Nat Commun* | ensemble-style nulls |
| 10 | Sensorimotor-association axis ↔ gene expression PC1 | Sydnor et al., 2021, *Neuron* | spin |

Maps and gene lists are recreated from open data or the papers' supplements (`data/README.md`); when a map cannot be reconstructed openly it is dropped and reported as such.

## Research questions and hypotheses

1. **Spatial-null concordance.** H1: for map-gene correlations, the five spatial null families give p-values that differ by more than a factor of 10 for at least a quarter of benchmark genes with |r| in 0.3-0.5; variogram and Moran nulls are more conservative than nearest-neighbour spin for maps with long-range structure (gradients), and less conservative for patchy maps.
2. **Ensemble vs random-gene nulls.** H2: at least half of the enrichment claims in the Table lose significance (q > 0.05) under the ensemble null built from the same spatial surrogates, while the map-level correlations survive; categories that drop out are enriched in high-differential-stability, spatially smooth genes (replicating Fulcher et al., 2021 in humans).
3. **Processing sensitivity.** H3: parcellation resolution changes the effective sample size and flips significance for at least 3 of 10 findings; gene filtering by DS increases map-gene correlations but also increases false-positive category rates under random-gene nulls.
4. **Decision rule.** H4: on planted-signal simulations (synthetic maps from known gene sets plus SA-matched noise), "significant under both a spatial null and the ensemble null" keeps FPR ≤ 5 % while retaining ≥ 80 % power for effects r ≥ 0.3 at 360 parcels; "any null" does not control FPR, "all five" over-corrects.
5. **Under-studied maps.** H5: the HCP myelin map's association with the hierarchy gradient survives all nulls (it is a strong, large-scale effect), but its enrichments do not survive the ensemble null except for oligodendrocyte/myelination categories; HCP-Aging thickness age-slopes decode to categories distinct from cross-sectional thickness and overlap with Alzheimer's-risk categories only under the protocol of H4.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| Allen Human Brain Atlas microarray | regional expression (abagen) | 6 donors, 3,702 samples, ~20k genes; ~4 GB | open | https://human.brain-map.org (via `abagen`) |
| HCP S1200 group-average maps (myelin, thickness) | benchmark maps | 32k fsLR | open via neuromaps; ConnectomeDB free registration | https://netneurolab.github.io/neuromaps ; https://db.humanconnectome.org |
| Margulies 2016 FC gradient | benchmark map | 32k fsLR | open via neuromaps | as above |
| HCP MEG power maps | benchmark map (#8) | 4k fsLR | open via neuromaps | as above |
| HCP-Aging structural (thickness/myelin per subject) | age-effect maps | ~1,200 subjects | NDA Data Use Certification (collection 2847) | https://nda.nih.gov |
| Glasser MMP1.0 parcellation | 360-parcel maps | label GIFTIs | free registration (BALSA) | https://balsa.wustl.edu |
| Schaefer 2018 parcellations | resolution sweep | 100-1000 parcels | open | via `netneurotools` / `nilearn` |
| Gene Ontology + NCBI gene2go | gene sets | ~20k human annotations | open | http://geneontology.org ; https://ftp.ncbi.nlm.nih.gov/gene/DATA/ |
| Literature gene lists (HSE, brain-specific, cell-type markers) | benchmark categories | small | open (paper supplements) | see `data/README.md` |

## Methods

1. **Expression** (`expression.py`): abagen with recommended defaults (DS probe selection, bidirectional mirroring, SRS normalisation, `missing='interpolate'`); per-donor tables for `differential_stability`; three gene-filter levels.
2. **Maps** (`maps.py`): parcellate neuromaps annotations and CIFTI files onto Glasser/Schaefer; parcel centroids on the sphere and great-circle or `wb_command` geodesic distances.
3. **Spatial nulls** (`nulls.py`): pure-numpy spin (nearest, Váša), BrainSMASH-style variogram surrogates, Moran spectral randomization (singleton and pair), naive shuffle; each validated against neuromaps/brainsmash reference implementations on the same inputs (a deliverable: agreement plots of null distributions). Eigenstrapping via the authors' package as an additional family.
4. **Enrichment** (`enrichment.py`): gene scores as Spearman/Pearson correlations; category statistic mean r or mean |r|; random-gene null (size-matched resampling) and ensemble null (category scores recomputed for each surrogate map, single matrix product); BH-FDR; `category_false_positive_rate` to reproduce the Fulcher diagnostic on human data.
5. **Factorial benchmark**: 10 findings x 5 spatial nulls x 2 gene nulls x 3 parcellations x 3 gene filters x 2 statistics = 1,800 cells (each with 5,000 surrogates); results stored as a tidy table and shown as survival heatmaps.
6. **Simulation study**: planted gene-set maps (`simulate_expression`, `make_gene_sets`) with SA-matched noise to estimate FPR/power of each decision rule (H4).
7. **Application**: HCP myelin map and HCP-Aging age-effect maps run through the recommended protocol; report which categories survive.

### Decision rules evaluated (H4)

| Rule | A category "survives" if | Corresponds to |
|---|---|---|
| R1 spatial-only | map-level r significant under one spatial null, category significant under the random-gene null | most current practice |
| R2 gene-only | category significant under the random-gene null (no spatial null) | classic GSEA |
| R3 either | significant under the random-gene null *or* the ensemble null | permissive |
| R4 both (candidate protocol) | map-level r significant under a spatial null *and* category significant under the ensemble null built from the same surrogates | Fulcher et al. (2021) + 2026 preprint recommendation |
| R5 unanimous | R4 under all five spatial null families | most conservative |

Each rule is scored on simulated pure-noise SA maps (false-positive rate) and planted gene-set maps (power), then applied to the benchmark.

## Quick start (module API)

```python
import numpy as np, pandas as pd
from imgtx_nulls import compare_nulls, variogram_surrogates, enrichment_with_nulls
from imgtx_nulls.maps import load_parcellated_csv, align_map_to_expression

expr = pd.read_parquet("data/expression/glasser360_expression.parquet")      # regions x genes (abagen)
myelin = load_parcellated_csv("data/maps/hcps1200_myelinmap_glasser.csv")
y, E = align_map_to_expression(myelin, expr)
cen = pd.read_csv("data/parcellations/centroids_glasser.csv", index_col=0).loc[y.index]
lh = cen[cen.hemi == "L"][["x", "y", "z"]].to_numpy(); rh = cen[cen.hemi == "R"][["x", "y", "z"]].to_numpy()
D = np.load("data/parcellations/geodesic_glasser.npy")

# map-gene correlation under five nulls
print(compare_nulls(y.to_numpy(), E["PVALB"].to_numpy(), lh, rh, D=D, n_perm=5000))

# category enrichment with random-gene and ensemble (variogram-surrogate) nulls
surr = variogram_surrogates(y.to_numpy(), D, n_surr=5000)
enr = enrichment_with_nulls(E, y.to_numpy(), go_sets, null_maps=surr, n_gene_null=5000)
print(enr.query("q_gene < 0.05 and q_ensemble >= 0.05"))   # enrichments that do not survive the ensemble null
```

`nulls.neuromaps_nulls` / `nulls.brainsmash_surrogates` give the reference implementations for the validation step.

## Evaluation and statistics

- p-values with the +1 permutation correction; 5,000 surrogates per null (10,000 for the final protocol); FDR across categories within each analysis; findings declared "surviving" if q < 0.05 under the decision rule being evaluated.
- Concordance between null families quantified by log10-p Bland-Altman plots and by the proportion of categories whose significance flips.
- Simulation-based calibration: FPR at nominal 5 % under pure-noise SA maps (target ≤ 5 %), power curves vs planted effect size and parcel count.
- Leakage/independence: donors treated as the unit for expression uncertainty (leave-one-donor-out sensitivity); left-hemisphere-only vs mirrored analyses reported separately; no map is used to select genes that are then tested on the same map without the null accounting for it.
- Robustness: repeat with two abagen versions/settings (interpolate vs no missing fill) and with the Desikan atlas to match the original papers' resolution.

## Publishable angle

**Headline result:** a survival matrix of ten classic imaging-transcriptomics findings under five spatial nulls and two gene-set nulls, showing that map-level correlations mostly survive but a large share of published enrichment stories do not, together with a simulation-calibrated recommendation ("report spin/variogram for the map, ensemble null for categories, and require both") and a first null-controlled transcriptomic decoding of HCP myelin and HCP-Aging age-effect maps.

Target venues: *Nature Communications* or *PLOS Biology* (benchmark + protocol), *Imaging Neuroscience*, *NeuroImage*; a shorter methods note in *Network Neuroscience*. OHBM/Neuromatch talks.

Follow-ups: (i) extend to subcortex and cerebellum (fewer AHBA samples, different nulls); (ii) test cell-type deconvolution maps (Seidlitz et al., 2020) under the same protocol; (iii) a `pip`-installable `imgtx_nulls` with the ensemble null as a drop-in for existing GSEA toolboxes; (iv) apply to disease maps from ENIGMA.

## Risks, confounds and mitigations

| Risk | Mitigation |
|---|---|
| Original maps not openly reconstructible (NSPN adolescent change, some structural covariance) | use authors' supplementary parcel tables or drop with explicit statement; the benchmark reports coverage |
| Right-hemisphere expression from only two donors | left-only primary analysis; mirrored as sensitivity |
| Great-circle vs geodesic distances change variogram nulls | compute exact geodesics with `wb_command`; report both |
| Spin tests sensitive to spherical projection (2025 result) | include projection variants (sphere vs inflated-sphere from fsaverage/fsLR) as a factor for two findings |
| Categories with few genes unstable | minimum size 10, maximum 500, report size-stratified results |
| Multiple testing across the factorial makes "any survival" trivial | pre-registered decision rules; simulation-calibrated FPR |
| Compute for 1,800 cells x 5,000 surrogates | matrix formulation (`_scores_matrix`) makes enrichment O(genes x surrogates) per cell; cache surrogates per (map, null, parcellation) |

## Milestones

- [ ] AHBA + parcellations + geometry; abagen tables for Glasser/Schaefer/Desikan (month 1)
- [ ] Validate pure-numpy nulls against neuromaps/brainsmash (agreement figures) (month 1-2)
- [ ] Reconstruct benchmark maps and gene lists; document what is not reconstructible (month 2-3)
- [ ] Factorial benchmark on map-level correlations (H1, H3) (month 3-4)
- [ ] Ensemble-null enrichment across the benchmark (H2) (month 4-5)
- [ ] Simulation calibration of decision rules (H4) (month 5-6)
- [ ] HCP myelin + HCP-Aging age-effect decoding (H5; NDA approval in parallel) (month 6-9)
- [ ] Manuscript, released surrogate caches and survival table (month 9-12)

## Ethics and data-use notes

- AHBA, neuromaps annotations, GO and Schaefer parcellations are open; HCP S1200 group maps require ConnectomeDB terms; Glasser MMP1.0 requires BALSA registration; HCP-Aging requires an NDA DUC and participant-level HCP-A data must stay in the approved environment (only group-level parcel maps enter this repository's `data/`, which is git-ignored anyway).
- No participant-level data are sent to third-party services or LLM APIs.
- The benchmark is a reproducibility audit; report results per finding without singling out authors, and share full pipelines so that disagreements can be traced to specific choices.
