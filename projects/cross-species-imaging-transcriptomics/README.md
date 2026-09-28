# cross-species-imaging-transcriptomics

**Do human imaging-transcriptomics associations (AHBA gene maps vs MRI maps) replicate in the mouse when the mouse side is measured with whole-brain MERFISH instead of ISH, and when both species get spatial-autocorrelation-preserving and gene-ensemble nulls? A cross-species replication panel with a composition-vs-within-type decomposition.**

## Status / difficulty / timeline / compute

- Status: design + starter code (regional aggregation of MERFISH cells, composition decomposition, 3-D variogram-matched surrogates, gene-ensemble nulls, cross-species replication statistics). No data is shipped.
- Difficulty: MSc-level to early PhD. The atlases are open and well documented; the work is in the ortholog mapping, the nulls, and honest reporting.
- Timeline: 6-9 months (1-2 months data assembly and registration checks, 2 months human-side associations with abagen/neuromaps, 2 months mouse side + nulls, 1-2 months decomposition and writing).
- Compute: a workstation with >= 64 GB RAM. The Allen MERFISH cell tables (about 4 million cells x 500 genes, plus the Zhuang datasets with 1,122 genes) fit in memory as sparse/HDF5; no GPU needed.

## Background

Imaging transcriptomics correlates a spatial brain map (cortical thickness, T1w/T2w, functional gradients, case-control atrophy) with regional gene expression from the Allen Human Brain Atlas (AHBA; Hawrylycz et al., 2012 Nature; six donors, microarray). Best-practice pipelines exist (abagen: Markello et al., 2021 eLife; Arnatkeviciute et al., 2019 NeuroImage), and the field has learned that naive correlations are inflated by spatial autocorrelation (spin tests: Alexander-Bloch et al., 2018 NeuroImage; variogram surrogates: Burt et al., 2020 NeuroImage; null-model comparison: Markello & Misic, 2021 NeuroImage; eigenstrapping: Koussis et al., 2024 Imaging Neuroscience) and that gene-category enrichment is inflated by gene-gene co-expression (Fulcher, Arnatkeviciute & Fornito, 2021 Nature Communications, who used the Allen Mouse Brain Atlas ISH data across 213 regions).

Cross-species work exists but is thin and ISH-based. Fulcher, Murray, Zerbi & Wang (2019 PNAS) showed that mouse cortical gradients (T1w:T2w, gene expression) correspond to human ones (rho about 0.44 for the T1w:T2w-expression coupling; per-gene mouse-human correspondence rho about 0.25 across 2,951 orthologs). Beauchamp et al. (2022 eLife) built a latent gene-expression space over 2,835 homologous genes to compare mouse and human isocortical subdivisions (sensorimotor regions more similar than supramodal). "Whole brain alignment of spatial transcriptomics between humans and mice" (Nature Communications, 2024) aligned mouse and human region-level expression. All of these use the mouse ISH atlas (Lein et al., 2007 Nature), whose per-gene maps are single-animal, semi-quantitative and inconsistently sectioned.

The Allen Brain Cell (ABC) Atlas changes the mouse side. Yao et al. (2023 Nature) released ~4.3 million MERFISH cells with a 500-gene panel registered to CCFv3 with a full cell-type taxonomy (34 classes, 338 subclasses, 5,322 clusters); Zhang et al. (2023 Nature; Zhuang lab) released ~9 million cells across 245 sections with 1,122 genes. Both provide cell-level expression with cell-type labels, so a regional expression map can be decomposed into a cell-type composition part and a within-type part, something neither AHBA nor ISH can do.

## The research gap

What has been done:

- Human-only imaging transcriptomics with corrected nulls (many papers 2019-2025; see the related project `imaging-transcriptomics-nulls` in this repository, which audits human null models).
- Mouse-human correspondence of expression gradients using ISH (Fulcher et al., 2019 PNAS; Beauchamp et al., 2022 eLife; Nature Communications 2024 alignment). These compare expression to expression, or expression to one MRI map (T1w:T2w), and they do not ask whether specific published human gene-map associations hold in mouse.
- Mouse gene-category enrichment nulls (Fulcher et al., 2021 Nat Commun) on ISH.
- MERFISH validation against ISH and scRNA-seq at cluster level (Yao et al., 2023; Zhang et al., 2023: median cluster-level correlation 0.91 between 10x and MERFISH), but no one has used MERFISH-derived regional maps as the mouse side of imaging transcriptomics.

What is specifically missing (our angle):

1. A **replication panel**: take the most-cited human AHBA-vs-MRI associations (T1w/T2w vs PVALB/SST and the "hierarchy" gene set, Burt et al., 2018 Nature Neuroscience; the principal functional gradient vs gene PC1; cortical thickness/myelin development gradients; disorder atrophy maps vs candidate gene sets), recompute each in the mouse with MERFISH-derived regional expression and mouse MRI maps, and report which replicate under species-appropriate spatial nulls.
2. A **measurement-modality comparison**: same genes, same regions, ISH vs MERFISH vs MERFISH-imputed (Yao et al. provide imputed whole-transcriptome values); how much of the mouse-human discordance is mouse measurement noise?
3. A **composition decomposition**: because MERFISH cells carry subclass labels, each regional gene map splits into (a) the part predicted by cell-type proportions and (b) the residual within-type expression. We test whether human associations replicate through composition (e.g., interneuron density) or through within-type expression, which is the mechanistic question people assume they are answering.
4. **Panel-aware enrichment**: the 500-gene MERFISH panel was chosen to discriminate cell types, so any gene-set test on it must use the panel (or the imputed transcriptome with a panel-membership covariate) as background. We quantify how much enrichment claims change under panel-aware vs genome-wide backgrounds.
5. A **conjunction (double-null) replication criterion**: an association "replicates" only if it passes the spatial null in both species and the gene-ensemble null in both species; we report replication rates with confidence intervals rather than anecdotes.

## Research questions / hypotheses

1. **RQ1 (replication rate).** What fraction of a pre-registered panel of ~40 published human gene-map associations replicate in mouse under the conjunction criterion? H1: fewer than half; associations with sensorimotor-dominant maps (T1w:T2w, myelin) replicate more often than association-cortex maps (functional gradients, disorder maps), consistent with Beauchamp et al. (2022).
2. **RQ2 (modality).** H2: MERFISH-derived regional maps agree better with human AHBA orthologs than ISH-derived maps do (higher median per-gene mouse-human rho), and replication rates are higher with MERFISH; the ISH-MERFISH within-mouse agreement bounds the achievable cross-species agreement.
3. **RQ3 (composition vs within-type).** H3: for at least two thirds of replicating associations, the mouse association is carried by the composition component; associations that survive in the within-type residual are a minority and enriched for ion-channel / synaptic genes rather than cell-type markers.
4. **RQ4 (panel bias).** H4: gene-set enrichment p-values computed on the 500-gene panel with a genome-wide background are inflated by more than an order of magnitude relative to the panel-aware ensemble null.
5. **RQ5 (null severity).** H5: 3-D variogram-matched surrogates on CCF region centroids are conservative relative to region-label permutation in mouse in the same way spin/eigenstrapping nulls are in human; the false-positive rate for random gene maps under each null is calibrated (about 5% at alpha = 0.05), whereas naive permutation gives more than 20%.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| Allen Brain Cell Atlas, MERFISH-C57BL6J-638850 (Yao et al., 2023) | Mouse cell-level expression (500-gene panel), CCFv3 coordinates and parcellation, subclass labels; imputed whole-transcriptome values | ~4 M cells; ~10-30 GB depending on files | Open (public AWS S3 bucket `allen-brain-cell-atlas`, no login) | https://alleninstitute.github.io/abc_atlas_access/ |
| ABC Atlas, Zhuang-ABCA-1..4 (Zhang et al., 2023) | Second MERFISH dataset (1,122 genes, 4 animals) for within-mouse replication | ~9 M cells | Open (same bucket) | https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html |
| Allen Mouse Brain Atlas ISH (Lein et al., 2007) | ISH expression energy per CCF structure for modality comparison | ~20k genes x 213+ structures (via API) | Open (Allen Brain Map API / AllenSDK) | https://mouse.brain-map.org/ |
| Allen CCFv3 annotation volume + structure ontology | Region centroids and hierarchy for nulls | ~100 MB | Open (AllenSDK) | https://atlas.brain-map.org/ |
| Allen Human Brain Atlas via abagen | Human regional expression (Desikan-Killiany, Schaefer, or custom parcellations) | ~4 GB raw | Open (abagen downloads) | https://abagen.readthedocs.io/ |
| neuromaps | Human MRI maps (T1w/T2w, functional gradient, thickness, PET receptor maps) | small | Open | https://netneurolab.github.io/neuromaps/ |
| Mouse MRI templates / maps: DSURQE (Mouse Imaging Centre), AMBMC, T1w:T2w cortical map from Fulcher et al. 2019 | Mouse imaging maps on CCF | ~1 GB | Open | https://wiki.mouseimaging.ca/ ; https://imaging.org.au/AMBMC/ |
| Mouse resting-state fMRI (Grandjean et al., 2020 NeuroImage multi-centre collection) | Mouse functional gradient map | ~10 GB | Open (OpenNeuro / authors' repository) | https://openneuro.org/ |
| Ensembl BioMart one-to-one mouse-human orthologs | Gene homology table | ~2 MB | Open (REST) | https://www.ensembl.org/biomart |
| Allen Mouse Brain Connectivity Atlas | Mouse structural-connectivity maps (optional) | ~1 GB (region-level) | Open (AllenSDK) | https://connectivity.brain-map.org/ |

## Methods

Pipeline (modules in `src/xspecies_it/`):

1. **Mouse regional expression** (`regional_expression.py`). Load MERFISH `cell_metadata` (with `parcellation_*` columns from the CCF registration) and the cell-by-gene matrix; log1p-normalise counts per cell (counts per 1e4 then log1p); aggregate to CCF structures at a chosen ontology depth (default: the 213 "summary structures" used by Fulcher et al., plus a cortical-area-only analysis). Require >= 100 cells per structure. Build the subclass composition matrix (structure x subclass proportions).
2. **Composition decomposition** (`regional_expression.decompose_composition`). Regress each gene's regional profile on the composition matrix (ridge, leave-one-structure-out to choose lambda); store fitted (composition) and residual (within-type) components and per-gene R^2.
3. **Human regional expression**. abagen with default best-practice settings (donor-specific probe selection by intensity-based filtering and differential stability; mirror-symmetric sampling; scaled robust sigmoid normalisation) on a parcellation matched in region count to the mouse analysis (e.g., Schaefer-200 cortex + Tian subcortex, or Desikan-Killiany + aseg).
4. **Orthologs** (`cross_species.align_orthologs`). One-to-one orthologs from BioMart; restrict to genes present in both (panel or imputed) and expressed above threshold.
5. **Associations**. For each human map M_h with a mouse analogue M_m (table of pairs in `data/README.md`), compute per-gene Spearman correlations with the map in each species (`cross_species.association_profile`), then the cross-species agreement of the gene-score vectors and the per-gene-set statistics for the published gene sets (hierarchy genes of Burt et al., 2018; interneuron markers; oligodendrocyte and myelin sets; disorder risk-gene sets from GWAS).
6. **Nulls** (`nulls.py`). (a) Spatial: 3-D variogram-matched surrogates (Burt et al., 2020 algorithm re-implemented for region centroids in CCF mm coordinates) for mouse; spin or eigenstrapping surrogates (neuromaps) for human cortex; variogram surrogates for human subcortex. (b) Gene-ensemble: random gene sets matched on mean expression and on spatial autocorrelation (Moran's I bin), following Fulcher et al. (2021). (c) Panel-aware: background restricted to the MERFISH panel, or imputed transcriptome with panel membership as a covariate.
7. **Replication criterion** (`cross_species.conjunction_p`). p_rep = max(p_spatial_h, p_spatial_m, p_ens_h, p_ens_m); an association replicates if p_rep < 0.05 after BH-FDR across the panel. Effect sizes with bootstrap CIs over regions.
8. **Modality comparison**. Per gene: rho(ISH, MERFISH) across structures; rho(ISH, AHBA) vs rho(MERFISH, AHBA); attenuation-corrected cross-species correlation (divide by the square root of within-mouse reliability).

Tools: pandas/pyarrow, h5py/anndata (MERFISH tables), abagen, neuromaps, AllenSDK (ISH and CCF), scipy, scikit-learn, statsmodels.

## Evaluation & statistics

- Region-level statistics use Spearman correlations; all p-values come from surrogate/permutation nulls (>= 5,000 surrogates), never from parametric formulas.
- Multiple comparisons: BH-FDR at q = 0.05 across the ~40-association panel; within-association gene-level tests are reported as gene-set statistics, not per-gene lists.
- Calibration check of nulls: 1,000 random genes as "maps" against each MRI map; report empirical false-positive rate per null type (H5).
- Leakage: the association panel and gene sets are fixed in a pre-registration before the mouse side is computed; ortholog filtering does not look at map correlations.
- Reliability: split-animal replication using the four Zhuang animals; ISH vs MERFISH agreement as the noise ceiling.
- Sensitivity: ontology depth (213 structures vs cortex-only 40 areas), parcellation choice in human, normalisation (log1p vs Pearson residuals), inclusion of imputed genes.

## Publishable angle

Headline: "Of N published human imaging-transcriptomics associations, only k replicate in mouse under matched nulls; most that do are explained by cell-type composition rather than within-type expression, and MERFISH raises the cross-species ceiling relative to ISH." A quantitative replication table with modality and composition annotations would be a reference for the imaging-transcriptomics community and for anyone using mouse expression to interpret human MRI.

Target venues: Nature Communications or PLOS Biology (cross-species biology); NeuroImage or Imaging Neuroscience (methods and nulls); Network Neuroscience; OHBM / Cosyne abstracts for early results.

Follow-ups: extend to macaque (ABC atlas cross-species taxonomy includes ~840k macaque cells); use MERFISH composition maps to build a "cell-type-deconvolved" human map via transfer of subclass signatures; a Python package that wraps the mouse-side nulls for the community.

## Risks, confounds & mitigations

- Region homology is imperfect (mouse has no dorsolateral prefrontal cortex). Mitigation: restrict the main analysis to regions with accepted homology (Beauchamp et al., 2022 correspondence table; sensorimotor, visual, auditory, cingulate, retrosplenial/posterior parietal, hippocampus, striatum, thalamus, cerebellum); report a sensitivity analysis with the machine-learned correspondences.
- MERFISH panel bias. Mitigation: panel-aware backgrounds (H4), imputed transcriptome with panel covariate, and Zhuang panel as a second panel.
- Registration error of MERFISH sections to CCF. Mitigation: use only cells with confident parcellation labels; compare against ISH at structure level; exclude structures thinner than the section spacing.
- AHBA covers six donors, mostly left hemisphere, adult. Mitigation: mirror sampling, donor-level leave-one-out, report donor consistency.
- Spatial nulls in 3-D for a non-convex region set are approximate. Mitigation: the variogram-matching quality (correlation between surrogate and empirical variograms) is reported and surrogates below 0.9 are discarded; compare to label permutation within hierarchy-matched strata as a second null.
- Different spatial scales (mm in mouse, cm in human). Mitigation: nulls are species-specific; compare only rank-based statistics; do not pool regions across species.

## Milestones

- [ ] Download MERFISH-C57BL6J-638850 metadata and cell-by-gene, ISH expression energies, CCFv3 ontology; verify `--sample` pipeline
- [ ] Ortholog table built and filtered; expression thresholds fixed
- [ ] Pre-registered association panel (OSF) with human map, mouse map, gene sets and prediction
- [ ] Human side reproduced with abagen for each association (effect size within reported CIs)
- [ ] Mouse regional maps (MERFISH, Zhuang, ISH, imputed) with composition matrices
- [ ] Nulls implemented and calibrated (H5 false-positive-rate table)
- [ ] Replication table with conjunction p-values and BH-FDR
- [ ] Composition vs within-type decomposition for replicating associations
- [ ] Sensitivity analyses (ontology depth, parcellation, panel)
- [ ] Manuscript + code release

## Ethics / data-use notes

- All datasets are open: Allen Institute data are released under the Allen Institute Terms of Use (citation required); AHBA via abagen requires citing Hawrylycz et al. (2012) and Markello et al. (2021); OpenNeuro mouse fMRI is CC0/PDDL.
- No human subject-level data beyond the six AHBA donors' anonymised expression are used; no credentialed data are involved.
- Never commit downloaded data; `.gitignore` excludes `data/` and `outputs/`. Publish only derived region-level tables.
- Animal data are secondary-use; cite the primary studies and their ethics approvals.
