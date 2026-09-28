# gwas-celltype-enrichment

**Which spatially-resolved brain cell types carry the genetic signal of imaging-derived brain phenotypes?**
Cell-type and spatial-domain enrichment (MAGMA / S-LDSC / scDRS-style) of psychiatric, neurological
and — the under-explored part — *neuroimaging-derived* GWAS (cortical thickness, surface area, DTI FA/MD,
subcortical volumes from UK Biobank BIG40) against the Allen Brain Cell Atlas (mouse 10x + MERFISH),
the Siletti et al. human brain atlas, and the Allen Human Brain Atlas, with an explicit mouse-vs-human
resolution comparison.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No results yet.
- Difficulty: MSc thesis or first-year PhD project; 6–9 months for the core paper.
- Compute: a workstation with 64 GB RAM is enough for pseudobulk-level analyses. Per-cell scoring
  (scDRS-style) on the 4-million-cell MERFISH dataset or the 3.3-million-nucleus human atlas needs
  ~128 GB RAM or chunked/backed `anndata` reads (implemented in `gwas_ct.abc_loader`).
  MAGMA and S-LDSC are CPU-bound; ~1–3 CPU-hours per GWAS × annotation set. Total budget for
  ~60 GWAS × ~600 annotations is a few hundred CPU-hours.
- Storage: ~150 GB if you download the full ABC Atlas expression matrices; ~15 GB for the
  metadata + MERFISH-only subset used first.

## Background

GWAS of brain disorders and traits are highly polygenic, so a standard way to make them mechanistically
interpretable is to ask which *cell types* are enriched for heritability. The tools are mature:
MAGMA gene-property analysis (de Leeuw et al., 2015, PLoS Comput Biol), stratified LD-score regression
on specifically-expressed genes (Finucane et al., 2018, Nat Genet), EWCE bootstrap enrichment
(Skene et al., 2018, Nat Genet) and per-cell polygenic scoring with scDRS (Zhang et al., 2022, Nat Genet).
What changed in 2023–2025 is the reference data: the Allen Brain Cell (ABC) Atlas provides a
~4-million-cell whole-mouse-brain 10x taxonomy with 34 classes / 338 subclasses / 1,201 supertypes /
5,322 clusters (Yao et al., 2023, Nature) plus a 500-gene MERFISH dataset of ~4 million cells registered
to CCFv3 (Zhang et al., 2023, Nature; Allen MERFISH-C57BL6J-638850), and the human side has the
~3.3-million-nucleus Siletti et al. (2023, Science) atlas (also redistributed as ABC `WHB-10Xv3`).
For the first time one can ask not only *which cell type* but *which cell type in which anatomical
domain* is enriched, and compare the mouse and human answers at matched taxonomic depth.

## The research gap

**What has been done.**

- Cell-type prioritisation for psychiatric/neurological GWAS using mouse and human scRNA-seq at the
  cluster level: Skene et al. (2018, Nat Genet); Bryois et al. (2020, Nat Genet); Watanabe et al.
  (2019, Nat Commun, FUMA); Jagadeesh et al. (2022, Nat Genet, sc-linker).
- Psychiatric GWAS × the Siletti human atlas, with extrapolation from cell types to dissected brain
  regions and comparison to fMRI connectivity: Zeng et al. (2025, Nat Commun, "Connecting genomic
  results for psychiatric disorders to human brain cell types and regions...").
- Spatially-resolved GWAS mapping: gsMap (Song et al., 2025, Nature) maps trait-associated cells in
  spatial transcriptomics; the Spatial GWAS Atlas (Nucleic Acids Res, 2026 database issue; 3,854 GWAS ×
  635 ST datasets) and Spatial2GWAS (Nucleic Acids Res, 2026 database issue) are knowledge bases
  built on this idea. These are mostly disease/behavioural traits and use gsMap's own score; they do
  not test anatomical concordance with human imaging phenotypes.
- Method benchmarking: Li et al. (2025, medRxiv, "Benchmarking methods integrating GWAS and
  single-cell transcriptomic data...") evaluated 19 pipelines and recommended Cepo-based
  specificity plus a Cauchy combination of SC→GWAS and GWAS→SC strategies.
- Imaging GWAS: BIG40 released summary statistics for 3,935 IDPs (Smith et al., 2021, Nat Neurosci;
  Elliott et al., 2018, Nature). ENIGMA cortical GWAS (Grasby et al., 2020, Science) and the 2,347-IDP
  cortical study (Warrier et al., 2023, Nat Genet) ran cell-type enrichments only against
  *developmental / fetal* cortical cell types or bulk tissue; DTI GWAS (Zhao et al., 2021, Science) did
  tissue-level enrichment. None used the ABC Atlas MERFISH spatial domains or the adult whole-brain
  taxonomies, and none asked whether the enrichment is anatomically specific.

**What is missing (the angle of this project).**

1. *Anatomical concordance test.* An IDP is anatomically localised (e.g. thickness of the human
   precentral gyrus, FA of the corticospinal tract). With MERFISH cells registered to CCFv3 we can
   compute cell-type × spatial-domain specificity and test whether the GWAS of a *regional* IDP is
   enriched preferentially in cell types *of the homologous mouse region* (via a cortex/subcortex
   homology table, e.g. the Beauchamp et al., 2022, eLife mouse–human spatial-transcriptomic
   correspondence) versus the same cell type elsewhere. This is a falsifiable spatial prediction that
   no published IDP enrichment study has made.
2. *Resolution ablation, mouse vs human.* Run every enrichment at class → subclass → supertype →
   cluster (mouse) and superclass → cluster (human) and quantify where signals saturate, where they
   split, and whether mouse cluster-level results survive orthology mapping to human clusters. IDPs
   have lower polygenicity and SNP-heritability than schizophrenia, so power at fine resolution is a
   real open question.
3. *Estimator reliability for IDPs.* MAGMA gene-property, S-LDSC, EWCE bootstrap and scDRS-style
   per-cell scoring disagree in known ways; Li et al. (2025) benchmarked them on disease traits.
   We will quantify concordance specifically for imaging traits and report a consensus (Cauchy-combined)
   ranking.

If a 2026 paper appears that performs gsMap on BIG40 IDPs, the anatomical-concordance test (1) and the
resolution comparison (2) remain the differentiators; sharpen towards them.

## Research questions / hypotheses

1. **H1 (cell class).** Cortical-thickness and surface-area IDP GWAS are enriched in excitatory
   glutamatergic neuron subclasses (IT/ET/CT) and, for surface area, in radial-glia-like /
   progenitor-related gene programs, whereas DTI FA/MD IDPs are enriched in oligodendrocyte lineage
   (OPC, Oligo) cell types. Test: MAGMA gene-property + S-LDSC on top-decile specificity genes,
   FDR < 0.05 across (trait × cell type).
2. **H2 (spatial concordance).** For regional cortical IDPs, enrichment in a cell type is stronger
   when specificity is computed from MERFISH cells in the homologous CCF region than from the same
   cell type in non-homologous regions (paired comparison across cell types; one-sided Wilcoxon).
3. **H3 (resolution).** The number of significant cell types grows from class to subclass and then
   plateaus or declines at supertype/cluster level for IDPs, unlike schizophrenia which keeps gaining
   signal at cluster level (interaction test: trait-class × level on −log10 p).
4. **H4 (species).** Mouse (ABC) and human (Siletti) cell-type rankings agree at the subclass level
   (Spearman ρ > 0.5 across matched subclasses) but not at cluster level for IDPs.
5. **H5 (estimator concordance).** Rank agreement between MAGMA, S-LDSC and scDRS-style scores is lower
   for IDPs than for psychiatric traits, and the Cauchy-combined score has higher replication across
   BIG40 discovery vs replication summary statistics than any single estimator.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Brain Cell Atlas, WMB-10Xv3 / WMB-10Xv2 / WMB-10XMulti | Whole mouse brain scRNA-seq, cluster annotations (class/subclass/supertype/cluster), pseudobulk per cluster | ~4 M cells; 60+ GB h5ad | Open (public S3, no login) | https://alleninstitute.github.io/abc_atlas_access/ ; s3://allen-brain-cell-atlas/ |
| ABC Atlas MERFISH-C57BL6J-638850 (+ `-CCF`) | 500-gene MERFISH, ~4 M cells with CCFv3 parcellation labels; spatial-domain specificity | 7.6 GB h5ad + 0.7 GB metadata | Open | same manifest, datasets `MERFISH-C57BL6J-638850`, `MERFISH-C57BL6J-638850-CCF` |
| ABC Atlas Zhuang-ABCA-1..4 | Independent whole-brain MERFISH (1,122 genes) for replication of spatial results | ~3 M cells | Open | same manifest |
| ABC Atlas WHB-10Xv3 (Siletti et al. 2023 redistribution) | Human whole-brain snRNA-seq, 31 superclusters / 461 clusters, dissection ROI per cell | 3.3 M nuclei; ~40 GB | Open | same manifest, dataset `WHB-10Xv3`, `WHB-taxonomy` |
| Siletti et al. 2023 on CELLxGENE | Same data, curated h5ad per dissection; alternative download | 3.3 M nuclei | Open (CZ CELLxGENE) | https://cellxgene.cziscience.com/ (collection "Transcriptomic diversity of cell types across the adult human brain") |
| Allen Human Brain Atlas (microarray) | Regional bulk expression for region-level sanity checks via `abagen` | 6 donors, ~3,700 samples | Open | https://human.brain-map.org/ ; `abagen` |
| UK Biobank BIG40 | Summary statistics for 3,935 IDPs (discovery n≈22k; replication n≈11k) | ~4k files × ~300 MB | Open (no registration) | https://open.oxcin.ox.ac.uk/ukbiobank/big40/ |
| PGC GWAS (SCZ3, BD, MDD, ADHD, ASD, PTSD, ...) | Psychiatric summary statistics as positive controls | 10–15 traits | Open (click-through agreement) | https://pgc.unc.edu/for-researchers/download-results/ |
| GWAS Catalog summary statistics | AD (Bellenguez 2022), PD (Nalls 2019), stroke, epilepsy, intelligence, etc. | ~20 traits | Open (FTP / REST) | https://www.ebi.ac.uk/gwas/summary-statistics ; https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/ |
| 1000 Genomes EUR reference; MAGMA gene locations; LDSC baseline-LD v2.2 | LD reference / gene annotation | ~5 GB | Open | https://cncr.nl/research/magma/ ; https://github.com/bulik/ldsc |
| Ensembl BioMart / MGI orthology | Mouse→human one-to-one orthologs | small | Open | https://www.ensembl.org/biomart |

## Methods

1. **Reference construction** (`gwas_ct.abc_loader`, `gwas_ct.specificity`)
   - Pull the ABC manifest (`releases/20241115/manifest.json`), download `cell_metadata_with_cluster_annotation`
     and expression matrices (`log2` variant) per dissection region; MERFISH `cell_metadata_with_parcellation_annotation`
     for CCF labels (`parcellation_division` / `_structure` / `_substructure`).
   - Pseudobulk mean expression per cluster; aggregate to supertype / subclass / class with cell-count weights.
   - EWCE-style specificity: `spec[g, c] = mean_expr[g, c] / sum_c mean_expr[g, c]`; top-decile gene sets per
     cell type; 40-quantile continuous specificity for MAGMA gene-property.
   - Spatial-domain specificity from MERFISH: per (subclass × CCF division) and per division alone
     (500 genes only → used for the *anatomical* test, with the 10x-imputed MERFISH matrix
     `MERFISH-C57BL6J-638850-imputed` as a genome-wide extension).
   - Mouse→human orthologs (one-to-one only); human specificity from `WHB-10Xv3` superclusters/clusters.
2. **GWAS preprocessing** — harmonise to hg19/hg38 rsIDs, `munge_sumstats.py`, MAGMA SNP-to-gene
   annotation (35 kb up / 10 kb down window), gene-level Z (MAGMA `--gene-model snp-wise=mean`).
3. **Enrichment estimators** (`gwas_ct.magma_io`, `gwas_ct.enrichment`, `gwas_ct.scdrs_lite`)
   - MAGMA gene-set (top decile) and gene-property (continuous specificity, `--gene-covar ... condition-hide Average`).
   - S-LDSC: `make_annot.py` on top-decile genes ±100 kb, conditioned on baseline-LD v2.2 and an
     all-genes annotation; report enrichment coefficient τ* and p.
   - EWCE-style bootstrap (expression-matched random gene sets) — implemented here in numpy.
   - scDRS-lite per-cell scoring with expression-bin-matched control gene sets and cluster-level
     association z (`gwas_ct.scdrs_lite`), run on backed anndata chunks.
   - Cauchy combination across estimators (as in Li et al. 2025) for the consensus ranking.
4. **Anatomical concordance** — for each regional IDP, define the homologous CCF division set; compute
   enrichment with "in-homolog" vs "out-of-homolog" domain-restricted specificity; paired test.
5. **Baselines / controls** — schizophrenia (Trubetskoy et al., 2022, Nature), bipolar (Mullins et al.,
   2021, Nat Genet), Alzheimer (Bellenguez et al., 2022, Nat Genet), Parkinson (Nalls et al., 2019,
   Lancet Neurol), plus negative controls (height, type-2 diabetes) which should not enrich in neurons.

Tooling: `magma` v1.10, `ldsc`, `scdrs` (for cross-checking against the lite implementation),
`EWCE`/`MAGMA_Celltyping` (R, optional), `anndata`, `scanpy`, `abc_atlas_access`, `abagen`.

## Evaluation & statistics

- Primary unit of inference: (GWAS × annotation) pair. Multiple testing: Benjamini–Hochberg within
  each estimator across all pairs at a given resolution level; Bonferroni across levels for H3.
- Nulls: (a) EWCE bootstrap with expression-level matching (10,000 draws); (b) S-LDSC conditional on
  baseline-LD; (c) label-permutation null for spatial-domain tests (permute CCF division labels within
  subclass so that cell-type composition is preserved; 1,000 permutations).
- Replication: BIG40 discovery (n≈22k) vs replication (n≈11k) summary statistics — an enrichment is
  called replicated if p < 0.05 in replication with the same sign.
- Leakage: do not choose gene windows or resolution after seeing IDP results; pre-register the
  homology table and the analysis plan (OSF) before running IDPs; psychiatric traits serve as the
  tuning set only.
- Estimator agreement: Spearman ρ of −log10 p rankings and top-10 Jaccard, per trait.
- Sensitivity: window size (10/35 kb vs 100 kb), specificity definition (EWCE vs Cepo), MHC
  exclusion, N-weighted vs unweighted pseudobulk.

## Publishable angle

Headline result: a brain-wide, spatially-resolved map of which adult cell types carry the heritability
of ~150 regional imaging phenotypes, with an explicit test that regional IDP genetics localise to the
homologous anatomical domain's cell types — plus a quantitative statement of how much taxonomic
resolution IDP GWAS can actually support in mouse vs human.

Target venues: *Nature Communications*, *Biological Psychiatry* (imaging-genetics scope), *Imaging
Neuroscience*, *Genome Biology*; methods sub-paper at *Bioinformatics* or *NAR Genomics and Bioinformatics*.

Follow-ups: integrate ENIGMA and ABCD imaging GWAS as an out-of-cohort replication; extend to
snATAC-seq (chromatin) cell-type annotations; transcriptome-imputed (TWAS) IDPs per cell type.

## Risks, confounds & mitigations

| Risk | Mitigation |
|---|---|
| IDP GWAS underpowered (h² ~ 0.1–0.3, n≈33k) → few significant cell types | Focus on the ~200 IDPs with h² z > 7; use consensus scores; report calibrated nulls, not just significance counts |
| MERFISH panel (500 genes) too small for genome-wide enrichment | Use the ABC-imputed MERFISH matrix for gene sets; MERFISH only defines spatial *domains* and cell-type labels |
| Mouse–human region homology is contested for association cortex | Restrict H2 to primary sensory/motor cortex, hippocampus, thalamus, striatum; pre-registered table; sensitivity to an alternative table |
| Cell-type composition drives spatial "enrichment" | Permute domain labels within cell type; report domain-effects conditional on subclass |
| Dissection-driven batch in Siletti data | Use dissection ROI as a covariate; compare to ABC WHB reprocessing |
| Gene-length / GC / LD confounds in gene-set tests | MAGMA covariates (gene size, density, MAC); S-LDSC baseline conditioning; length-matched bootstraps |
| Winner's curse across estimators | Replicate in BIG40 replication set; pre-registration |

## Milestones

- [ ] Download ABC metadata + MERFISH-CCF metadata; build cluster→supertype→subclass→class tables (`scripts/download_data.py --sample`).
- [ ] Pseudobulk + specificity at 4 mouse levels; human superclusters/clusters; orthology map.
- [ ] MAGMA gene analysis for 10 psychiatric/neurological positive controls; reproduce Bryois 2020 top hits (sanity).
- [ ] Select ~200 high-h² IDPs from BIG40 (thickness, area, volume, DTI FA/MD, rfMRI amplitude); run all estimators.
- [ ] Spatial-domain specificity and anatomical concordance test (H2).
- [ ] Resolution ablation (H3) and species comparison (H4).
- [ ] Estimator concordance and replication (H5); Cauchy consensus.
- [ ] Manuscript + public results browser (per-trait tables).

## Ethics / data-use notes

- All GWAS summary statistics used here are open; do not attempt to obtain individual-level UK Biobank
  data for this project (requires an approved application). PGC downloads require accepting the PGC
  data-use terms; cite the primary GWAS papers.
- ABC Atlas and CELLxGENE data are CC-BY-4.0; cite Yao et al. 2023, Zhang et al. 2023, Siletti et al. 2023.
- No human individual-level data are handled, so no IRB is required for the core analysis; check
  institutional policy if you add ENIGMA/ABCD individual-level data later.
- Never commit downloaded data or GWAS files (`data/` is git-ignored).
