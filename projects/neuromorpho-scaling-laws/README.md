# neuromorpho-scaling-laws

**A batch-corrected, phylogenetically-aware meta-analysis of dendritic allometric scaling laws across 250k+ NeuroMorpho.org reconstructions — and an audit of how much "species difference" in the neuromorphology literature is really lab/protocol effect.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (API client, SWC morphometrics, mixed-effects and phylogenetic GLS helpers).
- Difficulty: MSc thesis to early-PhD chapter. Statistics-heavy, no GPU.
- Timeline: 6-9 months (2 months data pull + QC, 2 months morphometrics, 2-3 months modelling, 1-2 months writing).
- Compute: a laptop is enough for morphometrics on ~50k SWC files (each file is seconds); a small workstation (16 cores, 64 GB) makes the full 250k+ pass and bootstrap/permutation runs comfortable. Disk: ~40-80 GB for all CNG SWC files (can be streamed and discarded).

## Background

Dendritic arbors obey striking regularities. Cuntz, Mathy & Häusser (2012, PNAS) derived from minimum-spanning-tree wiring optimality that total dendritic length *L* scales with the number of branch points *n* and the spanning volume *V* as a power law — a 2/3 exponent for 3-D arbors (and 1/2 for planar ones) — and reported it holds across ~6,000 reconstructions from ~140 NeuroMorpho datasets. Earlier "one rule to grow them all" work (Cuntz et al., 2010, PLoS Comput Biol) framed dendrites as balancing wiring length against conduction (path) cost with a single balancing factor. Related "universal" claims exist for branch density (Snider, Pillai & Stevens, 2010, J Neurosci; Teeter & Stevens, 2011, PNAS), for the decomposition of Sholl profiles into density and length components (Bird & Cuntz, 2019, Cell Reports), and for connectivity-repertoire maximisation (Wen et al., 2009, PNAS; Chklovskii, 2004, Neuron).

These laws are increasingly used as priors: for synthetic morphology generators, for cross-species scaling arguments about human neurons (Deitcher et al., 2017, Cereb Cortex; Beaulieu-Laroche et al., 2018, Cell; Benavides-Piccione et al., 2024, Cereb Cortex; "A framework for comparative analysis of human and mouse cortical neurons", Nat Neurosci 2026), and for neuromorphic hardware design. If the exponents or the residual "species effects" are contaminated by who traced the cell, with what software, at what shrinkage correction, the downstream claims inherit that contamination.

## The research gap

**What has been done**

- Cuntz et al. (2012) tested the 2/3 law on a curated subset (~6k cells) without modelling lab or protocol as random effects and without accounting for the fact that species share evolutionary history.
- Polavaram, Gillette, Parekh & Ascoli (2014, Front Neuroanat) mined NeuroMorpho by metadata category and showed morphometric distributions differ across species/regions/cell types — but treated categories as fixed, independent labels and did not partition variance into biological vs. provenance components.
- NeuroMorpho itself now curates, per reconstruction, the archive (lab), reconstruction software, shrinkage reported/corrected status, slicing thickness, objective/magnification, stain, protocol (in vitro / in vivo / culture) and original file format (Ascoli, Donohue & Halavi, 2007, J Neurosci; Akram et al., 2018, Sci Data; Ascoli et al., 2024, FASEB BioAdvances). These are exactly the batch variables needed, and they are available through the REST API.
- Cross-species comparisons in 2023-2026 (human vs. mouse: Benavides-Piccione et al., 2024; "Of mice and men: dendritic architecture differentiates human from mice", 2025; the Nat Neurosci 2026 region-correspondence framework with 2,363 human and 16,011 mouse reconstructions) are two-species, cortex-only, and are largely single-consortium data, so lab and species are confounded by design.
- Comparative-neuroanatomy work across many mammals (e.g. Jacobs and colleagues on cerebellar and cortical neurons across afrotherians, carnivores, cetartiodactyls and primates, Front Neuroanat 2014) uses Golgi material from one lab and small n, and does not use phylogenetic comparative methods (PGLS; Felsenstein, 1985, Am Nat; Freckleton, Harvey & Pagel, 2002, Am Nat).

**What is missing (verified by searching 2023-2026 literature)**

1. No study has estimated allometric scaling exponents (L vs. n, L vs. V, Sholl-profile shape, branch-order distribution) with the archive/lab as a *random effect* and reconstruction software, shrinkage correction, protocol and slice thickness as covariates, across the full NeuroMorpho corpus.
2. No study has quantified what fraction of apparent between-species morphometric variance is attributable to lab/protocol, i.e. an attenuation estimate for "species effects" after batch adjustment. The identification comes from (a) labs that deposited >1 species and (b) species deposited by >1 lab — both are common in NeuroMorpho.
3. No neuromorphology scaling study has applied phylogenetic generalised least squares or Pagel's lambda to account for non-independence of species means (standard in comparative biology, absent here).
4. The dimensionality confound has not been treated: many Neurolucida reconstructions are effectively planar (thin slices, z-compression), for which wiring optimality predicts a 1/2 rather than 2/3 exponent. The effective dimensionality of each reconstruction is itself a lab/protocol property.

## Research questions / hypotheses

1. **H1 (exponent recovery).** After lab random effects and dimensionality adjustment, the pooled L ~ n^b V^c exponents are b = 2/3, c = 1/3 for 3-D arbors and b = 1/2, c = 1/2 for near-planar arbors. Test: Wald test of b against 2/3 (or 1/2) from the mixed model; report CI and equivalence bounds (±0.05).
2. **H2 (batch inflation).** Naive per-species scaling exponents and per-species morphometric means differ more than batch-adjusted ones. Prediction: at least 30% of naive between-species variance in total length, branch-point count and Sholl peak is absorbed by the archive random effect and software/shrinkage covariates. Test: attenuation ratio with bootstrap CI; permutation null (shuffle species labels across archives).
3. **H3 (phylogenetic signal).** Species-level residuals of the scaling law (after batch correction) show phylogenetic signal (Pagel's lambda > 0) and scale with brain mass (Herculano-Houzel-style allometry); the PGLS slope of log residual length vs. log brain mass differs from the OLS slope.
4. **H4 (region and cell-class dependence).** Exponents are conserved across brain regions for principal cells but deviate for interneuron classes and for axons (Cuntz 2012 predicted axons follow different cost weights; Sci Rep 2022 "How axon and dendrite branching are guided by time, energy, and space" argued axons minimise time rather than power). Test: region × class interaction terms with lab random effects.
5. **H5 (software/shrinkage bias direction).** Reconstructions flagged "shrinkage not corrected" have systematically smaller z-extent and lower effective dimensionality, and automated tracers (e.g. neuTube, Vaa3D, TREES) yield different branch-point counts at matched total length than manual Neurolucida traces. Test: software fixed effects within labs that used several tools.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| NeuroMorpho.org (v8.x) | Metadata for every reconstruction (species, region, cell type, archive/lab, software, shrinkage, protocol, slice thickness, magnification, stain, integrity); pre-computed L-Measure morphometry; standardised CNG.swc files | 250k+ reconstructions, ~1,000 labs, >100 species | Open, no registration (REST API v1) | https://neuromorpho.org/api/ , https://neuromorpho.org/apiReference.html |
| TimeTree 5 | Pairwise species divergence times to build the phylogenetic covariance matrix | ~150k species | Open (web query / bulk download) | http://timetree.org |
| Brain mass / neuron number tables (Herculano-Houzel and colleagues, published supplementary tables) | Species-level covariates for H3 | tens of species | Open (paper supplements) | see cited papers |
| Allen Cell Types Database (optional validation) | Single-consortium, single-protocol mouse + human reconstructions to check that within-consortium species differences agree with batch-adjusted NeuroMorpho estimates | ~1k reconstructions | Open (allensdk) | https://celltypes.brain-map.org |

No credentialed data are involved.

## Methods

1. **Metadata harvest** (`scripts/download_data.py`): page through `/api/neuron` (500 per page) or `/api/neuron/select` with JSON filters; store one JSONL per species; flatten list fields (`brain_region`, `cell_type`). Also harvest `/api/morphometry` (NeuroMorpho's L-Measure summary: `total_length`, `n_bifs`, `n_branch`, `max_branch_order`, `width/height/depth`, `volume`, `surface`, `partition_asymmetry`, `fractal_dimension`, ...) so a first pass needs no SWC files.
2. **Inclusion / QC**: `physical_Integrity` in {Dendrites Complete, Dendrites & Axon Complete}; `structural_domains` containing dendrites; min 20 branch points; exclude glia and `experiment_condition` other than control unless studying it; flag duplicates by neuron_name and PMID.
3. **Own morphometrics** (`src/nm_scaling/swc.py`) recomputed from CNG.swc for a consistent definition: total length per neurite type, branch points, tips, stems, max branch order, path/Euclidean extents, convex-hull spanning volume, Sholl profile (segment crossings at fixed and normalised radii), PCA-based effective dimensionality (participation ratio) and planarity.
4. **Allometric mixed models** (`src/nm_scaling/mixed_models.py`, statsmodels `MixedLM`): `log L = a + b log n + c log V + d·dim_eff + software + shrinkage + protocol + slice_thickness + species + region_top + class + (1 | archive)`; optional variance component for `reconstruction_software` nested in archive. Exponents are reported per species/region via interactions; ICC and lab variance fraction reported.
5. **Species-effect attenuation**: naive OLS species coefficients vs. mixed-model coefficients; attenuation = 1 − Var(adjusted species effects)/Var(naive species effects); bootstrap over archives (cluster bootstrap) for CI.
6. **Phylogenetic GLS** (`src/nm_scaling/phylo_gls.py`): species-level batch-adjusted residuals (BLUP-adjusted means with SEs) regressed on brain mass / neuron number under Brownian covariance from a TimeTree-derived Newick; Pagel's lambda estimated by ML on a grid; measurement-error variance added to the diagonal (weighted PGLS).
7. **Sholl and branch-order laws**: fit Bird & Cuntz-style density × length decomposition; compare normalised Sholl profiles across species after adjusting for lab, using functional mixed models on the profile (basis-spline coefficients as multivariate response).
8. **Tools**: `requests`, `pandas`, `numpy`, `scipy`, `statsmodels`, optional `neurom`/`morphopy` for cross-checking morphometrics, `ete3` or own parser for Newick.

## Evaluation & statistics

- Primary estimands: exponents b, c with 95% CI (Wald and cluster bootstrap by archive), lab variance fraction (ICC), attenuation ratio for species effects, Pagel's lambda with LR test vs. lambda = 0.
- Validation of batch model: hold out entire archives (leave-lab-out CV) and check predictive log-likelihood vs. naive model; check that within-lab multi-species contrasts (e.g. archives with both mouse and rat) match the mixed-model species contrasts.
- Leakage: no learning across cells beyond fixed/random effects; duplicates (same neuron in several archives, e.g. re-deposited datasets) detected by neuron_name/PMID and dropped.
- Multiple comparisons: Benjamini-Hochberg across species × region × class exponent tests; pre-registered primary hypotheses H1-H3, secondary H4-H5.
- Nulls: (a) permute species labels across archives to get the null attenuation; (b) permute archive labels within species for the null lab variance; (c) phylogenetic null by tip-shuffling on the tree.
- Sensitivity: convex hull vs. bounding box vs. alpha-shape volume; including/excluding "shrinkage not corrected"; morphometrics from NeuroMorpho L-Measure vs. own recomputation.

## Publishable angle

- **Headline**: "Roughly X% of reported cross-species differences in dendritic length/branching are attributable to laboratory and reconstruction protocol; after correction, the wiring-optimality exponent is conserved across N species and M brain regions, and species residuals show phylogenetic signal that tracks brain mass." Also a concrete, citable table of software/shrinkage bias magnitudes.
- Target venues: *PLoS Computational Biology*, *Neuroinformatics*, *Cerebral Cortex* (comparative angle), *eLife* (Tools & Resources or Research Advance), *Journal of Neuroscience* (if the phylogenetic result is strong).
- Follow-ups: a NeuroMorpho-wide "batch-adjusted morphometrics" release; applying the same design to axonal arbors and to Sholl-derived electrotonic proxies; using the lab random effects as a prior for automated reconstruction QC.

## Risks, confounds & mitigations

- **Species nested within labs**: many labs deposit only one species; identification relies on multi-species labs and multi-lab species. Mitigation: report the identifying subset explicitly; simulate power with the observed nesting structure; run models on the identifying subset and on all data.
- **Reconstruction incompleteness** (cut dendrites in slices): use `physical_Integrity`, slice thickness, and z-extent as covariates; sensitivity analysis on in-vivo/whole-mount only.
- **Region and cell-type ontology heterogeneity**: NeuroMorpho region labels are hierarchical and inconsistently deep. Mitigation: map to top-level regions (neocortex, hippocampus, cerebellum, ...) and use cell-class (principal vs. interneuron vs. sensory) tiers.
- **Convex hull volume for sparse/planar arbors** is unstable: use PCA effective dimensionality to switch between area and volume; alpha-shape as sensitivity.
- **Phylogeny uncertainty**: sample over TimeTree credible intervals; results should not depend on fine topology.
- **API rate limits / TLS quirks**: the client retries with backoff and respects `NEUROMORPHO_VERIFY_SSL`.

## Milestones

- [ ] Harvest full metadata + morphometry tables (JSONL → parquet), document counts by species/lab/software.
- [ ] QC and inclusion rules frozen; duplicate detection.
- [ ] Own morphometrics on a 20k stratified sample, agreement with L-Measure values (Bland-Altman).
- [ ] Mixed model for L ~ n, V with lab random effect; H1 tested on 3-D vs planar subsets.
- [ ] Species attenuation analysis (H2) with cluster bootstrap and permutation null.
- [ ] TimeTree covariance + PGLS on species residuals (H3).
- [ ] Region/class/axon extensions (H4), software/shrinkage bias table (H5).
- [ ] Sholl functional mixed model.
- [ ] Preprint + code/data release (adjusted morphometrics table).

## Ethics / data-use notes

- NeuroMorpho.org data are open; cite NeuroMorpho.org and the original depositing publications (each record carries `reference_pmid`/`reference_doi`) in any publication, as required by their terms.
- No human-identifiable data are involved (human reconstructions are de-identified resections/post-mortem tissue curated by the depositing labs).
- Do not commit downloaded SWC files or metadata dumps; `data/` is git-ignored.
- Be polite to the API: page size ≤ 500, backoff on 429/5xx, cache locally.
