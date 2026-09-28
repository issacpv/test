# dendritic-spine-metadata-mining

**A NeuroElectro-style, method-annotated database of dendritic spine densities mined from open-access literature and NeuroMorpho.org, with a meta-regression that quantifies how much of the reported variation is methodological (Golgi vs fluorescence vs EM, shrinkage, compartment, counting rules) and a calibration against exhaustive spine counts from dense EM connectomes.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (SWC spine detection and density audit; regex/unit-normalising extraction of spine-density statements with method/species/region context; Europe PMC full-text client; random-effects meta-regression with lab random effects).
- Difficulty: MSc thesis (mining + meta-regression) to PhD chapter (with EM calibration and a released resource).
- Timeline: 6-10 months (1 month NeuroMorpho audit, 2 months text mining + manual validation, 1-2 months curation, 1-2 months meta-regression, 1-2 months EM calibration, 1 month writing).
- Compute: laptop for mining and statistics; a workstation with ~100 GB disk for EM skeleton/synapse tables (no image volumes needed: CAVE/BossDB serve pre-computed synapse and skeleton tables).

## Background

Dendritic spine density (spines per micrometre of dendrite) is one of the most reported quantitative neuroanatomical measures: it is the standard readout in ageing, plasticity, psychiatric and neurodegenerative studies, and the basis of cross-species claims that human pyramidal cells carry more and larger spines (Elston, Benavides-Piccione & DeFelipe, 2001, Journal of Neuroscience; Benavides-Piccione, Ballesteros-Yanez, DeFelipe & Yuste, 2002, Journal of Neurocytology; Benavides-Piccione et al., 2013, Cerebral Cortex; "Principles for dendritic spine size and density in human and mouse", 2025). Yet the measurement depends heavily on method: Golgi impregnation hides spines behind the dendritic shaft (Feldman & Peters, 1979, Journal of Comparative Neurology, proposed a correction), fluorescent filling with confocal/two-photon imaging resolves more but is diffraction-limited, and serial EM resolves all spines (Harris, Jensen & Tsao, 1992, Journal of Neuroscience; Harris & Kater, 1994, Annual Review of Neuroscience). Automated detection tools add their own biases (NeuronStudio: Rodriguez et al., 2008, PLoS ONE).

Two developments make an audit timely. First, dense EM connectomes now provide exhaustive, method-independent spine and synapse counts on complete dendrites: mouse somatosensory cortex (Kasthuri et al., 2015, Cell; Motta et al., 2019, Science), a cubic millimetre of mouse visual cortex (MICrONS Consortium, 2025, Nature), human temporal cortex (H01: Shapson-Coe et al., 2024, Science) and cross-species comparisons (Loomba et al., 2022, Science). Second, open-access full text via Europe PMC makes large-scale extraction of reported values feasible, following the NeuroElectro model in which text-mined electrophysiology values were shown to depend strongly on methodological covariates (Tripathy et al., 2014, Frontiers in Neuroinformatics; Tripathy et al., 2015, Journal of Neurophysiology).

NeuroMorpho.org (Ascoli, Donohue & Halavi, 2007, Journal of Neuroscience; Akram et al., 2018, Scientific Data) hosts 250k+ reconstructions, but spines are rarely part of the digital tree (Halavi et al., 2012, Neuroinformatics, on reconstruction practice). How rarely, from which labs and tools, and whether the spine-bearing subset is representative, has not been quantified.

## The research gap

**What has been done**

- Careful single-lab, single-method datasets of human spine density and morphology (DeFelipe lab; the 2025 multi-hospital human spine dataset in Journal of Neurophysiology with ~4,000 reconstructed spines from 27 patients) and human-vs-mouse principles (2025).
- Disease-focused meta-analyses of spine density (schizophrenia: Glausier & Lewis, 2013, Neuroscience; Moyer, Shelton & Sweet, 2015, Neuroscience Letters) that pool studies without modelling method effects as the primary question.
- Methodological comparisons on small samples (Golgi vs intracellular fills; light vs EM) in individual papers, not archive-wide.
- Text-mined, method-annotated databases for electrophysiology (NeuroElectro) but not for spine density.

**What is missing (checked against 2023-2026 literature)**

1. No cross-study, method-annotated database of baseline (control-condition) spine densities across species, regions, cell types and dendritic compartments, and therefore no estimate of the variance explained by method versus biology.
2. No archive-wide quantification of the Golgi undercount, the confocal-vs-two-photon difference, the effect of shrinkage correction, or of counting rules (spines per linear micrometre vs per unit surface; inclusion of filopodia) on reported densities.
3. No calibration of literature light-microscopy densities against exhaustive EM counts from MICrONS/H01/Kasthuri on matched region, layer, cell type and compartment.
4. No audit of spine annotation in NeuroMorpho.org: fraction of reconstructions with spines encoded (custom SWC types or short terminal segments), by lab, software and year, and how the spine-bearing subset differs from the archive.
5. No reporting-quality analysis: how often papers report units per micrometre, the compartment, distance from soma, shrinkage correction, and animal-level n (needed for hierarchical meta-analysis).

## Research questions / hypotheses

1. **H1 (method variance).** In control-condition densities for the same species x region x cell type x compartment, method (Golgi / fluorescence-LM / EM), imaging modality and shrinkage correction explain at least 30% of between-study variance in log density. Test: random-effects meta-regression with lab random effects; R^2 between-study.
2. **H2 (Golgi undercount).** Golgi-derived densities are lower than fluorescence-LM densities from the same cell class by a factor of 1.3-2.0, and lower than EM counts by more. Test: method contrasts with 95% CIs; comparison with the Feldman-Peters correction factor.
3. **H3 (EM calibration).** For matched mouse V1/S1 L2/3 pyramidal basal dendrites, literature LM densities underestimate MICrONS/Kasthuri exhaustive counts by X%, and the gap is smaller for two-photon/STED studies. Test: paired comparison of literature-derived predicted density (from the meta-regression) vs EM density with bootstrap CIs.
4. **H4 (NeuroMorpho audit).** Fewer than 5% of NeuroMorpho dendritic reconstructions contain spine annotations; they come from a small number of labs and software packages (Neurolucida, NeuronStudio, Imaris) and are enriched for specific species/regions. Test: descriptive audit with per-lab/software breakdown; comparison of metadata distributions (chi-square) between spine-bearing and other reconstructions.
5. **H5 (compartment and distance).** Apical oblique and basal densities exceed apical trunk densities, and density peaks at intermediate distances from the soma; these gradients are consistent across LM methods once method is adjusted. Test: compartment and distance-bin moderators in the meta-regression.
6. **H6 (reporting quality over time).** The fraction of papers reporting compartment, shrinkage and animal-level n has increased since 2010 but remains below 50%. Test: logistic regression of reporting flags on year.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| Europe PMC (REST API) | Open-access full-text XML of papers reporting spine densities; search with `OPEN_ACCESS:Y AND HAS_FT:Y` | thousands of candidate papers | Open (no key) | https://europepmc.org/RestfulWebService |
| PubMed E-utilities | Abstract-level screening and metadata (year, journal, MeSH) for non-OA papers | metadata only | Open | https://www.ncbi.nlm.nih.gov/books/NBK25501/ |
| NeuroMorpho.org (v8.x) | Metadata + CNG.swc files; spine annotation audit (custom SWC types, short terminal segments), lab/software/year | 250k+ reconstructions | Open (REST API) | https://neuromorpho.org/api/ |
| MICrONS (MICrONS Consortium, 2025, Nature) | Proofread pyramidal-cell skeletons and synapse tables in mouse V1 (cubic millimetre); exhaustive spine/synapse counts per dendritic segment | ~10^5 cells, ~10^8 synapses (tables) | Free registration (CAVE token) | https://www.microns-explorer.org |
| H01 (Shapson-Coe et al., 2024, Science) | Human temporal cortex EM segmentation, skeletons and synapse tables | 1 mm^3 (tables) | Open (Google Cloud / Neuroglancer / BigQuery) | https://h01-release.storage.googleapis.com/landing.html |
| Kasthuri et al., 2015 (mouse S1) | Saturated reconstruction with spines, via BossDB | ~1,500 um^3 | Open | https://bossdb.org |
| Human dendritic spine dataset (J Neurophysiol, 2025) | ~4,000 reconstructed spines from 27 patients, 3 hospitals; validation of extracted human values | see paper | Check the paper's data-availability statement | https://journals.physiology.org/doi/full/10.1152/jn.00622.2024 |

No credentialed data; the human EM and patient datasets are de-identified and published.

## Methods

1. **Literature corpus** (`scripts/download_data.py --europepmc`): Europe PMC query such as `("spine density" OR "dendritic spines") AND ("spines/µm" OR "spines per" OR "per µm" OR "per 10 µm") AND OPEN_ACCESS:Y AND HAS_FT:Y`, paginated with `cursorMark`; full-text XML stored per PMCID; body text, figure captions and table cells extracted.
2. **Extraction** (`src/spine_mining/text_extraction.py`): regex grammar for numeric density statements with units (spines/um, per 10 um, per 100 um, spines per micron, spines/um^2 flagged as surface density), +/- SD/SEM, ranges; context window with keyword dictionaries for species, region, layer, cell type, compartment (basal, apical trunk/oblique/tuft), method (Golgi, Golgi-Cox, DiI, biocytin, Lucifer yellow, GFP/YFP, Thy1, EM/serial section/FIB-SEM/SBEM), imaging (bright-field, confocal, two-photon, STED, super-resolution), shrinkage correction, distance from soma, condition (control vs treated). Output: one row per statement with provenance (PMCID, sentence).
3. **Validation and curation**: 200 randomly sampled statements annotated by two curators; precision/recall of extraction; disagreements resolved; extraction rules iterated; final table restricted to control-condition values with method and compartment resolved.
4. **NeuroMorpho audit** (`src/spine_mining/swc_spines.py`): for every CNG.swc, detect custom SWC type codes (> 4) and short terminal segments (<= 3 um, <= 3 nodes) attached to dendrites; compute spine-like density per um of dendrite; cross-tabulate with metadata (archive, software, species, year); flag reconstructions with plausible spine annotation (density 0.2-5 per um) vs artefacts.
5. **EM calibration**: from MICrONS (CAVE tables), H01 and Kasthuri, select L2/3 and L5 pyramidal cells, split dendrites by compartment and distance bins, count spine synapses per um of skeleton; compare with literature-predicted densities for matched cells.
6. **Meta-regression** (`src/spine_mining/meta_regression.py`): log density ~ method + imaging + shrinkage + species + region + compartment + distance bin + year + (1 | lab) with study-level sampling variance from reported SD/n; method-of-moments tau^2 (DerSimonian-Laird generalisation) and REML cross-check; between-study R^2; leave-one-lab-out influence.
7. **Tools**: `requests`, `lxml`/`xml.etree`, `regex`, `pandas`, `numpy`, `statsmodels`, `scipy`; optional `scispacy` for entity linking; `caveclient` for MICrONS; `cloud-volume`/`intern` for H01/BossDB.

## Evaluation & statistics

- Extraction quality: precision/recall/F1 on the annotated set; per-field accuracy (value, unit, method, compartment); a second annotator for inter-rater agreement (kappa).
- Meta-regression estimands: method contrasts (log ratio with 95% CI), tau^2 and I^2, between-study R^2 for method-only vs biology-only vs full models; prediction intervals per species x region x cell type.
- Leakage/duplication: papers reporting the same dataset (same lab, same animals) detected by author overlap and identical values; only one entry retained per dataset; lab random effect handles residual clustering.
- Nulls: permutation of method labels across studies within species x region strata to obtain the null distribution of the method R^2.
- Multiple comparisons: Holm across method contrasts; BH across compartment/region moderators.
- Sensitivity: exclude automated-detection studies; exclude values without reported n; SEM vs SD ambiguity handled by sensitivity bounds; unit-ambiguous statements excluded.
- EM calibration uncertainty: bootstrap over EM cells; report the LM/EM ratio per method with CIs.

## Publishable angle

- **Headline**: "Across N studies, method explains X% of the variation in reported spine density; Golgi undercounts by a factor Y relative to fluorescence and Z relative to EM; literature values for mouse L2/3 pyramidal cells underestimate exhaustive EM counts by W%; only V% of NeuroMorpho reconstructions carry spines." Plus a released, method-annotated spine-density table (a "NeuroElectro for spines").
- Target venues: *Scientific Data* (resource), *Cerebral Cortex*, *Journal of Comparative Neurology*, *Frontiers in Neuroanatomy*, *eLife* (Tools & Resources).
- Follow-ups: extend to spine morphology classes (thin/stubby/mushroom proportions by method); integrate with NeuroMorpho as a spine-density metadata layer; use the method-adjusted normative values as priors for disease meta-analyses; LLM-assisted extraction evaluated against the curated gold standard (local models only, to keep the pipeline reproducible).

## Risks, confounds & mitigations

- **Extraction errors** (values from treated groups, from other measures such as synapse density): control-condition filter uses sentence and paragraph context; manual validation set; conservative exclusion of ambiguous statements.
- **Publication and reporting bias**: the corpus is OA-only; compare distribution of methods/years between OA and non-OA hits from PubMed metadata; report as a limitation.
- **Confounding of method with era, lab and species**: lab random effects, year covariate, species x method interaction; permutation null for R^2.
- **NeuroMorpho CNG standardisation may strip spine annotations**: compare CNG vs original-format files for a subset; treat the audit as "spines encoded in the distributed SWC", which is what re-users get.
- **EM matching**: EM volumes cover specific regions/layers; calibration is limited to those; report matched-only comparisons.
- **Unit heterogeneity** (per um vs per 10 um vs per 100 um vs per um^2): explicit unit grammar and normalisation; surface densities kept separate.

## Milestones

- [ ] NeuroMorpho metadata harvest + SWC audit for spine annotation; audit table by lab/software/species/year.
- [ ] Europe PMC corpus; extraction grammar; first pass over full texts; statement table with provenance.
- [ ] Annotation set (200 statements, 2 curators); precision/recall; grammar iteration; curated control-condition table.
- [ ] Meta-regression with method/biology moderators and lab random effects; permutation null; sensitivity analyses.
- [ ] EM calibration on MICrONS/H01/Kasthuri matched cells.
- [ ] Reporting-quality trends; released dataset (CSV + provenance) and code; manuscript.

## Ethics / data-use notes

- Text mining uses only open-access full text under licences permitting reuse (Europe PMC OA subset; respect each article's licence for redistribution of extracted sentences: distribute values + citations, not full text).
- NeuroMorpho and EM datasets are open; cite original depositors and consortium papers as required.
- The human EM (H01) and patient spine datasets are de-identified; no attempt at re-identification; report aggregates only.
- If LLM-assisted extraction is used, run local models; do not upload full-text corpora to third-party APIs where licences prohibit it.
- Never commit full-text XML, SWC files or EM tables; commit only the curated value table with citations.

## Related projects (kept self-contained here)

- `neuromorpho-scaling-laws` and `neuromorpho-software-fingerprinting` (lab/software effects in NeuroMorpho; this project audits a specific, rarely encoded structure).
- `unlabeled-adverse-event-mining` (text-mining pipeline patterns with manual validation).
