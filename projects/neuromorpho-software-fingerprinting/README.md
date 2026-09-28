# neuromorpho-software-fingerprinting

**Can a classifier tell which reconstruction software (or which lab) produced a neuron from the SWC geometry alone? Quantifying provenance leakage in NeuroMorpho.org as a detectability index, calibrating it on BigNeuron (same image, many tracers), and testing whether harmonisation removes it without erasing biology.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (SWC parser, morphometrics, "sampling fingerprint" features that capture node spacing, coordinate and radius quantisation; grouped-CV detectability index with label-permutation null; ComBat harmonisation and a kBET-style local mixing test; synthetic tree generator with controllable tracing artefacts).
- Difficulty: MSc thesis. CPU only.
- Timeline: 4-8 months (1 month metadata + SWC harvest, 1 month features, 1-2 months classifiers and nulls, 1 month BigNeuron calibration, 1 month harmonisation + writing).
- Compute: laptop for a 50k-cell subset; a 16-core workstation for the full 250k+ archive (feature extraction is seconds per file; gradient-boosting fits are minutes). Disk: 40-80 GB for all CNG SWC files, streamable.

## Background

NeuroMorpho.org curates 250k+ digital reconstructions from ~1,000 laboratories (Ascoli, Donohue & Halavi, 2007, Journal of Neuroscience; Akram et al., 2018, Scientific Data), with per-cell metadata including the reconstruction software, original file format, shrinkage correction, slice thickness, objective and protocol. Meta-analyses across the archive (Polavaram, Gillette, Parekh & Ascoli, 2014, Frontiers in Neuroanatomy), cell-type classifiers built on morphology (Gouwens et al., 2019, Nature Neuroscience; Laturnus, Kobak & Berens, 2020, Neuroinformatics), and topological descriptors (Kanari et al., 2018, Neuroinformatics) all assume that the geometry reflects the neuron rather than the tracing pipeline.

That assumption has been questioned for two decades: Scorcioni, Lazarewicz & Ascoli (2004, Journal of Comparative Neurology) found that hippocampal pyramidal cells differed more between reconstructing laboratories than between anatomical classes. Automated tracers differ systematically from manual ones and from each other (BigNeuron: Peng et al., 2015, Neuron; Manubens-Gil et al., 2023, Nature Methods; DIADEM: Brown et al., 2011, Neuroinformatics), and tools like Neurolucida, neuTube, Vaa3D, TREES and Imaris impose their own node spacing, radius estimation and z-sampling. In single-cell genomics this class of problem is handled with batch-detectability metrics (kBET: Büttner et al., 2019, Nature Methods) and empirical-Bayes harmonisation (ComBat: Johnson, Li & Rabinovic, 2007, Biostatistics). Neuromorphology has neither.

## The research gap

**What has been done**

- Lab-to-lab morphometric differences demonstrated on a few hundred cells (Scorcioni et al., 2004) and archive-wide metadata-stratified statistics (Polavaram et al., 2014), without a detectability framing.
- Algorithm-vs-gold-standard accuracy benchmarks (DIADEM; BigNeuron) that quantify tracing *error* relative to manual annotation, not the *signature* each algorithm leaves in downstream morphometrics.
- Morphology-based cell-type classification evaluated with random or stratified cross-validation, rarely with laboratory held out.
- A companion project in this repository (`neuromorpho-scaling-laws`) models lab as a random effect in allometric mixed models; it does not ask how *identifiable* provenance is or whether harmonisation removes it.

**What is missing (checked against 2023-2026 literature)**

1. No published estimate of how accurately reconstruction software or laboratory can be predicted from SWC geometry alone, archive-wide, with proper grouped cross-validation (whole archives held out) and a permutation null.
2. No separation of two sources of provenance signal: *sampling artefacts* (node spacing, coordinate/radius quantisation, z-step, planar compression) versus *morphometric distortion* (branch counts, lengths, tortuosity). The former can be removed by resampling; the latter cannot.
3. No calibration on a design where biology is held fixed: BigNeuron provides the same image stacks traced by dozens of algorithms plus manual annotation, i.e. a pure software effect. Signatures learned there can be tested for transfer to NeuroMorpho.
4. No "provenance leakage" audit of morphology-based cell-type classifiers: does accuracy drop when archives are held out, and is part of the reported accuracy attributable to lab signatures correlated with cell type (because labs specialise in cell types)?
5. No evaluation of harmonisation (ComBat on morphometrics; resampling of SWC geometry to a canonical node spacing/precision) by residual detectability and by preservation of known biological contrasts (e.g. pyramidal vs interneuron, apical vs basal statistics).

## Research questions / hypotheses

1. **H1 (detectability).** With archives held out (GroupKFold by archive), a gradient-boosting classifier predicts reconstruction software from geometry with balanced accuracy well above the permutation null (detectability index > 0.3 on a 0-1 scale), and predicts *archive* within a software class above chance as well. Test: grouped CV; label permutations at archive level; CIs by repeated CV.
2. **H2 (sampling vs morphometric signal).** Sampling-fingerprint features (inter-node distance distribution, coordinate decimal precision, radius uniqueness, z-step quantisation) carry most of the software signal; after resampling every tree to a fixed node spacing and rounding coordinates to a common precision, detectability falls by > 50% but remains above the null. Test: detectability with feature subsets and with resampled SWCs.
3. **H3 (BigNeuron calibration).** A classifier trained to distinguish tracing algorithms on BigNeuron (same images) transfers to NeuroMorpho reconstructions labelled with the same software (above-chance accuracy), demonstrating the signature is algorithmic, not biological. Test: train on BigNeuron, test on NeuroMorpho subset with matching software labels.
4. **H4 (leakage in cell-type classification).** Morphology-based cell-type accuracy (e.g. pyramidal vs interneuron subclasses within mouse neocortex) drops by a measurable margin when archives are held out versus random CV, and the drop correlates with per-archive provenance detectability. Test: paired CV designs; correlation across cell-type tasks.
5. **H5 (harmonisation).** ComBat on morphometric features with software/archive as batch reduces residual detectability to near null while preserving known contrasts (effect sizes for apical/basal, species, cell-class differences within a single-pipeline reference such as the Allen Cell Types Database). Test: pre/post detectability; kBET-like rejection rate; effect-size preservation on the reference.
6. **H6 (temporal drift).** Provenance signatures change with software version/deposition year, so detectability of *year* is also above chance within software. Test: ordinal classification of deposition year bins.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| NeuroMorpho.org (v8.x) | Metadata (reconstruction_software, archive, original_format, shrinkage, slicing thickness, objective, magnification, stain, protocol, deposition date, species, region, cell type) + standardised CNG.swc files | 250k+ reconstructions | Open (REST API, no registration) | https://neuromorpho.org/api/ , https://neuromorpho.org/apiReference.html |
| BigNeuron (Manubens-Gil et al., 2023, Nature Methods) | Gold-standard light-microscopy image stacks, each with manual annotation and reconstructions from dozens of automated tracing algorithms | > 100 image stacks x dozens of algorithms | Open (see the paper's data-availability statement; BigNeuron GitHub/Zenodo resources) | via the paper |
| DIADEM challenge (Brown et al., 2011, Neuroinformatics) | Gold-standard reconstructions and images for several tissue types | 6 datasets | Open | http://diademchallenge.org |
| Allen Cell Types Database | Single-pipeline mouse + human reconstructions (reference for effect-size preservation under harmonisation) | ~1k reconstructions | Open (`allensdk`) | https://celltypes.brain-map.org |
| MouseLight (Winnubst et al., 2019, Cell) and single-pipeline whole-brain fMOST reconstructions (Peng et al., 2021, Nature) | Large single-pipeline sets to estimate within-pipeline morphometric variance | ~1k and ~1.7k neurons | Open | http://ml-neuronbrowser.janelia.org , via BIL |

No credentialed data.

## Methods

1. **Harvest** (`scripts/download_data.py`): page the NeuroMorpho API, stratify by `reconstruction_software` x `archive`, and download CNG.swc files for a balanced design (cap per archive so that no lab dominates a software class). Record software version strings where present.
2. **Features** (`src/nm_fingerprint/swc_features.py`):
   - Morphometrics: node count, total length per neurite type, bifurcations, tips, stems, max branch order, mean/CV branch length, tortuosity, extents, PCA planarity (participation ratio), radius statistics.
   - Sampling fingerprint: inter-node distance quantiles and CV, grid-likeness (fraction of distances near the modal spacing), coordinate decimal precision, fraction of unique radii, radius rounding, z-step quantisation (unique z fraction, modal z-step), collinearity fraction, fraction of child nodes with radius identical to parent.
   - Optional: persistence-diagram summary (TMD) vectors via `tmd` for a topology-only arm.
3. **Detectability** (`src/nm_fingerprint/fingerprint.py`): HistGradientBoosting / random forest with GroupKFold by archive; balanced accuracy, macro-F1, per-class recall; detectability index = (BA - BA_null)/(1 - BA_null) with archive-level label permutations; permutation feature importance; per-record provenance risk score (max class probability).
4. **Resampling arm**: resample every SWC to 1 um node spacing with linear interpolation, round coordinates to 0.01 um, drop radii (or set to branch mean) and recompute detectability (H2).
5. **BigNeuron calibration**: features for every algorithm's reconstruction of every gold-standard image; algorithm classifier with image-level GroupKFold; transfer to NeuroMorpho subsets with matching software labels (H3).
6. **Leakage audit** (H4): cell-type classification tasks defined from NeuroMorpho metadata within one species/region; random CV vs leave-archive-out; relate the accuracy drop to per-archive detectability.
7. **Harmonisation** (`src/nm_fingerprint/harmonize.py`): parametric ComBat with software (or archive) as batch and species/region/cell class as covariates; kBET-style local mixing test; residual detectability; effect-size preservation on Allen reference contrasts.
8. **Tools**: `numpy`, `scipy`, `pandas`, `scikit-learn`, `statsmodels`, `requests`; optional `neurom` (morphometric cross-check), `tmd` (topological descriptors), `allensdk`.

## Evaluation & statistics

- Primary estimands: detectability index for software and for archive (with 95% CIs from 10x repeated grouped CV); its decomposition into sampling vs morphometric feature sets; residual detectability after harmonisation.
- Nulls: archive-level label permutation (labels shuffled across archives, keeping cells within an archive together) so that the null respects the grouping; 500+ permutations.
- Leakage prevention: GroupKFold by archive everywhere; hyper-parameters chosen by nested grouped CV; BigNeuron images held out by image id; duplicates across archives removed by neuron name/PMID.
- Class imbalance: cap per archive and per software; report balanced accuracy and per-class recall; minority software classes (< 500 cells) merged into "other" and reported separately.
- Multiple comparisons: Holm across feature-subset contrasts; BH across cell-type tasks in H4.
- Sensitivity: including/excluding radii; excluding "shrinkage not corrected"; dendrite-only vs dendrite+axon; per-species analyses.

## Publishable angle

- **Headline**: "Reconstruction software is predictable from SWC geometry with balanced accuracy X% (chance Y%) and laboratory with Z%; most of the signal is sampling artefact removable by resampling, but a residual morphometric signature survives and inflates morphology-based cell-type accuracy by W points under random cross-validation." Plus a per-record provenance risk score for the archive and a harmonisation recipe validated by residual detectability.
- Target venues: *Neuroinformatics*, *PLoS Computational Biology*, *Nature Methods* (Brief Communication if the leakage audit is striking), *Frontiers in Neuroinformatics*, *eLife* (Tools & Resources).
- Follow-ups: a NeuroMorpho-integrated "batch-corrected morphometrics" table; software-signature-aware priors for automated tracer QC; extension to EM-derived skeletons (MICrONS/H01) where the "software" is the segmentation pipeline.

## Risks, confounds & mitigations

- **Software confounded with species/cell type/lab** (e.g. one tool dominant in one archive): stratify and cap; test software *within* archives that used several tools; BigNeuron calibration isolates the algorithmic signal.
- **Metadata errors** in `reconstruction_software`: treat as label noise; report confusion matrices; manually verify a random sample of 100 records against original publications.
- **CNG standardisation** already rewrites SWC files (NeuroMorpho's CNG version resamples/normalises some fields): document exactly what CNG preserves; where possible obtain original files (`original_format`) for a subset to compare pre/post standardisation detectability.
- **Over-interpretation of "lab effects" as errors**: labs also study different neurons; the design contrasts sampling features (pure artefact) against morphometrics (mixed) and uses within-image BigNeuron evidence.
- **Harmonisation removing biology**: effect-size preservation checks on single-pipeline references.

## Milestones

- [ ] Harvest metadata; distribution of software x archive x species; balanced design; SWC download.
- [ ] Feature extraction (morphometrics + sampling fingerprint); synthetic validation of fingerprint features.
- [ ] Software/archive detectability with grouped CV and archive-level permutation nulls; feature-subset decomposition.
- [ ] Resampling arm; BigNeuron algorithm classifier and transfer test.
- [ ] Cell-type leakage audit (random vs leave-archive-out).
- [ ] ComBat + resampling harmonisation; residual detectability; effect-size preservation on Allen reference.
- [ ] Provenance risk scores for the archive; manuscript and code release.

## Ethics / data-use notes

- All data are open, non-human-identifiable reconstructions; cite NeuroMorpho.org and the original depositing publications (the API provides PMIDs) as required by NeuroMorpho's terms.
- Do not commit SWC files or bulk metadata dumps; commit only derived feature tables and results.
- Provenance risk scores name laboratories only in aggregate; the goal is to characterise pipelines, not to grade labs.

## Related projects (kept self-contained here)

- `neuromorpho-scaling-laws` (mixed-effects modelling of lab/protocol effects on allometric exponents; complementary to this project's detectability framing).
- `morphology-to-electrophysiology` and `morphology-dependent-stimulation` (downstream consumers of morphologies that inherit provenance bias).
