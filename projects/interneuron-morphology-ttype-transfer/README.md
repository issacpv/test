# interneuron-morphology-ttype-transfer

**How much transcriptomic-type information is carried by interneuron morphology alone, and does it transfer from mouse visual cortex to human middle temporal gyrus? A benchmark of hand-crafted vs. learned morphology embeddings under a harmonised cross-species subclass taxonomy, with laminar-position and laboratory confounds explicitly separated.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (SWC morphometrics and depth-normalised density maps, taxonomy harmonisation, CORAL-aligned cross-species classifier with leave-dataset-out evaluation and permutation nulls).
- Difficulty: MSc thesis to early-PhD chapter; machine learning on small-n biological data. A single GPU is optional (only for the self-supervised graph embedding baseline).
- Timeline: 6-9 months (1-2 months data assembly and taxonomy harmonisation, 2 months features/embeddings, 2 months transfer experiments, 1-2 months writing).
- Compute: laptop for hand-crafted features (~2,000 reconstructions); one GPU-day for a GraphDINO-style embedding.

## Background

Patch-seq links a neuron's transcriptome (t-type), electrophysiology and morphology in the same cell. In mouse visual cortex, Gouwens et al. (2020, *Cell*) defined morpho-electric-transcriptomic (MET) types for GABAergic interneurons and showed that morphology and electrophysiology only partly resolve t-types; Scala et al. (2021, *Nature*) found a continuum rather than discrete morpho-electric clusters in mouse motor cortex. In human neocortex, Lee et al. (2023, *Science*) characterised morpho-electric signatures of GABAergic interneurons mapped to the middle temporal gyrus (MTG) taxonomy of Hodge et al. (2019, *Nature*), and Chartrand et al. (2023, *Science*) showed that layer 1 interneuron types diverge between human and mouse in the very features that distinguish types. Consensus cross-species taxonomies (Bakken et al., 2021, *Nature*) allow subclasses (Pvalb, Sst, Vip, Lamp5, Sncg/PAX6) to be aligned between mouse and human.

Morphology-only representations have matured in parallel: Laturnus, Kobak & Berens (2020, *Neuroinformatics*) systematically evaluated interneuron morphology representations for cell-type discrimination; MorphVAE (Laturnus & Berens, 2021, *ICML*) and GraphDINO (Weis et al., "Self-supervised graph representation learning for neuronal morphologies", 2021 preprint / later journal version) learn embeddings without labels; Weis et al. (2025, *Nature Communications*) used such embeddings to map excitatory dendritic morphology across mouse visual cortex. Most recently, a cross-species transfer study mapped *electrophysiology* to transcriptomic identity of GABAergic interneurons from mouse to human using the Allen Patch-seq datasets (*Neuroinformatics*, 2026).

## The research gap

**What has been done**

- Within-species, within-lab prediction of subclass/MET-type from morphology and/or electrophysiology (Gouwens et al., 2020; Scala et al., 2021; Lee et al., 2023).
- Cross-species transfer for *electrophysiology* to transcriptomic subclass (Neuroinformatics, 2026), and cross-species *comparison* of morpho-electric properties for L1 (Chartrand et al., 2023).
- Morphology representation benchmarks on a single species and region (Laturnus et al., 2020), and unsupervised morphology maps in mouse (Weis et al., 2025).

**What is missing (checked against 2023-2026 literature)**

1. No study has trained a *morphology-only* t-type/subclass classifier on mouse Patch-seq interneurons and evaluated it on human MTG Patch-seq interneurons (or vice versa) under a harmonised taxonomy - the morphology analogue of the 2026 electrophysiology transfer.
2. Laminar position is a strong proxy for subclass in both species (soma depth alone predicts Lamp5 vs. Pvalb reasonably well); no morphology-to-t-type study has reported how much of the apparent morphological information is *positional* (depth, layer) rather than *shape*, and whether shape information transfers across species while positional information does not (human layers are thicker and L2/3 is expanded).
3. Representations have not been compared under distribution shift: hand-crafted morphometrics, 2-D density maps, persistence-based summaries and self-supervised graph embeddings differ in how they encode absolute scale (human interneurons are larger), which is exactly what breaks cross-species transfer.
4. Dataset/laboratory effects (Allen mouse V1 vs. Tolias-lab mouse M1 vs. Allen human MTG) are confounded with species in a single-pair comparison; a three-dataset design with leave-dataset-out evaluation can separate "lab shift" from "species shift".

Related projects in this repository: `morphology-to-electrophysiology` (predicting electrophysiological features from morphology) and `neuromorpho-scaling-laws` (lab effects on morphometrics). This project is self-contained and focuses on transcriptomic identity.

## Research questions / hypotheses

1. **H1 (within-species ceiling).** Morphology-only balanced accuracy for the five harmonised subclasses is > 0.6 in mouse V1 (n ~ 500 reconstructions) and > 0.5 in human MTG under stratified 5-fold CV; finer t-types are near chance after subclass is accounted for. Test: balanced accuracy with permutation null (labels shuffled within dataset), 1,000 permutations.
2. **H2 (transfer gap).** Train-on-mouse, test-on-human balanced accuracy drops by at least 0.15 relative to the within-human ceiling; feature standardisation per species plus CORAL alignment recovers at least half of the gap. Test: paired bootstrap over test cells.
3. **H3 (shape vs. position).** Removing soma depth and layer features reduces within-species accuracy by less than 0.1 but changes the transfer gap materially; scale-invariant features (normalised by total length / extent) transfer better than absolute ones. Test: ablation with bootstrap CIs.
4. **H4 (lab vs. species shift).** Leave-dataset-out transfer between the two mouse datasets (Allen V1, Tolias M1) loses less accuracy than mouse-to-human transfer of the same representation; the difference quantifies the species-specific part of the shift.
5. **H5 (representation ranking is not shift-invariant).** The best representation within species (expected: learned graph embedding) is not the best under transfer (expected: normalised density maps or persistence summaries); Kendall's tau between within- and cross-species rankings < 0.5.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Institute mouse V1 Patch-seq (Gouwens et al., 2020) | SWC reconstructions, soma depth/layer, t-type and MET-type labels, mapping confidence | ~4,200 cells with transcriptomes, ~500 with morphology | Open (Allen Cell Types / Brain Image Library; NWB electrophysiology on DANDI) | https://portal.brain-map.org/explore/classes/multimodal-characterization , https://celltypes.brain-map.org |
| Allen Institute human MTG Patch-seq interneurons (Lee et al., 2023) | SWC reconstructions, layer, t-type mapped to the MTG taxonomy | hundreds of cells with morphology | Open (Allen portal, DANDI for ephys; morphologies via portal / BIL) | https://portal.brain-map.org/explore/classes/multimodal-characterization |
| Tolias-lab mouse M1 Patch-seq "mini-atlas" (Scala et al., 2021) | SWC reconstructions (dendrite + axon), t-type labels, soma depth | ~1,300 cells with morphology | Open (GitHub / Zenodo release) | https://github.com/berenslab/mini-atlas |
| Human MTG snRNA-seq taxonomy (Hodge et al., 2019) and cross-species consensus taxonomy (Bakken et al., 2021) | Mapping of t-types to harmonised subclasses | taxonomy tables | Open | https://portal.brain-map.org/atlases-and-data/rnaseq |
| Laturnus et al. 2020 benchmark data | Reference representations and their evaluation code for the mouse V1 morphologies | ~600 cells | Open (Zenodo) | https://zenodo.org/records/3716519 |

No credentialed data are involved.

## Methods

1. **Assembly** (`scripts/download_data.py`, `data/README.md`). Download the Allen Patch-seq metadata tables and SWC files (allensdk / portal downloads), the mini-atlas repository, and the taxonomy tables. Keep for every cell: dataset, species, region, layer, normalised soma depth, t-type, mapping confidence, reconstruction completeness (axon present?).
2. **Taxonomy harmonisation** (`src/morph_ttype/taxonomy.py`). Parse t-type names into marker tokens and map to five harmonised subclasses (Pvalb, Sst, Vip, Lamp5, Sncg/PAX6) plus "other"; Sst Chodl and Meis2 handled explicitly; keep the finer t-type for within-species analyses. Human PAX6 types are aligned to mouse Sncg following the consensus taxonomy; this choice is a sensitivity parameter.
3. **Representations** (`src/morph_ttype/features.py`). (a) Hand-crafted morphometrics per neurite type (length, branch points, tips, order, Sholl, extents, asymmetry); (b) depth-normalised 2-D density maps (XZ projection with the depth axis expressed in cortical-thickness units) at two resolutions; (c) persistence-style branch length distributions (radial distance at branch start/end); (d) scale-normalised versions of (a)-(c); (e) optional learned embedding (GraphDINO-style; external code) exported as a feature matrix.
4. **Transfer** (`src/morph_ttype/transfer.py`). Per-dataset standardisation, CORAL second-order alignment (Sun, Feng & Saenko, 2016, *AAAI*), logistic regression / random forest classifiers (scikit-learn), evaluation designs: within-dataset stratified CV; leave-dataset-out; mouse-to-human; ablations (no position; scale-normalised only).
5. **Tools**: `numpy`, `scipy`, `pandas`, `scikit-learn`, optional `torch` + `torch-geometric` for the graph embedding, `allensdk` for downloads.

## Evaluation & statistics

- Primary metric: balanced accuracy and macro-F1 over harmonised subclasses; secondary: per-class recall, confusion matrices, log-loss with calibration (transfer typically miscalibrates).
- Validation: stratified 5-fold within dataset; leave-dataset-out for transfer; no cell appears in both training and test; mapping-confidence threshold (e.g. > 0.7) applied identically to all datasets; ambiguous-mapping cells reported separately.
- Nulls: label permutation within dataset (within-species chance); for transfer, permutation of the *training* labels (does the mouse model carry any human-relevant structure?); position-only baseline (depth + layer) as the reference floor.
- Multiple comparisons: five pre-registered hypotheses; representation x design grid summarised with bootstrap CIs, not p-values; Kendall's tau for H5.
- Confounds: axon reconstruction availability differs between datasets - all analyses run for "dendrite-only" and "dendrite+axon" subsets; shrinkage/slice thickness recorded and used as covariates; class imbalance handled by class weights.

## Publishable angle

- **Headline**: "Morphology alone recovers interneuron subclass at X balanced accuracy in mouse and Y in human; Z% of the apparent information is laminar position; scale-normalised shape features transfer across species with a gap of only W, whereas learned embeddings that win within species fail to transfer." Plus a public feature/embedding table for ~2,000 Patch-seq interneurons in one harmonised taxonomy.
- Target venues: *Neuroinformatics*, *PLoS Computational Biology*, *eLife* (Tools & Resources), *Cerebral Cortex*.
- Follow-ups: adding electrophysiology (a joint morphology+ephys transfer) and comparing with the 2026 ephys-only transfer; applying the transfer model to unlabeled human interneuron reconstructions on NeuroMorpho.org and EM datasets (H01, MICrONS) to estimate subclass composition; region-to-region transfer within species.

## Risks, confounds & mitigations

- **Small human n** for some subclasses (Sncg/PAX6, Sst Chodl). Mitigation: report per-class recall with CIs; pool rare classes into "other" in the primary analysis.
- **Taxonomy mapping ambiguity** (human PAX6 vs. mouse Sncg/Lamp5). Mitigation: sensitivity analysis with alternative mappings; report results excluding ambiguous classes.
- **Incomplete axons** in human slices (thicker tissue, larger cells). Mitigation: dendrite-only primary analysis.
- **Reconstruction protocol differences** across the three datasets are partly confounded with species; the three-dataset design (H4) bounds the lab component.
- **Overfitting of learned embeddings** to the mouse dataset: freeze embeddings; never fine-tune on human labels for the transfer condition.

## Milestones

- [ ] Assemble the three Patch-seq morphology sets with harmonised metadata; taxonomy table with audit.
- [ ] Hand-crafted features and density maps for all cells; QC (completeness, scale).
- [ ] Within-species baselines and permutation nulls (H1).
- [ ] Mouse-to-human and leave-dataset-out transfer (H2, H4); CORAL and standardisation ablations.
- [ ] Position vs. shape ablation (H3); scale-normalised representations.
- [ ] Optional learned embedding; representation ranking comparison (H5).
- [ ] Preprint, code, feature tables released.

## Ethics / data-use notes

- Allen Institute data are released under the Allen Institute terms of use (citation required); human Patch-seq tissue is de-identified neurosurgical material collected under the original IRB approvals; no identifiable information is handled.
- The mini-atlas data are released by the authors under an open licence; cite Scala et al. (2021).
- Do not commit SWC files, metadata tables or embeddings; `data/` and `outputs/` are git-ignored.
