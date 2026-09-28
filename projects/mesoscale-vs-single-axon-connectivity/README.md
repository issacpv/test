# mesoscale-vs-single-axon-connectivity

**How well do Allen mesoscale (bulk-tracer) projection maps predict the targets of individual axons from the same source region? A brain-wide single-neuron vs. bulk-tracer concordance benchmark with a per-region "projection heterogeneity index", and its consequences for connectome-based brain network models.**

## Status / difficulty / timeline / compute

- Status: proposal + starter code (allensdk connectivity fetcher, SWC axon → CCFv3 region assignment, concordance metrics, heterogeneity index with independent-sampling null and bootstrap CIs).
- Difficulty: MSc/early-PhD; heavy on data wrangling (coordinate frames, ontologies) and statistics, light on ML.
- Timeline: 6-9 months.
- Compute: CPU only. The 25 µm CCF annotation volume is ~0.5 GB in memory; a few thousand SWC files (each up to ~10^5 nodes) process in minutes. Optional network-model simulations (The Virtual Brain) add a few CPU-days.

## Background

The Allen Mouse Brain Connectivity Atlas (Oh et al., 2014, Nature; Harris et al., 2019, Nature; voxel model of Knox et al., 2019, Network Neuroscience) is the de-facto structural connectome for mouse computational neuroscience: it parameterises whole-brain network models such as The Virtual Mouse Brain (Melozzi et al., 2017, eNeuro; Melozzi et al., 2019, PNAS), voxel-level network analyses (Coletta et al., 2020, Sci Adv) and many spiking/mean-field models. Bulk anterograde tracing (AAV injections of ~10^2-10^3 neurons) yields *population-averaged* projection densities: it cannot say whether a source region's 30% projection to target A and 20% to target B come from the same neurons (divergent, multi-target axons) or from separate subpopulations (parallel channels).

Single-neuron full-morphology datasets registered to CCFv3 (Wang et al., 2020, Cell) now make this decomposition possible: MouseLight (Economo et al., 2016, eLife; Winnubst et al., 2019, Cell; >1,000 neurons), SEU-ALLEN (Peng et al., 2021, Nature; 1,741 neurons; expanded in later releases), the ION prefrontal and hippocampal projectomes (Gao et al., 2022, Nat Neurosci; Qiu et al., 2024, Science), a ~20,000-neuron cortical projectome analysis (Neuron, 2025) and MAPseq/BARseq barcoded projections (Kebschull et al., 2016, Neuron; Han et al., 2018, Nature). These consistently show that single neurons project to idiosyncratic subsets of the population's targets.

## The research gap

**What has been done**

- Winnubst et al. (2019) and Peng et al. (2021) qualitatively compared single-neuron target sets to bulk projection patterns and described divergent vs. convergent projection motifs; Muñoz-Castañeda et al. (2021, Nature) related single-neuron and bulk data within motor cortex.
- Fei et al. (2024, Nat Commun) used public AAV bulk tracing *and* single-neuron tracing to characterise interhemispheric (callosal) connections, proposed a "heterogeneity" metric for connection-pattern complexity, and showed with wide-field imaging that heterogeneous projections yield weaker but more stable functional connectivity. This is the closest work, but it is restricted to interhemispheric cortico-cortical projections and to a single metric without a sampling-corrected null.
- Nat Methods 2025 ("Reconstruction of a connectome of single neurons in mouse brains by cross-validating multi-scale multi-modality data") built a single-neuron bouton-based connectome from 1,877 neurons and found single-neuron connections correlate more strongly with gene co-expression than the full mesoscale connectome — a comparison at the network-statistics level, not a per-region predictive benchmark.
- "Connectivity of single neurons classifies cell subtypes" (Nat Methods, 2025), the Neuron 2025 cortical projectome and the Cell 2026 motor-cortex "projection-defined modules" paper use single-neuron data to *define* types; none asks how predictable a neuron's target set is from the bulk map of its source region.
- MAPseq studies (Han et al., 2018) quantified target combinatorics within visual cortex but without registration to the same CCF regions used by the Allen bulk atlas.

**What is missing (verified by 2023-2026 searches)**

1. A brain-wide, per-source-region *predictive benchmark*: treating the bulk projection vector as a classifier score for "does neuron n target region t?", with AUROC / precision-recall / weighted rank correlation, across all source regions with ≥ 10 reconstructed neurons.
2. A sampling-aware null: single neurons drawn *independently* from the bulk distribution already produce heterogeneous target sets (random sampling maximises motif diversity). Deviation from that null is what distinguishes structure from random divergence: a *deficit* of heterogeneity means neurons cluster into a limited set of projection motifs (parallel channels / subpopulations), an *excess* means mutually exclusive targeting beyond chance. No published heterogeneity metric includes this null or corrects for unequal neuron counts across regions (rarefaction).
3. A "how many neurons recover the bulk map?" analysis (subsampling curves of pooled single-neuron axon length vs. bulk density) per region, with its dependence on cell type (Cre line / soma layer) and on the axon-length vs. terminal-count weighting.
4. Quantified consequences for network models: re-parameterising a TVB-style mouse model with single-neuron-derived edge weights (and with heterogeneity-scaled edge variance) and measuring the change in simulated functional connectivity and in graph metrics against the bulk-parameterised model.

## Research questions / hypotheses

1. **H1 (bulk predicts targets only coarsely).** Using bulk normalised projection volume of the source region as a score, per-neuron AUROC for target membership is > 0.7 on average but with a wide range across regions; precision at k = 5 targets is < 0.5, i.e. the majority of bulk "top targets" are not targeted by any given neuron.
2. **H2 (motif structure beyond random sampling).** In most cortical source regions, the observed pairwise Jaccard distance between neurons falls *below* the independent-sampling-from-bulk null (z < −3): neurons cluster into a limited set of projection motifs (parallel channels) rather than sampling targets at random from the bulk map. The deficit is larger for cortical L5 ET / L2-3 IT populations than for thalamic relay neurons (which we expect to be close to the null). Regions with z > 3 (mutually exclusive targeting) are predicted to be rare and to coincide with known anti-correlated pathways (e.g. striatum-projecting vs. brainstem-projecting subclasses).
3. **H3 (recovery curve).** Pooling n single neurons recovers the bulk vector with Spearman rho > 0.8 for n ≈ 30-100 in most regions; the required n scales with the heterogeneity index.
4. **H4 (model consequences).** Replacing bulk-derived edge weights with single-neuron-derived weights changes simulated FC (TVB reduced-Wong-Wang or Wilson-Cowan) by a Frobenius-normalised difference > 0.2 for regions with high heterogeneity index, and shifts hub rankings (Kendall tau < 0.8).
5. **H5 (bias direction).** Bulk projection density over-represents targets reached by fibres of passage and by high-bouton-density collaterals; single-neuron terminal counts vs. axon length weighting reveal this systematically (terminal-based concordance < length-based concordance for white-matter-adjacent targets).

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Mouse Brain Connectivity Atlas (allensdk `MouseConnectivityCache`) | Wild-type and Cre-line anterograde experiments; structure unionizes → summary-structure projection matrix (projection density / normalised projection volume, ipsi/contra); CCFv3 annotation (25 µm) and structure tree | ~3,000 experiments; ~300 summary structures | Open, no registration | https://connectivity.brain-map.org , https://allensdk.readthedocs.io |
| Allen voxel-scale connectivity model (`mcmodels`, Knox 2019) | Predicted projection vector for an arbitrary soma voxel (finer than region-level injections) | 100 µm voxel model | Open | https://github.com/AllenInstitute/mouse_connectivity_models |
| MouseLight (Janelia) | Complete single-neuron axons in CCFv3 µm coordinates (JSON/SWC; nodes carry Allen structure id) | >1,000 neurons (Winnubst 2019 and later releases) | Open | https://ml-neuronbrowser.janelia.org , also on NeuroMorpho.org |
| SEU-ALLEN full morphologies (Peng et al., 2021 and BICCN releases) | Full single-neuron reconstructions (fMOST) registered to CCFv3 | 1,741 (2021), thousands in later BICCN releases | Open (Brain Image Library; NeuroMorpho.org) | https://www.brainimagelibrary.org , https://neuromorpho.org |
| ION single-neuron projectomes (Gao 2022 PFC; Qiu 2024 hippocampus) | Registered single-neuron axons for additional regions | thousands | Open per the papers' data-availability statements | see papers |
| MAPseq / BARseq target tables (Han et al., 2018) | Independent barcoded single-cell projection matrices for cross-validation in visual cortex | thousands of cells | Open (paper supplements) | see paper |

## Methods

1. **Bulk projection vectors** (`src/meso_vs_axon/allen_connectivity.py`): for each source summary structure, average `normalized_projection_volume` (and `projection_density`) over wild-type experiments whose injection is centred in the structure (injection fraction ≥ 0.5), split ipsi/contra; alternative: Cre-line experiments matched to the single-neuron dataset's Cre line; alternative: `mcmodels` voxel model evaluated at each neuron's soma voxel (removes injection-site mismatch).
2. **Single-neuron target vectors** (`ccf_assign.py`): axon nodes mapped to CCFv3 annotation voxels (25 µm), collapsed to summary structures via the ontology ancestor map, split ipsi/contra relative to the soma; two weightings: axon length per target and terminal (tip) count per target; normalised to fractions; binarised at ≥ 1% of axon length or ≥ 2 terminals.
3. **Concordance** (`concordance.py`): per neuron — Jaccard with the binarised bulk vector, Spearman and weighted Kendall tau (top-weighted), AUROC and precision/recall at k using the bulk vector as score; per region — pooled single-neuron vector vs. bulk correlation; subsampling curves; label-permutation null.
4. **Heterogeneity index** (`heterogeneity.py`): mean pairwise Jaccard distance among neurons of a region (PHI), motif entropy, divergence (targets per neuron), bulk-explained fraction; independent-sampling null (each neuron draws its observed number of targets without replacement with probabilities ∝ bulk vector); null-deviation z-score (negative = parallel channels, positive = mutually exclusive targeting); neuron-level bootstrap CIs; rarefaction to a common n across regions.
5. **Cell-type stratification**: soma layer (from CCF layer annotation) and Cre line (SEU-ALLEN metadata) as strata; compare PHI within vs. across strata to separate "type mixture" from "within-type divergence".
6. **Network-model consequences**: build region × region weight matrices from (a) bulk, (b) pooled single neurons, (c) bulk with per-edge variance from PHI; simulate with The Virtual Brain (mouse connectome pipeline of Melozzi 2017) and compare FC and graph metrics.
7. **Tools**: allensdk, mcmodels, numpy/scipy/pandas, scikit-learn (AUROC), tvb-library (optional), networkx.

## Evaluation & statistics

- Primary estimands per source region: mean per-neuron AUROC (95% neuron-bootstrap CI), precision@5, weighted tau; PHI with CI; null-deviation z (two-sided); n_50 (neurons needed for rho ≥ 0.8 with the bulk vector).
- Nulls: (a) independent sampling from bulk (H2); (b) target-label permutation (destroys region identity; H1 floor); (c) bulk vector from a *different* source region (specificity check: the correct region's bulk vector should predict better than a random region's).
- Leakage/validity: exclude single neurons whose soma is outside the injection structure or within 100 µm of its boundary; exclude bulk experiments with injection fraction < 0.5 or with reported leakage; ipsi/contra assignment checked against the MouseLight per-node `allenId` field.
- Multiple comparisons: BH-FDR across regions for null-deviation tests; effect sizes reported everywhere.
- Sensitivity: 10 vs. 25 µm annotation; summary structures vs. finer (layer-level) parcellation; length vs. terminal weighting; wild-type vs. Cre-matched bulk; MouseLight vs. SEU-ALLEN dataset (dataset as a covariate — different labelling, imaging and registration pipelines).

## Publishable angle

- **Headline**: "Bulk tracer maps predict individual axon targets with AUROC ≈ X but precision ≈ Y; in N of M source regions single neurons are organised into far fewer projection motifs than random sampling of the bulk map would produce, revealing parallel projection channels invisible to population tracing; re-parameterising a whole-brain mouse model with single-neuron weights changes simulated FC by Z." Plus a released per-region heterogeneity table for modellers.
- Target venues: *Nature Communications*, *PLoS Computational Biology*, *Network Neuroscience*, *Cell Reports* (resource-style), *eNeuro* (if narrower).
- Follow-ups: cross-validate with MAPseq/BARseq combinatorics; extend to bouton-level (synapse-proxy) weighting using the 2025 bouton-net; human/NHP mesoscale vs. single-axon once data exist.

## Risks, confounds & mitigations

- **Registration error** of single neurons to CCF (tens to hundreds of µm): use summary structures (coarse), boundary-exclusion buffers, and MouseLight's own per-node `allenId` as a consistency check.
- **Injection-site mismatch** (bulk injections cover multiple layers/types; single neurons come from specific layers/Cre lines): Cre-matched bulk vectors and the voxel model evaluated at soma position; report both.
- **Incomplete axon reconstructions** (thin collaterals missed): terminal counts are more sensitive than length; report both and use completeness proxies (total axon length, number of tips) as covariates.
- **Unequal n per region**: rarefaction to common n; regions with < 10 neurons excluded from primary analysis.
- **Ontology drift** across CCF versions (CCFv3 2017 vs. later annotations): pin `annotation/ccf_2017` and structure graph id 1 in allensdk.
- **Selection bias** in single-neuron datasets (bright, complete neurons preferred): compare soma distributions to bulk injection coverage; state as a limitation.

## Milestones

- [ ] Bulk projection matrix (wild-type, ipsi/contra, summary structures) cached; injection-fraction QC.
- [ ] Single-neuron ingest: MouseLight JSON + SEU-ALLEN SWC → CCF target vectors; agreement with MouseLight `allenId` > 95% of nodes.
- [ ] Concordance benchmark for all regions with ≥ 10 neurons (H1); specificity null.
- [ ] Heterogeneity index, independent-sampling null and null-deviation z, bootstrap CI, rarefaction (H2).
- [ ] Recovery curves and n_50 per region (H3).
- [ ] Cell-type stratification; length vs. terminal weighting (H5).
- [ ] TVB re-parameterisation experiment (H4).
- [ ] Preprint, per-region table and code release.

## Ethics / data-use notes

- All data are from mice under the depositing institutions' IACUC approvals; only open, de-identified datasets are used. Cite the Allen Institute Terms of Use, MouseLight (Janelia) and BICCN/BIL data-use policies and the source publications.
- Do not commit annotation volumes, unionize caches, SWC/JSON files; `data/` is git-ignored.
