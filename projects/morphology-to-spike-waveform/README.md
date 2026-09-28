# morphology-to-spike-waveform

**How much of the diversity of extracellular spike waveforms on high-density probes is explained by dendritic morphology rather than ion-channel composition? A factorial forward-modelling study on the Allen Cell Types reconstructions and biophysical models, validated against opto-tagged Neuropixels units, that yields morphology-conditioned waveform priors for cell-type identification.**

## Status / difficulty / timeline / compute

- Status: design + starter code (SWC parsing and morphology features, a line-source extracellular forward model with Neuropixels 1.0/2.0/Ultra geometries, spatiotemporal waveform features, cross-validated morphology-to-waveform regression and a factorial variance partition). No data is shipped.
- Difficulty: MSc-level (feature/forward-model track) to PhD-level (full NEURON simulations with the Allen all-active models). The starter code runs without NEURON; the full study needs NEURON + LFPy/BMTK.
- Timeline: 6-9 months (1 month data + morphology features, 2 months simulations with the Allen models, 1 month waveform features and regression, 1 month opto-tagged validation, 1-2 months writing).
- Compute: a workstation for the analytic forward model; a small cluster or 32+ cores for the NEURON simulations (about 500 morphologies x several channel-density sets x 3 probe geometries x several placements; each simulation is seconds to minutes).

## Background

The shape of an extracellular action potential (EAP) depends on the transmembrane currents and on the morphology through which the return currents flow (Gold, Henze, Koch & Buzsaki, 2006 J Neurophysiol; Pettersen & Einevoll, 2008 Biophys J). High-density probes (Neuropixels: Jun et al., 2017 Nature; Neuropixels 2.0: Steinmetz et al., 2021 Science; Neuropixels Ultra: Ye et al., 2025 Neuron) record the spatiotemporal footprint of each unit across tens of sites, exposing dendritic back-propagation and propagation along the axon (Jia et al., 2019 J Neurophysiol). Cell-type identification from EAPs is a long-standing goal: narrow vs broad spikes separate putative fast-spiking interneurons from pyramidal cells; WaveMAP (Lee et al., 2021 eLife) finds more classes; Beau et al. (2025 Cell) reach >95% accuracy in cerebellum with opto-tagged ground truth and a semi-supervised classifier on waveform, discharge statistics and layer; Ye et al. (2025 Neuron) show that opto-tagged PV, SST and VIP interneurons in mouse visual cortex have large, distinct spatial footprints (medians 40, 35, 42 um) and that many are not narrow-spiking. Buccino et al. (2018 J Neurophysiol) combined biophysical modelling with deep learning for MEA-based localisation and classification; a 2022 modelling study showed that cortical pyramidal and parvalbumin cells exhibit distinct spatiotemporal extracellular potentials.

On the morphology side, the Allen Cell Types Database (Gouwens et al., 2019 Nature Neuroscience; Gouwens et al., 2020 Cell) provides hundreds of mouse and human reconstructions with matched patch-clamp recordings, and the Allen biophysical models (Gouwens et al., 2018 Nature Communications, perisomatic and all-active; Nandi et al., 2022 Cell Reports, 9,200 models for 230 cells) put channel densities on those morphologies. Wei et al. (2023 Nature Communications) linked in-vitro, in-vivo (Neuropixels) and in-silico classes in mouse V1.

## The research gap

What has been done:

- EAP classifiers trained on in-vivo waveforms with opto-tag labels (Lee et al., 2021; Beau et al., 2025; Ye et al., 2025): no morphology, so they cannot say why a class has its footprint.
- Small-scale simulation studies (a few morphologies, one probe) showing that morphology shapes the EAP (Gold et al., 2006; Pettersen & Einevoll, 2008; Buccino et al., 2018; the 2022 pyramidal-vs-PV modelling study).
- Morphology-to-intracellular-electrophysiology prediction (a 2020 bioRxiv machine-learning study on the Allen Cell Types database) and the multimodal in-vitro/in-vivo class associations of Wei et al. (2023).
- The related project `morphology-to-electrophysiology` in this repository predicts intracellular features from morphology; the present project is about the extracellular waveform on a probe.

What is specifically missing (our angle):

1. A **factorial design** that separates morphology from channel composition: simulate EAPs with (a) each morphology under its own fitted all-active model, (b) each morphology with channel densities transplanted from other cells of the same and of different classes, and (c) each channel set on other morphologies. This gives a variance partition (morphology, channels, interaction, probe placement) for each waveform feature.
2. **Scale**: all reconstructed Allen mouse (and human) morphologies with models, three probe geometries (NP1.0, NP2.0, NP Ultra), and a systematic placement grid, rather than a handful of cells.
3. **Which morphological features matter**: cross-validated regression of EAP features (trough-to-peak, spatial footprint, decay exponent, propagation velocity, above/below asymmetry) on interpretable morphology features (soma size, dendritic surface area within 50/100/200 um, apical trunk diameter, polarity, axon initial segment position), with permutation importance and negative controls.
4. **Validation against in-vivo ground truth**: compare the simulated footprint distributions per class with the opto-tagged PV/SST/VIP units of Allen Visual Coding (NP1.0) and the NP Ultra opto-tagging of Ye et al. (2025), including the question of why many opto-tagged interneurons are not narrow-spiking.
5. **Morphology-conditioned priors**: a table of expected waveform-feature distributions per morphological type and probe geometry that can be used as priors in cell-type classifiers or as a plausibility check on sorted units.

## Research questions / hypotheses

1. **RQ1 (variance partition).** For each waveform feature, what fraction of the variance across simulated cells is due to morphology vs channel densities? H1: spatial features (footprint, decay exponent, asymmetry) are > 60% morphology; temporal features (trough-to-peak, repolarisation slope) are > 60% channels, with a large interaction for the peak-to-trough amplitude ratio.
2. **RQ2 (features).** H2: dendritic surface area within 100 um of the soma and the number of primary dendrites predict the footprint and amplitude decay (cross-validated R^2 > 0.5); apical presence and trunk diameter predict the above/below asymmetry.
3. **RQ3 (probe geometry).** H3: the morphology-explained fraction increases from NP1.0 to NP Ultra (denser sites resolve the footprint); trough-to-peak is invariant to the probe within 5%.
4. **RQ4 (in-vivo validation).** H4: the simulated footprint distributions for aspiny (Pvalb/Sst-like) and spiny morphologies reproduce the opto-tagged footprint medians of Ye et al. (2025) within 10 um, and the simulated fraction of broad-spiking PV-like models under morphology transplants matches the observed fraction of broad-spiking opto-tagged PV units.
5. **RQ5 (priors).** H5: adding morphology-conditioned priors to a waveform-only classifier improves opto-tagged class accuracy (Visual Coding opto-tagging) by >= 5 points at equal false-positive rate.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| Allen Cell Types Database (mouse + human) | SWC morphologies (~600 mouse, ~150 human reconstructed), intracellular waveform features, cell-type metadata (Cre line, dendrite type, layer) | ~2 GB (SWC + NWB) | Open (AllenSDK `CellTypesCache`; API) | https://celltypes.brain-map.org/ |
| Allen biophysical models (perisomatic + all-active; Gouwens et al., 2018) | Channel densities on the same morphologies (NEURON hoc/json) | ~ a few hundred models | Open (AllenSDK `BiophysicalApi`) | https://celltypes.brain-map.org/data?models |
| Nandi et al. (2022) all-active model ensembles | 9,200 models for 230 cells (channel-set transplants) | ~1 GB | Open (DANDI / GitHub per the paper's data availability) | https://www.cell.com/cell-reports/fulltext/S2211-1247(22)00989-5 |
| Allen Visual Coding - Neuropixels opto-tagging sessions | In-vivo mean waveforms across channels for opto-tagged Sst/Pvalb/Vip units (NP1.0) | ~1 TB for all sessions; ~20 GB for opto sessions | Open (AllenSDK; S3) | https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html |
| Neuropixels Ultra opto-tagging (Ye et al., 2025 Neuron) | Ultra-dense footprints of opto-tagged interneurons | tens of GB | Open (DANDI per the paper's data availability) | https://dandiarchive.org/ |
| NeuroMorpho.org | Additional mouse V1 morphologies for held-out morphology tests | via REST API | Open | https://neuromorpho.org/ |

## Methods

Pipeline (modules in `src/morph2wave/`):

1. **Morphology features** (`swc.py`). Parse SWC; compute soma radius, number of stems, total length and surface area by compartment type, dendritic surface area within radial shells (0-50, 50-100, 100-200, 200-400 um), dendritic polarity vector and asymmetry index, apical trunk presence/diameter, axon initial segment origin (distance and side), and a compartmentalised representation for the forward model.
2. **Forward model** (`eap_forward.py`). Line-source approximation of the extracellular potential from compartment transmembrane currents in a homogeneous, isotropic medium (sigma = 0.3 S/m; Holt & Koch, 1999). Two current sources: (a) full NEURON simulations of the Allen models exported as compartment currents (LFPy/BMTK; the recommended path), and (b) a fast phenomenological "propagating spike" current template (somatic Na/K current with distance-dependent delay and attenuation into dendrites) for large factorial sweeps and for tests. Probe geometries: NP1.0 (checkerboard, 20 um row pitch), NP2.0 (15 um pitch, 2 columns), NP Ultra (6 um pitch, 8 columns). Placement grid: soma-to-shank distance 10-100 um, depth offsets, rotation about the shank axis.
3. **Waveform features** (`waveform_features.py`). On the peak channel: trough-to-peak, half-width, repolarisation and recovery slopes, peak/trough ratio. Spatially: footprint (number of sites with |amplitude| > 12% of max and an equivalent Gaussian spread), amplitude decay exponent vs distance, above/below asymmetry, trough-latency propagation velocity along the shank.
4. **Regression and variance partition** (`regression.py`). Cross-validated ridge and gradient-boosted regression from morphology features to waveform features with grouped CV by cell; permutation importance; a two-way factorial variance partition (morphology x channel-set) with placement as a nested factor.
5. **In-vivo comparison**. Extract mean spatiotemporal waveforms of opto-tagged units (AllenSDK `session.mean_waveforms`), compute the same feature set, and compare distributions with the simulated ones per class.

Tools: AllenSDK, NEURON + LFPy 2.x or BMTK (for the full simulations), numpy/scipy, scikit-learn, pandas, NeuroM (optional morphology checks).

## Evaluation & statistics

- Regression: grouped 5-fold CV by cell (all placements of a cell in the same fold); report R^2 and RMSE with CIs over folds; negative control with shuffled morphology-feature rows.
- Variance partition: sum-of-squares decomposition with bootstrap CIs over cells; report eta^2 for morphology, channels, interaction, placement.
- Probe comparison: paired differences per cell across geometries.
- In-vivo validation: two-sample comparisons of feature distributions (simulated class vs opto-tagged class) with energy distance and permutation tests; calibration of the simulated footprint scale checked on the Ye et al. medians before any tuning.
- Multiple comparisons: BH-FDR over the waveform-feature panel.
- Sensitivity: sigma (0.2-0.4 S/m), line-source vs point-source, inclusion of the probe shank as an insulating boundary (method of images), temperature/channel-kinetics variants of the models.

## Publishable angle

Headline: "Across hundreds of reconstructed neurons, spatial EAP features are set mainly by dendritic morphology while temporal features are set by channel composition; morphology explains why many opto-tagged interneurons are broad-spiking, and morphology-conditioned priors improve waveform-based cell-type identification." A morphology-to-waveform lookup table per probe geometry would be immediately useful to every Neuropixels lab.

Target venues: Journal of Neuroscience; PLOS Computational Biology; eNeuro; Neuron (if the in-vivo validation is strong); Cosyne/SfN.

Follow-ups: human cortex (Allen human reconstructions + human Neuropixels recordings); axonal-arbor contributions using full axon reconstructions; use the forward model to generate training data for spike-sorting benchmarks (MEArec-style) with realistic cell-type mixtures.

## Risks, confounds & mitigations

- Reconstructions truncate dendrites and axons at slice boundaries; mitigation: use only "full" reconstructions per Allen QC flags, model slice-cut effects explicitly, and report robustness with NeuroMorpho V1 cells.
- Channel-density transplants can produce models that do not spike; mitigation: verify each transplant fires a single AP under the standard stimulus and drop failures (report the failure rate per class).
- Forward-model assumptions (homogeneous medium, no probe shank); mitigation: sensitivity to sigma and an image-charge shank model; compare amplitude scales against in-vivo distributions before feature-level comparisons.
- In-vivo waveforms are affected by spike sorting (merged/split units, drift); mitigation: use only isolated units with standard QC and compare distributions, not individual units.
- Opto-tagged classes are defined transcriptomically, morphology classes anatomically; mitigation: map via Cre line and dendrite type (aspiny/spiny), and via Gouwens et al. (2020) met-types where available.

## Milestones

- [ ] `--sample` downloads 10 mouse morphologies + ephys features; SWC features computed
- [ ] Analytic forward model validated against a NEURON/LFPy simulation for 5 cells (amplitude and footprint within 20%)
- [ ] Full NEURON export of compartment currents for all Allen all-active models
- [ ] Factorial simulation grid (morphology x channel set x probe x placement) run and stored
- [ ] Waveform-feature tables; regression and permutation importance
- [ ] Variance partition with CIs; probe-geometry comparison
- [ ] Opto-tagged in-vivo comparison (Visual Coding NP1.0; NP Ultra)
- [ ] Morphology-conditioned priors and classifier test
- [ ] Manuscript + release of the lookup tables and simulation code

## Ethics / data-use notes

- All data are open, secondary-use animal (and de-identified human surgical tissue) data released by the Allen Institute under its Terms of Use; cite the primary papers and the AllenSDK.
- No credentialed data; nothing under `data/` is committed (see `.gitignore`).
- Simulation outputs can be large; share summary tables and the code to regenerate them rather than raw traces.
