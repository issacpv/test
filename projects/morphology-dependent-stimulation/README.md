# Morphology-Dependent Stimulation Thresholds

**Do hundreds of real cortical reconstructions polarize and fire like the handful of canonical model neurons that the brain-stimulation dosing literature is built on — and is the residual morphological variability large enough to swamp the individualized E-field differences that dosing protocols try to correct for?**

## Status / difficulty / timeline / compute

| | |
|---|---|
| Status | Design complete; starter code runs on synthetic morphologies |
| Difficulty | MSc thesis to early-PhD. Needs comfort with cable theory and linear algebra, not wet-lab skills |
| Timeline | 6–9 months (2 months data + pipeline, 2 months sweeps, 2 months statistics, 2 months writing) |
| Compute | Modest. The numpy path solves one morphology's full orientation sweep in <1 s on a laptop (three sparse solves, then a matrix product). 500 morphologies × 6 waveforms ≈ 1 CPU-hour. The optional NEURON calibration subset (~40 cells × bisection) is ~10 CPU-hours; a single 8-core desktop suffices. No GPU. |
| Storage | <5 GB (SWC files are kilobytes; head models dominate) |

## Background

Every form of brain stimulation in clinical use — tDCS, tACS, TMS, DBS — delivers a **field**, not a spike. Whether a given neuron responds depends on how that field couples to its geometry. The coupling physics is old and well understood: a neurite polarizes where the extracellular potential has non-zero curvature along its axis, so bends, branch points and terminals are the sites that depolarize, and a cell aligned with the field polarizes far more than one perpendicular to it (Rattay, 1986, *IEEE Trans Biomed Eng*). Uniform-field slice experiments put concrete numbers on it: roughly 0.2 mV of somatic polarization per V/m in rat hippocampal pyramidal cells, with the sign set by somato-dendritic orientation (Bikson et al., 2004, *J Physiol*), and cell-type-dependent thresholds under uniform fields in cortex (Radman et al., 2009, *Brain Stimulation*).

The modelling field then made a decisive move: couple a macroscopic FEM head model to morphologically detailed neuron models, and predict which cortical element TMS actually activates. Aberra et al. (2018, *J Neural Eng*) built human-scaled, biophysically realistic cortical neuron models; Aberra et al. (2020, *Brain Stimulation*) coupled them to a TMS head model and showed that **axon terminals**, not somata or dendrites, are the lowest-threshold elements, with thresholds ordered by layer and cell type. Shirinpour et al. (2021, *Brain Stimulation*) packaged this multi-scale approach into a reusable toolbox. On the tES side, multi-scale models of axonal and dendritic polarization in realistic head geometry have extended the same logic to DC fields (bioRxiv 2023.08.23.554447), and in-vivo work confirmed that somato-dendritic orientation determines the sign of tDCS modulation in awake animals (bioRxiv 2023.02.18.529047, peer-reviewed version 2025).

This body of work is the basis for a fast-growing clinical claim: that **individualized E-field modelling is the route to individualized dose**. If we know each patient's field, we know each patient's neural effect. That inference has a hidden premise — that the neurons receiving the field are interchangeable enough that field magnitude is the dominant source of variance. Nobody has measured whether that premise holds.

## The research gap

**What has been done.**

- Aberra et al. (2018, *J Neural Eng*) and Aberra et al. (2020, *Brain Stimulation*) establish the canonical pipeline and the layer/cell-type threshold ordering for TMS. Their morphological variability comes from a small set of Blue-Brain-derived models (on the order of 25 cells, with clone families per layer and type), scaled to human dimensions.
- Shirinpour et al. (2021, *Brain Stimulation*) provide tooling for single-neuron and subcellular TMS responses.
- **Trotter, Pariz, Hutt & Lefebvre (2025, *Frontiers in Synaptic Neuroscience*, doi:10.3389/fnsyn.2025.1621352)** is the closest existing work and it partially closes the naive version of this project. Using uniform-field stimulation of Blue-Brain cortical models, they report a **null result**: no physical trait yielded layer- or cell-type-specific responses that survived significance testing, because within-type morphological variability blurred the between-type differences. This is exactly the right question, and it means the project below must not be framed as "does morphology matter" — that is answered.
- Recent histologically-informed multiscale modelling (2024–2025) has begun introducing measured variability into *morphological parameters* of model neurons, rather than using real reconstructions.

**What is specifically missing.** Four things, and they compound:

1. **Real reconstructions at population scale, not clone families.** All of the above vary morphology either within a small canonical set or by perturbing parameters of a generative model. NeuroMorpho.Org holds tens of thousands of real reconstructions; the variability in a clone family is a model artefact, the variability across archives is biological plus methodological. Nobody has run the field-coupling calculation across hundreds of independently traced cortical cells.
2. **Human versus mouse, with the lab confound actually modelled.** Species claims in this literature rest on *scaling* rodent models to human dimensions (Aberra et al., 2018), not on human reconstructions. Real human and mouse reconstructions come from largely disjoint sets of contributing labs with different fixation, shrinkage-correction and tracing conventions, so a naive species comparison is confounded by archive. A random intercept for archive is the minimum defence, and to our knowledge no stimulation-modelling paper has applied it.
3. **Cross-modality rank transfer.** Aberra-style work treats one modality at a time. If a morphology is hard to activate with TMS, is it also hard with tDCS? The waveform enters only through a membrane low-pass factor, so the *ranking* of morphologies should be nearly modality-invariant — an easily falsifiable prediction that nobody has tested, and one with direct practical value (it would mean a single morphological susceptibility index generalizes across devices).
4. **The dosing comparison that matters.** The width of the threshold distribution across morphologies has never been placed on the same axis as the between-subject spread of E-field magnitude at a target from head models. Until it is, "individualized dosing" is an engineering aspiration with no denominator.

**The sharpened angle.** Trotter et al. (2025) showed morphological variability *blurs cell-class specificity*. We take the complementary and more actionable step: quantify the **magnitude** of that variability in real reconstructions and compare it head-to-head against the E-field variability that individualized dosing corrects. The headline is a variance budget, not a specificity test.

## Research questions and hypotheses

1. **How wide is the threshold distribution within a cortical morphological class?**
   *H1:* Across real human cortical pyramidal reconstructions, minimum activation threshold spans ≥ 3-fold (interquartile range ≥ 0.2 dex) after controlling for archive — i.e. within-class morphological variability is not a small correction.
2. **Is that spread larger than the between-subject spread in E-field dose?**
   *H2 (headline):* The within-class morphological IQR of threshold exceeds the reported between-subject IQR of E-field magnitude at a fixed cortical target for a fixed stimulator output (which is roughly 1.5–2-fold in the head-modelling literature). If true, field-based individualization alone cannot account for most of the variance in who responds.
3. **Do cell class and layer predict threshold once morphological variability is admitted?**
   *H3:* Consistent with Trotter et al. (2025), the probability of superiority (AUC) for pyramidal-versus-interneuron threshold is < 0.70 — a real mean difference with heavy distributional overlap, so cell class is a poor predictor at the single-cell level. We test this in real rather than model-generated morphologies.
4. **Human versus mouse.**
   *H4:* Human cortical reconstructions have systematically lower thresholds than mouse (larger cells, longer electrotonic reach, hence more polarization per V/m), with a ratio of median thresholds whose 95% cluster-bootstrap CI excludes 1. *Falsification route:* if the archive random intercept absorbs the effect, the species claim is a lab artefact and we report it as such.
5. **Does the morphological ranking transfer across modality?**
   *H5:* Spearman ρ between per-morphology thresholds under tDCS and under TMS is > 0.9, and orientation anisotropy (max/min threshold over field direction) is modality-invariant, because the waveform enters only as a scalar attenuation.
6. **Orientation.**
   *H6:* Anisotropy ratio is larger for pyramidal than for interneuron morphologies (dominant apical axis versus more isotropic arborization), making pyramidal responses more sensitive to the field-direction errors that coil placement introduces.
7. **Disease and age (exploratory).**
   *H7:* Reconstructions annotated with Alzheimer's disease or epilepsy have higher thresholds than controls from the same archive (dendritic regression reduces electrotonic reach). Powered only for a within-archive comparison; reported as hypothesis-generating.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| NeuroMorpho.Org v8+ | Human and mouse cortical pyramidal / interneuron SWC reconstructions (CNG-standardised), plus metadata: species, brain region, cell type, archive, experiment condition, age classification | Target 300–800 usable cells; files are 10–500 kB each | **Open**, no registration; REST API | https://neuromorpho.org/api.jsp |
| Allen Cell Types Database | Human (~413 cells) and mouse (~1920 cells) reconstructions with paired electrophysiology; perisomatic and all-active biophysical model packages for the NEURON calibration arm | ~100 human + ~100 mouse reconstructions used; model packages ~10 MB each | **Open**, no registration; `allensdk` | https://celltypes.brain-map.org |
| SimNIBS example dataset (`ernie`, MNI152) | Cortical surface normals and the empirical distribution of E-field direction and magnitude at targets, used to weight the orientation sweep by *physiologically reachable* field directions | ~3 GB (ships with SimNIBS) | **Open** with SimNIBS install | https://simnibs.github.io/simnibs/build/html/dataset.html |
| MIDA head model | Higher-resolution anatomy for a sensitivity check on field-direction priors | ~1 GB | **Free registration** (IT'IS Foundation) | https://itis.swiss/virtual-population/regional-human-models/mida-model/ |
| Blue Brain / Aberra model neurons *(optional)* | Published reference morphologies with complete axonal arbors, used to bound the error introduced by NeuroMorpho's missing axons | ~25 cells | **Open** (Zenodo) | https://zenodo.org/records/2488573 |

No credentialed or patient-identifiable data is used. Nothing here requires a DUA.

## Methods

**Step 1 — cohort construction.** `scripts/download_data.py` queries the NeuroMorpho REST API (`/api/neuron/select?q=species:human&q=brain_region:neocortex&q=cell_type:pyramidal,interneuron`, paginated at 100/page) and downloads CNG-standardised SWC files. Cohorts: human cortical, mouse cortical, and a disease/aging arm keyed on `experiment_condition`. Inclusion criteria applied after parsing: ≥ 200 nodes, total neurite length ≥ 500 µm, exactly one connected component after root repair, no coordinate duplicates exceeding 5% of nodes. Record the archive for every cell — it is the grouping variable for all inference.

**Step 2 — cable model.** `morph_stim.swc_cable` builds one compartment per SWC node and assembles the axial conductance Laplacian `L` (siemens) and leak conductance `D`. Under the quasi-uniform approximation the extracellular potential is `phi_e(x) = -E·x`, and the steady-state polarization solves

```
(L + D) v = -L phi_e
```

a sparse symmetric positive-definite system (`scipy.sparse.linalg.spsolve`). The right-hand side is the discrete activating function. Passive parameters: `Ra = 150 Ω·cm`, `Rm = 30 000 Ω·cm²`, `Cm = 1 µF/cm²`, with a myelination multiplier available for axonal compartments.

**Step 3 — orientation sweep.** Because the system is linear in `E`, three solves (one per cartesian unit field) give an `(n, 3)` sensitivity matrix in mV per V/m; any of 128 Fibonacci-lattice directions is then a matrix product. `morph_stim.thresholds` converts polarization to a threshold as `delta_v_th / (attenuation × peak depolarization at excitable sites)`, with `attenuation` the first-order membrane low-pass factor for the waveform: 1 for DC, `1/sqrt(1+(ωτ)²)` for tACS, `1 - exp(-pw/τ)` for TMS (0.2 ms) and DBS (60 µs). Reported per cell: minimum and median threshold over directions, best direction, anisotropy ratio, threshold parallel and perpendicular to the cell's own principal axis, and effective polarization length in µm.

**Step 4 — NEURON calibration.** `morph_stim.neuron_driver` rebuilds the same geometry in NEURON (Hines & Carnevale, 1997, *Neural Computation*) with active channels and `e_extracellular`, and bisects on field amplitude for a true spiking threshold. Run on ~40 cells stratified by species, class and surrogate-threshold decile. Report Spearman ρ, the log-log calibration slope and the median fold error. **The population claims are stated in terms of rank and relative spread, which survive a biased but monotone surrogate; absolute V/m values are reported only for the calibrated subset.**

**Step 5 — field-direction prior.** Extract cortical surface normals and E-field vectors from the SimNIBS `ernie` model at candidate targets (M1, DLPFC) to obtain the distribution of *achievable* field directions. Re-weight the orientation sweep by this prior, so the reported thresholds are the ones a real coil or montage could deliver rather than the unconstrained optimum.

**Step 6 — statistics.** `morph_stim.popstats` fits `log10(threshold) ~ species * cell_class + log10(total_length) + layer + (1 | archive)` with `statsmodels` `MixedLM`, falling back to OLS with archive-clustered standard errors when the random intercept is not identifiable. Coefficients are exponentiated to fold changes. Variance is partitioned across species, class and archive; the human/mouse median ratio gets a **cluster bootstrap resampling archives**, not cells.

**Baselines to beat.** (a) A single canonical pyramidal morphology, the current field default — show its threshold sits at an arbitrary percentile of the real distribution. (b) Cell class + layer alone, as a predictor of threshold. (c) Total neurite length alone, the simplest morphometric — does anything else add explanatory power beyond size?

**Tooling.** numpy, scipy.sparse, pandas, statsmodels, requests; `neuron` and `allensdk` optional; `simnibs` optional for step 5.

## Evaluation and statistics

- **Primary outcome:** IQR and 10th–90th percentile range of `log10(threshold_min)` within human cortical pyramidal cells, after archive adjustment. Reported in dex and as a fold range.
- **Headline comparison (H2):** that range placed side by side with the between-subject IQR of E-field magnitude at a matched target from the head-modelling literature, with the comparison stated as a ratio of variances and the literature value cited rather than re-derived.
- **Class discrimination (H3):** Cohen's *d* **and** probability of superiority (AUC) for every class pair. An AUC near 0.5 is reported as an informative null, not a failure.
- **Validation scheme:** archive-held-out cross-validation for any predictive model of threshold from morphometrics — split by *archive*, never by cell, since cells from one archive share slices, tracing software and an operator.
- **Leakage prevention:** the largest risk here is not train/test leakage but **clone leakage** — several NeuroMorpho entries can be reconstructions of the same cell, or siblings from one slice. Deduplicate on (archive, experiment, coordinates hash) and on pairwise morphometric near-identity before splitting.
- **Multiple comparisons:** the confirmatory family is H1–H6 (6 tests); Benjamini–Hochberg at q = 0.05 across that family. H7 and all per-layer breakdowns are explicitly exploratory and reported unadjusted with that label.
- **Nulls:** (i) *orientation null* — shuffle field directions across cells to confirm the anisotropy statistic is not a geometric artefact of the sampling lattice; (ii) *archive permutation null* — permute species labels within archive to confirm the species effect is not carried by lab; (iii) *size-matched null* — resample human and mouse cells matched on total neurite length, to separate "human cells are bigger" from "human cells are shaped differently".
- **Sensitivity analyses, all pre-registered:** `delta_v_th` ∈ {8, 12, 20} mV; excitable-site definition ∈ {axon terminal, any terminal, soma}; `Rm` ∈ {10 k, 30 k, 60 k} Ω·cm²; myelin factor ∈ {1, 100}; with and without cells lacking axons. Conclusions must survive all of them or be stated conditionally.

## Publishable angle

**Headline result that would make a paper.** *"Across several hundred real human cortical reconstructions, the within-class spread of stimulation threshold is X-fold — larger than the between-subject spread of E-field dose that individualized modelling corrects. Morphology, not field magnitude, is the dominant unmodelled source of variance in single-neuron response, and the canonical model neuron used throughout the dosing literature sits at the Nth percentile of the real distribution."* Secondary results: a modality-invariant morphological susceptibility index (H5), and the honest human-versus-mouse ratio with the lab confound removed (H4).

**Target venues.**
- *Journal of Neural Engineering* — the natural home for the cable-modelling methodology and the dosing implication.
- *Brain Stimulation* — where the Aberra and Shirinpour papers live; highest-impact fit if the dosing framing lands.
- *Imaging Neuroscience* (MIT Press) — receptive to variance-budget and methodological-audit work, open access.
- *PLOS Computational Biology* — fallback with room for the full sensitivity-analysis apparatus.

**Follow-ups.** (1) Fold the threshold distribution into a head model as a spatially varying susceptibility prior, turning an E-field map into a predicted *activated-fraction* map. (2) Repeat with patch-seq-linked Allen morphologies to ask whether transcriptomic type predicts susceptibility better than morphological class. (3) Extend to DBS with a genuinely non-uniform near-field solution, where the quasi-uniform approximation fails and the morphological effect should be larger still.

## Risks, confounds and mitigations

| Risk | Why it bites | Mitigation |
|---|---|---|
| **NeuroMorpho reconstructions mostly lack complete axons** | Axon terminals are the lowest-threshold element (Aberra et al., 2020). A dendrite-only reconstruction gives the wrong absolute threshold and possibly the wrong ranking. **This is the single biggest threat to the project.** | Stratify the whole analysis by axon completeness (`frac_axon_nodes`, reported per cell). Run the primary analysis on the axon-bearing subset; report the dendrite-only subset separately as a lower-resolution replication. Bound the bias by comparing against Blue Brain / Aberra models that do have full axons. Pre-register that the headline claim is about *relative* spread within a completeness stratum. |
| **Archive confounded with species** | Human and mouse cells come from different labs; shrinkage correction and tracing conventions differ systematically | Random intercept for archive in every model; within-archive species comparison where any archive has both; archive-permutation null; cluster bootstrap over archives |
| Polarization-threshold surrogate is not a real spike threshold | Absolute V/m values would be wrong | NEURON calibration subset; claims framed in rank and relative terms; sensitivity analysis on `delta_v_th` |
| Quasi-uniform approximation fails for DBS | Near-field gradients are steep within ~2 mm of a contact | State the validity radius explicitly; treat DBS results as applicable only to the far field, or drop DBS to a supplementary analysis |
| Shrinkage and fixation artefacts inflate apparent size differences | Diameters especially are unreliable across archives | Report results with and without diameter renormalization to an archive-specific median; make diameter a sensitivity axis, not a headline |
| Selection bias in which cells get reconstructed | Labs trace complete, well-filled, often large cells | Acknowledge as a bound on generalization; compare the morphometric distribution against Allen's systematically sampled cells as a reality check |
| Layer metadata is sparse and free-text | Layer is a covariate in H3 | Parse conservatively (`_infer_layer`), treat missing as its own level, never guess |
| Trotter et al. (2025) or a 2026 paper closes more of the gap | The specificity question is already answered | The framing is deliberately the variance budget and the dosing denominator, not specificity. Re-check the literature at write-up and cite as concurrent work |

## Milestones

- [ ] **M1 (week 3)** — NeuroMorpho fetcher running; human + mouse cortical cohorts downloaded; inclusion criteria applied; cohort table with per-archive counts and axon-completeness distribution
- [ ] **M2 (week 6)** — Cable model validated: reproduce the analytic `lambda = sqrt(d·Rm/(4·Ra))` polarization length on straight cables and the ~0.2 mV/(V/m) somatic figure of Bikson et al. (2004) on a pyramidal morphology
- [ ] **M3 (week 10)** — Orientation sweeps complete for all cohorts × 6 waveforms; tidy results table written to `outputs/`
- [ ] **M4 (week 14)** — NEURON calibration on 40 stratified cells; calibration report (ρ, slope, fold error) in hand
- [ ] **M5 (week 18)** — Mixed models, variance partition, cluster bootstrap, all three nulls, full sensitivity grid
- [ ] **M6 (week 22)** — SimNIBS field-direction prior integrated; H2 variance-budget figure drafted
- [ ] **M7 (week 26)** — Disease/aging exploratory arm; pre-registration of the confirmatory family retrospectively documented
- [ ] **M8 (week 32–36)** — Manuscript; code and results archived on Zenodo with the exact NeuroMorpho query strings and retrieval dates

## Ethics and data-use notes

- All primary data here is **open and non-human-identifiable at the level used**. NeuroMorpho and Allen reconstructions are published, de-identified single-cell geometries. No credentialed access, no DUA, no IRB requirement for secondary use of these resources.
- **Human tissue provenance.** Human cortical reconstructions derive from neurosurgical resections and post-mortem tissue collected under the originating studies' consent and ethics approvals. Cite the source publications for every archive used (NeuroMorpho's `/api/literature` endpoint gives the references) — attribution is the data-use condition these repositories ask for.
- **Be polite to shared infrastructure.** NeuroMorpho.Org is an academic service. Keep the request delay in `NeuroMorphoClient` (default 0.5 s), cache aggressively, and never parallelize the crawl. Record the retrieval date: the database is versioned and grows.
- **Never commit data.** `data/` and `outputs/` are gitignored. Commit the query strings and the manifest, not the SWC files — the point is that anyone can re-derive the cohort from the API.
- **PhysioNet-style restrictions do not apply to this project**, but note the general rule for the wider programme: credentialed data (MIMIC and other PhysioNet resources) may not be sent to third-party LLM APIs except as permitted by PhysioNet's responsible-use policy. Nothing in this repository is subject to that constraint, and nothing here should be mixed with data that is.
- If the SimNIBS or MIDA head models are used, follow their own licences; MIDA requires registration and is not redistributable.
