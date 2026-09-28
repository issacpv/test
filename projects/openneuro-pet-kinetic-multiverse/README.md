# openneuro-pet-kinetic-multiverse — How much of a PET binding-potential result is the kinetic-modelling pipeline? A multiverse and test-retest audit across OpenNeuro PET-BIDS datasets

Run every defensible combination of reference-tissue model (SRTM, SRTM2, Logan reference, MRTM2, late-frame SUVR), reference-region definition, t*, frame weighting, scan truncation and k2′ handling on the dynamic, reference-region-tracer datasets shared on OpenNeuro, and quantify (i) how much between-subject and between-region variance in BP_ND is attributable to analytic choices, (ii) which choices maximize test-retest reliability, and (iii) whether published group/region contrasts survive the specification curve.

## Status / difficulty / timeline / compute

- Status: design + starter code (BIDS-PET sidecar/TAC handling, SRTM/SRTM2/Logan/MRTM2/SUVR implementations in pure numpy/scipy with a forward simulator, a multiverse runner with specification-curve summaries and variance decomposition, and test-retest reliability metrics).
- Difficulty: MSc-level (6-9 months) if restricted to shipped/derived TACs; PhD-chapter (9-12 months) if the preprocessing stage (motion correction, coregistration, PVC) is added as multiverse factors.
- Compute: kinetic fits are cheap (seconds per subject per specification); a 2,000-specification multiverse over ~300 scans × 20 regions is a few CPU-hours. Preprocessing with PETPrep/petsurfer (FreeSurfer-based) is ~1-3 CPU-hours per scan.
- Related project in this repository: `fmri-pipeline-multiverse` (same philosophy for fMRI). This folder is self-contained.

## Background

Quantitative PET outcome measures (non-displaceable binding potential BP_ND, distribution-volume ratio DVR) come from kinetic models applied to regional time-activity curves (TACs). For tracers with a reference region, the standard choices are the simplified reference tissue model (SRTM; Lammertsma and Hume, 1996, NeuroImage; basis-function implementation Gunn et al., 1997, NeuroImage), SRTM2 with a fixed reference efflux constant k2′ (Wu and Carson, 2002, JCBFM), Logan graphical analysis with a reference region (Logan et al., 1996, JCBFM), the multilinear reference tissue models MRTM/MRTM2 (Ichise et al., 2003, JCBFM) and simple late-frame ratios (SUVR). Each has tunable choices — t*, frame weighting, k2′ source, scan duration, reference-region delineation (cerebellar grey vs whole cerebellum, eroded or not) — and each is known to be biased under specific assumption violations (Salinas et al., 2015, JCBFM). The consensus nomenclature (Innis et al., 2007, JCBFM) standardized names but not pipelines.

The reproducibility literature in fMRI has shown that analytic flexibility alone can flip conclusions (Botvinik-Nezer et al., 2020, Nature) and that multiverse/specification-curve analyses (Steegen et al., 2016, Perspectives on Psychological Science; Simonsohn et al., 2020, Nature Human Behaviour) make that flexibility visible. PET has had one such effort for *preprocessing*: Nørgaard et al. (2019, NeuroImage; 2020, JCBFM) ran ~1,000 preprocessing pipelines on a single-site [11C]DASB dataset and showed that preprocessing choices change both reliability and the conclusions of a group comparison. Kinetic-model choices were not the object of that study, the data were not public, and one tracer was used.

## The research gap

What has been done:

- Preprocessing multiverse on one in-house [11C]DASB dataset (Nørgaard et al., 2019, 2020): preprocessing choices change conclusions; kinetic model fixed.
- Tools for reproducible modelling exist: kinfitr (Tjerkaski et al., 2020, EJNMMI Research), which implements 14 models with a common syntax; PETPrep, petsurfer-bids and related BIDS apps under the OpenNeuroPET initiative ("Building an open ecosystem for molecular neuroimaging", 2026), which standardize preprocessing. They enable, but have not yet produced, a cross-dataset multiverse.
- Open data: PET-BIDS (Nørgaard et al., 2022, Scientific Data) and the OpenNeuro PET portal now list ~32 PET-BIDS datasets (as of the 2026 ecosystem paper), several dynamic with reference-region tracers and some with test-retest designs.
- Reliability guidance: Matheson (2019, PeerJ) argued that test-retest studies should report ICC-based reliability to inform study design; per-model reliability comparisons exist tracer by tracer in the older literature but not as a systematic multiverse on shared data.

What is specifically missing:

1. A kinetic-modelling multiverse (model × reference region × t* × weighting × truncation × k2′ handling) run identically on *multiple public* dynamic datasets and tracers, with the results released as a re-usable table.
2. Variance decomposition: what fraction of BP_ND variance across scans is "pipeline" versus "subject" versus "region", per tracer, and which single factor dominates (t*? model? reference region?).
3. Reliability-optimal specifications: for datasets with test-retest scans, which specifications maximize ICC and minimize within-subject variability, and whether the reliability-optimal choice matches the field's default.
4. Robustness of published findings: for datasets attached to a published contrast (group difference, age effect, regional ordering), the fraction of specifications reproducing the sign and significance.
5. Interaction with scan duration: how much scan time (60 vs 90 min) can be cut before reliability drops, per model — directly actionable for study design.

## Research questions / hypotheses

1. RQ1 (variance). Across dynamic reference-tissue datasets on OpenNeuro, what share of BP_ND variance is attributable to specification factors? H1: 10-30% of total variance; model family and t* are the largest factors; reference-region definition is the largest for tracers with cerebellar off-target signal.
2. RQ2 (bias structure). Are model differences constant offsets (harmless for contrasts) or BP-dependent (harmful)? H2: Logan and SUVR are BP-dependently biased (noise-induced underestimation at high BP for Logan; flow/wash-out dependence for SUVR), producing specification-dependent slopes in region-wise comparisons.
3. RQ3 (reliability). Which specification maximizes test-retest ICC? H3: SRTM2/MRTM2 with population k2′ and frame-duration weighting is most reliable; SUVR is reliable but biased; ICC differences between best and default specifications exceed 0.05 in at least one tracer.
4. RQ4 (conclusions). For datasets with a published contrast, what fraction of specifications reproduces it? H4: sign is reproduced in > 90% of specifications, but statistical significance in < 60% for medium-effect contrasts.
5. RQ5 (duration). At what truncation does ICC fall by > 0.05 relative to full length? H5: 60 min suffices for SRTM-type models for fast tracers ([11C]raclopride-like), not for slow tracers.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| OpenNeuro PET-BIDS datasets (dynamic, reference-region tracers) | 4D PET, `_pet.json` (TracerName, FrameTimesStart, FrameDuration, decay correction), T1w for ROI definition, participants.tsv; test-retest sessions where present | ~32 PET datasets listed on the OpenNeuro PET portal (2026); roughly a third are dynamic with reference-tissue tracers; tens of subjects each | Open (CC0); GraphQL API / openneuro-py / DataLad / S3 | https://openneuro.org/pet , https://github.com/OpenNeuroDatasets |
| ds004869 (example) | [11C]MC1 (COX-2) dynamic PET, NIMH | tens of subjects | Open | https://openneuro.org/datasets/ds004869 |
| ds004230 (example) | Dynamic PET dataset used as the PETPrep worked example; verify tracer and blood data in its `dataset_description.json` | tens of subjects | Open | https://openneuro.org/datasets/ds004230 |
| OpenNeuroPET example/phantom datasets | Pipeline validation | small | Open | https://github.com/openneuropet |
| Cimbi database (NRU Copenhagen) | Additional [11C]DASB, [11C]SB207145, [11C]Cimbi-36 and other dynamic scans with test-retest subsets, for replication | hundreds of scans | Application to NRU | https://nru.dk/index.php/cimbi-database |

The downloader enumerates OpenNeuro PET datasets through the API and classifies each by tracer (reference-region tracers, e.g. raclopride, DASB, SB207145, Cimbi-36, AZ10419369, PiB/FTP with cerebellar reference vs. tracers requiring arterial input) and by whether it is dynamic (more than one frame). The analysis uses only dynamic, reference-region datasets; arterial-input datasets are an optional extension with 2TCM added to the multiverse.

## Methods

1. Discovery and ingestion (`scripts/download_data.py`, `pet_multiverse.tacs`): list PET datasets, download sidecars and participants tables, classify tracers, then download 4D PET and T1w for the selected datasets. Frame timing from `FrameTimesStart`/`FrameDuration` (seconds → minutes, mid-frame times); check `ImageDecayCorrected`.
2. Regional TACs: PETPrep or petsurfer-bids (FreeSurfer-based ROIs, optional PVC), or the built-in `extract_tacs` on a label image. Reference regions as multiverse levels: cerebellar grey (FreeSurfer), whole cerebellum, eroded cerebellar grey (2 mm), and — for [11C]raclopride-like tracers — a supervised-cluster reference if available.
3. Models (`pet_multiverse.models`): SRTM via basis functions (grid over k2a), SRTM2 (k2′ fixed to a population value derived from SRTM in a high-binding region), Logan reference with and without the k2′ term, MRTM and MRTM2, SUVR over late windows. All are pure numpy/scipy, frame-integrated on a fine time grid, with optional frame-duration weighting.
4. Multiverse (`pet_multiverse.multiverse`): specification grid → tidy results (dataset, subject, session, region, specification, BP_ND, fit diagnostics); specification-curve summaries; variance decomposition by nested ANOVA (subject, region, specification factors, residual) with η² per factor.
5. Reliability (`pet_multiverse.reliability`): ICC(2,1)/ICC(3,1), within-subject coefficient of variation, absolute variability, Bland-Altman limits, computed per specification for datasets with test-retest sessions; "reliability specification curve".
6. Conclusion robustness: for each dataset with a published contrast, fit the contrast under every specification; report the fraction with the same sign and p < 0.05, and the distribution of effect sizes.
7. Optional preprocessing factors (motion correction on/off, PVC on/off, coregistration target) via PETPrep flags for a subset, to place the kinetic-model variance next to the preprocessing variance reported by Nørgaard et al.

Libraries: numpy, scipy, pandas, statsmodels; nibabel for images; optional kinfitr (R) as an external validation of the model implementations on the same TACs; PETPrep/petsurfer via containers.

## Evaluation and statistics

- Model implementation validation: recover known parameters from simulated TACs (forward SRTM with noise) within 5% for SRTM/SRTM2/MRTM2 and characterize Logan/SUVR bias vs noise; cross-check against kinfitr on ≥ 1 real dataset (report Lin's concordance).
- Variance decomposition: mixed/nested ANOVA with η² and 95% bootstrap CIs (resampling subjects); report separately per tracer.
- Reliability: ICC with 95% CIs (F-based); comparisons between specifications by bootstrap over subjects.
- Conclusion robustness: specification curves with joint inference by permutation (shuffle group labels / ages within dataset; Simonsohn et al., 2020).
- Leakage/hygiene: k2′ population values estimated within dataset with leave-one-subject-out when used for that subject; t* chosen per specification, never per subject post hoc; all specifications pre-registered as a grid before any results are inspected.
- Multiple comparisons: the specification curve is descriptive; the joint permutation test provides the single inferential statement per dataset.

## Publishable angle

Headline: "On open PET data, the kinetic-modelling pipeline accounts for X% of the variance in binding potential; t* and reference-region choices dominate; the reliability-optimal specification is not the default; and published contrasts keep their sign but not their p-value across the multiverse." Deliverables: a public specification-level results table for every OpenNeuro reference-tissue dataset, a recommended default per tracer class with reliability evidence, and a reusable BIDS-compatible multiverse runner.

Target venues: Journal of Cerebral Blood Flow & Metabolism; EJNMMI Research; NeuroImage; Journal of Nuclear Medicine (methods); Scientific Data for the results table.

Follow-ups: add arterial-input models (1TCM/2TCM/Logan plasma) for datasets with blood data; voxel-wise parametric-map multiverse; extend to the Cimbi database as a replication set.

## Risks, confounds and mitigations

- Few usable datasets: even 4-6 dynamic reference-tissue datasets across 3-4 tracers suffice for the design; report the sampling frame honestly.
- Heterogeneous preprocessing state of shared data (some datasets share motion-corrected images, some raw): record and treat as a covariate; re-run PETPrep uniformly where possible.
- Reference-region delineation depends on T1 availability and quality: fall back to atlas-based definitions in MNI space; treat as a specification level.
- Decay correction / unit inconsistencies in sidecars: validate units (Bq/mL vs kBq/mL) and decay flags; the loader raises on inconsistencies.
- Model implementation errors: simulation recovery tests and kinfitr cross-check are part of the milestones, not afterthoughts.
- Test-retest scarcity: reliability analyses restricted to datasets with repeat sessions; report n.

## Milestones

- [ ] Dataset discovery table (tracer, dynamic, reference-region, test-retest, blood) from the OpenNeuro API.
- [ ] TAC extraction for selected datasets (PETPrep/petsurfer or built-in), with QC plots.
- [ ] Model validation on simulated TACs and kinfitr cross-check.
- [ ] Multiverse run; variance decomposition per tracer.
- [ ] Reliability specification curves for test-retest datasets.
- [ ] Conclusion-robustness analysis for datasets with published contrasts.
- [ ] Results table release + manuscript.

## Ethics / data-use notes

- OpenNeuro PET data are CC0 but derived from human participants; do not attempt re-identification; respect any dataset-specific citation requests in `dataset_description.json`.
- Cimbi database data require an application and a data-use agreement; no redistribution.
- Never commit imaging data or participant-level TACs from DUA sources; the released results table should contain only specification-level summaries or CC0-derived values.
- Do not upload participant-level data to third-party services.
