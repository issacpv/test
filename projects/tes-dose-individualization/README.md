# tES dose individualization: does the individual electric field explain behavioural response, once spatial and montage nulls are respected?

One-sentence pitch: Re-analyse open tDCS/tACS datasets that ship a T1-weighted MRI, simulate each participant's electric field (E-field) with SimNIBS, and test whether individual target-site E-field predicts the behavioural effect, using spatial (spin/variogram) and "montage" null models that existing E-field–behaviour correlation studies have not used; then derive a normative dose-to-current mapping from large open T1 cohorts.

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (OpenNeuro registry search, analytical sphere E-field toy model, spin-test nulls, dose-response and random-effects meta-analysis).
- Difficulty: MSc/early-PhD level (BME or neuroscience); 6-9 months for the re-analysis paper, +3 months for the normative-dose extension.
- Compute: SimNIBS head modelling (`charm`) takes ~1-2 h per T1 on a single CPU core; 300 heads is a few days on a 16-core workstation. No GPU required. Storage ~2 GB per subject (mesh + fields).
- Related projects (same repo family, kept separate): `tms-target-connectivity` (targeting from connectivity; shares SimNIBS tooling), `imaging-transcriptomics-nulls` (spatial null models), `fmri-pipeline-multiverse`.

## Background

Transcranial electrical stimulation (tDCS/tACS) delivers a fixed current (typically 1-2 mA) regardless of head anatomy, yet finite-element simulations show 2-3-fold inter-individual variability in the cortical E-field for the same montage, driven by skull thickness, CSF volume and gyral geometry (Opitz et al., 2015, NeuroImage; Laakso et al., 2019, Sci Rep; Antonenko et al., 2021, NeuroImage). The physiological threshold for acute network effects is on the order of 0.5-1 V/m (Vöröslakos et al., 2018, Nat Commun), so many participants in 1-2 mA studies may receive sub-threshold fields. This is the leading mechanistic hypothesis for the notorious variability of tES behavioural effects, and it motivates "dose-controlled" stimulation where current is scaled per individual to hit a target E-field (Evans et al., 2020, Brain Stimulation; Caulfield et al., 2020, Brain Stimulation).

## The research gap

What has been done:

- Individual E-field explains part of the physiological response: tACS alpha after-effects (Kasten et al., 2019, Nat Commun), tDCS-induced GABA change (Nandi et al., 2022, Brain Stimulation), fMRI/behaviour in older adults (Antonenko et al., 2019, Brain Stimulation), TMS motor thresholds (Mikkonen et al., 2020, Brain Stimulation).
- Clinical retrospective analyses correlate simulated prefrontal E-field with depression outcomes (Suen et al., 2021, Eur Arch Psychiatry Clin Neurosci, ELECT-TDCS trial).
- Meta-analytic "performance-E-field correlation" maps for working-memory tDCS (Wischnewski et al., 2021, Neurosci Biobehav Rev), which regress study-level effect sizes on template-head E-fields, voxel-wise.
- A 2025 review of personalised tES modelling and optimisation (arXiv 2509.01192) and 2025-2026 work on lesioned heads (medRxiv 2025) and E-field directionality in OCD (Transl Psychiatry, 2026) confirm that individual modelling is standard, but that behaviour-level validation remains sparse and mostly single-study.

What is specifically missing:

1. Voxel-wise/vertex-wise E-field-to-outcome maps (study-level or subject-level) are evaluated with parametric or naive permutation p-values. E-field maps from a given montage are strongly spatially autocorrelated and nearly identical in shape across subjects (they differ mostly in gain), so map-level "significant" regions can be a montage artefact. No published E-field–behaviour analysis we could find uses spatial null models (spin tests, Alexander-Bloch et al., 2018, NeuroImage; variogram-matched surrogates, Burt et al., 2020, NeuroImage; comparison in Markello & Misic, 2021, NeuroImage).
2. No pooled subject-level re-analysis across independent open datasets exists; single studies have N = 10-40 and cannot separate the gain-vs-shape question or estimate the dose-response slope with useful precision.
3. Normative dose-to-current mappings (how many mA are needed for a 0.2 V/m target given age/sex/skull metrics) are available only from small study samples, not from large open T1 cohorts.

Sharpened angle: a pre-registered, multi-dataset re-analysis with (a) subject-level dose-response models, (b) a "montage null" that separates E-field gain from E-field shape, and (c) spatial nulls for any map-level claim; plus (d) a normative dose atlas from ~800 open T1s.

## Research questions / hypotheses

1. H1 (gain): Across open tES datasets with T1 and a behavioural outcome, the individual target-ROI E-field magnitude (V/m per mA x applied current) predicts the within-subject behavioural effect (active minus sham), with a positive pooled slope (random-effects meta-analysis across datasets, p < 0.05, I^2 reported).
2. H2 (shape vs gain): After regressing out the subject's global E-field gain (mean |E| over grey matter), residual shape variation at the target does not add predictive value (likelihood-ratio test of nested mixed models). If it does, that is evidence for focal targeting mattering beyond dose.
3. H3 (null audit): Vertex-wise E-field–behaviour correlation maps evaluated with spin/variogram nulls retain < 25% of the clusters that survive naive permutation or parametric thresholds.
4. H4 (normative dose): In ~800 open T1s (MPI-LEMON, IXI, HCP-style OpenNeuro anatomy) the current required to reach 0.2 V/m at M1 or DLPFC varies 2-fold and is predicted (R^2 > 0.4) by age, sex, skull thickness and CSF fraction.
5. H5 (transportability): A dose model fitted on one cohort predicts the required current in a held-out cohort with MAE < 0.2 mA.

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| OpenNeuro tES datasets (programmatic registry; e.g. ds003670 concurrent EEG/ECG/behaviour with multiple tES doses; ds006126 tDCS visual cortex + motor imagery) | Behavioural dose-response; T1 where present | ds003670: multi-dose tES, EEG/behaviour; ds006126: 5 subjects x 3 sessions | Open (CC0) | https://openneuro.org |
| MPI-Leipzig Mind-Brain-Body (LEMON), OpenNeuro ds000221 | Normative head models + cognitive covariates | ~228 subjects, T1/T2/FLAIR | Open | https://openneuro.org/datasets/ds000221 |
| IXI | Normative T1s across ages | ~600 T1 | Open (CC BY-SA 3.0) | https://brain-development.org/ixi-dataset/ |
| SimNIBS example data ("ernie") | Pipeline tests | 1 head | Open | https://simnibs.github.io/simnibs/ |
| neuromaps / spin-test surfaces (fsaverage spheres) | Spatial nulls | small | Open | https://netneurolab.github.io/neuromaps/ |

Note: at the time of writing, OpenNeuro has few tES datasets that include both T1w and a behavioural outcome; `scripts/download_data.py --search` queries the OpenNeuro GraphQL API for datasets whose name/README mention tDCS/tACS/tES and reports which include `anat` T1w, so the registry (`data/registry.csv`) is built programmatically rather than from memory. Datasets without a T1 are analysed with the template head (MNI152) and enter H1 with a "template" flag; datasets with individual T1s drive H2-H3.

## Methods

1. Registry: query OpenNeuro (`tes_dose.openneuro`) for tES datasets; record modality, N, task, montage (from README/`*_events.json`), whether T1w exists, and behavioural outcome columns. Hand-verify montage and electrode positions; store in `data/registry.csv`.
2. Head models: SimNIBS 4 `charm` per subject T1 (T2 if available). Electrode placement from reported EEG-10-20 positions; conductivities: SimNIBS defaults. Simulate 1 mA; E-fields scale linearly with current (`tes_dose.efield.scale_to_current`).
3. Dose metrics per subject: ROI mean |E| and normal component (E_n) in a target ROI defined a priori (e.g. left M1 hand knob or DLPFC from the study's stated target), 99th percentile |E|, focality (grey-matter volume above 50% of peak), global grey-matter mean |E| (the "gain").
4. Behavioural effect: within-subject active minus sham (or pre/post) standardised by the dataset's control SD; sign-aligned so that positive = intended direction.
5. Dose-response (`tes_dose.dose_response`): per dataset OLS/mixed model effect ~ dose + covariates (age, sex); pooled by DerSimonian-Laird random effects. Nested model for H2 (dose_target vs gain + residual).
6. Map-level analysis: vertex-wise correlation between |E| (fsaverage surface) and behavioural effect across subjects; nulls via `tes_dose.spatial_nulls.spin_test_correlation` (rotations on the sphere) and via a subject-label permutation "montage null" that keeps each subject's map intact.
7. Normative dose atlas: `charm` on LEMON + IXI, simulate two canonical montages (M1-SO, F3-Fp2), compute required current for 0.2 V/m at target; regress on age, sex, skull thickness (from SimNIBS tissue masks) and CSF fraction; cross-cohort validation.
8. Tools: SimNIBS 4.x, nibabel, nilearn, neuromaps, statsmodels, scipy, openneuro-py/datalad.

Toy model shipped with the code: `tes_dose.efield.sphere_efield` is the analytical homogeneous-sphere solution for two surface point electrodes (Legendre series); it is used to unit-test scaling, ROI metrics and the spin test without SimNIBS.

## Evaluation & statistics

- Pre-registration (OSF) of ROI definitions, effect sign conventions and models before running SimNIBS on any dataset with behaviour.
- Slopes with 95% CI; random-effects pooling with I^2 and prediction interval; leave-one-dataset-out sensitivity.
- Nulls: 5,000 spin rotations for surface maps; 5,000 subject-label permutations; report both, and the fraction of naive-significant vertices that survive (H3).
- Multiple comparisons: two ROIs x two dose metrics pre-specified; Holm correction within family; map-level cluster inference with TFCE-free max-statistic over spin nulls.
- Leakage: dose atlas models are fitted on LEMON and tested on IXI (and vice versa); no subject appears in both fit and test sets.
- Equivalence testing (TOST) for "no effect" claims, with the smallest effect size of interest set at r = 0.2.
- Power: with pooled N ~ 150-250 subjects across datasets, 80% power to detect r = 0.2-0.25.

## Publishable angle

Headline result candidates: (i) "Individual E-field gain, not shape, predicts tDCS behavioural response across N open datasets", or the equally publishable null "E-field–behaviour correlations do not survive spatial and montage null models"; (ii) a normative dose-to-current atlas from ~800 open heads with a validated calculator.

Target venues: Brain Stimulation; NeuroImage; Journal of Neural Engineering; Imaging Neuroscience; conference: Brain Stimulation & Imaging Meeting (OHBM satellite).

Follow-ups: extend to tACS/EEG datasets (dose vs entrainment), include lesioned heads (stroke datasets on OpenNeuro), test whether dose-controlled trials (retrospectively) show reduced variance.

## Risks, confounds & mitigations

- Few open tES datasets have both T1 and behaviour: mitigate with template-head analyses flagged separately, and by contacting authors of datasets on Zenodo/OSF; the atlas aim (H4-H5) does not depend on tES datasets.
- Electrode positions are often reported only as 10-20 labels: simulate a sensitivity range (+/- 1 cm) and propagate to dose uncertainty.
- Conductivity uncertainty (skull) dominates absolute |E|: report relative dose (subject/mean) as primary; absolute as secondary.
- Behavioural effects in small datasets are noisy (reliability of the difference score): estimate test-retest reliability where sham-sham sessions exist and attenuate slopes accordingly.
- Global gain and ROI dose are collinear: this is the point of H2; use nested models and report VIFs.
- Segmentation failures in older/pathological heads: automatic QC (`charm` reports) and manual review; exclude with pre-specified criteria.

## Milestones

- [ ] Build and hand-verify the OpenNeuro tES registry (`scripts/download_data.py --search`).
- [ ] Pre-register ROIs, effect conventions and models on OSF.
- [ ] Run `charm` + simulations on all tES datasets with T1 (and template head for the rest).
- [ ] Compute dose metrics and behavioural effects; fit per-dataset and pooled models (H1-H2).
- [ ] Vertex-wise maps with spin and montage nulls (H3).
- [ ] Normative atlas on LEMON + IXI; cross-cohort validation (H4-H5).
- [ ] Release atlas, calculator and code; write manuscript.

## Ethics / data-use notes

- All listed datasets are open; respect each dataset's licence (OpenNeuro CC0; IXI CC BY-SA 3.0, which requires share-alike for derived data such as head models).
- Do not redistribute participant-level T1s or head meshes with the paper; share derived, non-identifiable dose metrics only. Defaced T1s are still potentially re-identifiable; do not upload them to third-party services.
- Never commit data or meshes; `data/` and `outputs/` are git-ignored.
