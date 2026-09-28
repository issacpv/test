# scanner-upgrade-discontinuity — Scanner upgrades as natural experiments: counterfactual atrophy trajectories and a clinical-outcome benchmark for harmonization

Treat the real scanner transitions inside OASIS-3 (Siemens Vision/Sonata 1.5T → TIM Trio 3T → Biograph mMR / Vida) and ADNI (1.5T ADNI-1 → 3T ADNI-GO/2/3, plus site-level scanner swaps) as regression-discontinuity-in-time natural experiments, estimate the counterfactual "no-upgrade" atrophy trajectory of every subject who crossed an upgrade, and benchmark harmonization methods (scanner covariate, cross-sectional ComBat, longitudinal ComBat, ComBat-GAM, image-level DeepHarmony-style correction, and an RD-anchored correction proposed here) on what matters clinically: whether the atrophy rates they produce still predict CDR progression.

## Status / difficulty / timeline / compute

- Status: design + starter code (session/transition tables, RDiT estimator with cluster-robust SEs, placebo and donut checks, cross-sectional and longitudinal ComBat, RD-anchored correction, longitudinal cohort simulator).
- Difficulty: MSc thesis (tabular arm, FreeSurfer outputs only) to PhD chapter (adding the image-level harmonization arm). 6-9 months for the tabular paper.
- Compute: the tabular arm runs on a laptop (OASIS-3 ships FreeSurfer 5.3 outputs; ADNI ships UCSF FreeSurfer tables). Re-running FreeSurfer 7 longitudinal stream on ~3,000 sessions is ~10 CPU-hours per session-pair (optional, cluster). The image-level harmonization arm (a 3D U-Net trained on the OASIS-3 Trio/mMR traveling-subject pairs) needs one 16-24 GB GPU for a few days.

## Background

Longitudinal MRI cohorts outlive their scanners. OASIS-3 spans 30 years of Knight ADRC studies and its MR sessions come from several Siemens platforms (1.5T Vision and Sonata, 3T TIM Trio, the 3T Biograph mMR PET/MR and later 3T systems; the exact model is recorded per session in the OASIS-3 MR session table). ADNI moved from 1.5T (ADNI-1) to 3T (ADNI-GO/2/3) and 44 of 58 sites used more than one scanner or upgraded during the study. The measured effect of an upgrade is not small: in a traveling-subject study of the Siemens TIM Trio → Prisma upgrade, the same brains showed region-specific shifts in cortical thickness (increases in frontal/temporal/cingulate, decreases in parietal) and shifts of several percent in hippocampal, amygdala, striatal and thalamic volumes (Plitman et al., 2021, NeuroImage). Annual hippocampal atrophy in cognitively normal older adults is roughly 0.5-1.5 %/year, so an upgrade-induced offset of 1-3 % is worth one to several years of true aging, and in a mixed-effects model it loads onto the slope of exactly those subjects who were scanned longest.

Harmonization methods exist (ComBat, Fortin et al., 2018, NeuroImage; ComBat-GAM, Pomponio et al., 2020, NeuroImage; longitudinal ComBat, Beer et al., 2020, NeuroImage; DeepHarmony, Dewey et al., 2019, Magnetic Resonance Imaging; CovBat, Chen et al., 2022, Human Brain Mapping) and reviews of them (Hu et al., 2023, NeuroImage). They are validated on traveling subjects (same people scanned on both systems within days) or on the detectability of the batch label, and they are almost never validated on the question a clinical cohort actually asks: after harmonization, is the within-subject atrophy rate across an upgrade unbiased, and does it still predict clinical progression?

## The research gap

What has been done:

- Reliability studies across upgrades and field strengths with repeated scans of the same volunteers: Han et al., 2006, NeuroImage (cortical thickness across field strength, upgrade and manufacturer); Jovicich et al., 2009 and 2013, NeuroImage (subcortical volumes and cortical thickness test-retest across sites/upgrades); Plitman et al., 2021, NeuroImage (TIM Trio → Prisma with volumetric navigators). These quantify offsets in healthy volunteers over days, not in a cohort followed for years with clinical endpoints.
- OASIS-3 itself ships a traveling-subject validation subset: 69 participants scanned on the TIM Trio and the Biograph mMR within two weeks (LaMontagne et al., 2019, medRxiv; OASIS-3 imaging data dictionary). It has been used to describe scanner offsets, not to validate cohort-level harmonization against outcomes.
- Longitudinal ComBat (Beer et al., 2020) was developed on ADNI cortical thickness and shown to control type-I error and improve power for group × time effects relative to cross-sectional ComBat, using scanner (including field strength) as the batch. Its evaluation criterion was statistical (detecting group differences in slopes), not the bias of individual atrophy rates across an upgrade nor the association of those rates with progression.
- DeepHarmony (Dewey et al., 2019) was explicitly built for a scanner/protocol change (paired scans of MS patients before and after) and evaluated by image similarity and segmentation-volume consistency, not on an aging cohort with clinical outcomes.
- Lee et al., 2019, NeuroImage estimated scanner-change effects on whole-brain volume change in MS trials by modelling the change as a nuisance term; the design was not a discontinuity design and no counterfactual per-subject trajectory was reported.
- Regression discontinuity in time (RDiT) is well developed in economics and epidemiology (Hausman and Rapson, 2018, Annual Review of Resource Economics; interrupted time series tutorial, Bernal, Cummins and Gasparrini, 2017, International Journal of Epidemiology; robust local-polynomial inference, Calonico, Cattaneo and Titiunik, 2014, Econometrica). We found no application of RDiT to scanner transitions in a neuroimaging cohort.

What is specifically missing:

1. No study estimates the upgrade effect *in the cohort itself* with a discontinuity design that separates the level shift (additive scanner offset) from a slope change (multiplicative or interaction effect), with placebo cutoffs, donut exclusions and covariate-continuity checks that make the estimate credible in the presence of selection into scanning.
2. No harmonization benchmark uses clinical outcomes as the criterion: after harmonization, does the subject-level hippocampal atrophy rate estimated across an upgrade (a) equal the rate estimated within a single scanner, and (b) predict CDR-SB progression / conversion as well as single-scanner rates do?
3. No "counterfactual trajectory" output for the ~hundreds of OASIS-3 and ADNI subjects who crossed an upgrade, which is what downstream users (brain-age, subtype trajectories, normative models) actually need.
4. The two natural validation subsets (OASIS-3 Trio/mMR 69 traveling subjects; ADNI-1's subset of participants scanned at both 1.5T and 3T at the same visit) have not been used to validate a cohort-derived RDiT estimate against a paired ground truth.

The angle is therefore methodological-epidemiological: a natural-experiment design for a nuisance everyone has, plus a benchmark that grades harmonization by clinical validity instead of by batch detectability.

## Research questions / hypotheses

1. RQ1 (size of the discontinuity). For FreeSurfer hippocampal, amygdala, lateral-ventricle, total cortical GM volumes and lobar mean thickness, what is the level shift at the upgrade in the cognitively normal (CDR = 0) reference group, expressed in % and in "aging-equivalent years" (shift / |CN slope|)? H1: hippocampus 1-3 % (1-3 years), ventricles < 2 %, cortical thickness region-dependent in sign (as in Plitman et al., 2021).
2. RQ2 (additive vs interaction). Does the upgrade change the *slope* (scanner × time interaction) or only the level? H2: level shift dominates; slope change is not distinguishable from zero for volumes, but is present for thickness where partial-volume behaviour differs between 1.5T and 3T.
3. RQ3 (harmonization benchmark). Ranked by (a) residual discontinuity after correction, (b) bias of subject slopes versus single-scanner slopes, (c) preserved CN age-slope and CDR-group × time effects, (d) AUC / C-index of atrophy rate for CDR progression: longitudinal ComBat and the RD-anchored correction ≥ ComBat-GAM ≥ cross-sectional ComBat ≥ scanner covariate only. H3: cross-sectional ComBat removes the offset but shrinks true slopes in subjects with unbalanced pre/post sessions.
4. RQ4 (validation against paired ground truth). Does the RDiT estimate of the Trio → mMR offset from the cohort agree with the paired offset from the 69 traveling subjects within its confidence interval? H4: yes for volumes; thickness offsets agree in sign but the cohort estimate is noisier.
5. RQ5 (selection). Is there discontinuity in covariates (age, sex, CDR, APOE, time since entry) at the cutoff, i.e., were different people scanned after the upgrade? H5: small but non-zero discontinuities in age and CDR in OASIS-3 (enrolment waves), which is why subject fixed effects and donut RD are needed.
6. RQ6 (transportability). Do the ADNI 1.5T → 3T estimates and the OASIS-3 estimates agree after conversion to aging-equivalent years? H6: agreement for hippocampus; disagreement for cortical thickness because ADNI's FreeSurfer version and sequence (MPRAGE vs later accelerated MPRAGE) also changed.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| OASIS-3 (Knight ADRC) | MR session table with scanner model and day-from-entry; FreeSurfer 5.3 `aseg.stats`, `?h.aparc.stats`; ADRC clinical table (CDR, CDR-SB, MMSE, APOE); Trio/mMR traveling-subject pairs | ~1,378 participants, ~2,842 MR sessions, ~2,000 FreeSurfer runs; 69 paired Trio/mMR sessions | Free registration + Data Use Agreement (NITRC / XNAT) | https://sites.wustl.edu/oasisbrains/ , https://www.nitrc.org/projects/oasis3/ |
| ADNI 1/GO/2/3 | `MRILIST.csv` / `MRI3META` and `MRIMETA` (scanner manufacturer, model, field strength per scan, exam date), UCSF FreeSurfer longitudinal tables (`UCSFFSL*`, `UCSFFSX*`), `ADNIMERGE.csv` (CDR-SB, diagnosis, dates) | ~2,000 participants; ADNI-1 subset scanned at both 1.5T and 3T at the same visits | Application via LONI IDA (DUA) | https://adni.loni.usc.edu/ , https://ida.loni.usc.edu/ |
| OASIS-4 (optional) | Clinical memory-clinic cohort, one scanner generation; used as an external distribution check for harmonized hippocampal volumes | 663 participants | Same NITRC DUA | https://www.nitrc.org/projects/oasis4/ |

Neither OASIS-3 nor ADNI may be redistributed. OASIS-3 provides days-from-entry rather than calendar dates, so the OASIS-3 design is a *subject-anchored event-time* RD (time centred on each subject's first post-upgrade session); ADNI provides exam dates, so the ADNI design is a calendar-time RDiT with site-specific upgrade dates derived from the MRI metadata tables.

## Methods

1. Session and transition tables (`scanner_rd.sessions`): parse `OAS30001_MR_d0129` IDs into subject and day, attach scanner model, detect each subject's scanner transitions, define the cutoff as the first post-transition session, compute event time in years, and flag subjects with ≥ 2 sessions on each side. For ADNI, derive site-level upgrade dates from the first date a site's scans carry the new model/field strength, and define the running variable as exam date minus site cutoff.
2. Features: FreeSurfer volumes normalised to eTIV (or expressed as % change from each subject's first session), lobar mean thickness, and the FreeSurfer version held fixed (OASIS-3: 5.3; ADNI: use one UCSF table version per analysis).
3. RDiT estimator (`scanner_rd.rdit`): local-linear regression on both sides of the cutoff with a triangular kernel, subject fixed effects (weighted within-transformation), covariates (age at session, sex, CDR at session), cluster-robust standard errors by subject, cross-validated bandwidth, donut exclusion of sessions within ±d months of the cutoff (protocol shake-down), placebo cutoffs at ±1, ±2 years, and covariate-continuity tests. A random-intercept/random-slope mixed-model version (statsmodels `MixedLM`) is provided as a sensitivity analysis.
4. Counterfactual trajectory: for each transitioned subject, the post-upgrade values minus the estimated level shift (and slope change) from the CDR = 0 reference group, with bootstrap (subject-resampled) uncertainty bands.
5. Harmonization arms (`scanner_rd.harmonize`): (a) scanner as fixed covariate; (b) cross-sectional parametric ComBat fitted on baseline sessions only; (c) longitudinal ComBat (subject random intercept; simplified re-implementation of Beer et al., 2020, with the R reference implementation as cross-check); (d) ComBat-GAM via `neuroHarmonize`; (e) RD-anchored correction: subtract the RDiT jump/slope change estimated in the CN group; (f) optional image-level correction trained on the 69 Trio/mMR pairs, then FreeSurfer re-run. All fitted inside cross-validation folds or on reference groups only, never on the evaluation subjects' outcomes.
6. Clinical validity: per-subject hippocampal atrophy rate (mixed-model BLUP slope) from harmonized data → logistic / discrete-time hazard model of CDR-SB progression (≥ 1 point increase or CDR 0 → ≥ 0.5), compared with rates from single-scanner subjects matched on age, sex, follow-up length and baseline CDR.
7. Simulation (`scanner_rd.simulate`): a longitudinal generator with known slopes by clinical group, additive and multiplicative scanner effects, session-to-session noise and unbalanced transition timing, used for power and bias calibration of all estimators.

Libraries: numpy, pandas, scipy, statsmodels; `neuroHarmonize` (ComBat-GAM cross-check); optional nibabel, FreeSurfer 7 (longitudinal stream), MONAI/PyTorch (image-level arm), `rdrobust` (R or Python port) for robust bias-corrected RD confidence intervals as a second opinion.

## Evaluation and statistics

- Estimands: level shift τ (units and % of CN mean), slope change Δβ (units/year), aging-equivalent years τ/|β_CN|, per-feature. 95 % CIs from cluster-robust SEs and from subject-level bootstrap (2,000 resamples); both reported.
- Credibility checks (pre-registered): placebo cutoffs (expect |τ_placebo| < SE), donut RD with d ∈ {0, 3, 6} months, bandwidth sensitivity (h ∈ {1, 2, 3, 5} years and CV-optimal), covariate-continuity (age, sex, CDR, APOE, MMSE, time-since-entry; expect no discontinuity after fixed effects), density test of session timing around the cutoff.
- Harmonization benchmark metrics: residual τ after correction (target 0), slope bias (harmonized-transitioned slopes vs. single-scanner slopes; target 0, paired by matched design), preservation of the CN age slope and of the CDR ≥ 0.5 × time interaction (relative change < 10 %), and clinical validity (AUC, C-index and calibration slope of atrophy-rate → progression, compared between transitioned and single-scanner subjects with DeLong / bootstrap tests).
- Leakage: harmonization parameters estimated in CN reference subjects or inside CV folds; outcome (CDR progression) never enters harmonization; FreeSurfer version fixed.
- Multiple comparisons: features are grouped (subcortical volumes, ventricles, lobar thickness); Holm within group; primary endpoint is hippocampal volume.
- Nulls: placebo cutoffs and simulated cohorts without scanner effects (expect τ ≈ 0 with nominal type-I error); simulated cohorts with known τ (expect coverage ≥ 93 %).

## Publishable angle

Headline: "Scanner upgrades in OASIS-3 and ADNI shift hippocampal volume by X % (≈ Y years of normal aging); only longitudinal ComBat and the RD-anchored correction leave atrophy-rate → CDR-progression prediction unbiased, while cross-sectional ComBat shrinks true atrophy in subjects with unbalanced follow-up." A secondary deliverable is a released table of estimated scanner offsets (feature × transition) and a `scanner_rd` package that any longitudinal cohort with a scanner change can run.

Target venues: NeuroImage (methods/validation), Human Brain Mapping, Imaging Neuroscience, Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring (clinical validity of atrophy rates), Medical Image Analysis (if the image-level arm is included).

Follow-ups: apply the same design to PET (PiB → AV45 tracer switch in OASIS-3 as a discontinuity), to diffusion metrics across sequence changes, and to the OASIS-3 → OASIS-4 scanner generation change; provide the counterfactual trajectories as inputs to `atrophy-subtype-trajectories`, `brain-age-transportability` and `lifespan-normative-models` in this repository.

## Risks, confounds and mitigations

- Protocol changes co-occurring with the upgrade (new sequence, resolution, FreeSurfer re-processing): donut RD, separate transitions by protocol string when available, fix FreeSurfer version.
- Selection into scanning after the upgrade (new enrolment waves, sicker participants scanned on the PET/MR): subject fixed effects, covariate-continuity tests, restrict to subjects with ≥ 2 sessions per side, CDR-stratified estimates.
- Few transitions for some scanner pairs: report per-pair n; pool pairs only under a homogeneity test.
- Non-linear aging trajectories within the bandwidth: local-linear on both sides, bandwidth sensitivity, quadratic robustness check.
- OASIS-3 provides no calendar dates: event-time design; scanner era can still be reconstructed by ordering, and the ADNI arm (with dates) checks that event-time and calendar-time designs agree.
- Traveling-subject ground truth is on Trio/mMR only: validation of RQ4 is limited to that pair; other pairs are validated by simulation and by the ADNI dual-field-strength subset.

## Milestones

- [ ] DUA/applications: OASIS-3 (NITRC), ADNI (LONI). Confirm scanner model per session in OASIS-3 MR table.
- [ ] Build session/transition tables; count transitions per scanner pair; define analysis cohorts.
- [ ] Simulation study: bias/coverage of RDiT with fixed effects vs mixed model; bandwidth/donut choices; pre-register.
- [ ] RQ1-2: discontinuity estimates for all features; placebo and covariate checks; figures of counterfactual trajectories.
- [ ] RQ4: validate Trio/mMR estimate against traveling-subject pairs.
- [ ] Harmonization benchmark (RQ3): five tabular arms; clinical validity models.
- [ ] ADNI replication (RQ6) with calendar RDiT and dual-field-strength subset.
- [ ] Optional image-level arm; release offsets table and package; write-up.

## Ethics / data-use notes

- OASIS-3/4 and ADNI are governed by their DUAs: no redistribution, no attempt at re-identification, acknowledgement text as required by each. Derived tables that could identify participants (session-level dates) are never committed; only aggregate offset tables are released.
- No data are sent to third-party APIs. Everything in `data/` and `outputs/` is git-ignored.
- ADNI requires manuscript review via the ADNI Data and Publications Committee before submission.
