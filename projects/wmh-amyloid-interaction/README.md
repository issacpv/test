# wmh-amyloid-interaction — Do white-matter hyperintensities amplify amyloid's effect on cognitive decline? A time-varying, additive-scale, segmentation-robust test in OASIS-3 FLAIR

Use the under-exploited OASIS-3 FLAIR series (with PiB/AV45 Centiloids, CDR/CDR-SB and psychometrics over up to ~15 years) to test whether white-matter hyperintensity (WMH) burden *modifies* the effect of amyloid on cognitive trajectories and CDR progression, treating WMH as a time-varying exposure (its own trajectory, not a baseline number), reporting interaction on both the multiplicative and the additive scale (RERI, attributable proportion), separating periventricular / deep / posterior WMH, and showing that the answer survives the WMH segmentation tool and the 1.5T → 3T FLAIR protocol change; replicated in ADNI with the UC Davis WMH tables.

Related project in this repository: `amyloid-from-mri` (predicting amyloid status from T1). This project is self-contained.

## Status / difficulty / timeline / compute

- Status: design + starter code (cohort/session matching with Centiloid thresholds and progression events, WMH regional features from masks with tool-agreement metrics, mixed-effects three-way interaction models, additive-interaction measures with bootstrap CIs, discrete-time hazard model on person-period data, longitudinal simulator).
- Difficulty: MSc thesis (tabular arm using FreeSurfer WM-hypointensity volumes and one FLAIR segmenter; 6-9 months) to PhD chapter (segmentation multiverse + joint models + ADNI replication; 12-18 months).
- Compute: WMH segmentation for ~2,000 FLAIR sessions with 3-4 tools (LST-LPA, BIANCA, SAMSEG-lesion, a deep-learning segmenter) is ~5-20 min per session per tool on CPU/GPU, i.e. a few hundred CPU-hours or ~2 GPU-days; statistics run on a laptop.

## Background

Cerebral small-vessel disease, visible as WMH on FLAIR (STRIVE criteria: Wardlaw et al., 2013, Lancet Neurology; STRIVE-2: Duering et al., 2023, Lancet Neurology), co-occurs with amyloid pathology in most older adults. Whether the two act additively or synergistically on cognition determines whether vascular prevention changes the course of amyloid-positive individuals and how anti-amyloid trials should stratify. ADNI-based work suggested that WMH and amyloid are jointly "necessary and sufficient" for clinical AD (Provenzano et al., 2013, JAMA Neurology); Mayo Clinic data found vascular and amyloid pathologies to be *independent* predictors of decline in normal elderly (Vemuri et al., 2015, Brain); the Harvard Aging Brain Study found a vascular-risk × amyloid interaction on decline (Rabin et al., 2018, JAMA Neurology); parietal WMH progression predicted incident AD in WHICAP (Brickman et al., 2015, Neurobiology of Aging); periventricular WMH have been associated with amyloid burden itself (Marnane et al., 2016, Neurology). In 2024-2025 a cross-sectional analysis of WMH spatial patterns reported that periventricular and basal-ganglia WMH components predicted poorer episodic memory only in the presence of elevated amyloid, and the A4 pre-randomisation cohort showed that WMH modulate the amyloid → p-tau217 → cognition chain and contribute to decline independently ("White matter hyperintensity modulates the amyloid-tau-cognition association and anti-amyloid treatment efficacy in asymptomatic older adults", 2025).

OASIS-3 (LaMontagne et al., 2019, medRxiv) contains FLAIR in a large fraction of its ~2,842 MR sessions, PiB (~1,000) and AV45 (~500) PET with Centiloids from the PET Unified Pipeline (Su et al., 2018, NeuroImage: Clinical), annual CDR and a psychometric battery, on ~1,378 participants followed for up to ~15 years. It has been used far less than ADNI for WMH questions because no WMH table is distributed: the volumes must be computed.

## The research gap

What has been done:

- Baseline-WMH × amyloid interactions on cross-sectional cognition or on slopes, mostly in ADNI, HABS and Mayo (above); spatial-pattern × amyloid interaction cross-sectionally (2024).
- Longitudinal WMH trajectories pooled across ADNI, AIBL and OASIS-3 by the ADOPIC consortium ("White matter hyperintensity onset, trajectories, and associations...", 2024/2025; "Regional growth rates of white matter hyperintensities...", 2025/2026), reporting that WMH accumulation accelerates around age 60 and exacerbates decline in amyloid-positive individuals. These used a single segmentation pipeline and modelled WMH as the exposure of interest rather than as a time-varying modifier of the amyloid effect on clinical progression.
- Segmentation benchmarking (Kuijf et al., 2019, IEEE TMI, WMH challenge; BIANCA, Griffanti et al., 2016, NeuroImage; LST, Schmidt et al., 2012, NeuroImage; SAMSEG lesion, Cerri et al., 2021, NeuroImage) — accuracy against manual masks, not the sensitivity of a downstream interaction estimate to the tool.
- Epidemiological guidance that interaction should be reported on the additive scale (RERI, attributable proportion, synergy index) as well as the multiplicative scale (Knol and VanderWeele, 2012, International Journal of Epidemiology; VanderWeele and Knol, 2014, Epidemiologic Methods) — never followed in the WMH-amyloid literature, where only product terms in linear/logistic models are reported.

What is specifically missing:

1. Time-varying WMH: no study models the *concurrent* WMH trajectory (subject-specific level and growth) as a modifier of the amyloid effect on cognitive slopes and on CDR progression, with joint longitudinal-survival or person-period hazard models that handle the fact that WMH is measured with error and only at scan visits.
2. Additive-scale interaction with CIs, which is the public-health-relevant scale (how many progressions are attributable to the co-occurrence), and the distinction moderator vs mediator (does WMH lie on the amyloid → cognition path, or amplify it?) with temporal ordering.
3. Segmentation and field-strength robustness: OASIS-3 spans 1.5T Vision/Sonata and 3T Trio/mMR FLAIR; the interaction estimate must be shown to be stable across tools and scanner generations, or the field has been reporting a pipeline artefact.
4. Regional specificity with a longitudinal design: periventricular vs deep vs posterior (parietal/occipital) WMH growth × amyloid on decline, testing the "posterior WMH = AD-related" hypothesis prospectively.
5. External replication in ADNI (UC Davis WMH volumes + Centiloids + CDR-SB) with the identical model, and a harmonised effect size.

## Research questions / hypotheses

1. RQ1 (multiplicative interaction on slopes). In mixed models of cognition (global composite, episodic memory, CDR-SB) on time × amyloid × WMH(t), is the three-way term significant? H1: amyloid-positive participants with high or growing WMH decline faster than predicted by the two main effects (three-way coefficient < 0 for cognition, > 0 for CDR-SB); effect concentrated in episodic memory.
2. RQ2 (additive interaction on progression). For CDR 0 → ≥ 0.5 progression, is RERI > 0 with the lower CI bound above 0? H2: RERI ≈ 0.5-1.5 with AP 20-40 %; synergy index > 1.
3. RQ3 (time-varying vs baseline). Does a time-varying WMH exposure (current level or growth rate) explain more of the amyloid-related decline than baseline WMH? H3: WMH growth rate is the stronger modifier; baseline WMH alone underestimates the interaction by ≥ 30 %.
4. RQ4 (regional). Which WMH compartment carries the interaction? H4: periventricular and posterior WMH; deep frontal WMH interact weakly or not at all.
5. RQ5 (moderation vs mediation). Does WMH mediate part of the amyloid → cognition effect (amyloid → later WMH growth → decline)? H5: mediation is small (< 10 % of the total effect); the dominant structure is moderation.
6. RQ6 (robustness). Do the interaction estimates agree across four segmentation tools, across 1.5T and 3T sessions, and between OASIS-3 and ADNI? H6: rank-order agreement across tools (ICC of subject WMH volumes > 0.85); interaction coefficients within each other's CIs; the 1.5T era shows the same sign with wider CIs.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| OASIS-3 (Knight ADRC) | FLAIR + T1 per MR session (scanner model in session table), FreeSurfer 5.3 (eTIV, WM-hypointensities, ventricles), PUP PET (PiB/AV45 Centiloid), ADRC clinical (CDR, CDR-SB, MMSE), psychometrics (logical memory, digit symbol, etc.), APOE, demographics | ~1,378 participants; ~2,842 MR sessions; ~1,500 amyloid PET; FLAIR in a large fraction of sessions (count from the session scan list) | Free registration + DUA (NITRC / XNAT) | https://sites.wustl.edu/oasisbrains/ , https://www.nitrc.org/projects/oasis3/ |
| ADNI (2/3/4) | FLAIR-based WMH volumes from the UC Davis tables, UC Berkeley amyloid Centiloids, ADNIMERGE (CDR-SB, diagnosis, cognition composites), FreeSurfer ICV | ~1,500 with WMH + amyloid | Application via LONI IDA (DUA) | https://adni.loni.usc.edu/ |
| OASIS-4 (optional) | Clinical cohort FLAIR for segmentation-tool robustness only (no PET) | 663 | NITRC DUA | https://www.nitrc.org/projects/oasis4/ |
| WMH Segmentation Challenge (MICCAI 2017) | Manual WMH masks (60 training cases, 3 scanners) for tool calibration | small | Open (registration) | https://wmh.isi.uu.nl/ |

## Methods

1. Cohort and matching (`wmh_amyloid.cohort`): parse OASIS-3 IDs (`OAS30001_MR_d0129`), keep sessions with FLAIR, attach the nearest amyloid PET within ±365 days (or carry the baseline PET forward with an amyloid-time covariate), the nearest clinical visit within ±180 days, age at session, APOE ε4 count, years since first FLAIR; amyloid positivity from Centiloids with tracer-specific thresholds (PiB 16.4 CL, AV45 20.6 CL per OASIS-3 documentation) and a continuous Centiloid arm; progression event definitions (CDR 0 → ≥ 0.5; CDR-SB increase ≥ 1).
2. WMH features (`wmh_amyloid.wmh_features`): total WMH volume, periventricular/deep split by distance to the ventricular mask (distance-transform rule, 10 mm and 3 mm variants), lobar volumes and posterior fraction from a lobar label map, ICV normalisation and log transform; tool agreement (Dice, volume ICC, Bland-Altman) between LST-LPA, BIANCA, SAMSEG-lesion and a deep-learning segmenter, all run on the same FLAIR+T1; FreeSurfer WM-hypointensities (T1-only) as the zero-cost comparator.
3. Models (`wmh_amyloid.models`): (a) linear mixed models with random intercept and slope: `cognition ~ time * amyloid * WMH(t) + age0 + sex + education + APOE4 + scanner_generation`; (b) person-period discrete-time hazard (logistic) for progression with time-varying WMH and amyloid; (c) additive interaction measures RERI, AP and synergy index from the hazard model with subject-bootstrap CIs (Hosmer and Lemeshow, 1992, Epidemiology; Knol and VanderWeele, 2012); (d) joint longitudinal-survival model (R `JMbayes2`) linking the subject-specific WMH trajectory to the hazard (production arm); (e) mediation with temporal ordering (amyloid at t0 → WMH growth t0-t1 → decline t1-t2) via the counterfactual mediation framework (VanderWeele, 2015).
4. Multiverse: 4 segmentation tools × 2 PV/deep rules × binary vs continuous amyloid × 3 outcomes, summarised by specification curve; primary specification pre-registered (LST-LPA, 10 mm, binary amyloid, global composite).
5. Simulation (`wmh_amyloid.simulate`): longitudinal cohort with known slopes, WMH growth, interaction and progression hazard, to check bias and power of (a)-(c) at OASIS-3-like visit spacing and dropout.

Libraries: numpy, pandas, scipy, statsmodels; nibabel and scipy.ndimage for masks; FSL (BIANCA), LST (SPM), FreeSurfer 7.3 SAMSEG, a deep-learning WMH tool; R `JMbayes2` and `lme4` as cross-checks.

## Evaluation and statistics

- Primary estimand: the three-way `time × amyloid × WMH` coefficient (per SD of log-WMH) in the mixed model for the global cognitive composite; secondary: RERI for progression.
- CIs: model-based (Satterthwaite/cluster-robust) and subject-level bootstrap (2,000) for all interaction quantities; RERI CI by bootstrap (delta-method as sensitivity).
- Baseline balance and confounding: age, sex, education, APOE, scanner generation, hypertension/diabetes if available in the ADRC data; E-values for the interaction.
- Missing data and dropout: informative dropout addressed by the joint model; comparison with mixed-model estimates.
- Measurement error in WMH: tool-agreement ICC used in a regression-calibration sensitivity analysis.
- Multiple comparisons: three outcomes and four regions pre-specified; Holm within family; specification-curve inference for the multiverse (median effect and the fraction of specifications with p < 0.05).
- Leakage: no PET-derived variable enters WMH segmentation; amyloid thresholds fixed a priori; ADNI used only for replication after the OASIS-3 analysis is frozen.
- Nulls: permutation of amyloid status within age-sex strata (interaction should vanish); simulated cohorts with zero interaction (type-I error), and with known interaction (coverage).

## Publishable angle

Headline: "In OASIS-3, growing periventricular/posterior WMH multiply the effect of amyloid on cognitive decline and CDR progression (RERI ≈ X, attributable proportion ≈ Y %), the effect is stable across four segmentation tools and two scanner generations and replicates in ADNI; baseline WMH understates it." Deliverables: an OASIS-3 WMH table (session-level volumes from four tools, shared under the OASIS DUA terms via NITRC if permitted), the additive-interaction report, and a pre-registered analysis code base.

Target venues: Neurology, Alzheimer's & Dementia, Brain, JAMA Neurology (if the ADNI replication is strong), NeuroImage: Clinical (segmentation-robustness component).

Follow-ups: add tau PET (AV1451 subset in OASIS-3) for a three-pathology model; use the WMH table for the `amyloid-from-mri` project's FLAIR arm; test blood-pressure trajectories from the ADRC data as the upstream cause of WMH growth.

## Risks, confounds and mitigations

- FLAIR availability and protocol heterogeneity across OASIS-3 eras: scanner-generation covariate, era-stratified analyses, tool multiverse.
- Amyloid measured at a different time than FLAIR: ±365-day matching with an "amyloid-time" covariate; continuous Centiloid interpolation as a sensitivity analysis; restrict to sessions with PET within 1 year in the primary.
- Survivor bias and dropout (sicker participants stop being scanned): joint models; inverse-probability-of-censoring weights as an alternative.
- Collinearity of WMH with age: age-centred models and within-subject WMH change; report the interaction at fixed ages.
- Small numbers of progressors in amyloid-negative/high-WMH cells: report cell counts; use CDR-SB as a continuous secondary outcome; pool with ADNI for the additive-scale estimate.
- Segmentation failures on 1.5T FLAIR: QC by visual review of a random 10 % plus outlier detection; exclusion sensitivity analysis.

## Milestones

- [ ] DUA (OASIS-3), application (ADNI); download FLAIR + T1 + FreeSurfer + PUP + clinical tables; tabulate FLAIR sessions per scanner.
- [ ] Run four WMH segmenters on all FLAIR sessions; QC; tool-agreement report; regional features.
- [ ] Simulation study and pre-registration of the primary specification.
- [ ] RQ1-RQ2 primary models; RQ3 time-varying vs baseline; RQ4 regional.
- [ ] RQ5 mediation; joint models; RQ6 tool/era robustness and specification curve.
- [ ] ADNI replication with UC Davis WMH tables; pooled additive-interaction estimate.
- [ ] Share the WMH table (per OASIS terms), release code; write-up.

## Ethics / data-use notes

- OASIS-3/4 and ADNI DUAs: no redistribution of raw data, no re-identification; the derived WMH table can only be shared through channels the OASIS team approves; ADNI manuscripts pass the ADNI publications committee.
- Clinical data (CDR, psychometrics) are sensitive; only aggregate results leave the analysis environment; no third-party APIs; `data/` and `outputs/` are git-ignored.
