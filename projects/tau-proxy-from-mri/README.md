# tau-proxy-from-mri — Can T1 + FLAIR triage patients for tau PET without any PET or plasma input?

An externally validated audit of "PET-free" tau proxies: predict flortaucipir (AV1451) positivity and 2024-criteria tau stage (T2 medial-temporal vs neocortical) from structural MRI (FreeSurfer T1 features + FLAIR white-matter hyperintensities) and intake covariates, with an explicit "is this just neurodegeneration?" test on tau/atrophy-discordant participants, OASIS-3 ↔ ADNI cross-cohort transport, and a decision-curve model for tau-PET triage.

## Status / difficulty / timeline / compute

- Status: design + starter code (OASIS-3 table builders, tau labelling from PUP ROI tables, FreeSurfer feature extraction with tau-signature composites, nested subject-grouped CV with a covariate baseline, discordance test, decision curves).
- Difficulty: MSc thesis to early-PhD chapter. 6-9 months for the ROI-model paper; +3 months if a 3D-CNN arm is added.
- Compute: ROI models run on a laptop in minutes. FreeSurfer re-processing (only if you want a single FreeSurfer version across cohorts) is 6-10 CPU-hours per scan; OASIS-3 and ADNI ship pre-computed FreeSurfer outputs so this is optional. WMH segmentation (e.g., SynthSeg-WMH, LST-AI, or BIANCA) is ~2-5 minutes per FLAIR on CPU. The optional CNN arm needs one 16-24 GB GPU for a few days.
- Related project in this repository: `amyloid-from-mri` (same cohorts, same audit philosophy, amyloid target). This folder is self-contained; nothing is imported from it.

## Background

Tau PET is the biomarker most tightly coupled to cognitive decline and progression in Alzheimer's disease, and under the 2024 Alzheimer's Association revised criteria (Jack et al., 2024, Alzheimer's & Dementia) tau PET defines the "T2" biological stage: medial-temporal (stage b) versus moderate/high neocortical (stages c/d) tau. Tau PET is more expensive and less available than amyloid PET, has tracer-specific off-target binding, and is not reimbursed for routine care in most systems. Anti-amyloid treatment decisions increasingly reference tau load (high-tau participants benefited less in phase 3 lecanemab/donanemab analyses), so a cheap way of deciding *who needs a tau PET* has direct clinical and trial-screening value.

Structural MRI is a plausible proxy for tau for a mechanistic reason that does not hold for amyloid: neurodegeneration follows tau topographically. Baseline tau PET predicts the spatial pattern and rate of subsequent atrophy (La Joie et al., 2020, Science Translational Medicine), tau PET patterns mirror clinical and neuroanatomical variants of AD (Ossenkoppele et al., 2016, Brain), and data-driven tau subtypes have distinct atrophy signatures (Vogel et al., 2021, Nature Medicine). The question is therefore not whether MRI carries tau information (it does) but whether that information is separable from plain neurodegeneration, whether it transports across cohorts, and whether it is decision-relevant compared with a coordinator-computable covariate model.

## The research gap

What has been done (2023-2026):

- Lew et al., 2023, Radiology: 3D CNNs on T1 MRI to classify A, T and N status in ADNI. Tau status was predictable at moderate accuracy but the model was internal to ADNI and not compared with a covariate-only baseline or with a neurodegeneration-only explanation.
- Karlsson et al. (medRxiv 2024; published 2025): machine-learning prediction of tau-PET load and distribution from plasma, MRI and clinical variables. Plasma p-tau217 dominated prediction of total tau load (R² ~0.66-0.69); MRI variables were the best predictors of hemispheric asymmetry of tau. MRI-only, PET-free performance was not the headline and no triage/decision analysis was reported.
- "Classification of tau status with machine learning models" (2025/2026; ADNI n = 410 with external validation): logistic regression on structural MRI, amyloid PET Centiloid and demographics reached AUC 0.92 for Braak III/IV positivity. The model *requires amyloid PET*, so it does not answer the PET-free question; its performance without the Centiloid input is not the primary result.
- "Prediction of amyloid and tau status in nondemented older adults using tree-based ensemble models" (2024/2025) and "A machine learning approach to predict tau positivity using clinical features in amyloid-positive individuals" (2024/2025): clinical/demographic feature sets, restricted to amyloid-positive or non-demented participants, without FLAIR and without a discordance (T vs N) analysis.
- Tau-PET harmonization has matured: the CenTauR scale and masks (Villemagne et al., 2023, Alzheimer's & Dementia: DADM) and the joint propagation model (Leuzy et al., 2024, Alzheimer's & Dementia) allow tracer-independent thresholds; almost all MRI-proxy work still uses cohort-specific SUVR cut-offs.

What is specifically missing:

1. A PET-free, plasma-free model (T1 + FLAIR + age, sex, APOE ε4, education, MMSE/CDR-SB) evaluated against the covariate-only baseline with paired statistics. Existing MRI tau models either include amyloid PET as an input or do not report the increment over intake variables.
2. A neurodegeneration-discordance test. Tau and atrophy are correlated by construction; an MRI model can reach AUC 0.8 by learning "N". The decisive test is performance in T+N− (tau without measurable atrophy) and T−N+ (atrophy without tau: SNAP/LATE/vascular) participants, and whether adding FLAIR WMH volume helps separate non-AD neurodegeneration from tau.
3. Ordinal staging rather than binary positivity: predicting the 2024-criteria T2 stage (none / medial temporal / neocortical) is the clinically actionable target and has not been modelled from MRI alone.
4. OASIS-3's AV1451 subset (451 tau PET sessions; NITRC sub-project OASIS-3_AV1451, PIs Benzinger and Morris) is dominated by cognitively normal older adults with low tau prevalence. It has not been used as an external validation set for ADNI-trained tau proxies (or vice versa), and it is the realistic low-prevalence screening scenario.
5. No decision-curve / net-benefit or "tau PET scans avoided per 1,000" analysis exists for an MRI-based tau triage, whereas such analyses exist for plasma p-tau217. Without them AUCs cannot be translated into a screening policy.
6. Threshold and harmonization sensitivity: positivity depends on meta-temporal SUVR cut-off (1.20-1.30 depending on reference region and partial-volume correction) or CenTauR thresholds; label sensitivity analyses are rarely reported.

The angle is therefore an audit with a mechanistic twist: MRI should proxy tau better than it proxies amyloid because atrophy is downstream of tau, and the paper quantifies how much of that signal is "tau-specific" versus generic neurodegeneration, whether it transports, and what it is worth as a triage tool.

## Research questions / hypotheses

1. RQ1 (increment). In nested subject-grouped CV on OASIS-3 AV1451 participants, does an ROI model (68 cortical thicknesses, eTIV-normalized subcortical volumes, hippocampal and entorhinal measures, WMH volume) improve AUC and Brier score over age + sex + APOE ε4 + education + MMSE + CDR-SB? H1: ΔAUC 0.05-0.12 in the full sample; ≥ 0.05 in the CDR = 0 stratum (larger than the corresponding amyloid increment).
2. RQ2 (tau-specificity). Restricting to participants without measurable neurodegeneration (N−: hippocampal volume and AD-signature thickness within 1 SD of A− CN norms), does the MRI model still discriminate T+ from T−? H2: AUC drops to 0.60-0.68 but remains above 0.5 with 95% CI excluding 0.5; in T−N+ participants the model's false-positive rate is ≥ 2× that in T−N−, and WMH volume reduces it.
3. RQ3 (staging). Can an ordinal model (none / MTL / neocortical) from MRI recover T2 stage with macro-averaged AUC ≥ 0.75 and adjacent-category accuracy ≥ 80%? H3: MTL-only stage is the hardest class (it is the class where atrophy is subtle).
4. RQ4 (transport). Do OASIS-3-trained models transport to ADNI (and ADNI-trained to OASIS-3) after ComBat by scanner fitted on the training cohort? H4: discrimination transports (AUC within 0.05) but calibration intercept shifts with prevalence (OASIS-3 CN T+ prevalence ~15-25% vs ADNI mixed ~35-45%); intercept-only recalibration restores calibration-in-the-large.
5. RQ5 (utility). At 90% sensitivity for T+, how many tau PET scans per 1,000 does the MRI + covariate model avoid compared with the covariate model, and how does this compare with published plasma p-tau217 operating points? H5: MRI + covariates avoids 25-40% of scans at 90% sensitivity in mixed samples and < 25% in CN, i.e. it is a useful triage only where MRI is already acquired.
6. RQ6 (label sensitivity). Do conclusions hold across meta-temporal SUVR thresholds (1.20, 1.25, 1.30), CenTauR-based thresholds, and exclusion of a grey zone? H6: rank-order results hold; ΔAUC estimates shift by < 0.03.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| OASIS-3 (Knight ADRC, WashU) | T1w, FLAIR, FreeSurfer outputs, ADRC clinical data (CDR, MMSE, APOE, education, demographics), PiB/AV45 amyloid PET (PUP, Centiloid) for the N− definition and A-stratified analyses | ~1,300 participants, ~2,800 MR sessions | Free registration + Data Use Agreement on NITRC / XNAT Central | https://sites.wustl.edu/oasisbrains/ , https://www.nitrc.org/projects/oasis3/ |
| OASIS-3_AV1451 (sub-project) | Flortaucipir PET sessions with PUP ROI SUVR tables (FreeSurfer-based ROIs, with and without partial-volume correction) | 451 tau PET sessions | Same NITRC DUA; separate project on NITRC | https://www.nitrc.org/projects/oasis3_av1451/ |
| OASIS3_AV1451_Longitudinal (sub-project) | Longitudinal flortaucipir sessions for conversion analyses | subset of the above | Same NITRC DUA | https://www.nitrc.org/projects/oasis3_av1451l/ |
| ADNI (2/3/4) | T1w, FLAIR, UCSF FreeSurfer tables, UC Berkeley flortaucipir tables (meta-temporal and Braak-stage ROI SUVRs, with reference region options), ADNIMERGE covariates, amyloid Centiloids | > 1,500 participants with tau PET (2024 release) | Application via LONI IDA; DUA; no redistribution | https://adni.loni.usc.edu/ , https://ida.loni.usc.edu/ |
| A4 / LEARN (optional) | Screening-phase flortaucipir tau PET in a subset of cognitively normal participants, T1 MRI, APOE, PACC | several hundred with tau PET | Application via LONI IDA (project "A4") | https://ida.loni.usc.edu/ |

Labels. OASIS-3 AV1451 PUP tables give ROI SUVRs (cerebellar-cortex reference) in FreeSurfer regions. The code builds a meta-temporal composite (entorhinal, amygdala, parahippocampal, fusiform, inferior temporal, middle temporal; Jack et al., 2017, Alzheimer's & Dementia) and a neocortical composite, and applies parameterized thresholds; the default binary cut-off is 1.25 SUVR on the non-PVC composite with a sensitivity range 1.20-1.30, and an alternative using Braak-region masks for the stage variable. ADNI's Berkeley tables provide the same composites directly. Where CenTauR conversion parameters are available for the tracer/pipeline, CenTauR units are used for a harmonized secondary label; otherwise the code falls back to cohort-specific SUVR thresholds and reports both.

## Methods

1. Table construction (`tau_proxy.tables`): parse OASIS IDs (`OAS30001_AV1451_d1234`, `OAS30001_MR_d1200`), match each tau PET to the nearest MR session within ±365 days and the nearest clinical visit within ±180 days; compute age at scan; APOE ε4 allele count; CDR-based cognitive status; one PET-MR pair per participant for the primary analysis (earliest), all pairs with subject-grouped CV for the secondary analysis.
2. Labels (`tau_proxy.labels`): meta-temporal and neocortical composites from ROI SUVR tables; binary positivity; ordinal T2 stage (0 none / 1 MTL-only / 2 neocortical) following the 2024 criteria's logic (MTL positivity without neocortical positivity = stage b); grey-zone exclusion; N status from hippocampal volume and AD-signature thickness z-scores relative to A− CN.
3. Features (`tau_proxy.features`): FreeSurfer `aseg.stats` and `?h.aparc.stats` parsing → 68 thicknesses, subcortical volumes normalized to eTIV, AD-signature meta-ROI thickness, hippocampal occupancy score, WMH volume (FreeSurfer WM-hypointensities as a fallback; FLAIR-based WMH volume from LST-AI/BIANCA/SynthSeg-WMH when available), asymmetry indices (left-right thickness differences, motivated by Karlsson et al.). ComBat by scanner is fitted inside the training fold.
4. Models (`tau_proxy.models`): (a) covariate logistic regression; (b) elastic-net logistic regression on ROI features; (c) HistGradientBoosting; (d) covariates + ROI; (e) ordinal logistic (proportional odds via statsmodels or a cumulative-link implementation) for stage. Nested CV: outer 5-fold `StratifiedGroupKFold` × 5 repeats, inner 3-fold grouped grid search, Platt scaling on inner OOF logits.
5. Discordance test (`tau_proxy.discordance`): define T/N quadrants; report AUC within N− and false-positive rate within T−N+ vs T−N−; test whether WMH volume as an added feature lowers T−N+ false positives (paired bootstrap).
6. External validation: freeze OASIS-3 models (including ComBat parameters, estimating ADNI scanner offsets from ADNI T− CN only) and predict ADNI; repeat in the other direction; report discrimination, calibration slope/intercept, and intercept-only recalibration.
7. Decision analysis (`tau_proxy.decision`): net benefit vs threshold (Vickers and Elkin, 2006, Medical Decision Making), scans avoided per 1,000 at fixed sensitivity, cost per identified T+, comparator arms from literature p-tau217 operating points entered as parameters.
8. Optional CNN arm (`scripts/train_cnn.py`, to be added): MONAI 3D DenseNet on 1.5 mm MNI T1 + FLAIR channels; occlusion saliency to check whether attention lands on medial/inferior temporal cortex.

Libraries: pandas, numpy, scipy, scikit-learn, statsmodels; optional nibabel, MONAI/PyTorch, neuroHarmonize (cross-check of built-in ComBat), LST-AI or FSL BIANCA for WMH.

## Evaluation and statistics

- Discrimination: AUC with 2,000-resample subject-level bootstrap CIs; paired ΔAUC via the same bootstrap; DeLong as a cross-check. Ordinal: macro one-vs-rest AUC, adjacent-category accuracy, Somers' D.
- Calibration: Brier, calibration slope and intercept, loess calibration curves per cohort and stratum.
- Clinical utility: net benefit at thresholds 0.10-0.60; scans avoided at 80/90/95% sensitivity; sensitivity analysis over cost parameters.
- Leakage prevention: all splits grouped by participant; ComBat, scalers, feature selection and Platt scaling fitted inside training folds; the external cohort is never used for model selection; thresholds fixed on OASIS-3 OOF predictions.
- Multiple comparisons: one primary comparison (ROI + covariates vs covariates, full sample, ΔAUC); strata and alternative labels are secondary and Holm-corrected.
- Nulls: label permutation within grouped folds (1,000×); age-sex-only model as the proxy ceiling; a "N-only" model (hippocampal volume + AD-signature thickness) as the neurodegeneration ceiling — the primary claim requires the full model to beat the N-only model in the N− stratum.
- Reporting: TRIPOD+AI (Collins et al., 2024, BMJ).

## Publishable angle

Headline result: "T1 + FLAIR MRI predicts tau-PET positivity with AUC ≈ 0.8 in mixed samples, but roughly half of that signal is generic neurodegeneration; the tau-specific residual is real (AUC ≈ 0.65 in N− participants), transports between OASIS-3 and ADNI after recalibration, and avoids ~30% of tau PET scans at 90% sensitivity — comparable to nothing in the plasma era unless the MRI is already acquired." Either direction of the discordance result is publishable: a tau-specific signal would motivate MRI-based triage where plasma assays are unavailable; its absence would be a useful negative result against "MRI-derived tau" claims.

Target venues: Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring; NeuroImage: Clinical; Radiology: Artificial Intelligence; Journal of Nuclear Medicine (if CenTauR harmonization is central); MICCAI/MIDL workshop for the CNN arm.

Follow-ups: longitudinal conversion T− → T+ from baseline MRI (OASIS3_AV1451_Longitudinal); adding plasma p-tau217 where available (ADNI) to quantify the marginal value of MRI on top of plasma; regional tau load regression (not just positivity) with asymmetry targets.

## Risks, confounds and mitigations

- Tau-neurodegeneration collinearity: addressed head-on by the discordance design and the N-only ceiling model; the paper's claim is conditional on that result.
- Low prevalence in OASIS-3 CN: wide CIs; use repeated CV and report precision-recall curves; pool ADNI + OASIS-3 for a secondary mixed-cohort CV with cohort as a group factor.
- Off-target flortaucipir binding (choroid plexus, meninges, basal ganglia) contaminates MTL composites: use the PVC and non-PVC tables, exclude hippocampus from the composite (standard), and run the sensitivity analysis over thresholds.
- FreeSurfer version differences (OASIS-3 FS 5.3 vs ADNI FS 6/7): ComBat by cohort × scanner; sensitivity analysis re-running FS 7 on a subset.
- FLAIR availability: not all OASIS-3 sessions have FLAIR; the WMH feature uses FreeSurfer WM-hypointensities as fallback and reports the FLAIR-only subset separately.
- Amyloid status as a hidden confound: report A-stratified results (tau positivity in A− participants is rare and often non-AD; main analysis in A+ and all).
- Multiple sessions per participant: subject-grouped CV throughout.

## Milestones

- [ ] NITRC DUA approved for OASIS-3 and OASIS-3_AV1451; ADNI application approved.
- [ ] `tau_proxy.tables` produces one PET-MR-clinical row per participant with QC flags (n, prevalence table by CDR).
- [ ] Labels: binary and ordinal T2 stage; threshold sensitivity table.
- [ ] Features: FreeSurfer parsing validated against a hand-checked subject; WMH volumes for the FLAIR subset.
- [ ] Nested CV results for covariate, ROI, N-only, and combined models; primary ΔAUC with CI.
- [ ] Discordance analysis (T/N quadrants) and WMH effect.
- [ ] External validation OASIS-3 ↔ ADNI with recalibration.
- [ ] Decision curves and scans-avoided table; comparator arms.
- [ ] Label-sensitivity and permutation nulls.
- [ ] Manuscript (TRIPOD+AI checklist) and code release with synthetic example data.

## Ethics / data-use notes

- OASIS-3 and ADNI data are under Data Use Agreements: no redistribution, no attempts at re-identification, acknowledgement text required in publications (OASIS: P30 AG066444 and related grants; ADNI: standard acknowledgement and ADNI author list).
- Never commit data, derived participant-level tables, or credentials. `data/` is git-ignored; credentials are read from environment variables.
- Do not send participant-level data to third-party LLM or cloud APIs. Aggregate results only in manuscripts.
- Report subgroup performance (sex, education, race where available) since triage models can encode disparities in who receives confirmatory testing.
