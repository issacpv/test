# amyloid-from-mri — Does structural MRI beat age + APOE + MMSE as a pre-screen for amyloid-PET positivity?

An honest, externally validated audit of "amyloid-from-T1" models (ROI-based and 3D-CNN) against the cheapest possible baseline, with an age/atrophy-proxy test in amyloid-positive cognitively normal adults and a decision-curve + cost model for using MRI to reduce PET scans in trial screening.

## Status / difficulty / timeline / compute

- Status: design + starter code (table builders, ROI features, nested-CV baselines, decision-curve/cost model, optional MONAI CNN script).
- Difficulty: MSc thesis to early-PhD chapter. 6-9 months for the ROI-model paper; +3-4 months if the 3D-CNN arm is included.
- Compute: ROI models run on a laptop (minutes). ComBat/harmonization is trivial. The optional 3D-CNN arm needs one 16-24 GB GPU for ~1-2 days per training configuration (≈3,000-5,000 T1 volumes at 1.5 mm). FreeSurfer re-runs (if you choose to re-process to a single version) are ~6-10 CPU-hours per scan; OASIS-3 and ADNI ship pre-computed FreeSurfer outputs so this is optional.

## Background

Anti-amyloid trials and treatment pathways require proof of amyloid pathology (PET or CSF) before enrolment. Amyloid PET costs roughly USD 3,000-5,000 per scan and screen-failure rates in preclinical trials are high (in A4, roughly 70% of PET-scanned cognitively normal volunteers were amyloid-negative; Sperling et al., 2020, JAMA Neurology). Any cheap, already-acquired signal that enriches the population sent to PET reduces trial cost. Structural T1 MRI is acquired anyway for safety screening (ARIA baseline), so a model that reads amyloid status off the T1 would be free at the margin.

The competing pre-screen in 2024-2026 is plasma p-tau217. Multicentre analyses report that a p-tau217 pre-screen with an intermediate threshold reduces PET scans by about 64% in cognitively unimpaired and 46% in cognitively impaired participants ("Reference charts of plasma p-tau217 as a pre-screening tool in AD clinical trials", Alzheimer's & Dementia, 2025; "Use of plasma p-tau217 as a pre-screening method for detecting amyloid-PET positivity in cognitively unimpaired participants", 2024/2025), and the ALTITUDE-AD phase 2 programme reported roughly USD 10M (~40%) screening savings from a p-tau217 assay. MRI-only screening therefore has to be evaluated on the same decision-analytic footing, not on AUC alone.

## The research gap

What has been done:

- Deep learning on T1 (and T1+FLAIR) to predict PET amyloid status was trained on ADNI + OASIS-3 + A4 (4,056 examinations) with EfficientNet and externally tested on 149 Stanford ADRC examinations: external AUC 0.65 (95% CI 0.60-0.71), best subgroup MCI (AUC 0.71) ("Deep Learning-Based Prediction of PET Amyloid Status Using Multi-Contrast MRI", AJNR, 2025; arXiv 2411.12061). This is the closest prior work and it establishes that the signal is weak-to-moderate.
- Lew et al., 2023, Radiology: MRI-based CNN assessment of A/T/N biomarker status across the AD spectrum in ADNI; amyloid was the hardest of the three targets.
- A 2024 Frontiers in Neuroscience comparison of DL architectures for amyloid positivity from T1 in 1,847 participants (661 CN / 889 MCI / 297 dementia) found usable performance mainly in MCI.
- ROI-level work in ADNI/A4 (Jacobian maps, medial temporal atrophy, "MRI-based medial temporal atrophy may reflect amyloid PET positivity", 2025) reports AUCs 0.70-0.83 in mixed-diagnosis samples, i.e. samples in which amyloid status is confounded with diagnosis, age and atrophy.
- MRI-to-PET synthesis (European Radiology, 2025) reports high accuracy but in internal or semi-external settings.

What is specifically missing (verified against 2023-2026 literature; none of the above reports all of the following):

1. No head-to-head against the zero-cost covariate model age + sex + APOE ε4 + MMSE (+ education) with paired statistics (bootstrap ΔAUC, ΔBrier, decision curves). Several papers omit APOE entirely; the trial-screening question is whether MRI adds anything to a model that a coordinator can compute from the intake form.
2. No age/atrophy-proxy test. Amyloid positivity rises with age and with diagnosis; a T1 model can reach AUC ~0.7 in a mixed CN/MCI/AD sample by learning atrophy and age. The screening use-case is cognitively normal volunteers (A4-like), where the AUC has never been reported after age-matching or age-residualization of features, nor restricted to amyloid-positive CN with no measurable atrophy.
3. No decision-curve / net-benefit analysis and no cost model for MRI-based pre-screening, whereas these exist for plasma p-tau217. Without them, "AUC 0.65" cannot be translated into "PET scans avoided per 1,000 screened at 90% sensitivity".
4. Leakage and harmonization audits are rarely reported: OASIS-3 has several MR sessions per subject and multiple scanners (Siemens TIM Trio 3T, Biograph mMR); ADNI is multi-site with two FreeSurfer versions. Subject-level splits and ComBat-by-scanner are needed and their effect on the headline AUC should be shown, not assumed.
5. Calibration and prevalence transportability (OASIS-3 ~ 30-35% A+ among CN vs ADNI ~55% overall vs A4 screening ~30%) are unreported, though they determine net benefit.

The angle of this project is therefore not "a better CNN" but the audit: honest external validation, a counterfactual against the covariate baseline, an explicit proxy test in amyloid-positive CN, and a cost-effectiveness comparison with plasma p-tau217 operating points taken from the literature.

## Research questions / hypotheses

1. RQ1 (increment). In nested subject-level CV on OASIS-3, does an ROI model (FreeSurfer thickness + ICV-normalized volumes) or a 3D CNN improve AUC/Brier over age + sex + APOE ε4 count + MMSE (+ education)? H1: ΔAUC ≤ 0.05 in the CDR = 0 stratum; larger (0.05-0.10) in CDR ≥ 0.5.
2. RQ2 (proxy). Does the MRI signal survive (a) age-matching (1:1 caliper matching of A+ to A- within CN), (b) residualizing all features on age, sex and ICV using A- controls, (c) restriction to CN subjects with hippocampal volume z > -1? H2: AUC in the residualized CN stratum falls to 0.55-0.62 and is not significantly better than the covariate model.
3. RQ3 (external validity). Do OASIS-3-trained models transport to ADNI and to the A4/LEARN pre-randomization MRI subset (all CN)? H3: rank-order performance transports (AUC within 0.05) but calibration does not (slope < 0.8) without intercept re-calibration.
4. RQ4 (utility). At 90% sensitivity for A+, how many PET scans per 1,000 screened does each model avoid, and what is the cost per enrolled A+ participant, versus (a) no pre-screen, (b) covariate model, (c) published plasma p-tau217 operating points? H4: MRI-only pre-screening avoids < 30% of PET scans at 90% sensitivity in CN, versus ~60% for p-tau217; MRI + covariates is cost-saving only if the MRI is already acquired (marginal MRI cost ≈ 0).
5. RQ5 (harmonization/leakage). How much does the internal AUC change when (a) sessions of the same subject are allowed across folds, (b) ComBat-by-scanner is applied inside vs outside the CV loop? H5: session-level (leaky) CV inflates AUC by ≥ 0.03; ComBat fitted on the full data (leaky) inflates by a smaller amount.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| OASIS-3 (Knight ADRC, WashU) | T1w MRI, FreeSurfer 5.3 outputs, PiB and AV45 PET processed with PUP (Centiloid), ADRC clinical data (CDR, MMSE, APOE genotype, demographics) | ~1,300 participants, ~2,800 MR sessions, ~1,500 amyloid PET sessions (999 PiB, 492 AV45 in the 2020 release) | Free registration + Data Use Agreement on NITRC (XNAT central) | https://sites.wustl.edu/oasisbrains/ , https://www.nitrc.org/projects/oasis3/ |
| ADNI (1/GO/2/3/4) | T1w MRI, UCSF FreeSurfer tables, UC Berkeley amyloid PET Centiloid tables (florbetapir/florbetaben), ADNIMERGE (age, sex, APOE, MMSE, education, diagnosis) | ~2,000+ participants with amyloid PET | Application via LONI IDA; DUA; no redistribution | https://adni.loni.usc.edu/ , https://ida.loni.usc.edu/ |
| A4 / LEARN pre-randomization data | Screening florbetapir PET (SUVr and Centiloid), T1 MRI on screen-eligible participants (A+ in A4, A- in LEARN), APOE, PACC, demographics | ~4,400 screened with PET; ~1,300 A+ with MRI; ~500 A- (LEARN) with MRI | Application via LONI IDA (A4 study data) | https://ida.loni.usc.edu/ (project "A4") |
| OASIS-4 (optional) | Clinical memory-clinic cohort, T1 + FreeSurfer, CSF in a subset, no PET | 663 individuals, 676 MR sessions | Same NITRC DUA as OASIS-3 | https://www.nitrc.org/projects/oasis4/ |

OASIS-4 has no amyloid PET, so it cannot provide labels; it is useful only as a "clinical population" distribution-shift check of the model's predicted-probability histogram and, for the CSF subset, an Aβ42/40-based secondary label.

Amyloid labels: OASIS-3 distributes Centiloid values computed by the PET Unified Pipeline; the OASIS-3 documentation uses tracer-specific positivity thresholds of 16.4 CL for PiB (equivalent to SUVR 1.42 with RSF partial-volume correction; Su et al., 2018, NeuroImage: Clinical) and 20.6 CL for AV45. ADNI's UC Berkeley tables provide Centiloids for florbetapir/florbetaben. A4 eligibility used florbetapir SUVr ≥ 1.15 (Sperling et al., 2020); Centiloid conversions are available in the A4 release. The code takes thresholds as parameters and the analysis plan includes a sensitivity analysis excluding a 10-25 CL grey zone.

## Methods

1. Table construction (`amyloid_mri.oasis_tables`): parse OASIS IDs (`OAS30001_MR_d0129` → subject, days from entry), match each MR session to the nearest amyloid PET within ±365 days and the nearest clinical visit within ±180 days, compute age at scan (`ageAtEntry + day/365.25`), APOE ε4 allele count from genotype, cognitive status (CDR 0 vs ≥ 0.5). Primary analysis: one session per subject (earliest with PET); secondary: all sessions with subject-grouped CV.
2. ROI features (`amyloid_mri.roi_features`): parse FreeSurfer `aseg.stats` and `?h.aparc.stats`; 68 cortical thickness values, ~20 subcortical volumes normalized to eTIV, WM-hypointensity volume. Residualize on age/sex/ICV using A- controls (for the proxy test only). Harmonize by scanner/site with a parametric ComBat implementation (Johnson et al., 2007; Fortin et al., 2018) fitted inside the CV training fold.
3. Baseline and ROI models (`amyloid_mri.models`): (a) covariate logistic regression; (b) elastic-net logistic regression on ROI features; (c) HistGradientBoosting on ROI features; (d) covariates + ROI. Nested CV: outer 5-fold `StratifiedGroupKFold` repeated 5 times; inner 3-fold grouped grid search; Platt scaling fitted on inner out-of-fold logits. Out-of-fold predictions are stored per repeat.
4. Optional 3D CNN (`scripts/train_cnn.py`, MONAI DenseNet121-3D on 1.5 mm MNI-registered, skull-stripped T1; subject-level splits; mixed precision; label-permutation null). Saliency by occlusion on the held-out fold to check whether attention lands on precuneus/medial temporal (amyloid-related) or on ventricles/global atrophy (age-related).
5. External validation: freeze OASIS-3 models (including ComBat parameters, using ADNI/A4 A- controls only to estimate site offsets), predict ADNI (all diagnoses; then CN-only) and A4/LEARN (CN-only).
6. Decision curves and cost model (`amyloid_mri.decision_curve`): net benefit vs threshold (Vickers and Elkin, 2006); screening policy analysis at fixed sensitivity; PET scans avoided per 1,000; cost per enrolled A+ with parameters: PET cost, MRI marginal cost (0 if already acquired, else full), blood-test cost, and literature sensitivity/specificity for p-tau217 as a comparator arm.

Libraries: pandas, numpy, scikit-learn, statsmodels, scipy; optional MONAI + PyTorch, nibabel, ANTsPy for registration; neuroHarmonize as a cross-check of the built-in ComBat.

## Evaluation and statistics

- Discrimination: AUC with 2,000-resample subject-level cluster bootstrap CIs; paired ΔAUC between models via the same bootstrap (DeLong as a cross-check).
- Calibration: Brier score, calibration slope and intercept (logistic recalibration of y on logit p), calibration curves with loess; report separately for CN and CI strata and for each external cohort; intercept-only re-calibration for prevalence shift.
- Clinical utility: net benefit at thresholds 0.10-0.60; PET scans avoided at 80/90/95% sensitivity; cost per enrolled A+.
- Leakage prevention: all splits grouped by subject; ComBat, scalers, feature selection and Platt scaling fitted inside training folds; external cohorts never used for model selection; thresholds for the screening policy chosen on OASIS-3 OOF predictions and applied unchanged externally.
- Multiple comparisons: one pre-registered primary comparison (ROI + covariates vs covariates, CN stratum, ΔAUC); secondary strata and models corrected with Holm.
- Null models: label permutation within subject-grouped folds (1,000×) for the CNN; for the proxy test, a model trained on age + sex only serves as the "proxy ceiling".
- Reporting: TRIPOD+AI (Collins et al., 2024, BMJ).

## Publishable angle

Headline result (either direction is publishable): "In cognitively normal adults, T1-MRI models add ΔAUC = x (95% CI) over age + APOE + MMSE, avoid y PET scans per 1,000 screened at 90% sensitivity, and are (not) cost-saving relative to plasma p-tau217 pre-screening; most of the apparent MRI signal in mixed samples is an age/atrophy proxy." The paper is an audit with a decision-analytic conclusion, which is what trialists need and what the CNN papers lack.

Target venues: Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring; Radiology: Artificial Intelligence; NeuroImage: Clinical; Journal of Prevention of Alzheimer's Disease (screening/cost angle); MIDL or MICCAI (workshop) for the CNN/harmonization methodology.

Follow-ups: add FLAIR (the AJNR paper found FLAIR helps); combine MRI with plasma p-tau217 where both exist (ADNI has plasma p-tau217 on a subset) to test whether MRI adds to blood; tau-PET positivity as a second target; test a two-stage policy (blood first, MRI to resolve intermediate p-tau217 values).

## Risks, confounds and mitigations

- Label noise near the Centiloid threshold: sensitivity analysis excluding 10-25 CL; report AUC against continuous Centiloid (Spearman) as well.
- Tracer differences (PiB vs AV45) and PUP version: tracer as covariate; stratified results.
- MRI-PET time gap: primary ±365 days, sensitivity ±180 days; time gap as covariate.
- Scanner/site: ComBat inside folds; leave-one-scanner-out check in OASIS-3.
- FreeSurfer version mismatch (OASIS-3 v5.3 vs ADNI v6/7): either re-run FreeSurfer 7.x on both (compute cost) or include version as a batch in ComBat and show the results are stable.
- Selection in A4: only PET-screen-eligible participants received MRI, and MRI was acquired after PET; A- MRI comes from LEARN. Treat A4/LEARN as CN-only external validation and report the selection explicitly.
- Prevalence shift changes net benefit: report decision curves per cohort with cohort-specific prevalence and intercept-recalibrated versions.
- CNN overfitting to site: subject-level splits plus a site-prediction probe on CNN embeddings.
- APOE missingness: multiple imputation is inappropriate for a screening covariate; analyse complete cases and report the APOE-free covariate model too.

## Milestones

- [ ] DUA/applications: OASIS-3 (NITRC), ADNI (LONI), A4 (LONI).
- [ ] Download OASIS-3 tables, FreeSurfer outputs, PUP Centiloid tables; build subject table; freeze label definitions and thresholds (pre-registration document).
- [ ] ROI feature table; ComBat; covariate baseline; nested CV; RQ1 results in CN and CI strata.
- [ ] Proxy test (RQ2): age-matching, residualization, atrophy-free subgroup.
- [ ] External validation on ADNI and A4/LEARN (RQ3); calibration and recalibration.
- [ ] Decision curves and cost model with p-tau217 comparator (RQ4).
- [ ] Leakage/harmonization ablation (RQ5).
- [ ] Optional 3D CNN arm + occlusion saliency.
- [ ] Manuscript (TRIPOD+AI checklist), code release with synthetic-data tests.

## Ethics / data-use notes

- OASIS-3/OASIS-4: accept the OASIS Data Use Agreement on NITRC; acknowledge the OASIS grants and cite LaMontagne et al., 2019 (medRxiv) in publications. Data may not be redistributed.
- ADNI: data are obtained under the ADNI Data Use Agreement via LONI; manuscripts must be submitted to the ADNI Data and Publications Committee before publication and must include the ADNI acknowledgement and author list statement.
- A4: same LONI process; acknowledge the A4 Study and its funders per the data use terms.
- Never commit imaging or tabular participant data; `data/` and `outputs/` are git-ignored. Do not send participant-level data to third-party APIs (including LLM APIs); only aggregate results leave the analysis environment.
- Report results in a way that cannot be misused as a "diagnosis from MRI" claim: the target use is trial pre-screening enrichment, and calibration/net-benefit results must be shown alongside any AUC.
