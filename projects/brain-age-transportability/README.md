# Brain-age transportability audit

**One sentence:** A systematic audit of how brain-age delta, its age-bias corrections and scanner harmonisation transport across HCP Young Adult, HCP-Aging/Development and OASIS-3, using OASIS-3's longitudinal MRI + CDR/MMSE trajectories as the ground truth for within-person change.

| | |
|---|---|
| Status | proposal + starter code (loaders, ComBat, baselines, corrections, mixed models) |
| Difficulty / timeline | MSc thesis or first PhD paper; 6-9 months with data access in hand (add 1-3 months for NDA/DUA approvals) |
| Compute | CPU only for the FreeSurfer-feature pipeline (minutes per model). Optional: re-running FreeSurfer 7 on ~4,000 sessions (~1-2 h/session CPU; a 64-core node for ~1 week) or a GPU for a CNN comparator |
| Package | `src/brainage_transport` |

## Background

"Brain age" is a chronological-age prediction from structural MRI; the residual (brain-age delta, or gap) is used as a biomarker of accelerated ageing in dementia, psychiatry and population studies (Franke & Gaser, 2019, *Front Neurol*; Cole et al., 2018, *Mol Psychiatry*). Two technical facts undermine its use as a *transportable* biomarker:

1. **Age bias / regression to the mean.** Any regressor with R² < 1 predicts young people as older and old people as younger, so raw delta correlates negatively with age (Le et al., 2018, *Front Aging Neurosci*; Liang et al., 2019, *Hum Brain Mapp*; Smith et al., 2019, *NeuroImage*). Corrections are applied post hoc: regressing delta on age (Beheshti et al., 2019, *NeuroImage: Clin*), rescaling predictions by the reference slope (Cole et al., 2018), or the de Lange & Cole (2020, *NeuroImage: Clin*) variant. Butler et al. (2021, *Hum Brain Mapp*) showed that where and how the correction is calibrated changes downstream associations; de Lange et al. (2022, *Hum Brain Mapp*) showed that performance metrics themselves are affected.
2. **Scanner/protocol shift.** Delta absorbs scanner effects. Jirsaraie et al. (2023, *Hum Brain Mapp*) benchmarked brain-age models across scanners and found that scanner variance and prediction bias interact; the usual remedy is ComBat-style harmonisation (Fortin et al., 2018, *NeuroImage*; Pomponio et al., 2020, *NeuroImage*), which is almost always fitted on the pooled sample, including the test cohort.

Both problems are usually handled inside one cohort and evaluated cross-sectionally. What a clinician needs is different: a delta computed on a *new* scanner and age range, with corrections calibrated elsewhere, that still tracks *within-person* decline.

## The research gap

**What exists.**
- More et al. (2023, *NeuroImage*) compared 128 ML workflows for brain age and, importantly, evaluated test-retest reliability (CoRR) and longitudinal consistency on OASIS-3 (N = 127 pairs, 3-4 years apart). They did not vary bias correction or harmonisation, and did not relate delta change to CDR/MMSE change.
- Zhang et al. (2023, *NeuroImage: Clin*) formalised sample-level vs age-level bias correction and showed Beheshti and de Lange corrections are algebraically equivalent; cross-sectional only.
- Vidal-Piñeiro et al. (2021, *eLife*) showed cross-sectional delta relates to early-life factors more than to longitudinal brain change, i.e. delta is largely a trait, which makes *change in delta* the quantity that needs auditing.
- Cumplido-Mayoral et al. (2023, *eLife*) validated a FreeSurfer-feature brain-age model against AD biomarkers across several cohorts, with a single correction method.
- Wegmann et al. (2025, *medRxiv*) and related work fit longitudinal CNN architectures directly to OASIS-3 change, showing cross-sectional models poorly capture individual trajectories; harmonisation and correction choice were fixed.
- OpenMAP-BrainAge (2025, arXiv:2506.17597) trained on ADNI + OASIS-3 (ages 42-95) and found rank-order performance transfers but absolute calibration does not under demographic shift.

**What is missing (as of the 2023-2026 literature checked in Sept 2026).** No study has crossed the three design choices that a transported brain-age pipeline must make (which cohort calibrates the bias correction; whether harmonisation is fitted with or without the target cohort; which correction formula) and scored each cell of that design against longitudinal clinical ground truth. In particular:
- Under Beheshti/de Lange correction, corrected within-person change is `ΔD − a·Δt`; under Cole it is `(ΔD + Δt)/a − Δt`, where `a` is the reference slope. Because `a` depends on the reference cohort's age range and scanner, the same person can be labelled as ageing faster or slower depending on where the correction was calibrated. This has been noted algebraically but never quantified against CDR/MMSE progression.
- Harmonisation fitted on the pooled sample uses the target cohort's own data (leakage in the transport setting) and, when the batch is confounded with disease prevalence (OASIS-3's 1.5T scans are older and enriched for CDR > 0), can remove disease signal. Nobody has compared "harmonise-then-train", "train-then-ComBat-transform (fit on reference only)" and "no harmonisation" on the same longitudinal clinical endpoint.
- HCP-YA (22-37) + HCP-D (5-21) + HCP-A (36-100+) together span the lifespan on one scanner family (Siemens Prisma/Skyra, 0.7-0.8 mm MPRAGE), which lets the age-range and scanner axes of transport be varied *separately*, something ADNI/UKB-based work cannot do.

## Research questions and hypotheses

1. **Transport of the bias slope.** H1: the slope `a` of `delta ~ age` estimated on HCP-Lifespan differs from that estimated on OASIS-3 controls by more than its 95 % CI, and the difference scales with the mismatch in training vs test age range.
2. **Correction × calibration cohort.** H2: for OASIS-3 participants with ≥ 3 scans, the per-person delta slope (years of delta per year) computed under {none, Beheshti, Cole, Smith-quadratic} × {calibrated on HCP-Lifespan, on OASIS-3 CDR = 0, on the pooled sample} varies in sign for a non-trivial fraction (> 10 %) of participants.
3. **Clinical validity of within-person change.** H3: the association between delta slope and CDR-SB / MMSE slope (mixed model, `years:converter` interaction) is largest when correction is calibrated on age-matched, scanner-matched controls and shrinks when calibrated on HCP-YA; Cole correction gives a noisier but less calibration-dependent estimate than Beheshti.
4. **Harmonisation and disease signal.** H4: ComBat fitted on all OASIS-3 scans (including CDR > 0) attenuates the delta-vs-CDR association relative to ComBat fitted on CDR = 0 scans and transformed onto the rest; the attenuation is larger for the 1.5T batch, where scanner and diagnosis are confounded.
5. **Reliability floor.** H5: test-retest ICC(2,1) of delta for same-day/within-6-month OASIS-3 pairs is > 0.85 within scanner and drops below 0.7 across scanners without harmonisation; harmonisation restores ICC but not the clinical association (H4), demonstrating that reliability and validity can move in opposite directions.
6. **Age-range extrapolation.** H6: models trained on HCP-YA + HCP-D (≤ 37 y) applied to OASIS-3 (42-95 y) show a bias slope steeper than −0.5, and no post-hoc correction restores a within-person association with CDR; a model trained on HCP-A alone (36-100+) does transport.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| HCP Young Adult S1200 | training/test cohort, ages 22-37; FreeSurfer 5.3-HCP stats; Siemens Skyra 3T, 0.7 mm | 1,113 subjects with FreeSurfer; 45 retest sessions | free registration + Open Access Data Use Terms; exact age needs Restricted Access | https://db.humanconnectome.org ; S3 `hcp-openaccess` |
| HCP-Aging (Lifespan) | reference cohort ages 36-100+, same protocol family | ~1,200 subjects | NDA Data Use Certification (collection 2847) | https://nda.nih.gov |
| HCP-Development (Lifespan) | ages 5-21, extrapolation tests | ~1,300 subjects | NDA Data Use Certification (collection 2846) | https://nda.nih.gov |
| OASIS-3 | longitudinal target: ~2,800 MR sessions, 1,378 participants (42-95 y), 4 Siemens scanners (1.5T Vision/Sonata, 3T TIM Trio, 3T Biograph mMR), FreeSurfer 5.3-HCP assessors, CDR/MMSE at ~annual ADRC visits | ~1,100 participants with ≥ 2 MR sessions | free registration + Data Use Agreement (XNAT Central) | https://www.oasis-brains.org ; https://central.xnat.org |
| CoRR (optional) | additional test-retest across scanners | ~1,600 subjects | open (INDI) | http://fcon_1000.projects.nitrc.org/indi/CoRR/html/ |

Detailed download steps: `data/README.md`; scripted parts: `scripts/download_data.py`.

## Cohort definitions and key variables (OASIS-3)

| Definition | Rule |
|---|---|
| Baseline scan | first MR session with an ADRC visit within ±365 days (`align_clinical_to_scans`) |
| Converter | CDR = 0 at baseline scan and CDR ≥ 0.5 at ≥ 2 consecutive later ADRC visits |
| Stable control | CDR = 0 at every ADRC visit, ≥ 2 MR sessions ≥ 2 years apart |
| Prevalent impairment | CDR ≥ 0.5 at baseline; cross-sectional associations only, excluded from trajectory models |
| Reliability pair | two MR sessions of one participant < 180 days apart; split into same-scanner and cross-scanner subsets |
| Session exclusions | FreeSurfer failure, eTIV or total-grey outlier > 3 MAD within scanner, missing T1w |
| Calibration sets | (i) HCP-Lifespan out-of-fold predictions; (ii) OASIS-3 stable controls (baseline scan only); (iii) pooled |

Per-session variables: `subject`, `session`, `days` (OASIS `dXXXX`), `years` since first scan, `age`, `age_baseline`, `sex`, `scanner` (TIM Trio 3T / Biograph mMR 3T / Vision 1.5T / Sonata 1.5T), `field_strength`, `mriqc_flag` (optional), `delta_raw`, `delta_<method>_<calibration>`, `cdr`, `cdr_sb`, `mmse`, `apoe4`, `education`, `converter`.

## Quick start (module API)

```python
import numpy as np, pandas as pd
from brainage_transport import (build_feature_table, ComBat, BrainAgeModel,
                                cross_validated_predictions, BiasCorrector)
from brainage_transport.longitudinal import align_clinical_to_scans, fit_delta_trajectory_model

# 1. features: {session_id: path/to/stats}
X_ref = build_feature_table({s: f"data/hcp_lifespan/hcp_aging/{s}/stats" for s in hcpa_sessions})
X_oas = build_feature_table({s: f"data/oasis3/freesurfer/{s}/stats" for s in oasis_fs_ids})

# 2. harmonisation fitted on controls only, transformed onto every OASIS-3 scan (no refit)
cb = ComBat().fit(X_ctrl, scanner_ctrl, covars=np.c_[age_ctrl, sex_ctrl])
X_oas_h = cb.transform(X_oas, scanner_oas, covars=np.c_[age_oas, sex_oas])

# 3. reference model, out-of-fold predictions -> bias correction calibrated on the reference
oof = cross_validated_predictions(X_ref, age_ref, groups=subject_ref, kind="ridge")
bc = BiasCorrector(method="cole").fit(oof["age"], oof["pred"])
model = BrainAgeModel(kind="ridge").fit(X_ref, age_ref)
pred = model.predict_external(X_oas_h, age=age_oas)          # flags extrapolated ages
pred["delta_corr"] = bc.corrected_delta(pred["age"], pred["pred"])

# 4. longitudinal evaluation against CDR conversion
scans = align_clinical_to_scans(scan_table.assign(delta_corr=pred["delta_corr"].values), clinical_table)
res = fit_delta_trajectory_model(scans, delta_col="delta_corr", group_col="converter")
print(res.params["years:converter"], res.pvalues["years:converter"])
```

Swap `method` / calibration set / harmonisation design in loops to fill the 240-cell grid; `bias_correction.apply_all_methods` and `longitudinal.pipeline_comparison_table` collect the results.

## Methods

1. **Features** (`freesurfer_stats.py`): parse `aseg.stats` + `lh/rh.aparc.stats` into 33 subcortical volumes, 68 × {thickness, area, volume}, eTIV and global volumes (identical feature vector for all cohorts; all are FreeSurfer 5.3-HCP, which removes the FreeSurfer-version confound between HCP and OASIS-3). Optional eTIV normalisation.
2. **Models** (`brainage.py`): ridge (RidgeCV) and HistGradientBoosting on standardised features, subject-grouped 5-fold CV; out-of-fold predictions feed the corrections. A CNN comparator (SFCN, Peng et al., 2021, *Med Image Anal*) is a stretch goal, not required for the audit.
3. **Training designs** (the "source" axis): (a) HCP-A only; (b) HCP-A + HCP-D + HCP-YA (lifespan); (c) OASIS-3 CDR = 0 baseline scans; (d) pooled.
4. **Harmonisation designs** (`harmonize.py`): none; ComBat fitted on the pooled sample; ComBat fitted on reference/controls only and *transformed* onto target scans (`ComBat.fit` / `.transform`); mean-only ComBat; age and sex as preserved covariates. Cross-check against `neuroCombat` on identical inputs.
5. **Correction designs** (`bias_correction.py`): none, Beheshti, de Lange (equivalent, kept for reporting), Cole, Smith-quadratic; each calibrated on (i) held-out reference predictions, (ii) OASIS-3 CDR = 0, (iii) pooled.
6. **Longitudinal evaluation** (`longitudinal.py`): align each MR session to the nearest ADRC visit (±365 d); mixed model `delta ~ years * converter + age_baseline + sex + scanner` with random intercept (and slope) per participant; per-person slopes; ICC(2,1) for scan pairs < 6 months apart, stratified by same/different scanner.
7. **Full factorial**: 4 sources × 4 harmonisations × 5 corrections × 3 calibrations = 240 pipelines; each yields (bias slope on target, MAE, ICC, `years:converter` coefficient, delta-vs-MMSE slope). Results are summarised as a specification curve.

## Evaluation and statistics

- **Primary endpoint**: the `years:converter` interaction (difference in delta slope between participants who progress from CDR 0 to ≥ 0.5 and those who stay at 0) and the within-person correlation of delta slope with MMSE slope, with 95 % CIs from participant-level bootstrap (1,000 resamples).
- **Bias metrics**: slope and r of `delta ~ age` on the target before and after correction (`age_bias_slope`); fraction of participants whose per-person slope changes sign across pipelines.
- **Leakage prevention**: subject-grouped folds (no repeat visits across folds); corrections and ComBat fitted only on the designated calibration set; OASIS-3 clinical labels never enter training; HCP-YA exact age only used if Restricted Access is granted, otherwise HCP-YA is test-only.
- **Reliability**: ICC(2,1) with Shrout-Fleiss CIs; Bland-Altman limits of agreement for cross-scanner pairs.
- **Multiple comparisons**: the 240 pipelines are treated as a multiverse (Steegen et al., 2016, *Perspect Psychol Sci*); inference is on the distribution of estimates (median, IQR, sign consistency) rather than on any single cell, plus a joint permutation test that shuffles converter labels within age strata to get a null for the specification curve.
- **Nulls / sanity**: `simulate_regression_to_mean` reproduces the expected slope `sqrt(R²) − 1`; `simulate_longitudinal_cohort` gives a known `years:converter` effect that every pipeline must recover in the no-shift case.

## Publishable angle

**Headline result:** a figure showing, for the same OASIS-3 participants, that the estimated brain-ageing acceleration in CDR converters ranges from clearly positive to null (or negative) purely as a function of where the bias correction and harmonisation were calibrated, together with a recommended protocol (controls-only ComBat transform + Cole or controls-calibrated Beheshti correction + report the reference slope) that keeps the clinical effect stable.

Target venues: *NeuroImage*, *Human Brain Mapping*, *Imaging Neuroscience*, *Alzheimer's & Dementia: DADM*. Methods-focused alternative: *MELBA* or a MICCAI/MIDL workshop paper on the transport formulae.

Follow-ups: (i) extend to UK Biobank longitudinal repeat imaging; (ii) a "reference slope registry" that model authors publish alongside weights; (iii) test whether longitudinal ComBat (Beer et al., 2020, *NeuroImage*) or ComBat-GAM changes the conclusions; (iv) CNN models to check whether the effect is feature-set-specific.

## Risks, confounds and mitigations

| Risk | Mitigation |
|---|---|
| HCP-YA exact age unavailable (open tier gives 5-year bins) | apply for Restricted Access; otherwise use HCP-YA only as a test cohort with bin midpoints and report sensitivity |
| FreeSurfer version differences (HCP 5.3-HCP vs a re-run) | all three cohorts ship FreeSurfer 5.3-HCP outputs; if re-running, use one container for all and treat version as a batch |
| Scanner confounded with diagnosis and calendar time in OASIS-3 (1.5T scans are earlier and more impaired) | stratified ComBat fit on CDR = 0 only; report scanner-within-participant pairs; include scan year as covariate in sensitivity analyses |
| Converters are older and have more visits (selection) | age-stratified permutation null; inverse-probability weighting for visit count |
| HCP-A protocol (Prisma, 0.8 mm) vs HCP-YA (Skyra, 0.7 mm) differ, so "same scanner family" is approximate | treat as separate batches; include a within-HCP transport (YA→A) as a positive control |
| NDA/DUA delays | start with OASIS-3 + HCP-YA (fast), add Lifespan when approved |
| Small number of converters with ≥ 3 scans (~150-250) | use continuous CDR-SB and MMSE slopes as well as the binary converter definition |

## Milestones

- [ ] Access: ConnectomeDB (week 1), OASIS-3 DUA (weeks 1-3), NDA DUC (weeks 2-8)
- [ ] Feature tables for HCP-YA and OASIS-3 (`build_feature_table`), QC on eTIV/outliers
- [ ] Baseline models + out-of-fold predictions per source design; bias slopes per cohort (H1)
- [ ] Correction × calibration grid on OASIS-3; sign-flip statistics (H2)
- [ ] Clinical alignment + mixed models; specification curve (H3, H4)
- [ ] Test-retest / cross-scanner ICC analysis (H5)
- [ ] Add HCP-A/HCP-D; age-range extrapolation experiments (H6)
- [ ] Preregistration-style analysis plan posted (OSF) before the clinical analysis
- [ ] Manuscript + released reference slopes and harmonisation parameters

## Ethics and data-use notes

- OASIS-3 and HCP require DUA/DUT acceptance; NDA requires an institutional DUC. Participant-level data stay in `data/` (git-ignored) and are never uploaded to third-party services or LLM APIs.
- HCP Restricted Data (exact age) must be analysed in a way that does not allow reconstruction from shared outputs; publish only aggregate slopes and model coefficients.
- Report scanner and demographic composition of every calibration set, because the audit's whole point is that corrections carry cohort-specific assumptions.
