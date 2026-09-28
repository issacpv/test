# Sleep staging in disease: automated-staging error as a measurement-error problem for sleep epidemiology

**One-sentence pitch.** Characterise how automated sleep-staging errors (especially N3 -> N2 collapse) depend on obstructive sleep apnoea (OSA) severity and heart failure in the NSRR cohorts (SHHS, MESA, MrOS, CFS), test whether that error is *differential* with respect to cardiovascular outcomes, and quantify and correct the resulting bias in hazard ratios for "slow-wave sleep -> incident hypertension / CVD / mortality" using regression calibration and SIMEX with human scoring as the validation instrument.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis to early-PhD scale. 6-9 months for the error-structure + measurement-error paper; 12 months if a from-scratch deep stager is trained rather than using pretrained U-Sleep/YASA.
- Compute: CPU is enough for feature-based staging (`stagebias.features`) and all statistics. Running U-Sleep or YASA over ~12,000 PSGs (SHHS1+2, MESA, MrOS, CFS) needs ~1-2 CPU-days with YASA or a single GPU-day with U-Sleep. Storage: SHHS ~ 1.3 TB of EDF (SHHS1 5,804 + SHHS2 2,651 PSGs); MESA Sleep ~ 2,056 PSGs; MrOS Sleep ~ 2,900 PSGs (visit 1) ; CFS ~ 730 PSGs. Sleep-EDF Expanded is ~ 8 GB.
- Related project in this repo: `sleep-spindle-aging-biomarker` (spindle metrics in NSRR cohorts; shares the NSRR download tooling). This project is self-contained.

## Background

Deep-learning sleep stagers now match human inter-scorer agreement on average (U-Sleep, Perslev et al., 2021, *npj Digital Medicine*; YASA, Vallat & Walker, 2021, *eLife*; SleepTransformer, Phan et al., 2022, *IEEE TBME*), and are increasingly used to score epidemiological cohorts and consumer devices at scale. Sleep-architecture variables produced this way (N3 percentage, REM latency, N1 fragmentation) are then entered as *exposures* in outcome models: N3 percentage is associated with incident hypertension in SHHS (Javaheri et al., 2018, *Sleep*) and slow-wave-sleep loss with incident dementia in Framingham (Himali et al., 2023, *JAMA Neurology*).

Two facts make this a measurement-error problem rather than a benchmark problem. First, staging error is not uniform: N3 has the lowest F1 of all stages in patients with sleep-disordered breathing, and both automated and human N3 scoring degrade with age, cortical arousals and respiratory events (the AASM inter-scorer reliability programme reports the lowest agreement for N1 and N3; Rosenberg & Van Hout, 2013, *JCSM*). Second, the same variables that drive staging error (AHI, arousal index, age, heart failure, beta-blocker use) are also causes of the cardiovascular outcomes under study. Error that depends on outcome-related covariates is *differential* misclassification, which can bias effect estimates in either direction, whereas classical non-differential error attenuates them. Epidemiology has standard tools for this (regression calibration, Rosner, Spiegelman & Willett, 1990, *Am J Epidemiol*; SIMEX, Cook & Stefanski, 1994, *JASA*; Carroll et al., 2006, *Measurement Error in Nonlinear Models*), but they have not been applied to AI-derived sleep exposures.

## The research gap

**What has been done.**

- Stager generalisation across cohorts and disorders: U-Sleep was trained on 15,660 PSGs from 16 cohorts (mostly NSRR) and evaluated per cohort (Perslev et al., 2021); Olesen et al. (2021, *Sleep*) showed lower agreement in patients with sleep disorders in a mixed-cohort residual-network stager; Stephansen et al. (2018, *Nat Commun*) reported staging in narcolepsy.
- Fairness/bias of staging: SLEEPYLAND (Fiorillo et al., 2025, *npj Digital Medicine*) evaluated several open stagers and an ensemble (SOMNUS) on ~220,000 h of NSRR and out-of-domain PSG and reported that per-recording performance and derived clinical markers (e.g., N3 duration, REM latency) degrade with AHI, PLMI and male sex, with no architecture removing the bias.
- Cardiorespiratory (EEG-free) staging validated against PSG across OSA severities (e.g., Sun et al., 2020, *Sleep*; Sridhar et al., 2020, *npj Digital Medicine*), including demonstrations that automated stages "reproduce" known cross-sectional associations with age, sex and AHI.
- Outcome epidemiology with human-scored N3: Javaheri et al. (2018) in SHHS; Himali et al. (2023) in Framingham; multiple SHHS/MESA papers on sleep architecture and CVD.

**What is specifically missing.**

1. **Error structure conditional on disease, at the epoch level.** SLEEPYLAND reports recording-level performance as a function of AHI. Nobody has characterised *where* in the night the errors occur relative to scored respiratory events and arousals (NSRR XML annotations give event onsets/durations), whether N3 -> N2 errors cluster in the minutes after apnoea-related arousals, and whether the error rate for a *given true stage* depends on prevalent heart failure, beta-blocker use, or age after adjusting for AHI.
2. **Differential vs non-differential misclassification with respect to outcomes.** Whether P(error | true N3) depends on the outcome (incident CVD, hypertension, death) or its strong causes has never been tested. This determines the direction of bias and which correction is valid.
3. **Propagation of staging error into hazard ratios.** No study has swapped automated for human N3 percentage in a published outcome model (e.g., N3% -> incident hypertension in SHHS) and reported the change in the hazard ratio, nor applied regression calibration / SIMEX with the human-scored subset as a validation study. SLEEPYLAND shows marker bias; it does not show what that does to an epidemiological inference.
4. **Source-domain dependence.** Stagers trained on healthy young adults (Sleep-EDF) vs. disease-enriched cohorts (SHHS) should differ in the *structure* of their N3 errors, not only in mean accuracy. Whether disease-enriched training makes error more or less differential is open.
5. **Uncertainty-aware exposures.** Stagers emit per-epoch posteriors; using the expected N3% (or multiple imputation from the posteriors) instead of argmax stages is a cheap correction whose effect on outcome models is untested.

## Research questions / hypotheses

1. **H1 (stage-specific degradation).** For pretrained U-Sleep and YASA, N3 sensitivity falls monotonically across AHI strata (< 5, 5-15, 15-30, >= 30 events/h) in SHHS1 and MESA, with N3 -> N2 as the dominant confusion; N3 sensitivity in AHI >= 30 is at least 0.15 lower than in AHI < 5, whereas REM and Wake sensitivity fall by < 0.05.
2. **H2 (event-locked error).** Epoch-level staging error is elevated within +/- 2 epochs of scored apnoeas/hypopnoeas and arousals (odds ratio > 1.5 vs event-free NREM epochs), and this explains most of the AHI gradient in H1 (the gradient shrinks by > 50% after excluding event-adjacent epochs).
3. **H3 (differential misclassification).** Among true-N3 epochs, the probability of misclassification depends on prevalent heart failure and age after adjusting for AHI and arousal index (cluster-robust logistic regression, OR for HF > 1.2); i.e., error is differential with respect to CVD risk.
4. **H4 (bias in hazard ratios).** Replacing human-scored N3% with automated N3% in a Cox model for incident hypertension (SHHS1 -> SHHS2 design as in Javaheri et al., 2018) and for incident CVD / all-cause mortality (SHHS CVD outcomes) changes the HR per 10-percentage-point N3 by more than the bootstrap 95% interval of the human-scored estimate, with the direction depending on cohort AHI distribution (attenuation in low-AHI strata, over-estimation when N3 under-scoring tracks HF).
5. **H5 (correction).** Regression calibration using a 10-20% human-scored validation subset, and SIMEX with the empirically estimated error variance, recover the human-scored HR to within its 95% CI; using expected N3% from posteriors reduces the attenuation by at least half relative to argmax staging.
6. **H6 (source domain).** A feature-based stager trained only on Sleep-EDF (healthy, 25-101 y) shows a *steeper* AHI gradient in N3 sensitivity than the same architecture trained on SHHS1, but its error is *less* differential with respect to HF after conditioning on AHI.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Sleep-EDF Expanded (PhysioNet `sleep-edfx` 1.0.0) | 153 Sleep-Cassette + 44 Sleep-Telemetry PSGs (Fpz-Cz, Pz-Oz, EOG, 100 Hz) with R&K hypnograms (EDF+); healthy source domain | 197 recordings, ~8 GB | Open (PhysioNet, no login) | https://physionet.org/content/sleep-edfx/1.0.0/ |
| SHHS (NSRR) | SHHS1 (5,804 PSGs, 1995-98) and SHHS2 (2,651 PSGs, 2001-03): C3/C4-A1/A2 EEG at 125 Hz, EOG, EMG, airflow, SpO2; XML hypnograms + respiratory/arousal events; AHI (3%/4% definitions); baseline CVD history; `shhs-cvd-summary` incident CHD/CVD/CHF/stroke, vital status and censoring dates | ~1.3 TB EDF | DUA (free NSRR Data Access Request; IRB/DUA sign-off) | https://sleepdata.org/datasets/shhs |
| MESA Sleep (NSRR) | 2,056 PSGs (2010-13; Fz/Cz/Oz EEG at 256 Hz) with AASM staging, AHI, arousals; MESA Exam 5 covariates; CVD events available via MESA/NSRR linkage | ~600 GB | DUA (NSRR) | https://sleepdata.org/datasets/mesa |
| MrOS Sleep (NSRR) | ~2,900 men >= 65 y (visit 1, 2003-05), PSG + AHI + CVD/mortality follow-up | ~700 GB | DUA (NSRR) | https://sleepdata.org/datasets/mros |
| CFS (NSRR) | 730 PSGs, family-based cohort enriched for OSA; used for external validation of error structure | ~200 GB | DUA (NSRR) | https://sleepdata.org/datasets/cfs |
| Pretrained stagers | U-Sleep (open weights/code, `utime`), YASA (LightGBM, pip), optional SleepFM / SLEEPYLAND containers | - | Open (MIT/BSD) | https://github.com/perslev/U-Time ; https://github.com/raphaelvallat/yasa ; https://github.com/biomedical-signal-processing/sleepyland |

Outcome variables (SHHS): incident hypertension between SHHS1 and SHHS2 (BP >= 140/90 or antihypertensive use at SHHS2 among normotensive at SHHS1, following Javaheri et al., 2018); incident CVD/CHF and all-cause mortality from `shhs-cvd-summary` (adjudicated through ~2011). Exposure: N3% of total sleep time (human vs automated), plus N3 minutes and slow-wave activity as secondary exposures.

## Methods

1. **Ingest (`stagebias.hypnogram`).** Parse NSRR Compumedics "profusion" XML (`<SleepStages>` 30-s codes 0/1/2/3/4/5, with stage 4 merged into N3; `<ScoredEvent>` respiratory/arousal events) and Sleep-EDF EDF+ annotations into per-epoch integer stages {W, N1, N2, N3, R} plus an epoch-level event mask (+/- k epochs around scored events).
2. **Signals.** One central EEG derivation per cohort (SHHS C4-A1 at 125 Hz; MESA Cz-Oz at 256 Hz; Sleep-EDF Fpz-Cz at 100 Hz), plus EOG; resampled to 100 Hz; 30-s epochs. The resampling/filtering is done with MNE or pyedflib outside the package; the package works on arrays.
3. **Stagers.** (a) Pretrained U-Sleep and YASA applied as released (no fine-tuning) - the deployment scenario. (b) A transparent feature-based stager (`stagebias.features`: log band powers, relative powers, spectral edge, Hjorth parameters, temporal context features; `HistGradientBoostingClassifier`) trained under two source regimes: Sleep-EDF only vs SHHS1 training split (subject-wise), to test H6. (c) Per-epoch posteriors from all stagers are retained.
4. **Error structure (`stagebias.error_structure`).** Confusion matrices per AHI stratum, per-stage sensitivity/PPV/F1 and Cohen's kappa; per-recording N3% bias with Bland-Altman limits vs AHI; event-locked error odds ratios; cluster-robust logistic regression of misclassification among true-N3 epochs on AHI, arousal index, age, sex, prevalent HF, beta-blocker use (H3).
5. **Outcome models (`stagebias.measurement_error`).** Cox proportional-hazards (statsmodels `PHReg`) for incident CVD/mortality and logistic regression for incident hypertension, with covariates as in Javaheri et al. (age, sex, race, BMI, AHI, smoking, diabetes). Three exposure versions: human N3%, automated argmax N3%, automated expected N3% (mean posterior). Corrections: regression calibration (validation subset with human staging), SIMEX (quadratic extrapolation; error variance from the validation subset), and multiple imputation drawing epoch stages from posteriors.
6. **Domain-shift diagnostics.** Feature-space MMD and PSD divergence between Sleep-EDF and SHHS backgrounds, used descriptively to interpret H6.

## Evaluation & statistics

- Primary endpoints: (i) N3 sensitivity by AHI stratum (H1) with subject-level bootstrap CIs (2,000 draws); (ii) HR per 10-point N3% under human vs automated exposure (H4) with paired subject-level bootstrap of the difference; (iii) coverage of the human-scored HR by corrected estimates (H5).
- Validation scheme: pretrained stagers are never fine-tuned on NSRR test recordings. The feature stager uses subject-wise 5-fold CV within the training cohort and is then applied unchanged to other cohorts. Validation subsets for regression calibration are drawn by stratified random sampling (by AHI stratum and sex) and pre-registered.
- Leakage prevention: SHHS subjects appear in both SHHS1 and SHHS2 - splits are by `nsrrid`, never by recording; U-Sleep's training set overlaps NSRR cohorts (it used SHHS, MESA, MrOS, CFS among others), so U-Sleep results on those cohorts are reported as *in-domain* and YASA (trained on a different NSRR subset) and the Sleep-EDF stager as *out-of-domain*; this is stated explicitly in every table.
- Multiple comparisons: H1-H6 are pre-registered families; Holm-Bonferroni within family; per-stage comparisons use stage as a fixed factor in a mixed model (subject random effect) rather than separate tests.
- Nulls: (a) label-permuted stagers to calibrate kappa; (b) a *non-differential* null for H3 by shuffling HF/outcome labels across subjects within AHI strata; (c) a *simulation* null for H4 where automated N3% is replaced by human N3% plus classical Gaussian error of matched variance - if the observed HR shift exceeds this null, the error is not classical.
- Reporting: STROBE-style cohort flow, per-cohort tables, and a "swap-in" figure (HR vs exposure version) per outcome.

## Publishable angle

- **Headline result.** "Automated N3 is under-scored by X points in severe OSA and by Y points in heart failure independent of AHI; this error is differential with respect to cardiovascular outcomes, and substituting automated for human N3% moves the N3-hypertension hazard ratio from A to B. Regression calibration with a 15% human-scored subset restores it." That is a general recipe for AI-derived exposures in cohort studies.
- **Venues.** *Sleep* or *Journal of Clinical Sleep Medicine* (clinical/epidemiological readership); *npj Digital Medicine* (AI-in-epidemiology framing); *American Journal of Epidemiology* or *Epidemiology* (measurement-error methods); *IEEE JBHI* for the error-structure/staging part.
- **Follow-ups.** (i) Wearable/cardiorespiratory staging, where N3 error is larger and the validation-subset design is the only option; (ii) applying the same framework to automated arousal and respiratory-event scoring (AHI itself as an error-prone exposure); (iii) a fairness extension: whether error is differential by race/ethnicity in MESA.

## Risks, confounds & mitigations

- **Scoring-rule shift.** SHHS was scored with R&K (stages 3+4), MESA/MrOS with AASM; N3 definitions differ. Mitigation: merge R&K 3/4 into N3 everywhere; analyse cohorts separately; treat scoring rule as a covariate in pooled models.
- **U-Sleep/YASA training overlap with NSRR.** Reported as in-domain; the Sleep-EDF-trained stager and the SHHS1-trained feature stager (with subject-wise splits) give clean out-of-domain estimates.
- **Human scoring is itself noisy.** Human N3 is the reference, not the truth. Mitigation: use SHHS inter-scorer reliability data and the AASM ISR literature to bound reference error; a sensitivity analysis treats human scoring as error-prone with known kappa (Berkson-type correction).
- **Outcome-model misspecification.** Non-linear N3 effects; competing risks for mortality. Mitigation: splines for N3%, Fine-Gray sensitivity analysis.
- **Low event counts in strata.** Incident CHF is rare. Mitigation: pool CVD outcomes; report HF as a secondary outcome; power analysis from SHHS event counts before pre-registration.
- **Channel/derivation differences across cohorts** confound stager error with hardware. Mitigation: per-cohort analyses; derivation as a covariate; report MESA (256 Hz, Fz/Cz/Oz) and SHHS (125 Hz, C3/C4) separately.

## Cohort, exposure and outcome definitions (pre-specified)

| Element | Definition | Source variables (NSRR) |
|---|---|---|
| Analysis cohort A (incident hypertension) | SHHS1 participants with a scorable PSG (>= 4 h TST), normotensive at SHHS1 (SBP < 140 and DBP < 90, no antihypertensive use), with SHHS2 follow-up; outcome as in Javaheri et al. (2018) | SHHS1/SHHS2 datasets: blood-pressure and medication variables; `nsrrid` for linkage |
| Analysis cohort B (incident CVD, CHF, all-cause mortality) | All SHHS1 participants with a scorable PSG; adjudicated events and censoring dates | `shhs-cvd-summary-dataset`: incident CHD/CVD/CHF/stroke indicators and dates, vital status, censoring date |
| Replication cohort | MESA Sleep exam (2010-13) with MESA Exam 5 covariates; CVD events via a MESA data request; MrOS for older men | `mesa-sleep-dataset`, MESA event files; `mros-visit1-dataset` |
| Exposure (gold) | N3% of TST from human scoring (R&K stages 3+4 merged for SHHS; AASM N3 for MESA/MrOS) | profusion XML `SleepStages` |
| Exposure (automated) | N3% from U-Sleep, YASA and the feature stager; argmax and posterior-expected versions | model outputs (per-epoch posteriors, parquet) |
| Effect modifiers | AHI stratum (< 5, 5-15, 15-30, >= 30 events/h; 3% and 4% desaturation definitions), arousal index, prevalent heart failure, beta-blocker use, age, sex | SHHS1 `ahi_a0h3a` / `ahi_a0h4`, arousal index, CVD-history and medication variables; NSRR harmonized `nsrr_*` equivalents where available |
| Covariates (outcome models) | Age, sex, race/ethnicity, BMI, AHI, smoking, diabetes, baseline SBP | SHHS1 / MESA / MrOS datasets |
| Validation subset | 15% of each cohort, stratified by AHI stratum and sex, fixed by seed before any model is run | `data/manifests/validation_ids.csv` (IDs only; restricted, not committed) |

Epoch-level analysis file (one row per scored epoch; restricted): `record`, `epoch`, `y_true`, `y_pred_<stager>`,
`p_<stage>_<stager>`, `resp_event_adjacent` (+/- 2 epochs), `arousal_adjacent`, plus record-level modifiers broadcast.

## Pre-specified outputs

1. Table 1: cohort characteristics by AHI stratum, including human vs automated N3% and Bland-Altman bias.
2. Figure 1: per-stage sensitivity vs AHI stratum for every stager with subject-level bootstrap CIs (H1); in-domain vs out-of-domain stagers marked.
3. Figure 2: epoch-level error rate as a function of time from the nearest respiratory event / arousal (H2), with the odds ratio inside vs outside event-adjacent epochs.
4. Table 2: cluster-robust logistic regression of misclassification among true-N3 epochs on AHI, arousal index, age, sex, prevalent HF, beta-blocker use (H3).
5. Figure 3: "swap-in" plot of hazard / odds ratios per 10-point N3% for each exposure version and outcome, with the classical-error null band (H4).
6. Table 3: corrected estimates (regression calibration, SIMEX, multiple imputation from posteriors) and whether they cover the human-scored estimate (H5).
7. Supplement: Sleep-EDF-trained vs SHHS1-trained feature stager (H6); R&K vs AASM sensitivity; MESA and MrOS replication.

## Repository layout and quick start

```
src/stagebias/   hypnogram.py (XML/EDF+ ingest, event masks)   features.py (spectral features, stager)
                 error_structure.py (confusions, kappa, N3 bias, event-locked and differential error)
                 measurement_error.py (Cox/logistic, swap-in, regression calibration, SIMEX, nulls)
scripts/download_data.py   tests/test_stagebias.py   data/README.md
```

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q                                     # synthetic-data tests (no downloads)
python scripts/download_data.py --dataset sleep-edfx --sample
python - <<'EOF'
from stagebias.hypnogram import parse_profusion_xml, events_to_epoch_mask
xml = open("data/nsrr/shhs/polysomnography/annotations-events-profusion/shhs1/shhs1-200001-profusion.xml").read()
stages, events = parse_profusion_xml(xml)
mask = events_to_epoch_mask(events.respiratory(), len(stages), pad_epochs=2)
EOF
```

## Milestones

- [ ] NSRR DUA approved for SHHS, MESA, MrOS, CFS; Sleep-EDF downloaded (`scripts/download_data.py`).
- [ ] Manifest of recordings with AHI, arousal index, HF status, outcomes; cohort flow diagram.
- [ ] XML/EDF+ ingest validated against NSRR-provided stage summaries (N3 minutes match to < 1 epoch).
- [ ] Run YASA and U-Sleep on all PSGs; store per-epoch posteriors (parquet).
- [ ] Train the feature stager on Sleep-EDF and on SHHS1 (subject-wise CV); freeze.
- [ ] Error-structure analyses H1-H3 with bootstrap CIs; event-locked error figure.
- [ ] Replicate the human-scored N3% -> incident hypertension model (Javaheri et al., 2018) as the anchor.
- [ ] Swap-in analysis (H4), regression calibration, SIMEX, multiple imputation (H5).
- [ ] Pre-registration (OSF) after the pipeline is frozen on a 10% pilot; then full run.
- [ ] Manuscript + code release with fixed manifests and validation-subset IDs.

## Ethics / data-use notes

- NSRR data (SHHS, MESA, MrOS, CFS) are governed by a Data Access and Use Agreement; they may not be redistributed, and derived per-subject data (including per-epoch stage files) are treated as restricted. Only aggregate tables and code are committed. Do not upload NSRR EDFs, XMLs or per-subject tables to third-party services (including LLM APIs).
- Sleep-EDF is open, but the same rule about not committing data applies (`.gitignore` excludes `data/`).
- Outcome data (mortality, CVD events) are sensitive; keep all analyses on institutional storage; follow the NSRR publication acknowledgement requirements (cite Zhang et al., 2018, *JAMIA*, and the cohort papers: Quan et al., 1997, *Sleep* for SHHS; Chen et al., 2015, *Sleep* for MESA Sleep).
- Pre-register the outcome models to avoid choosing the exposure version post hoc.
