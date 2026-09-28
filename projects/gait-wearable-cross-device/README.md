# GaitShift-PD: cross-device, cross-placement generalisation of Parkinson's gait models across open PhysioNet/UCI/Kaggle datasets, with free-living validation on PPMI Verily Study Watch data

One-sentence pitch: Build a WILDS-style leave-one-dataset-out benchmark for Parkinson's disease detection, severity and freezing-of-gait from eight open gait datasets that differ in sensor (force plates, footswitches, IMUs, smartwatches), body placement (foot, ankle, thigh, lower back, wrist), sampling rate and protocol, quantify the deployment gap for hand-crafted, deep and self-supervised accelerometry models, and validate the survivors on PPMI's free-living wrist data.

## Status / difficulty / timeline / compute

- Status: proposal + starter code (dataset registry and harmonisation: units, resampling, gravity removal, orientation-invariant channels, windowing; gait features including cadence, stride regularity, harmonic ratio, freeze index and VGRF stride-time variability; leave-one-dataset-out benchmark with subject-level bootstrap and a classifier two-sample shift test).
- Difficulty: MSc-level for the benchmark; PhD-level with the self-supervised transfer and PPMI free-living validation. 6-9 months.
- Compute: CPU for features and classical models; one GPU (8-16 GB) for 1-D CNN / SSL fine-tuning. Total data < 50 GB.
- Related projects (kept separate): `cross-dataset-seizure-generalization` and `ecg-cross-dataset-generalization` (same leave-one-dataset-out methodology in other modalities), `icu-model-transportability`.

## Background

Wearable gait analysis is among the most mature digital-biomarker areas in Parkinson's disease (PD): wrist inertial sensors track motor fluctuations in daily life (Powers et al., 2021, Sci Transl Med), smartphone apps capture bradykinesia and gait (Lipsmeier et al., 2018, Mov Disord; mPower, Bot et al., 2016, Sci Data), and freezing of gait (FoG) can be detected from leg/back accelerometers (Bachlin et al., 2010, IEEE Trans Inf Technol Biomed; freeze index, Moore et al., 2008, J Neurosci Methods). Open datasets now span very different hardware: PhysioNet's VGRF walking database (Yogev et al., 2005, Eur J Neurosci; Frenkel-Toledo et al., 2005, Mov Disord; Hausdorff et al., 2000, J Appl Physiol for neurodegenerative gait), the Daphnet FoG set, the Kaggle/TLVMC FoG contest data (Salomon et al., 2024, Nat Commun), the PADS smartwatch dataset (Varghese et al., 2024; PhysioNet), the 2025-2026 WearGait-PD release (Sci Data, 2026), the LTMM free-living lower-back set (Weiss et al., 2013, Neurorehabil Neural Repair), and PPMI's Verily Study Watch programme (343 participants with ~485 days each of hourly derived measures; npj Parkinson's Disease, 2024 and 2026). Self-supervised accelerometry models pre-trained on 700k person-days of UK Biobank wrist data (Yuan et al., 2024, npj Digit Med) offer a transferable backbone.

## The research gap

What has been done:

- Cross-dataset FoG detection: Sigcha et al. (2024) compared deep models across three FoG datasets; a 2026 Sensors study reported a laboratory F1 of ~1.0 collapsing to ~0.55 in daily-living data (an "83% deployment gap"); FM-FoG (arXiv 2025) proposes a foundation-model wearable system.
- Within-dataset PD-vs-control and severity models on PADS, mPower and VGRF data reach high accuracies but are trained and tested on the same device and protocol.
- Domain generalisation benchmarks (WILDS; Koh et al., 2021, ICML) exist for images/text, not for PD gait.

What is specifically missing:

1. No benchmark holds out an entire *device/placement family* (e.g., train on foot/back IMUs, test on wrist smartwatch; train on VGRF, test on IMU) for tasks beyond FoG: PD vs control, MDS-UPDRS-III severity strata, gait-speed/stride-variability regression.
2. Harmonisation choices (gravity removal, orientation invariance, resampling, unit conversion, window length) are never ablated across devices, so it is unknown which part of the deployment gap is preprocessing versus physiology.
3. Transfer of UK-Biobank-pretrained self-supervised wrist models to non-wrist placements and to PD tasks is untested, as is transfer of lab-trained models to PPMI free-living wrist measures.

Sharpened angle: a reusable benchmark (GaitShift-PD) with a fixed harmonisation pipeline, three task families, subject-level bootstrap and per-shift-type analysis (device, placement, protocol, population), plus an ablation isolating preprocessing from physiological transfer and an SSL-transfer study, ending in PPMI free-living validation.

## Research questions / hypotheses

1. H1 (deployment gap): For PD-vs-control classification, leave-one-dataset-out (LODO) balanced accuracy is at least 15 points lower than within-dataset grouped-CV accuracy for all model families; the gap is largest for placement shifts (wrist <-> lower body) and smallest for protocol shifts within the same placement.
2. H2 (harmonisation): Orientation-invariant channels (magnitude, PCA-vertical) plus gravity removal recover >= 30% of the LODO gap relative to raw-axis inputs (paired bootstrap of the difference); resampling rate (50 vs 100 Hz) has < 3 points effect.
3. H3 (SSL transfer): A UK-Biobank self-supervised wrist backbone fine-tuned on one dataset transfers to other *wrist* datasets (PADS, PPMI) better than a from-scratch CNN (>= 5 points balanced accuracy), but not to lower-body placements.
4. H4 (task transfer): Gait-timing features (cadence, stride-time CV, stride regularity) transfer across VGRF and IMU placements (LODO regression R^2 for stride-time CV >= 0.5), whereas amplitude features do not; a timing-only model is therefore more device-robust for severity estimation.
5. H5 (free-living): Lab-trained severity models applied to PPMI Verily-derived walking measures correlate with MDS-UPDRS-III (Spearman rho >= 0.3) only after within-person aggregation over >= 14 days; single-day estimates do not.

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| Gait in Parkinson's Disease (gaitpdb) | VGRF (8 sensors/foot, 100 Hz), PD vs control, UPDRS/H&Y, dual task | 93 PD, 73 controls | Open (PhysioNet, ODC-By) | https://physionet.org/content/gaitpdb/1.0.0/ |
| Gait in Neurodegenerative Disease (gaitndd) | Footswitch stride timing; PD/HD/ALS/control | 64 subjects | Open (PhysioNet) | https://physionet.org/content/gaitndd/1.0.0/ |
| Long Term Movement Monitoring (ltmm) | 3-day free-living lower-back accelerometer (100 Hz), older adults, falls | 71 subjects | Open (PhysioNet) | https://physionet.org/content/ltmm/1.0.0/ |
| PADS - Parkinson's Disease Smartwatch | Bilateral wrist accelerometer/gyroscope, 11 movement tasks, PD / differential diagnoses / controls | 469 participants | Open (PhysioNet) | https://physionet.org/content/pads-parkinsons-disease-smartwatch/1.0.0/ |
| Daphnet Freezing of Gait | Ankle/thigh/trunk accelerometers (64 Hz), FoG annotations | 10 PD | Open (UCI, CC BY) | https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait |
| TLVMC Parkinson's FoG prediction (Kaggle 2023: tdcsfog 128 Hz, defog 100 Hz, daily living) | Lower-back accelerometer, FoG events, lab + home | ~100 subjects | Kaggle account + competition rules | https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction |
| WearGait-PD (Sci Data, 2026) | Multi-sensor wearables, PD + age-matched controls | see paper | Open (verify host in the paper's data-availability statement) | https://www.nature.com/articles/s41597-026-06806-2 |
| mPower | Smartphone (pocket) walking/balance, self-reported PD | >10k participants | Synapse registration + qualified-researcher terms | https://www.synapse.org/mpower |
| PPMI Verily Study Watch (derived hourly measures; raw on request) | Free-living wrist validation, MDS-UPDRS-III | 343 participants, ~485 days each | Application to PPMI (online form + Data Use Agreement, free; derived data via LONI IDA download portal) | https://www.ppmi-info.org/access-data-specimens/download-data |

## Methods

1. Registry and harmonisation (`gaitshift.harmonize`): per-dataset metadata (device, placement, fs, units); pipeline = unit conversion to m/s^2 -> resampling to 50 Hz (polyphase) -> gravity estimation (0.5 Hz low-pass) and removal -> orientation-invariant channels (magnitude, PCA-vertical, horizontal magnitude) -> 5 s windows with 50% overlap, subject-indexed. VGRF and footswitch data enter through stride-timing features only.
2. Features (`gaitshift.features`): cadence by autocorrelation, stride regularity (Moe-Nilssen & Helbostad, 2004, J Biomech), harmonic ratio, freeze index (3-8 Hz / 0.5-3 Hz power), RMS, jerk, spectral entropy, band powers; VGRF heel-strike detection and stride-time CV.
3. Models: (a) feature + logistic regression / gradient boosting; (b) 1-D CNN on harmonised windows; (c) UK-Biobank SSL backbone (Yuan et al., 2024) fine-tuned; (d) domain-generalisation baselines (group DRO, CORAL) and test-time normalisation.
4. Tasks: PD vs control (gaitpdb, PADS, WearGait-PD, mPower); severity strata (H&Y 2 vs 3, UPDRS tertiles where available); FoG episode detection (Daphnet, Kaggle FoG); stride-time CV regression (VGRF <-> IMU).
5. Benchmark protocol (`gaitshift.benchmark`): LODO with subject-level bootstrap CIs; within-dataset GroupKFold for the in-distribution reference; deployment gap = ID - LODO; shift diagnostics via classifier two-sample test on harmonised features; shift typed as device / placement / protocol / population.
6. Free-living validation: PPMI Verily walking measures (step counts, gait-speed proxies where provided) aggregated per person over 14/30/90-day windows; association with MDS-UPDRS-III at the nearest visit (mixed models).
7. Tools: numpy/scipy, scikit-learn, PyTorch, wfdb (PhysioNet), synapseclient (mPower), kaggle CLI.

## Evaluation & statistics

- Primary metric: balanced accuracy and AUROC (classification), R^2 and Spearman rho (regression), event-level F1 with 1 s tolerance (FoG).
- Subject-level bootstrap (2,000 resamples) for CIs; paired bootstrap for model/harmonisation comparisons; Holm correction within each hypothesis family.
- Leakage: subject-grouped splits everywhere; harmonisation statistics (PCA axes, normalisation) fitted per window or on the training datasets only; no test-dataset tuning.
- Nulls: label permutation within dataset (chance LODO); "dataset-identity" probe to quantify how much of the signal is dataset membership rather than disease (a model trained to predict PD should not exceed chance when the target dataset's labels are shuffled but dataset identity is preserved).
- Reporting: per-shift-type tables and a leaderboard file format so others can add datasets.

## Publishable angle

Headline: "Across eight open PD gait datasets, models lose most of their accuracy when the sensor or placement changes; gait-timing features and orientation-invariant harmonisation recover a third of the loss, and only aggregated multi-week free-living wrist data reproduce lab severity estimates."

Target venues: npj Digital Medicine; IEEE Journal of Biomedical and Health Informatics; Journal of NeuroEngineering and Rehabilitation; NeurIPS Datasets & Benchmarks track (benchmark paper); Movement Disorders (clinical short report on the PPMI validation).

Follow-ups: add gyroscope-only tasks, tremor detection across devices, and non-PD gait disorders (gaitndd HD/ALS) as out-of-distribution negatives; release harmonised windows as a PhysioNet derived dataset where licences allow.

## Risks, confounds & mitigations

- Different tasks/protocols across datasets (dual-task walking, turns, ADL): restrict primary comparisons to straight-walking bouts detected by cadence stability; report protocol shift separately.
- Age and sex imbalance between PD and controls differ per dataset: covariate-adjusted metrics and age-matched subsampling sensitivity analyses.
- Label heterogeneity (self-reported PD in mPower vs clinician-confirmed elsewhere): mPower used only as an out-of-distribution test with a clearly labelled "self-report" flag.
- Kaggle FoG licence limits redistribution: derived features only; check the competition terms before any release.
- PPMI derived measures are proprietary aggregates (Verily): treat as a validation-only cohort; document exact variable names from the LONI download.
- Small FoG cohorts (Daphnet n = 10): subject-level bootstrap will be wide; FoG is a secondary task.

## Milestones

- [ ] Download open datasets (`scripts/download_data.py --all-open`); build the registry; run harmonisation and feature extraction.
- [ ] In-distribution baselines per dataset (GroupKFold); LODO benchmark v0 for PD vs control (H1).
- [ ] Harmonisation ablation (H2); timing-vs-amplitude transfer (H4).
- [ ] SSL wrist-backbone transfer (H3); FoG cross-dataset task.
- [ ] PPMI application; Verily derived-measure validation (H5).
- [ ] Benchmark release (code, leaderboard format) and manuscript.

## Ethics / data-use notes

- PhysioNet open datasets (gaitpdb, gaitndd, ltmm, PADS) carry their own licences (ODC-By / PhysioNet open licences): cite and attribute; UCI Daphnet is CC BY; Kaggle data are governed by competition rules; mPower requires Synapse qualified-researcher terms and prohibits re-identification; PPMI requires a signed DUA and forbids redistribution and re-identification.
- Free-living accelerometry is behaviourally sensitive: never commit raw data; publish only aggregated results; do not send participant-level data to third-party services.
- Report performance by sex and age; PD cohorts are predominantly older and male in several datasets.
