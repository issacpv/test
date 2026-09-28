# An open, bias-corrected neonatal EEG maturation clock: functional brain age from open term-range cohorts, normative feature centiles, and EEG-age deviation as a marker of encephalopathy severity and seizure burden

**One-sentence pitch.** Build the first fully open and reproducible neonatal "EEG age" clock (predicted vs postmenstrual age, PMA) from the Helsinki seizure-annotated cohort and the HIE background-grading cohort, correct its deviation score for the regression-to-the-mean bias that the adult brain-age literature has documented but the neonatal literature has not, release normative centiles of maturational EEG features (interburst intervals, continuity, rEEG, spectral slope, interhemispheric synchrony) against PMA, and test whether the corrected EEG-age deviation tracks hypoxic-ischaemic encephalopathy (HIE) grade and seizure burden, and whether it is a trait of the infant or a state of the recording.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis (feature clock, bias correction, outcome associations on the two open cohorts, ~6-8 months) to PhD chapter (deep clock, preterm extension via data-sharing agreements, state/trait decomposition, ~12-18 months).
- Compute: CPU only for the feature clock (open cohorts total < 200 h of 19-channel EEG at 256 Hz). One GPU for an optional CNN clock. Storage ~15 GB.
- Related projects: `cross-dataset-seizure-generalization` uses the Helsinki cohort for seizure detection (age 0 extreme); `lifespan-normative-models` and `brain-age-transportability` cover MRI brain-age methodology (bias correction, normative modelling) whose logic is reused here conceptually. No code is shared.

## Background

The neonatal EEG matures on a scale of weeks: tracé discontinu gives way to continuity as interburst intervals shorten, delta brushes and other transients disappear around 36-38 weeks PMA, sleep states differentiate, and interhemispheric synchrony increases (developmental features and glossary: André et al., 2010, *Neurophysiol. Clin.*). Because these changes are lawful, the EEG can be used to estimate maturational age, and the difference between EEG-estimated and true PMA ("functional brain age" deviation) is a candidate biomarker of delayed or disrupted maturation. Automated estimators exist: O'Toole et al., 2016 (*Clin. Neurophysiol.*; support-vector regression on spontaneous-activity-transient features in very/extremely preterm infants, error ~1.3 weeks), Stevenson et al., 2017 (*Sci. Rep.*; serial recordings in preterm infants), the NEURAL feature set (O'Toole & Boylan, 2017, arXiv 1704.05694), and the 2025 NeoNaid multi-task deep model with quality control (medRxiv 2025.10.16.25338113; ~1,300 h from 124 age-labelled recordings across centres). Pavlidis et al., 2025 (*J. Clin. Med.*) compared visual and automated EEG maturational-age estimation in preterm infants; two 2025 *Clin. Neurophysiol.* papers report automated EEG-maturity estimation in preterm neonates and machine-learning prediction of cognitive outcome from preterm EEG; a 2025 *J. Neurosci.* study used dense-array EEG and a functional-brain-age measure to resolve cortical maturation across the birth transition.

Two open cohorts make a reproducible term-range clock possible: the Helsinki neonatal EEG dataset (Stevenson et al., 2019, *Sci. Data*; 79 term neonates, 19-channel 256 Hz, three expert seizure annotators, clinical information; PMA at recording within 35-45 weeks by design) and the HIE background-grading dataset (O'Toole et al., 2023, *Sci. Data*; one-hour multichannel EEG epochs from term neonates with HIE, each graded for background severity by experts, on Zenodo). Adult and paediatric MRI brain-age work has shown that the raw age deviation is biased by regression to the mean and must be corrected before being related to outcomes (Le et al., 2018, *Front. Aging Neurosci.*; Smith et al., 2019, *NeuroImage*; Butler et al., 2021, *Hum. Brain Mapp.*); no neonatal EEG-age paper applies such a correction.

## The research gap

**What has been done.**

- FBA estimators trained on private cohorts (O'Toole et al., 2016; Stevenson et al., 2017; NeoNaid, 2025) with age errors of ~1-2 weeks; multi-centre validation (NeoNaid) with quality control for out-of-distribution inputs.
- Clinical use of deviation: delayed FBA associated with adverse outcome in preterm cohorts (Stevenson et al., 2017 and the 2025 *Clin. Neurophysiol.* outcome paper).
- HIE background grading and its prognostic value (Murray et al., 2009, *Pediatrics*; O'Toole et al., 2023 dataset); automated grading models exist, but grading and "EEG age" have not been put on the same axis.
- Brain-age methodology: bias correction and normative modelling are standard in imaging (Smith et al., 2019; Rutherford et al., 2022, *eLife*) and absent in neonatal EEG.

**What is specifically missing.**

1. **An open clock with released weights and fixed splits** that anyone can apply to a BIDS/EDF neonatal recording, trained only on open data.
2. **Bias-corrected deviation.** Whether the EEG-age delta correlates with PMA itself (regression dilution) in neonates, and how much outcome associations change after correction.
3. **Normative centiles of maturational features vs PMA** (interburst interval, continuity, rEEG, spectral slope, synchrony) so that individual recordings can be placed on centile charts, as is done for growth and for MRI.
4. **State vs trait.** Whether the deviation is stable across epochs and sleep states within an infant (trait) or moves with vigilance, medication and recording quality (state); test-retest ICC has not been reported for neonatal EEG age.
5. **Deviation vs HIE grade and seizure burden on open data**, with PMA-adjusted, permutation-tested associations.
6. **A transparent statement of the range.** The open cohorts cover 35-45 weeks; a preterm extension needs data-sharing agreements (candidate cohorts listed below), so the clock and its limitations are reported honestly by PMA bin.

## Research questions / hypotheses

1. **H1 (accuracy).** On seizure-free, artefact-screened Helsinki epochs, subject-grouped CV gives MAE <= 1.5 weeks for PMA within 35-45 weeks with the interpretable feature clock; a CNN clock does not improve MAE by more than 0.3 weeks.
2. **H2 (bias).** The raw delta correlates negatively with PMA (Pearson r < -0.3); after Smith-style correction the correlation is ~0 and the Spearman correlation between raw and corrected subject rankings is < 0.9 (i.e. correction changes who looks "delayed").
3. **H3 (HIE).** Corrected delta decreases monotonically with HIE background grade (Jonckheere-Terpstra trend, permutation p < 0.01); grade >= 2 recordings have a median delta <= -2 weeks, and the clock's out-of-distribution flag (multivariate feature deviation) fires in the majority of grade-3/4 recordings.
4. **H4 (seizure burden).** In Helsinki, consensus seizure burden (min/h) is associated with more negative corrected delta (partial Spearman rho < -0.3 adjusting for PMA), computed on seizure-free epochs only.
5. **H5 (state vs trait).** Within-infant ICC of delta across 1-h epochs is > 0.6; the active-vs-quiet-sleep difference in delta is < 1 week; recording-quality covariates explain < 20% of between-infant variance.
6. **H6 (transfer).** A Helsinki-trained clock applied to grade-0/1 HIE recordings gives MAE <= 2 weeks after per-record robust scaling; reducing the montage from 19 to 8 channels costs < 0.5 weeks.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Helsinki neonatal EEG with seizure annotations (Stevenson et al., 2019) | 79 term neonates, 19-channel 256 Hz EDF, three annotators, clinical information (gestational and postnatal age fields used to derive PMA); seizure-free epochs for training, seizure burden as outcome | ~74 h, ~4 GB | Open (Zenodo, CC-BY) | https://zenodo.org/ (search the dataset title; DATASETS.md lists record 2547147) |
| Neonatal EEG graded for severity of background abnormalities in HIE (O'Toole et al., 2023) | One-hour multichannel EEG epochs from term neonates with HIE, expert background grades; external validation and H3 | tens of hours | Open (Zenodo, CC-BY) | https://zenodo.org/ (search the dataset title) |
| Candidate preterm cohorts (not open; require agreements) | Serial preterm recordings (Helsinki, Stevenson et al., 2017), Cork preterm cohorts (O'Toole/Boylan), NeoNaid centres; needed to extend the clock below 35 weeks | ~ | DUA / collaboration | see the respective papers |
| Optional | Any neonatal/infant EEG dataset appearing on OpenNeuro with PMA or age metadata (none confirmed at scaffold time) | ~ | Open | https://openneuro.org |

## Methods

1. **Ingestion (`scripts/download_data.py`).** Zenodo REST API (record listing with pagination for searches) for both cohorts; `records.csv` (dataset, record_id, subject_id, fs, n_channels, duration_s, pma_weeks, ga_weeks, grade) and `events.csv` (seizure annotations per annotator, Helsinki). PMA = gestational age at birth + postnatal age at recording; records with missing fields are excluded from the clock but kept for feature QC.
2. **Pre-processing.** Common bipolar neonatal montage (Fp1-T3, T3-O1, Fp1-C3, C3-O1, Fp2-C4, C4-O2, Fp2-T4, T4-O2 plus midline pairs where available), 0.5-30 Hz band-pass, 50 Hz notch, artefact screening (amplitude > 500 uV, flat channels, electrode pops via derivative outliers); 1-h and 5-min epochs.
3. **Features (`neoclock.features_neural`).** Log absolute and relative power in delta1 (0.5-2), delta2 (2-4), theta (4-7), alpha (7-13), beta (13-30); spectral edge 90/95; spectral slope (log-log, 2-20 Hz); rEEG (range EEG, 2-s peak-to-peak) percentiles 5/50/95 and asymmetry; burst/interburst structure from a smoothed amplitude envelope (median/max/95th-percentile interburst interval, burst fraction = continuity, bursts per minute); interhemispheric synchrony (correlation of 2-s envelopes across homologous pairs); median across channels per epoch.
4. **Clock (`neoclock.clock`).** Ridge (default) and gradient-boosting regressors on epoch features aggregated per recording; subject-grouped 5-fold CV with fixed folds; delta = predicted - PMA; bias correction by regressing delta on PMA in training folds and subtracting the fit (Smith et al., 2019); ICC(1) for within-infant reliability; MAE by PMA bin.
5. **Normative centiles (`neoclock.normative`).** For each feature, a Gaussian model with polynomial mean and log-SD in PMA (fitted on seizure-free Helsinki recordings) gives centiles and z-scores; a multivariate deviation score (mean |z| across features) flags out-of-distribution recordings.
6. **Outcomes (`neoclock.outcomes`).** Jonckheere-Terpstra trend of delta across HIE grades with permutation p; partial Spearman correlation of delta with seizure burden adjusting for PMA; AUROC of delta for grade >= 2; consensus seizure burden from the three annotators.
7. **State analysis.** Sleep-state proxy from discontinuity/rEEG (quiet sleep = more discontinuous, higher rEEG asymmetry); delta computed per state; mixed model with infant random intercept.

## Evaluation & statistics

- Primary endpoint: subject-grouped CV MAE (weeks) and Pearson r between predicted and true PMA; bias slope of delta vs PMA before and after correction.
- Uncertainty: subject-level bootstrap (2,000 draws) for MAE and for the delta-outcome associations; paired bootstrap for model comparisons.
- Leakage prevention: all recordings of an infant in the same fold; normative models and bias-correction coefficients fitted on training folds only; seizure epochs removed before feature extraction for the clock; annotator consensus computed once and frozen.
- Nulls: PMA permutation across infants (clock MAE at chance); outcome permutation for H3/H4; epoch shuffling within infant for the ICC null.
- Multiple comparisons: Holm-Bonferroni within hypothesis families; centile tables reported descriptively.
- Reporting: MAE per PMA bin (35-37, 37-39, 39-41, 41-43, 43-45) to make the narrow-range limitation explicit; calibration plot of predicted vs true PMA.

## Publishable angle

- **Headline result.** "An open neonatal EEG clock reaches ~1.5-week accuracy in the term range; its raw deviation is biased by X weeks per week of PMA, and after correction it separates HIE grades and tracks seizure burden with effect sizes Y and Z, while remaining stable within infants across epochs and states." Delivered with released weights, normative centile charts and a command-line tool for EDF input.
- **Venues.** *Clinical Neurophysiology* (clinical neurophysiology audience); *Pediatric Research* or *Archives of Disease in Childhood - Fetal and Neonatal Edition* (neonatology); *NeuroImage* or *Journal of Neural Engineering* (methods, normative modelling); *Scientific Data* for the centile tables and manifests.
- **Follow-ups.** Preterm extension through data-sharing agreements; longitudinal FBA trajectories as outcome predictors; harmonization with the NeoNaid quality-control framework; use of the clock as an age covariate in neonatal seizure detectors (see `cross-dataset-seizure-generalization`).

## Risks, confounds & mitigations

- **Narrow PMA range (35-45 weeks) in open data.** Mitigation: report per-bin MAE and calibration; frame as a term-range clock; list preterm cohorts and agreements needed.
- **Cohort pathology.** Both open cohorts are clinical (seizures, HIE); "normal" training data are seizure-free epochs from the less-affected infants. Mitigation: train on seizure-free epochs with normal/mildly abnormal background where grades exist; sensitivity analysis excluding infants with any seizure; state the limitation.
- **Medication (phenobarbital, sedatives) suppresses the EEG** and mimics immaturity. Mitigation: use clinical information where available as covariate; sensitivity analysis.
- **Montage and reference differences between cohorts.** Mitigation: common bipolar montage; per-record robust scaling; montage-subsampling experiment (H6).
- **PMA uncertainty** (gestational-age dating error ~1 week). Mitigation: treat MAE below ~1 week as indistinguishable from dating noise; use rank-based outcome statistics.
- **Small n for HIE grades 3-4.** Mitigation: trend tests rather than pairwise comparisons; bootstrap CIs; report counts.

## Milestones

- [ ] Download Helsinki and HIE datasets (`scripts/download_data.py`); build `records.csv` with PMA and grades.
- [ ] Implement montage harmonization and artefact screening; QC report per recording.
- [ ] Extract features for 5-min and 1-h epochs; sanity plots of interburst interval and continuity vs PMA.
- [ ] Freeze subject folds; fit ridge and GBM clocks; MAE by PMA bin (H1).
- [ ] Delta bias analysis and correction (H2).
- [ ] Normative centiles and multivariate deviation score.
- [ ] HIE-grade and seizure-burden associations (H3, H4).
- [ ] Within-infant ICC and sleep-state analysis (H5).
- [ ] Cross-cohort transfer and montage subsampling (H6).
- [ ] Release weights, centile tables and CLI; write paper.

## Ethics / data-use notes

- Both cohorts are de-identified and CC-BY licensed on Zenodo; cite Stevenson et al., 2019 and O'Toole et al., 2023 and the Zenodo records.
- Clinical fields (gestational age, grades) are sensitive; publish only aggregated statistics and per-record derived scores without re-identifying combinations.
- Do not send raw EEG to third-party LLM/ML APIs; keep processing local.
- Never commit data; `data/` and `*.edf` are git-ignored. Manifests with record ids, PMA (weeks) and grades already public in the source datasets may be committed.
- Any preterm extension under a data-sharing agreement must keep those data outside this repository.
