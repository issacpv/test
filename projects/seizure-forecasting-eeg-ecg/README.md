# Seizure forecasting from scalp EEG + ECG on open multimodal data: does the heart add anything the brain does not already say?

**One-sentence pitch.** Build the first open-data, patient-independent *forecasting* (not detection) benchmark that pairs scalp EEG with simultaneously recorded ECG (Siena, SeizeIT2, TUSZ EKG channels), and quantify, under seizure-time surrogate nulls and time-in-warning-matched chance levels, how much pre-ictal heart-rate-variability (HRV) information adds to EEG, how much of it is circadian/vigilance confounding, and whether ECG-derived models transfer across cohorts better than montage-bound EEG models.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis (baseline benchmark, ~6-9 months) to early PhD (adding deep multimodal models and pseudo-prospective evaluation, ~12-15 months).
- Compute: HRV + spectral-EEG baselines run on CPU (SeizeIT2 is ~11,600 h but only 250 Hz, 4 EEG + 1 ECG channels; windowed features fit in a few GB). One 16-24 GB GPU for the optional CNN/transformer fusion models. Storage ~250 GB (SeizeIT2 is the bulk; TUSZ ~70 GB).
- Related project in this repository: `cross-dataset-seizure-generalization` (seizure *detection*, leakage audit, montage harmonization). This project is self-contained and reuses none of its code; the scoring problem here (forecasting horizons, time-in-warning) is different.

## Background

Seizure forecasting aims to raise a warning minutes-to-hours before a seizure. The decisive demonstration used chronic intracranial recordings (Cook et al., 2013, *Lancet Neurol.*), and the field has since moved toward non-invasive and wearable signals (Kuhlmann et al., 2018, *Nat. Rev. Neurol.*; Nasseri et al., 2021, *Sci. Rep.*, wrist-worn forecasting; Meisel et al., 2020, *Epilepsia*, wristband forecasting in a paediatric EMU). Peri-ictal autonomic changes are well documented: ictal tachycardia in the majority of focal seizures, and HRV changes reported minutes before onset (Billeci et al., 2018, *PLoS ONE*; Leal et al., 2021, *Sci. Rep.*; Jeppesen et al., 2019, *Epilepsia*, ECG-based detection with a wearable). ECG is attractive for forecasting because it is montage-free, cheap, and tolerated for weeks, which scalp EEG is not.

The methodological problem is that almost all HRV "prediction" results are (i) patient-specific, (ii) on private data with tens of seizures, (iii) evaluated with a prediction horizon of zero or a few tens of seconds, which makes them *early detection* rather than forecasting, and (iv) rarely tested against the seizure-time surrogate nulls that the EEG forecasting community adopted twenty years ago (Andrzejak et al., 2003, *Phys. Rev. E*; Winterhalder et al., 2003, *Epilepsy Behav.*; Schelter et al., 2006, *Chaos*; Snyder et al., 2008, *J. Neural Eng.*). Until 2025 there was no open dataset with hundreds of seizures and simultaneous EEG + ECG on which to do this properly.

That changed with SeizeIT2 (Bhagubai et al., 2025, *Sci. Data*; OpenNeuro ds005873): 125 patients with focal epilepsy from five European epilepsy monitoring units, ~11,600 h of behind-the-ear EEG, ECG, EMG and accelerometry/gyroscope from one wearable device (250 Hz; movement 25 Hz), 886 annotated focal seizures, BIDS-formatted and openly licensed. Together with Siena (Detti et al., 2020, *Processes*; 14 adults, full 10-20 scalp EEG + EKG at 512 Hz, 47 seizures) and the EKG channel carried by many TUH EEG Seizure Corpus records (Shah et al., 2018, *Front. Neuroinform.*), this is enough to ask the forecasting question on open data.

## The research gap

**What has been done.**

- ECG/HRV seizure detection and "prediction" with small private cohorts, mostly patient-specific: Billeci et al., 2018 (*PLoS ONE*; HRV + recurrence quantification, 7 patients); Pavei et al., 2017 (*Front. Neurol.*); Leal et al., 2021 (*Sci. Rep.*, HRV pre-ictal interval identification, EPILEPSIAE data); Jeppesen et al., 2019 (*Epilepsia*) and Vandecasteele et al., 2017 (*Sensors*) for wearable-ECG detection. A 2025 preprint reports multimodal detection with long-term ambulatory ECG (medRxiv, 10.64898/2025.12.16.25342428) and a 2025 *Sensors* paper (25(24):7687) reports ECG-based detection in real-world wearable recordings; both are detection, not forecasting.
- Wearable forecasting: Meisel et al., 2020 (*Epilepsia*), Nasseri et al., 2021 (*Sci. Rep.*), Stirling et al., 2021 (*Front. Neurol.*, cycles + wearables), Karoly et al., 2021 (*Nat. Rev. Neurol.*, multiday cycles). None uses simultaneous scalp EEG as a comparator on open data.
- EEG-only forecasting/prediction on open data: many CHB-MIT studies (reviewed in a 2026 *PMC* review "EEG-based seizure prediction approaches within clinically relevant ...", and in CG-MambaNet, arXiv 2606.08226, which introduces event-level clinical evaluation for cross-patient prediction). CHB-MIT has no usable ECG for most subjects, so these are unimodal by necessity.
- Multimodal wearable detection on SeizeIT2: "Multimodal wearable EEG, EMG and accelerometry measurements improve ..." (arXiv 2403.13066, 2024) and the SeizeIT2-based 2025 seizure detection challenge (biomedepi.github.io/seizure_detection_challenge) target *detection* with SzCORE scoring (Dan et al., 2024, *Epilepsia*). A 2026 review (arXiv 2601.05095) lists multimodal *prediction* as an open direction.

**What is specifically missing.**

1. **A joint EEG + ECG forecasting benchmark on open data with a real horizon.** No published study trains a patient-independent forecaster on hundreds of seizures with simultaneous scalp EEG and ECG and a seizure prediction horizon (SPH) of >= 5 min, then reports sensitivity at matched time-in-warning against surrogate-seizure nulls.
2. **Attribution of ECG "pre-ictal" signal to confounds.** Pre-ictal HRV differences are compatible with circadian phase, sleep/wake state, arousal before EMU seizures, and antiseizure-medication tapering. Nobody has residualized the ECG signal against time-of-day and EEG-derived vigilance and asked how much discriminability survives.
3. **Transportability of ECG vs EEG models.** ECG features are device- and montage-independent; scalp-EEG features are not (full 10-20 at 512 Hz in Siena vs 4-channel behind-the-ear at 250 Hz in SeizeIT2). The hypothesis that ECG-based forecasters lose *less* across cohorts than EEG-based ones has not been tested.
4. **Horizon curves.** Forecast skill as a function of SPH/SOP (0-60 min) for EEG, ECG and fusion, with the peri-ictal (< 1 min) autonomic changes explicitly excluded, would separate genuine forecasting from early detection.

## Research questions / hypotheses

1. **H1 (forecastable subset).** With ECG-only HRV features and patient-independent models, pre-ictal-vs-inter-ictal discriminability (AUROC at SPH 5 min / SOP 30 min) exceeds the seizure-time surrogate null (p < 0.05) in a minority of SeizeIT2 patients (pre-registered expectation: 20-40%), consistent with the "forecastable patient" literature.
2. **H2 (fusion gain).** Late fusion of EEG and ECG improves improvement-over-chance (IoC) sensitivity at matched time-in-warning by >= 0.05 over EEG-only in the forecastable subset, and by < 0.02 in the remainder.
3. **H3 (confounding).** Residualizing ECG features on time-of-day (sine/cosine of clock time) and EEG-derived vigilance (sleep-wake proxy from delta/alpha ratio) reduces ECG-only AUROC by >= 50% of its excess over 0.5.
4. **H4 (transportability).** The cross-cohort drop (Siena -> SeizeIT2 and reverse) in IoC is smaller for ECG-only than for EEG-only models (paired patient-level bootstrap).
5. **H5 (horizon decay).** IoC decays monotonically with SPH; at SPH >= 30 min neither modality is distinguishable from the surrogate null in > 80% of patients.
6. **H6 (patient-specific gain).** Fine-tuning on a patient's first seizures increases IoC more for ECG than for EEG (autonomic signatures are idiosyncratic), tested with leave-future-seizures-out evaluation.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| SeizeIT2 (OpenNeuro ds005873) | 125 focal-epilepsy patients, 5 EMUs, behind-the-ear EEG + ECG + EMG (250 Hz) + ACC/gyro (25 Hz), 886 focal seizures with onset/offset in BIDS `events.tsv`; primary corpus | ~11,600 h, ~2,850 recordings | Open (OpenNeuro, BIDS; CC license per dataset page) | https://openneuro.org/datasets/ds005873 |
| Siena Scalp EEG | 14 adults, 10-20 scalp EEG + EKG channel, 512 Hz, 47 seizures; full-montage cohort for transportability | ~128 h, ~20 GB | Open (PhysioNet, no login) | https://physionet.org/content/siena-scalp-eeg/1.0.0/ |
| TUH EEG Seizure Corpus (TUSZ v2.0.3) | Records that carry an EKG channel (label variants `EEG EKG1-REF`, `EKG-REF`; availability is per record and must be read from the EDF header); mixed ages; thousands of seizures; optional large-scale extension | ~1,600 h, ~70 GB | Free registration (TUH data-use form; rsync) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| Post-Ictal Heart Rate Oscillations in Partial Epilepsy (szdb) | 7 single-lead ECG records from 5 patients with 11 seizures and seizure times (Al-Aweel et al., 1999, *Neurology*); ECG-only sanity/external check for HRV code | small | Open (PhysioNet) | https://physionet.org/content/szdb/1.0.0/ |
| Optional: CHB-MIT | A minority of records carry an `ECG` channel; used only for an EEG-only comparison with the sibling project | ~42 GB | Open (PhysioNet) | https://physionet.org/content/chbmit/1.0.0/ |

Notes. (a) No open wrist-worn (PPG/EDA) seizure dataset with simultaneous scalp EEG was confirmed during the scaffold's literature check; SeizeIT2's wearable device supplies ECG, EMG and movement, which is the "wearable" arm of this project. (b) Both SeizeIT2 and Siena are EMU recordings: medication tapering and sleep deprivation are part of the setting and are treated as covariates, not ignored.

## Methods

1. **Ingestion (`szforecast.windows`, `scripts/download_data.py`).** SeizeIT2 is read from its BIDS layout (`sub-*/ses-*/eeg/*_eeg.edf`, `*_events.tsv`); Siena from `PNxx/*.edf` with `Seizures-list-PNxx.txt`; TUSZ from `*.csv_bi`. One `records.csv` (subject, record, start clock time, duration, fs, has_ecg, ecg_channel) and one `events.csv` (onset_s, offset_s, type) per corpus. Clock time is retained for circadian covariates.
2. **ECG -> HRV (`szforecast.ecg_hrv`).** Band-pass 5-30 Hz, derivative-square-integrate R-peak detection (Pan-Tompkins style) with refinement on the filtered signal; RR cleaning (physiological bounds 0.3-2.0 s, > 20% deviation from a 5-beat running median -> interpolated); 5-min windows, 1-min step: mean HR, SDNN, RMSSD, pNN50, LF (0.04-0.15 Hz), HF (0.15-0.4 Hz), LF/HF, total power (Lomb-Scargle on the unevenly sampled RR series), sample entropy; a signal-quality index (fraction of rejected beats) is carried as a feature and used to exclude windows with SQI > 0.2.
3. **EEG features.** Per channel, 5-s sub-windows aggregated to the same 5-min windows: log band powers (delta, theta, alpha, beta, gamma), relative powers, spectral edge 95%, line length, Hjorth mobility/complexity; for Siena the 4 channels closest to the behind-the-ear positions (T3/T4-region pairs) are used in the "matched-montage" condition and the full montage in the "full" condition.
4. **Labels (`szforecast.windows.label_windows`).** Pre-ictal = [onset - SPH - SOP, onset - SPH); ictal and a 60-min post-ictal period are excluded; a lead-seizure rule merges seizures < SOP apart; inter-ictal = windows >= 2 h from any onset/offset. Primary setting SPH = 5 min, SOP = 30 min; grid SPH in {1, 5, 15, 30} min x SOP in {15, 30, 60} min.
5. **Models (`szforecast.fusion`).** (a) Logistic regression (standardized, class-balanced) per modality; (b) late fusion: stacked logistic on out-of-fold modality probabilities; (c) early fusion: concatenation; (d) gradient boosting (sklearn HistGradientBoosting) as a nonlinear check; (e) optional: 1-D CNN on raw 250 Hz EEG+ECG (torch, not included). Patient-independent training with GroupKFold over subjects; patient-specific fine-tuning by leave-future-seizures-out.
6. **Alarm generation (`szforecast.forecast_eval.firing_power`).** Firing-power smoothing (Teixeira et al., 2011, *Comput. Methods Programs Biomed.*) over the SOP length, alarm when >= 0.5, refractory period = SOP.
7. **Nulls (`szforecast.windows.seizure_surrogates`, `forecast_eval.chance_sensitivity`).** (i) Constrained seizure-time surrogates: shuffle inter-seizure intervals and circularly shift onsets within each record (Andrzejak et al., 2003); the full pipeline is re-run on each surrogate set (>= 200 draws). (ii) Analytical unspecific random predictor: chance sensitivity 1 - exp(-FPR x SOP) at the model's own false-alarm rate (Winterhalder et al., 2003; Schelter et al., 2006), with a binomial test on the number of forecast seizures.
8. **Confound analysis (`szforecast.fusion.residualize`).** ECG features are residualized on sin/cos(clock hour), the EEG delta/alpha ratio (vigilance proxy), and accelerometer activity (SeizeIT2), then re-evaluated.

## Evaluation & statistics

- Primary endpoint: IoC = sensitivity (fraction of lead seizures with an alarm inside the SOP) minus chance sensitivity at the achieved time-in-warning, at SPH 5 / SOP 30, per patient, summarized as median and IQR across patients.
- Secondary: AUROC and Brier score of the pre-ictal probability at the window level; sensitivity at matched time-in-warning (10%, 20%); horizon curves over the SPH x SOP grid.
- Validation: patient-independent GroupKFold (5 folds, seizures balanced), thresholds and firing-power hyperparameters fixed on training folds only; cross-cohort runs train on one corpus and evaluate on the other with no target-side tuning.
- Leakage prevention: subject ids from BIDS `sub-*`, `PNxx`, TUSZ 8-digit ids; `assert_no_subject_leakage` runs before every fit; windows overlapping fold boundaries are dropped; feature scalers fit on training folds only.
- Significance: per patient, surrogate p-value = fraction of surrogate IoC >= observed (>= 200 surrogates); population-level test of "IoC > 0" via Wilcoxon signed-rank across patients; H2/H4 via paired patient-level bootstrap (2,000 draws) of the difference between modalities; Holm-Bonferroni within each hypothesis family.
- Multiple comparisons across the SPH x SOP grid: report the full grid, pre-register the primary cell, and control FDR (Benjamini-Hochberg) over the grid for exploratory claims.
- Reporting follows the seizure-forecasting reporting recommendations of the SzCORE / seizure-detection-challenge community for event definitions (lead seizures, refractory periods).

## Publishable angle

- **Headline result.** "On 886 open focal seizures with simultaneous EEG and ECG, ECG adds X IoC points to EEG forecasting at a 5-min horizon in the Y% of patients that are forecastable at all; half of the ECG signal is circadian/vigilance confounding; and ECG models transfer across cohorts with Z% of the drop seen for EEG." Delivered with fixed splits, surrogate machinery and a leaderboard others can enter.
- **Venues.** *Epilepsia* or *Brain Communications* (clinical forecasting audience); *Journal of Neural Engineering* / *IEEE JBHI* (methods); *NeurIPS Datasets & Benchmarks* or the SeizeIT2 challenge venue for the benchmark release; *Scientific Data* companion for the harmonized manifests and window labels.
- **Follow-ups.** Multiday cycle covariates (Karoly-style) once longer recordings are available; PPG-derived HRV from consumer wearables using the same pipeline; conformal prediction intervals on the forecast probability; ECG-only "first-line" screening of forecastable patients before prescribing EEG-based devices.

## Risks, confounds & mitigations

- **Early detection masquerading as forecasting.** Ictal tachycardia can precede the EEG onset mark by seconds to a minute. Mitigation: SPH >= 1 min in all analyses, primary SPH = 5 min; horizon curves reported.
- **EMU setting.** Medication reduction and sleep deprivation raise seizure likelihood and change HRV. Mitigation: day-in-EMU and time-of-day covariates; sensitivity analysis on day-1 vs later recordings.
- **Few seizures per patient** (median in SeizeIT2 is small; Siena 1-10). Mitigation: patient-independent models as the primary design; patient-specific only as H6 with leave-future-out; report per-patient seizure counts.
- **ECG quality in wearables.** Electrode contact and movement artefacts inflate HRV features. Mitigation: SQI gating; accelerometer-based motion exclusion in SeizeIT2; report the fraction of excluded windows per patient.
- **Annotation conventions.** Onset in SeizeIT2 is the EEG onset from video-EEG review; Siena lists wall-clock times; TUSZ uses term-based `csv_bi`. Mitigation: one `events.csv` schema; +/- 30 s onset tolerance sensitivity analysis.
- **Class imbalance and window autocorrelation.** Overlapping 5-min windows are not independent. Mitigation: statistics at the seizure/patient level (IoC, bootstrap over patients), never at the window level for inference.
- **Dataset availability.** OpenNeuro data are versioned; pin ds005873 version in `records.csv`.

## Milestones

- [ ] Download Siena and szdb; register for TUSZ; fetch SeizeIT2 (`scripts/download_data.py`).
- [ ] Build `records.csv` / `events.csv`; record which TUSZ/CHB-MIT records carry an ECG channel.
- [ ] Validate R-peak detection on szdb and on 20 hand-checked SeizeIT2 minutes (sensitivity/PPV of beats).
- [ ] Compute HRV and EEG features for all windows; SQI distributions per cohort.
- [ ] Freeze the label grid and patient folds (`splits/*.json`).
- [ ] Run ECG-only, EEG-only, fusion models; surrogate nulls; produce the forecastable-patient table (H1, H2).
- [ ] Confound residualization (H3); horizon curves (H5).
- [ ] Cross-cohort Siena <-> SeizeIT2 (H4); patient-specific fine-tuning (H6).
- [ ] Release benchmark repo with fixed manifests, surrogates and scorer tests; write paper.

## Ethics / data-use notes

- SeizeIT2 (OpenNeuro) and Siena/szdb (PhysioNet) are de-identified and openly licensed; cite the dataset papers and repositories. Respect the licence stated on the OpenNeuro dataset page for redistribution of derived data.
- TUSZ requires a signed TUH EEG data-use agreement; credentials live in `TUH_USERNAME` / `TUH_PASSWORD` environment variables, never in code or git.
- Do not upload raw EEG/ECG from any of these corpora to third-party LLM/ML APIs; PhysioNet's responsible-use terms and the TUH agreement restrict redistribution and third-party processing.
- Never commit data; `data/` and `*.edf` are git-ignored. Only manifests with subject ids and event times that are already public may be committed.
- Forecasting outputs are research artefacts, not medical devices; any pseudo-prospective evaluation must be labelled as such.
