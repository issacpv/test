# EEG depth-of-anesthesia models across datasets and agents: replacing BIS with pharmacological and behavioural targets to test propofol-to-sevoflurane transfer

**One-sentence pitch.** Most "cross-dataset" depth-of-anaesthesia (DoA) papers train EEG models to reproduce the Bispectral Index (BIS), a proprietary, smoothed and agent-dependent number, so what they measure is BIS reproduction, not anaesthetic depth; this project uses VitalDB's raw frontal EEG together with its TCI propofol effect-site concentrations and end-tidal sevoflurane, plus the open Cambridge propofol-sedation volunteer set with plasma levels and behavioural responsiveness, to ask whether EEG models transfer between agents and datasets when the target is pharmacological (Ce / age-adjusted MAC fraction) or behavioural (responsiveness, induction/emergence events) instead of BIS, and whether six interpretable spectral features beat deep models on that transfer.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis (feature models, Pk evaluation, agent transfer on VitalDB, ~6-8 months) to PhD chapter (deep models, behavioural external validation, age analysis, ~12 months).
- Compute: VitalDB frontal EEG is 2 channels at 128 Hz; feature extraction for ~2,000 general-anaesthesia cases runs on CPU in hours. One 16 GB GPU for the optional CNN. Storage ~50 GB (EEG waveforms + numeric tracks) plus the Cambridge set (~10 GB).
- Related projects: `cuffless-bp-mimic-waveform` (also uses VitalDB, for blood pressure) and `icu-model-transportability` (transportability framing). No code is shared.

## Background

Anaesthetic-induced unconsciousness has well-characterized frontal EEG signatures: with propofol, loss of consciousness coincides with the appearance of coherent frontal alpha (8-12 Hz) and large slow-delta oscillations (Purdon et al., 2013, *PNAS*); sevoflurane produces a similar alpha-delta pattern with additional theta (Akeju et al., 2014, *Anesthesiology*); ketamine and dexmedetomidine look different (Purdon et al., 2015, *Anesthesiology*, clinical EEG for anaesthesiologists). These signatures are strongly age-dependent: alpha power and coherence fall with age under both propofol and sevoflurane (Purdon et al., 2015, *Br. J. Anaesth.*). Commercial monitors compress the EEG into an index (BIS: Rampil, 1998, *Anesthesiology*) that is smoothed over tens of seconds, proprietary, and known to behave differently across agents.

VitalDB (Lee et al., 2022, *Sci. Data*) is an open database of 6,388 surgical cases with raw BIS-monitor EEG waveforms (`BIS/EEG1_WAV`, `BIS/EEG2_WAV`, 128 Hz), BIS-derived numerics (`BIS/BIS`, `BIS/SEF`, `BIS/SR`, `BIS/EMG`, `BIS/SQI`), target-controlled-infusion propofol/remifentanil effect-site concentrations (`Orchestra/PPF20_CE`, `Orchestra/RFTN20_CE`), anaesthesia-machine gas concentrations (`Primus/EXP_SEVO`, `Primus/EXP_DES`, `Primus/MAC`), and clinical timestamps (anaesthesia/operation start and end) with age, sex and ASA class. That combination - raw EEG, drug exposure and clinical events in the same cases - is what makes it possible to evaluate DoA models against something other than BIS. The Cambridge propofol-sedation dataset (Chennu et al., 2016, *PLoS Comput. Biol.*; 20 healthy volunteers, 91-channel EEG at baseline / mild / moderate sedation / recovery with measured plasma propofol and a behavioural responsiveness task) adds a behaviourally labelled, hardware-different external test.

## The research gap

**What has been done.**

- BIS-regression models on VitalDB and private sets: a 2025 ResNet-SE study reports RMSE ~5.5 on VitalDB and ~6.6 on an NTUH set; an LSTM-with-attention model reports RMSE 5.07 on VitalDB (5,566 cases) vs 8.61 on a 90-case AMC set; a multitask time-frequency deep model (*Diagnostics*, 2026, 16(12):1937) and an EEG-processor design (2024, *Brain Informatics*) also regress BIS; "Predicting depth of anaesthesia from single-channel EEG" (*Sci. Rep.*, 2026) and AnesNet-style future-BIS prediction from multimodal VitalDB signals continue the BIS-as-target line; "Anesthesia depth prediction from drug infusion history" (*BMC Med. Inform. Decis. Mak.*, 2025) predicts BIS from dosing alone, which shows how much of BIS is explained by pharmacology without any EEG.
- Physiology: aperiodic (1/f) slope tracks arousal and propofol anaesthesia (Lendner et al., 2020, *eLife*); spectral parameterization (Donoghue et al., 2020, *Nat. Neurosci.*); burst-suppression quantification (Rampil, 1998); age-adjusted MAC (Nickalls & Mapleson, 2003, *Br. J. Anaesth.*); MAC-awake ~0.3 MAC (Eger, 2001, *Anesth. Analg.*); propofol effect-site modelling (Schnider et al., 1998/1999, *Anesthesiology*).
- Evaluation: prediction probability Pk (Smith, Dutton & Smith, 1996, *Anesthesiology*) is the accepted DoA-indicator metric but is rarely reported by deep-learning papers, which report RMSE against BIS.

**What is specifically missing.**

1. **Targets other than BIS on open data.** No open-data study evaluates EEG models against pharmacological exposure (propofol Ce, age-adjusted MAC fraction) or clinical events (induction/emergence timing) as the reference, although both are available in VitalDB.
2. **Agent transfer with the confound removed.** Propofol -> sevoflurane transfer has only been assessed through BIS, whose own agent-dependence is inseparable from the EEG shift. A pharmacological target makes the two agents comparable on a common "fraction of hypnotic effect" scale.
3. **Interpretable vs deep on transfer.** Six physiologically motivated features (frontal alpha power, delta power, aperiodic exponent, SEF95, burst-suppression ratio, alpha-delta ratio) have never been benchmarked against CNNs for cross-agent and cross-dataset transfer with Pk.
4. **Age-stratified transfer.** Despite the known age-dependence of anaesthetic EEG, no cross-dataset DoA paper reports performance by age decade or age-aware training.
5. **Behavioural external validation.** No VitalDB-trained model has been tested on a behaviourally labelled, hardware-different dataset (Cambridge volunteers) - a genuine out-of-distribution test for "depth".

## Research questions / hypotheses

1. **H1 (BIS ceiling and agent bias).** A BIS-reproduction model trained on propofol TIVA cases has RMSE <= 6 on held-out propofol cases and >= 2 points worse on sevoflurane cases; >= 50% of the increase is explained by the different BIS-vs-exposure relationship of the two agents (fit on BIS numerics alone, no EEG), not by EEG feature shift.
2. **H2 (pharmacological target).** Models predicting normalized exposure (propofol Ce / 3 ug mL^-1; sevoflurane age-adjusted MAC fraction) show a cross-agent Pk drop < 0.05, versus > 0.10 for BIS-reproduction models.
3. **H3 (interpretable features).** The six-feature model reaches Pk within 0.03 of a CNN within-agent and a higher Pk than the CNN cross-agent and on the Cambridge set.
4. **H4 (age).** Pk and calibration slope decline with age decade for all models; the cross-agent gap is largest above 65 y; adding age as an input closes >= 50% of that gap.
5. **H5 (events).** EEG-derived depth detects the induction (LOC) and emergence (ROC) transitions with smaller absolute timing error against clinical timestamps than BIS thresholds (BIS < 60 / > 80), particularly under sevoflurane.
6. **H6 (behavioural validation).** The propofol six-feature model, trained on VitalDB frontal EEG, separates responsive from unresponsive Cambridge epochs (frontal channels only) with AUROC > 0.85; deep models trained on VitalDB do not exceed 0.80.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| VitalDB (open dataset) | General-anaesthesia cases with `BIS/EEG1_WAV`, `BIS/EEG2_WAV` (128 Hz), BIS numerics, `Orchestra/PPF20_CE` (propofol TCI), `Orchestra/RFTN20_CE`, `Primus/EXP_SEVO`/`EXP_DES`/`MAC`, `Solar8000/*` vitals; clinical table (age, sex, ASA, ane_type, anestart/aneend, opstart/opend, drugs) | 6,388 cases; EEG subset ~50 GB | Open (free; REST API `api.vitaldb.net` and the `vitaldb` Python package; web downloader needs site registration and acceptance of the data-use terms) | https://vitaldb.net/dataset/ |
| Cambridge propofol sedation EEG (Chennu et al., 2016) | 20 healthy volunteers, 91-channel EEG, four states (baseline, mild, moderate sedation, recovery), plasma propofol, behavioural responsiveness; frontal channels used for external validation | ~10 GB | Open (University of Cambridge Apollo repository; CC licence stated on the record) | https://www.repository.cam.ac.uk/ (search "Chennu propofol sedation EEG") |
| Optional, to verify before use | An EEG set of 44 surgical patients (27 propofol, 17 sevoflurane) described in a 2025-2026 paper on GABAergic anaesthetic-induced unconsciousness and recovery; hosting and licence were not confirmed at scaffold time | ~ | unverified | see the paper's Data Availability |
| Optional | OpenNeuro EEG datasets tagged anaesthesia/propofol/sevoflurane, if any appear (none confirmed at scaffold time) | ~ | Open | https://openneuro.org |

Case selection (VitalDB): `ane_type == "General"`, EEG waveform present with `BIS/SQI` >= 50 for >= 60% of the maintenance phase, age >= 18; propofol arm = cases with `Orchestra/PPF20_CE` and no volatile; sevoflurane arm = cases with `Primus/EXP_SEVO` > 0.2 vol% for >= 30 min and no propofol infusion after induction. Mixed and ketamine/N2O cases are excluded from the primary analysis and used as a sensitivity set.

## Methods

1. **Ingestion (`doa_xfer.vitaldb_io`).** REST API helpers with local caching; case/track selection; numeric tracks aligned to a 1-s grid by last-observation-carried-forward; EEG waveform cut into 4-s epochs (50% overlap) with artefact flags (amplitude > 500 uV, flat line, `BIS/SQI` < 50).
2. **Features (`doa_xfer.eeg_features`).** Welch PSD (0.5-47 Hz): log power in slow (0.1-1), delta (1-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-47) bands, relative powers, SEF95, spectral entropy, aperiodic exponent and offset via a robust log-log fit with iterative peak exclusion (FOOOF-lite), alpha peak power above the aperiodic fit, alpha/delta ratio, burst-suppression ratio (fraction of epoch with |x| < 5 uV for >= 0.5 s). The "six-feature" model uses frontal alpha, delta, aperiodic exponent, SEF95, BSR and alpha/delta.
3. **Targets (`doa_xfer.targets`).** (a) BIS with an explicit lag alignment (BIS is smoothed; the lag is estimated per case by cross-correlation with SEF95 and fixed at the population median); (b) normalized exposure: propofol Ce / 3.0 ug mL^-1 (approximate Ce50 for loss of response; treated as a scale, not a threshold) and sevoflurane end-tidal / age-adjusted MAC (Nickalls & Mapleson: MAC(age) = MAC40 x 10^(-0.00269 (age - 40)), MAC40 = 1.80 vol% for sevoflurane); (c) phase labels from clinical timestamps (pre-induction, induction, maintenance, emergence, post-emergence); (d) Cambridge: responsive vs unresponsive per state from the behavioural task.
4. **Models.** Ridge / gradient boosting on features (case-grouped CV), the six-feature ridge model, and a 1-D CNN on raw epochs (torch, optional). Age-aware variants add age (and sex) as inputs.
5. **Transfer protocol (`doa_xfer.transfer_eval`).** Train on propofol arm -> test on sevoflurane arm and vice versa (no target-side tuning); train on VitalDB -> test on Cambridge frontal channels; age-stratified evaluation (decades); label-permutation null (permute case labels).
6. **Event timing.** Depth trajectory smoothed (30 s); LOC = first crossing of the maintenance median +/- 2 SD band after induction start; ROC = last crossing before aneend; compared with clinical timestamps and with BIS-threshold timing.

## Evaluation & statistics

- Primary metric: prediction probability Pk (Smith et al., 1996) between model output and the target (BIS, normalized exposure, or ordinal state), per case, summarized as median (IQR); jackknife SE per case.
- Secondary: RMSE and Lin's concordance correlation coefficient against BIS; Bland-Altman bias and limits of agreement; calibration slope of predicted vs target exposure; AUROC for responsiveness (Cambridge); absolute timing error (s) for LOC/ROC.
- Validation: case-grouped 5-fold CV within each arm; strict cross-arm and cross-dataset evaluation; feature scalers and any lag/threshold constants fitted on training cases only.
- Leakage prevention: `caseid` and `subjectid` (VitalDB has repeat patients) grouped together; Cambridge volunteers never used for any fitting.
- Comparisons: paired case-level bootstrap (2,000 draws) of Pk differences between models and between targets; Holm-Bonferroni within each hypothesis family; age effects via linear mixed model (case random intercept) of per-case Pk on age decade.
- Nulls: label permutation across cases (Pk -> 0.5); within-case circular time shift of the target (tests temporal alignment rather than case-level depth).

## Publishable angle

- **Headline result.** "EEG DoA models that transfer across agents on BIS do so because BIS is agent-biased; on a pharmacological scale, a six-feature model transfers propofol <-> sevoflurane with Pk loss < 0.05, beats CNNs out of distribution, and its age-related failure is predictable from the known decline of frontal alpha." Delivered with an open evaluation kit (Pk, targets, case lists) for VitalDB.
- **Venues.** *British Journal of Anaesthesia* or *Anesthesiology* (clinical); *Journal of Clinical Monitoring and Computing* (monitoring methods); *IEEE TBME* / *Journal of Neural Engineering* (methods); *Scientific Data* companion for the harmonized case list and targets.
- **Follow-ups.** Ketamine and dexmedetomidine cases (agent-specific signatures); remifentanil interaction surfaces (Ce propofol x Ce remifentanil); paediatric extension if an open paediatric anaesthesia EEG set appears (cf. Cornelissen et al., 2015, *eLife*, infant sevoflurane EEG); prospective validation with a clinical partner.

## Risks, confounds & mitigations

- **Exposure is not depth.** Ce and MAC fraction are inputs, not brain states; individual sensitivity varies. Mitigation: treat exposure as a scale for *relative* ordering (Pk), complement with behavioural (Cambridge) and event-timing endpoints; report all three.
- **BIS lag and smoothing.** Mitigation: explicit lag estimation; report results with and without lag correction.
- **Opioid and adjuvant effects.** Remifentanil changes EEG and reduces hypnotic requirements. Mitigation: include `RFTN20_CE` as covariate; sensitivity analysis on low-opioid cases.
- **Artefacts (electrocautery, EMG).** Mitigation: SQI and amplitude gating; BSR only on clean epochs; report the fraction excluded per case.
- **Hardware shift for Cambridge.** 91-channel lab EEG vs BIS sensor. Mitigation: use only frontal channels re-referenced to approximate the BIS montage; per-record robust scaling; treat as an out-of-distribution test.
- **VitalDB single-centre.** Mitigation: state the limitation; the Cambridge set and any verified additional set give the only true external checks.

## Milestones

- [ ] Pull VitalDB clinical and track tables; select propofol and sevoflurane arms (`scripts/download_data.py --dataset vitaldb --sample` first).
- [ ] Download the Cambridge propofol set; map channels to a frontal montage.
- [ ] Feature extraction and artefact gating for all cases; QC summary.
- [ ] Build targets: lag-aligned BIS, normalized exposure, phase labels; validate MAC adjustment against `Primus/MAC`.
- [ ] Within-arm models (features, six-feature, CNN); Pk and RMSE tables (H1).
- [ ] Cross-agent transfer on BIS vs exposure targets (H2, H3).
- [ ] Age-stratified analysis and age-aware training (H4).
- [ ] LOC/ROC timing analysis (H5).
- [ ] Cambridge behavioural validation (H6).
- [ ] Release evaluation kit; write paper.

## Ethics / data-use notes

- VitalDB is de-identified and provided under its own data-use terms; cite Lee et al., 2022 and follow the terms on redistribution (publish only case ids and derived features/labels).
- The Cambridge dataset is openly licensed; cite Chennu et al., 2016 and the repository record.
- Do not send raw EEG or clinical tables to third-party LLM/ML APIs; keep processing local.
- Never commit data; `data/` and waveform caches are git-ignored. Case lists with `caseid` and derived targets may be committed.
- Depth-of-anaesthesia outputs are research artefacts, not monitoring devices.
