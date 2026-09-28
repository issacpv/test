# cuffless-bp-pregnancy

**Does the pulse-transit-time / PPG-to-blood-pressure relationship shift in pregnancy and in hypertensive disorders of pregnancy (HDP)? A physiologically grounded transportability audit of cuffless BP models on the only open waveform records of pregnant patients (MIMIC-III/IV waveforms linked to obstetric admissions), against VitalDB and PulseDB reference populations.**

## Status / difficulty / timeline / compute

- Status: design + starter code (beat detection, PAT/PTT extraction, cohort flags from ICD codes, mixed-model calibration analysis, ISO 81060-2 / IEEE 1708 metrics, Bramwell-Hill shift simulator). No data is shipped.
- Difficulty: MSc thesis to early-PhD. The hard parts are (i) the small pregnant cohort and the resulting need for beat-level mixed models rather than subject-level means, and (ii) careful signal quality control.
- Timeline: 6-9 months (1 month credentialing + record inventory, 2 months signal pipeline, 2 months modelling/statistics, 1-2 months writing). A feasibility stop/go is built in at month 2 (see Milestones).
- Compute: one workstation; selected waveform records only (download record-by-record with `wfdb`, tens of GB at most). No GPU needed for the primary analysis; a GPU is optional for retraining PPG deep models on PulseDB.

## Background

Cuffless BP estimation from pulse arrival time (PAT, ECG R-peak to PPG foot), pulse transit time (PTT, between two arterial sites) and PPG morphology rests on the Bramwell-Hill / Moens-Korteweg relation: pulse wave velocity rises with arterial stiffness, and stiffness rises with distending pressure (Bramwell & Hill, 1922 Proc R Soc B; Mukkamala et al., 2015 IEEE TBME). Every PTT-based estimator therefore embeds an assumption about the subject's arterial compliance, and every calibration-free PPG model embeds the compliance distribution of its training population.

Pregnancy changes exactly those quantities: plasma volume rises 40-50%, cardiac output 30-50%, heart rate 15-20%, systemic vascular resistance falls, and arterial compliance rises through mid-gestation (Sanghavi & Rutherford, 2014 Circulation). Preeclampsia reverses part of this: arterial stiffness (carotid-femoral PWV, augmentation index) is elevated compared with normotensive pregnancy (Hausvater et al., 2012 J Hypertens, meta-analysis). HDP affect 5-10% of pregnancies and remain a leading cause of maternal mortality, so cuffless monitoring in pregnancy is an obvious clinical target; a 2025 review of wearable BP in pregnancy (Arch Gynecol Obstet, doi:10.1007/s00404-025-08301-2) and a 2025 case report of a wrist cuffless device against 24-h ABPM in pregnancy (Blood Pressure, doi:10.1080/08037051.2025.2563615) both conclude that cuffless devices have not been formally validated in pregnancy, and the case report found nighttime disagreement.

Consumer and research cuffless devices are validated (when they are validated at all) on general adult populations under the ESH 2023 recommendations (Stergiou et al., 2023 J Hypertens) and IEEE 1708-2014/1708a-2019. Pregnancy is precisely the physiological state in which those validations are least likely to transfer, and the state in which BP thresholds (140/90, 160/110) trigger high-stakes decisions.

## The research gap

What has been done (2023-2026):

- Large PPG-BP benchmarks now exist: PulseDB (Wang et al., 2023 Front Digit Health; ~5.2M 10-s segments from MIMIC-III matched waveforms and VitalDB with subject identifiers), a 2026 benchmarking paper of PPG-based cuffless methods (arXiv:2602.04725) and a 2026 change-point-aware evaluation / re-calibration study (arXiv:2608.18639). All treat the population as homogeneous adults; none stratify by pregnancy, and MIMIC-derived benchmarks silently include a small number of obstetric ICU patients without labelling them.
- Aurora-BP (Mukkamala et al., 2023 Hypertension) showed that PTT/PPG features explain little between-subject BP variance without calibration and that calibration drifts; the cohort was non-pregnant adults.
- Cuffless devices in pregnancy: one case report (Blood Pressure, 2025) and narrative reviews; no open-data analysis of whether PAT-BP calibration differs in pregnancy or HDP.
- Prediction of spinal-anaesthesia hypotension at caesarean section from pre-induction PPG/ECG is an active area (registered trials NCT06158542, NCT06847737; neural-network work in BMC Anesthesiol 2020), but it is closed-data and does not estimate BP itself.
- Physiological studies of PWV in pregnancy use tonometry / cf-PWV, not PAT-from-wearables, and do not connect to cuffless estimator error.

What is specifically missing (our angle):

1. A **quantified pregnancy shift** of the PAT-SBP calibration (intercept and slope) versus matched non-pregnant women, from beat-level invasive ABP or intermittent NIBP references, on open data.
2. A **within-pregnancy HDP contrast**: does the slope steepen (stiffer arteries) in preeclampsia / eclampsia / HELLP relative to normotensive pregnancy?
3. A **transportability audit** of general-population models (PulseDB-trained PPG models; VitalDB-trained PAT models) evaluated on pregnant records under ISO 81060-2 and IEEE 1708 criteria, with the error decomposed into bias (intercept), slope and morphology components, and with the question "does one-point calibration fix it?".
4. A **mechanistic explanation**: a Bramwell-Hill / exponential-elastance arterial model with pregnancy-specific parameters that predicts the direction and approximate size of the shifts in 1-2, tested against the empirical estimates.
5. A **pre-ejection-period (PEP) confound estimate**: PAT includes PEP; pregnancy increases contractility and heart rate. In ABP-tier records, R-to-ABP-foot time (proximal PAT) and ABP-foot-to-PPG-foot (true distal PTT) can be separated.

Related project in this repo: `cuffless-bp-mimic-waveform` (general-population cuffless BP on MIMIC waveforms). This project is the pregnancy-shift arm and is self-contained; it does not import code from it.

## Research questions / hypotheses

A note on functional form, because it decides what each hypothesis tests. With exponential elastance E(P) = E0 exp(alpha P) and Moens-Korteweg PWV, ln(PTT) = c - alpha P / 2, so d SBP / d ln(PTT) = -2/alpha depends only on alpha (the pressure-sensitivity of stiffness), whereas d SBP / d PTT = -2/(alpha PTT) is steeper when baseline stiffness E0 is higher (shorter PTT). Baseline stiffness therefore shows up as an intercept shift (and a linear-slope change), not as a log-slope change. The primary calibration model is linear in PAT (mmHg per ms); the log model is secondary and isolates alpha.

1. **RQ1 (calibration shift).** In beat-level linear mixed models SBP ~ PAT x group + HR with random subject intercepts and slopes, H1: the group intercept for normotensive pregnancy is positive at the population median PAT (longer PAT at equal SBP, i.e. higher compliance / lower E0) with an effect of >= 5 mmHg; the effect is absent in matched non-pregnant women of the same age.
2. **RQ2 (HDP slope).** H2: the linear |slope| d(SBP)/d(PAT) in mmHg/ms is larger in HDP (preeclampsia/eclampsia/HELLP, ICD-9 642.4-642.7; ICD-10 O11, O14, O15) than in normotensive pregnancy, by >= 20% (bootstrap CI excluding 0), and the HDP intercept shift at the reference PAT is negative (shorter PAT at equal SBP). Exploratory: whether the log-slope (i.e. alpha) also differs.
3. **RQ3 (transportability).** H3: PulseDB-trained calibration-free PPG models and VitalDB-trained PAT models fail ISO 81060-2 criterion 1 (|ME| <= 5, SD <= 8 mmHg) on the pregnant cohort with a predominantly bias-type error; H3b: one-point calibration (first 60 s of each record) restores criterion 1 in normotensive pregnancy but not in HDP, where slope error remains.
4. **RQ4 (drift).** H4: within-record calibration drift (IEEE 1708 style, error vs time since calibration) is faster in HDP records (labile BP, magnesium sulfate, antihypertensive boluses) than in controls.
5. **RQ5 (mechanism).** H5: a Bramwell-Hill model with E0 reduced by 20-30% (pregnancy) and E0/alpha increased (preeclampsia) reproduces the sign and order of magnitude of the intercept and slope shifts estimated in RQ1-RQ2.
6. **RQ6 (PEP).** H6: the proximal component (R-to-ABP-foot) accounts for >= 50% of the pregnancy PAT offset in ABP-tier records.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-III Waveform Database Matched Subset v1.0 | ECG II, PLETH, ABP waveforms (125 Hz) and numerics (NIBP, HR) for ICU patients linkable to MIMIC-III clinical | 10,282 patients, 22,317 records | Open (no credentialing) | https://physionet.org/content/mimic3wdb-matched/1.0/ |
| MIMIC-III Clinical Database v1.4 | ICD-9 pregnancy / HDP codes, demographics, care unit, medications (MgSO4, labetalol, hydralazine, nifedipine) for labelling waveform records | 46,520 patients | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciii/1.4/ |
| MIMIC-IV Waveform Database v0.1.0 (larger release announced) | Same as above for MIMIC-IV era | 200 records in v0.1.0 | PhysioNet credentialed | https://physionet.org/content/mimic4wdb/0.1.0/ |
| MIMIC-IV v3.1 (hosp module) | ICD-10 O-codes, obstetric care units (Labor & Delivery, Obstetrics Antepartum/Postpartum), demographics | ~546k admissions | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| VitalDB v1.0 | Non-pregnant anaesthetised reference population (gynaecology / general surgery women 18-45), ECG/PLETH/ART at 500/100 Hz; spinal-anaesthesia cases for rapid BP change | 6,388 cases | Open dataset (web API / `vitaldb` package; accept terms) | https://vitaldb.net/dataset/ |
| PulseDB | Cleaned 10-s ECG/PPG/ABP segments with subject ids from MIMIC-III matched + VitalDB, for training general-population models and for subject-overlap exclusion | ~5.2M segments | Open (GitHub release; MATLAB files) | https://github.com/pulselabteam/PulseDB |

Notes. VitalDB's listed departments are general surgery, thoracic surgery, urology and gynaecology; obstetric (caesarean) cases are not an advertised category. The downloader filters `opname` for "cesarean"/"C-sec" and reports the count so this can be settled empirically before relying on it. If no obstetric cases exist, VitalDB serves only as the anaesthetised non-pregnant reference.

## Methods

Pipeline (each step maps to a module in `src/cuffless_preg/`):

1. **Cohort labelling** (`cohort.py`). From `DIAGNOSES_ICD` (MIMIC-III) and `diagnoses_icd` (MIMIC-IV): pregnant = any ICD-9 640-679, V22, V23, V27 or ICD-10 O00-O9A, Z33, Z34, Z37 code on the admission; HDP = ICD-9 642.x / ICD-10 O10-O16; severe HDP = 642.5, 642.6, 642.7 / O14.1, O14.2, O15, O11. Controls: female, age 18-45, no pregnancy code on any admission, matched 1:3 on 5-year age band and first ICU care unit (exact) with a random draw. MIMIC-IV adds care-unit evidence (`transfers.careunit` containing "Labor & Delivery" or "Obstetrics").
2. **Record inventory and tiering** (`cohort.py`, `scripts/download_data.py`). Parse the matched-subset `RECORDS` index and per-record headers with `wfdb`; Tier A = ECG II + PLETH + ABP, >= 10 min; Tier B = ECG II + PLETH + NIBP numerics (>= 5 NIBP readings). Tier B is the wearable-like scenario and the larger cohort.
3. **Signal processing** (`signals.py`). Pan-Tompkins-style R-peak detection; PPG foot (minimum before systolic peak) and maximum-slope fiducials; ABP systolic/diastolic beats; PAT = R to PPG foot (window 0.1-0.6 s); proximal PAT = R to ABP foot; per-beat template-correlation SQI (drop beats with r < 0.8, drop 30-s windows with > 20% bad beats); PPG morphology (amplitude, rise time, width at half height).
4. **Beat table** (`signals.py`). Aligned beat-level table (record, time, PAT, proximal PAT, HR, SBP, DBP, MAP, morphology, SQI). Tier B: PAT averaged over the 30 s preceding each NIBP reading.
5. **Calibration models** (`ptt_model.py`). Per-subject OLS SBP = a + b PAT (primary, mmHg/ms) and SBP = a + b ln(PAT) (secondary, isolates alpha); population linear mixed model with group (control / pregnant normotensive / HDP) x PAT interaction, HR covariate, random intercept and slope per subject (statsmodels `MixedLM`), PAT centred at the control median so intercepts are interpretable.
6. **Transportability audit** (`ptt_model.py`, `metrics.py`). (a) Train PAT-based population models on VitalDB women 18-45 and on PulseDB (excluding any MIMIC-III subject in our cohorts); (b) evaluate calibration-free and one-point-calibrated predictions on Tier A/B pregnant and control records; (c) decompose error into intercept, slope and residual components by refitting a 2-parameter correction per subject and attributing variance.
7. **Mechanistic model** (`ptt_model.py`). Exponential elastance E(P) = E0 exp(alpha P) (Hughes et al., 1979) with Moens-Korteweg PWV = sqrt(E h / (2 rho r)); PAT = PEP + L / PWV. Simulate PAT-SBP curves for control, pregnancy (E0 x 0.7, r x 1.1, PEP x 0.85) and preeclampsia (E0 x 1.4, PEP x 0.9; alpha as a sensitivity parameter) and compare the implied intercept, linear slope and log-slope with the RQ1-RQ2 estimates. The simulator makes the prediction explicit: pregnancy moves the intercept up, preeclampsia moves it down and steepens the linear slope, and only a change in alpha would move the log-slope.
8. **Drift analysis** (`metrics.py`). Error vs minutes since one-point calibration, binned; slope of |error| vs time per group.

Tools: `wfdb`, `numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn`, `vitaldb` (VitalDB API), `matplotlib`.

## Evaluation & statistics

- Unit of inference: subject (record) for group contrasts, beat for within-subject calibration. All CIs by cluster bootstrap over subjects (1000 resamples).
- Mixed models: `MixedLM` with random intercept + random slope on ln(PAT); report fixed-effect group x ln(PAT) coefficients with Wald CIs and a likelihood-ratio test against the no-interaction model.
- Transportability metrics (`metrics.py`): ME, SD, MAE, ISO 81060-2 criterion 1 and criterion 2 (subject-level SD limit interpolated from the ISO table), BHS grade (% within 5/10/15 mmHg), per-subject calibration drift slope (IEEE 1708 style).
- Leakage prevention: subject-level exclusion of any PulseDB/MIMIC-III subject in the pregnant or control cohort from model training; calibration beats (first 60 s) never used in evaluation; no hyper-parameter tuning on pregnant records.
- Multiple comparisons: three pre-specified primary hypotheses (H1, H2, H3) with Holm correction; RQ4-RQ6 are exploratory.
- Null models: (i) permutation of group labels across subjects (2000 permutations) for the interaction terms; (ii) within-subject time-shuffled PAT (destroys beat-level pairing, preserves subject means) to show that the calibration slope is a beat-level phenomenon; (iii) negative-control outcome: DBP-from-PAT slope should shift less than SBP slope (PAT tracks SBP/MAP more than DBP), which is a sanity check rather than a hypothesis.
- Sensitivity: exclude records during vasoactive infusions (MIMIC `INPUTEVENTS` / `inputevents` for norepinephrine, phenylephrine, labetalol, hydralazine, MgSO4) and repeat.

## Publishable angle

Headline result: "PAT-based and PPG-based cuffless BP models calibrated on general adult populations are biased by X mmHg in pregnancy and mis-scaled in preeclampsia; a one-point calibration removes the bias in normotensive pregnancy but not the slope error in HDP, and a Bramwell-Hill model with pregnancy-specific compliance explains the pattern." Even a small-n result is publishable because it is the first open-data quantification and it comes with a mechanistic model and a released beat-level pipeline.

Target venues: IEEE Journal of Biomedical and Health Informatics; Physiological Measurement; Pregnancy Hypertension (Int Soc Study of Hypertension in Pregnancy); Blood Pressure Monitoring; IEEE EMBC (early results).

Follow-ups: (a) apply the same pipeline to the larger MIMIC-IV Waveform release when published; (b) prospective validation with an obstetric partner site under ISO 81060-2 pregnancy protocol; (c) extend to postpartum (first 72 h) where HDP-related strokes cluster.

## Risks, confounds & mitigations

- **Small pregnant cohort.** ICU obstetric admissions are rare (< 1% of ICU stays) and only a fraction have waveforms. Mitigation: Tier B (NIBP numerics) roughly triples eligible records; beat-level mixed models; MIMIC-III and MIMIC-IV combined; stop/go rule at month 2 (>= 30 Tier A+B pregnant records with >= 30 min clean data, else the paper is re-scoped to a methods + feasibility + mechanistic-model paper).
- **ICU obstetric patients are not typical pregnant women** (severe HDP, haemorrhage, sepsis). Mitigation: state this explicitly; the HDP contrast is within the ICU obstetric population; report indication for ICU admission from diagnoses.
- **Vasoactive drugs and MgSO4 alter vascular tone independently of BP.** Mitigation: sensitivity analysis excluding infusion periods; drug exposure as covariate.
- **Arterial-line damping and PPG site (finger vs ear).** Mitigation: SQI; ABP waveform quality index (damped-waveform check via dP/dt max and dicrotic-notch presence); PPG site not always known, so include record as random effect.
- **PEP confound in PAT.** Mitigation: RQ6 separates proximal and distal components in Tier A; interpret PAT results as "wearable-realistic".
- **Subject overlap between PulseDB and our cohort.** Mitigation: explicit subject-id exclusion using PulseDB's subject fields.
- **Date shifting and record time alignment.** Waveform and clinical timestamps are shifted consistently within subject in the matched subset; we still verify alignment by cross-correlating waveform-derived HR with charted HR.
- **VitalDB obstetric cases may not exist.** Mitigation: VitalDB is used as an anaesthetised non-pregnant reference only, unless the `opname` filter finds caesarean cases.

## Milestones

- [ ] PhysioNet credentialing (CITI) for MIMIC-III v1.4, MIMIC-IV v3.1, MIMIC-IV Waveform; accept VitalDB terms.
- [ ] Record inventory: parse `RECORDS`/headers of the MIMIC-III matched subset; join to MIMIC-III `DIAGNOSES_ICD`; produce tiered counts for pregnant / HDP / control (stop/go).
- [ ] Signal pipeline validated on 20 records with manual beat checks (>= 95% beat agreement).
- [ ] Beat tables for all eligible records; SQI summary.
- [ ] RQ1-RQ2 mixed models with cluster bootstrap and permutation nulls.
- [ ] Train reference models on VitalDB women and PulseDB (with overlap exclusion); RQ3 audit with ISO/BHS/IEEE metrics; RQ4 drift.
- [ ] Bramwell-Hill simulation (RQ5) and PEP decomposition (RQ6).
- [ ] Sensitivity analyses (drugs, damping, Tier A vs B).
- [ ] Manuscript + code release (pipeline, no data).

## Ethics / data-use notes

- MIMIC-III, MIMIC-IV and the MIMIC-IV Waveform Database are PhysioNet credentialed resources: complete CITI training, sign the DUA, and never redistribute data. The MIMIC-III Waveform Matched Subset is open but becomes re-identifiable in combination with clinical data, so treat derived beat tables as restricted.
- Per PhysioNet's responsible-use policy, credentialed data must not be sent to third-party LLM or cloud APIs; all processing here is local.
- Credentials are read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD` environment variables; never hard-code or commit them. `data/` is git-ignored.
- Obstetric ICU patients are a vulnerable, small group; report counts >= 11 per cell and avoid publishing subject-level trajectories that could be re-identifying.
