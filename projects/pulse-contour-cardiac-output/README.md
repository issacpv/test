# pulse-contour-cardiac-output

**Open, reproducible pulse-contour stroke-volume algorithms validated against three independent reference standards (oesophageal Doppler in VitalDB, PA-catheter thermodilution and echo LVOT-VTI in MIMIC) across the operating room and the ICU, with damping and arrhythmia stratification.**

## Status / difficulty / timeline / compute

- Status: design + starter code (Windkessel waveform generator with known stroke volume, slope-sum beat detector, per-beat features, four classical estimators plus a fitted impedance correction, calibration, Bland-Altman with repeated measures, percentage error, four-quadrant and polar trending statistics, echo-report parser, VitalDB REST loader, time alignment). No data is shipped.
- Difficulty: MSc-level for the VitalDB arm; PhD-level for the three-reference, two-setting study with echo linkage. 6-9 months.
- Compute: one workstation; 500 Hz VitalDB waveforms for a few hundred cases fit in memory case by case; MIMIC-III waveform records are read with `wfdb` per record. A GPU is only needed if the optional deep-learning estimator (1D CNN on beats) is trained.

## Background

Arterial pulse-contour analysis estimates stroke volume (SV) from the shape of the arterial pressure pulse. The lineage runs from Liljestrand & Zander (1928; PP/(SBP+DBP)) and the systolic-area method (Kouchoukos et al., 1970) through Wesseling's corrected-impedance cZ and the three-element Windkessel "Modelflow" (Wesseling et al., J Appl Physiol 1993) to today's commercial devices (FloTrac/EV1000, PiCCO, LiDCOrapid, PRAM), whose algorithms are proprietary. Validation studies compare a device with pulmonary-artery thermodilution using Bland-Altman limits of agreement, the Critchley percentage error (< 30% for interchangeability; Critchley & Critchley, J Clin Monit Comput 1999) and trending statistics (four-quadrant and polar plots; Critchley et al., Anesth Analg 2010). Meta-analyses of uncalibrated devices consistently report percentage errors above 40% in vasoplegic or unstable patients (e.g., Slagt et al., Br J Anaesth 2014 for FloTrac), which is exactly where cardiac output matters.

Two open resources now make an independent, reproducible validation possible. VitalDB (Lee et al., Sci Data 2022; ~6,300 surgical cases) contains 500 Hz arterial waveforms with simultaneous SV from oesophageal Doppler (CardioQ; a method that does not use the arterial waveform) and from FloTrac/EV1000/Vigileo (commercial pulse contour, a comparator). MIMIC-III's Waveform Matched Subset (10,282 ICU patients, ABP at 125 Hz) links to chartevents containing thermodilution and continuous cardiac output from PA catheters and to ~45,000 echocardiography reports with LVOT diameter and VTI; MIMIC-IV-ECHO (v0.1: 7,243 studies, 4,579 patients, 2017-2019) adds DICOM studies and, in later versions, cardiologist-recorded measurements, linkable to MIMIC-IV Waveform records as that database grows beyond its first 200-record release.

## The research gap

What has been done (2019-2026):

- Deep-learning SV from the arterial waveform: a 2019 study ("Deep learning-based stroke volume estimation outperforms conventional arterial contour method in patients with hemodynamic instability", single-centre OR data, thermodilution reference) and, more recently, an 881-case OR study reporting MAE 10.8 mL/beat with percentage limits of agreement of about +-27-30%, plus a 2026 paediatric SVI model (BMC Med Inform Decis Mak). All are single-centre, OR-only, one reference standard, closed data or closed code.
- Cross-modal waveform conversion (MD-ViSCo, arXiv 2025) and PPG-to-ABP reconstruction (ArterialNet, arXiv 2024) use MIMIC waveforms but do not estimate cardiac output.
- Device validation studies remain small (tens of patients) and compare one device with one reference in one setting (cardiac surgery, liver transplant, septic shock).
- The influence of arterial-line damping and resonance on pulse-wave-analysis cardiac output was shown in a 2022 Br J Anaesth study (underdamping and resonance filters change PWA cardiac output), again in a small cohort.

What is specifically missing (our angle):

1. **One open algorithm suite, three reference standards, two care settings.** Nobody has validated the same pulse-contour estimators against oesophageal Doppler (OR), thermodilution (ICU) and echo LVOT-VTI (ICU) on open data with public code.
2. **Trend versus absolute accuracy under calibration regimes.** Uncalibrated, one-point calibrated and drift-recalibrated performance, reported with the full Critchley toolbox (percentage error, four-quadrant, polar), stratified by vasopressor changes, arrhythmia and haemodynamic instability.
3. **Waveform-quality stratification.** Damping class (from the sibling project `arterial-line-damping-detection`), sampling rate (500 Hz vs 125 Hz) and beat-quality filters as explicit moderators of error, quantified at cohort scale.
4. **Commercial comparator on the same beats.** In VitalDB the FloTrac/EV1000 SV is recorded alongside CardioQ, so the open algorithms can be compared with a commercial pulse-contour device *and* with an independent reference on identical data.
5. **Echo as a reference at scale.** Echo LVOT-VTI SV has never been used as a pulse-contour reference beyond small studies; MIMIC provides thousands of echo-waveform pairs (after time alignment).

## Research questions / hypotheses

1. **RQ1 (absolute accuracy).** H1: after one-point calibration, the Windkessel and systolic-area estimators reach a percentage error < 30% against CardioQ in haemodynamically stable VitalDB segments but > 40% in segments with vasopressor boluses or SVV > 13%; Liljestrand-Zander and Herd are worse in every stratum.
2. **RQ2 (trending).** H2: four-quadrant concordance > 90% and polar radial LoA within +-30 degrees for the Windkessel estimator against CardioQ for changes > 10%; concordance against the commercial FloTrac SV is higher (both are pulse-contour), against thermodilution lower.
3. **RQ3 (setting transfer).** H3: calibration factors fitted in the OR (VitalDB) transfer to the ICU (MIMIC-III thermodilution) with a systematic bias explained by heart-rate and MAP differences; the fitted impedance correction removes most of it.
4. **RQ4 (waveform quality).** H4: underdamped segments inflate systolic-area SV by > 15% and roughly double the percentage error; 125 Hz sampling costs < 3 percentage points versus 500 Hz.
5. **RQ5 (echo reference).** H5: echo LVOT-VTI SV agrees with the calibrated Windkessel estimate with bias < 5 mL and percentage error < 35% within +-30 min alignment windows; agreement decays with alignment gap.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| VitalDB | OR arm: `SNUADC/ART` 500 Hz ABP; `CardioQ/SV, /CO, /FTc` (oesophageal Doppler reference); `EV1000/SV`, `Vigileo/SV` (commercial pulse-contour comparator); NIBP; clinical info (`cases.csv`) | ~6,300 cases; SV references in a subset (hundreds of cases) | Free registration on vitaldb.net (CC BY-NC-SA); REST API | https://vitaldb.net/dataset/ |
| MIMIC-III Waveform Database Matched Subset v1.0 | ICU arm: ABP at 125 Hz linked to clinical data | 22,317 waveform records, 10,282 patients | PhysioNet credentialed | https://physionet.org/content/mimic3wdb-matched/1.0/ |
| MIMIC-III v1.4 clinical | `NOTEEVENTS` echo reports (LVOT diameter, VTI, EF); `CHARTEVENTS`/`D_ITEMS` thermodilution and continuous CO | ~45k echo reports; PA-catheter CO in cardiac-surgery stays | PhysioNet credentialed | https://physionet.org/content/mimiciii/1.4/ |
| MIMIC-IV Waveform Database v0.1.0 | Pilot linkage to MIMIC-IV-ECHO and chartevents; larger releases expected | 200 records, 198 patients | PhysioNet credentialed | https://physionet.org/content/mimic4wdb/0.1.0/ |
| MIMIC-IV-ECHO | Echo studies (DICOM) with, in later versions, measurement tables; linkage by `subject_id` | v0.1: 7,243 studies, 4,579 patients | PhysioNet credentialed | https://physionet.org/content/mimic-iv-echo/ |
| MIMIC-IV v3.1 `icu` | `chartevents` cardiac output items; vasopressor `inputevents` for instability strata | ~94k stays | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |

## Methods

Pipeline (each step maps to a module in `src/pulse_contour/`):

1. **Waveform ingestion.** VitalDB tracks via REST/`vitaldb` (`references.vitaldb_tracks`); MIMIC via `wfdb`. Resample VitalDB ABP to 125 Hz for the sampling-rate experiment.
2. **Beat detection and features** (`beats.py`). Slope-sum onset detection, systolic peak, dicrotic notch (second-derivative maximum), systolic area, dP/dt max, diastolic time constant tau, physiologic plausibility flag.
3. **Estimators** (`estimators.py`). Liljestrand-Zander, Herd, systolic area, two-element Windkessel (uses per-beat tau), and a fitted impedance correction (log SV/area ~ HR + MAP) as the open analogue of cZ. Optional 1D-CNN on beat sequences (PyTorch) trained on VitalDB only.
4. **Calibration regimes.** Uncalibrated (trend only), one-point ratio calibration at the first reference, OLS calibration on the first 10 min, and hourly recalibration.
5. **Reference alignment** (`references.py`). CardioQ/EV1000 numeric tracks are aligned to 20-s windows of beat estimates; thermodilution values from chartevents to +-5 min; echo LVOT-VTI SV to +-30/60 min windows around the study time.
6. **Agreement** (`agreement.py`). Bland-Altman with the repeated-measures variance decomposition (Bland & Altman, 2007), percentage error, four-quadrant concordance with a 10% exclusion zone, polar plots with +-30 degree limits, cluster (case-level) bootstrap CIs.
7. **Strata.** Vasopressor bolus/infusion changes (VitalDB drug tracks; MIMIC `inputevents`), SVV, atrial fibrillation (RR irregularity), damping class (from `arterial-line-damping-detection` features), sampling rate.

### Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/download_data.py --dataset vitaldb --out data/vitaldb --tracks --sample
```

```python
from pulse_contour import beats, estimators, agreement
s = beats.synthetic_abp(fs=125, n_beats=60, sv_ml=70, sv_variation=0.2)
f = beats.beat_features(s.abp, s.fs, beats.detect_onsets(s.abp, s.fs))
sv = estimators.calibrate(estimators.estimate(f, "windkessel2"), s.sv_true[: len(f)]).apply(estimators.estimate(f, "windkessel2"))
print(agreement.bland_altman(s.sv_true[: len(f)], sv))
```

## Evaluation & statistics

- Unit of analysis: aligned reference-estimate pairs nested in cases; all CIs by case-level bootstrap (1000).
- Primary metrics: Bland-Altman bias and LoA (repeated measures), percentage error, four-quadrant concordance, polar angular bias and radial LoA; secondary: MAE, Spearman rho of trends.
- Validation: estimator hyper-parameters (notch search window, beat filters) and impedance-correction coefficients fitted on a VitalDB development split (60% of cases), evaluated on the held-out 40% and on all MIMIC arms without refitting; case-level splits only.
- Leakage prevention: calibration uses only reference values at or before the current time; the impedance correction is never fitted on MIMIC.
- Multiple comparisons: 5 estimators x 3 references x 4 calibration regimes; Benjamini-Hochberg; H1-H5 pre-registered.
- Nulls and controls: (i) the Windkessel generator (`beats.synthetic_abp`) with known SV is the positive control (tests require Spearman > 0.8 for every estimator); (ii) shuffling reference values across cases gives the concordance null; (iii) a "constant SV" predictor gives the floor for trending statistics.
- Sensitivity: 125 vs 500 Hz; beat-quality threshold; alignment window; exclusion-zone size (10% vs 15%).

## Publishable angle

Headline result: "Open pulse-contour estimators match commercial FloTrac SV trends on identical beats, but against independent references both fall outside interchangeability during vasoactive changes; damping status explains a third of the excess error." A public, versioned benchmark of pulse-contour accuracy across OR and ICU references, with code, would be the first of its kind.

Target venues: British Journal of Anaesthesia; Journal of Clinical Monitoring and Computing; IEEE Transactions on Biomedical Engineering; Anesthesiology (if the OR arm is the lead story).

Follow-ups: deep-learning SV with uncertainty; fluid-responsiveness prediction from PPV/SVV on MIMIC boluses (backlog item 58); integration with `cuffless-bp-mimic-waveform` (shared beat detection and quality pipeline).

## Risks, confounds & mitigations

- Oesophageal Doppler is itself operator dependent and measures descending-aortic flow: report CardioQ as one reference among three; the direction of disagreement across references is informative.
- Sparse thermodilution values in MIMIC chartevents (cardiac surgery mostly): report n per stratum; treat continuous CO (CCO) separately since it is also pulse-derived on some devices.
- Echo field names in MIMIC-III reports vary by era: `parse_echo_measurements` keeps units and is validated against a manually checked sample of 200 reports (categories only, on the credentialed machine).
- MIMIC-IV Waveform v0.1.0 is small: the echo arm is planned on MIMIC-III first; the MIMIC-IV arm is a pilot until larger releases appear.
- Time alignment errors (charted time vs measurement time): sensitivity to window size; echo alignment reported at +-30 and +-60 min.
- Damping and arrhythmia confound estimator error: pre-specified strata; damping features imported rather than re-invented.

## Milestones

- [ ] VitalDB cases with CardioQ/EV1000 SV and ART identified; tracks downloaded; development/held-out split fixed
- [ ] Beat pipeline validated on VitalDB (onset sensitivity > 99% vs annotated subset of 20 cases)
- [ ] Estimators + calibration regimes evaluated against CardioQ and FloTrac; strata analyses
- [ ] PhysioNet credentialing; MIMIC-III waveform records with ABP and PA-catheter CO selected; thermodilution arm
- [ ] Echo report parser validated; echo-waveform alignment; echo arm
- [ ] Damping/arrhythmia/sampling-rate moderators
- [ ] Pre-registration (OSF) before MIMIC evaluation; manuscript and public benchmark release

## Repository layout

```
README.md                          this document
requirements.txt                   numpy/scipy/pandas/scikit-learn, wfdb, vitaldb (optional), requests
data/README.md                     acquisition steps for VitalDB (open) and the MIMIC arms (credentialed)
scripts/download_data.py           VitalDB REST downloader (track filtering, --sample) + PhysioNet staging
src/pulse_contour/beats.py         Windkessel generator with known SV, slope-sum onset detector, per-beat features
src/pulse_contour/estimators.py    Liljestrand-Zander, Herd, systolic area, 2-element Windkessel, fitted impedance correction, calibration
src/pulse_contour/agreement.py     Bland-Altman (repeated measures), percentage error, four-quadrant / polar concordance, cluster bootstrap
src/pulse_contour/references.py    echo LVOT parsing, d_items lookup for thermodilution CO, VitalDB loader, time alignment
tests/test_pulse_contour.py        synthetic Windkessel tests (all estimators must track true SV; agreement statistics)
```

Analysis outputs (`outputs/`, git-ignored): per-case aligned reference/estimate tables, agreement tables per
estimator x reference x calibration regime x stratum, and figures (Bland-Altman, four-quadrant, polar).

## Ethics / data-use notes

- MIMIC-III, MIMIC-IV, MIMIC-IV Waveform and MIMIC-IV-ECHO are PhysioNet credentialed resources: CITI training, signed DUAs, approved encrypted storage; no redistribution; credentialed data are never sent to third-party LLM APIs or other external services.
- VitalDB is open under CC BY-NC-SA for research; cite Lee et al. (Sci Data 2022) and respect the non-commercial clause.
- Never commit data; `.gitignore` excludes `data/`, waveform files and `outputs/`. Publish aggregate agreement statistics only.
- Credentials are read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`; never hard-code them.
