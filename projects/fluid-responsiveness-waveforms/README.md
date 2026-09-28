# fluid-responsiveness-waveforms

**Pulse-pressure variation "in the wild": waveform-derived PPV/SPV computed automatically on MIMIC arterial lines, its predictive value for the haemodynamic response to real fluid boluses recorded in `inputevents`, stratified by whether the classic validity prerequisites hold, and benchmarked against sham-bolus windows that quantify regression to the mean.**

## Status / difficulty / timeline / compute

- Status: design + starter code (ABP beat detection and signal-quality mask, windowed PPV/SPV, respiratory-rate estimate from PP modulation, bolus identification from `inputevents`, pre/post response windows, sham-bolus sampler, prerequisite flags, AUROC with cluster bootstrap, gray-zone estimation). No data is shipped.
- Difficulty: MSc-level for the descriptive study; PhD-level if the waveform-ML and causal parts are pursued. The hard parts are signal quality, linking waveforms to clinical timestamps, and the outcome definition without a cardiac-output monitor.
- Timeline: 6-9 months (1 month credentialing + waveform indexing, 2 months bolus/response tables and chart validation, 2 months PPV pipeline + VitalDB algorithm validation, 2 months analyses, 1-2 months writing).
- Compute: workstation with fast storage; MIMIC-III matched waveforms are tens of TB in total but only the +/- 1 h around each bolus is needed (stream with `wfdb` and cache). Optional GPU for waveform-ML baselines.

## Background

Dynamic indices of preload responsiveness (pulse-pressure variation, PPV; stroke-volume variation, SVV; systolic-pressure variation, SPV) predict whether a fluid bolus will increase stroke volume (Michard & Teboul, 2002, Chest; Michard, 2005, Anesthesiology; Monnet, Marik & Teboul, 2016, Ann Intensive Care). Meta-analyses give AUC ~0.87 for PPV with a threshold near 11-12% (Crit Care, 2024, 40 PPV studies), and a "gray zone" of roughly 9-13% in which PPV is inconclusive (Cannesson et al., 2011, Anesthesiology). Validity, however, requires controlled ventilation with tidal volume >= 8 mL/kg, no spontaneous effort, a regular rhythm and a closed chest (De Backer et al., 2005, Intensive Care Med for low tidal volumes). A prospective multicentre point-prevalence study found that only a small minority of ICU patients meet all PPV validity criteria at a given moment (Mahjoub et al., 2014, Br J Anaesth). Automated PPV algorithms on the arterial waveform have been validated in the operating room (Cannesson et al., 2008, Anesth Analg), and waveform machine learning has been applied to fluid responsiveness in animals (Sci Rep, 2024: 394 boluses in 58 pigs, random forest on ABP features vs PPV AUROC 0.73) and to hypotension prediction in humans (Hatib et al., 2018, Anesthesiology).

MIMIC is the only open resource where bedside arterial waveforms co-exist with the timing and volume of real fluid boluses (`inputevents` with start/end/amount), ventilator settings, rhythm and vasopressor changes for thousands of ICU patients. Retrospective MIMIC studies of bolus response exist (Girkar et al., 2018, arXiv: attention networks predicting blood-pressure response to boluses in MIMIC-III; a MIMIC-III matched-waveform study of fluid responsiveness in sepsis), but they use clinical variables or opaque waveform features and do not compute the guideline index itself.

## The research gap

What is missing (2023-2026):

1. **No real-world estimate of how PPV performs when the prerequisites are *not* met.** All validation cohorts select patients who satisfy the criteria; the point-prevalence study shows that is rarely the case at the bedside. MIMIC allows PPV to be computed for every bolus and the validity criteria to be checked from charted ventilator settings, spontaneous breathing (set vs total respiratory rate, mode) and rhythm (RR-interval irregularity from the ECG/ABP).
2. **Regression to the mean is never controlled.** Boluses are given when pressure is low; pressure then tends to rise anyway. Sham-bolus windows matched on pre-window MAP, drawn from the same patients at times with no bolus, give the "response rate" expected under no treatment. No bolus-response study on MIMIC has reported it.
3. **Outcome definitions without cardiac output.** MIMIC has thermodilution/continuous cardiac output for a minority; for the rest, PP (a stroke-volume surrogate) and MAP responses are the only options. Comparing definitions, and validating the PP surrogate against device stroke volume on VitalDB, is itself a contribution.
4. **Co-interventions.** Vasopressor rate changes within the response window contaminate the outcome; they can be identified in `inputevents` and used as exclusion or adjustment.
5. **An open algorithm-validation step.** VitalDB provides ABP waveforms with EV1000/Vigileo SVV and SV; our PPV/SPV implementation can be validated against device values before it is applied to MIMIC.

The angle is a methodological audit of the guideline index under real ICU conditions, with an explicit regression-to-the-mean null, rather than another black-box predictor.

## Research questions / hypotheses

1. **H1 (real-world PPV).** In unselected ICU boluses with an arterial line, automated PPV has AUROC 0.60-0.70 for a >= 10% MAP (or >= 15% PP) response; in the subset that satisfies all prerequisites (controlled ventilation, Vt >= 8 mL/kg PBW, no spontaneous triggering, regular rhythm) AUROC >= 0.80 and the gray zone narrows.
2. **H2 (regression to the mean).** Sham windows matched on pre-window MAP show a "response" rate that is at least half of the crude bolus response rate; the bolus-attributable response fraction is reported per pre-MAP stratum.
3. **H3 (co-interventions).** Excluding windows with a vasopressor rate change alters the responder rate by more than 20% relative and changes the PPV AUROC by >= 0.03.
4. **H4 (outcome definition).** PP-based and MAP-based responder labels agree only moderately (Cohen's kappa < 0.6); PPV predicts the PP-based label better, consistent with PP tracking stroke volume.
5. **H5 (dose-response).** The MAP/PP response increases sublinearly with bolus volume per kg after adjustment for pre-MAP (mixed model with patient random intercept).
6. **H6 (algorithm validation on VitalDB).** Waveform PPV agrees with EV1000/Vigileo SVV/PPV with mean bias < 2 percentage points and limits of agreement narrower than +/- 6 points during periods without arrhythmia.
7. **H7 (waveform ML).** A pre-bolus waveform feature model (PPV, SPV, dP/dt max, PP/HR trends, morphology) improves on PPV alone in the unselected set but the increment shrinks in the prerequisite-satisfying subset.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-III Waveform Database Matched Subset v1.0 | ABP (125 Hz), ECG, PLETH, numerics (1/min) linked to MIMIC-III patients | 22,317 waveform records, 10,282 patients | Open | https://physionet.org/content/mimic3wdb-matched/1.0/ |
| MIMIC-III Clinical Database v1.4 | `INPUTEVENTS_MV` (boluses, vasopressors), `CHARTEVENTS` (ventilator settings, rhythm, CO), `D_ITEMS`, `ICUSTAYS` | ~61k stays (MetaVision era 2008-2012 has bolus start/end) | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciii/1.4/ |
| MIMIC-IV Waveform Database v0.1.0 | Newer waveform release linked to MIMIC-IV (initial release is small; grows with later versions) | ~200 records initially | Open (waveforms); linked clinical data credentialed | https://physionet.org/content/mimic4wdb/0.1.0/ |
| MIMIC-IV v3.1 `icu` | `inputevents`, `chartevents`, `d_items`, `icustays` for the MIMIC-IV waveform subset | ~94k stays | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV demo v2.2 | Pipeline smoke tests for the clinical side | 100 patients | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| VitalDB | Intra-operative ABP (500 Hz) with EV1000/Vigileo SV, SVV, CO for algorithm validation | 6,388 cases | Open (free registration for the web tools; API is open) | https://vitaldb.net/dataset/ |

## Methods

1. **Waveform indexing.** Map `subject_id` -> matched waveform records; for each candidate bolus, locate the ABP segment covering [-60, +60] min; fetch with `wfdb` and cache per bolus (never the whole database).
2. **Beats and quality** (`abp_beats.py`). Systolic peaks by prominence/refractory peak detection; diastolic minima; per-beat SBP/DBP/PP/MAP/RR-interval; a signal-quality mask (physiologic ranges, beat-to-beat jumps, flatline, PP >= 10 mmHg). Prerequisite-related metrics: RR-interval irregularity from beats (and ECG when available).
3. **Indices** (`ppv.py`). PPV = 100 x (PPmax - PPmin) / ((PPmax + PPmin)/2) over 8-s windows stepped by 2 s, aggregated by median over the pre-bolus analysis window (last 5 min before the bolus with >= 80% good beats); SPV likewise; respiratory rate from the PP-modulation spectrum as a cross-check against charted RR.
4. **Boluses and responses** (`boluses.py`). Bolus = crystalloid/colloid `inputevents` row with >= 250 mL delivered in <= 30 min and no other bolus in [-60, +30] min; blood products separate. Pre window [-15, 0) min and post window [end, end+30] min on 1-min numerics (or beat-derived series). Labels: MAP >= 10% and PP >= 15% increases (mean and peak variants); vasopressor rate change flag; sham windows matched on pre-MAP within the same patient; prerequisite flags from charted mode, set/total RR, tidal volume / PBW (ARDSNet formula), rhythm.
5. **Evaluation** (`evaluate.py`). AUROC with cluster bootstrap by patient, gray zone (thresholds at 90% sensitivity and 90% specificity with bootstrap), attributable response vs sham windows, stratified AUROC by prerequisite status.
6. **VitalDB validation.** Compute PPV/SPV on `SNUADC/ART` and compare with `EV1000/SVV` or `Vigileo/SVV` per minute (Bland-Altman, concordance), excluding arrhythmic minutes.
7. **Models.** Mixed-effects logistic (statsmodels `BinomialBayesMixedGLM` or R `lme4`) for response with patient random intercept; gradient boosting on pre-bolus waveform features (H7) with grouped CV.

Tools: `wfdb`, NumPy/SciPy, pandas, DuckDB, scikit-learn, statsmodels; `vitaldb` Python package for VitalDB.

## Evaluation & statistics

- Unit: bolus; clusters: patients. Splits by patient for any learned model. PPV is computed only from pre-bolus data; the post window never enters any feature.
- Primary metric: AUROC and gray-zone width; secondary: calibration of a logistic model on PPV, net benefit at decision thresholds of 30-70% probability of response.
- Nulls: (i) sham windows for the outcome base rate; (ii) PPV computed on the *post* window as a negative control for the direction of causality (should not predict the pre-to-post change beyond regression to the mean); (iii) PPV from a randomly chosen other patient (permutation across patients) -> AUROC 0.5.
- Multiple comparisons: Benjamini-Hochberg across strata and label definitions; H1-H3 pre-registered.
- Reporting: STARD-style flow of boluses excluded for signal quality, missing waveform, co-interventions.

## Publishable angle

Headline: "Outside the conditions of its validation studies, automated PPV is a weak predictor of the pressure response to a fluid bolus, half of the observed 'responses' are expected from regression to the mean, and the gray zone covers most real ICU boluses; within the prerequisite-satisfying minority, PPV works as advertised." An accompanying open pipeline (beat detection, PPV, bolus linkage, sham windows) makes the audit repeatable on later MIMIC-IV waveform releases.

Target venues: Critical Care; British Journal of Anaesthesia; Anesthesiology; Intensive Care Medicine; Journal of Clinical Monitoring and Computing; Physiological Measurement; Computing in Cardiology (CinC) for the algorithm-validation part.

Follow-ups: a target-trial emulation of restrictive vs liberal bolus strategies using PPV strata; waveform foundation-model embeddings as fluid-responsiveness features; extension to the plethysmographic variability index from PLETH.

Related project: `cuffless-bp-mimic-waveform` (shares the MIMIC waveform indexing and ABP beat-detection approach; this project is self-contained).

## Risks, confounds & mitigations

- **Waveform coverage.** Not every bolus has an ABP segment; the initial MIMIC-IV waveform release is small. Mitigation: MIMIC-III matched subset is the primary source; report coverage; re-run on later MIMIC-IV waveform versions.
- **Timestamp alignment between waveform and clinical records.** Mitigation: use numerics-vs-chartevents cross-correlation of HR/MAP to verify alignment per record and exclude mismatches.
- **No stroke volume in most stays.** Mitigation: PP surrogate validated on VitalDB; CO subset analysed separately; results framed as "pressure response".
- **Damped/under-damped arterial lines.** Mitigation: SQI, fast-flush artefact detection heuristics, sensitivity analysis on strict SQI; this is also the topic of a separate backlog seed.
- **Confounding by indication in the dose-response analysis.** Mitigation: adjust for pre-MAP, vasopressors, and use sham windows; interpret associations, not effects.

## Milestones

- [ ] Credentialing; waveform index; `inputevents` bolus table on the MIMIC-IV demo and on MIMIC-III.
- [ ] Beat detection + SQI validated on 50 manually reviewed segments; VitalDB PPV vs SVV agreement (H6).
- [ ] Response tables with MAP/PP labels, co-intervention flags, sham windows (H2-H4).
- [ ] Prerequisite flags from ventilator/rhythm charting; stratified AUROC and gray zones (H1).
- [ ] Mixed-model dose-response (H5); waveform-ML comparison (H7).
- [ ] Pre-registration; manuscript; pipeline release.

## Ethics / data-use notes

- MIMIC-III/IV clinical data are PhysioNet credentialed; the waveform databases are open but link to credentialed records, so treat derived tables as credentialed. Credentials from `PHYSIONET_USER`/`PHYSIONET_PASS` only; `data/` is git-ignored; never commit waveform excerpts or derived tables.
- No credentialed data to third-party LLM APIs or external services (PhysioNet responsible-use policy). VitalDB is open but its terms of use must be followed.
- The analysis is observational; nothing here is a bedside decision tool.
