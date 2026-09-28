# circadian-label-bias-icu

**Circadian label bias: how workflow-driven timing of ICU outcome labels (discharge, death, sepsis onset, AKI, intubation) imprints time-of-day and staffing structure on benchmark targets in MIMIC-IV and eICU-CRD, how much benchmark performance is "clock" rather than physiology, and how to evaluate models so the clock cannot inflate them.**

## Status / difficulty / timeline / compute

- Status: design + starter code (circular statistics for event timing, documentation-delay and phase-randomisation nulls for windowed labels, clock-feature ablation with hour-stratified metrics). No data is shipped.
- Difficulty: MSc-level (timing audit + ablation) to early-PhD (label-timing noise model and deconvolution). The novelty is in the framing and the nulls, not in heavy modelling.
- Timeline: 5-8 months (1 month credentialing + extraction, 1-2 months timing audit on both databases, 2 months ablations and nulls on standard benchmarks, 1 month staffing/hospital-level analysis in eICU, 1 month writing).
- Compute: a workstation with 64 GB RAM; DuckDB over MIMIC-IV/eICU; gradient-boosted trees and logistic regression only.

## Background

ICU outcomes have strong time-of-day structure that is not physiological. ICU discharge hazards drop sharply after 16:00 and are lowest between 00:00 and 03:59; discharge decisions are made at morning rounds ("Time of day and its association with risk of death and chance of discharge in critically ill patients", 2019). Night and early-morning ICU admissions carry higher mortality (a 2026 Scientific Reports analysis of MIMIC-IV sepsis admissions; Wallace et al., 2012 NEJM on night-time intensivist staffing). Documentation is shift-structured: in MIMIC-IV, `charttime` is when staff considered a value valid and `storetime` when it was entered (Johnson et al., 2023 Scientific Data); nurses chart on the hour; labs are drawn in early-morning batches; a 2026 preprint on informative sampling in MIMIC-IV notes reports that night-shift notes are fewer but 2.7-3.0 times more likely to describe acute deterioration. Vital-sign circadian rhythms themselves change before discharge (a 2020 retrospective ICU analysis).

Benchmarks (MIMIC-III benchmark: Harutyunyan et al., 2019 Scientific Data; YAIB: van de Water et al., 2024 ICLR; MIMIC-IV pipelines: Gupta et al., 2022 MLHC) define labels relative to clock times: mortality within 48 h of a prediction time, length-of-stay remaining, discharge in the next 24 h, Sepsis-3 onset (defined from antibiotic-order and culture times), KDIGO AKI (defined from creatinine-draw times). Every one of these labels inherits the workflow clock.

## The research gap

What has been done (2023-2026):

- Descriptive epidemiology of admission/discharge timing and mortality (2019 registry study; 2026 MIMIC-IV sepsis study; NEWS time dependence in Resuscitation 2023).
- Decision-curve analysis of ICU discharge-prediction models that incorporates staffing and time constraints (medRxiv 2026, MIMIC-IV, 8:30 rounds), i.e., the clock enters at the *deployment* stage.
- Informative-sampling and observation-process modelling for irregular EHR time series (several 2025-2026 preprints), which treat measurement timing as a feature, not label timing as a bias.
- A critical reading of MIMIC-IV's construction (Barlas, 2025 AIES) that discusses timestamp quirks qualitatively.
- The related project `icu-model-transportability` in this repository treats time-of-day as one nuisance in cross-site shift; the present project isolates it and makes the *label* clock the object of study.

What is specifically missing (our angle):

1. A **label-timing audit**: for each benchmark label, the hour-of-day and day-of-week distribution of the defining event (ICU out-time, death time, antibiotic order/administration time, culture time, first creatinine meeting KDIGO, intubation, vasopressor start), with circular statistics and phase-locking to shift changes (07:00/19:00) and rounds, in MIMIC-IV and eICU (208 hospitals, so hospital-level variation in the phase-locking can be related to hospital characteristics).
2. A **decomposition of benchmark performance into clock and physiology**: train each standard task with (a) full features, (b) features minus all clock information (hour, weekday, and the measurement-time patterns that encode them), (c) clock-only features; report AUROC/AUPRC/calibration stratified by prediction hour and by admission hour.
3. **Phase-randomisation nulls for windowed labels**: shift each event's clock time by a random offset while preserving its calendar day and ordering, recompute windowed labels (e.g., 48-h mortality at a fixed prediction hour), and measure the label-flip rate and the resulting metric change; this quantifies how much of a label is "hour of the clock".
4. A **label-timing noise model**: observed event time = physiological time + workflow delay(hour of day, weekday, hospital); delays estimated from `storetime - charttime`, lab order-to-result intervals and discharge-readiness-to-discharge gaps (where "ready" can be proxied by the last vasopressor stop / extubation), and used to deconvolve outcome-hazard curves.
5. **Equity and deployment**: whether night-time/weekend admissions receive systematically worse calibration, and a **clock-robust evaluation protocol** (hour-stratified metrics, phase-randomised nulls, "clock-only" baselines) proposed as a reporting standard for ICU benchmarks.

## Research questions / hypotheses

1. **RQ1 (timing).** How strongly are label-defining events phase-locked to the clock? H1: ICU discharge and death show mean resultant lengths R > 0.3 with peaks in the afternoon; deaths after withdrawal of life-sustaining therapy cluster in daytime; Sepsis-3 onset time (culture/antibiotic) is bimodal around morning rounds and evening admissions; AKI onset by creatinine is locked to the 04:00-06:00 lab draw (R > 0.5).
2. **RQ2 (clock share).** H2: a clock-only model (hour of prediction, hour of admission, weekday, hours since admission) reaches AUROC > 0.6 for next-24-h discharge and > 0.55 for 48-h mortality; removing clock features from a full model costs < 0.01 AUROC overall but > 0.03 in the night-time strata.
3. **RQ3 (phase-randomisation).** H3: randomising event clock time flips > 10% of 48-h mortality labels at a fixed prediction hour and > 25% of next-24-h discharge labels; benchmark AUROC drops by more than the reported difference between competing architectures.
4. **RQ4 (delays).** H4: `storetime - charttime` delays are longest in the 04:00-08:00 and 16:00-20:00 windows (shift handover) and are larger for deterioration-related values; deconvolving these delays shifts the apparent hazard peaks by 1-3 h.
5. **RQ5 (hospital variation, eICU).** H5: phase-locking of discharge time varies across eICU hospitals (R from < 0.2 to > 0.5) and relates to hospital size/teaching status; models trained on strongly phase-locked hospitals transfer worse to weakly phase-locked ones after matching case mix.
6. **RQ6 (equity).** H6: calibration intercepts differ by admission-hour band (night vs day) with night admissions under-predicted for mortality, and the gap is not removed by recalibration on pooled data.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (hosp + icu) | Timestamps for admissions/transfers/`icustays`, `deathtime`, `chartevents` (`charttime`, `storetime`), `labevents` (`charttime`, `storetime`), `inputevents`, `procedureevents`, `prescriptions`/`emar`, `microbiologyevents`; all benchmark labels | ~94k ICU stays; ~30 GB | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV Clinical Database Demo v2.2 | Pipeline development and tests | ~40 MB | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| eICU-CRD v2.0 | Same labels across 208 hospitals; `hospital` table (bed size, teaching status, region); offsets in minutes from ICU admission plus `hospitaladmittime24`/`unitadmittime24` clock times | ~200k stays; ~20 GB | PhysioNet credentialed | https://physionet.org/content/eicu-crd/2.0/ |
| eICU-CRD Demo v2.0.1 | Pipeline development and tests | ~50 MB | Open | https://physionet.org/content/eicu-crd-demo/2.0.1/ |
| MIMIC-IV-ED v2.2 (optional) | ED arrival and triage times for admission-path analyses | ~1 GB | PhysioNet credentialed | https://physionet.org/content/mimic-iv-ed/2.2/ |
| YAIB / mimic-code task definitions | Standard label definitions reproduced exactly | small | Open | https://github.com/rvandewater/YAIB ; https://github.com/MIT-LCP/mimic-code |

Note on MIMIC-IV time shifting: dates are shifted per patient into 2100-2200, but the time of day and day of week are preserved (Johnson et al., 2023), which is what this project needs. eICU provides `unitadmittime24` and offsets, so event clock times are reconstructed as admission clock time + offset.

## Methods

Pipeline (modules in `src/circadian_bias/`):

1. **Event tables.** DuckDB extraction of every label-defining event per stay with its clock time, weekday, and (where available) `storetime`. Standard labels reproduced from YAIB/mimic-code definitions: 48-h and in-hospital mortality, next-24-h ICU discharge, remaining LOS, Sepsis-3 onset (mimic-code `sepsis3`), KDIGO AKI onset, intubation, vasopressor start.
2. **Timing audit** (`circular.py`). Hour-of-day histograms; mean resultant length and Rayleigh test; circular mean; peak-to-trough ratio; a shift-change phase-locking statistic (mass within +/- 1 h of 07:00/19:00 relative to a uniform expectation, with a permutation p-value); weekday effects; eICU hospital-level versions with bootstrap CIs.
3. **Documentation delays** (`label_timing.py`). Distribution of `storetime - charttime` by hour, item and value abnormality (flag); lab order-to-result intervals; delay curves used as the kernel of the label-timing noise model.
4. **Phase-randomisation null** (`label_timing.py`). For each stay, add a random offset in [0, 24) h to all label-defining event times (preserving order and calendar day structure), recompute windowed labels at the fixed prediction hours, compute label-flip rates and re-evaluate frozen models; repeat 200 times.
5. **Clock ablation** (`clock_ablation.py`). Feature sets: physiology (YAIB feature set), physiology + clock (hour sin/cos, weekday, night flag, hours since admission), clock only. Models: L2 logistic regression and gradient-boosted trees. Metrics overall and stratified by prediction hour (6 bins) and admission hour; calibration intercept/slope per stratum; the "clock share" = AUROC(clock-only) - 0.5 and the "clock increment" = AUROC(full) - AUROC(no clock), each with bootstrap CIs.
6. **Hospital-level analysis (eICU).** Phase-locking per hospital vs hospital characteristics (mixed model with hospital random effect); cross-hospital transfer of discharge models stratified by phase-locking tertile with case-mix matching.
7. **Deconvolution.** Hazard-by-hour curves for discharge/death deconvolved with the estimated delay kernels (Richardson-Lucy on circular histograms) to estimate the "physiological" event timing.

Tools: DuckDB, pandas, numpy/scipy (circular statistics), scikit-learn, LightGBM (optional), statsmodels (mixed models).

## Evaluation & statistics

- Circular statistics: Rayleigh test with n >= 100 events per stratum; CIs by bootstrap over stays; multiple labels x strata corrected with BH-FDR.
- Model evaluation: patient-level 70/15/15 splits fixed once; all feature sets share the split; bootstrap CIs (1,000) for AUROC/AUPRC/Brier and for differences between feature sets (paired bootstrap).
- Stratified metrics reported with minimum stratum size 500 stays; calibration via logistic recalibration per stratum.
- Nulls: phase-randomisation (label clock destroyed), label permutation (all information destroyed) and a positive control where a synthetic hour-dependent label is injected.
- Leakage: clock features are computed from prediction-time information only; `storetime` never enters the feature set except in the delay analysis.
- Sensitivity: prediction-time grid (every 6 h vs hourly), horizon (24/48 h), exclusion of deaths after withdrawal-of-care documentation, and eICU hospitals with incomplete `unitadmittime24`.

## Publishable angle

Headline: "ICU benchmark labels are phase-locked to the hospital clock; a clock-only model reaches AUROC X on discharge prediction, randomising the label clock flips Y% of labels and moves AUROC more than architecture choices do; night-time admissions are systematically mis-calibrated. We propose hour-stratified, phase-randomised evaluation as a reporting standard." This is a methodological audit with direct implications for every published ICU benchmark.

Target venues: npj Digital Medicine; JAMIA; Critical Care (audit + equity framing); ML4H / CHIL proceedings for the evaluation-protocol paper.

Follow-ups: apply the protocol to ward deterioration scores (NEWS-based alerts); add HiRID and AmsterdamUMCdb (different countries, different rounding cultures); model the clock as a causal nuisance (front-door adjustment via delay kernels); a `clockcheck` package that emits the audit for any OMOP-formatted dataset.

## Risks, confounds & mitigations

- Time-of-day correlates with case mix (night admissions come from the ED and are sicker): mitigation: stratified analyses within admission source and severity bands; the phase-randomisation null holds the patient fixed and only moves the label clock.
- MIMIC-IV date shifting preserves time of day but not season or year; mitigation: no seasonal claims; weekday effects use the preserved weekday.
- eICU clock reconstruction depends on `unitadmittime24` accuracy; mitigation: validate against `hospitaladmittime24` consistency and exclude inconsistent hospitals.
- Deaths include withdrawal of care, which is a decision time, not a physiological time; mitigation: separate analyses excluding stays with comfort-care documentation (chartevents code status) and report both.
- Ablating clock information from physiology features is imperfect (measurement counts encode the clock): mitigation: a "clock-blind" feature set that also removes measurement-timing features, with the residual clock share measured by a clock-from-features probe.
- Multiple labels and strata inflate false positives: BH-FDR, pre-registered H1-H6.

## Milestones

- [ ] Pipeline runs end to end on the open demo databases (`--sample`), including the audit tables
- [ ] PhysioNet credentialing complete; full event tables extracted for MIMIC-IV and eICU
- [ ] Timing audit tables and figures (circular statistics per label, per hospital in eICU)
- [ ] Documentation-delay kernels estimated
- [ ] Standard benchmark labels reproduced within +-2% prevalence of YAIB
- [ ] Clock ablation with hour-stratified metrics and calibration
- [ ] Phase-randomisation nulls and label-flip rates
- [ ] Hospital-level phase-locking and transfer analysis (eICU)
- [ ] Deconvolution of hazard curves; equity analysis by admission hour
- [ ] Pre-registration (OSF); manuscript and `clockcheck` release

## Ethics / data-use notes

- MIMIC-IV, MIMIC-IV-ED and eICU-CRD are PhysioNet credentialed resources (CITI training + DUA). Follow PhysioNet's responsible-use policy: no patient-level data, timestamps or free text may be sent to third-party LLM APIs; use local tooling only.
- `scripts/download_data.py` reads `PHYSIONET_USER` / `PHYSIONET_PASS` from the environment; credentials are never written to disk or committed.
- Never commit data or patient-level derived tables; `.gitignore` covers `data/`, `outputs/`, DuckDB files. Publish only aggregate results (cell counts >= 10).
- Time-of-day disparities touch on staffing and equity; report them descriptively with hospital-level context and without identifying individual units.
