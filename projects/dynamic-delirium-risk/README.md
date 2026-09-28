# dynamic-delirium-risk

**Sedation-aware dynamic delirium risk in the ICU: multi-state landmark models that treat unassessable (comatose) windows and missing CAM-ICU screens as states rather than noise, quantify how much of a "dynamic" model's performance is sedation-policy leakage, and test transportability from MIMIC-IV to eICU.**

## Status / difficulty / timeline / compute

- Status: design + starter code (CAM-ICU/RASS parsing with regex-resolved item ids, 12-h window state machine, landmark dataset builder with three label schemes, sedative-exposure and trajectory features, cause-specific/multinomial landmark models, transition-intensity regression, inverse-probability-of-assessment weights, sedation-leakage ablation). No data is shipped.
- Difficulty: MSc to early-PhD. The modelling is standard; the contribution is the outcome/state definition, the causal framing of sedation, and the evaluation under realistic assessment schedules.
- Timeline: 6-9 months (1 month credentialing/extraction, 2 months state machine + validation against chart review, 2 months models + ablations, 1-2 months eICU external validation, 1-2 months writing).
- Compute: workstation, DuckDB over `chartevents` (~330M rows in MIMIC-IV v3.1; filter by item id on read). Gradient-boosted trees and logistic models; no GPU required. Optional GRU/transformer baselines fit on one GPU in hours.

## Background

ICU delirium is screened with the CAM-ICU (Ely et al., 2001, JAMA) and interpreted together with the Richmond Agitation-Sedation Scale (Sessler et al., 2002, Am J Respir Crit Care Med): a CAM-ICU is only valid when the patient is arousable (RASS >= -3); at RASS -4/-5 the patient is unassessable and counts as comatose. Delirium and coma are jointly tracked in trials as "days alive without delirium or coma" (e.g. MENDS, Pandharipande et al., 2007, JAMA; SEDCOM, Riker et al., 2009, JAMA), which showed that sedative choice (dexmedetomidine vs benzodiazepines) changes the time spent in each state. Static admission-time risk scores (PRE-DELIRIC, van den Boogaard et al., 2012, BMJ; E-PRE-DELIRIC, Wassenaar et al., 2015, Intensive Care Med) were followed by machine-learning models, reviewed by Ruppert et al. (2020, Crit Care Explor), and since 2023 by *dynamic* models that update every 12 h on EHR time series (Contreras et al., 2023; DeLLiriuM, Contreras et al., 2025, Sci Rep, 104,303 patients from eICU, MIMIC-IV and the University of Florida, with delirium defined as CAM-ICU positive with RASS >= -3 in any 12-h interval after 24 h). Post-cardiac-surgery and sepsis-associated delirium models trained on MIMIC-IV and validated on eICU appeared in 2025-2026 (JMIR Med Inform, 2026; Sci Prog, 2026), most with a single 24-h landmark and a binary outcome.

## The research gap

What the 2023-2026 dynamic models share:

1. **The label ignores the state structure.** Windows in which the patient is comatose (RASS -4/-5) or simply not screened are coded as "no delirium" or dropped. Under the first coding, deeper sedation *lowers* the label rate, so sedatives appear protective; under the second, the population at risk is selected by the sedation policy. Neither is what a clinician wants to know.
2. **Sedation policy leaks into "risk".** Sedative infusion rates, RASS targets and RASS values are strong features in every published dynamic model. They are partly *treatment* variables chosen by the team, so a high score may reflect the plan to lighten sedation (which makes delirium detectable) rather than the patient's vulnerability. No paper quantifies how much of the reported AUROC comes from these variables.
3. **Missing screens are treated as ignorable.** CAM-ICU completion depends on prior results, RASS, nursing workload and unit culture; the fraction of windows screened differs markedly between MIMIC-IV and eICU hospitals. Informative missingness biases both the labels and the apparent transportability.
4. **No competing risks.** Discharge and death within the horizon are dropped or ignored; for 12-24-h horizons they are common and informative.
5. **Fairness and transportability are single-number.** External AUROC on eICU is reported without stratification by assessment frequency, hospital, age band, sex, language or race.

Our angle: a multi-state (normal / delirium / coma-unassessable / unscreened / discharged / dead) landmark framework with sedation modelled explicitly as a time-varying exposure, an inverse-probability-of-assessment correction, an ablation that isolates sedation-policy leakage, and a transportability analysis that conditions on assessment density. All of it on MIMIC-IV (development, temporal validation) and eICU-CRD (external validation across hospitals).

## Research questions / hypotheses

1. **H1 (label scheme).** Coding coma/unscreened windows as negative vs excluding them vs treating them as a separate state changes the AUROC of the same feature set by >= 0.05 and flips the sign of at least one sedative's coefficient (benzodiazepine or propofol dose in the previous 12 h).
2. **H2 (sedation leakage).** Removing sedation-policy features (sedative doses, RASS target, RASS mean/min) from a dynamic model reduces cross-validated AUROC by >= 30% of the gain of the dynamic model over an admission-only model; the physiology-only model keeps calibration in the assessable population.
3. **H3 (transition-specific effects).** In a multi-state model, benzodiazepine exposure in the previous 12 h increases the normal -> delirium and coma -> delirium intensities, while dexmedetomidine does not increase normal -> delirium (direction consistent with MENDS/SEDCOM); propofol mainly increases normal -> coma.
4. **H4 (informative missingness).** P(CAM-ICU performed in a window) depends on the previous window's result and RASS (odds ratio for "previous window positive" > 1.5); inverse-probability-of-assessment weighting changes the estimated 12-h delirium incidence by a measurable amount in low-screening ICUs.
5. **H5 (transportability).** The MIMIC-IV model loses more discrimination on eICU hospitals in the lowest tertile of screening density than in the highest; per-hospital intercept recalibration restores calibration-in-the-large but not the subgroup gaps.
6. **H6 (equity).** Calibration differs by age band, sex, race and primary language (MIMIC-IV `admissions.language`), and the gap is larger for the binary-negative coding than for the multi-state coding.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 `icu` | `chartevents` (CAM-ICU items, "Delirium assessment", Richmond-RAS Scale, GCS, vitals), `inputevents` (propofol, midazolam, lorazepam, dexmedetomidine, opioids, vasopressors), `procedureevents` (ventilation), `icustays` | ~94k stays | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV v3.1 `hosp` | `patients`, `admissions` (language, race, deathtime), `labevents` (creatinine, lactate, sodium, glucose) | ~365k admissions | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV demo v2.2 | Pipeline smoke tests | 100 patients | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| eICU-CRD v2.0 | External validation: `nurseCharting` (delirium scale "CAM-ICU", sedation scale RASS), `infusionDrug`, `vitalPeriodic`, `lab`, `patient` (hospital id) | ~200k stays, 208 hospitals | PhysioNet credentialed | https://physionet.org/content/eicu-crd/2.0/ |

Item ids are resolved at run time by regex over `d_items.label` (patterns: `CAM-ICU`, `Delirium assessment`, `Richmond-RAS`) and printed; the same applies to sedatives. Do not copy ids from memory.

## Methods

1. **Assessment timeline** (`assessments.py`). Parse every CAM-ICU-related chart row into `positive / negative / uta` (unable to assess) using the four-feature rule (feature 1 and 2 and (3 or 4)) when only features are charted and the summary item when present; parse RASS to integers. 12-h windows from ICU admission. Window state: `delirium` if any positive CAM; else `normal` if any negative CAM; else `coma` if any UTA or all RASS <= -4; else `unscreened`. Terminal states `discharged` / `dead` from `icustays.outtime` and `admissions.deathtime`. A synthetic simulator (`simulate_stays`) generates state sequences with sedation-dependent transitions and an informative screening process for tests and power calculations.
2. **Landmark datasets** (`landmark.py`). For every landmark window t >= 2 (i.e. after 24 h) with a non-terminal state, the label is the state at t + h (h = 1 or 2 windows). Three label schemes are produced from the same rows: `multistate`, `binary_drop_coma` (rows with future coma/unscreened removed) and `binary_coma_negative`. Delirium/coma-free days at 14 days as the patient-level outcome.
3. **Features** (`features.py`). Physiology: last/mean/min/max of vitals, GCS, labs over 12 h and 24 h; ventilation; age, sex, admission type. Sedation-policy block: propofol, benzodiazepine, dexmedetomidine, opioid cumulative doses (mg or mcg per kg) over the previous 12 and 24 h from `inputevents`; RASS mean/min/max/slope; RASS target when charted. Assessment-process block: hours since last CAM, previous window's state, fraction of windows screened so far.
4. **Models** (`models.py`). Cause-specific logistic and multinomial landmark models and gradient boosting (LightGBM) per label scheme; transition-intensity Poisson regressions (statsmodels GLM with window-count exposure) per transition with sedative exposures as covariates; inverse-probability-of-assessment weights from a logistic model of screening; the sedation-leakage ablation (full vs physiology-only vs admission-only feature sets) with grouped cross-validation.
5. **External validation.** Re-implement the state machine on eICU `nurseCharting` (values are free-text categories; mapping table versioned in `data/`); apply the frozen MIMIC-IV model; per-hospital recalibration with n = 50/100/200 windows.

Tools: DuckDB, pandas/pyarrow, scikit-learn, LightGBM, statsmodels; optional PyTorch GRU baseline.

## Evaluation & statistics

- Splits by `subject_id`; temporal validation by `anchor_year_group` (train 2008-2016, test 2017-2022); eICU external.
- Discrimination: AUROC/AUPRC per landmark index and pooled; multi-state: one-vs-rest AUROC per future state and the Brier score of the multinomial forecast. Calibration: intercept/slope, ECE; decision curves at alert rates of 10-30%.
- CIs by cluster bootstrap over patients (1000 resamples); H1/H2 as paired differences across the same rows.
- Multiple comparisons: Benjamini-Hochberg over transitions x exposures (H3) and over subgroups (H6); H1-H5 pre-registered.
- Nulls/controls: (i) permuting window order within a patient must remove trajectory information (AUROC of trajectory features -> chance); (ii) a "screening model" that predicts *whether* a CAM-ICU will be done must not be confused with the delirium model (report both); (iii) simulation with the synthetic generator where sedation truly has no effect on delirium must give a null coefficient under the multi-state coding and a spurious protective one under the negative coding (validates H1's mechanism).
- Leakage: features use data strictly before the landmark; no labels from the landmark window itself; RASS values from the *future* window are never features.

## Publishable angle

Headline: "Dynamic ICU delirium models owe a large share of their reported performance to knowing how sedated the patient is and whether a screen will be done; treating coma and non-assessment as states, and sedation as an exposure, gives lower but honest discrimination, transportable calibration, and transition-specific sedative effects consistent with randomised trials."

Target venues: Critical Care Medicine; Intensive Care Medicine; Critical Care; JAMIA; npj Digital Medicine; conferences CHIL, ML4H, SCCM Congress.

Follow-ups: a target-trial emulation of early dexmedetomidine vs propofol on delirium-free days using the multi-state framework; a nursing-workload analysis of screening completion; extension to AmsterdamUMCdb if its delirium charting is usable.

## Risks, confounds & mitigations

- **CAM-ICU charting conventions differ across MIMIC-IV years and across eICU hospitals.** Mitigation: versioned value-mapping tables, chart-review validation of 200 windows, and sensitivity analyses using the summary item only vs the four-feature rule.
- **Sedation is both confounder and mediator.** Mitigation: state clearly that the landmark models are predictive, not causal; use the transition models with time-varying exposures and negative-control exposures (e.g. acetaminophen) for H3.
- **Sparse screening in eICU.** Mitigation: H5 stratifies on screening density; report the screening model.
- **Class imbalance and short stays.** Mitigation: landmark from 24 h; competing-risk handling; report AUPRC.
- **Race/language fields are administrative.** Mitigation: report as recorded; do not build "fair" models by removing the fields; interpret gaps as properties of the health-system process.

## Milestones

- [ ] Credentialing; download; DuckDB extraction of chart items resolved by regex; demo run.
- [ ] State machine; chart-review validation; descriptive transition matrix by year.
- [ ] Landmark datasets with three label schemes; admission-only, physiology-only and full models (H1, H2).
- [ ] Transition-intensity models with sedative exposures and negative controls (H3).
- [ ] Screening model and IPAW analysis (H4).
- [ ] eICU state machine, external validation, recalibration curves (H5), subgroup calibration (H6).
- [ ] Pre-registration; manuscript; code release with synthetic example data.

## Ethics / data-use notes

- MIMIC-IV and eICU-CRD are PhysioNet credentialed; credentials via `PHYSIONET_USER` / `PHYSIONET_PASS` only; `data/` is git-ignored; never commit extracts or row-level derived tables.
- No credentialed data may be sent to third-party LLM APIs or any external service (PhysioNet responsible-use policy). Everything here runs locally; LLM baselines, if any, must be local open-weights models.
- Models are research artefacts, not decision tools. Sedation findings are observational and must not be read as dosing advice.
