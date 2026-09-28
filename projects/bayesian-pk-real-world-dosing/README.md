# bayesian-pk-real-world-dosing

**An open external-evaluation benchmark of published vancomycin (and gentamicin) population-PK models for Bayesian forecasting on MIMIC-IV, quantifying how real-world dosing-record noise, practice drift and patient subgroup propagate into AUC estimates and dosing decisions.**

## Status / difficulty / timeline / compute

- Status: design + starter code (dose/level table builder from `emar`/`inputevents`/`labevents`, 1- and 2-compartment infusion models, MAP-Bayesian forecasting with model averaging, prediction-error metrics, timing-perturbation experiment, eGFR/CrCl helpers). No data is shipped.
- Difficulty: MSc-level with a pharmacometrics co-supervisor; the PK maths is standard, the difficulty is the EHR data engineering (dose provenance, level timing) and the statistics of a fair model comparison.
- Timeline: 6-9 months (1 month credentialing + extraction, 2 months dosing/level curation and validation against chart review rules, 2 months model library + forecasting, 2 months analyses, 1-2 months writing).
- Compute: laptop/workstation. DuckDB over the MIMIC-IV csv.gz files (hosp `emar`/`emar_detail` are the largest tables at a few GB). Full-posterior fits (Stan/Torsten or PyMC) for a few thousand courses take hours on 8 cores; MAP fits take minutes. No GPU.

## Background

Vancomycin dosing moved from trough targeting to AUC-guided, Bayesian model-informed precision dosing (MIPD) with the 2020 ASHP/IDSA/PIDS/SIDP consensus guideline (Rybak et al., 2020, Am J Health-Syst Pharm). Commercial and academic Bayesian tools embed one or more published population-PK (popPK) models and estimate individual clearance and volume from one or two measured levels. The choice of model matters: Broeker et al. (2019, Clin Microbiol Infect) evaluated 31 vancomycin models for Bayesian forecasting and found large differences in predictive performance; Guo et al. (2019, Antimicrob Agents Chemother) externally evaluated ICU models on large ICU cohorts and found that even ICU-derived models mis-predict in the ICU; Uster et al. (2021, Clin Pharmacol Ther) showed that model averaging/selection improves forecasting; Hughes & Keizer (2021, Clin Pharmacol Ther) showed that "continuous learning" (refitting priors on local data) helps. Neely et al. (2018, Antimicrob Agents Chemother) and the PROVIDE study (Lodise et al., 2020, Clin Infect Dis) link AUC exposure to nephrotoxicity and efficacy, which is why the AUC estimate, not the level, is the quantity that drives decisions.

Almost all of that evidence comes from curated pharmacokinetic datasets (protocolised sampling, verified infusion start/stop times) or from proprietary MIPD-platform data (e.g. the 79,600-course, 84-health-system analysis of model choice by age and BMI presented at IDWeek 2023). What a hospital's Bayesian tool actually receives is EHR data: administration timestamps from an eMAR, levels labelled "trough" whether or not they were drawn before the dose, missing infusion durations, and sparse weights. MIMIC-IV (Johnson et al., 2023, Sci Data) is the only large, open, credentialed ICU dataset that exposes exactly that layer: `emar`/`emar_detail` (barcode-scanned administrations with dose and rate), ICU `inputevents` (infusion start/end/rate), `prescriptions`, `labevents` vancomycin and aminoglycoside levels with draw times, creatinine, weight, CRRT and vasopressor records, and outcomes (KDIGO AKI, mortality).

## The research gap

What has been done with MIMIC-IV and vancomycin (2022-2026):

- Outcome epidemiology: first trough concentration and KDIGO AKI in 3,917 ICU patients treated > 48 h (MIMIC-IV retrospective study, 2022); vancomycin TDM vs mortality in 18,056 ICU patients of whom 7,451 had at least one level (2024), and a further TDM-outcome cohort (Sci Rep, 2026). These use the raw level values as the exposure; none reconstruct the dosing history or estimate AUC with a PK model.
- A registered study (NCT06431412) plans to use MIMIC-IV only as an external test set for a machine-learning trough predictor built on single-centre data.
- Hybrid popPK + machine-learning clearance predictors (2025) and Bayesian exposure estimation in special populations (Antibiotics, 2026) are built on closed hospital datasets.
- External-evaluation papers for vancomycin models (Guo et al., 2019; a two-centre evaluation in 2021; obese/critically-ill model evaluations in 2023) all use curated TDM datasets that cannot be redistributed, so no two groups evaluate on the same data.

What is specifically missing (our angle):

1. **An open, reproducible external-evaluation benchmark** for Bayesian forecasting: a fixed MIMIC-IV cohort, fixed course definitions, a versioned library of published model parameterisations, and a scoring protocol (a priori / a posteriori after 1, 2, 3 levels; rBias, rRMSE, AUC24 agreement, target attainment). Anyone with PhysioNet credentials can rerun it and add a model.
2. **Dosing-record provenance as a source of forecasting error.** MIMIC-IV has three overlapping dose sources with different fidelity (`inputevents` start/end/rate in the ICU; `emar` barcode administration times for admissions after the eMAR go-live; `prescriptions` scheduled orders). We measure how much AUC24 and next-level forecasts change between sources and run a planted timing-error experiment to quantify the sensitivity of AUC estimates to +/- 15-60 min administration-time errors and to unrecorded infusion durations.
3. **Practice drift and continuous learning across MIMIC-IV's anchor-year groups (2008-2022)**, which span the trough-to-AUC guideline transition: does a prior refit on 2008-2013 courses mis-predict 2017-2022 courses, and does model averaging or local refitting fix it?
4. **Equity of MIPD**: forecast error and target attainment by sex, race, obesity class and renal function, including the effect of replacing the race-adjusted CKD-EPI 2009 eGFR with the race-free CKD-EPI 2021 equation (Inker et al., 2021, N Engl J Med) as the clearance covariate. To our knowledge no MIPD external evaluation has reported subgroup forecasting error.
5. **Exposure-response with measurement-error correction**: day-2 Bayesian AUC24 (with its posterior uncertainty) vs first trough as predictors of KDIGO AKI, in a joint/errors-in-variables model rather than plugging in point estimates.

Gentamicin is included as a secondary drug (levels exist in `labevents`; extended-interval dosing) to test whether the provenance findings generalise to a drug with a very different sampling design.

## Research questions / hypotheses

1. **H1 (model disagreement).** For the same course, the between-model spread (max/min) of AUC24 estimated a posteriori from the first level exceeds 1.25 in more than 30% of courses across >= 8 published ICU vancomycin models, and falls below 1.15 after two levels.
2. **H2 (provenance).** Using `emar` administration times instead of `inputevents` start/end changes the a posteriori AUC24 by more than 10% in a substantial fraction of courses (pre-registered threshold: >= 15% of courses); planted administration-time errors of SD 30 min inflate next-level rRMSE by a measurable amount that scales with the ratio of infusion duration to dosing interval.
3. **H3 (model averaging).** OFV-weighted model averaging has lower next-level rRMSE than every single model, with the largest gain in AKI, CRRT and obesity subgroups.
4. **H4 (equity).** Forecast rBias differs between subgroups by sex, race and BMI class after adjusting for renal function; switching to CKD-EPI 2021 changes the typical clearance prediction in Black patients by a quantifiable amount and shifts a priori target attainment.
5. **H5 (drift).** A priori target attainment (AUC24 400-600 mg*h/L) rises and the fraction of troughs > 20 mg/L falls across anchor-year groups; a prior refit on early years has higher rRMSE on late years than a prior refit on late years (temporal validation gap), and continuous learning closes most of it.
6. **H6 (exposure-response).** Day-2 Bayesian AUC24 predicts KDIGO stage >= 1 AKI within 7 days better than the first trough (higher AUROC and net benefit), and the effect estimate is attenuated when posterior uncertainty in AUC24 is propagated.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1, `hosp` module | `emar`, `emar_detail`, `prescriptions`, `labevents` (vancomycin, gentamicin, tobramycin, amikacin levels; creatinine), `patients`, `admissions`, `omr` (weight/height) | ~365k admissions; emar ~ 40M rows | PhysioNet credentialed (CITI "Data or Specimens Only Research" + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV v3.1, `icu` module | `inputevents` (vancomycin/aminoglycoside infusions, start/end/rate, patient weight), `chartevents` (weights), `procedureevents` (CRRT), `icustays` | ~94k ICU stays | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV Clinical Database Demo v2.2 | Pipeline smoke tests (100 patients, same schema) | ~100 MB | Open (no credentials) | https://physionet.org/content/mimic-iv-demo/2.2/ |
| eICU-CRD v2.0 (optional) | External check of subgroup findings (medication + lab vancomycin levels; coarse timing) | ~200k stays | PhysioNet credentialed | https://physionet.org/content/eicu-crd/2.0/ |
| Published popPK model library (curated by this project) | Model parameterisations transcribed from the original papers (typical values, omega, sigma, covariate equations) | ~10-15 models | Open (CSV/JSON in this repo, each entry with its source citation and a checker) | `data/models/` (to be populated) |

Candidate models for the library (parameters must be transcribed from the papers, never from memory; the starter code ships only a clearly labelled illustrative prior): the ICU and general adult vancomycin models evaluated by Broeker et al. (2019) and Guo et al. (2019), the pooled adult model of Colin et al. (2019, Clin Pharmacokinet), and the obese-patient models externally evaluated in 2023. For gentamicin, the adult extended-interval models used in Bayesian aminoglycoside tools.

## Methods

Pipeline (each step is a module in `src/bayes_pk/`):

1. **Cohort and courses** (`dosing_records.py`). Adults (>= 18), ICU stay with >= 1 IV vancomycin administration and >= 1 vancomycin level. Item ids are resolved by regex over `d_items` / `d_labitems` labels at run time and printed, so nothing is hard-coded. A *course* is a maximal run of doses with no gap > 72 h; each course carries its dose events `(start, end, amount_mg, source)` from `inputevents` (primary in ICU), `emar`+`emar_detail` (secondary; duration imputed from dose when `infusion_rate` is missing) and `prescriptions` (scheduled-time proxy). Levels carry `(charttime, value, label_trough)` where `label_trough` is inferred from the interval to the next dose start (<= 1 h) and from the storetime/charttime relationship. Weight uses the closest `inputevents.patientweight` / `chartevents` daily weight before the level. Creatinine, CRRT, vasopressors, KDIGO AKI (creatinine and urine-output criteria), age, sex, race and anchor-year group are joined per course.
2. **PK models** (`pk_models.py`). Exact superposition solutions for zero-order infusions into 1- and 2-compartment models (macro-constant form), with covariate models supplied as callables on a `ModelSpec` (typical values, `omega` as log-normal inter-individual variances, `sigma` as proportional + additive residual error). AUC over any interval by fine-grid integration of the analytic concentration.
3. **Bayesian forecasting** (`bayes_forecast.py`). MAP estimation of the individual random effects (the estimator that MIPD software uses) by L-BFGS-B on -2 log posterior, Laplace standard errors, a priori / a posteriori predictions, next-level forecasts, OFV-weighted model averaging (Uster et al., 2021). Full posteriors for a subset with Stan/Torsten or PyMC (optional; the MAP results are what a bedside tool would show).
4. **Continuous learning.** Refit typical values and omegas per anchor-year group with `nlmixr2` (R, SAEM/FOCEI) or `pharmpy`; compare priors from early vs late years on the held-out late years.
5. **Evaluation** (`evaluation.py`). rBias, rRMSE, MPE with patient-level bootstrap CIs; target attainment (AUC24 400-600 mg*h/L; trough 10-20 mg/L for historical comparison); planted timing-error experiment (`timing_perturbation_experiment`), CKD-EPI 2021 and Cockcroft-Gault helpers.
6. **Exposure-response.** Logistic/Cox for KDIGO AKI with day-2 AUC24 as exposure; regression calibration or a Bayesian errors-in-variables model that uses the per-course posterior SD of AUC24.

Tools: DuckDB, pandas/pyarrow, NumPy/SciPy, statsmodels; `nlmixr2`/`pharmpy` for population refits; Stan/Torsten or PyMC for full posteriors; scikit-learn for AKI models.

## Evaluation & statistics

- Unit of analysis: the course; CIs by cluster bootstrap over patients (a patient can have several courses).
- Forecast evaluation is strictly sequential: the forecast of level k uses doses and levels before level k only; covariates are the most recent values before the forecast time. No future creatinine.
- Temporal validation: fit priors on `anchor_year_group` 2008-2013, test on 2017-2022 (and reverse) to estimate drift; patient-level split within years for the non-temporal analyses.
- Model comparison: paired differences in rRMSE per course, mixed model with patient random effect; Benjamini-Hochberg across models and subgroups; pre-registered primary comparisons (H1-H3).
- Nulls and controls: (i) shuffling dose times within a course must destroy a posteriori gains (positive control for the pipeline); (ii) a simulation with the same dosing/sampling schedule as the real cohort and known parameters must recover the parameters (checks identifiability under TDM-style sampling); (iii) levels that are flagged as "drawn during the infusion" are analysed separately as a data-quality stratum.
- Informative sampling caveat: TDM levels are drawn more often in sicker patients and after abnormal results; we report results for first levels (least selected) and all levels separately and use inverse-probability-of-sampling weights as a sensitivity analysis.
- Exposure-response: AUROC, calibration and decision curves for AKI; E-values for unmeasured confounding.

## Publishable angle

Headline: "On a fully open ICU cohort, published vancomycin models disagree on AUC24 by more than 25% for one in three patients at the first level; the choice of dose-record source changes the estimate as much as the choice of model; model averaging and local refitting remove most of the disagreement, and forecasting error is not equal across sex, race and body-size subgroups." A second paper can be the benchmark itself (data-descriptor style with the model library and scoring code).

Target venues: Clinical Pharmacology & Therapeutics; CPT: Pharmacometrics & Systems Pharmacology; British Journal of Clinical Pharmacology; Antimicrobial Agents and Chemotherapy; Journal of Antimicrobial Chemotherapy; conferences PAGE and ACoP.

Follow-ups: extend to beta-lactams once MIMIC-IV-style level data exist; a reinforcement-learning dose-selection study built on the validated simulator; an eICU replication of the subgroup findings.

## Risks, confounds & mitigations

- **Level timing errors and mislabelled troughs.** Mitigation: infer trough status from the dose schedule rather than from the label; treat "level during infusion" as its own stratum; sensitivity analysis excluding levels within 30 min of a dose.
- **Three dose sources with different coverage** (`emar` only for admissions after the eMAR go-live). Mitigation: provenance is a study variable, not a nuisance; report coverage per year.
- **Weight sparsity and CRRT.** Mitigation: weight carried forward with a horizon; CRRT and ECMO as strata; models without a CRRT term are evaluated with CRRT patients excluded and included.
- **Informative TDM sampling.** Mitigation: as in the statistics section; discuss as a limitation of any real-world MIPD evaluation.
- **Parameter transcription errors in the model library.** Mitigation: each model entry has a checker that reproduces a published simulation (e.g. a typical-patient trough) within tolerance before it is used.
- **Race categories in MIMIC-IV are administrative.** Mitigation: report as recorded, do not collapse groups, and interpret the eGFR-equation analysis as a policy sensitivity analysis, not as biology.

## Milestones

- [ ] PhysioNet credentialing; download `hosp` + `icu`; DuckDB views; run the pipeline on the demo database.
- [ ] Dose/level/course tables with provenance and trough inference; manual validation of 100 courses against the raw tables.
- [ ] Model library (>= 8 vancomycin, >= 2 gentamicin models) with per-model checkers.
- [ ] MAP forecasting benchmark (a priori, after 1/2/3 levels) with rBias/rRMSE and AUC24 disagreement (H1).
- [ ] Provenance comparison and timing-perturbation experiment (H2).
- [ ] Model averaging and continuous learning across anchor-year groups (H3, H5).
- [ ] Subgroup and eGFR-equation analyses (H4).
- [ ] AKI exposure-response with uncertainty propagation (H6).
- [ ] Pre-registration (OSF) before H1-H6 analyses; manuscript; benchmark release with code and model library.

## Ethics / data-use notes

- MIMIC-IV and eICU-CRD are PhysioNet credentialed resources. Credentials are read from `PHYSIONET_USER` / `PHYSIONET_PASS` and are never written to disk by this code. Data live under `data/` which is git-ignored; never commit extracts, even aggregated tables that could re-identify patients.
- Per PhysioNet's responsible-use policy, credentialed data must not be sent to third-party LLM APIs or any external service. All modelling here is local (NumPy/SciPy/Stan/nlmixr2).
- Results are for research; the model library and scoring code are not a dosing tool and must not be used clinically.
- Report subgroup results with care: differences in forecasting error are properties of the models and the data-generating health system, not of patients.
