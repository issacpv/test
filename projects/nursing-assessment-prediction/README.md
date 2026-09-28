# nursing-assessment-prediction

**Nursing assessments as time-varying signals: hospital-acquired pressure-injury risk in MIMIC-IV ICU stays from Braden-scale trajectories and their observation process — a label-definition multiverse for a database without present-on-admission flags, dynamic landmark prediction versus the static admission Braden score, and treatment-aware evaluation (with an exploratory in-hospital fall arm).**

## Status / difficulty / timeline / compute

- Status: design + starter code (chartevents item resolution with label verification, Braden total reconstruction from subscales, ICD and nursing-documentation HAPI labels with a multiverse/agreement report, landmark dataset builder with leakage guards, landmark supermodel vs static baseline, dynamic AUROC / calibration / treatment-aware summaries). No data is shipped.
- Difficulty: MSc-level; 5-8 months. The hard parts are the label definitions and the informative-observation statistics, not the models.
- Compute: workstation with >= 32 GB RAM (MIMIC-IV `chartevents` is ~430M rows; filter by itemid with DuckDB before loading). No GPU.

## Background

Hospital-acquired pressure injuries (HAPI) cost the US system an estimated $26.8 billion per year (Padula & Delarmente, 2019 Int Wound J) and are a nursing-sensitive quality indicator. Risk is assessed with the Braden Scale (Bergstrom et al., 1987 Nurs Res; six subscales, total 6-23, higher = lower risk), re-scored every shift in most ICUs. Machine-learning HAPI models (Alderden et al., 2018 Am J Crit Care; Cramer et al., 2019 eGEMs; Song et al., 2021 JAMIA using nursing assessment phenotypes) typically outperform the static Braden score, but almost all use admission-time or aggregated features and single-centre labels.

MIMIC-IV records every Braden subscale entry in `icu.chartevents` (MetaVision items "Braden Sensory Perception", "Braden Moisture", "Braden Activity", "Braden Mobility", "Braden Nutrition", "Braden Friction/Shear"), pressure-injury skin documentation (site, stage), pressure-relief interventions (specialty surfaces, repositioning) and ICD-9/10 pressure-ulcer diagnosis codes — but, crucially, no present-on-admission (POA) indicator. Recent MIMIC-IV papers have used the Braden score mainly as a mortality predictor (several 2025 Sci Rep papers on short-term mortality and on ARDS in elderly sepsis) or built static PI models with ICD labels (a 2025 J Clin Nurs interpretable model on 1,774 ICU PI patients; a 2025 SHAP-XGBoost model in ventilated patients, PMC11928553). A 2024 JMIR Med Inform paper ("Implementable prediction of pressure injuries in ICU", e51842) emphasised deployability, and a 2023 JMIR Med Inform comparative study showed that HAPI definitions built from ICD codes, note keywords and stage thresholds change the identified cohort substantially.

## The research gap

What has been done (2023-2026):

- Braden as a prognostic marker for mortality / ARDS on MIMIC-IV: saturated; not our target.
- Static ML HAPI models on MIMIC-IV with ICD-code labels: several; none address POA ambiguity, none use Braden as a trajectory, none model the assessment frequency, none evaluate the effect of documented prevention on the label.
- The 2023 JMIR HAPI-definition study quantified definition sensitivity on MIMIC-III (ICD-9 + notes); it did not propagate the definitions into prediction models or into MIMIC-IV's structured skin assessments.
- Dynamic (landmark) prediction is established in survival methodology (van Houwelingen & Putter, 2012) and used for ICU deterioration, but not for HAPI with nursing-assessment trajectories on open data.
- The self-fulfilling-prophecy problem for prognostic models under treatment (Sperrin et al., 2019 JAMIA; Lenert et al., 2019 JAMIA) is directly relevant: a low Braden score triggers prevention, which lowers the outcome, which makes Braden look weak.

What is specifically missing (our angle):

1. **A HAPI label multiverse for MIMIC-IV** (ICD any / ICD stage >= 2 / nursing-documented first PI after 24 h / after 48 h / stage >= 2 / ICD AND nursing), with prevalence, pairwise agreement (kappa, Jaccard) and the fraction of ICD-coded PIs that are documented within the first 24 h (i.e. probably present on admission) — an audit useful to everyone who has used MIMIC-IV for PI work.
2. **Dynamic landmark prediction** from the Braden trajectory (last, minimum, 48-h slope), subscales, and the **observation process** (assessments per 24 h, hours since last assessment) at daily landmarks, against the static admission-Braden baseline and against static ML models — with time-dependent AUROC and calibration by landmark day.
3. **Treatment-aware evaluation**: documented pressure-relief surfaces and repositioning as time-varying covariates; performance stratified by prevention status; a target-trial-style estimate of the association between documented prevention after a low Braden score and subsequent HAPI, to interpret model performance under intervention.
4. **Informative observation**: whether assessment frequency itself predicts HAPI beyond the Braden values (nurses assess more often when worried), and whether models that use it degrade when frequency is randomised (a robustness/null test for deployment).
5. **Exploratory fall arm**: whether a Morse Fall Scale (or any fall-risk item) exists in MIMIC-IV `d_items` (to be verified; not guaranteed), and in-hospital falls from ICD-10 W00-W19 with place-of-occurrence Y92.23 (hospital) as a weak label, with the same multiverse logic.

Related project in this repo: none directly; the label-multiverse pattern parallels `icu-model-transportability`'s ontology verification. This project is self-contained.

## Research questions / hypotheses

1. **RQ1 (labels).** H1: HAPI prevalence among adult ICU stays ranges by >= 3x across definitions; >= 30% of ICD-coded pressure-ulcer stays have skin documentation of a PI within the first 24 h (probable POA); ICD-vs-nursing agreement is only moderate (kappa 0.4-0.6).
2. **RQ2 (dynamic vs static).** H2: at landmark days 2-7, the landmark supermodel with Braden trajectory + observation-process features has AUROC >= 0.05 higher than the admission Braden total for HAPI within 72 h (nursing-documented, hospital-acquired definition), with better calibration slope (closer to 1).
3. **RQ3 (observation process).** H3: assessments per 24 h and hours since last assessment carry independent information (likelihood-ratio test p < 0.001) after conditioning on Braden values; randomising them within stay reduces AUROC by <= 0.02 (i.e. the gain in H2 is not mainly an observation-process artefact) — this is a two-sided pre-specified test.
4. **RQ4 (treatment-aware).** H4: among landmarks with Braden <= 12, documented prevention within the next 24 h is associated with lower 72-h HAPI (adjusted OR < 0.8); model discrimination is lower in the prevented subgroup (an expected consequence, not a model failure).
5. **RQ5 (subscale structure).** H5: mobility and moisture subscales dominate the dynamic signal; sensory perception adds little once sedation (RASS) is included.
6. **RQ6 (falls, exploratory).** If a fall-risk item exists: same analysis for in-hospital falls; otherwise report the absence as a data note and the ICD-based fall prevalence.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (icu module) | chartevents (Braden subscales, skin/PI documentation, pressure-relief devices, repositioning, RASS), d_items, icustays | ~94k ICU stays; chartevents ~430M rows | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV v3.1 (hosp module) | diagnoses_icd (L89.*, 707.*; W00-W19, Y92.23), admissions, patients | ~546k admissions | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV Clinical Database Demo v2.2 | Dry run with real item labels (`d_items`) and 100 patients | 100 patients | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| eICU-CRD v2.0 (optional) | External check of label/feature availability (`nursecharting`, `nurseassessment`) if a Braden-like item exists | ~200k stays | PhysioNet credentialed | https://physionet.org/content/eicu-crd/2.0/ |
| mimic-code (MIT-LCP) | Reference for item handling conventions (e.g. GCS component carry-forward) | code | Open | https://github.com/MIT-LCP/mimic-code |

Item ids are resolved from `d_items` by label regex at run time and verified (`items.py`), never hard-coded silently; the expected MetaVision ids for the six Braden subscales fall in the 224054-224059 range and are confirmed on the demo before use.

## Methods

Pipeline (modules in `src/nursing_risk/`):

1. **Item resolution** (`items.py`). Regex concepts: `braden_sensory`, `braden_moisture`, `braden_activity`, `braden_mobility`, `braden_nutrition`, `braden_friction`, `skin_pi_stage` (labels containing "pressure ulcer"/"pressure injury"/"impaired skin" and "stage"), `skin_pi_site`, `pressure_relief` ("pressure reducing device", "specialty bed", "air mattress"), `reposition` ("turn", "reposition"), `rass`, `fall_risk` (any label containing "fall"). `verify_expected()` refuses ids whose label does not match.
2. **Braden reconstruction** (`items.py`). Wide table per (stay_id, charttime) of the six subscales; components carried forward within a 1-h window (mirrors the mimic-code GCS convention); `braden_total` only when all six are present; per-stay assessment frequency.
3. **Labels** (`labels.py`). ICD-10 L89 (stage digit: 0 unstageable, 1-4 stages, 6 deep-tissue, 9 unspecified) and ICD-9 707.0x/707.2x; nursing-documented first PI time from `skin_pi_stage` values (stage >= 1, or >= 2), hospital-acquired if first documentation > 24 h (sensitivity 48 h) after hospital admission and no documentation before; exclusion of stays with prior-admission PI codes. `label_multiverse()` reports prevalence, kappa/Jaccard and the probable-POA fraction.
4. **Landmark dataset** (`dynamic.py`). Landmarks at 24, 48, ... h after ICU admission up to day 14 while the stay is ongoing and no PI has occurred. Features strictly from before the landmark: last Braden total, minimum so far, mean over 48 h, 48-h slope, subscales (last), assessments per 24 h, hours since last, prevention documented in the last 24 h, RASS last, plus static covariates (age, sex, admission type, ventilation flag at landmark). Outcome: nursing-documented HAPI within 72 h (sensitivity 7 d); administrative censoring (discharge before horizon) flagged.
5. **Models** (`dynamic.py`). Landmark supermodel: pooled logistic regression with landmark-day main effects and feature x landmark-day interactions (van Houwelingen & Putter, 2012), L2-regularised; gradient-boosted alternative (LightGBM) with landmark day as a feature. Static baselines: admission Braden total alone; admission-time ML model.
6. **Evaluation** (`evaluate.py`). Dynamic AUROC and AUPRC per landmark day with stay-level cluster bootstrap; calibration intercept/slope per landmark; decision curves at 5-20% thresholds; observation-process null (permute assessment-frequency features within stay); treatment-aware stratification and the adjusted OR for prevention after low Braden.

Tools: DuckDB (chartevents filtering by itemid), pandas, scikit-learn, LightGBM, statsmodels.

## Evaluation & statistics

- Splits: patient-level 70/15/15 (train/validation/test); all landmarks of a patient in one split; hyper-parameters on validation only; test used once.
- Leakage guards (tested): features use only charttime <= landmark; the outcome window starts strictly after the landmark; stays with PI before the landmark are removed from the risk set; skin-documentation items used to define the label are excluded from the features.
- Metrics: time-dependent AUROC/AUPRC per landmark, pooled AUROC with landmark-stratified bootstrap, calibration intercept/slope, ECE, net benefit; CIs from 1000 stay-level bootstrap resamples.
- Multiple comparisons: primary hypotheses H1-H3 (Holm); landmark-day-specific comparisons BH-FDR.
- Nulls: (i) within-stay permutation of assessment-frequency features (H3); (ii) label-permutation null for the supermodel (AUROC ~ 0.5); (iii) negative-control outcome: a same-horizon outcome unrelated to skin care (e.g. new hyperkalaemia) should not be predicted by Braden trajectory beyond severity covariates.
- Sensitivity: 24 h vs 48 h hospital-acquired threshold; stage >= 1 vs >= 2; ICD vs nursing labels; 72 h vs 7 d horizon; excluding stays < 48 h.

## Publishable angle

Headline result: "HAPI prevalence in MIMIC-IV ICU stays varies 3-fold with the label definition and a third of ICD-coded cases are probably present on admission; a landmark model using the Braden trajectory and how often nurses reassess predicts 72-hour HAPI with AUROC ~0.8 at every ICU day versus ~0.7 for the admission Braden score, and the improvement survives a permutation of the observation process; documented prevention after low scores halves subsequent HAPI, which explains why static Braden looks weak." The label audit alone is a citable data note.

Target venues: Journal of the American Medical Informatics Association; International Journal of Nursing Studies; Journal of Clinical Nursing (methods/informatics); Critical Care Medicine (research letter on the POA audit); ML4H / CHIL (dynamic prediction + treatment-aware evaluation).

Follow-ups: (a) external validation on eICU if a Braden-like item is found; (b) nurse-facing dashboard prototype with the landmark model; (c) extend the multiverse/landmark framework to falls once a fall-risk item and a fall event source are confirmed.

## Risks, confounds & mitigations

- **Label validity**: nursing skin documentation may lag or be incomplete; ICD coding under-captures stage 1. Mitigation: the multiverse is the primary product; report each definition; manual review of 100 charts via MIMIC-IV-Note (local only) for agreement with documentation.
- **No POA flag**: 24-h proxy is imperfect. Mitigation: 48-h sensitivity; prior-admission exclusion; compare with `admissions.admission_location` (transfers from other hospitals more likely POA).
- **Treatment paradox**: prevention lowers outcomes among high-risk patients. Mitigation: treatment-aware stratification; report performance with and without prevention features; do not claim causal effects beyond the target-trial-style estimate with its stated assumptions.
- **Informative observation**: assessment frequency may proxy nurse concern. Mitigation: explicit test (H3) and the permutation null; deployment note.
- **Item drift across MetaVision versions / MIMIC-IV releases**: verify ids on each release; regex resolution + verification.
- **Fall arm may be infeasible** (no Morse in ICU chartevents). Mitigation: pre-declared as exploratory; absence reported as a finding.
- **chartevents size**: filter by resolved itemids with DuckDB before pandas.

## Milestones

- [ ] PhysioNet credentialing; dry run on the MIMIC-IV demo (`scripts/download_data.py --sample`) including item resolution on the demo `d_items`.
- [ ] DuckDB extraction of Braden, skin, prevention, RASS items; Braden reconstruction QC (distribution 6-23, per-shift frequency).
- [ ] Label multiverse table with prevalence, kappa/Jaccard, probable-POA fraction (RQ1).
- [ ] Landmark dataset with leakage tests; descriptive trajectories before HAPI.
- [ ] Supermodel vs static baselines; per-landmark AUROC/calibration (RQ2); observation-process tests (RQ3).
- [ ] Treatment-aware evaluation and prevention OR (RQ4); subscale analysis (RQ5).
- [ ] Fall feasibility check (RQ6); sensitivity analyses.
- [ ] Pre-registration, manuscript, code release without data.

## Ethics / data-use notes

- MIMIC-IV (and eICU if used) are PhysioNet credentialed (CITI + DUA); never redistribute data; `data/` is git-ignored; credentials via `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`.
- Any chart review uses MIMIC-IV-Note locally; no text is sent to third-party LLM or cloud APIs (PhysioNet responsible-use policy).
- HAPI is a nursing-sensitive indicator; results must be framed as system-level and must not be used to evaluate individual nurses or units (unit identifiers are not analysed).
- Report cells >= 11; no patient-level trajectories in figures without aggregation.
