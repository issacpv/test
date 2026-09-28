# ED-Triage: mis-triage equity audit and text-augmented, temporally validated triage prediction on MIMIC-IV-ED

**One-sentence pitch.** Reproduce the MIMIC-IV-ED triage benchmark (Xie et al., 2022), then use it to (a) audit *outcome-anchored* ESI mis-triage (under- and over-triage) by race, sex, age, language and insurance, (b) test whether locally-run clinical language models on free-text chief complaints reduce mis-triage for the groups most affected, and (c) measure how both the nurse-assigned ESI and the models drift across the 2008-2022 `anchor_year_group` eras.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data shipped; all data are PhysioNet-credentialed.
- Difficulty: MSc-level (clinical informatics + ML + biostatistics); the equity audit alone is a self-contained paper.
- Timeline: 6-8 months (1 month credentialing + cohort reproduction, 2 months mis-triage audit, 2 months text models + temporal validation, remainder writing).
- Compute: CPU workstation with 32 GB RAM and DuckDB for the tabular pipeline (MIMIC-IV-ED is ~ 450k stays; the largest joined table, `vitalsign`, is ~1.6M rows). One 12-24 GB GPU for fine-tuning a small clinical transformer (BioClinicalBERT / Clinical-Longformer / a 1-4B open LLM in inference mode) on ~450k short chief-complaint strings. Everything runs *locally*: PhysioNet's policy prohibits sending MIMIC text to external LLM APIs.

## Background

ED triage assigns an Emergency Severity Index (ESI 1-5) in minutes, using vitals, chief complaint and expected resource use. In a 5-million-encounter Kaiser Permanente cohort, Sax et al. (2023, JAMA Netw Open) found ~one-third of encounters mis-triaged under ESI v4, with disparities by race/ethnicity, sex, age and neighbourhood poverty. MIMIC-IV-ED (Johnson et al., 2023, PhysioNet; Johnson et al., 2023, Sci Data for MIMIC-IV) is the only large public ED dataset with triage vitals, free-text chief complaint, ESI, medication reconciliation, dispensing (pyxis), and linkage to hospital/ICU outcomes and to discharge/radiology notes (MIMIC-IV-Note). Xie et al. (2022, Sci Data) built the reference benchmark on it (hospitalisation, critical outcome = ICU transfer or death within 12 h, 72-h ED reattendance; 441,437 adult visits) and released code (nliulab/mimic4ed-benchmark).

## The research gap

**What has been done (2022-2026):**

- Xie et al. (2022, Sci Data): benchmark with structured features; logistic regression, gradient boosting, MLP; random split; no text, no fairness, no temporal validation.
- PREDICT-ED (GitHub Doclecodeur/predict_ed, 2025): hospital-admission prediction with structured triage variables + TF-IDF chief complaint and a temporal validation on the 2017-2019 era (AUROC 0.888) - shows the value of text and temporal splits but only for admission, without mis-triage or equity analyses.
- A 2025 medRxiv preprint (doi:10.1101/2025.07.22.25332000) trains XGBoost on MIMIC-IV triage data for ICU admission/death with an external validation.
- LLM triage studies: "From promising capability to pervasive bias" (arXiv:2504.16273, 2025) and EQUITRIAGE (arXiv:2605.03998, 2026) build vignettes from MIMIC-IV-ED to audit gender/race bias of *LLM-assigned* ESI; Williams et al. (2024, JAMA Netw Open) and Lafuente & Rahim (2025) study LLM acuity assessment. These audit the *model*, not the *nurses' ESI against outcomes*, and use API-based LLMs on derived vignettes.
- "Fairness in healthcare processes: a quantitative analysis of decision making in triage" (arXiv:2601.11065, 2026) uses MIMIC-IV-ED to test whether time-to-triage, re-triage and deviation differ by race, gender, age, language and insurance per ESI level - a process-fairness view, not outcome-anchored mis-triage.
- MIMIC-IV-Ext-MDS-ED (Alcaraz & Strodthoff, 2024, PhysioNet) adds ECG waveforms to ED decision support benchmarks; the MIMIC-IV-Ext Triage Instruction Corpus (PhysioNet) provides instruction-tuning data.
- Temporal drift: MIMIC-IV `anchor_year_group` has been used to show vital-sign distribution shifts between 2008-2010 and 2020-2022, but not for ED triage models or for ESI itself.

**What is missing (the gap this project fills):**

1. No **outcome-anchored mis-triage audit** on MIMIC-IV-ED: under-triage (ESI 3-5 followed by a critical outcome) and over-triage (ESI 1-2 without admission, ICU or procedures) rates by race, sex, age, *language* (English vs non-English) and insurance, with adjustment for vitals and chief-complaint category. Sax et al. could not release data; MIMIC allows a fully reproducible audit.
2. No study asks whether a **text-augmented, locally-run model** narrows the *subgroup-specific* under-triage gap, or whether it inherits the nurses' pattern (the ESI itself is a label that encodes those disparities; the outcome labels do not).
3. No **temporal drift** analysis of mis-triage or of triage models across the four MIMIC eras, although the benchmark mixes 2011-2022 data in random splits.
4. No published **decision-curve / calibration** comparison of ESI vs models at deployable operating points (e.g. matched over-triage rate), which is what determines whether a model could safely re-prioritise the ESI-3 pool (60 % of visits in Sax et al.).

## Research questions / hypotheses

1. **RQ1.** What are the under- and over-triage rates in MIMIC-IV-ED under Sax-style outcome criteria, overall and by ESI level? *H1:* under-triage of critical outcomes is concentrated in ESI 3 (>= 60 % of under-triaged critical outcomes).
2. **RQ2.** Do under-triage odds differ by race/ethnicity, sex, age, language and insurance after adjusting for triage vitals, pain score, arrival mode and chief-complaint category? *H2:* non-English-speaking and Black patients have higher adjusted odds of under-triage (OR > 1.15) and lower odds of over-triage.
3. **RQ3.** Does adding chief-complaint text (TF-IDF, then a locally fine-tuned clinical transformer) to the structured benchmark model reduce the *subgroup* under-triage rate at a fixed overall over-triage rate matched to nurses' ESI? *H3:* text reduces the absolute under-triage gap between language groups by >= 30 %.
4. **RQ4.** How much do AUROC, calibration slope and net benefit degrade when training on 2011-2016 eras and testing on 2017-2019 and 2020-2022? *H4:* discrimination is stable (Delta AUROC < 0.02) but calibration intercept drifts (> 0.1 logits) with prevalence changes; ESI under-triage rates also change across eras.
5. **RQ5.** Is model mis-triage less unequal than nurse mis-triage at the same operating point (equalised-odds gaps)? *H5:* models trained on outcomes (not ESI) have smaller TPR gaps across language groups than ESI.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV-ED v2.2 | `edstays`, `triage` (vitals, pain, acuity, chiefcomplaint), `vitalsign`, `medrecon`, `pyxis`, `diagnosis` | 425,087 ED stays (v2.2), 2011-2022, Beth Israel Deaconess | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimic-iv-ed/2.2/ |
| MIMIC-IV v3.1 (hosp, icu) | `patients` (anchor_age, anchor_year_group, dod), `admissions` (language, insurance, race, deathtime), `transfers`, `icustays`, `procedures_icd`, `labevents` | ~365k patients | Credentialed (PhysioNet) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV-Note v2.2 | `discharge` and `radiology` notes for outcome adjudication (procedures, critical care) and optional weak labels | 331k discharge, 2.3M radiology reports | Credentialed (PhysioNet) | https://physionet.org/content/mimic-iv-note/2.2/ |
| MIMIC-IV-ED Demo v2.2 | Pipeline smoke tests (100 patients) | 100 patients | Open | https://physionet.org/content/mimic-iv-ed-demo/2.2/ |
| MIMIC-IV-Ext-MDS-ED (optional) | Cross-check of critical-outcome definitions | derived | Credentialed | https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/ |

Note: `language` and `insurance` live in `admissions` (available for visits with a hospital admission and, via any prior/subsequent admission of the same subject, for most others); `race` is in `edstays`. Missingness by group is itself reported.

## Methods

1. **Cohort builders** (`ed_triage.cohort`): DuckDB SQL over the CSV.gz files (no database server). Reproduce Xie et al.: adults (age at ED arrival >= 18 computed from `anchor_age` + year offset), one row per `stay_id`, outcomes: `hospitalization` (admitted disposition with `hadm_id`), `critical_outcome` (ICU transfer within 12 h of ED departure or in-hospital death within 12 h), `revisit_72h` (next ED stay within 72 h of `outtime`). Extensions: `resource_count` proxies from `pyxis` (medications) and `labevents`/`procedures_icd`; `nonenglish` from `admissions.language`. A pandas implementation of the same logic is included for tests and for the open demo.
2. **Chief-complaint features** (`ed_triage.text_features`): normalisation (lower-casing, punctuation, abbreviation expansion, e.g. "sob" -> "shortness of breath", "cp" -> "chest pain"), splitting of multi-complaint strings, TF-IDF (word 1-2 grams + char 3-5 grams) and an optional local transformer embedding (`emilyalsentzer/Bio_ClinicalBERT` or any HF checkpoint loaded from disk; no network calls at inference).
3. **Mis-triage definitions** (`ed_triage.mistriage`): following Sax et al. (2023): under-triage = ESI 3-5 and critical outcome (or >= 2 resources + ICU admission proxy); over-triage = ESI 1-2 and discharged home without ICU, procedures or death. Equity metrics: rates with Wilson CIs, crude and adjusted odds ratios (numpy IRLS logistic regression with Wald CIs; `statsmodels` optional), equalised-odds gaps at a matched operating point, and a *model-vs-nurse* comparison where the model's threshold is chosen to match the nurses' overall over-triage rate.
4. **Models**: logistic regression, LightGBM/HistGradientBoosting on structured features (Xie replication); + TF-IDF text; + transformer embeddings; fine-tuned transformer with structured features concatenated. Sex, race, language are *not* model inputs (audit variables only); a sensitivity analysis includes them.
5. **Temporal validation** (`ed_triage.validation`): era splits by `anchor_year_group` (2008-2010, 2011-2013, 2014-2016, 2017-2019, 2020-2022); rolling-origin evaluation; feature drift (PSI, KS) and outcome-prevalence drift; recalibration (Platt / isotonic on the first 3 months of the target era) as a mitigation arm.
6. **Calibration and decision curves** (`ed_triage.validation`): calibration slope/intercept, ECE, net benefit vs threshold probability for ESI-as-classifier and for models.

Tools: `duckdb`, `pandas`, `scikit-learn`, `lightgbm` (optional), `transformers` + `torch` (optional, local), `statsmodels` (optional).

### Outcome and mis-triage definitions (as implemented)

| Variable | Definition | Table(s) |
|---|---|---|
| `hospitalization` | `edstays.disposition = 'ADMITTED'` and `hadm_id` not null | edstays |
| `critical_outcome` | `icustays.intime` within 12 h after `edstays.outtime`, or `admissions.deathtime` within 12 h after `outtime` | edstays, icustays, admissions |
| `revisit_72h` | next `edstays.intime` of the same `subject_id` <= `outtime` + 72 h | edstays |
| `age` | `anchor_age + (year(intime) - anchor_year)`; adults >= 18 | patients |
| `era` | `anchor_year_group` (2008-2010 ... 2020-2022) | patients |
| `nonenglish` | `admissions.language != 'ENGLISH'` (visit's admission, else subject's latest admission); missing kept as NaN | admissions |
| `race6` | White / Black / Hispanic / Asian / Other / Unknown collapsed from `edstays.race` | edstays |
| `n_meds_pyxis` | count of `pyxis` dispensing rows per `stay_id` (resource proxy) | pyxis |
| `under_triage` | `acuity` in {3,4,5} and `critical_outcome = 1` (variant: or `acuity` in {4,5} and admitted) | derived |
| `over_triage` | `acuity` in {1,2} and not admitted and no critical outcome | derived |
| eligible denominators | under: `critical_outcome = 1`; over: not admitted and no critical outcome | derived |

### Analysis arms

| Arm | Train | Test | Text | Output |
|---|---|---|---|---|
| A0 replication | random 70 % patients | random 30 % | none | AUROC vs Xie et al. Table |
| A1 audit | - | all adult visits | complaint category only (covariate) | mis-triage rates, adjusted ORs, E-values |
| A2 text | eras <= 2016 | 2017-2019 (val), 2020-2022 (test) | TF-IDF; local transformer | AUROC, calibration, subgroup under-triage at matched over-triage |
| A3 drift | rolling origin over eras | next era | as A2 | AUROC/calibration decay, PSI, prevalence drift |
| A4 fairness | as A2 | as A2 | as A2 | equalised-odds gaps: nurse ESI vs models |

## Evaluation & statistics

- Discrimination: AUROC, AUPRC with 95 % bootstrap CIs (patient-clustered, 2,000 resamples). Calibration: slope, intercept, ECE, reliability plots. Utility: net benefit at threshold probabilities 5-30 % for critical outcome.
- Mis-triage audit: adjusted logistic models per outcome (under-triage; over-triage) with covariates age, sex, race, language, insurance, arrival transport, vitals, pain, chief-complaint category (top-50 TF-IDF clusters), era; cluster-robust SEs by `subject_id`. E-values for unmeasured confounding on the main ORs.
- Subgroup fairness: TPR/FPR gaps (equalised odds) and calibration-in-the-large per group; report both nurse-ESI and each model at the *same* overall over-triage rate.
- Validation scheme: (i) Xie-style random patient-level 70/30 for replication only; (ii) primary: temporal - train <= 2016 eras, validate 2017-2019, test 2020-2022; (iii) leave-one-era-out for drift curves. Patients never straddle splits.
- Leakage prevention: text vectorisers and any imputers fitted on training era only; outcomes defined from tables timestamped after ED departure; no discharge-note features used for prediction (notes only for outcome adjudication).
- Multiple comparisons: Holm correction within each family (5 demographic axes x 2 mis-triage outcomes); pre-registered primary contrasts (language, race) with the rest exploratory.
- Nulls: permutation of group labels within ESI strata to obtain the null distribution of the gap statistics; a model trained on ESI instead of outcomes to show inherited bias.

## Publishable angle

- Headline: "In 400k+ public ED visits, X % of critical outcomes were under-triaged, disproportionately among non-English-speaking patients; a locally-run chief-complaint language model reduced the under-triage gap by Y % at the nurses' over-triage rate, and its calibration - but not its discrimination - drifted across the 2011-2022 eras."
- Deliverables: reproducible DuckDB cohort SQL, mis-triage definitions as code, temporal benchmark splits, and a fairness report card for the benchmark.
- Venues: *Annals of Emergency Medicine*; *JAMA Network Open*; *Journal of the American Medical Informatics Association*; *npj Digital Medicine*; ML4H / CHIL for the benchmark extension.
- Follow-ups: adding ECG waveforms (MIMIC-IV-Ext-MDS-ED) for chest-pain sub-cohorts; pediatric contrast with other datasets; prospective silent-mode evaluation with a partner ED; interpretability of which complaint phrasings drive under-triage.

## Risks, confounds & mitigations

- **ESI encodes expected resource use, not only acuity** (especially levels 3-5). Mitigation: report both critical-outcome-anchored and resource-anchored under-triage; use Sax et al. definitions for comparability.
- **Language/insurance missing for non-admitted visits.** Mitigation: propagate from other admissions of the same patient; report missingness by group; multiple imputation sensitivity.
- **Outcome ascertainment differs by era** (v2.2 covers 2011-2022; ICU transfer definitions depend on `transfers.careunit`). Mitigation: fixed careunit list; era-stratified checks.
- **Race categories in MIMIC are free-text-derived and inconsistent.** Mitigation: collapse to the standard 6-category scheme used in MIMIC papers; sensitivity with finer categories.
- **Date shifting**: only `anchor_year_group` is real-time-anchored; within-era ordering is not recoverable. Mitigation: treat eras as blocks; do not claim month-level drift.
- **Text models may memorise identifiers.** Mitigation: chief complaints are short and de-identified; still, weights trained on MIMIC are not redistributed.

## Milestones

- [ ] PhysioNet credentialing; download MIMIC-IV-ED, MIMIC-IV hosp/icu, MIMIC-IV-Note; run `scripts/download_data.py --verify`.
- [ ] Run the pipeline on the open MIMIC-IV-ED demo (`--sample`) end-to-end.
- [ ] Reproduce Xie et al. cohort counts and baseline AUROCs (report a comparison table).
- [ ] Mis-triage definitions frozen; audit tables and adjusted ORs; E-values.
- [ ] TF-IDF and local transformer text features; matched-operating-point comparison.
- [ ] Temporal validation and drift report; recalibration arm.
- [ ] Decision curves; fairness report card; manuscript.

## Ethics / data-use notes

- All MIMIC modules are credentialed: CITI training + DUA per user; store data encrypted; never commit data, notes or derived row-level tables.
- PhysioNet's responsible-use policy prohibits sharing MIMIC data with third-party services, including sending notes or chief complaints to external LLM APIs. All language models in this project run locally on institution-controlled hardware; the code contains no API clients.
- Fairness variables (race, language, insurance) are used only for auditing and are excluded from the primary models to avoid encoding disparities.
- Findings on mis-triage should be framed at the system level (triage protocol, staffing, interpreter access), not as individual-nurse performance.
