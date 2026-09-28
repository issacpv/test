# ed-language-disparity

**Language, documentation and triage: does the information captured at ED triage mediate outcome and process disparities for non-English-preferring patients? A MIMIC-IV-ED study using the granular MIMIC-IV v3 language field, chief-complaint informativeness, under-triage calibration of the ESI, note-derived interpreter mentions, and triage-model fairness by preferred language.**

## Status / difficulty / timeline / compute

- Status: design + starter code (language/race harmonisation and visit table, chief-complaint text features and interpreter-mention regex, adjusted odds, under-triage calibration by group, exact matching, mediation with bootstrap, triage-model fairness metrics). No data is shipped.
- Difficulty: MSc-level (epidemiology + ML); 5-8 months. The hard parts are the causal framing (language vs ethnicity vs insurance) and the propagation of a hospital-admission-level language field to ED visits.
- Compute: laptop/workstation; MIMIC-IV-ED is ~425k ED stays and fits in pandas/DuckDB. No GPU needed (gradient-boosted trees; optional local text embeddings).

## Background

Patients with limited English proficiency (LEP) receive lower-quality care unless professional interpreters are used (Karliner et al., 2007 Health Serv Res; Diamond et al., 2019 J Gen Intern Med). In the ED, LEP is associated with more unplanned 72-h revisits (Ngai et al., 2016 Ann Emerg Med), different admission and testing patterns (Schulson et al., 2018 J Gen Intern Med), and, in a 2025/2026 stroke cohort, longer door-to-groin-puncture times (PMC12802779). ED triage with the Emergency Severity Index (ESI; Gilboy et al., AHRQ handbook) relies on a brief history — the chief complaint — that is precisely what a language barrier degrades. The mechanism most often invoked ("less information at triage leads to mis-triage and delay") has never been measured on open data.

MIMIC-IV-ED (Johnson et al., PhysioNet v2.2) holds ~425k ED visits (2011-2019) with triage vitals, ESI acuity, free-text chief complaint, medication administration (pyxis), disposition and timestamps, and links to MIMIC-IV hospital admissions and MIMIC-IV-Note. MIMIC-IV v3.0 (July 2024) replaced the old `ENGLISH` / `?` language field with a standardised primary language for non-English patients, so, for the first time, language-specific groups (e.g. Spanish, Portuguese, Chinese languages, Russian, Haitian Creole, Vietnamese, Arabic; verify the exact value set in `admissions.language`) can be analysed rather than a binary flag.

## The research gap

What has been done (2023-2026):

- A 2025 arXiv study (arXiv:2503.22781) used MIMIC-IV-ED to associate demographic factors (including language) with triage acuity, admission and length of stay by regression. It used the binary English / non-English field, reported associations only, and did not examine mechanisms.
- Xie et al. (2022 Sci Data) published the MIMIC-IV-ED benchmark (hospitalisation, critical outcome, 72-h reattendance) with fairness discussed by race/ethnicity but not language.
- Several single-centre studies (2024-2026, e.g. a 2026 J Immigrant Minority Health paper on language-based disparities in ED referral) report LEP effects on referral, admission and waiting, without open data or documentation-quality measures.
- Interpreter use is never captured as structured data in MIMIC; its mention in discharge notes has not been exploited.

What is specifically missing (our angle):

1. **Granular language groups** (MIMIC-IV >= 3.0) instead of a binary flag: are disparities uniform across languages, or concentrated in languages with fewer in-house interpreters?
2. **Documentation informativeness as a measurable mediator**: chief-complaint length, number of complaints, vagueness, explicit language-barrier markers ("language barrier", "per family", "unable to obtain history"), missing pain score and missing vitals at triage, by language.
3. **Under-triage calibration of the ESI by language**: within each ESI level, the rate of critical outcome (ICU transfer within 12 h or in-hospital death; Xie et al., 2022 definition) by language group. A higher critical-outcome rate at the same ESI level is direct evidence of under-triage; the discrimination (AUROC) of ESI for critical outcome per group measures how much information the triage nurse could act on.
4. **Language separated from ethnicity and insurance**: contrasts within race/ethnicity strata (e.g. Spanish-preferring vs English-preferring Hispanic/Latino patients) and exact matching on ESI, age, sex, arrival mode, race group and insurance.
5. **Note-derived interpreter mentions** (local regex on MIMIC-IV-Note discharge summaries) as a proxy exposure modifier: are disparities smaller in visits whose admission notes document an interpreter?
6. **Triage-model fairness by language**: do the benchmark ML models (with and without chief-complaint text features) have worse calibration or lower sensitivity for non-English-preferring patients, and does adding text features widen or narrow the gap?

Related project in this repo: `ed-triage-mimic-ed` (triage prediction on MIMIC-IV-ED). This project is independent and focuses on the language axis; its fairness module can evaluate any triage model's predictions.

## Research questions / hypotheses

1. **RQ1 (process and outcomes).** Adjusted for age, sex, ESI, arrival mode, race group, insurance, hour/weekday, and chief-complaint cluster: H1a non-English-preferring (NEP) patients have higher odds of admission at ESI 3-5 (OR > 1.1) but not at ESI 1-2; H1b longer ED LOS (>= 20 min adjusted median difference); H1c longer time to first analgesic among pain complaints; H1d higher 72-h return odds.
2. **RQ2 (documentation).** H2: chief complaints for NEP visits are shorter (fewer tokens), have fewer listed complaints, more vagueness and more language-barrier markers; pain score is more often missing. Effects present in every language group but largest for languages other than Spanish.
3. **RQ3 (under-triage).** H3a: at ESI 3, the critical-outcome rate is higher for NEP than for English-preferring patients (risk ratio > 1.2); H3b: the AUROC of ESI for critical outcome is lower in NEP visits by >= 0.02.
4. **RQ4 (mediation).** H4: chief-complaint informativeness and missing pain score mediate >= 20% of the language effect on under-triage (H3a) and on time to first medication.
5. **RQ5 (within-ethnicity).** H5: the language effects in H1 and H3 persist within the Hispanic/Latino stratum (Spanish vs English preferring), i.e. they are not explained by ethnicity.
6. **RQ6 (interpreter mentions).** H6: among admitted NEP patients, visits whose discharge summary mentions an interpreter have smaller LOS and under-triage gaps than those without.
7. **RQ7 (model fairness).** H7: benchmark triage models have lower sensitivity at a fixed 10% alert rate for NEP visits; adding chief-complaint text features widens the gap because the text is less informative for NEP patients.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV-ED v2.2 | edstays (intime, outtime, disposition, arrival_transport, race, gender), triage (vitals, pain, acuity, chiefcomplaint), pyxis, medrecon, vitalsign, diagnosis | ~425k ED stays, ~205k patients | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimic-iv-ed/2.2/ |
| MIMIC-IV v3.1 (hosp, icu) | admissions.language (granular since v3.0), insurance, marital_status; patients (age); icustays (ICU transfer); transfers | ~546k admissions | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV-Note v2.2 | discharge summaries for interpreter-mention regex (local only) | ~331k discharge notes | PhysioNet credentialed | https://physionet.org/content/mimic-iv-note/2.2/ |
| MIMIC-IV-ED Demo v2.2 + MIMIC-IV Demo v2.2 | Pipeline dry-run without credentials | 100 patients | Open | https://physionet.org/content/mimic-iv-ed-demo/2.2/ , https://physionet.org/content/mimic-iv-demo/2.2/ |
| MIMIC-IV-ED benchmark (Xie et al., 2022) | Reference cohort/outcome definitions and baseline models | code | Open (GitHub) | https://github.com/nliulab/mimic4ed-benchmark |

Language is recorded at the hospital-admission level; ED-only patients who were never admitted have no language value. The primary cohort is therefore ED visits by subjects with >= 1 hospital admission (any time), with the language taken as the modal value across the subject's admissions and a conflict flag when admissions disagree. Selection into this cohort is itself analysed (RQ1 sensitivity).

## Methods

Pipeline (modules in `src/ed_language/`):

1. **Visit table** (`cohort.py`). Join edstays + triage + patients (age at visit from `anchor_age`/`anchor_year`) + admissions (language, insurance, marital status, hadm_id for admitted visits) + icustays. Harmonise `language` into groups (English, Spanish, Portuguese, Chinese languages, Russian, Haitian Creole, Vietnamese, Arabic, Other, Unknown) and `race` into 6 groups (White, Black, Hispanic/Latino, Asian, Other, Unknown). Outcomes: admitted, LWBS/eloped, ED LOS (h), critical outcome (ICU-in within 12 h of ED exit or in-hospital death), time to first pyxis medication, 72-h ED return (next edstay for the same subject within 72 h of outtime).
2. **Chief-complaint features** (`text_features.py`). Token count, number of complaint items (split on `,`, `/`, `;`, ` and `), vagueness score (share of tokens from a generic list), language-barrier marker flag (regex), pain-score missingness, number of missing triage vitals. Chief-complaint clusters: 30 clusters from TF-IDF + k-means on the lower-cased complaint (fitted once, used as a covariate).
3. **Interpreter mentions** (`text_features.py`). Regex over discharge notes: `interpreter`, `language line`, `via phone interpreter`, `spanish-speaking only`, etc. Counted per hadm_id. Local processing only.
4. **Adjusted associations** (`disparity.py`). Logistic (binary outcomes) and log-linear / quantile regression (LOS, time to medication) via statsmodels with the covariate set above; cluster-robust SEs by subject.
5. **Under-triage** (`disparity.py`). Critical-outcome rate by ESI x language group with Wilson CIs; risk ratios at each ESI level; AUROC of ESI (as an ordinal score) for critical outcome by language group with bootstrap CIs.
6. **Matching** (`disparity.py`). Exact matching on ESI, sex, 10-year age band, arrival transport, race group and insurance, 1:k with k <= 4; conditional logistic / paired comparisons.
7. **Mediation** (`disparity.py`). Regression-based natural direct/indirect effects (VanderWeele, 2015) with the mediator model (informativeness ~ language + covariates) and outcome model (outcome ~ language + mediator + covariates); bootstrap CIs (1000 resamples clustered by subject).
8. **Model fairness** (`disparity.py`). For any set of predictions (e.g. the Xie et al. benchmark models retrained locally, with and without text features): AUROC, calibration intercept/slope, sensitivity and FPR at 10% alert rate per language group; max-min gaps with bootstrap CIs.

Tools: pandas, DuckDB, statsmodels, scikit-learn, LightGBM (benchmark models), regex; no external NLP APIs.

## Evaluation & statistics

- Unit: ED visit, with cluster-robust SEs / cluster bootstrap by subject (patients have multiple visits).
- Pre-specified primary hypotheses: H1a, H3a, H4 (three tests, Holm). Everything else exploratory with BH-FDR within families (languages x outcomes).
- Negative-control outcome: triage temperature (should not differ by language after adjustment) and arrival hour distribution (negative-control exposure check for residual confounding via care-seeking patterns).
- Positive control: known association of ESI with admission must be reproduced in every language group.
- Leakage: for model-fairness analyses, predictions come from models trained with patient-level splits; language is never a feature; text features are derived from triage-time text only.
- Selection into the language-known cohort: compare ED-only (no admission ever) visits with cohort visits on triage variables; inverse-probability weighting sensitivity analysis.
- Missingness of language (`Unknown`) is analysed as its own group, not dropped.

## Publishable angle

Headline result: "Non-English-preferring patients arrive with less informative triage documentation, are under-triaged at ESI 3 (higher critical-outcome rate at the same acuity), and wait longer for first medication; roughly a fifth of the gap runs through documentation informativeness, the gap persists within Hispanic/Latino patients, is smaller when an interpreter is documented, and triage ML models inherit it." The granular-language and mediation results are new; the fairness result bears directly on deployed ED triage tools.

Target venues: Annals of Emergency Medicine; JAMA Network Open; Academic Emergency Medicine; JAMIA (fairness/method component); CHIL or ML4H (fairness track).

Follow-ups: (a) chief-complaint text embeddings with a local model to refine informativeness; (b) external replication on another ED dataset with a language field; (c) a nurse-facing triage prompt intervention study design.

## Risks, confounds & mitigations

- **Language is admission-level, not visit-level**; ED-only patients lack it. Mitigation: modal language per subject; selection analysis and IPW sensitivity; report the unknown group.
- **Language vs ethnicity vs immigration/insurance confounding.** Mitigation: within-ethnicity contrasts (H5); insurance and marital status as covariates; matching.
- **Chief-complaint text is de-identified and terse** (~2-6 tokens); informativeness features are coarse. Mitigation: multiple features, cluster covariate, sensitivity to feature definitions; treat the mediation estimate as a lower bound.
- **Interpreter mentions are incomplete and only for admitted patients.** Mitigation: use as exposure modifier only, not as a causal treatment; report note availability rates by language.
- **Critical-outcome ascertainment differs for discharged patients** (out-of-hospital deaths are unobserved). Mitigation: restrict death to in-hospital; use 72-h return as an additional outcome.
- **Date shifting removes secular trends** (interpreter services improved over 2011-2019). Mitigation: `anchor_year_group` as covariate.
- **Multiple small language groups.** Mitigation: minimum 500 visits per group for group-specific estimates; pooled "Other" otherwise; cells < 11 suppressed.

## Milestones

- [ ] Credentialing for MIMIC-IV-ED, MIMIC-IV (>= 3.0 for granular language), MIMIC-IV-Note.
- [ ] Dry-run on the ED + hospital demos (`scripts/download_data.py --sample`).
- [ ] Visit table with language/race harmonisation, outcomes and selection audit.
- [ ] Chief-complaint features, clusters and interpreter-mention counts; descriptive tables by language.
- [ ] RQ1 adjusted models; RQ2 documentation models; negative/positive controls.
- [ ] RQ3 under-triage calibration and ESI discrimination by language.
- [ ] RQ4 mediation with cluster bootstrap; RQ5 within-ethnicity; RQ6 interpreter modifier.
- [ ] RQ7 retrain benchmark triage models (patient-level split) and run fairness metrics.
- [ ] Pre-registration (OSF), manuscript, code release without data.

## Ethics / data-use notes

- MIMIC-IV-ED, MIMIC-IV and MIMIC-IV-Note are PhysioNet credentialed; complete CITI training and sign the DUA; never redistribute data; `data/` is git-ignored; credentials via `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`.
- Free-text notes must not be sent to third-party LLM or cloud APIs (PhysioNet responsible-use policy); the interpreter-mention extractor is a local regex.
- Language and race are sensitive attributes; results are reported at the group level with cell sizes >= 11 and framed as system-level care disparities, never as patient-level risk.
- Findings could be misused to justify differential care; the manuscript should include an explicit equity framing and recommendations (interpreter access, triage documentation prompts).
