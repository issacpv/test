# sepsis-definition-multiverse

**A factorial multiverse of Sepsis-3 operationalisations on MIMIC-IV and eICU-CRD: how much do suspected-infection rules, SOFA windows and baselines change the cohort, the onset time, early-warning benchmarks, model rankings and treatment-effect estimates, and which decisions matter most?**

## Status / difficulty / timeline / compute

- Status: design + starter code (SOFA component scoring with configurable missingness handling, suspected-infection pairing rules, a specification grid with cohort/onset/Jaccard summaries, main-effects variance decomposition, and a definition-transfer benchmark). No data is shipped.
- Difficulty: MSc-level (cohort multiverse) to PhD-level (downstream causal and RL re-analyses). Mostly SQL/pandas engineering and careful statistics.
- Timeline: 6-9 months (1 month credentialing + extraction, 2 months implementing and validating the grid against mimic-code and ricu, 2 months benchmarks and treatment-effect analyses, 1-2 months writing).
- Compute: workstation with >= 64 GB RAM and DuckDB over MIMIC-IV/eICU CSV/parquet; the full grid (~300 specifications x 2 databases) is embarrassingly parallel and CPU-only; the early-warning models are gradient-boosted trees (a GRU baseline is optional).

## Background

Sepsis-3 (Singer et al., 2016 JAMA) defines sepsis as suspected infection plus an acute rise of >= 2 SOFA points; Seymour et al. (2016 JAMA) operationalised suspected infection as a culture-antibiotic pair within asymmetric windows (antibiotic first: culture within 24 h; culture first: antibiotic within 72 h) and looked for the SOFA rise in a window from 48 h before to 24 h after. Every retrospective study must fill in unspecified details: which antibiotics count, whether a culture is required at all (eICU culture timing is sparse), how SOFA components are imputed when unmeasured, what the SOFA baseline is (zero, the pre-infection minimum, ICU admission), the exact window, and whether onset is the suspicion time or the SOFA-rise time. Johnson et al. (2018 Critical Care Medicine) compared sepsis identification methods on MIMIC-III and found large disagreement between Sepsis-3, Angus and CDC criteria. The MIMIC code repository (mimic-code `sepsis3.sql`), `ricu` (Bennett et al., 2023 GigaScience) and YAIB (van de Water et al., 2024 ICLR) each ship a different implementation, and ML papers routinely use their own.

Downstream, sepsis labels feed early-warning benchmarks (Moor et al., 2021 Frontiers in Medicine systematic review; Moor et al., 2023 eClinicalMedicine), off-policy reinforcement learning (AI Clinician: Komorowski et al., 2018 Nature Medicine) and observational treatment-effect studies (time-to-antibiotics, fluids), all of which are sensitive to who is in the cohort and when t = 0 is.

## The research gap

What has been done (2023-2026):

- Cohen et al. (2024 Scientific Reports, "Subtle variation in sepsis-III definitions markedly influences predictive performance within and across methods") compared three interpretations of sepsis onset on MIMIC and showed that the definition changes model performance by more than the differences between models. Three variants, one database, one task.
- A 2026 medRxiv preprint ("Variability in automated sepsis case detection", with a systematic source-code analysis supplement) audited six public Sepsis-3 implementations (four on MIMIC-III, two on eICU-CRD) and classified their decisions into six domains (D1-D6) along the pipeline from raw data to label. It characterises the repositories; it does not enumerate the full factorial space, does not use MIMIC-IV, and does not carry the variation through to benchmarks or causal estimates.
- MIMIC-Sepsis (arXiv, October 2025) provides a curated trajectory benchmark under one definition.
- Falsification testing of sepsis prediction models (medRxiv 2026) and observation-process/domain-shift studies (medRxiv 2026) address model validity but hold the definition fixed.
- Correspondence in 2025 raised concerns about ICD-based sepsis identification in MIMIC-IV.
- The related project `icu-model-transportability` in this repository treats Sepsis-3 operationalisation as a nuisance for cross-site transfer; the present project makes the operationalisation itself the object of study.

What is specifically missing (our angle):

1. A **full factorial grid** (about 300 specifications) over the decision dimensions, run identically on MIMIC-IV v3.1 and eICU-CRD v2.0, with each dimension traceable to a published implementation (mimic-code, ricu, YAIB, AI Clinician, Seymour 2016), and validated by reproducing those implementations exactly as grid points.
2. **Cohort geometry**: pairwise Jaccard overlap of sepsis cohorts, onset-time shifts (median and IQR of the difference in t = 0 between specifications), and the fraction of patients whose onset moves across the "before ICU admission" boundary (which silently changes the prediction task).
3. **Variance decomposition**: main-effects and interaction eta^2 of each decision dimension on cohort size, mortality, and downstream metrics; the answer to "which decisions matter?" has not been published.
4. **Downstream consequences**: early-warning AUROC/AUPRC/calibration at fixed horizons under every specification; model *rankings* (LR vs GBM vs GRU) across specifications; a **definition-transfer matrix** (train under definition A, evaluate under B); time-to-antibiotics effect estimates; the value of a fixed treatment policy under the AI-Clinician cohort vs alternatives.
5. **Recommendation**: a small "core set" of specifications that spans most of the variance, to be reported with any sepsis benchmark, and code that emits it.

## Research questions / hypotheses

1. **RQ1 (cohort).** How much do cohort size and hospital mortality vary across the grid? H1: cohort size varies by more than 2-fold and mortality by more than 5 absolute points on both databases; the suspected-infection rule (culture required vs antibiotic-only) is the largest main effect on size, the SOFA baseline the largest on mortality.
2. **RQ2 (onset time).** H2: median onset time shifts by > 6 h between the suspicion-time and SOFA-rise-time conventions, and > 20% of patients change status between "sepsis at ICU admission" and "ICU-acquired sepsis" across specifications.
3. **RQ3 (benchmarks).** H3: the range of early-warning AUROC across specifications (same model, same features) exceeds the range across models (same specification); the ranking of models is not stable across the grid (Kendall's W < 0.7).
4. **RQ4 (transfer).** H4: models trained under one specification lose > 0.05 AUROC when evaluated under a specification that differs in the SOFA window, but < 0.02 when only the antibiotic list differs; calibration intercepts track the prevalence change.
5. **RQ5 (causal).** H5: the estimated association between time-to-antibiotics and mortality changes sign or loses significance across specifications, driven by the onset-time convention (immortal-time structure).
6. **RQ6 (cross-database).** H6: the ranking of decision dimensions by eta^2 is the same in MIMIC-IV and eICU (Spearman > 0.8), except that the culture requirement matters far more in eICU because culture timing is sparse there.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (hosp + icu) | Adult ICU stays; labs, vitals, vasopressors, ventilation, GCS, urine output, antibiotics (`prescriptions`, `emar`), cultures (`microbiologyevents`), mortality | ~94k ICU stays; ~30 GB | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV Clinical Database Demo v2.2 | Open 100-patient subset for developing and testing the SQL/pandas pipeline | ~40 MB | Open (PhysioNet, no credentialing) | https://physionet.org/content/mimic-iv-demo/2.2/ |
| eICU-CRD v2.0 | Second database (208 US hospitals); `lab`, `vitalPeriodic`, `infusionDrug`, `medication`, `microLab`, `treatment`, `respiratoryCare` | ~200k stays; ~20 GB | PhysioNet credentialed | https://physionet.org/content/eicu-crd/2.0/ |
| eICU-CRD Demo v2.0.1 | Open subset for pipeline development | ~50 MB | Open | https://physionet.org/content/eicu-crd-demo/2.0.1/ |
| mimic-code concepts (`sepsis3.sql`, `suspicion_of_infection.sql`, `sofa.sql`) | Reference implementation reproduced as a grid point | small | Open (MIT licence) | https://github.com/MIT-LCP/mimic-code |
| ricu (R) `sep3`, `susp_inf`, `sofa` concepts; YAIB task definitions | Additional reference grid points | small | Open | https://github.com/eth-mds/ricu ; https://github.com/rvandewater/YAIB |

## Methods

Pipeline (modules in `src/sepsis_multiverse/`):

1. **Extraction (DuckDB SQL)** into a long hourly table per stay: `(stay_id, hour, concept, value)` for PaO2, FiO2, ventilation flag, platelets, bilirubin, MAP, vasopressor doses (norepinephrine, epinephrine, dopamine, dobutamine), GCS, creatinine, urine output; plus antibiotic administrations and culture events with timestamps. eICU concept mapping follows ricu.
2. **SOFA** (`sofa.py`). Component scores per hour with configurable carry-forward horizons (per concept), "assume normal when never measured" vs "missing", GCS handling under sedation (last pre-sedation value vs verbal-component imputation), and urine output aggregation windows (24-h vs 6-h extrapolated).
3. **Suspected infection** (`suspected_infection.py`). Rules: `culture_abx` (Seymour asymmetric windows, both directions), `abx_first_only`, `culture_first_only`, `abx_only` (any qualifying antibiotic; eICU fallback), `abx_2plus` (two administrations of the same antibiotic within 96 h, a common variant); antibiotic list = mimic-code list vs broad ATC J01; the suspicion time = earlier of the pair vs antibiotic time.
4. **Labels** (`multiverse.py`). `SepsisSpec` holds all decisions; `label_sepsis` finds the first hour in the window `[t_si - before, t_si + after]` where SOFA - baseline >= delta, with baseline in {zero, min over the 24/48 h before t_si, first ICU value, minimum over the whole stay before the window}. Onset convention: suspicion time vs SOFA-rise time. Cohort filters: adults, first ICU stay, onset relative to ICU admission (any / after / within 24 h before).
5. **Grid and geometry** (`multiverse.py`). Enumerate the factorial grid; for each specification store cohort ids, onset times, mortality; compute pairwise Jaccard, onset shifts and the eta^2 decomposition (`variance_decomposition`) of each outcome over dimensions (one-hot main effects, then pairwise interactions).
6. **Benchmarks** (`benchmark.py`). Early-warning task: at each hour from ICU admission until onset (cases) or discharge (controls), predict onset within 6/12 h from features of the previous 24 h. Models: L2 logistic regression, gradient-boosted trees, optional GRU. Metrics per specification; definition-transfer matrix; Kendall's W of model rankings across specifications.
7. **Causal re-analyses**. Time-to-antibiotics (from onset) vs hospital mortality with a target-trial emulation (cloning-censoring-weighting) per specification; AI-Clinician-style cohort with a fixed policy evaluated by weighted importance sampling per specification.

Tools: DuckDB, pandas/pyarrow, scikit-learn, LightGBM (optional), statsmodels, ricu (R, for concept cross-checks).

## Evaluation & statistics

- Every reference implementation is reproduced as a grid point and checked against its published prevalence (mimic-code Sepsis-3 on MIMIC-IV; YAIB sepsis prevalence; Johnson et al. 2018 numbers on MIMIC-III where applicable). The grid is not trusted until these match within 2%.
- Patient-level splits (70/15/15) fixed once and shared across all specifications so that differences are due to labels, not to resampling.
- Metrics: AUROC, AUPRC, Brier, calibration slope/intercept, net benefit at 10-30% thresholds; bootstrap CIs (1,000 stay-level resamples).
- Variance decomposition: eta^2 per dimension with bootstrap CIs; specification curves (Simonsohn et al., 2020 Nature Human Behaviour) for each outcome.
- Model-ranking stability: Kendall's W across specifications; Friedman test.
- Multiple comparisons: the grid is descriptive; pre-registered hypotheses H1-H6 are the confirmatory tests, BH-FDR within each.
- Negative control: a "random label" specification (onset times shuffled across stays) to calibrate what benchmark variation looks like when labels carry no information.

## Publishable angle

Headline: "Across ~300 Sepsis-3 operationalisations, cohort size varies N-fold and early-warning AUROC varies more across definitions than across models; three decisions (culture requirement, SOFA baseline, onset convention) explain most of the variance, and treatment-effect estimates flip with the onset convention." Deliverables: the grid, the eta^2 table, a recommended core set, and code that emits labels for all core-set definitions from a MIMIC-IV/eICU DuckDB.

Target venues: Critical Care Medicine or Intensive Care Medicine (clinical audience); JAMIA or npj Digital Medicine (informatics); ML4H / CHIL proceedings for the benchmark-ranking result.

Follow-ups: extend the grid to HiRID and AmsterdamUMCdb via ricu; a "definition-robust" training objective (multi-label over the core set); apply the multiverse to AKI (KDIGO baseline creatinine choices) and ARDS.

## Risks, confounds & mitigations

- Antibiotic and culture data differ structurally between MIMIC-IV (`prescriptions`, `emar`, `microbiologyevents`) and eICU (`medication`, `infusionDrug`, `microLab`, `treatment`): mitigation: per-database concept dictionaries validated against ricu; eICU culture-based rules flagged as low-confidence with the antibiotic-only rule as the primary eICU specification.
- Grid explosion: mitigation: a fractional factorial for interactions beyond pairwise; all main effects fully crossed.
- Leakage in early-warning benchmarks (features after onset): mitigation: strict prediction-time cut-offs and a leakage unit test on synthetic data.
- Immortal-time bias in the time-to-antibiotics analysis: mitigation: target-trial emulation with cloning; report the naive estimate alongside to show the artefact.
- Overinterpreting AUROC differences: mitigation: CIs from a shared bootstrap; report AUPRC and calibration; negative control.
- Reproducing reference implementations may reveal bugs in them: report discrepancies transparently and pin the versions used.

## Milestones

- [ ] PhysioNet credentialing (MIMIC-IV, eICU); demo databases used to develop and test the pipeline end to end
- [ ] Hourly concept tables extracted for both databases; concept dictionaries validated against ricu
- [ ] `SepsisSpec` grid enumerated; reference implementations reproduced within 2% prevalence
- [ ] Cohort geometry tables (sizes, mortality, Jaccard, onset shifts) with specification curves
- [ ] Variance decomposition with bootstrap CIs on both databases
- [ ] Early-warning benchmarks under all specifications; definition-transfer matrix; ranking stability
- [ ] Time-to-antibiotics target-trial emulation and policy-value sensitivity
- [ ] Core-set recommendation and label-emitting code
- [ ] Pre-registration (OSF) of H1-H6 before running the benchmark grid; manuscript

## Ethics / data-use notes

- MIMIC-IV and eICU-CRD are PhysioNet credentialed resources: complete CITI training, sign the DUAs, keep data on approved encrypted storage, and follow PhysioNet's responsible-use policy. Patient-level data (including free-text antibiotic or culture names) must not be sent to third-party LLM APIs; only local tooling touches the data.
- The download script reads `PHYSIONET_USER` / `PHYSIONET_PASS` from the environment and never stores them; the demo datasets are open and used for development.
- Never commit data or patient-level derived tables (`.gitignore` covers `data/`, `outputs/`, DuckDB files). Publish only aggregate results with cell counts >= 10.
- Sepsis definitions affect patients through quality metrics and alerts; report the multiverse without recommending any single definition as "true".
