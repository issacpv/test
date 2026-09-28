# synthetic-ehr-transportability

**Does a model trained on synthetic EHR data transport to a different hospital as well as one trained on the real source data? A source-to-target decomposition of the "synthetic gap" versus the "site gap" for Synthea, copula/GAN/diffusion generators trained on MIMIC-IV, and eICU as the external target.**

## Status / difficulty / timeline / compute

- Status: design + starter code (shared MIMIC-IV / eICU / Synthea feature ontology with a working Synthea loader, a from-scratch Gaussian-copula generator plus adapters for SDV/synthcity, fidelity metrics incl. pMSE/MMD/DCR, and the TSTR gap-decomposition with bootstrap). No data is shipped.
- Difficulty: MSc-level (tabular models, careful evaluation design); PhD-level only if longitudinal generators are trained from scratch. 6-9 months.
- Compute: workstation with 64 GB RAM (DuckDB over MIMIC-IV and eICU CSVs); one GPU (8-16 GB) for CTGAN/TVAE/diffusion generators; Synthea generation needs Java 11+ and ~1 h for 100k patients.

## Background

Synthetic EHR data is promoted as a way to share, teach and prototype without privacy risk. Generators fall into two families: *rule-based* simulators that never see real records (Synthea; Walonoski et al., 2018, JAMIA) and *learned* generators fitted to a real database (medGAN, Choi et al., 2017, MLHC; EHR-Safe, Yoon et al., 2023, npj Digit Med; HALO, Theodorou, Xiao & Sun, 2023, Nat Commun; EHR-M-GAN, Li et al., 2023, npj Digit Med; tabular diffusion models). The standard utility metric is Train-on-Synthetic-Test-on-Real (TSTR): a model fitted on synthetic data is scored on a held-out slice of the *same* real database. Benchmarks now exist: Yan et al. (2022, Nat Commun) evaluated generators on fidelity, utility and privacy for one database; a 2024 methodological review with benchmarks (Chen et al., arXiv:2411.04281) trained seven open-source generators on MIMIC-III and, notably, tested "transportability" by training on MIMIC-III and testing on MIMIC-IV, i.e. the same hospital a decade later with a different EHR system. Kaabachi et al. (2025, npj Digit Med) catalogue the privacy and utility metrics in use. On the rule-based side, Chen et al. (2019, BMC Med Inform Decis Mak) showed Synthea's clinical-quality-measure rates diverge from real populations.

Two things are missing from this literature. First, TSTR on the source database answers "does synthetic data preserve the source distribution?", not "does a model built on synthetic data survive deployment at another hospital?". The latter is the question that matters when synthetic data is used to build models in the absence of local data. Second, learned generators fit the *measurement process* of the source (which labs are ordered, when, how missingness looks) as much as the physiology; those observation-process features are known to be a major driver of cross-site shift in ICU models (see the `icu-model-transportability` project in this repository). A generator that faithfully copies the source's ordering habits may therefore produce models that transport *worse* than the real source data would, an amplification effect that no benchmark measures.

## The research gap

What has been done (2022-2026):

- Single-database TSTR benchmarks (Yan et al., 2022; EHR-Safe 2023 on MIMIC-III and eICU separately; HALO 2023 on MIMIC-III/IV separately).
- Same-hospital cross-era transport (MIMIC-III -> MIMIC-IV) in the 2024 review; no cross-institution target, no rule-based generator, no decomposition of the gap.
- Synthea validity studies compare aggregate rates to population statistics, not downstream-model transport.
- Metric scoping reviews (Kaabachi et al., 2025) call for standardised utility evaluation but do not propose transport-aware metrics.

What is specifically missing (our angle):

1. **A 2x2 transport design**: {real source, synthetic source} x {source holdout, external target} for the same task, feature ontology and model class, yielding four numbers per generator from which the *synthetic gap*, the *site gap* and their *interaction* (amplification) are identified (`transport.gap_decomposition`).
2. **Three generator families side by side**, including a rule-based simulator (Synthea) that has never seen MIMIC, learned tabular generators (copula, CTGAN/TVAE, diffusion) and a learned longitudinal generator (HALO-style), on a shared encounter-level ontology.
3. **Observation-process ablation**: generators and models trained with and without measurement-count / missingness features, to test whether amplification is driven by copied ordering behaviour.
4. **Subgroup transport**: does synthetic training change the *fairness* of the transported model (sex, age band, race where available)?
5. **A calibrated prescription**: how many real target records are needed to recalibrate a synthetic-trained model to parity with a real-trained one.

## Research questions / hypotheses

1. **RQ1 (amplification).** Is the site gap larger for synthetic-trained models than for real-trained ones? H1: for learned generators, amplification = (syn->src - syn->tgt) - (real->src - real->tgt) is positive for AUROC and for calibration-in-the-large, with bootstrap CIs excluding zero; for Synthea the site gap is *smaller* than for real data (Synthea is not tied to any site) but the synthetic gap is much larger.
2. **RQ2 (observation process).** H2: removing measurement-count and missingness features reduces amplification by >= 50% for learned generators.
3. **RQ3 (fidelity does not predict transport).** H3: standard fidelity metrics (per-feature KS, correlation distance, pMSE, MMD) rank generators differently from the transported utility; pMSE correlates with source TSTR (rho > 0.7) but not with target utility (rho < 0.3).
4. **RQ4 (fairness).** H4: subgroup AUROC / calibration gaps at the target are larger for synthetic-trained models, especially for subgroups that are rare in the source (generators under-represent tails).
5. **RQ5 (recalibration).** H5: intercept + slope recalibration with n = 500 target records closes >= 80% of the synthetic-vs-real calibration gap at the target, but does not close the discrimination gap.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (hosp + icu modules) | source site (BIDMC): real training data and training data for learned generators | ~546k admissions, ~94k ICU stays; ~30 GB csv.gz | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV demo v2.2 | pipeline development (100 patients) | small | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| eICU-CRD v2.0 | external target: 208 US hospitals; hospital id allows within-eICU multi-site analysis | ~200k ICU stays; ~20 GB | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/eicu-crd/2.0/ |
| eICU-CRD demo v2.0.1 | pipeline development | small | Open | https://physionet.org/content/eicu-crd-demo/2.0.1/ |
| Synthea (rule-based simulator) + sample CSV exports | rule-based synthetic source that has never seen real records | any size; 1k-patient sample CSV ~ 30 MB | Open (Apache-2.0) | https://github.com/synthetichealth/synthea ; https://synthea.mitre.org/downloads |
| SDV (CTGAN, TVAE, GaussianCopula), synthcity (DDPM, TabDDPM-style) | learned tabular generators | software | Open (SDV: BSL-1.1; synthcity: Apache-2.0) | https://sdv.dev ; https://github.com/vanderschaarlab/synthcity |
| HALO reference implementation | learned longitudinal generator (optional stratum) | software | Open | https://github.com/btheodorou99/HALO_Inpatient |

Task definitions (encounter level, shared by all three sources): (T1) in-hospital mortality for adult inpatient admissions with >= 1 lab; (T2) 30-day unplanned readmission. ICU-level tasks (48-h mortality) are run for MIMIC-IV -> eICU only, since Synthea has no ICU physiology.

## Methods

Pipeline (modules in `src/synth_transport/`):

1. **Shared ontology** (`ontology.py`). 26 encounter-level concepts with source-specific definitions: demographics (age, sex), admission context (emergency flag, prior encounters in the last year), six chronic-condition flags via ICD-9/10 prefixes (MIMIC/eICU) or SNOMED codes (Synthea), ten first-24-h labs via `itemid` (MIMIC), `labname` (eICU) or LOINC (Synthea), plus per-lab measurement counts and missingness indicators (the *observation-process* block). `verify_spec` checks completeness; `load_synthea_encounters` builds the table from Synthea CSV exports; SQL templates for MIMIC-IV and eICU (DuckDB) are documented in the module.
2. **Generators** (`generators.py`). A dependency-free `GaussianCopulaGenerator` (empirical marginals + Gaussian copula; the classical baseline), `IndependentMarginalsGenerator` (negative control: destroys correlations), `BootstrapJitterGenerator` (privacy-leaky positive control), and import-guarded adapters `SDVAdapter` (CTGAN/TVAE) and `SynthcityAdapter` (DDPM). All expose `fit(df)` / `sample(n)`. Synthea is not a fitted generator: its output is simply loaded as a source table.
3. **Fidelity** (`fidelity.py`). Per-feature KS / total variation, correlation-matrix distance, propensity-score MSE with its null-expected ratio (Snoke et al., 2018, JRSS-A), RBF-MMD, distance-to-closest-record (DCR) percentiles against a real-vs-real baseline (privacy proxy), and missingness-pattern agreement.
4. **Transport** (`transport.py`). Fit logistic regression and gradient boosting on each training set; evaluate AUROC, AUPRC, Brier, calibration intercept/slope, ECE on each test set (`tstr_matrix`); compute the decomposition with stay-level bootstrap CIs (`gap_decomposition`, `bootstrap_decomposition`); subgroup metrics (`subgroup_metrics`).
5. **Ablation and recalibration.** Re-run 1-4 with the observation-process block removed; few-shot intercept/slope recalibration on n in {100, 250, 500, 1000, 2500} target records, 200 repeats.
6. **Within-eICU replication.** Treat the ten largest eICU hospitals as separate targets to obtain a distribution of amplification values rather than a single number.

Tools: DuckDB, pandas, scikit-learn, LightGBM, SDV, synthcity, Synthea (Java), statsmodels.

### Concept coverage across sources

| Block | Concepts | MIMIC-IV | eICU-CRD | Synthea |
|---|---|---|---|---|
| demographics | age, sex | `patients.anchor_age/anchor_year`, `gender` | `patient.age` ('> 89' -> 90), `gender` | `BIRTHDATE`, `GENDER` |
| context | emergency admission; prior encounters in 365 d | `admissions.admission_type`; admissions history | `hospitaladmitsource`; not available (single-stay records) | `ENCOUNTERCLASS == 'emergency'`; encounter history |
| conditions | diabetes, hypertension, heart failure, COPD, CKD, cancer | `diagnoses_icd` ICD-9/10 prefixes | `diagnosis.icd9code`, `pastHistory` | `conditions.CODE` (SNOMED) active at START |
| labs | creatinine, sodium, potassium, glucose, haemoglobin, WBC, platelets, bicarbonate, BUN, lactate (first value, 24 h) | `labevents.itemid` | `lab.labname` | `observations.CODE` (LOINC) |
| observation | per-lab measurement count and missingness indicator | counts in 24 h | counts in 24 h | counts within encounter |
| outcomes | in-hospital mortality; 30-day readmission | `hospital_expire_flag`; next `admittime` | `hospitaldischargestatus`; not available | `DEATHDATE` in [START, STOP]; next inpatient START |

The exact identifiers are in `ontology.CONDITIONS`, `ontology.LABS` and `ontology.FEATURE_SPEC`; `ontology.verify_spec()` fails if any concept lacks a definition for any source. Concepts unavailable in a source (eICU prior encounters and readmission) are set to NaN and excluded from the comparisons that involve that source, and the resulting coverage table is reported in the paper.

## Evaluation & statistics

- Splits: MIMIC-IV patients split 70/30 (generator + model training / source holdout) by `subject_id`; eICU is never used for fitting except in the recalibration experiment, where the recalibration sample is disjoint from the evaluation sample.
- Synthetic sample sizes match the real training set; five generator seeds; metrics averaged over seeds with between-seed SD reported.
- Leakage: labs restricted to the first 24 h; outcomes from discharge tables only; no generator sees the holdout.
- Metrics: AUROC, AUPRC, Brier, log-loss, calibration intercept/slope, ECE (10 quantile bins), net benefit at 5-20% thresholds. Decomposition components with 1,000-resample bootstrap CIs (resampling test encounters, and generator seeds as a second level).
- Multiple comparisons: five pre-registered hypotheses (H1-H5), Holm-adjusted; per-generator and per-hospital results are descriptive.
- Nulls / controls: the decomposition applied to two random halves of the source must give a site gap of ~0; `IndependentMarginalsGenerator` must show a large synthetic gap; `BootstrapJitterGenerator` must show ~0 synthetic gap and the same site gap as real data.
- Privacy is reported (DCR, membership-inference AUC) so that utility gains are not achieved by memorisation.

## Publishable angle

Headline: "Synthetic EHR data that scores well on standard fidelity and TSTR metrics produces models whose external-site performance drops X AUROC points *more* than models trained on the real source data; the excess is attributable to copied observation-process features and is not predicted by any current fidelity metric, whereas rule-based Synthea data transports uniformly but poorly."

Target venues: npj Digital Medicine; JAMIA; Journal of Biomedical Informatics; ML4H / CHIL (proceedings). Data-sharing angle for Scientific Data if the harmonised ontology is released.

Follow-ups: longitudinal (time-series) generators on the ICU tasks with HiRID as a second target; federated generator training across sites; a "transport-aware" utility metric proposal.

## Related project

`icu-model-transportability` decomposes real-to-real cross-site shift into covariate/label/concept components on four ICU databases; this project asks the orthogonal question of what synthetic *training* data does to that shift and re-uses only the idea of a shared, verified ontology. The two projects share no code.

## Risks, confounds & mitigations

- **Ontology mismatch across sources** (eICU labs are free-text names; Synthea lacks many labs). Mitigation: `verify_spec` plus manual review; Synthea comparisons restricted to concepts it emits; report coverage per concept.
- **Generator quality confounds transport.** Mitigation: report the synthetic gap at the source alongside; the amplification term is defined relative to it.
- **eICU is multi-site; pooled results hide heterogeneity.** Mitigation: per-hospital analysis for the ten largest hospitals.
- **Learned generators may memorise.** Mitigation: DCR and membership-inference audits; discard generator configurations with DCR below the real-vs-real baseline.
- **Synthea version drift.** Pin the Synthea release and export the generation seed and module list with the data.
- **Compute for diffusion/GAN generators.** Start with the copula baseline (minutes on CPU) to establish the full pipeline before GPU generators.

## Milestones

- [ ] Credentialing (CITI, PhysioNet DUAs) for MIMIC-IV and eICU; develop on the demo databases.
- [ ] Ontology SQL for MIMIC-IV and eICU; Synthea loader; `verify_spec` passes; coverage table.
- [ ] Baseline 2x2 with real data and the copula generator; negative/positive controls behave.
- [ ] CTGAN/TVAE/DDPM generators, five seeds; fidelity and privacy tables.
- [ ] Decomposition with bootstrap CIs; observation-process ablation; per-hospital eICU analysis.
- [ ] Fairness and recalibration experiments.
- [ ] Pre-registration of H1-H5 before eICU evaluation; manuscript; release of ontology + code.

## Ethics / data-use notes

- MIMIC-IV and eICU are credentialed (PhysioNet CITI training + DUA); data must stay on approved machines and must never be committed or uploaded. Synthetic data derived from MIMIC-IV is still a derivative of a credentialed dataset: do not redistribute it outside the DUA terms.
- Do not send credentialed data to external services or third-party APIs; all generators run locally.
- Synthea data is fully synthetic and freely shareable.
- Subgroup analyses use race/ethnicity fields only where present (MIMIC-IV, eICU) and follow the fairness-reporting guidance of the respective databases.
