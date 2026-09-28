# faers-signal-ehr-validation

**How often is a FAERS disproportionality signal real? Validating PRR/ROR/BCPNN/EBGM signals from openFDA against lab-defined adverse outcomes in MIMIC-IV with an active-comparator new-user target-trial design, to build a drug-by-drug, outcome-by-outcome positive-predictive-value map and an EHR-derived reference set.**

## Status / difficulty / timeline / compute

- Status: design + starter code (openFDA client, disproportionality statistics with CIs and an MGPS/EBGM implementation, RxNorm mapping helpers, MIMIC-IV lab-outcome SQL templates, active-comparator cohort + propensity-score matching and empirical calibration). No data shipped.
- Difficulty: MSc-level for one outcome; PhD-level for the full map with empirical calibration.
- Timeline: 6-9 months (openFDA extraction 1 month, MIMIC-IV cohort engineering 2-3 months, analysis 2 months, writing 1-2 months).
- Compute: laptop/workstation. openFDA bulk files are ~30 GB of JSON (or use the API's `count` endpoints); MIMIC-IV `labevents` (~120 M rows) needs DuckDB with 32 GB RAM or a machine with a fast SSD. No GPU.

## Background

Spontaneous reporting systems such as the FDA Adverse Event Reporting System (FAERS) are the workhorse of post-marketing signal detection. Disproportionality statistics - proportional reporting ratio (PRR; Evans, Waller & Davis, 2001 Pharmacoepidemiol Drug Saf), reporting odds ratio (ROR; van Puijenbroek et al., 2002; Rothman, Lanes & Sacks, 2004), the Bayesian confidence propagation neural network information component (BCPNN IC; Bate et al., 1998 Eur J Clin Pharmacol; Norén et al., 2013 Stat Methods Med Res) and the multi-item gamma-Poisson shrinker (MGPS/EBGM; DuMouchel, 1999 Am Stat) - are computed on millions of reports and generate thousands of "signals" per year. openFDA (Kass-Hout et al., 2016 JAMIA) makes FAERS queryable through a public API, and hundreds of 2023-2026 papers publish disproportionality analyses of single drugs or events from it.

The known weaknesses are notoriety and stimulated reporting, confounding by indication, masking, lack of denominators and duplicate reports. To know how well the statistics work, the field uses reference sets of "known" drug-event pairs: OMOP (Ryan et al., 2013 Drug Saf; 165 positive / 234 negative controls across acute liver injury, acute MI, acute kidney injury and GI bleeding), EU-ADR (Coloma et al., 2013 Drug Saf; 94 pairs, 10 events), a time-indexed standard (Harpaz et al., 2014 Sci Data), and newer sets built from labels (Ther Innov Regul Sci, 2024) or from EU documents with LLM assistance (arXiv, 2026). On these sets, FAERS algorithms reach AUROCs around 0.6-0.75 (Harpaz et al., 2013 Clin Pharmacol Ther; Candore et al., 2015 Drug Saf). But those reference sets are themselves built from product labels and literature that were largely informed by spontaneous reports, so the evaluation is partly circular ("can we trust negative controls?", Drug Saf 2016/2017).

## The research gap

What exists:

- EHR-based signal detection on its own (LePendu et al., 2013 CPT with clinical notes; the EU-ADR LEOPARD approach; a 2023 scoping review in Drug Safety of EHR signal identification found that most studies apply disproportionality-like methods to EHR data with minimal confounding control).
- Combining SRS and EHR *signal scores* (Li et al., 2015 Drug Saf; Harpaz et al., 2012 CPT), i.e. fusing two noisy detectors, not validating one against a measured outcome.
- Lab-anchored EHR pharmacovigilance for single outcomes: e.g. a two-stage design with propensity-score-matched cohorts for drug-induced thrombocytopenia in paediatric EMR (Front Pharmacol, 2021); QT-prolongation prediction from ECGs plus prescribing data (QTNet, JACC Clin Electrophysiol 2024; Heart Rhythm 2025 on ML-enabled drug-induced QT risk).
- MIMIC-IV has the pieces - `hosp/prescriptions` with NDC, `hosp/emar` administrations, `hosp/labevents`, `hosp/diagnoses_icd`, and MIMIC-IV-ECG (Gow et al., 2023 PhysioNet) with machine-measured intervals for ~800k 12-lead ECGs - but no published study uses it to validate FAERS signals.

What is missing (our angle):

1. **A quantitative PPV map**: for every drug ingredient with >= 100 suspect reports in FAERS and each of 7 lab-/ECG-definable outcomes (hyperkalaemia, hyponatraemia, AKI by KDIGO creatinine, hepatocellular injury by ALT/bilirubin, thrombocytopenia, neutropenia, QTc >= 500 ms / delta-QTc >= 60 ms), the FAERS signal (PRR/ROR/IC025/EB05 with the usual thresholds) is scored against an EHR effect estimate from an **active-comparator new-user cohort with propensity-score matching** in MIMIC-IV. PPV, sensitivity and the dose-response between signal strength and EHR risk ratio have never been reported drug-by-drug and outcome-by-outcome against *measured* outcomes.
2. **An EHR-derived reference set** (positive / negative / indeterminate drug-outcome pairs with effect sizes and CIs) that does not depend on labels or literature, released as a table for methodologists, complementing OMOP/EU-ADR.
3. **Which reporting features predict false positives**: reporter qualification, year of report (Weber effect), indication overlap with the outcome, "concomitant" vs "suspect" characterisation, and stimulated-reporting spikes, using the EHR result as ground truth.
4. **Empirical calibration** (Schuemie et al., 2014 Stat Med; 2018 PNAS) with outcome-specific negative-control drugs so that the EHR side has calibrated p-values and the map is not itself a pile of false positives.

## Research questions / hypotheses

1. **RQ1.** What is the PPV of a FAERS signal (PRR >= 2, chi-square >= 4, n >= 3) for each lab-defined outcome, taking a calibrated EHR risk ratio >= 1.5 with calibrated p < 0.05 as truth? H1: PPV is highest for hyperkalaemia and AKI (mechanistically simple, monitored routinely; PPV > 0.5) and lowest for hepatotoxicity and QTc prolongation (PPV < 0.3) because these are reported on notoriety.
2. **RQ2.** H2: EB05 and IC025 have higher PPV than PRR/ROR at matched sensitivity, and the *rank* of the signal statistic is monotonically related to the EHR risk ratio (Spearman rho > 0.4).
3. **RQ3.** H3: signals driven mostly by non-health-professional reporters, by reports from the two years after approval, or by reports in which the drug is "concomitant" rather than "suspect" have lower PPV (odds ratio < 0.5 for being EHR-confirmed).
4. **RQ4.** H4: at least 20% of drug-outcome pairs with strong EHR evidence (calibrated RR >= 2) have *no* FAERS signal, mostly for outcomes that are detected by routine labs rather than symptoms (hyponatraemia, thrombocytopenia), quantifying the sensitivity gap of spontaneous reporting.
5. **RQ5.** H5: EHR effect estimates without an active comparator (vs non-users) and without PS matching agree with FAERS signals *more* than the properly adjusted estimates do, because both share confounding by indication - a warning about naive EHR "validation".

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| openFDA drug adverse event API (FAERS) | Report counts per drug (openfda.generic_name / rxcui), event (MedDRA PT), characterisation, reporter, year; 2x2 tables | ~18 M reports (2004-2026); JSON bulk files ~30 GB | Open; API key (free) raises the limit to 240 req/min, 120k/day | https://open.fda.gov/apis/drug/event/ |
| openFDA bulk downloads | Full-data reproducibility (duplicates, case versions) | quarterly zipped JSON | Open | https://api.fda.gov/download.json |
| MIMIC-IV v3.1 (hosp module: prescriptions, emar, emar_detail, labevents, d_labitems, diagnoses_icd, admissions, patients) | Exposures, lab-defined outcomes, covariates | ~65k patients in ICU, ~200k in hosp; labevents ~120 M rows | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV-ECG v1.0 (machine_measurements.csv, record_list.csv) | QT/QTc outcomes from machine measurements (rr_interval, qrs_onset, t_end) | ~800k ECGs, ~160k patients | PhysioNet open access (linkable to MIMIC-IV, which is credentialed) | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| RxNav / RxNorm REST API | NDC -> RxCUI -> ingredient mapping; name normalisation | API | Open (NLM) | https://rxnav.nlm.nih.gov/RxNormAPIs.html |
| OMOP reference set (Ryan et al., 2013); EU-ADR set (Coloma et al., 2013) | External comparison of the EHR-derived reference set | 399 / 94 pairs | Open (supplementary tables) | https://doi.org/10.1007/s40264-013-0097-8 |

## Methods

1. **FAERS side** (`openfda_client.py`, `disproportionality.py`). For each ingredient and outcome PT group: a = reports with drug (suspect) and event, b, c, d from `meta.results.total` of four searches; also stratified counts by reporter qualification, report year and characterisation for RQ3. PRR (+chi-square), ROR with 95% CI, IC with IC025 (gamma posterior), EBGM/EB05 with the MGPS prior fitted by marginal likelihood over the whole drug x event table. Signals: Evans criteria; IC025 > 0; EB05 >= 2.
2. **Mapping** (`rxnorm.py`). MIMIC-IV `prescriptions.ndc` -> RxCUI -> ingredient via RxNav; fallback by drug name (approximateTerm); FAERS side uses `patient.drug.openfda.generic_name` and `openfda.rxcui`. Ingredient-level analysis; salts and combinations collapse to ingredient.
3. **EHR outcomes** (`outcomes.py`). DuckDB SQL over `labevents` for potassium (50971/50822), sodium (50983/50824), creatinine (50912), ALT (50861), AST (50878), total bilirubin (50885), platelets (51265), absolute neutrophils (52075) / WBC (51301); QTc from MIMIC-IV-ECG `machine_measurements` (QT = t_end - qrs_onset; Bazett with rr_interval). Incident outcome = baseline within 7 days before index meets "normal" criteria and the first post-index value within the outcome window crosses the threshold (hyperkalaemia K >= 5.5; hyponatraemia Na < 130; AKI KDIGO stage >= 1; hepatocellular injury ALT >= 3x ULN, severe if bilirubin >= 2x ULN; thrombocytopenia platelets < 100 or >= 50% drop; neutropenia ANC < 1.0; QTc >= 500 ms or delta-QTc >= 60 ms).
4. **Target trial** (`cohort.py`). Eligibility: adults; first inpatient administration (`emar.event_txt = 'Administered'`) of drug A or comparator B with no administration of either in the prior 180 days (across admissions); baseline outcome-defining lab available and normal. Treatment strategies: initiate A vs initiate B (same indication class, e.g. piperacillin-tazobactam vs cefepime for AKI; trimethoprim-sulfamethoxazole vs doxycycline for hyperkalaemia; haloperidol vs olanzapine for QTc; heparin vs enoxaparin for thrombocytopenia). Follow-up: index to outcome, discharge, death or window end (7-14 days). Confounders measured at index: age, sex, admission type, Elixhauser groups from ICD-9/10, baseline lab value, eGFR, concurrent nephrotoxins/QT drugs/K-sparing drugs, ICU status, SOFA components where available. Propensity score by L2 logistic regression (or gradient boosting), 1:1 nearest-neighbour matching with caliper 0.2 SD of the logit; balance by standardised mean differences; risk ratio with cluster-robust CI; IPTW as sensitivity analysis.
5. **Empirical calibration.** For each outcome, 20-40 negative-control ingredients (no plausible mechanism, no label mention, no FAERS signal) run through the same pipeline; fit the systematic-error distribution; report calibrated p-values and CIs.
6. **Scoring.** EHR truth: calibrated RR >= 1.5 and calibrated p < 0.05 = positive; calibrated RR upper CI < 1.25 = negative; else indeterminate. PPV, sensitivity, specificity, AUROC of each FAERS statistic vs the EHR truth; logistic regression of "EHR-confirmed" on reporting features (RQ3).

Tools: `requests`, DuckDB, pandas, scikit-learn, scipy, statsmodels (optional for GEE), RxNav REST.

### Outcome definitions (MIMIC-IV labevents itemids; thresholds are parameters in `outcomes.make_outcomes`)

| Outcome | Labs / source | Baseline normal (within 7 d before index) | Incident event (window) |
|---|---|---|---|
| Hyperkalaemia | K 50971 / 50822 | K < 5.0 | K >= 5.5 mmol/L (7 d) |
| Hyponatraemia | Na 50983 / 50824 | Na >= 135 | Na < 130 mmol/L (7 d) |
| AKI | Creatinine 50912 | Cr < 1.5 mg/dL | KDIGO: >= 1.5x baseline or +0.3 mg/dL (7 d) |
| Hepatocellular injury | ALT 50861, bilirubin 50885 | ALT < 2x ULN and bilirubin < 2x ULN | ALT >= 3x ULN (14 d); severe if bilirubin >= 2x ULN |
| Thrombocytopenia | Platelets 51265 | >= 150 x10^9/L | < 100 or >= 50% drop (14 d) |
| Neutropenia | ANC 52075 | >= 1.5 x10^9/L | < 1.0 x10^9/L (14 d) |
| QTc prolongation | MIMIC-IV-ECG machine measurements (QT = t_end - qrs_onset, Bazett) | pre-index ECG with QTc < 500, QRS < 120 ms | QTc >= 500 ms or delta-QTc >= 60 ms (3 d) |

FAERS MedDRA PT groups per outcome are in `openfda_client.OUTCOME_PT` (British spelling, e.g.
HYPERKALAEMIA, ELECTROCARDIOGRAM QT PROLONGED, TORSADE DE POINTES).

### Quickstart with the starter code

```python
import numpy as np, pandas as pd, duckdb
from faers_ehr import openfda_client as ofc, disproportionality as dp, rxnorm, outcomes, cohort

# FAERS side: 2x2 tables and signal statistics (OPENFDA_API_KEY optional)
client = ofc.OpenFDAClient()
tbl = client.drug_event_table(["trimethoprim", "amiodarone", "vancomycin", "haloperidol"],
                              ["hyperkalaemia", "aki", "qt_prolongation"])
sig = dp.signal_table(tbl)              # PRR/chi2, ROR CI, IC/IC025, EBGM/EB05 (MGPS prior fitted), signal flags
strata = client.stratified_counts("amiodarone", "qt_prolongation", "primarysource.qualification")

# EHR side: exposures, outcomes, active-comparator target trial
con = duckdb.connect()
rx = pd.read_csv("data/mimiciv/3.1/hosp/prescriptions.csv.gz", usecols=["ndc", "drug"])
lookup = rxnorm.RxNavClient().map_prescriptions(rx)            # ndc/drug -> ingredient (cached)
con.register("ingredient_lookup", lookup)
exposures = con.execute(outcomes.exposures_sql("data/mimiciv/3.1")).df()
labs = con.execute(outcomes.labs_sql("data/mimiciv/3.1", ("potassium",))).df()
c = cohort.new_user_cohort(exposures, "trimethoprim", "doxycycline", washout_days=180)
c = outcomes.incident_outcome(labs, c, outcomes.make_outcomes()["hyperkalaemia"]).query("eligible")
ps = cohort.propensity_scores(c[covariate_cols], c["treated"])
m = cohort.match_nearest(ps, c["treated"], caliper_sd=0.2)
print(cohort.standardized_mean_differences(c[covariate_cols].iloc[np.r_[m.treated_idx, m.control_idx]],
                                           np.r_[np.ones(len(m)), np.zeros(len(m))]).max())
print(cohort.matched_risk_ratio(c["outcome"].to_numpy(), m))
# empirical calibration with negative-control drugs, then score FAERS signals
syserr = cohort.fit_systematic_error(nc["log_rr"], nc["se_log_rr"])
truth = sig.apply(lambda r: cohort.classify_pair(r.rr, r.rr_lo, r.rr_hi, syserr.calibrated_p(r.log_rr, r.se_log_rr)), axis=1)
print(dp.signal_performance(sig["ic"], truth.map({"positive": 1, "negative": 0}), sig["signal_ic"]))
```

`tests/test_faers_ehr.py` checks PRR/ROR/IC against hand-computed values, that the MGPS prior
fitted on a simulated drug x event table shrinks small counts more than large ones, that PS
matching removes confounding in a simulated cohort (SMD < 0.1 and RR closer to truth than the
crude estimate), and that empirical calibration widens naive p-values.

## Evaluation & statistics

- Unit of analysis: drug-outcome pair. Expected ~150-300 pairs with adequate EHR power (>= 50 exposed new users with a baseline lab).
- Power: with 500 matched pairs and 5% baseline risk, RR = 2 is detectable at 80% power; pairs below this are "indeterminate", never "negative".
- Multiplicity: Benjamini-Hochberg across pairs within outcome; empirical calibration handles systematic error; the PPV map reports CIs by Wilson intervals over pairs and a pair-level bootstrap.
- Nulls: (i) negative-control pairs must give calibrated RR ~ 1; (ii) shuffling FAERS signals across drugs within an outcome gives the chance PPV; (iii) a "sham" outcome (e.g. serum chloride change) with no plausible drug relation should show PPV at chance.
- Leakage / circularity: the EHR truth is computed without any use of FAERS; the negative-control list is fixed before results; labs used to define outcomes are excluded from the propensity model except as baseline values.
- Sensitivity: outcome thresholds (K >= 5.5 vs 6.0), window (7 vs 14 days), suspect-only vs any characterisation in FAERS, deduplication of FAERS cases (`safetyreportid` + `safetyreportversion`), MIMIC-IV ICU-only vs all inpatients.

## Publishable angle

Headline: "Across N drug-outcome pairs, only X% of FAERS disproportionality signals are confirmed by calibrated active-comparator EHR analyses; confirmation is predictable from reporting features, and Y% of EHR-confirmed lab-detected harms have no FAERS signal at all." Plus the released EHR-derived reference set with effect sizes.

Target venues: Drug Safety; Clinical Pharmacology & Therapeutics; Pharmacoepidemiology and Drug Safety; JAMIA. Methods/benchmark: OHDSI Symposium; AMIA Annual Symposium.

Follow-ups: repeat on eICU-CRD (multi-centre) and on AmsterdamUMCdb to test transportability of the reference set; extend FAERS features with text (narratives are not in openFDA, but reaction lists and outcomes are); use the reference set to retrain signal-detection thresholds; extend to MIMIC-IV-Note for symptom-defined outcomes (local NLP only, never third-party LLM APIs).

## Risks, confounds & mitigations

- MIMIC-IV is a single centre with inpatient follow-up only: outcomes are restricted to acute lab-defined events; sensitivity analysis on window length; multi-centre replication planned (eICU has labs and drug infusions but coarser exposure data).
- Confounding by indication survives PS adjustment: active comparators within class and empirical calibration with negative controls; report E-values for confirmed pairs.
- Detection bias (more labs drawn in sicker patients): require baseline and follow-up labs for both arms (equal monitoring), and adjust for the number of labs in the previous 48 h.
- Drug mapping errors: RxNorm-based, with a manual audit of the top-200 ingredients; report mapping coverage.
- FAERS duplicates and versions: use bulk files for deduplication by case id; API counts serve as the primary but are cross-checked against bulk.
- Small numbers for rare outcomes (neutropenia): treat as indeterminate; do not report negatives.
- MedDRA PT grouping choices: pre-registered PT lists per outcome; sensitivity with SMQ-like broad lists.

## Milestones

- [ ] openFDA extraction: 2x2 tables for all ingredients with >= 100 suspect reports x 7 outcome PT groups (+ stratified counts)
- [ ] Disproportionality table with PRR/ROR/IC025/EB05 and signal flags; MGPS prior fitted
- [ ] RxNorm mapping of MIMIC-IV prescriptions; coverage report
- [ ] Lab-defined outcome tables and baseline definitions validated against chart review of 50 cases per outcome
- [ ] Active-comparator pairs and negative-control lists pre-registered (OSF)
- [ ] PS-matched cohorts with balance diagnostics; empirical calibration
- [ ] PPV map, sensitivity gap, reporting-feature model
- [ ] Reference-set release (aggregate only) + manuscript

## Ethics / data-use notes

- openFDA data are public and de-identified; do not attempt re-identification and respect the API terms (rate limits; an API key is personal - keep it in `OPENFDA_API_KEY`, never in code or git).
- MIMIC-IV and MIMIC-IV-ECG linkage requires PhysioNet credentialing (CITI training, DUA). PhysioNet's responsible-use policy prohibits sharing the data with third parties, which includes sending patient-level data (including drug lists, lab values or ECG text reports) to third-party LLM APIs. Any NLP must run locally.
- Never commit data or patient-level intermediates (`data/`, `outputs/` are git-ignored). Released tables are aggregate with cell counts >= 10.
- Download scripts read `PHYSIONET_USER` / `PHYSIONET_PASS` and `OPENFDA_API_KEY` from the environment.
- The PPV map is a methodological result about signal-detection statistics, not clinical guidance; report it with the caveats about single-centre inpatient data.
