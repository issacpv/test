# icu-model-transportability

**Where and why ICU prediction models break across four open ICU databases: a shift-decomposition and subgroup-fairness study of sepsis, AKI and 48-h mortality models over MIMIC-IV, eICU-CRD, HiRID and AmsterdamUMCdb, with few-shot site recalibration.**

## Status / difficulty / timeline / compute

- Status: design + starter code (cohort SQL templates, shift decomposition, calibration and parity metrics). No data is shipped.
- Difficulty: MSc thesis to early-PhD level. The hard parts are (i) getting the shared feature ontology right on four databases and (ii) the statistics of the decomposition, not the models.
- Timeline: 6-9 months (2 months data access + harmonisation, 2 months modelling, 2 months shift/fairness analysis, 1-2 months writing).
- Compute: one workstation with >=64 GB RAM and a fast SSD (DuckDB over the raw CSV/parquet files; HiRID and eICU are the largest at tens of GB). GPU optional (gradient-boosted trees are the main models; a GRU baseline is optional).

## Background

Every widely used open ICU database now has a benchmark: MIMIC-III/IV (Johnson et al., 2016 Sci Data; Johnson et al., 2023 Sci Data), eICU-CRD (Pollard et al., 2018 Sci Data), HiRID (Hyland et al., 2020 Nat Med; Faltys et al., PhysioNet) and AmsterdamUMCdb (Thoral et al., 2021 Crit Care Med). Two harmonisation efforts make it possible to run the same task definition on all four: the `ricu` R package (Bennett et al., 2023 GigaScience; 119 clinical concepts mapped to OMOP-grounded ids for MIMIC-III/IV, eICU, HiRID, AUMCdb) and YAIB, "Yet Another ICU Benchmark" (van de Water et al., 2024 ICLR), which builds on ricu to define mortality, AKI, sepsis, kidney-function and length-of-stay tasks and to train/transfer models across databases.

Cross-database performance drops are well documented (Moor et al., 2023 eClinicalMedicine, sepsis across MIMIC-III, eICU, HiRID, AUMCdb; van de Water et al., 2024; Wong et al., 2021 JAMA Intern Med for the Epic sepsis model). What is almost never reported is *why* a model fails at a new site. The distribution-shift literature (Lipton et al., 2018 ICML for label shift; Subbaswamy & Saria, 2020 Biostatistics; Finlayson et al., 2021 NEJM) gives a clean vocabulary — covariate shift P(X), label shift P(Y), concept shift P(Y|X) — but ICU transfer papers report a single AUROC delta and stop. Calibration, the quantity that actually matters for a threshold-based alert, is reported even less often, and subgroup performance across sites almost never.

## The research gap

What has been done (2023-2026):

- YAIB (van de Water et al., 2024 ICLR) shows within- and cross-database results for five tasks on MIMIC-IV, eICU, HiRID and AUMCdb, using AUROC/AUPRC. It does not decompose the transfer gap, does not report calibration slope/intercept, and does not report subgroup metrics.
- Moor et al. (2023 eClinicalMedicine) trained deep sepsis models across international sites and showed large transfer losses; again a single-number gap, no decomposition, no fairness.
- A multi-centre study on HiRID, MIMIC-IV and eICU (216,536 stays; medRxiv 2025, "Evaluating deep learning sepsis prediction models in ICUs under distribution shift", later in npj Digital Medicine 2026) quantified feature-level shift (18-29 features per site pair differ in all summary statistics) and compared deployment strategies. It is the closest work; it does not attribute performance loss to covariate vs label vs concept shift, does not include AUMCdb, and does not test few-shot recalibration or subgroup parity.
- Preprints on calibration drift for ICU mortality across institutions (medRxiv 2026, "Calibration drift under cross-institutional deployment") report that discrimination transfers but calibration intercepts drift, attributing this to prevalence-driven label shift and showing intercept-only correction helps. This is two-database work (US only), mortality only, and does not separate label from concept shift.
- A cross-database MIMIC-IV/eICU study (medRxiv 2026, "Observation-process features are associated with larger domain shift in sepsis mortality prediction") shows that measurement-frequency features improve internal AUROC but worsen external calibration — an important feature-choice confound our design must control.
- Domain-generalisation methods (anchor regression for ICU, arXiv 2025) and data interventions for subgroup fairness in the ICU (arXiv 2026) exist, but are single-task or single-database.

What is specifically missing (our angle):

1. A **formal decomposition** of the cross-site performance gap into covariate, label and concept components for each of the 12 ordered site pairs and 3 tasks, with confidence intervals. Nobody has published this on four ICU databases.
2. A **calibration-first** view: calibration-in-the-large, slope, ECE and decision-curve net benefit across sites, not only AUROC.
3. **Subgroup parity across sites**: does a model that is fair (in the calibration/equalised-odds sense) at the source site stay fair at the target site, by sex, age band and (where available) race?
4. **Few-shot site adaptation**: how many labelled target-site stays are needed for intercept-only / Platt / temperature / isotonic recalibration to restore calibration, and does recalibration also fix subgroup calibration gaps?
5. Doing 1-4 on a **verified shared ontology** (ricu/YAIB concept ids, cross-checked programmatically against each database's item dictionary) so that "shift" is not an artefact of mismatched variables.

## Research questions / hypotheses

1. **RQ1 (decomposition).** For each task and site pair, what fraction of the AUROC / Brier / log-loss gap is explained by covariate shift (importance-weighted source evaluation), label shift (BBSE-estimated prior change), and the residual concept shift? H1: label shift dominates the calibration-intercept drift; concept shift dominates the discrimination loss for sepsis (because Sepsis-3 operationalisation depends on local antibiotic/culture practice); covariate shift dominates for AKI (creatinine and urine-output measurement frequency).
2. **RQ2 (calibration).** H2: calibration slope < 1 (overfitting-like) and intercept != 0 at every external site; the intercept magnitude correlates with log(prevalence ratio) between sites (Spearman rho > 0.7).
3. **RQ3 (parity transfer).** H3: subgroup calibration gaps (by sex, age band) are larger at external sites than internally, and the increase is not explained by subgroup prevalence differences alone (i.e., it persists after label-shift correction).
4. **RQ4 (few-shot recalibration).** H4: intercept-only recalibration with n = 100 labelled target stays removes >= 80% of the calibration-in-the-large error; slope correction needs n >= 500; isotonic needs n >= 2000. H4b: recalibration on the pooled target sample does not close subgroup calibration gaps; group-wise recalibration does, at the cost of larger n.
5. **RQ5 (ontology sensitivity).** H5: replacing "measurement-count" features by clinical values only reduces the concept-shift component but lowers internal AUROC (replicating the observation-process finding across four sites).

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (hosp + icu modules) | Source site A (US, BIDMC). Vitals, labs, meds, outputs, demographics incl. race | ~94k ICU stays; ~30 GB csv.gz | PhysioNet credentialed (CITI "Data or Specimens Only Research" + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| eICU-CRD v2.0 | Site B (US, 208 hospitals, multi-centre) incl. ethnicity | ~200k ICU stays; ~20 GB csv.gz | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/eicu-crd/2.0/ |
| HiRID v1.1.1 | Site C (Bern, Switzerland; 2-min resolution) | ~34k stays; ~40 GB raw | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/hirid/1.1.1/ |
| AmsterdamUMCdb v1.0.2 | Site D (Amsterdam, NL) | ~23k admissions; ~80 GB csv | Free registration + end-user licence via Amsterdam Medical Data Science (not PhysioNet) | https://amsterdammedicaldatascience.nl/amsterdamumcdb/ |
| ricu concept dictionary (R package, CRAN) | Ground truth for concept ids | small | Open (GPL-3) | https://github.com/eth-mds/ricu |
| YAIB cohorts / task definitions | Task definitions and baseline configs | small | Open (MIT) | https://github.com/rvandewater/YAIB |

Race/ethnicity is available only in MIMIC-IV and eICU; HiRID and AUMCdb provide sex and (banded) age only. AUMCdb ages are 10-year bands, which fixes the age-band definition for all sites.

## Methods

Pipeline (each step is a script or module in `src/icu_transport/`):

1. **Extraction.** DuckDB SQL templates over the raw CSV/parquet files (`cohorts.py`). One template per database for the stay table, one per concept per database for the time series. Output: long table `(stay_id, hours_since_admit, concept, value)`.
2. **Shared ontology.** 38 concepts (vitals, blood gases, chemistry, haematology, GCS, urine output, vasopressor rate, ventilation flag, antibiotics, cultures) with per-database ids taken from ricu/YAIB. `verify_ontology()` checks every id against the database's own item dictionary (`d_items`, `d_labitems`, eICU `lab.labname`, HiRID `hirid_variable_reference.csv`, AUMCdb `numericitems.item`) and refuses to run when the label does not match the concept regex.
3. **Cohorts and labels** (YAIB-compatible): adults, first ICU stay, LOS >= 6 h. Prediction time 24 h after ICU admission with a 24-h observation window. Labels: 48-h mortality (death within 48 h of prediction time), KDIGO stage >= 1 AKI in the next 48 h (creatinine + urine-output criteria, baseline = lowest creatinine in the previous 7 days else first value), Sepsis-3 onset in the next 12 h (suspected infection = antibiotic start and culture within 72/24 h per Seymour et al., 2016 JAMA; SOFA increase >= 2 in the [-48 h, +24 h] window). eICU culture timing is sparse; the sepsis task there uses the ricu `susp_inf` fallback and is flagged as lower-confidence.
4. **Features.** Hourly grid, forward-fill with a per-concept maximum carry-forward horizon, mean/min/max/last/count per 24-h window, missingness indicators. Two feature sets: `clinical` (no counts) and `clinical+process` (adds measurement counts) to test RQ5.
5. **Models.** Logistic regression (L2) and LightGBM per task and source site; optional GRU via YAIB for comparison. Hyper-parameters selected by 5-fold grouped CV on the source site only.
6. **Shift decomposition** (`shift.py`). Covariate shift: a domain classifier (LightGBM) on X gives importance weights w(x) = P(T|x)/P(S|x) * n_S/n_T; evaluating the source model on source data weighted by w(x) gives the metric the model would have under the target covariate distribution with source P(Y|X). Label shift: Black-Box Shift Estimation (Lipton et al., 2018) from the source confusion matrix and the target predicted-label distribution gives w(y) = q(y)/p(y). Concept shift: the remainder after both reweightings. Because the sequential attribution depends on order, we report the Shapley average over the two orderings and bootstrap CIs (1000 resamples of stays).
7. **Calibration** (`calibration.py`). Calibration-in-the-large (intercept), slope (logistic recalibration of logit(p)), ECE with quantile bins, Brier decomposition, decision curves (Vickers & Elkin, 2006) at 5-30% thresholds.
8. **Few-shot recalibration** (`calibration.py`). For n in {25, 50, 100, 200, 500, 1000, 2000}, draw n labelled target stays, fit intercept-only / Platt / temperature / isotonic, evaluate on the remaining target stays; 200 repeats. Also group-wise versions.
9. **Subgroup parity** (`fairness.py`). Per-group AUROC, AUPRC, intercept, slope, ECE, TPR/FPR at the source-site operating point (alert rate 10%); max-min gaps with bootstrap CIs and permutation nulls.

Tools: DuckDB, pandas/pyarrow, scikit-learn, LightGBM, YAIB (optional, for cohort cross-checks), `ricu` (R, for concept-id cross-checks).

### Cohort and label definitions (shared across the four databases)

| Item | Definition | Notes |
|---|---|---|
| Population | Adults (>= 18), first ICU stay per patient, LOS >= 6 h | AUMCdb age is banded; band `18-39` is the adult floor |
| Prediction time | 24 h after ICU admission (12 h in sensitivity analysis) | Observation window = [0, 24) h |
| 48-h mortality | Death in (24, 72] h after ICU admission | HiRID: last observation time for `discharge_status = 'dead'` |
| AKI | KDIGO stage >= 1 in (24, 72] h: creatinine +0.3 mg/dL / 48 h or >= 1.5x baseline (min of previous 7 d), or urine output < 0.5 mL/kg/h for 6 h | Weight from the admission weight concept; AUMCdb weight is banded |
| Sepsis-3 onset | Suspected infection (antibiotic + culture within 72 h / 24 h) and SOFA rise >= 2 in [-48, +24] h; onset in (24, 36] h; prevalent cases (onset <= 24 h) excluded | eICU culture timing sparse -> secondary |
| Subgroups | Sex (all sites), age band (all), race/ethnicity (MIMIC-IV, eICU; harmonised to 5 groups) | Minimum 50 stays per subgroup for metrics |
| Feature sets | `clinical`: mean/min/max/last of 38 concepts + missingness; `clinical+process`: adds measurement counts | RQ5 |

### Quickstart with the starter code

```python
import duckdb, numpy as np, pandas as pd
from icu_transport import cohorts, shift, calibration, fairness

con = duckdb.connect()
root = "data/raw/physionet.org/files/mimiciv/3.1"
print(cohorts.verify_ontology(con, "mimiciv", root).query("status != 'ok'"))   # must be empty
stays = cohorts.load_stays(con, "mimiciv", root)
long = pd.concat([cohorts.load_concept(con, "mimiciv", c, root, max_hours=72)
                  for c in ["hr", "map", "creatinine", "platelets", "bilirubin", "norepinephrine"]])
grid = cohorts.hourly_grid(long, stays, n_hours=72)
X = cohorts.window_features(grid, 0, 24, include_counts=False)
y = cohorts.label_mortality(stays, pred_hour=24, horizon_h=48).reindex(X.index)

# ... train a model on the source site, then on a target site:
res = shift.full_decomposition(model.predict_proba_fn, X_src_holdout, y_src_holdout, X_tgt, y_tgt,
                               metrics=("auroc", "brier", "citl"), n_boot=200)
print({m: r.as_dict() for m, r in res.items()})          # covariate / label / concept components + CIs
print(calibration.calibration_report(y_tgt, p_tgt))       # intercept, slope, ECE, ICI, Brier decomposition
curve = pd.DataFrame(calibration.few_shot_learning_curve(p_tgt, y_tgt, groups=sex_tgt))
thr = fairness.operating_threshold(p_src_holdout, alert_rate=0.10)
print(fairness.subgroup_metrics(y_tgt, p_tgt, sex_tgt, thr))
```

The synthetic tests (`tests/test_shift_calibration.py`) show the expected behaviour of every
function: BBSE recovers an induced prior change, the domain classifier stays at AUROC ~0.5 for
two halves of one site, the decomposition attributes a pure label shift to the label component,
and intercept-only recalibration removes calibration-in-the-large error but not slope error.

## Evaluation & statistics

- Validation: train on site S (5-fold grouped CV by subject for tuning), evaluate on the held-out 20% of S and on 100% of each other site. Never tune on target data except in the few-shot experiment, where the recalibration sample is disjoint from the evaluation sample.
- Leakage: patient-level splits; all features restricted to the observation window; labels use only post-prediction-time data; in MIMIC-IV, `hosp` labs charted before ICU admission are excluded unless within the 24-h window.
- Metrics: AUROC, AUPRC, Brier, log-loss, calibration intercept/slope with 95% CI, ECE (10 quantile bins), net benefit; decomposition components with bootstrap CIs.
- Multiple comparisons: 3 tasks x 12 site pairs x 3 components; report all with Benjamini-Hochberg-adjusted p-values for "component != 0"; primary hypotheses are pre-registered (H1-H4).
- Nulls: (i) the decomposition run between two random halves of the same site must give components ~0 (negative control); (ii) permutation of subgroup labels for parity gaps; (iii) a synthetic-shift positive control where we induce a known label shift by subsampling positives at the target and check BBSE recovers it.
- Sensitivity: feature set (`clinical` vs `clinical+process`), model family, sepsis definition variants (ricu `sep3` vs YAIB), prediction time (12 h / 24 h).

## Publishable angle

Headline result: "Across four ICU databases and three tasks, most of the transfer loss in calibration is label shift and is fixable with ~100 labelled stays, whereas the transfer loss in discrimination for sepsis is concept shift and is not fixable by recalibration at any sample size; subgroup calibration gaps widen at external sites and require group-wise recalibration." A quantitative table of (covariate, label, concept) fractions per task and site pair would be a reference for the field.

Target venues: npj Digital Medicine; Journal of the American Medical Informatics Association; Critical Care Medicine (clinical audience); ML4H or CHIL (proceedings track) for the methods paper.

Follow-ups: extend to temporal shift within MIMIC-IV (anchor_year groups); add SICdb (Salzburg) and NWICU; test shift-stable learning (anchor regression, DRO) against the decomposition; a "site-adaptation budget" calculator for hospitals adopting a published model.

## Risks, confounds & mitigations

- Ontology mismatch masquerading as shift: mitigated by `verify_ontology()`, unit conversion tests, and comparing marginal distributions of every concept per site before modelling (flag any concept with a > 3-fold median difference for manual review).
- Sepsis-3 operationalisation differs by site (antibiotic tables, culture availability): report the sepsis task with two definitions and treat eICU sepsis as secondary.
- Time resolution (HiRID 2 min vs eICU 5 min vs MIMIC hourly charting): aggregate to hourly before feature computation; include count features only in the `clinical+process` set.
- eICU is multi-centre: hospital-level clustering inflates apparent within-site performance; use hospital-grouped CV within eICU and report hospital-level heterogeneity.
- Race categories are not comparable across countries: race analyses limited to MIMIC-IV and eICU with harmonised categories (White, Black, Hispanic, Asian, Other/Unknown).
- Decomposition is not unique when shifts interact: report the Shapley-averaged attribution and both orderings; show the negative-control run.
- AUMCdb has banded age and weight: age band is the shared age variable everywhere.

## Milestones

- [ ] Credentialing complete for PhysioNet (MIMIC-IV, eICU, HiRID) and AUMCdb licence signed
- [ ] `verify_ontology()` passes on all four databases; concept marginals compared
- [ ] Cohorts and three labels reproduced within +-2% of YAIB prevalence per database
- [ ] Source-site models trained (LR, LightGBM) for 3 tasks x 4 sites
- [ ] Cross-site evaluation table (12 ordered pairs x 3 tasks) with calibration metrics
- [ ] Shift decomposition with bootstrap CIs and negative/positive controls
- [ ] Subgroup parity tables and permutation tests
- [ ] Few-shot recalibration learning curves (pooled and group-wise)
- [ ] Sensitivity analyses (feature set, sepsis definition, prediction time)
- [ ] Pre-registration (OSF) before external evaluation; manuscript

## Ethics / data-use notes

- MIMIC-IV, eICU-CRD and HiRID are PhysioNet credentialed resources: complete CITI training, sign each DUA, and keep the data on approved, encrypted storage. AmsterdamUMCdb has its own end-user licence; the same handling applies.
- PhysioNet's responsible-use policy prohibits sharing credentialed data with third parties, which includes sending records or free text to third-party LLM APIs. Do not paste rows, notes or item labels derived from patient data into hosted LLMs; use local tooling only.
- Never commit data or derived patient-level tables. `.gitignore` excludes `data/` and `outputs/`. Only aggregate results (counts >= 10 per cell) may be published.
- Downloader scripts read `PHYSIONET_USER` / `PHYSIONET_PASS` from the environment; never hard-code credentials.
- Subgroup analyses are descriptive; report them with the cautions of the fairness literature (race is a social construct captured inconsistently across sites).
