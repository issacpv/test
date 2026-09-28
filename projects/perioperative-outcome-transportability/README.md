# perioperative-outcome-transportability

**Why preoperative risk models fail to travel: a three-health-system transportability study (INSPIRE / Seoul, MOVER / Irvine, MIMIC-IV surgical cohort / Boston) of postoperative mortality, unplanned ICU admission and AKI models, with shift decomposition, calibration-first evaluation, a surgical-cohort-definition multiverse, and VitalDB as the within-site high-resolution subset.**

## Status / difficulty / timeline / compute

- Status: design + starter code (surgical-encounter definitions for MIMIC-IV, KDIGO AKI and ICU/mortality labellers, harmonised preoperative feature ontology with Charlson, shift decomposition with BBSE and importance weighting, calibration / few-shot recalibration, subgroup metrics). No data is shipped.
- Difficulty: MSc thesis to early-PhD. Hard parts: (i) harmonising three coding systems and lab dictionaries without leaking site identity into features, (ii) the surgical-cohort definition in MIMIC-IV, which has no anaesthesia record, (iii) statistics of the decomposition.
- Timeline: 7-10 months (2 months access + harmonisation, 2 months labels/features, 2 months modelling and decomposition, 1-2 months writing).
- Compute: workstation with >= 64 GB RAM (DuckDB over csv/parquet; MIMIC-IV `labevents` is ~120M rows). GPU not needed (gradient-boosted trees and logistic regression).

## Background

Preoperative risk calculators (ACS NSQIP, Bilimoria et al., 2013 J Am Coll Surg) and EHR-derived deep models (Lee et al., 2018 Anesthesiology; Fritz et al., 2019 Br J Anaesth; Hill et al., 2019 Br J Anaesth) predict postoperative mortality with AUROC 0.85-0.95 internally. Until recently every such model was developed and validated inside one health system. Three open perioperative datasets now make cross-system validation possible: VitalDB (Lee et al., 2022 Sci Data; 6,388 non-cardiac surgery cases from Seoul National University Hospital, 2016-2017, with 500 Hz waveforms), INSPIRE (Lim et al., 2024 Sci Data; ~130k surgical encounters, SNUH 2011-2020, with OR/ward/ICU vitals, labs from 6 months before to 6 months after admission, medications, diagnoses and death/ICU outcomes), and MOVER (Samad et al., 2023 JAMIA Open; 58,799 patients / 83,468 surgeries from UC Irvine Medical Center 2015-2022 with EHR and waveforms, released under a data-use agreement). MIMIC-IV (Johnson et al., 2023 Sci Data) contains tens of thousands of surgical admissions from Beth Israel Deaconess (2008-2019) identifiable through PACU transfers, ICD procedure codes and surgical services, although it has no anaesthesia record.

A first cross-continental study (medRxiv 2025, "Cross-Continental Transfer of Perioperative Mortality Prediction: Intraoperative Features Generalize Where Preoperative Features Fail"; INSPIRE n = 127,413 and MOVER n = 57,545, in-hospital mortality 1.1% vs 1.4%) reported internal AUROC 0.81-0.89 (INSPIRE-trained) and 0.91-0.95 (MOVER-trained) and found that models built on intraoperative physiology transfer better than models built on preoperative variables. A 2026 scoping review of end-to-end ML for surgical risk (arXiv:2607.29090) confirms that external validation across health systems remains rare and that calibration is seldom reported.

## The research gap

What has been done (2023-2026):

- INSPIRE <-> MOVER mortality transfer (medRxiv 2025): single-number AUROC per direction, preoperative vs intraoperative feature sets, no decomposition of *why* preoperative features fail, no calibration slope/intercept, no subgroup analysis, no third site.
- AKI after non-cardiac surgery with internal and external validation on open datasets (2024; PMC11204685): two datasets, discrimination-focused.
- VitalBench (arXiv:2511.13757): a multi-centre benchmark for *intraoperative vital-sign forecasting*, not outcome prediction.
- Several single-dataset INSPIRE or VitalDB outcome models (mortality, ICU admission, AKI), all internal validation.
- An important structural fact that the seed ("VitalDB + INSPIRE + MIMIC-IV") glosses over: VitalDB and INSPIRE come from the same hospital (SNUH) and overlapping years, so a VitalDB <-> INSPIRE comparison is a within-site temporal/resolution comparison, not a transportability test.

What is specifically missing (our angle):

1. A **third, differently coded site**: the MIMIC-IV surgical cohort (BIDMC; MetaVision/Epic era, ICD-9/10-CM and PCS, US academic centre) added to INSPIRE (Korea, ICD-10 diagnosis codes, ICD-10-PCS procedures) and MOVER (US, CPT/ICD-10). Six ordered site pairs instead of two.
2. A **decomposition of the preoperative transfer failure** into covariate shift (case-mix, procedure mix, lab availability), label shift (outcome prevalence: 1.1% vs 1.4% mortality, very different ICU-admission policies) and concept shift (different P(Y|X), e.g. different ICU-admission thresholds or different coding of comorbidities), with bootstrap CIs and a negative-control split.
3. **Calibration-first evaluation** and the sample size needed for few-shot recalibration at a new site (intercept-only / Platt / temperature / isotonic), i.e. what a hospital must label before deploying a foreign model.
4. A **surgical-cohort definition multiverse** for MIMIC-IV: PACU transfer vs surgical ICD procedure code vs surgical service vs combinations. Cohort size, outcome prevalence and transported AUROC as a function of the definition; the same exercise for "unplanned" ICU admission (which is a policy-laden label).
5. **Subgroup parity across sites** by sex and age band (race where available: MOVER and MIMIC-IV), asking whether a model that is calibrated within subgroups at the source stays so at the target.
6. **Within-site temporal and resolution checks** using VitalDB as the SNUH high-resolution subset (2016-2017) and INSPIRE year strata (2011-2020): how much of the "external" gap is already present as temporal drift inside one site?

Related project in this repo: `icu-model-transportability` (ICU databases; same decomposition philosophy). This project is independent code and targets the perioperative setting where preoperative features are the deployable ones.

## Research questions / hypotheses

1. **RQ1 (decomposition).** For each of the 6 site pairs and 3 outcomes (30-day in-hospital mortality, unplanned ICU admission within 24 h of surgery end, KDIGO stage >= 1 AKI within 7 days), what fraction of the AUROC and log-loss gap is due to covariate, label and concept shift? H1: for mortality, label shift dominates the calibration-intercept error while concept shift dominates the discrimination loss; for ICU admission, concept shift dominates everything (policy label); for AKI, covariate shift (creatinine measurement availability) dominates.
2. **RQ2 (calibration).** H2: every externally applied model has calibration slope < 1 and |intercept| > 0.3 on the logit scale; intercept error correlates with log prevalence ratio (Spearman rho > 0.7 across 18 pair-outcome combinations).
3. **RQ3 (few-shot recalibration).** H3: intercept-only recalibration with 200 labelled target encounters (about 2-3 events for mortality) removes >= 80% of calibration-in-the-large error; slope correction needs >= 2,000 encounters for mortality but <= 500 for AKI.
4. **RQ4 (definition multiverse).** H4: across MIMIC-IV surgical-cohort definitions, cohort size varies by > 2x and mortality prevalence by > 1.5x; the transported AUROC of an INSPIRE-trained model varies by >= 0.05 across definitions, i.e. definition choice is as large as the site effect.
5. **RQ5 (parity).** H5: subgroup calibration gaps (sex, age band) grow at external sites and are not closed by pooled recalibration; group-wise recalibration closes them at the cost of larger n.
6. **RQ6 (temporal vs geographic).** H6: the INSPIRE 2011-2013 -> 2018-2020 temporal gap in calibration intercept is >= 50% of the INSPIRE -> MOVER geographic gap.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| INSPIRE v1.3 (check PhysioNet for the current version) | Site A (SNUH, Korea, 2011-2020): operations, preop labs, diagnoses, medications, ICU and death outcomes, ward vitals | ~130k operations | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/inspire/ |
| MOVER | Site B (UCI Medical Center, USA, 2015-2022): patient info, surgery info, labs, ICD/CPT, post-op complications | 58,799 patients / 83,468 surgeries | UCI data-use agreement (free, signed) | https://mover.ics.uci.edu/ |
| MIMIC-IV v3.1 (hosp + icu) | Site C (BIDMC, USA, 2008-2019): transfers (PACU), procedures_icd, services, labevents, diagnoses_icd, icustays, admissions, patients | ~546k admissions; surgical subset ~50-90k depending on definition | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| VitalDB v1.0 | Within-SNUH high-resolution subset (2016-2017) for RQ6 and intraoperative sensitivity analyses | 6,388 cases | Open (API; accept terms) | https://vitaldb.net/dataset/ |
| MIMIC-IV Clinical Database Demo v2.2 | Pipeline dry-run without credentials | 100 patients | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| mimic-code concepts (MIT-LCP) | Reference SQL for Charlson, KDIGO creatinine, ICU stay details | small | Open (MIT) | https://github.com/MIT-LCP/mimic-code |

ASA class exists in INSPIRE and MOVER but not in MIMIC-IV; race exists in MOVER and MIMIC-IV but not in INSPIRE (single-ethnicity population). Features are harmonised to the intersection, with ASA and race used only in sensitivity analyses.

## Methods

Pipeline (module names in `src/periop_transport/`):

1. **Surgical encounter tables** (`cohorts.py`). INSPIRE: one row per `op_id` from `operations` (op start/end, admission/discharge, ICU-in, death time, department, `antype`, `emop`, ASA). MOVER: one row per surgery from the surgery-info table. MIMIC-IV: three candidate definitions computed per `hadm_id` — (D1) any `transfers.careunit == 'PACU'`; (D2) any `procedures_icd` code in a surgical range (ICD-9 01-86 operative chapters excluding diagnostic 87-99; ICD-10-PCS section 0 "Medical and Surgical" excluding root operations that are inspections/drainage-only); (D3) a surgical `services` entry (CSURG, NSURG, ORTHO, SURG, TSURG, VSURG, GU, GYN, ENT, PSURG). D4 = D1 AND D2, D5 = D1 OR D2, D6 = D2 AND D3. Surgery time in MIMIC-IV = PACU transfer intime (D1) or `procedures_icd.chartdate` (D2). `definition_multiverse()` reports sizes, pairwise Jaccard and outcome prevalence per definition.
2. **Labels** (`cohorts.py`). Mortality: in-hospital death within 30 days of surgery end. Unplanned ICU admission: ICU-in within 24 h of surgery end excluding cardiac, neuro- and transplant surgery (planned ICU) and patients already in ICU before surgery. AKI: KDIGO creatinine criteria over 7 postoperative days with baseline = lowest creatinine in the 7 days before surgery (else the last preoperative value within 30 days); requires >= 1 postoperative creatinine (missingness itself is analysed as covariate shift).
3. **Preoperative features** (`features.py`). Harmonised ontology of 12 labs (haemoglobin, WBC, platelets, sodium, potassium, creatinine, BUN, glucose, albumin, INR, total bilirubin, AST) — last value in the 30 days before surgery, plus a missing indicator; age, sex, emergency flag, anaesthesia type (general/regional/sedation), surgery group (11 body-system groups mapped from ICD-10-PCS second character, ICD-9 procedure chapter, CPT range or department), Charlson comorbidity index from ICD codes (abbreviated Quan et al., 2005 prefix mapping; cross-checked against mimic-code `charlson.sql`). `verify_ontology()` refuses to run when a site's lab label does not match the concept regex.
4. **Models**. L2 logistic regression and LightGBM per outcome and site; hyper-parameters by 5-fold grouped CV (by subject) on the source only. Feature sets: `preop_core` (intersection of all sites) and `preop_plus` (adds ASA where available).
5. **Shift decomposition** (`shift.py`). Covariate shift: domain classifier (gradient boosting) on X gives w(x) = P(T|x)/P(S|x) x n_S/n_T; evaluating the source model on source data weighted by w(x) yields the metric under target covariates with source P(Y|X). Label shift: Black-Box Shift Estimation (Lipton et al., 2018) gives w(y) = q(y)/p(y). Concept shift = remainder. Shapley average over both orderings; 1000-resample bootstrap by subject.
6. **Calibration and recalibration** (`shift.py`, `metrics.py`). Intercept, slope (logistic recalibration), ECE (10 quantile bins), Brier decomposition, decision curves; few-shot recalibration curves for n in {50, 100, 200, 500, 1000, 2000, 5000} with 200 repeats; group-wise variants.
7. **Subgroup parity** (`metrics.py`). Per-group AUROC, intercept, slope, ECE and TPR/FPR at the source operating point (alert rate 5%); max-min gaps with bootstrap CIs.
8. **Temporal check** (RQ6). INSPIRE year strata and VitalDB cases (mapped to INSPIRE by year/department, not by patient) as the within-site comparison.

Tools: DuckDB, pandas/pyarrow, scikit-learn, LightGBM, statsmodels, `vitaldb`.

## Evaluation & statistics

- Validation: train on source (grouped 5-fold CV for tuning; held-out 20% for internal metrics), evaluate on 100% of each other site. Target labels are used only in the few-shot experiment, with disjoint recalibration and evaluation samples.
- Leakage prevention: subject-level splits; features restricted to before surgery start; no measurement-count features in `preop_core` (they encode site practice; treated as a separate sensitivity set); site identity never enters a feature.
- Metrics: AUROC, AUPRC, log-loss, Brier, calibration intercept/slope with 95% CI, ECE, net benefit; decomposition components with bootstrap CIs.
- Multiple comparisons: 6 pairs x 3 outcomes x 3 components = 54 tests; Benjamini-Hochberg at q = 0.05; primary hypotheses H1-H4 pre-registered (OSF).
- Nulls: (i) decomposition between two random halves of the same site must yield components ~0; (ii) synthetic label shift (subsampling positives at the target) must be recovered by BBSE within its CI; (iii) permutation of subgroup labels for parity gaps.
- Sample-size note: mortality ~1.1-1.5% means ~1,400-1,900 events in INSPIRE and ~800 in MOVER; AKI (~5-10%) and ICU admission (~5-15%) are better powered. MIMIC-IV surgical mortality depends on the definition (RQ4).

## Publishable angle

Headline result: "Preoperative risk models lose most of their external discrimination to concept shift and most of their calibration to label shift; 200 labelled cases restore calibration-in-the-large but subgroup calibration requires group-wise recalibration; and the choice of how a 'surgical encounter' is defined in an EHR without an anaesthesia record moves external AUROC as much as changing continent." The definition-multiverse result (RQ4) is novel on its own for anyone using MIMIC-IV as a surgical cohort.

Target venues: British Journal of Anaesthesia (methods/data science); Anesthesiology; npj Digital Medicine; JAMIA; ML4H / CHIL (proceedings track) for the decomposition methodology.

Follow-ups: (a) add intraoperative features from VitalDB/MOVER waveforms and repeat the decomposition to explain why intraoperative features transfer better; (b) federated recalibration without sharing data; (c) extend to a fourth site if further perioperative datasets are released.

## Risks, confounds & mitigations

- **INSPIRE and VitalDB share patients** (same hospital, overlapping years). Mitigation: never treat them as independent sites; VitalDB is used only for within-site analyses; report any patient-level overlap if identifiers allow.
- **MOVER access and format changes.** MOVER is released under a UCI DUA and has had re-releases. Mitigation: pin the release date in `data/README.md`; keep loaders schema-driven.
- **No anaesthesia record in MIMIC-IV** (surgery time is approximate). Mitigation: RQ4 treats this as a first-class question; surgery time resolution is bounded by PACU transfer times (minute resolution) or `chartdate` (day resolution); sensitivity with both.
- **Coding-system heterogeneity masquerading as concept shift.** Mitigation: Charlson from both ICD versions with the same mapping; surgery-group mapping validated by two annotators on 200 sampled encounters per site; report agreement.
- **Different outcome ascertainment** (in-hospital vs 30-day death, ICU policy). Mitigation: restrict to in-hospital events within a fixed window; treat ICU admission explicitly as a policy label.
- **Rare outcomes** at MOVER (~800 deaths). Mitigation: bootstrap CIs; report AUPRC; do not over-interpret component estimates whose CIs include 0.
- **Date shifting in MIMIC-IV** removes calendar year but keeps `anchor_year_group`; temporal analyses use INSPIRE only.

## Milestones

- [ ] Credentialing (PhysioNet CITI for INSPIRE, MIMIC-IV) and MOVER DUA; accept VitalDB terms.
- [ ] Dry-run the pipeline on the MIMIC-IV demo (`scripts/download_data.py --sample`).
- [ ] Build surgical encounter tables for all sites; RQ4 multiverse table for MIMIC-IV.
- [ ] Labels (mortality, unplanned ICU, AKI) with prevalence table and missing-creatinine audit.
- [ ] Harmonised preoperative features; `verify_ontology()` passes on every site; annotator check of surgery groups.
- [ ] Internal models per site; external evaluation matrix (6 pairs x 3 outcomes).
- [ ] Shift decomposition with bootstrap CIs and negative controls.
- [ ] Calibration, few-shot recalibration curves, subgroup parity.
- [ ] RQ6 temporal analysis on INSPIRE/VitalDB.
- [ ] Pre-registration, manuscript, code release (no data).

## Ethics / data-use notes

- INSPIRE and MIMIC-IV are PhysioNet credentialed (CITI training + DUA); MOVER requires the UCI data-use agreement; VitalDB requires acceptance of its terms. No data are redistributed and `data/` is git-ignored.
- Credentials come from environment variables (`PHYSIONET_USERNAME`, `PHYSIONET_PASSWORD`); never commit them.
- Per PhysioNet's responsible-use policy, credentialed data must not be sent to third-party LLM or cloud APIs; all modelling is local.
- Cross-country comparisons invite ecological over-interpretation; site names are reported, but differences are framed as data-generating-process differences, not quality judgements.
- Report subgroup results with cell sizes >= 11.
