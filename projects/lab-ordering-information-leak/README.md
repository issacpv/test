# lab-ordering-information-leak

**How much of ICU outcome prediction on MIMIC-IV and eICU is *which labs were ordered, when and how urgently* rather than what they showed, and does order-agnostic training buy robustness when the ordering policy changes across 200 hospitals and 12 years?**

## Status / difficulty / timeline / compute

- Status: design + starter code (semi-synthetic generator with a tunable "ordering leak", value/mask feature builder, exact two-player Shapley attribution, thinning stress test, order-dropout training, intensity reweighting, DuckDB extraction templates for MIMIC-IV and eICU). No data is shipped.
- Difficulty: MSc thesis for the attribution and stress-test core; early-PhD for the full cross-hospital, cross-era and equity programme. The hard parts are cohort/feature hygiene and the statistics, not the models.
- Timeline: 5-8 months (1 month credentialing + extraction, 1 month features/tasks, 2 months attribution + shift experiments, 1 month order-agnostic methods, 1-2 months writing).
- Compute: one workstation with >= 32 GB RAM and a fast SSD (DuckDB over csv.gz; `labevents` is ~13 GB raw). No GPU needed for the primary models (logistic regression, gradient-boosted trees); optional GPU for a GRU-D / joint observation-process baseline.

## Background

ICU laboratory tests are not sampled at random. A clinician orders a lactate at 03:00 because they are worried, and the presence and timing of that order carries information about the patient's future that the value itself may not. Agniel, Kohane & Weber (BMJ, 2018) showed that the timing of lab orders predicts 3-year survival independently of the values; Lipton, Kale & Wetzel (MLHC, 2016) and Che et al. (Sci Rep, 2018; GRU-D) built recurrent models that consume missingness masks and time-since-last-measurement and found the masks alone carry most of the signal; Sharafoddini et al. (JMIR Med Inform, 2019) showed missingness indicators improve mortality prediction in MIMIC-III. The methodological reviews by Groenwold (Diagn Progn Res, 2020, "the curse of knowing what we don't know") and Sisk et al. (JAMIA, 2021, "informative presence and observation") make the point that this is *informative observation*, not classical MAR/MNAR missingness (Rubin, 1976).

The practical consequence is that every widely used ICU benchmark bakes the observation process into its features: MIMIC-III benchmark (Harutyunyan et al., Sci Data, 2019), MIMIC-Extract (Wang et al., CHIL, 2020), YAIB (van de Water et al., ICLR, 2024) all carry masks, counts or time-since-last features. Two problems follow. First, the order process is partly the *clinician's own forecast* of deterioration, so a model that learns it is learning a soft label leak rather than physiology. Second, ordering intensity is the most site-specific part of the data: daily-lab-reduction initiatives change it within a hospital, and it differs several-fold between hospitals and eras, so mask-heavy models are the ones that fail on transfer.

## The research gap

What has been done (2022-2026):

- **Informativeness is established.** Beyond the classic papers above, 2025 ICU work reports that an ordering-pattern-only model with *no lab values at all* reaches AUROC 0.849 versus 0.831 for a vitals-only model. A 2024 arXiv study (GRU-D on MIMIC-IV) characterises age-specific temporal missingness. 
- **Clinical presence shift has a name and a model.** Jeanselme et al. (ML4H, 2022) showed imputation choices under informative presence change subgroup performance; their DeepJoint line of work (arXiv, 2022; extended in a 2025 preprint on survival under clinical presence shift) jointly models the observation process to be robust to its shift.
- **Cross-hospital missingness shift is on the agenda.** A 2026 arXiv paper does missingness-aware conformal prediction under cross-hospital shift on eICU and MIMIC-IV; a 2026 medRxiv study on MIMIC-IV/eICU sepsis mortality finds observation-process features raise internal AUROC but worsen external calibration (the same finding motivates RQ5 of the sibling project `icu-model-transportability`).
- **Label leakage via same-admission diagnosis codes** was quantified in JAMA Network Open (2025), a different leak of the same family.

What is specifically missing (our angle):

1. **A quantitative attribution.** Nobody reports "X% of this benchmark's predictive information is the ordering channel". We compute an exact two-player Shapley split of held-out log-likelihood gain between values and masks, per task, horizon and model class, with bootstrap CIs.
2. **Anatomy of the leak.** MIMIC-IV is the only open ICU database with `labevents.priority` (STAT vs ROUTINE), `poe.ordertime` (order placed) distinct from `charttime` (specimen collected) and `storetime` (result available). No study has used these to separate *routine schedule* information (physiology-neutral) from *ad hoc / STAT* orders (clinician suspicion) and from *ordered-but-not-yet-resulted* states at prediction time.
3. **A stress test under ordering-policy shift**, not just "external AUROC": cross-hospital (eICU hospitals binned by ordering intensity), cross-era (MIMIC-IV `anchor_year_group` 2008-2010 to 2020-2022), cross-unit, and synthetic thinning curves, with degradation reported *as a function of the model's mask reliance*.
4. **Order-agnostic training as a designed trade-off.** Order-level dropout augmentation, schedule standardisation (routine-panel-only view), ordering-intensity importance weighting, and a joint observation-process model compared on the same internal-loss vs external-gain axes.
5. **Equity.** Ordering intensity differs by insurance, language and race (all in MIMIC-IV `admissions`); we test whether the mask channel encodes demographics and whether removing it narrows subgroup calibration gaps.

## Research questions / hypotheses

1. **RQ1 (attribution).** What fraction of the joint model's log-likelihood gain is attributable to the ordering channel? H1: >= 25% for 48-h mortality at a 24-h prediction time, larger at longer horizons (7-day, in-hospital) and for gradient-boosted trees than for logistic regression.
2. **RQ2 (anatomy).** Where does the ordering information live? H2: >= 60% of the mask channel's Shapley gain is carried by off-schedule/STAT orders in the last 6 h before prediction time; routine morning panels carry < 15%; "ordered but not resulted" flags carry a measurable extra share for short horizons.
3. **RQ3 (policy shift).** H3: across model configurations, the cross-hospital and cross-era AUROC drop correlates with mask share (Spearman rho > 0.6); calibration-in-the-large shift is the dominant failure mode and scales with the log ratio of ordering intensities.
4. **RQ4 (order-agnostic training).** H4: order-level dropout halves the degradation area under thinning and cross-hospital transfer at <= 0.01 internal AUROC cost; schedule standardisation costs more internally (>= 0.02) but transfers best in calibration.
5. **RQ5 (equity).** H5: mask-only models predict insurance (Medicaid vs private) and non-English language with AUROC > 0.60; removing the mask channel narrows subgroup calibration-intercept gaps by >= 30%.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (`hosp`, `icu`) | `labevents` (priority, charttime, storetime), `poe` (lab order times), `d_labitems`, `patients` (anchor_year_group), `admissions` (race, insurance, language, mortality), `icustays` | ~94k ICU stays; ~30 GB csv.gz for the modules, ~4 GB for the files used | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| eICU-CRD v2.0 | `lab`, `patient` (hospitalid, unittype, unitadmittime24), `hospital` (bed size, teaching status, region), `apachePatientResult` | ~200k stays, 208 hospitals | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/eicu-crd/2.0/ |
| YAIB task definitions | Cross-check cohort/task definitions | small | Open (MIT) | https://github.com/rvandewater/YAIB |
| Semi-synthetic cohort (this repo, `lab_leak.simulate`) | Ground truth for attribution and shift (known ordering informativeness gamma, known policy shift) | any | Generated locally | - |

## Methods

Pipeline (each step maps to a module in `src/lab_leak/`):

1. **Extraction** (`sql.py`). DuckDB templates produce a long table `(stay_id, itemid, t_hours, hod, valuenum, priority, result_delay_hours)` for 26 common labs (itemids listed in `MIMIC_LAB_ITEMS`, verified against `d_labitems` by `verify_itemids`; eICU `labname` strings in `EICU_LAB_NAMES`) plus a stays table with demographics, era, hospital and outcomes. A second template pulls `poe` rows of `order_type = 'Lab'` for order-placed times.
2. **Cohort and tasks.** Adults, first ICU stay, LOS >= 24 h (so a full observation window exists), prediction at 24 h after ICU admission. Tasks: 48-h mortality, in-hospital mortality, KDIGO stage >= 1 AKI in the next 48 h (creatinine criterion), ICU LOS > 7 days. Task definitions follow YAIB for comparability.
3. **Features** (`ordering.py`). Two aligned wide tables: `values` (last/mean/min/max per lab) and `masks` (count, any, hours since last, STAT count, off-schedule count per lab; stay totals). "Routine" = inside the morning draw window (default 04:00-08:00 local; hour of day is preserved by the MIMIC-IV date shift, and eICU provides `unitadmittime24`) and not STAT; everything else is "off-schedule". A third table of "ordered but not resulted at prediction time" flags is built from `storetime > prediction time`.
4. **Models** (`attribution.py`). Logistic regression (median imputation + scaling + L2) and HistGradientBoosting (native NaN). Optional GRU-D and a joint observation-process model (DeepJoint-style) as neural comparators.
5. **Attribution.** For each fold, four fits (base rate, V, M, V+M) give the exact two-player Shapley split of held-out mean log-likelihood gain; nested three-player splits (routine masks, off-schedule masks, values) need eight fits and localise the leak. Stay-level bootstrap (1000) for CIs on shares.
6. **Shift experiments.** (a) eICU: hospitals with >= 500 stays binned into ordering-intensity tertiles; train on one tertile, test on the others, hospital-grouped CV inside tertiles. (b) MIMIC-IV: train on 2008-2013 eras, test on 2017-2019 and 2020-2022. (c) Unit type (MICU vs CVICU vs SICU). (d) Synthetic thinning curves (`thinning_stress_curve`: keep 100/75/50/25/10% of test-time orders, all or off-schedule only) summarised by `degradation_area`.
7. **Order-agnostic training.** `fit_order_dropout` (train on replicas thinned to several intensities), `schedule_standardize` (routine-only view), `intensity_weights` (density-ratio weights on orders per stay), and the joint model. Report internal AUROC/log-loss cost vs external gain.
8. **Equity.** Mask-only models predicting insurance/language/race; subgroup calibration intercept and slope with and without the mask channel.

Tools: DuckDB, pandas/pyarrow, scikit-learn; optional PyTorch for neural baselines; YAIB for cohort cross-checks.

## Evaluation & statistics

- Validation: subject-level splits; hospital-grouped CV inside eICU; 5x2 CV for attribution; never tune on target hospitals/eras.
- Leakage prevention: only measurements with `charttime <= prediction time` enter values; `storetime` is used to flag results not yet available (those enter the mask table only, never the value table); outcomes use only post-prediction-time data; AKI baseline creatinine uses the prior 7 days.
- Metrics: AUROC, AUPRC, mean log-likelihood gain (nats/stay), Brier, calibration intercept/slope, ECE (10 quantile bins), Shapley shares, degradation area; stay-level bootstrap CIs (1000) and cluster (hospital) bootstrap in eICU.
- Multiple comparisons: 4 tasks x 3 horizons x 2 model classes x 4 shift settings; Benjamini-Hochberg across the family; H1-H5 pre-registered on OSF.
- Nulls and controls: (i) permute order times within stay (keeps counts, destroys timing) to isolate timing information; (ii) permute mask rows across stays matched on `n_total` (destroys which-lab information); (iii) simulator negative control gamma = 0 must give mask share ~ 0 and positive control gamma > 0 must recover a monotone share (both are in `tests/`); (iv) within-hospital random-split "transfer" must show no degradation.
- Sensitivity: imputation scheme (median vs MICE), morning window (03:00-07:00 vs 04:00-08:00), STAT definition, lab panel (26 vs 10 labs).

## Publishable angle

Headline result: "About a third of the 48-h mortality signal in MIMIC-IV lab features is clinicians' ordering behaviour, concentrated in STAT and off-schedule orders in the last six hours before prediction; models that depend on it lose most of their advantage at hospitals with different ordering intensity, and order-dropout training recovers that robustness for a 0.01 AUROC price." A table of ordering-channel shares per benchmark task would be reused by every group building ICU models.

Target venues: JAMIA; npj Digital Medicine; CHIL or ML4H (proceedings) for the methods; Critical Care Medicine or Intensive Care Medicine for the clinical message on ordering practice.

Follow-ups: the same attribution for vitals charting frequency (nursing attention), imaging orders (MIMIC-CXR order timing), ED labs (MIMIC-IV-ED); combine with the shift decomposition in `icu-model-transportability`; a "leak budget" reporting standard for EHR prediction papers.

## Risks, confounds & mitigations

- Hour-of-day preservation under date shifting: MIMIC-IV preserves time of day; verify day-of-week claims against the current MIMIC-IV documentation before using weekday features. eICU uses explicit clock times.
- eICU has no STAT/priority flag and no order-placed time: priority and order-latency analyses are MIMIC-IV only; eICU is used for the cross-hospital intensity shift.
- Ordering intensity is confounded with case-mix across hospitals: stratify by APACHE IV predicted mortality and compare within strata; report hospital-level covariates (teaching status, bed size).
- Imputed values also carry presence information: run median imputation and MICE and report both; include a "value-only with indicator-free MICE" arm.
- Two-player Shapley is exact but depends on the model class: report LR and GBT; add the neural comparators as sensitivity.
- "Order-agnostic" models may lose accuracy at the home site: this is the point of the paper; report the trade-off curve rather than a single winner.
- Small-cell risk in equity analyses: report subgroup results only for groups with >= 500 stays.

## Milestones

- [ ] PhysioNet credentialing complete (MIMIC-IV, eICU-CRD); files staged via `scripts/download_data.py`
- [ ] `verify_itemids` and eICU `labname` checks pass; marginal distributions compared across databases
- [ ] Simulator study: shares and thinning curves as a function of gamma and site shift (calibrates expectations)
- [ ] Cohorts and four task labels reproduced within +-2% of YAIB prevalence
- [ ] Attribution tables (Shapley shares) per task x horizon x model with bootstrap CIs
- [ ] Leak anatomy: routine vs off-schedule vs STAT vs not-yet-resulted splits; order-latency analysis with `poe`
- [ ] Cross-hospital (eICU tertiles), cross-era (MIMIC-IV) and thinning stress tests; degradation vs mask share
- [ ] Order-agnostic methods compared on internal cost vs external gain
- [ ] Equity analyses; pre-registration (OSF) before external evaluation; manuscript

## Ethics / data-use notes

- MIMIC-IV and eICU-CRD are PhysioNet credentialed resources: complete CITI training, sign each DUA, keep data on approved encrypted storage, and never redistribute rows or derived patient-level tables.
- PhysioNet's responsible-use policy prohibits sharing credentialed data with third parties, which includes sending records to third-party LLM APIs. All processing here is local (DuckDB, scikit-learn).
- Never commit data; `.gitignore` excludes `data/` and `outputs/`. Publish only aggregate results with cells >= 10.
- Credentials are read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`; never hard-code them.
- Findings about ordering by insurance/language/race are descriptive and should be reported with the caveats of the fairness literature; they describe practice variation, not patient traits.
