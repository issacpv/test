# recall-lag-survival

**From first serious MAUDE report to recall: a survival analysis of the lag between adverse-event signals and recall initiation for FDA-regulated devices, decomposed into its reporting, detection and regulatory components, and stratified by device class, regulatory pathway, root cause and firm, using openFDA only.**

## Status / difficulty / timeline / compute

- Status: design + starter code (openFDA client with cursor pagination and the bulk manifest, device-key construction and recall linkage, first-event and Poisson-CUSUM signal dates, left-truncated time-to-event table builder, Kaplan-Meier with Greenwood CIs, log-rank test, Cox PH via statsmodels `PHReg` with entry times, lag decomposition). No data is shipped; all sources are open.
- Difficulty: MSc-level. The statistics are standard survival methods; the difficulty is the entity linkage between MAUDE device/manufacturer fields and recall records, and the careful handling of truncation and reverse causation.
- Timeline: 4-6 months (1 month data + linkage, 1 month signal definitions and validation, 1-2 months survival analyses and ITS, 1 month writing).
- Compute: laptop; MAUDE bulk JSON (~20 GB) processed once into monthly counts per device key with DuckDB or streaming.

## Background

Device recalls are the main post-market corrective action in the US. Their determinants have been studied from the pre-market side: Zuckerman et al. (2011, Arch Intern Med) on high-risk recalls of 510(k)-cleared devices; Dubin et al. (2021, JAMA Netw Open) on recall risk by pathway; Everhart et al. (2023, JAMA) on submission characteristics associated with recalls of 510(k) devices; and Lee et al. (2025, JAMA Health Forum) on AI-enabled devices, nearly half of whose recall events occurred within a year of authorisation. The post-market side is the Manufacturer and User Facility Device Experience database (MAUDE), whose structure and linkage are described by Ensign & Cohen (2017, eGEMs). A 2025 study measured *late* manufacturer reporting to MAUDE (days from manufacturer awareness to FDA receipt, and the proportion beyond the 30-day requirement; PMID 40081838), and the GAO (2026, GAO-26-107619) reviewed FDA's recall process using Recall Enterprise System data for FY2020-2024 and identified limitations in oversight. Machine-learning recall predictors exist (a 2025 West J Emerg Med study using Google Trends and PubMed signals with lead times up to 12 months; RecallRisk-BERT, arXiv 2026, on MAUDE narratives). Regulatory-affairs practice treats clusters of MAUDE reports as an early warning of recalls, and there are documented cases where reports accumulated for months before a Class I recall.

What no published study has done is to treat the *time* between the first signal in MAUDE and the recall as the outcome.

## The research gap

What exists (2023-2026): recall risk factors (cross-sectional, pre-market predictors); late-reporting delays (the event -> FDA-receipt component only); recall-prediction classifiers (binary, with lead time reported as a by-product); recall process audits (GAO) that do not use MAUDE.

What is missing (our angle):

1. **A time-to-event model of the MAUDE-to-recall lag** with the right observation scheme: origin = first serious report (death/injury) or a CUSUM signal on monthly counts; event = recall initiation (`event_date_initiated`); right-censoring at the data cut; left truncation for device keys whose origin precedes the start of reliable recall records; exclusion of reports that follow the recall (publicity-driven reporting = reverse causation).
2. **Decomposition of the lag** into (a) event -> FDA receipt (reporting delay), (b) receipt -> detectable signal (cluster threshold), (c) signal -> firm-initiated recall, (d) initiation -> FDA classification/posting (`center_classification_date`, `event_date_posted`), each with its own distribution and covariates.
3. **Stratification** by recall class (I/II/III), pathway (510(k)/PMA/De Novo), device class, panel, recall root cause (software vs design vs manufacturing), firm size (number of listed devices), and software/AI status; Cox models with firm-level clustering.
4. **Policy change points**: the end of the Alternative Summary Reporting programme and the release of its data in 2019 changed what is visible in MAUDE; an interrupted-time-series on the lag by recall-initiation year tests whether lags shortened afterwards.
5. **Detection lead time as a design quantity**: for a given false-alarm rate of a simple Poisson CUSUM on MAUDE counts, what fraction of eventually recalled device keys would have been flagged >= 6 months before initiation? This turns the survival distribution into an actionable number for surveillance system design and complements the classifier work in `device-recall-prediction`.

## Research questions / hypotheses

1. **H1 (lag magnitude).** Among device keys with >= 1 serious report before a recall, the median lag from first serious report to recall initiation exceeds 12 months for Class II recalls; Class I recalls have shorter lags (Cox HR for recall > 1 vs Class II, adjusting for panel and pathway).
2. **H2 (pathway).** PMA devices have shorter signal-to-recall lags than 510(k) devices in the same panel (HR > 1.2), consistent with more intensive post-market obligations.
3. **H3 (firm size).** Larger firms have shorter signal-to-initiation lags but longer event-to-receipt (late reporting) components; the two components are negatively correlated across firms.
4. **H4 (root cause).** Software-related recalls have shorter lags than design/manufacturing recalls (software faults cluster quickly in MAUDE).
5. **H5 (policy).** Lags for recalls initiated after 2019 are shorter than before, beyond the secular trend (interrupted time series with a 2019 break).
6. **H6 (lead time).** A Poisson CUSUM on monthly serious-report counts with a false-alarm rate <= 1 per 100 device-key-years flags >= 50% of eventually recalled keys >= 6 months before recall initiation.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| openFDA `device/event` (MAUDE) | Report dates (`date_of_event`, `date_received`), event type, product code, manufacturer, brand, UDI | millions of reports; bulk partitions | Open (optional `OPENFDA_API_KEY`) | https://open.fda.gov/apis/device/event/ |
| openFDA `device/recall` | Recall events: `event_date_initiated`, `event_date_posted`, `product_code`, `recalling_firm`, `root_cause_description`, `k_numbers`, `pma_numbers`, `res_event_number` | ~100k | Open | https://open.fda.gov/apis/device/recall/ |
| openFDA `device/enforcement` | Recall classification (I/II/III), `recall_initiation_date`, `center_classification_date` | ~100k | Open | https://open.fda.gov/apis/device/enforcement/ |
| openFDA `device/510k`, `device/pma`, `device/classification` | Pathway, decision dates, applicant, device class, panel | ~170k 510(k) | Open | https://open.fda.gov/apis/device/ |
| openFDA `device/registrationlisting` | Firm size proxy (number of listed devices per firm) | large | Open | https://open.fda.gov/apis/device/registrationlisting/ |
| FDA AI-enabled device list | Software/AI status flag | > 1,300 | Open (manual download) | https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices |

## Methods

1. **Device keys** (`linkage.py`). Primary key = product code x normalised recalling/manufacturer firm name; secondary key = 510(k)/PMA number where recalls list `k_numbers`/`pma_numbers` (then linked to MAUDE via the submission's product code and applicant). Firm names normalised (case, punctuation, legal suffixes); a manual alias table for large firms with many subsidiaries.
2. **Signals** (`signals.py`). Monthly counts of all and serious (death/injury) reports per key; first-serious-event date; Poisson CUSUM with baseline from the first 12 months (or the product-code average for sparse keys), detection multiplier 2 and threshold h chosen for a target false-alarm rate on never-recalled keys; signal date = first month the CUSUM crosses h.
3. **Time-to-event table** (`linkage.py: build_time_to_event`). Per key: origin (first serious report / CUSUM signal / authorisation date, as sensitivity analyses), recall initiation (first recall after origin), censoring at data cut, left-truncation entry time when origin precedes the reliable start of recall records (2002 for RES data; sensitivity 2010); keys whose only recall precedes the origin are tabulated separately ("recall before signal"). Reports after the recall initiation are never used for signals.
4. **Survival models** (`survival.py`). Kaplan-Meier with Greenwood CIs and left truncation; log-rank tests; Cox PH (`statsmodels PHReg` with `entry`) with cluster-robust SEs by firm; covariates: recall class (time-fixed at the recall, so used for description and for a Fine-Gray-style sub-analysis rather than as a Cox covariate), pathway, device class, panel, root cause, firm size tertile, software/AI flag, origin year; Schoenfeld residual checks; AFT (log-normal) models for the lag decomposition components.
5. **Lag decomposition.** Per recalled key: median event -> receipt (from reports before the recall), receipt -> signal, signal -> initiation, initiation -> classification, initiation -> posting; stacked bar summaries by stratum; correlations across firms (H3).
6. **ITS** (H5). Segmented regression of median log-lag by recall-initiation quarter with a break at 2019 Q3, Newey-West SEs.
7. **Lead time** (H6). CUSUM lead-time distribution vs false-alarm rate curve (ROC-like) across h.

Tools: requests, pandas, DuckDB, statsmodels (`PHReg`, GLM, OLS with HAC), SciPy; `lifelines` optional for Fine-Gray/AFT cross-checks; rapidfuzz optional.

## Evaluation & statistics

- Unit: device key; clustering by firm for all inference; 95% CIs; Benjamini-Hochberg across strata; H1-H3 pre-registered.
- Truncation and censoring handled explicitly (entry times in KM/Cox); sensitivity to the reliable-start date and to the key definition (product code x firm vs submission number).
- Reverse causation: only reports before recall initiation contribute to signals; a negative control uses reports received in the 90 days *after* initiation, which should show the publicity spike and not be predictive of anything.
- Immortal-time bias: keys enter the risk set at the origin, never earlier; keys with zero reports before recall are described separately (they are informative about MAUDE's sensitivity, not about lag).
- Detection performance: lead time vs false-alarm rate evaluated on keys split by product-code panel to avoid tuning on the same keys.
- Nulls: (i) permuting recall dates across keys within panel gives the null distribution of lag-stratum differences; (ii) simulated Poisson report streams without a change-point give the CUSUM false-alarm rate to compare with the empirical never-recalled keys.

## Publishable angle

Headline: "For most recalled devices there is a year or more between the first serious MAUDE report and the recall; the largest component is not late reporting but the time from a detectable signal to firm-initiated action, and it is longest for 510(k) Class II devices from large firms; a simple CUSUM on MAUDE counts would have flagged half of recalled devices six months earlier at a low false-alarm rate." The paper is timely with the GAO's 2026 review of the recall process.

Target venues: JAMA Internal Medicine (research letter or original investigation); Health Affairs; BMJ Surgery, Interventions, & Health Technologies; npj Digital Medicine; Therapeutic Innovation & Regulatory Science.

Follow-ups: a prospective evaluation of the CUSUM surveillance on new MAUDE releases; extension to EU vigilance once EUDAMED data are public; joint modelling with the narrative taxonomy of `ai-device-failure-taxonomy` (does the *content* of early reports predict lag?).

Related projects: `device-recall-prediction` (classifier of recall status), `ai-device-failure-taxonomy` (failure-mode content). This project is self-contained.

## Risks, confounds & mitigations

- **Key linkage errors** (subsidiaries, distributors, product-code changes). Mitigation: alias table, submission-number key as a second definition, sensitivity analyses, manual adjudication of 200 recalled keys.
- **Reporting intensity varies by device type and over time.** Mitigation: CUSUM baselines per key and per product code; year as covariate; ITS for policy effects.
- **Recalls unrelated to the reported events.** Mitigation: match root cause to report problem codes for a "concordant recall" sub-analysis.
- **MAUDE completeness before 2000 and the ASR era.** Mitigation: left truncation, sensitivity to the start date, and separate analysis of the 2019 ASR data release.
- **Recall class is determined after initiation.** Mitigation: not used as a Cox covariate; used for description and competing-risk sub-analyses by class.

## Milestones

- [ ] Download recall/enforcement/510k/pma/classification; MAUDE bulk to monthly counts per key.
- [ ] Key linkage with alias table; adjudicate 200 recalled keys; publish key table.
- [ ] Signal definitions (first serious, CUSUM); false-alarm calibration on never-recalled keys.
- [ ] Time-to-event table; KM/log-rank by stratum; Cox models (H1-H4).
- [ ] Lag decomposition; firm-level correlations (H3); ITS (H5); lead-time curves (H6).
- [ ] Pre-registration; manuscript; code and key-table release.

## Ethics / data-use notes

- All data are public FDA data; MDRs are unverified and do not establish causation (FDA disclaimer to be included). Firm-level results must be reported with denominators and uncertainty, and not as league tables.
- `OPENFDA_API_KEY` is optional and read from the environment only; nothing under `data/` is committed.
- Public data: the PhysioNet restriction on external LLM services does not apply; no narrative text is needed for this project's core analyses in any case.
