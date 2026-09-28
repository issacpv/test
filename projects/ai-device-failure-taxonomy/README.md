# ai-device-failure-taxonomy

**What breaks when AI-enabled medical devices fail: a reproducible linkage of the FDA AI-enabled device list to MAUDE adverse-event reports, an AI-specific failure-mode taxonomy applied to report narratives with local NLP and human validation, and denominator-adjusted reporting rates against non-AI comparator devices in the same product codes.**

## Status / difficulty / timeline / compute

- Status: design + starter code (openFDA client with cursor pagination and rate limiting, FDA-list-to-MAUDE linkage with tiered matching and a linkage report, a rule-based taxonomy labeller with evidence spans and a TF-IDF classifier to generalise it, device-year rate ratios, negative-binomial rate models, pre/post software-update counts). No data is shipped; all sources are open.
- Difficulty: MSc-level; the difficult parts are entity linkage (brand names, product codes, distributors vs manufacturers) and annotation design, not the modelling.
- Timeline: 4-7 months (1 month data + linkage, 1 month taxonomy + double annotation of 500 narratives, 1-2 months rates/comparators/update analysis, 1 month writing).
- Compute: laptop. The full MAUDE bulk files are ~20 GB of JSON; DuckDB or streaming JSON handles them. Local transformer models (optional) run on CPU for a few hundred thousand narratives.

## Background

The FDA maintains a public list of AI-enabled medical devices authorised through 510(k), De Novo and PMA; it passed 950 devices in 2024 and more than 1,300 by 2026. Wu et al. (2021, Nat Med) showed how thin the pre-market evidence is (few prospective or multi-site evaluations). Post-market evidence is thinner still. MAUDE (Manufacturer and User Facility Device Experience) holds millions of medical device reports (MDRs) with free-text narratives and FDA-coded product/patient problems; openFDA exposes it through the `device/event` endpoint together with `device/510k`, `device/pma`, `device/classification`, `device/recall` and `device/udi`. Lyell et al. (2023, J Am Med Inform Assoc; "More than algorithms") manually analysed MAUDE events for machine-learning-enabled devices and found that many safety events were not algorithm failures but problems of data input, integration and use. In 2025, an analysis of the 950 devices authorised through August 2024 reported 489 MDRs across 36 devices (458 malfunctions, 30 injuries, 1 death) and 113 recalls of 40 devices, 75% software-related, and Lee et al. (2025, JAMA Health Forum) showed that devices without clinical validation and devices from publicly traded companies were recalled more often. A 2025 review of 27 years of AI/ML device recalls and a 2026 npj Digital Medicine paper on scaling regulatory science with large language models make the same point: the narratives are where the failure modes are, and nobody has classified them at scale with an AI-specific scheme.

## The research gap

What has been done (2023-2026):

- Manual, small-scale event classification for ML devices (Lyell et al., 2023): informative but pre-2022 and not reproducible at scale.
- Counting studies on the 950-device list (2025): number of MDRs and recalls per device, recall causes from the FDA recall database; no narrative taxonomy, no denominators (device-years on the market), no comparison with non-AI devices in the same product codes, and linkage methods that are not released.
- Radiology-specific tallies (2025): 151 MDRs for 43 radiology AI products, 40 attributed to software defects; 45% of radiology AI devices had software version updates. Again descriptive, single specialty.
- A JMIR Research Protocols systematic-review protocol (2024) on AI-as-a-medical-device adverse-event reporting, which highlights the lack of standardised classification.
- The FDA's own problem codes cannot express AI-specific failure modes such as distribution shift or silent degradation (noted in the governance literature, npj Digit Med 2025-2026).

What is specifically missing (our angle):

1. **A released, versioned linkage** between the FDA AI list and MAUDE via submission number -> `device/510k` / `device/pma` -> product code + applicant -> `device/event` (product code + brand/manufacturer fuzzy matching, UDI where present), with precision/recall estimated on a manually adjudicated sample. Every later study can reuse it.
2. **An AI-specific failure-mode taxonomy** (incorrect output; no/delayed output; input data quality; integration/interface; version/update-related; hardware of the AI device; user/workflow; cybersecurity) applied to narratives with rule-based weak labels, a locally trained classifier and double human annotation, and compared with FDA's structured `product_problems` codes (agreement statistics).
3. **Denominator-adjusted rates**: events per device-year since authorisation, compared with non-AI devices in the same product codes and authorisation years (negative-binomial models with panel, pathway, class, firm size).
4. **Temporal analysis around software updates and recalls**: a self-controlled comparison of reporting in the 180 days before vs after successive clearances of the same device, and the link to recall (the `recall-lag-survival` project models timing; this project models content).
5. **Reporter and outcome structure**: manufacturer vs user-facility vs voluntary reports and the malfunction/injury/death mix for AI vs non-AI devices.

## Research questions / hypotheses

1. **H1 (algorithm share).** In linked AI-device MDRs, narratives implicating the algorithm's output (incorrect / missed / no output) are fewer than 40% of reports; input-quality, integration and hardware categories account for the majority (replicating Lyell et al. at scale).
2. **H2 (AI vs comparator).** Per device-year, AI devices have higher rates of incorrect-output and integration/interface reports than non-AI devices in the same product code and authorisation-year stratum (incidence rate ratio > 1.5 with 95% CI excluding 1), and similar hardware rates.
3. **H3 (updates).** Reporting rate is higher in the 180 days after a new clearance of the same device than in the 180 days before (rate ratio > 1.2), consistent with update-related regressions.
4. **H4 (codes vs narratives).** Agreement between the narrative taxonomy and FDA `product_problems` codes is weak (Cohen's kappa < 0.4 for the "incorrect output" category), i.e. structured codes under-capture AI failure modes.
5. **H5 (validation and recalls).** Devices listed as having clinical validation at authorisation (using the annotation of Lee et al., 2025) have lower incorrect-output rates; devices with >= 1 incorrect-output report have a higher subsequent recall probability.
6. **H6 (reporters).** The share of user-facility and voluntary reports is higher for AI devices than for comparators (manufacturers see fewer of the failures that happen in the workflow).

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| FDA Artificial Intelligence-Enabled Medical Devices list | Device, company, submission number, primary product code, decision date, panel | > 1,300 rows (2026) | Open (Excel/CSV download on the FDA page; URL changes, see data/README.md) | https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices |
| openFDA `device/event` (MAUDE) | MDRs: narratives (`mdr_text`), event type, dates, device brand/generic name, product code, manufacturer, product/patient problem codes, reporter | millions of reports; bulk zip partitions | Open (optional `OPENFDA_API_KEY`) | https://open.fda.gov/apis/device/event/ |
| openFDA `device/510k`, `device/pma`, `device/classification` | Submission -> product code, applicant, decision date, device class, panel; comparator device universe | ~170k 510(k)s | Open | https://open.fda.gov/apis/device/ |
| openFDA `device/recall`, `device/enforcement` | Recall events for H5 | ~100k | Open | https://open.fda.gov/apis/device/recall/ |
| openFDA `device/udi` (GUDID) | Brand names and version/model identifiers for linkage | ~4M | Open | https://open.fda.gov/apis/device/udi/ |
| Lee et al., 2025 supplementary data | Clinical-validation annotation per AI device | 950 devices | Open (journal supplement) | JAMA Health Forum 2025 |

## Methods

1. **Ingest** (`openfda_client.py`, `scripts/download_data.py`). Bulk MAUDE partitions from the openFDA download manifest (full mode) or paged API queries by product code (sample mode); `510k`/`pma`/`classification`/`recall` via paged queries. Records are flattened to a tabular schema with the narrative types kept separate (`Description of Event or Problem`, `Additional Manufacturer Narrative`).
2. **Linkage** (`linkage.py`). Normalise names (case, punctuation, legal suffixes); tier 1 = product code match + brand-name similarity >= 0.85 to the listed device name; tier 2 = product code + manufacturer similarity >= 0.80; tier 3 = product code only (used solely for comparator construction). Adjudicate 200 tier-1/2 matches and 100 tier-3 non-matches manually; report precision/recall; publish the mapping with tiers.
3. **Taxonomy** (`taxonomy.py`). Eight failure-mode categories with regex rules and evidence spans; an outcome layer (patient harm, delay in care) kept separate. Weak labels -> TF-IDF + logistic one-vs-rest classifier (or a local sentence-embedding model) evaluated against 500 double-annotated narratives (kappa per category). The rules and the annotation guideline are versioned.
4. **Rates** (`rates.py`). Device-years from decision date to data cut; comparators = non-AI devices in the same product code cleared within +/- 2 years; negative-binomial regression with offset log(device-years) and covariates (panel, pathway, class, firm size by number of listed devices); rate ratios per taxonomy category. Pre/post update counts around successive clearances of the same device.
5. **Agreement with FDA codes.** Map `product_problems` (FDA problem codes) onto the taxonomy where a mapping is defensible and compute kappa with narrative labels.

Tools: requests, pandas, DuckDB, rapidfuzz (optional; difflib fallback), scikit-learn, statsmodels; optional local transformer models via `sentence-transformers`.

## Evaluation & statistics

- Linkage precision/recall with Wilson CIs on adjudicated samples; sensitivity analysis on similarity thresholds (0.80-0.95).
- Taxonomy: per-category kappa (two annotators), classifier F1 on the held-out annotated set; report rule-only vs classifier.
- Rates: negative-binomial IRRs with 95% CIs; overdispersion reported; cluster-robust SEs by manufacturer; Benjamini-Hochberg across 8 categories; H1-H3 pre-registered.
- Reporting-bias controls: (i) event counts normalised by *all* reports of the product code and year (reporting intensity), (ii) analysis restricted to manufacturer reports (mandatory) vs all, (iii) exclusion of the 90 days after a recall announcement (publicity-driven reporting).
- Nulls: (i) shuffling AI-status among devices within product code x year strata gives the null IRR distribution; (ii) taxonomy applied to a random sample of non-AI reports to check that rule hits are not trivially higher for AI narratives due to vocabulary (e.g. "algorithm" appearing in device names).
- Leakage/contamination: brand-name tokens are removed from narratives before classification so that the classifier cannot learn device identity.

## Publishable angle

Headline: "Most reported failures of AI-enabled devices are not algorithm errors; but per device-year, AI devices generate several times more incorrect-output and integration reports than non-AI devices in the same product codes, reporting spikes after software updates, and FDA problem codes capture little of this." Deliverables: the linkage table, the taxonomy with annotation guideline, and an open dashboard-ready dataset.

Target venues: npj Digital Medicine; JAMIA; The Lancet Digital Health; JAMA Network Open; BMJ Health & Care Informatics; AMIA Annual Symposium.

Follow-ups: prospective monitoring of new clearances; extension to EU vigilance data (EUDAMED, once public); integration with the `recall-lag-survival` timing model; comparison with FAERS-style disproportionality methods on MAUDE.

Related projects: `device-recall-prediction` (recall classification from MAUDE features), `recall-lag-survival` (time from MAUDE signal to recall). This project is self-contained.

## Risks, confounds & mitigations

- **Linkage errors** (distributors, re-branded devices, multiple product codes per device). Mitigation: tiers, manual adjudication, publish uncertainty; UDI/GUDID brand names as a second key.
- **Reporting bias** (manufacturer awareness, publicity). Mitigation: denominators, reporter stratification, recall-window exclusion, normalisation by product-code reporting intensity.
- **Narrative quality** (templated manufacturer text, redaction). Mitigation: de-duplicate templated narratives, analyse report types separately.
- **Comparator validity** (non-AI devices in the same code may be very different). Mitigation: restrict to codes with both AI and non-AI devices and similar authorisation years; sensitivity analysis on the closest predicate devices from `510k` records.
- **Device-year denominators ignore installed base.** Mitigation: state clearly; use within-device pre/post designs (H3) as a denominator-free complement.

## Milestones

- [ ] Download FDA list, openFDA tables (sample then bulk); build device universe with product codes and dates.
- [ ] Linkage with tiers; adjudicate 300 pairs; release v1 mapping.
- [ ] Taxonomy v1; annotation guideline; 500 narratives double-annotated; classifier; kappa (H4).
- [ ] Category shares (H1); comparator rates (H2, H6); update analysis (H3); validation/recall links (H5).
- [ ] Pre-registration; manuscript; data and code release.

## Ethics / data-use notes

- All sources are public and de-identified; openFDA data carry FDA's disclaimer that MDRs are not verified and do not establish causation, which must be stated in every output. Manufacturer names are public; avoid ranking firms without denominators and uncertainty.
- The optional `OPENFDA_API_KEY` is read from the environment and never written to files. `data/` is git-ignored; bulk downloads are large and must not be committed.
- openFDA data are public, so the PhysioNet-style restriction on external LLM APIs does not apply; for reproducibility the classifier and any LLM-based annotation should still use local, versioned models, and any external service use must be documented.
