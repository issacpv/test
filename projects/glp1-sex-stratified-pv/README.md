# glp1-sex-stratified-pv: separating exposure, reporting propensity and biology in the female predominance of GLP-1 receptor agonist adverse-event reports

**One-sentence pitch.** Every FAERS study of GLP-1 receptor agonists (GLP-1 RAs) notes that 65–75 % of reports come from women, but none has separated *who takes the drug* (exposure), *who reports* (propensity) and *who is harmed* (biology); this project builds a sex-difference atlas of GLP-1 RA adverse events with sex-specific backgrounds, indication adjustment, survey-derived exposure denominators and label expectedness, timed to the 2023–2026 semaglutide/tirzepatide boom and the compounded-product era.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc-level pharmacoepidemiology/biostatistics; 6–8 months to a first paper; 10–12 months with the denominator and label components.
- Compute: CPU only. GLP-1 RA reports are on the order of a few hundred thousand; a 10 % background sample of FAERS is enough for sex-specific backgrounds. Everything runs on a laptop.

## Background

Semaglutide (Ozempic 2017, Rybelsus 2019, Wegovy 2021) and tirzepatide (Mounjaro 2022, Zepbound 2023) have become the most-discussed drugs in the US, with a reporting surge in FAERS from 2022, high-profile safety questions (gastroparesis/ileus label update for Ozempic in 2023; Sodhi et al., 2023, *JAMA* on gastrointestinal events; the 2023–2024 FDA and EMA evaluations of suicidal ideation, alopecia and pulmonary aspiration; Wang et al., 2024, *Nat Med* finding no elevated suicidal-ideation risk in EHR data), a widespread compounded-product market during the 2023–2025 shortages, and known sex differences in pharmacodynamics and tolerability of GLP-1 RAs in trials (e.g., Rentzeperi et al., 2022, *J Pers Med*). Women are over-represented in obesity-indication prescriptions and in adverse-event reporting generally (Watson et al., 2019, *EClinicalMedicine*), so a naive female share of reports is uninterpretable.

## The research gap

**What has been done (2024–2026 FAERS literature, all confirming the descriptive female predominance).**

- Neurological adverse events of GLP-1 RAs, FAERS 2005 Q2–2024 Q3: 65.3 % female vs 30.5 % male reports; sex-specific reporting patterns discussed descriptively (*Scientific Reports*, 2025, s41598-025-01206-9).
- Tirzepatide FAERS 2022 Q1–2025 Q1: 75.8 % female; sex-stratified RORs suggested sleep disorder, delayed gastric emptying and medullary thyroid cancer more often reported in men, starvation ketoacidosis and injection-site errors in women (*Drug, Healthcare and Patient Safety*, 2025).
- Psychiatric signals of GLP-1 RAs (FAERS 2021 Q2–2025 Q3; *Pharmaceuticals*, 2026, doi:10.3390/ph19060953) with 70.3 % women in the GLP-1 group vs 53.7 % in the comparator; neuropsychiatric events (*European Psychiatry*); male sexual dysfunction (*International Journal of Impotence Research*, 2025); GLP-1 vs insulin comparisons showing 65.7 % vs 57.4 % female share and a rising female share after 2021.
- FDA's own communications on compounded semaglutide/tirzepatide dosing errors (2024) and pharmacovigilance summaries of compounded-product reports.

**What is specifically missing.**

1. **No exposure denominators.** No study converts sex-specific counts into reporting rates per user. MEPS and NHANES provide survey-weighted, sex-specific numbers of GLP-1 RA users in the US population, which turns the female share into an "excess reporting propensity" statistic.
2. **No indication adjustment.** Women dominate the weight-management brands (Wegovy, Zepbound, Saxenda), men are relatively more represented in T2D brands; sex-stratified RORs that ignore indication confound biology with product. The drug × sex interaction adjusted for indication (and age band) has not been reported.
3. **Pooled backgrounds.** Published sex-stratified RORs are usually computed against the whole database, whereas women's background reporting profile differs (more alopecia, more psychiatric, fewer cardiovascular PTs). Sex-specific backgrounds are needed.
4. **No PT-wide, multiplicity-controlled atlas.** Sex differences are reported for hand-picked PTs; a full-vocabulary screen with BH control and formal interaction tests does not exist for this class.
5. **Label expectedness by sex.** GLP-1 RA labels list adverse reactions pooled across sexes; whether the sex-enriched unlabelled events (e.g., alopecia, menstrual/contraceptive-failure events relevant to the tirzepatide oral-contraceptive interaction warning) cluster in women is untested.
6. **Compounded products.** Whether the sex distribution and event profile of compounded-product reports (dosing errors, injection-site events) differ from branded reports has not been analysed with a comparator design.

## Research questions / hypotheses

1. **H1 (exposure vs propensity).** The observed female share of semaglutide/tirzepatide reports exceeds the exposure-expected share from MEPS/NHANES by a propensity ratio of 1.1–1.3; after age standardisation the female/male reporting-rate ratio is < 1.5 (i.e., most of the raw 2–3× female excess is exposure).
2. **H2 (atlas).** In a PT-wide screen with sex-specific backgrounds, ≥ 5 % of GLP-1 RA signals show a drug × sex interaction (BH q ≤ 0.05); female-enriched signals concentrate in alopecia, nausea/vomiting, cholelithiasis and psychiatric PTs, male-enriched in pancreatitis, acute kidney injury and sexual-dysfunction PTs.
3. **H3 (indication confounding).** At least one third of crude sex differences (H2) attenuate to non-significance after adjustment for indication (obesity vs T2D) and age band; the residual set is the candidate "biological" sex-difference list.
4. **H4 (active comparators).** Sex-difference ratios against active comparators (SGLT2 inhibitors, DPP-4 inhibitors, insulin for T2D; phentermine-containing products for obesity) are smaller than against all-of-FAERS, and gastrointestinal sex differences persist against every comparator.
5. **H5 (labels).** Unlabelled PTs are more frequent in women's reports than men's for the obesity brands (difference ≥ 5 percentage points), driven by alopecia and reproductive-system PTs; for T2D brands the sex difference in unlabelled fraction is < 2 points.
6. **H6 (compounded products).** Compounded-product reports have a higher female share, a higher fraction of medication-error and injection-site PTs, and are more often consumer-reported than branded-product reports in the same quarters; the atlas restricted to branded products does not change its top-20 sex-different PTs.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| openFDA drug/event (FAERS) | GLP-1 RA reports 2018–2026 (brand/generic/verbatim names, `drugindication`, sex, age, reactions, reporter, seriousness); comparator-class reports; 10 % background sample | ~20 M reports total; ~0.5 M exposed | open (free API key optional) | https://open.fda.gov/apis/drug/event/ |
| FAERS quarterly ASCII | `CASEID` dedup; `INDI` indication table; `DRUG.DOSE_VBM`/`ROUTE` for compounded-product detection | ~1.5 GB/yr | open | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| MEPS HC Prescribed Medicines + Full-Year Consolidated | survey-weighted GLP-1 RA users by sex, age; 2018–2023 panels | ~300 k fills/yr | open | https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp |
| NHANES RXQ_RX + DEMO (2017–2020, 2021–2023) | cross-check of users by sex | ~10 k participants/cycle | open | https://wwwn.cdc.gov/nchs/nhanes/ |
| openFDA drug/label (SPL) | adverse reactions, warnings, specific-populations, drug-interactions sections per brand | 1 label/brand | open | https://open.fda.gov/apis/drug/label/ |
| Medicaid State Drug Utilization Data | quarterly prescriptions per NDC (no sex) for time trends | open | open | https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data |
| EudraVigilance (DAP) / VigiBase | replication in a second system (optional) | — | web access / VigiBase requires application | https://www.adrreports.eu/ |

## Methods

1. **Cohort (`glp1_sexpv.cohort`).** Match agents and brands from `openfda.generic_name`/`brand_name`/`medicinalproduct`; classify indication from `drugindication` text with brand fallback; flag compounded products (string markers or unmapped bare INN); tag active-comparator classes. Deduplicate via `CASEID` (ASCII) or hash.
2. **Sex-specific disproportionality (`glp1_sexpv.sex_signals`).** For each PT with ≥ 3 exposed reports in each sex: female and male 2×2 tables with same-sex backgrounds; ROR_f, ROR_m, ratio with Wald CI and LRT for drug × sex; BH across PTs (H2). Backgrounds: all-of-FAERS 10 % sample; each active-comparator class (H4).
3. **Indication/age adjustment.** Aggregated weighted logistic model `event ~ drug*sex + C(indication) + drug:C(indication) + sex:C(indication) [+ age band terms]`; report crude vs adjusted ratios; Mantel–Haenszel RORs per sex across indication strata (H3).
4. **Denominators (`glp1_sexpv.denominators`).** MEPS fills flagged by name regex → persons → survey-weighted users by sex and age band; reporting rate per 10 000 users with exact Poisson CIs; female/male rate ratio; age-standardised rates; excess-female-reporting propensity ratio with binomial test (H1).
5. **Labels.** Fetch SPL sections per brand; classify each report's PTs as labelled/unlabelled (synonym table, British/American spellings); compare unlabelled fractions by sex × brand (H5). (Approach shared with `unlabeled-adverse-event-mining`, re-implemented here.)
6. **Compounded products.** Same atlas and reporter-type comparisons for compounded vs branded reports within calendar quarter (H6).
7. **Time.** Quarterly sex shares and top-PT trajectories 2018–2026 with change-points at Wegovy (2021-06), the 2023 media wave, the Ozempic ileus label update (2023-09) and the FDA compounded-product communications (2024).

## Evaluation & statistics

- Primary endpoints: propensity ratio with binomial CI (H1); number and identity of PTs with BH q ≤ 0.05 for drug × sex (H2), reported with ratio and 95 % CI.
- Sensitivity: suspect-only role; exclude consumer reports; exclude 2023 (media peak); alternative min_a (3/5/10); sex-unknown reports handled by complete-case and by multiple imputation from age/indication/brand.
- Nulls: permute sex within (exposure, indication, age-band) strata to obtain the null distribution of the count of sex-different PTs; negative-control PTs (e.g., "off label use", "product storage error") expected to show no sex difference beyond propensity.
- Multiplicity: BH within the atlas; Holm across the six hypotheses' primary tests.
- Denominator uncertainty: MEPS replicate-weight (or Taylor-linearised) variance for user counts propagated into rate CIs by parametric bootstrap.
- Reporting per READUS-PV (Fusaroli et al., 2024, *Drug Saf*).

## Publishable angle

- **Headline result.** "Of the 2.3-fold female excess in GLP-1 RA reports, X is explained by exposure, Y by reporting propensity, and only Z PTs show a residual sex-specific disproportionality after indication adjustment — among them alopecia and cholelithiasis in women and pancreatitis in men." A quantitative decomposition that current descriptive papers cannot provide.
- **Venues.** *Drug Safety*; *Diabetes, Obesity and Metabolism*; *Pharmacoepidemiology and Drug Safety*; *Biology of Sex Differences*; *JAMA Network Open* (denominator/propensity angle).
- **Follow-ups.** Oral-contraceptive interaction and unintended-pregnancy reports with tirzepatide (interaction ROR, see `faers-ddi-signals`); replication in EudraVigilance; EHR validation of the residual sex-specific signals (see `faers-signal-ehr-validation`).

## Risks, confounds & mitigations

- **Indication misclassification** (missing `drugindication` for ~30 % of entries). Mitigation: brand fallback; sensitivity restricted to reports with explicit indication; multiple imputation from brand/age/sex.
- **Notoriety and media-stimulated reporting** (2023 surge; celebrity coverage skews to women). Mitigation: quarter-stratified analyses; exclude peak quarters; reporter-type stratification.
- **Denominator mismatch** (MEPS civilian non-institutionalised population; compounded products invisible in MEPS/NHANES). Mitigation: restrict rate analyses to branded products; triangulate with Medicaid SDUD trends; present rates as sensitivity, atlas as primary.
- **Sex-unknown reports** (10–20 %). Mitigation: complete-case + imputation; report both.
- **Competition/masking** by the dominant GI events. Mitigation: masking analysis removing GI PTs from backgrounds.
- **Reverse causation** for weight-related PTs (e.g., "weight decreased" is the intended effect). Mitigation: pre-specified exclusion list of on-target PTs.

## Milestones

- [ ] Bulk FAERS pull; cohort, comparator and 10 % background extraction; dedup
- [ ] Indication classification validated on 500 hand-checked reports (kappa)
- [ ] MEPS/NHANES users by sex × age; rates and propensity ratio (H1)
- [ ] Sex-difference atlas with sex-specific backgrounds; BH; permutation null (H2)
- [ ] Indication/age-adjusted interactions; MH pooling; active comparators (H3, H4)
- [ ] Label expectedness by sex (H5); compounded vs branded (H6)
- [ ] Time-trend/change-point figures; sensitivity analyses
- [ ] Manuscript (READUS-PV), released atlas table and code

## Ethics / data-use notes

- FAERS, SPL, MEPS and NHANES are public, de-identified data; no re-identification attempts; typical secondary-use exemption (confirm locally).
- Findings are hypothesis-generating; sex-specific signals must not be presented as incidence differences without the denominator caveats.
- Never commit downloaded data; `data/` is git-ignored; API key from `OPENFDA_API_KEY`.

## Related projects in this repository

`faers-reporting-bias` (reporter-type and stimulated-reporting adjustment; sex-stratified ROR utilities), `unlabeled-adverse-event-mining` (label expectedness), `faers-ddi-signals` (interaction terms), `faers-signal-ehr-validation`. This project is self-contained.
