# faers-reporting-bias — denominator-aware, bias-adjusted FAERS signal detection with sex-stratified signals

**Pitch.** Quantify the structural reporting biases of FAERS (sex/age reporting disparities relative to real prescribing denominators, stimulated reporting after FDA Drug Safety Communications, and reporter-type effects) and turn those measurements into a bias-adjusted, sex-stratified disproportionality framework that separates "women report more" from "women are harmed more".

**Status:** starter code + protocol. **Difficulty:** MSc thesis to early-PhD; **timeline:** 6–9 months. **Compute:** a laptop; full FAERS (2004–2025, ~20M reports as quarterly JSON) needs ~60 GB disk and a few hours of pandas/pyarrow processing, no GPU.

## Background

FAERS is the largest public spontaneous-reporting system and the basis of thousands of disproportionality papers per year, yet nearly all of them treat report counts as if they were proportional to harm. Three structural biases are known but almost never modelled jointly:

1. **Missing denominators.** FAERS has no exposure data; the ROR/PRR compare a drug's reaction mix to the rest of the database, not to a population. Women file the majority of reports and experience more ADRs (Zucker & Prendergast, 2020, *Biol Sex Differ*; Watson et al., 2019, *EClinicalMedicine*), but they also receive more prescriptions, so a sex-specific ROR cannot say whether a difference is biology, prescribing or reporting behaviour.
2. **Stimulated reporting / notoriety bias.** Regulatory communications and media coverage change who reports what (Pariente et al., 2007, *Drug Saf*; Hoffman et al., 2014, *Drug Saf*). Empirical findings disagree: Hoffman et al. found only modest overall effects of FDA alerts, and Neha et al. (2021, *Hosp Pharm*) found no notoriety-driven change in signal estimates, whereas single-drug case studies repeatedly show sharp post-communication spikes. No study has estimated the distribution of stimulation effects across many communications with a common, denominator-aware design.
3. **Reporter type.** Consumer and lawyer reports differ systematically from physician reports. A 2026 analysis of cancer-therapy alopecia (*Pharmaceuticals* 19(3):445) showed drug rankings flip between all-reporter and HCP-only analyses, and a 2026 *J Clin Med* paper on paracetamol–autism reports found 98.6% of them consumer-submitted, driven by litigation and media.

Sex-stratified disproportionality has become common in 2024–2026 (e.g. statin-associated diabetes in *BMJ Open Diabetes Res Care* 2024; antiarrhythmic QT signals; growth-hormone and antibiotic analyses), but these papers report sex-specific RORs without denominators, interaction tests or bias adjustment, and their conclusions therefore conflate the three mechanisms above.

## The research gap

**What exists.**
- FAERS/VigiBase descriptive sex analyses (Watson et al., 2019; Brabete et al., 2022, *Pharmaceuticals*, scoping review) and one Dutch study with dispensing denominators (de Vries et al., 2019, *Br J Clin Pharmacol*).
- A US analysis adjusting sex differences in adverse drug events for baseline rates of drug use (2023, PMID 37603336) — the closest precedent — but at the aggregate level, not as a per-signal detection method.
- Stimulated-reporting studies of FDA alerts (Hoffman et al., 2014; Neha et al., 2021) using pre/post proportions, not interrupted time series with denominators or reporter strata; and, on the methods side, ITS tutorials (Bernal et al., 2017, *Int J Epidemiol*) and covariate-adjusted disproportionality via propensity scoring (Tatonetti et al., 2012, *Sci Transl Med*) that have not been combined with exposure offsets.
- Reporter-type sensitivity analyses restricted to one drug class (alopecia 2026; paracetamol–ASD 2026).

**What is missing (and what this project does).**
1. A **database-wide, denominator-aware** estimate of sex- and age-specific reporting rates per exposed person (FAERS numerators; MEPS person-level prescribing with sex/age, Medicare Part D for the ≥65 population), for hundreds of drugs, not one class.
2. An **ITS meta-analysis of stimulated reporting** across all FDA Drug Safety Communications since 2010, with negative-binomial segmented regression, all-FAERS offsets, reporter-type strata and a placebo-date permutation null — i.e. a distribution of stimulation ratios rather than one-drug anecdotes.
3. A **bias-adjusted disproportionality estimator**: logistic/Poisson disproportionality with reporter type, DSC exposure windows, sex and an exposure offset, plus a formal **drug × sex interaction test**, applied as a screen so that "sex-specific signals" are reported only when they survive adjustment.
4. An evaluation of how many published 2023–2026 "sex-specific FAERS signals" survive the adjusted framework.

## Research questions / hypotheses

1. **H1 (denominators).** After dividing by MEPS-weighted users, the female:male reporting-rate ratio (RRR) is < the crude FAERS female:male report ratio for most drugs; for ≥ 30% of drugs with a crude ratio > 1.5 the denominator-adjusted RRR is not significantly different from 1.
2. **H2 (age).** Reports per 10 000 Part D beneficiaries decline with age band within drug (under-reporting in the ≥ 75 group), and this gradient is steeper for consumer than for physician reports.
3. **H3 (stimulated reporting).** The median level-change rate ratio in the 6 months after a DSC is > 1.3 for the named drug–event pair but ≈ 1 for the drug's other events; the effect is larger for consumer reports and for communications with boxed-warning language; slope changes decay within 12–18 months.
4. **H4 (reporter type).** Consumer- and lawyer-dominated drug–event pairs have higher crude ROR but lower IC025 stability across quarters; restricting to HCP reports removes ≥ 25% of ROR-based signals with a ≥ 3 reports.
5. **H5 (sex-stratified signals).** With the interaction test and exposure adjustment, the number of drug–event pairs with a significant sex difference (BH-FDR 5%) is an order of magnitude smaller than the number implied by "ROR_f signal & ROR_m null" logic used in recent papers; surviving pairs are enriched for known PK sex differences (Zucker & Prendergast list).

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| FAERS via openFDA `drug/event` (API + quarterly bulk JSON) | numerators: reports by drug, event PT, sex, age, reporter, date | ~20M ICSRs 2004–2025; ~60 GB zipped JSON | open (free API key optional) | https://open.fda.gov/apis/drug/event/ |
| FDA FAERS quarterly extracts (ASCII/XML) | cross-check of report source (`RPSR`) and de-duplication | same | open | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| MEPS-HC Prescribed Medicines + Full-Year Consolidated files (2018–2023) | sex × age exposure denominators (persons with ≥ 1 fill, survey-weighted) | ~300k fills/yr, ~30k persons/yr | open | https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp |
| Medicare Part D Spending by Drug; Prescribers by Provider and Drug | ≥ 65 claim/beneficiary denominators, secular prescribing trends | ~3k generics × 5 yr; ~25M prescriber-drug rows | open | https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug/medicare-part-d-spending-by-drug |
| FDA Drug Safety Communications index (+ SrLC database for boxed warnings) | event dates for ITS | ~30–60 DSCs/yr since 2010 | open | https://www.fda.gov/drugs/drug-safety-and-availability/drug-safety-communications |
| Google Trends / GDELT (optional) | media-attention covariate | small | open (unofficial API) | https://trends.google.com |

Details and directory layout: `data/README.md`. Sample pull: `python scripts/download_data.py --sample`.

## Methods

1. **Ingest & de-duplicate.** Flatten bulk FAERS JSON (`faers_bias.openfda_client.flatten_faers_record`), keep the latest version per `safetyreportid`, restrict to US reports (`occurcountry == US`) when merging with US denominators, map `primarysource.qualification` to reporter type, normalise generic names (`denominators.normalize_drug_name`) to the Multum/Part D naming.
2. **Denominators.** MEPS: weighted persons with ≥ 1 fill per (drug, sex, age band, year); pool 2018–2023 for sparse drugs; design-based CIs via `VARSTR/VARPSU` (optional `samplics`). Part D: `Tot_Benes` per generic-year. Merge to (drug, year, sex[, age band]) and compute reports per 10 000 exposed with exact Poisson CIs (`denominators.merge_denominators`) and the female:male RRR (`reporting_rate_ratio`).
3. **Stimulated reporting.** For each DSC row (`data/dsc_events_seed.csv`, extended to the full index), build monthly series of (a) the named drug–event pair, (b) the drug's other events, (c) all FAERS, by reporter type; fit negative-binomial segmented regression with annual Fourier terms, HAC covariance and the all-FAERS offset (`its.fit_its`); calibrate with 200 placebo dates (`its.placebo_its`); pool level-change log-RRs with a random-effects meta-analysis (statsmodels `combine_effects`) and meta-regress on communication type (boxed warning vs. information), reporter type, and Google-Trends peak.
4. **Reporter-type effects.** For every drug–event pair with a ≥ 3: ROR/IC per reporter stratum (`disproportionality.reporter_stratified_signal`); flag "reporter-discordant" pairs; quarter-by-quarter IC stability.
5. **Bias-adjusted, sex-stratified disproportionality.** Report-level logistic model `event ~ drug + reporter + post_dsc + drug:post_dsc + sex + drug:sex` (`bias_adjusted_ror`) and, where denominators exist, an aggregated Poisson rate model with `log(exposed persons)` offset. Sex difference tested by Wald and LRT (`sex_interaction_test`); multiplicity controlled by BH-FDR over all pairs.
6. **Re-analysis of published claims.** Collect 2023–2026 FAERS papers claiming sex-specific signals (PubMed search "FAERS AND sex AND disproportionality"); re-estimate each claimed pair under the adjusted framework; report the survival rate.

Libraries: pandas/pyarrow, statsmodels (GLM NB/Poisson, HAC, meta-analysis), scipy, scikit-learn (propensity-score covariate balance as a sensitivity analysis), matplotlib.

## Evaluation & statistics

- **Primary estimands:** RRR (female:male reports per exposed person), DSC level-change RR and stimulation ratio (observed / counterfactual in 6 months), adjusted ROR and ratio-of-ROR (female/male).
- **Validation:** hold-out-in-time — fit stimulation models on DSCs 2010–2018, predict post-DSC excess for 2019–2024 communications (calibration slope). For denominators, compare MEPS-based and Part D-based RRRs in the ≥ 65 stratum (agreement = Lin's concordance).
- **Nulls:** placebo-date permutation for ITS; label-permutation of sex within drug for interaction tests; negative-control drug–event pairs (Ryan et al.-style, e.g. from the OMOP reference set) to check the adjusted ROR's false-positive rate.
- **Multiple comparisons:** BH-FDR 5% across all drug–event pairs; report q-values and the number of pairs tested.
- **Leakage prevention:** DSC effect estimates use only pre-DSC data for the counterfactual; the re-analysis of published claims uses the same data cut as the original paper when stated.
- **Sensitivity:** suspect-only vs. suspect+concomitant drugs; US-only vs. all countries; exclusion of duplicates and of literature reports; alternative denominators (claims vs. persons vs. dosage units).

## Publishable angle

- **Headline result:** "Across N drugs and M FDA communications, X% of FAERS sex differences disappear after exposure adjustment, FDA communications inflate reporting of the named event by a median RR of Y for Z months (almost entirely via consumer reports), and only K of J recently published sex-specific FAERS signals survive a bias-adjusted interaction test."
- **Venues:** *Drug Safety*; *Pharmacoepidemiology and Drug Safety*; *Clinical Pharmacology & Therapeutics*; *JAMIA* (methods framing); *Biology of Sex Differences* (sex-difference framing).
- **Follow-ups:** apply the same offsets to VigiBase/EudraVigilance (age/sex denominators from IQVIA or national dispensing), extend to age-stratified paediatric signals, and release the adjusted signal table as a resource alongside the FDA "potential signals" list.

## Risks, confounds & mitigations

- **Denominator mismatch** (MEPS = civilian non-institutionalised; FAERS includes hospital and foreign reports) → restrict to US reports; sensitivity with Part D for ≥ 65; treat RRR as a relative, not absolute, rate.
- **Drug-name mapping** (salts, combinations, brand-only reports) → `normalize_drug_name`, RxNorm ingredient mapping via openFDA `rxcui`; audit 200 random mismatches.
- **Sparse cells** (many drugs have < 100 MEPS users) → pool years, require unweighted n ≥ 30, use Bayesian shrinkage (IC) for sparse pairs.
- **Concurrent events** near DSC dates (generic entry, lawsuits, drug withdrawal) → code confounding events from the SrLC database and Google Trends; exclude DSCs within 6 months of another event for the same drug in a sensitivity analysis.
- **Autocorrelation / over-dispersion** → NB family, HAC SEs, placebo permutation.
- **Reporter misclassification** (missing `qualification` in ~15% of reports) → separate "unknown" stratum; multiple imputation as sensitivity.
- **Garbage-in on published claims** → pre-register the paper list and re-analysis protocol (OSF) before running it.

## Milestones

- [ ] Ingest bulk FAERS 2010–2024, de-duplicate, flatten, write parquet (`data/processed/faers.parquet`).
- [ ] Build MEPS 2018–2023 and Part D denominators; drug-name mapping audit.
- [ ] Reports-per-exposed tables by sex/age; H1–H2 figures.
- [ ] Complete DSC event list (2010–2024) with confounder coding; ITS + placebo runs; meta-analysis (H3).
- [ ] Reporter-stratified signal tables; stability analysis (H4).
- [ ] Bias-adjusted sex-stratified screen with FDR; re-analysis of published claims (H5).
- [ ] Negative-control calibration; sensitivity analyses; manuscript + released signal table.

## Ethics / data-use notes

- FAERS, MEPS and Part D public use files are de-identified and openly licensed; MEPS requires citing AHRQ and forbids attempts at re-identification. Never commit raw data (see `.gitignore`).
- FAERS narratives are not used; if narrative fields are ever processed, do not send them to third-party LLM APIs without institutional review.
- Report findings as reporting-rate differences, not incidence; FDA explicitly cautions against incidence inference from FAERS.
