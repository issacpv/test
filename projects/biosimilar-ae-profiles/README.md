# biosimilar-ae-profiles: launch-aligned, label-controlled comparison of biosimilar and originator adverse-event profiles in FAERS

**One-sentence pitch.** Use the 2023–2025 US biosimilar "waves" (ten adalimumab products in 2023, ustekinumab and denosumab in 2025, aflibercept and tocilizumab in 2024) as natural experiments to test whether biosimilars differ from their originators in reported adverse-event profiles once product attribution, time-since-launch (Weber effect), reporter mix and label expectedness are controlled — and to quantify how much of FAERS biologic reporting is even attributable to a product.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc-level pharmacoepidemiology / data-science project; 6–8 months to a first paper, 9–12 months including the traceability and nocebo sub-studies.
- Compute: CPU only. Biologic-family reports are ~2–3 % of FAERS (a few hundred thousand reports); everything fits in memory on a laptop after the bulk filter step (~10 GB zipped JSON to scan once).

## Background

Biosimilars are approved on the basis of analytical similarity and limited comparative clinical data; post-marketing safety surveillance is therefore explicitly part of their regulatory bargain. Two features make FAERS-based comparison unusually tricky. First, **traceability**: a report may name a brand, a suffixed nonproprietary name (e.g. adalimumab-atto), the originator brand, or only the shared INN, and INN-only reports cannot be attributed (Vermeer et al., 2013, *Drug Saf*, on FAERS and EudraVigilance; Vermeer et al., 2019, *Clin Pharmacol Ther*). FDA's four-letter suffix convention (guidance "Nonproprietary Naming of Biological Products", 2017) was introduced to fix this, and the 2023 adalimumab wave is the first large-scale test of whether it works in spontaneous reports. Second, **time-since-launch**: reporting peaks in the first two years after a launch (Weber, 1984; Hoffman et al., 2014, *Drug Saf*, on FAERS), so a young biosimilar compared with a 20-year-old originator is confounded by design.

Switching studies (Cohen et al., 2018, *Drugs*; Barbier et al., 2020, *Clin Pharmacol Ther*) find no efficacy or immunogenicity penalty, but nocebo-type discontinuations after non-medical switching are well documented in open-label settings — exactly the kind of "drug ineffective / condition aggravated" reports that spontaneous systems collect.

## The research gap

**What has been done.**

- A disproportionality analysis of FAERS June 2018–December 2022 for rituximab, bevacizumab and trastuzumab and their marketed biosimilars found similar disproportionate-reporting signals between originators and biosimilars, except for death with biosimilar bevacizumab (*Expert Opinion on Drug Safety*, 2024, doi:10.1080/14740338.2024.2348577). An earlier FAERS comparison of marketed biosimilar and biological products appeared in 2023 (PubMed 36892184).
- Italian spontaneous-report comparison of biologics vs non-biologics (Cutroneo et al., 2014, *Drug Saf*); traceability audits of biologics in European reports (Klein et al., 2016, *Drug Saf*; Vermeer et al., 2019).
- Single-product FAERS disproportionality papers on individual biosimilars (2024–2025), which use all-of-FAERS backgrounds and no launch alignment.

**What is specifically missing.**

1. **The subcutaneous immunology wave is unanalysed.** The oncology mAb work stops at 2022; adalimumab (ten products, three interchangeable, high- and low-concentration, citrate-free formulations, different autoinjectors), ustekinumab, denosumab, tocilizumab and aflibercept biosimilars entered 2023–2025 and have never been compared in FAERS. Self-injected products bring device- and injection-site-related events that intravenous oncology biosimilars cannot show.
2. **Attribution is treated as given.** No FAERS study reports what fraction of family reports are INN-only, how that fraction changed after the suffix convention and after each launch, or whether attributable and unattributable reports differ in seriousness — i.e., whether the naming policy achieved traceability.
3. **No launch alignment.** Existing comparisons use the originator's entire reporting history as background. Comparing a biosimilar's first 24 months with (a) the originator in the same calendar months and (b) sibling biosimilars at equal months since launch has not been done.
4. **No label control.** Biosimilar labels copy the reference label, so labelled events are identical by construction; the interesting quantity is the fraction of *unlabelled* PTs and its dependence on product, reporter type and report source (manufacturer patient-support programmes generate many solicited reports).
5. **Nocebo/substitution signals are unmeasured.** "Drug ineffective", "condition aggravated", "product substitution issue" and device PTs are the pre-specified secondary endpoints here; they are typically discarded as non-medical noise.

## Research questions / hypotheses

1. **H1 (attributability).** Among adalimumab reports received 2023-01 → 2026-06, the INN-only fraction is < 25 % and decreases with calendar time; among infliximab reports 2016–2019 (pre-suffix products Inflectra/Renflexis carry suffixes, Remicade does not) it is > 40 %. Attributable and INN-only reports differ in seriousness (OR ≠ 1, adjusted for reporter type).
2. **H2 (calendar-aligned profiles).** Within each family, the Jensen–Shannon divergence between a biosimilar's PT profile in its first 24 months and the originator's profile in the same calendar window is not larger than the divergence between two random halves of the originator (permutation p > 0.05) for ≥ 70 % of biosimilars; exceptions are concentrated in injection-site/device panels.
3. **H3 (Weber).** Biosimilar launch curves show the classical early peak (early/late ratio > 1.5) and the peak month is earlier for interchangeable products (pharmacy-level substitution) than for non-interchangeable ones.
4. **H4 (nocebo/substitution).** Effectiveness-panel and substitution-panel PTs are enriched in biosimilar vs originator reports (panel ROR > 1.5, Fisher p < 0.01) primarily among consumer-reported cases and not among physician-reported cases (reporter × product interaction).
5. **H5 (label control).** After restricting to physician/pharmacist reports, the fraction of reports carrying any unlabelled PT does not differ between biosimilar and originator (difference < 5 percentage points), whereas consumer and manufacturer-solicited reports show a product difference driven by device/injection-site PTs.
6. **H6 (sibling comparison).** At equal months since launch, sibling adalimumab biosimilars differ in device-panel fractions (chi-square across ten products, p < 0.01), consistent with autoinjector design rather than the molecule.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| openFDA drug/event (FAERS) | reports 2015–2026 mentioning any catalogued family: brand/generic/substance names, `manufacturer_name`, roles, reactions, seriousness, `reporttype`, reporter qualification, sex, age | ~20 M total; ~500 k in families | open (free API key optional) | https://open.fda.gov/apis/drug/event/ |
| FAERS quarterly ASCII | `CASEID`/`CASEVERSION` dedup; `RPSR_COD` report source; `DRUG.NDA_NUM` (BLA numbers) | ~1.5 GB/yr | open | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| FDA Purple Book | biosimilar/interchangeable catalogue, BLA numbers, approval and (where listed) marketing status | monthly CSV | open | https://purplebooksearch.fda.gov/ |
| openFDA drug/label (SPL) | boxed warning, warnings & precautions, adverse reactions sections per brand; version history | ~1 label/brand | open | https://open.fda.gov/apis/drug/label/ |
| DailyMed SPL archive | full SPL XML with section LOINC codes (for label-history analysis) | — | open | https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm |
| Medicaid State Drug Utilization Data | quarterly units/prescriptions per NDC → biosimilar uptake denominators | open CSV | open | https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data |
| Medicare Part B / Part D drug spending dashboards | annual claims per HCPCS/product for IV biologics | open | open | https://data.cms.gov/ |

## Methods

1. **Extraction and attribution (`biosim_ae.products`).** Scan bulk FAERS; keep reports with any catalogued family; attribute each drug entry by the hierarchy *biosimilar brand > suffixed INN > originator brand > INN-only*; record `manufacturer_name` for a secondary sponsor-based attribution and flag conflicts. Deduplicate by `CASEID` (ASCII files) or by (date, sex, age, country, drugs, reactions) hash for openFDA-only runs.
2. **Traceability outcomes.** INN-only fraction by family, quarter and reporter type; logistic model `inn_only ~ quarter + reporter + reporttype + serious` (H1).
3. **Launch alignment (`biosim_ae.profiles`).** For each biosimilar with a known US launch month: calendar window (first 24 months) vs originator in the same months; sibling windows at equal months since launch; launch curves and Weber descriptors (H3).
4. **Profile comparison.** Jensen–Shannon divergence on the top-200 PT vocabulary with report-level permutation (H2); comparator-restricted ROR per PT with BH; Mantel–Haenszel ROR stratified by reporter type and sex; PT panels (effectiveness, injection-site, device, hypersensitivity, substitution) as secondary endpoints (H4, H6).
5. **Label control (`biosim_ae.labels`).** Fetch each brand's latest SPL sections; confirm label similarity (k-shingle Jaccard) originator vs biosimilar; classify each report's PTs as labelled/unlabelled; compare unlabelled fractions by product × reporter (H5).
6. **Denominators (optional).** Medicaid SDUD prescriptions per product-quarter to convert counts into crude reporting rates; report both count-based and rate-based comparisons.
7. **Sensitivity.** Suspect-only roles; exclude manufacturer-solicited reports (`reporttype` ≠ 1 and `qualification` = consumer with company source); exclude the first 3 months after launch; alternative vocab sizes (100/500 PTs).

## Evaluation & statistics

- Primary endpoints: per-family JS divergence with permutation p (H2); INN-only fraction trend with 95 % CI (H1).
- Secondary: panel RORs with Fisher exact p and BH across 5 panels × families; PT-level comparator RORs with BH (q ≤ 0.05, a ≥ 3).
- Interaction tests: logistic `panel_event ~ product × reporter_type` per family (H4, H5); chi-square across sibling products with post-hoc pairwise tests (H6).
- Multiplicity: BH within each family-level PT screen; Holm across the six hypotheses' primary tests.
- Nulls: random splits of originator reports (empirical null for JS); permuted launch dates (null for Weber descriptors); negative-control families (filgrastim/pegfilgrastim, on the market since 2015–2018, should show stable profiles).
- Leakage/bias controls: no all-of-FAERS backgrounds; calendar and time-since-launch alignment reported side by side; attribution conflicts reported, not silently resolved.
- Reporting per READUS-PV (Fusaroli et al., 2024, *Drug Saf*).

## Publishable angle

- **Headline result.** "In the first three years of the US adalimumab biosimilar market, X % of FAERS reports were attributable to a product, biosimilar and originator profiles were indistinguishable after launch alignment except for device- and effectiveness-related reports that were carried by consumer and manufacturer-solicited cases." A regulatory-relevant statement about both naming policy and nocebo.
- **Venues.** *Drug Safety*; *BioDrugs*; *Pharmacoepidemiology and Drug Safety*; *Clinical Pharmacology & Therapeutics*; *JAMA Network Open* (for the traceability/naming-policy angle).
- **Follow-ups.** Repeat for ustekinumab/denosumab as their 24-month windows mature (2027); link SPL version history to test whether biosimilar labels lag originator label updates (see `unlabeled-adverse-event-mining`); EudraVigilance replication where batch numbers are more often recorded.

## Risks, confounds & mitigations

- **Channelling**: early biosimilar users may be switchers with longer disease duration or payer-driven substitution. Mitigation: compare within reporter type, restrict to reports with an indication field, sibling comparisons at equal time since launch.
- **Manufacturer-solicited reporting** (patient-support programmes) inflates non-serious consumer reports for whichever company runs the larger programme. Mitigation: stratify by `reporttype`/`qualification` and report-source codes from the ASCII files; treat solicited reports as a separate stratum.
- **Attribution error** (brand mentioned in narrative but not coded; `manufacturer_name` reflects the label repackager). Mitigation: hierarchy with conflict flags; sensitivity restricting to reports with a unique product.
- **Catalogue drift**: launches, interchangeability designations and withdrawals change quarterly. Mitigation: re-verify against the Purple Book at analysis time; `us_launch=None` products are excluded from launch-aligned analyses.
- **Small numbers** for recently launched families (eculizumab, natalizumab). Mitigation: pre-specify minimum 200 reports per product-window; otherwise descriptive only.
- **Same-label assumption fails** for carve-out indications and device instructions. Mitigation: label similarity is measured, not assumed; sections with low similarity are excluded from the expectedness text.

## Milestones

- [ ] Bulk FAERS scan; family extraction; attribution with conflict report
- [ ] Purple Book reconciliation of catalogue (launch months, interchangeability)
- [ ] Traceability analysis (H1) and figure: INN-only fraction by family × quarter
- [ ] Launch curves, Weber descriptors for all launched biosimilars (H3)
- [ ] Calendar- and sibling-aligned JS/ROR/panel comparisons (H2, H4, H6)
- [ ] SPL retrieval, label similarity, unlabelled-fraction analysis (H5)
- [ ] Denominator-based rates (Medicaid SDUD), sensitivity analyses
- [ ] Manuscript (READUS-PV checklist), code and attribution tables released

## Ethics / data-use notes

- FAERS and SPL data are public and de-identified; no patient contact; secondary-use exemption typical, confirm locally.
- Disproportionality outputs are not incidence comparisons; product-level statements must carry the openFDA/FAERS limitations disclaimer, especially for named commercial products.
- Never commit downloaded data; `data/` is git-ignored; the API key comes from `OPENFDA_API_KEY`.

## Related projects in this repository

`faers-reporting-bias` (reporter-type and stimulated-reporting adjustment), `unlabeled-adverse-event-mining` (label-expectedness mining), `faers-signal-ehr-validation`. This project is self-contained.
