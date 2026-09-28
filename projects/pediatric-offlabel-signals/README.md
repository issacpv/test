# pediatric-offlabel-signals: machine-readable labelled age floors turn FAERS into an age-off-label pharmacovigilance system for children

**One-sentence pitch.** Extract the labelled paediatric age floor of every US drug from SPL `pediatric_use`/`indications_and_usage` text, classify each paediatric FAERS report as on-label-age, below-floor (probable off-label by age) or no-paediatric-labelling, and test — with age-matched backgrounds and FDA paediatric labelling changes as natural experiments — whether age-off-label reports carry different, more serious and more *unlabelled* adverse events.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc-level (pharmacoepidemiology + NLP-lite); 6–9 months to the first paper; 12 months including the labelling-change ITS study and the manual validation of age floors.
- Compute: CPU only. Paediatric reports are ~7–8 % of FAERS (~1.5 M); labels are ~50 k SPL documents (openFDA `drug/label` bulk ≈ 1 GB). Everything fits on a laptop.

## Background

Off-label prescribing to children is common (Kimland & Odlind, 2012, *Clin Pharmacol Ther*; Yackey et al., 2019, *Hosp Pediatr*; AAP policy statement, Frattarelli et al., 2014, *Pediatrics*) and is associated with adverse drug reactions in prospective hospital cohorts (Horen, Montastruc & Lapeyre-Mestre, 2002, *Br J Clin Pharmacol*; Neubert et al., 2004, *Drug Saf*; Bellis et al., 2013, *BMC Med*; Bellis et al., 2014, *Br J Clin Pharmacol*). BPCA/PREA have produced hundreds of paediatric labelling changes since 1998, each of which converts a previously off-label age range into a labelled one — a natural experiment that spontaneous reports have never been used to evaluate. Spontaneous-report work in children (Star et al., 2011, *Drug Saf*, VigiBase; the GRiP paediatric reference set, Osokogu et al., 2015, *Drug Saf*; age-stratified signal detection, Osokogu et al., 2016, *Drug Saf*) has focused on drug–event signals, not on the on-/off-label status of the reported use, because that status is not coded in FAERS. The one FAERS field that could reveal it — age — is precise to the day for most paediatric reports, and the labelled age floor is written in every SPL in a small number of stereotyped phrasings.

## The research gap

**What has been done.**

- Many single-drug or single-class paediatric FAERS disproportionality studies in 2024–2026 (ADHD medications with age stratification 2014–2024; third-generation antiseizure drugs in children, *Therapeutic Advances in Drug Safety*, 2025, noting reports below the approved age as evidence of off-label use; TNF-α inhibitors in children, 2004–2024; dasatinib in children; drug-related aggression in under-18s, 2004–2025 Q2). Off-label use is inferred informally, drug by drug.
- A 2026 cross-system benchmark of paediatric signal concordance using FAERS as source and Canada Vigilance / JADER as comparators (*Frontiers in Pharmacology*, 2026); machine-learning prediction of paediatric ADRs from chemical/biological features (*Communications Chemistry*, 2025).
- Reference sets and method evaluations for paediatric signal detection (GRiP; Osokogu et al., 2015, 2016).
- FDA's Pediatric Labeling Changes table (public) lists every labelling change with date and age range but has not been linked to reporting data.

**What is specifically missing.**

1. **No systematic, machine-readable age-floor resource.** Labelled paediatric age ranges have never been extracted at scale from SPL and linked to FAERS ages; consequently "off-label by age" has never been measured across the whole paediatric FAERS.
2. **Age-matched comparison of off-label vs on-label reports.** Existing below-approved-age observations are counts, not disproportionality analyses with age-specific backgrounds (a below-floor report is necessarily younger than an on-label one, so age must be matched at 1-year resolution).
3. **Natural experiments.** Whether paediatric labelling changes (a) change the volume/share of below-floor reports and (b) change the profile of reported events (more labelled, fewer unlabelled) is unknown; interrupted time series around BPCA/PREA changes has not been attempted.
4. **Seriousness and label expectedness.** Whether age-off-label reports are more often serious/fatal and more often carry events absent from the label (the mechanism by which off-label use generates *unknown* risk) has not been quantified outside small hospital cohorts.

## Research questions / hypotheses

1. **H1 (extraction accuracy).** A regex extractor of labelled age floors from `pediatric_use` + `indications_and_usage` achieves ≥ 90 % exact agreement with a hand-labelled sample of 300 labels and with the FDA Pediatric Labeling Changes table for changes 2015–2025.
2. **H2 (prevalence).** Among paediatric FAERS reports 2015–2026 with a numeric age and a mapped suspect drug, 25–40 % are below-floor or no-paediatric-labelling; the share is highest in infants and for psychotropics, proton-pump inhibitors and antiemetics.
3. **H3 (seriousness).** Below-floor reports have higher adjusted odds of serious outcome and death than on-label-age reports of the same drugs (OR ≥ 1.3 after adjustment for 1-year age, sex, reporter type and drug).
4. **H4 (profile).** With 1-year-age-matched backgrounds, ≥ 10 % of drugs show at least one PT with a below-floor excess (ratio of age-specific RORs > 1, BH q ≤ 0.05); the excess PTs are enriched for dosing/medication-error PTs and for events absent from the label's adverse-reactions section.
5. **H5 (labelling changes).** After a paediatric labelling change that lowers the floor, the below-floor share of reports drops (level change < 0.7 in segmented Poisson ITS) while total paediatric reports for the drug rise (stimulated reporting); the fraction of reports with unlabelled events falls.
6. **H6 (age-group coding).** Reports with only `patientagegroup` (no numeric age) are more often serious and more often from consumers; excluding them biases the off-label share downwards by ≥ 5 points (a measurable coding-completeness bias).

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| openFDA drug/event (FAERS) | paediatric reports 2015–2026: onset age + unit, `patientagegroup`, suspect drugs, indications, reactions, seriousness, reporter | ~1.5 M paediatric of ~20 M | open (free API key optional) | https://open.fda.gov/apis/drug/event/ |
| openFDA drug/label (SPL) | `pediatric_use`, `indications_and_usage`, `dosage_and_administration`, `adverse_reactions`, `boxed_warning`; `effective_time`, `version` | ~50 k human Rx labels | open | https://open.fda.gov/apis/drug/label/ |
| DailyMed SPL archive | label version history (LOINC 34081-0 = pediatric use) for change dates | ~10 GB | open | https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm |
| FDA Pediatric Labeling Changes table | validation set for floors; dates for the ITS analysis | ~900 rows | open (web table / spreadsheet) | https://www.fda.gov/science-research/pediatrics/pediatric-labeling-changes |
| FAERS quarterly ASCII | `CASEID` dedup; `AGE`, `AGE_COD`, `AGE_GRP`; `INDI_PT` | ~1.5 GB/yr | open | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| GRiP paediatric reference set (Osokogu et al., 2015) | known paediatric drug–event pairs for method calibration | ~ hundreds of pairs | open (paper supplement) | https://doi.org/10.1007/s40264-015-0265-0 |
| MEPS / NHANES (optional) | paediatric users by age for denominators | — | open | https://meps.ahrq.gov/ |

## Methods

1. **Paediatric cohort (`peds_offlabel.age`).** Normalise onset age to years (unit codes 800–805), assign ICH E11 bands (neonate/infant/child/adolescent), fall back to `patientagegroup` when numeric age is missing (analysed separately, H6). Flatten suspect drugs with `openfda.generic_name` (unmapped verbatim names retained and flagged).
2. **Age floors (`peds_offlabel.label_ages`).** For each generic name, take the latest SPL with a `pediatric_use` section; extract the minimum established age from stereotyped phrases ("X years of age and older", "X months to less than Y years", "from birth", "neonates"), use "below X … not established" as a consistent floor, classify labels with only a general "not established" statement as no-paediatric-labelling, flag weight-based thresholds. Validate against 300 hand-labelled labels and the FDA table (H1); disagreements are adjudicated and the regex iterated (transparent, auditable; no LLM required, though an LLM-assisted pass can be used as a second annotator).
3. **Classification (`peds_offlabel.offlabel_signals.classify_reports`).** Each (report, suspect drug) → on-label-age / below-floor / no-paediatric-labelling / unknown; report-level class = worst class among suspect drugs.
4. **Age-matched signals (`offlabel_signal_table`).** For each drug and PT: Mantel–Haenszel ROR of the drug vs all other paediatric reports within 1-year age strata, separately for below-floor and on-label-age reports; ratio of the two RORs with Wald CI; BH across PTs per drug; enrichment of excess PTs in medication-error HLGTs and in unlabelled PTs (label `adverse_reactions` matching, as in `unlabeled-adverse-event-mining`).
5. **Seriousness (`seriousness_model`).** Report-level logistic regression of serious/death on below-floor with 1-year age (spline or bins), sex, reporter type and drug fixed effects (H3).
6. **Natural experiments (`segmented_poisson_its`).** For each labelling change 2015–2024 that lowered the floor (from the FDA table), monthly below-floor share and total paediatric reports 36 months before/after; segmented Poisson with offset; meta-analysis of level changes across drugs (H5).
7. **Denominators (optional).** MEPS paediatric users by age to convert to crude rates.

## Evaluation & statistics

- Primary endpoints: extraction accuracy (exact agreement, 95 % CI; H1); adjusted OR for seriousness (H3); pooled ITS level change with random-effects meta-analysis (H5).
- Signal screen: BH q ≤ 0.05 per drug; minimum 3 reports per class; sensitivity with min 5 and with 2-year age bins.
- Nulls: permute label class within 1-year age × drug strata (null for ratio-of-ROR); placebo labelling-change dates (random dates for ITS); negative-control drugs with no paediatric labelling change.
- Confounding: age matched at 1-year resolution (bands are too coarse — a floor inside a band leaves residual age confounding, demonstrated in the tests); reporter type and country as covariates; indication-restricted sensitivity analyses (e.g., quetiapine for insomnia vs bipolar).
- Leakage: floors are derived from labels only, never from FAERS; validation labels are held out from regex development.
- Reporting per READUS-PV (Fusaroli et al., 2024, *Drug Saf*).

## Publishable angle

- **Headline result.** "One in three paediatric FAERS reports concerns a use below the labelled age; these reports are X times more often fatal and carry Y % more unlabelled events, and paediatric labelling changes reduce the below-floor share by Z % within two years." Plus a released age-floor table for all US drugs (a reusable resource).
- **Venues.** *Pediatrics* or *JAMA Pediatrics* (policy-relevant headline); *Drug Safety*; *Pharmacoepidemiology and Drug Safety*; *Journal of the American Medical Informatics Association* or *Journal of Biomedical Informatics* (age-floor extraction resource).
- **Follow-ups.** Weight-based and dose-per-kg off-label use from `drugdosagetext`; extension to EudraVigilance (age fields comparable); LLM-based floor extraction benchmarked against the regex.

## Risks, confounds & mitigations

- **Age is not use.** Below-floor age is only a proxy for off-label use (indication may still be on-label in a lower age range for a different formulation). Mitigation: floors per generic name across all formulations (minimum); sensitivity restricted to reports with an indication that appears in the label.
- **Label heterogeneity** (multiple SPLs per generic; repackagers; outdated labels). Mitigation: take the minimum floor across current labels; use DailyMed history to date floors at report time rather than the current label.
- **Reporting bias by age**: paediatric reports come disproportionately from specialists and litigation (e.g., antipsychotics). Mitigation: reporter-type stratification, lawyer-report exclusion, drug fixed effects.
- **Stimulated reporting after labelling changes** confounds the ITS share outcome. Mitigation: model shares and counts jointly; placebo dates.
- **Regex brittleness.** Mitigation: validation set, adjudication, publication of the evidence sentence per floor.

## Milestones

- [ ] Bulk label pull; floor extraction for all generic names; evidence table
- [ ] 300-label validation set + FDA table reconciliation (H1)
- [ ] Paediatric FAERS extraction, dedup, classification; prevalence figures (H2, H6)
- [ ] Seriousness models (H3)
- [ ] Age-matched signal tables, medication-error and unlabelled-PT enrichment (H4)
- [ ] ITS around labelling changes with meta-analysis and placebo dates (H5)
- [ ] Resource release (age-floor table) and manuscript

## Ethics / data-use notes

- FAERS and SPL are public, de-identified; paediatric ages are not identifying at the precision used, and no narrative text is used. Confirm secondary-use exemption locally.
- Statements about specific drugs' off-label safety are hypothesis-generating and must be framed as such; avoid implying prescriber fault.
- Never commit downloaded data; `data/` is git-ignored; API key from `OPENFDA_API_KEY`.

## Related projects in this repository

`unlabeled-adverse-event-mining` (label-expectedness mining, from which the unlabelled-PT enrichment step is adapted), `faers-reporting-bias` (reporter-type adjustment, stimulated-reporting ITS). This project is self-contained.
