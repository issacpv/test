# label-change-prediction

**From FAERS trajectory to boxed warning: a landmark (dynamic) survival model that predicts which drug-event pairs will receive an FDA safety-related labeling change, built with strict temporal cutoffs, lead-time estimation and an explicit notoriety-bias audit.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (DailyMed SPL version-history client and LOINC-section parser/differ, SrLC export parser, openFDA quarterly count client, BCPNN/ROR trajectory features with landmark cutoffs, landmark dataset builder, pooled-logistic hazard model with time-dependent AUC, lead-time and notoriety statistics).
- Difficulty: MSc-to-PhD level (pharmacoepidemiology + survival/dynamic prediction). No GPU.
- Timeline: 8-12 months (3 months label-event extraction and curation, 2 months FAERS feature pipeline, 3 months modelling and validation, 2 months writing).
- Compute: laptop for the openFDA-count design; a workstation with ~64 GB RAM if the FAERS quarterly ASCII files are used to rebuild the drug x PT x quarter table (~25M reports).

## Background

Since FDAAA 2007 (section 505(o)(4)) FDA can require safety labeling changes (SLCs) when new safety information emerges; boxed warnings are the strongest. FDA staff analyses show most post-approval SLCs cite spontaneous reports (AERS/FAERS) as a major evidence source (Lester et al., 2013, Pharmacoepidemiol Drug Saf; Ishiguro et al., 2012, Pharmacoepidemiol Drug Saf), and roughly a third of novel therapeutics acquire a post-market safety event within ~10 years (Downing et al., 2017, JAMA; Pinnow et al., 2018, Clin Pharmacol Ther). A 2024 longitudinal review found ~78% of boxed warnings rest on post-marketing evidence (A Longitudinal Analysis of Black Box Warnings, Cureus, 2024). Since 2016 the FDA Drug Safety-related Labeling Changes (SrLC) database records every SLC with the section changed and the date, and DailyMed keeps the full version history of each Structured Product Label (SPL). Together these give, for the first time, an event time for each (drug, adverse event) pair: the quarter in which the event entered the Boxed Warning or Warnings and Precautions section.

The question a pharmacovigilance scientist actually faces is dynamic: *given the FAERS history of this pair up to today, how likely is a labeling change in the next two years, and how much earlier could it have been called?* That is a landmark / dynamic prediction problem (van Houwelingen & Putter, 2011), not the cross-sectional classification the earlier literature solved.

## The research gap

**What has been done**

- Gurulingappa et al. (2013, Pharmacoepidemiol Drug Saf) predicted 2010 label changes from automatically detected adverse-event mentions (text + FAERS) with a cross-sectional design; later ensemble approaches added label text as features. These snapshot designs cannot say *when* a signal became predictive and risk using post-change data (reports stimulated by the change itself).
- Signal-detection benchmarking (Harpaz et al., 2013, Clin Pharmacol Ther; Ryan et al., 2013, Drug Saf, the OMOP reference set) evaluates algorithms against static "known ADR" reference sets, not against dated label events, and ignores time-to-detection.
- Notoriety bias, i.e. reporting stimulated by safety communications, is documented (Pariente et al., 2007, Drug Saf) but is not routinely handled in FAERS prediction models.
- Descriptive SrLC/boxed-warning epidemiology exists (Solotke et al., 2018, Expert Opin Drug Saf; Cureus 2024) without predictive modelling.
- Shrinkage disproportionality (BCPNN; Norén, Hopstadius & Bate, 2013, Stat Methods Med Res) is standard, but trajectory features (slope, persistence) are rarely used as predictors.

**What is missing (checked against 2023-2026 literature)**

1. A dated, pair-level label-event table (drug x MedDRA PT x section x effective quarter) built from SrLC + DailyMed SPL history, released as a reusable benchmark with explicit "already labelled at baseline" status.
2. A landmark survival formulation with strict feature cutoffs (only reports received on or before the landmark quarter), dynamic discrimination (time-dependent AUC at 2/4/8 quarters), calibration, and grouped-by-drug and temporal validation.
3. A lead-time analysis: distribution of quarters between first sustained FAERS signal and the labeling change, stratified by section (boxed vs W&P) and by whether the SLC cites post-marketing evidence.
4. A quantified notoriety-bias audit: how much cross-sectional designs overstate performance because they include post-change stimulated reports.

## Research questions / hypotheses

1. **H1 (dynamic predictability).** FAERS trajectory features computed strictly up to landmark L (IC025 level, 4- and 8-quarter IC slope, consecutive quarters with IC025 > 0, cumulative and recent report counts, serious and healthcare-professional report shares, drug age, class-warning indicator) predict an SLC for the pair within 8 quarters with time-dependent AUC >= 0.75 in grouped-by-drug CV, exceeding a static-IC025 baseline by >= 0.05 AUC and a drug-age-only baseline by >= 0.15.
2. **H2 (lead time).** For SLCs whose SrLC summary cites post-marketing reports, the median lead between first sustained signal (IC025 > 0 for 2 consecutive quarters) and the effective quarter is >= 4 quarters; for trial-driven SLCs the lead is centred near 0 or negative.
3. **H3 (section difference).** Warnings-and-Precautions additions are more FAERS-predictable than boxed warnings (which more often come from trials, REMS or class actions); test AUC(W&P) > AUC(boxed) with a bootstrap difference CI.
4. **H4 (class labelling).** A pair whose same-EPC-class drugs already carry the warning has a higher hazard (HR >= 2) independent of its own FAERS signal.
5. **H5 (notoriety).** Reports for the pair rise after the SLC (ratio of the 4 post- to 4 pre-quarters >= 1.5 after normalising by the drug's total reports); a deliberately leaky design (features including post-change quarters) inflates AUC by >= 0.05 relative to the strict design.
6. **H6 (temporal transportability).** A model trained on landmarks <= 2019Q4 keeps AUC within 0.05 on landmarks 2020Q1-2024Q4.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| FDA Drug Safety-related Labeling Changes (SrLC) | Event table: drug, application, section (Boxed Warning, W&P, Contraindications, Adverse Reactions, ...), approval/effective date, summary text | all SLCs since Jan 2016, quarterly updates; downloadable | Open | https://www.fda.gov/drugs/drug-safety-and-availability/drug-safety-related-labeling-changes-srlc-database-overview-updates-safety-information-fda-approved |
| DailyMed SPL version history + versioned SPL XML (web services v2) | Extend event table before 2016 and validate SrLC dates by diffing LOINC-coded sections between versions | ~150k SPLs, multiple versions each | Open | https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm |
| openFDA `drug/label` | Current label incl. `boxed_warning`, `warnings_and_cautions`, `adverse_reactions`, `set_id`, `effective_time`, `version` | ~150k labels | Open | https://api.fda.gov/drug/label.json |
| FAERS via openFDA `drug/event` (count queries) | Quarterly drug x PT counts (a, b, c, d) by `receivedate`; serious and reporter-type splits | 2004-present | Open (optional `OPENFDA_API_KEY`) | https://api.fda.gov/drug/event.json |
| FAERS quarterly ASCII files | Optional exact rebuild with case de-duplication | ~1-2 GB/quarter | Open | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| openFDA `drug/drugsfda` (Drugs@FDA) | Approval dates (drug age), application numbers to join SrLC | ~25k applications | Open | https://api.fda.gov/drug/drugsfda.json |
| FDA FAERS "Potential Signals of Serious Risks" quarterly postings (FDAAA 921) | Intermediate outcome: FDA-acknowledged signal dates | quarterly lists since 2008 | Open | https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/potential-signals-serious-risksnew-safety-information-identified-fda-adverse-event-reporting-system-faers |
| MedDRA (PT -> HLT/SOC hierarchy) | Grouping and term normalisation | - | Licence required (free for academic/non-profit via MSSO registration) | https://www.meddra.org |
| OMOP / Harpaz reference sets | Sanity check of static signal detection | ~400 pairs | Open (paper supplements) | see cited papers |

No credentialed clinical data are involved.

## Methods

1. **Label-event table** (`src/label_change/dailymed_spl.py`):
   - Parse the SrLC export (`parse_srlc_export`) into (drug, application, section, date, summary); map summary text to MedDRA PTs with a PT/LLT dictionary plus manual curation of the ~2,000 boxed/W&P rows (two annotators, kappa reported).
   - For each drug, list SPL set IDs (`DailyMedClient.search_spls`), fetch `history` (version numbers + effective dates), download consecutive versions and diff the LOINC sections (`34066-1` boxed warning, `43685-7` W&P, `34070-3` contraindications, `34084-4` adverse reactions) with `diff_sections`/`new_terms`; a PT entering the boxed or W&P section defines an event quarter. Reconcile with SrLC (agreement rate, median date offset).
   - Baseline label status: PTs present in the earliest available version are "already labelled" and excluded from the risk set.
2. **FAERS pair-quarter table** (`openfda_counts.py`): for each drug (primary suspect) and each PT in the union of (labelled PTs, top-300 reported PTs for that drug), quarterly counts of a via `count=receivedate`; drug totals and PT totals and N per quarter from separate count queries; build cumulative 2x2 tables per quarter (`faers_trajectories.cumulative_2x2`).
3. **Trajectory features at landmarks** (`faers_trajectories.landmark_features`): IC and IC025 (BCPNN shrinkage), ROR and lower CI, slopes, persistence, recent share, serious share, HCP share, drug age, class-warning indicator; only quarters <= L enter.
4. **Landmark dataset** (`landmark_model.build_landmark_dataset`): landmarks every 2 quarters 2008Q1-2023Q4; risk set = pairs with >= 3 cumulative reports and no prior event; outcome = event within horizon h in {2, 4, 8} quarters; censoring flagged when follow-up is incomplete.
5. **Models**: pooled logistic (discrete-time hazard) with landmark-time interactions (super-landmark model); gradient boosting (sklearn `HistGradientBoostingClassifier`) as flexible comparator; baselines: static IC025 threshold, drug-age only. Grouped CV by drug; temporal split for H6.
6. **Lead time and notoriety** (`landmark_model.lead_times`, `faers_trajectories.notoriety_ratio`).
7. **Tools**: `requests`, `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `lxml`/`xml.etree`; optional `scispacy` for PT extraction from summaries.

## Evaluation & statistics

- Primary: time-dependent AUC at h = 8 quarters (inverse-probability-of-censoring weighted where follow-up is incomplete), plus AUC at h = 2, 4; Brier score and calibration slope/intercept; decision curves at plausible review thresholds.
- Validation: 5-fold grouped CV by drug (a drug's pairs never straddle folds); temporal validation (train landmarks <= 2019Q4, test >= 2020Q1); bootstrap CIs (pairs clustered by drug).
- Leakage prevention: features use `receivedate <= landmark quarter end`; label events dated by SPL effective date; pairs already labelled at the first available SPL excluded; MedDRA version frozen; no features derived from label text after the landmark.
- Multiple comparisons: H1-H3 confirmatory in fixed order; exploratory feature-importance and SOC-level analyses with Benjamini-Hochberg.
- Nulls: permute event quarters across pairs within drug (preserves drug-level base rates) to obtain the null AUC distribution; negative-control pairs from the OMOP reference set should have low predicted risk.
- Sensitivity: IC vs ROR features; `receivedate` vs FAERS release quarter; with/without literature/lawyer reports; with/without class-warning feature.

## Publishable angle

- **Headline**: "A dynamic model using only FAERS reports received before each landmark predicts safety-related labeling changes within two years with time-dependent AUC of X, gives a median lead of Y quarters for post-marketing-driven changes, and shows that cross-sectional designs overstate performance by Z AUC points because of reporting stimulated by the change itself." Plus a public benchmark: the dated pair-level label-event table.
- Target venues: *Drug Safety*; *Clinical Pharmacology & Therapeutics*; *Pharmacoepidemiology and Drug Safety*; *JAMIA* (methods/benchmark); *npj Digital Medicine*.
- Follow-ups: add EudraVigilance and VigiBase (WHO, restricted) trajectories for transportability; incorporate FDA 921 signal postings as an intermediate state (multi-state model); LLM-assisted extraction of SLC summaries (local models only); apply to biologics and biosimilars.

## Risks, confounds & mitigations

- **Event-date imprecision**: SPL effective dates lag FDA approval of the SLC; SrLC gives the approval date. Mitigation: use SrLC dates when available, DailyMed for validation and pre-2016 extension; sensitivity +/- 1 quarter.
- **Notoriety and reverse causation**: handled by strict cutoffs; the leaky design is estimated only to quantify bias (H5).
- **Reporting-rate confounding by drug**: normalise by drug totals; drug fixed effects are not possible in prediction, so use drug-level covariates (age, class, total volume) and grouped CV.
- **Term mapping noise**: SrLC summaries are free text; DailyMed section diffs contain non-event edits. Mitigation: dual annotation, restrict to PT-level matches, report inter-annotator agreement, sensitivity to HLT-level matching.
- **Class effects and multiplicity of events per drug**: cluster bootstrap by drug; class-warning feature; report drug-level performance.
- **openFDA limits**: 1,000 count buckets per call and 25,000-record skip ceiling; the count design and date windowing avoid both.
- **FAERS paper-mill perception**: pre-register; the contribution is the dated benchmark and dynamic evaluation, not another disproportionality table.

## Milestones

- [ ] SrLC export parsed; PT mapping curated; DailyMed diffs for the same drugs; agreement report.
- [ ] Pair-quarter FAERS table for all drugs with an SrLC entry + matched controls.
- [ ] Landmark dataset built; descriptive lead-time and notoriety analysis (H2, H5).
- [ ] Pooled logistic + gradient boosting; grouped CV; time-dependent AUC/Brier/calibration (H1, H3, H4).
- [ ] Temporal validation (H6); sensitivity analyses.
- [ ] Benchmark release (event table + landmark features), preprint.

## Ethics / data-use notes

- All sources are public and de-identified. FAERS reports must not be re-identified; keep the `OPENFDA_API_KEY` in the environment only.
- MedDRA is licensed: do not redistribute the dictionary; release PT names only as they appear in public FAERS/label data.
- Cite FDA SrLC, DailyMed/NLM and openFDA; follow DailyMed's request-rate guidance.
- Never commit downloaded labels, counts or derived pair tables (`data/` is git-ignored).

## Related projects

- `unlabeled-adverse-event-mining` (static label-vs-FAERS comparison; this project adds the time axis and the label-change outcome).
- `faers-reporting-bias` (reporter-type and stimulated-reporting models, reusable for the notoriety audit).
