# sex-stratified-brain-age — Does training brain-age models separately by sex make the brain-age delta fairer, more reliable, or more clinically informative?

A pre-registered factorial audit of brain-age modelling strategies (pooled without sex, pooled with sex covariate, sex-stratified, pooled with sex-specific bias correction) × head-size handling × feature set across five open lifespan cohorts, judged on sex-gap in error, spurious sex differences in delta, test-retest reliability, cross-cohort transport and association with cognition/vascular risk — with a simulation that shows which strategy *manufactures* a sex difference in brain-age delta.

## Status / difficulty / timeline / compute

- Status: design + starter code (FreeSurfer table parsing with four head-size corrections, strategy-aware brain-age estimator with bias correction and grouped CV, evaluation metrics including fairness gaps, ICC and outcome associations, and a simulation engine with known ground truth).
- Difficulty: MSc-level for the ROI-feature version (6-9 months); PhD-chapter if a voxel/CNN arm (pyment/DeepBrainNet re-training) is included (+4 months).
- Compute: ROI models train in seconds; the full factorial (4 strategies × 5 TIV corrections × 3 feature sets × 5 cohorts × 10 CV repeats) is a few CPU-hours. CNN re-training per sex requires one 24 GB GPU for ~1-2 days per configuration.
- Related project in this repository: `brain-age-transportability` (cross-cohort transport of brain-age models) and `lifespan-normative-models`. This folder is self-contained.

## Background

Brain-age prediction (Cole and Franke, 2017, Trends in Neurosciences; Franke and Gaser, 2019, Frontiers in Neurology) produces a delta (predicted − chronological age) used as a marker of deviation from normative ageing and associated with mortality (Cole et al., 2018, Molecular Psychiatry), disease and cognition. Delta estimates are known to be biased by age (regression to the mean; Smith et al., 2019, NeuroImage; Beheshti et al., 2019, NeuroImage: Clinical; Liang et al., 2019, Human Brain Mapping; de Lange and Cole, 2020, NeuroImage: Clinical). Sex is handled inconsistently: many models ignore it, some include it as a covariate, and large consortium pipelines (e.g., the ENIGMA-MDD brain-age model, Han et al., 2021, Molecular Psychiatry) train separate models for males and females by convention. Sex differences in delta are then reported as biology: for instance, recent conference reports along the Alzheimer's trajectory find that female brains "appear older" in most diagnostic groups after covariate adjustment, while other ensembles report no sex difference in cognitively normal adults.

Two facts make this a live methodological problem. First, head size (total intracranial volume, TIV) differs by ~10% between sexes and drives most raw sex differences in regional volumes (Sanchis-Segura et al., 2019, Biology of Sex Differences); how TIV is handled changes which sex differences survive. Second, a 14-cohort longitudinal analysis of 12,638 MRIs (Ravndal, Fjell et al., 2025, PNAS) found that sex differences in healthy structural ageing trajectories are small and mostly show *greater* decline in men, and concluded they are unlikely to explain higher AD prevalence in women. If true sex differences in ageing are small, then sizeable sex differences in brain-age delta are likely artefacts of modelling choices — which is testable.

## The research gap

What has been done:

- Fairness audits: Piçarra and Glocker (2023, FAIMI workshop at MICCAI; arXiv 2309.10835) found statistically significant differences in brain-age model error between males and females (and between racial groups) in a large UK-Biobank-trained model; Dibaji et al. (2023, same workshop) found that models trained on one sex generalize worse to the other and that performance depends on training-set sex balance. Neither evaluated clinical informativeness of the resulting deltas or cross-cohort transport.
- Sex-specific predictors of delta: Sanford et al. (2022, Human Brain Mapping) examined sex differences in predictors and regional patterns of brain-age gap using a pooled model; sex-stratified training was not compared against pooled training.
- Workflow benchmarks: More et al. (2023, NeuroImage) compared 128 brain-age workflows (features × algorithms) on multiple cohorts, but treated sex only as a covariate and did not analyse sex-gap metrics.
- Bias-correction literature (Smith 2019; Beheshti 2019; de Lange and Cole 2020) corrects the age dependence of delta but is silent on whether correction should be done within sex.
- A 2025 Imaging Neuroscience study on bias and generalizability of brain-age models reports sex-related error differences across models, again without a stratified-vs-pooled design.

What is specifically missing:

1. A factorial comparison of strategies (pooled without sex, pooled + sex covariate, sex-stratified, pooled + sex-specific bias correction) with the *same* features, folds and cohorts, evaluated on (a) sex gap in MAE, (b) residual association of delta with sex, (c) test-retest reliability, (d) transport to unseen cohorts, and (e) strength of association with cognition, CDR and vascular risk — the last being the only reason to prefer a strategy clinically.
2. Head-size handling as a crossed factor (no correction, proportion, residual, power-proportion; Liu et al., 2014, Frontiers in Neuroscience) — the most likely mechanism by which pooled models manufacture a sex difference in delta.
3. A ground-truth simulation: generate cohorts in which sex differs only in head size (no ageing difference), only in ageing slope, or both, and show which strategy recovers the true delta-outcome association without producing a spurious sex difference. No brain-age paper has done this.
4. Longitudinal validation: does the sex-specific delta predict within-person change (OASIS-3 follow-up MRI and CDR progression) better than the pooled delta? Sex-stratified training halves the sample; the trade-off between variance and bias has never been quantified.
5. Fairness-utility trade-off reporting: if stratification equalizes error but weakens the delta-outcome association, that is the actual finding practitioners need.

## Research questions / hypotheses

1. RQ1 (error gap). Across cohorts, does the sex gap in MAE (|MAE_F − MAE_M|) differ by strategy? H1: pooled-without-sex has the largest gap (0.3-0.8 years); stratified and pooled+sex reduce it to < 0.2 years; stratified has higher overall MAE (+0.2-0.5 years) because of halved training data.
2. RQ2 (spurious sex effect). After age-bias correction, is the mean delta different between sexes? H2: with uncorrected volumes and pooled training, women show a positive mean delta shift of 0.5-1.5 years driven by TIV; with residual/power-proportion TIV correction or sex-aware strategies the shift is < 0.3 years. The simulation with "head-size only" ground truth reproduces this pattern.
3. RQ3 (clinical informativeness). Is the partial association of delta with CDR-SB/MMSE (OASIS-3), fluid intelligence (Cam-CAN, HCP-A) and vascular risk (BP, BMI in Cam-CAN/HCP-A) stronger for stratified deltas? H3: no meaningful gain (|Δβ| < 10% of β); pooled+sex with TIV correction is not inferior.
4. RQ4 (reliability). Test-retest ICC of delta (HCP-YA retest; OASIS-3 same-session repeat scans) by strategy. H4: ICC is lower for stratified models (smaller training sets → noisier weights) by 0.02-0.05.
5. RQ5 (transport). Train on cohort A, test on B, by sex. H5: sex-stratified models transport worse (larger MAE increase) than pooled+sex, especially where the unseen cohort's sex ratio differs.
6. RQ6 (longitudinal). Does baseline delta predict annualized change in hippocampal volume / CDR progression in OASIS-3, and does the effect differ by strategy or sex? H6: association is present for all strategies and does not differ by sex.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| IXI | T1w, age, sex, site (3 London hospitals, 1.5T and 3T) | ~580 subjects, 20-86 y | Open (CC BY-SA 3.0) | https://brain-development.org/ixi-dataset/ |
| Cam-CAN | T1w, age, sex, cognition (fluid intelligence), BP/BMI, cardiovascular questionnaire | ~650 subjects, 18-88 y | Free registration + data request via Cam-CAN portal | https://cam-can.mrc-cbu.cam.ac.uk/dataset/ |
| HCP Young Adult (S1200) | T1w + FreeSurfer outputs, sex, age (5-year bins open; exact age restricted), retest subset (~45) | ~1,100 subjects, 22-37 y | Free registration (open access terms); exact age and family structure under restricted-data DUA | https://www.humanconnectome.org/study/hcp-young-adult , https://db.humanconnectome.org |
| HCP-Aging | T1w + FreeSurfer, age, sex, cognition, BP, HbA1c | ~1,200 subjects, 36-100+ y | NDA data use certification | https://www.humanconnectome.org/study/hcp-lifespan-aging , https://nda.nih.gov/ccf |
| OASIS-3 | T1w + FreeSurfer, age, sex, CDR/MMSE, longitudinal sessions, same-day repeat scans, APOE | ~1,300 subjects, ~2,800 sessions, 42-95 y | Free registration + DUA on NITRC / XNAT Central | https://sites.wustl.edu/oasisbrains/ , https://www.nitrc.org/projects/oasis3/ |
| SALD (Southwest University Adult Lifespan Dataset) | T1w, age, sex | 494 subjects, 19-80 y | Open (INDI; registration) | http://fcon_1000.projects.nitrc.org/indi/retro/sald.html |
| DLBS (Dallas Lifespan Brain Study) | T1w, age, sex, cognition | ~300 subjects, 20-89 y | Open (INDI; registration) | http://fcon_1000.projects.nitrc.org/indi/retro/dlbs.html |
| UK Biobank (optional) | IDPs, age, sex | ~50,000 | Application + fee | https://www.ukbiobank.ac.uk |

Feature extraction: FreeSurfer 7 `recon-all` (or the cohort-supplied FreeSurfer outputs, with version recorded and treated as a batch factor), `aparcstats2table` / `asegstats2table` outputs, Euler number for QC (Rosen et al., 2018, NeuroImage). Optional: pyment (Leonardsen et al., 2022, NeuroImage) and DeepBrainNet (Bashyam et al., 2020, Brain) as fixed pretrained references.

## Methods

1. Feature tables (`sexstrat_brainage.freesurfer`): read `aparcstats2table`/`asegstats2table` wide tables; assemble 68 thicknesses, 68 areas, ~20 subcortical volumes, ventricles, eTIV; apply one of five head-size corrections (none = raw volumes, eTIV as covariate, proportion, residual on eTIV fitted on training data, power-proportion); Euler-number QC exclusion (threshold chosen per cohort at the 5th percentile, with a sensitivity analysis).
2. Strategies (`sexstrat_brainage.models`): `BrainAgeEstimator(strategy=...)` wrapping ridge regression / kernel ridge / gradient boosting; strategies `pooled`, `pooled_sex` (sex as a feature), `stratified` (one model per sex), `pooled_sexbias` (pooled model, age-bias correction fitted separately per sex). Age-bias correction options: Beheshti-style (regress delta on age in training folds and subtract), de Lange/Cole-style (regress predicted on true age and invert). All corrections fitted inside CV training folds.
3. Cross-validation: 10-fold `GroupKFold` by subject (and family ID for HCP-YA), 10 repeats; cohort held out entirely for transport analyses; harmonization by cohort × scanner with ComBat fitted on training folds when pooling cohorts.
4. Evaluation (`sexstrat_brainage.evaluation`): MAE by sex and sex-gap with cluster bootstrap; delta ~ sex + age regression (spurious sex effect); partial association of delta with outcomes controlling age, sex, education and cohort (mixed model with cohort random intercept via statsmodels `MixedLM`); ICC(2,1) for retest; transport MAE by sex.
5. Simulation (`sexstrat_brainage.simulation`): lifespan cohorts with configurable sex differences in head size, in ageing slope and in outcome coupling; ground-truth "biological age" allows measuring which strategy recovers the true delta-outcome coefficient and which produces a spurious sex difference.
6. Longitudinal arm: OASIS-3 subjects with ≥ 2 sessions ≥ 1 year apart; annualized hippocampal change and CDR progression modelled on baseline delta × sex × strategy.

Libraries: numpy, pandas, scipy, scikit-learn, statsmodels; optional nibabel, pingouin (ICC cross-check), neuroHarmonize, torch (CNN arm).

## Evaluation and statistics

- Primary outcome: sex gap in MAE and spurious sex effect on corrected delta, per strategy × TIV correction, with subject-level bootstrap CIs (2,000 resamples).
- Secondary: outcome associations (standardized β with CI), ICC, transport MAE. All reported per cohort and pooled with cohort random effects.
- Model comparison: strategies compared by paired bootstrap on the same folds; permutation test for the strategy × sex interaction (1,000 permutations of sex labels within age strata).
- Multiple comparisons: two pre-registered primary contrasts (pooled vs stratified on the two primary outcomes); everything else is secondary with Holm correction within families.
- Leakage: all scalers, TIV residualization, ComBat, bias-correction and feature selection fitted on training folds; retest sessions never split across folds; family IDs grouped.
- Nulls: age-permuted models to verify that delta-outcome associations vanish; simulated cohorts with no sex difference to calibrate the false-positive rate of the "sex difference in delta" test.
- Reporting: TRIPOD+AI checklist; pre-registration on OSF before the outcome analyses.

## Publishable angle

Headline: "Reported sex differences in brain-age delta are largely a modelling artefact: with head-size-corrected features and sex-aware training the sex difference disappears, sex-stratified training equalizes error but costs accuracy, reliability and transportability, and it does not improve the delta's association with cognition or vascular risk." The paper delivers a recommendation (pooled model + sex covariate + TIV residualization + within-sex bias correction) with a simulation-based explanation and a reusable evaluation harness.

Target venues: NeuroImage; Human Brain Mapping; Imaging Neuroscience; Biology of Sex Differences (if the sex-difference framing is foregrounded); FAIMI workshop (MICCAI) for an early version.

Follow-ups: extend to race/ethnicity-stratified training where data allow (fairness parallel); apply to CNN brain-age models (re-train pyment per sex); test whether the same logic applies to "organ age" clocks.

## Risks, confounds and mitigations

- Age-range imbalance between sexes within cohorts (e.g., more older women in OASIS-3): age-stratified resampling and inverse-probability weights; report age distributions per sex.
- FreeSurfer version and scanner differences: cohort × scanner ComBat inside folds; Euler QC; version recorded.
- Exact age unavailable in HCP-YA open data: use 5-year bins as an interval outcome or obtain restricted data; HCP-YA mostly serves the retest and family-structure analyses.
- Outcome heterogeneity (different cognitive tests per cohort): use within-cohort standardized scores and meta-analytic pooling; treat fluid-intelligence measures as exchangeable only within a sensitivity analysis.
- Halving training data in stratified models confounds "strategy" with "sample size": add a pooled-subsampled control (pooled model trained on a random half) to separate the two.
- Hormonal/menopause status is unmeasured in most cohorts: acknowledge; Cam-CAN and HCP-A have partial data for an exploratory analysis.

## Milestones

- [ ] Access: IXI, SALD, DLBS downloaded; Cam-CAN request approved; HCP-YA/HCP-A NDA/ConnectomeDB access; OASIS-3 DUA.
- [ ] FreeSurfer tables for all cohorts with Euler QC; harmonized master table with cohort, scanner, sex, age, TIV.
- [ ] Simulation study complete (figure: spurious sex effect vs strategy × TIV correction).
- [ ] Factorial CV results (primary outcomes) on all cohorts.
- [ ] Reliability (HCP-YA retest, OASIS-3 repeats) and transport analyses.
- [ ] Outcome associations and longitudinal OASIS-3 analysis.
- [ ] Pre-registration filed before outcome analyses; manuscript; code + simulated data release.

## Ethics / data-use notes

- HCP restricted data (exact age, family structure) require a separate DUA and must not be combined with open data in ways that could re-identify participants; never publish exact ages for HCP subjects.
- OASIS-3, HCP-Aging (NDA) and Cam-CAN data are under DUAs prohibiting redistribution; keep participant-level tables out of git; acknowledgements per each dataset's policy.
- Sex is analysed as recorded in each dataset (typically sex assigned at birth); gender identity is not available and results should not be over-interpreted as gender effects.
- Do not upload participant-level data to third-party LLM or cloud services.
