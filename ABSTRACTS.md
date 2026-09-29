# Project Abstracts

Submission-ready structured abstracts (~2,500 characters each) for the 105 projects judged publishable. Each links to its project folder; the 8 projects rated at real risk (data availability, power or scope) have no abstract yet: `cuffless-bp-pregnancy`, `deid-residual-leakage-audit`, `digital-health-rct-ipd-reanalysis`, `effect-size-inflation-openneuro`, `meg-microstate-heritability`, `spike-sorter-multiverse`, `unified-biomedical-shift-benchmark`, `ventilator-asynchrony-detection`.

Abstracts for projects marked NARROWED in [litcheck/VERDICTS.md](litcheck/VERDICTS.md) use the sharpened angle and do not claim the part already published.


## Neuroimaging: structural MRI, aging & dementia

### Where the Correction Was Calibrated Changes the Answer: A Transportability Audit of Brain-Age Delta, Bias Correction and Harmonisation Against Longitudinal Decline in OASIS-3

*Project: [`brain-age-transportability`](projects/brain-age-transportability/)*

**Background.** Brain-age delta is used as a biomarker of accelerated ageing, but imperfect regressors produce an age-dependent bias, and delta absorbs scanner effects. Post-hoc corrections and ComBat harmonisation are usually calibrated within one cohort, often including the test data, and evaluated cross-sectionally. No study has crossed calibration cohort, harmonisation design and correction formula and scored each combination against within-person clinical change, although corrected change in delta depends on the reference slope, so the same person can appear to age faster or slower depending on where it was calibrated.

**Objective.** We will quantify how bias-correction calibration, harmonisation design and training age range change estimated within-person brain-ageing acceleration and its association with CDR and MMSE progression when models trained on HCP Young Adult, HCP-Aging and HCP-Development are applied to OASIS-3.

**Methods.** Identical FreeSurfer feature vectors will be built for HCP-YA (1,113 subjects), HCP-A (about 1,200), HCP-D (about 1,300) and OASIS-3 (about 2,800 sessions, 1,378 participants aged 42-95 on four Siemens scanners). Ridge and gradient-boosting models will be trained under four source designs with subject-grouped 5-fold CV; Beheshti, Cole and Smith-quadratic corrections will be calibrated on held-out reference predictions, OASIS-3 CDR 0 baseline scans, or the pooled sample; harmonisation is none, pooled ComBat, controls-only ComBat, or mean-only ComBat. Each of the 240 pipelines yields the target bias slope, ICC(2,1) for scan pairs under 180 days by scanner, and a mixed model of delta on years by converter (CDR 0 at baseline, at least 0.5 at two later visits) with participant random effects, plus the delta-slope versus MMSE-slope correlation, with participant-level bootstraps. Inference treats the grid as a multiverse with a specification curve and an age-stratified converter-label permutation null; clinical labels never enter training.

**Expected results.** We expect reference slopes to differ between HCP-Lifespan and OASIS-3 controls, per-person delta slopes to change sign for over 10% of participants across pipelines, the converter association to be largest with age- and scanner-matched calibration, pooled ComBat to attenuate it most in the 1.5T batch, harmonisation to restore ICC without restoring validity, and models trained below age 37 not to transport.

**Significance.** The audit yields a recommended transport protocol and published reference slopes so brain-age claims can travel between scanners and cohorts.

### Does Structural MRI Add to Age, APOE and MMSE as a Pre-Screen for Amyloid-PET Positivity? An Externally Validated, Decision-Analytic Audit

*Project: [`amyloid-from-mri`](projects/amyloid-from-mri/)*

**Background.** Anti-amyloid trials must confirm amyloid pathology before enrolment, and PET screen failure is costly. Deep-learning models that read amyloid status from T1 MRI report external AUCs near 0.65, and ROI models reach 0.70-0.83 only in mixed-diagnosis samples where amyloid status is confounded with age, diagnosis and atrophy. No study has tested MRI models against the zero-cost covariate model with paired statistics, checked whether the signal survives age-matching in cognitively normal adults, or translated AUC into PET scans avoided.

**Objective.** We will audit ROI-based and 3D-CNN amyloid-from-T1 models against age, sex, APOE e4 count, MMSE and education, with an age/atrophy-proxy test in cognitively normal participants, external validation, and a decision-curve and cost model benchmarked to published plasma p-tau217 operating points.

**Methods.** On OASIS-3 (about 1,300 participants, 1,500 amyloid PET sessions with Centiloid labels; PiB 16.4 CL, AV45 20.6 CL), MR sessions will be matched to PET within 365 days, one session per subject. FreeSurfer thickness and ICV-normalised volumes will feed elastic-net and gradient-boosting models, alone and with covariates, in nested subject-grouped 5x5 repeated cross-validation with ComBat, scaling and Platt calibration fitted inside training folds; an optional MONAI DenseNet on 1.5 mm T1 uses subject-level splits and a 1,000-fold label-permutation null. The proxy test uses 1:1 age-matched A+ and A- cognitively normal pairs, residualisation on age, sex and ICV in A- controls, and restriction to hippocampal z above -1. Frozen models will be validated on ADNI (over 2,000 with amyloid PET) and A4/LEARN pre-randomisation MRI (all cognitively normal), reporting calibration slope, intercept recalibration, paired bootstrap delta-AUC and delta-Brier, net benefit at thresholds 0.10-0.60, PET scans avoided per 1,000 at 90% sensitivity, and cost per enrolled A+ participant. A leakage ablation compares session-level with subject-level folds and ComBat fitted inside versus outside the loop.

**Expected results.** We expect delta-AUC of 0.05 or less over covariates in CDR 0, AUC falling to 0.55-0.62 after residualisation, rank order but not calibration transporting externally, fewer than 30% of PET scans avoided at 90% sensitivity versus about 60% for p-tau217, and leaky session-level CV inflating AUC by at least 0.03.

**Significance.** Either outcome gives trialists a decision-relevant answer, reported to TRIPOD+AI, on whether already-acquired MRI can enrich PET screening.

### How Much of the Structural MRI Tau Signal Is Just Neurodegeneration? A Covariate-Anchored, Discordance-Tested and Decision-Analytic Audit of PET-Free Tau Staging in OASIS-3 and ADNI

*Project: [`tau-proxy-from-mri`](projects/tau-proxy-from-mri/)*

**Background.** Tau PET defines the T2 stage under the 2024 Alzheimer's criteria and informs anti-amyloid treatment, yet it is costly and scarce. MRI signatures predicting tau positivity and staging have been shown in symptomatic ADNI participants, and models adding amyloid PET reach high discrimination. Unestablished are whether an MRI-only model adds anything over intake covariates, whether its signal is separable from generic neurodegeneration, whether it recovers T2 stage in a low-prevalence screening population, and what it is worth for triage.

**Objective.** We will quantify the increment of T1 plus FLAIR features over intake covariates for flortaucipir positivity and T2 stage, isolate the tau-specific component in tau/atrophy-discordant participants, test OASIS-3 to ADNI transport, and translate discrimination into scans avoided.

**Methods.** In the OASIS-3_AV1451 subset (451 flortaucipir sessions, predominantly cognitively normal), each PET will be matched to the nearest MR session and clinical visit. Labels use meta-temporal and neocortical SUVR composites at a cut-off of 1.25 (sensitivity 1.20-1.30), and a three-level stage (none, medial temporal, neocortical). Features comprise 68 cortical thicknesses, subcortical volumes, asymmetry indices and FLAIR white-matter hyperintensity (WMH) volume. Covariate logistic regression (age, sex, APOE e4, education, MMSE, CDR-SB), penalised and boosted ROI models, a neurodegeneration-only ceiling and ordinal models are compared in nested subject-grouped cross-validation with ComBat fitted inside folds, paired bootstrap change in AUC and label-permutation nulls. The discordance test reports AUC within N-negative participants and false-positive rates in T-negative N-positive versus N-negative groups, with and without WMH. Frozen models will be validated in ADNI (over 1,500 tau PET participants) and vice versa, and decision curves will give scans avoided per 1,000 at 90% sensitivity against published p-tau217 operating points.

**Expected results.** We expect a change in AUC of 0.05-0.12 over covariates; AUC of 0.60-0.68 within N-negative participants with the interval excluding 0.5; doubled false positives in atrophy-without-tau participants, reduced by WMH; macro AUC of at least 0.75 for stage; transported discrimination within 0.05 AUC after intercept recalibration; and 25-40% of scans avoided in mixed samples but under 25% in cognitively normal adults.

**Significance.** A tau-specific residual would support MRI triage where plasma assays are unavailable; its absence would correct MRI-derived tau claims.

### Do Cross-Sectional Atrophy Stages Progress Within People? A Null-Anchored Longitudinal Audit of Subtype-and-Stage Models in Amyloid-Positive and Amyloid-Negative Cognitive Impairment in OASIS-3

*Project: [`atrophy-subtype-trajectories`](projects/atrophy-subtype-trajectories/)*

**Background.** Subtype-and-stage inference orders regional atrophy from cross-sectional MRI and treats the ordering as temporal. Recent multi-cohort work shows subtypes replicate, stages increase at follow-up in amyloid-defined Alzheimer's disease samples, and stage carries cross-sectional prognostic value. Two gaps remain: reported monotonicity has no within-subject null, with many ties at early stages, most session pairs can be non-decreasing by chance. And amyloid-negative cognitively impaired individuals, a heterogeneous non-Alzheimer group, are excluded from or forced into Alzheimer sequences; no dedicated model has been longitudinally validated for them.

**Objective.** We will audit stage monotonicity, subtype stability and within-subject coupling of stage change to CDR-SB change against permutation nulls in OASIS-3, and fit a dedicated subtype-and-stage model for amyloid-negative cognitive impairment.

**Methods.** From OASIS-3 (about 1,300 participants, 2,800 MR sessions, roughly 60% with two or more sessions, 1,500 amyloid PET) we will build session timelines with FreeSurfer regional volumes and thickness, CDR-SB within 180 days, and amyloid status by a once-positive Centiloid rule. Features will be z-scored against amyloid-negative CDR 0 controls adjusted for age, sex, eTIV and scanner. Models will be fitted on baseline sessions only with pySuStaIn, selecting K of 1 to 5 by 10-fold cross-validated log-likelihood; follow-up sessions will be staged with the frozen model and never enter fitting. Monotonicity will be compared with 1,000 within-subject session-order permutations; subtype stability will be reported by baseline stage and posterior confidence against the marginal-frequency null; mixed models of CDR-SB on stage with subject random intercepts will be compared with hippocampal z by AIC and likelihood-ratio tests. Simulations from known sequences calibrate expected monotonicity; same-scanner pairs form the primary analysis, and ADNI serves as a held-out replication cohort.

**Expected results.** In amyloid-positive impairment we expect at least 80% non-decreasing pairs, exceeding the null by 20 points, with within-subject stage change explaining CDR-SB change better than hippocampal change. In amyloid-negative impairment we expect a limbic-first subtype with above-null monotonicity and otherwise largely non-progressive stages, and subtype stability below 60% at stage 2 or less.

**Significance.** The audit supplies the null models the field lacks and tests whether MRI staging is meaningful for the non-Alzheimer patients it ignores.

### How Far Do Brain Centiles Travel? A Transportability Audit of Lifespan Normative Models for Cortical Thickness, Subcortical Volume and Tract Diffusion Metrics

*Project: [`lifespan-normative-models`](projects/lifespan-normative-models/)*

**Background.** Lifespan normative charts exist for morphometry and, since 2024-2026, for tract FA and MD, promising that any individual can be scored against an open reference. Whether it holds depends on site and protocol effects, the reference model and how many local controls are needed. No study has reported, for the same individuals, how much centiles and extreme-deviation flags disagree between reference models, how held-out-site centiles behave under realistic adaptation budgets for diffusion, whether diffusion deviations add clinical detection in aging, or centile test-retest reliability.

**Objective.** We will fit quantile-regression and heteroscedastic normative models on the protocol-harmonised HCP Lifespan backbone plus open sites and audit centile transportability across sites, reference models, adaptation budgets, clinical labels and repeat scans.

**Methods.** Reference data are healthy participants from HCP-Development (~1,300, 5-21 y), HCP Young Adult (~1,100), HCP-Aging (~1,200, 36-100+ y), OASIS-3 (CDR 0, amyloid-negative), UCLA CNP (130 controls), QTIM (1,202), MPI-LEMON (~228) and AOMIC (~928). Features are 68 regional thicknesses, subcortical volumes and tract-mean FA and MD from one b = 1000 pipeline. Models use an age spline with sex and site effects at 11 quantiles and shift-scale site adaptation. Leave-one-site-out scoring with 0-100 adaptation controls reports the KS statistic against uniformity (bootstrap CIs) and extreme rates at |z| > 1.96 (Wilson CIs). Agreement between BrainChart, PCNtoolkit and local centiles uses Spearman rho and Cohen's kappa on extreme flags. Clinical value is the AUC (DeLong CI) of deviation counts for CDR >= 0.5 versus 0 and amyloid status in OASIS-3, with bootstrap delta-AUC over morphometry and a label-permutation null. Reliability uses ICCs from the HCP-YA 45-subject retest and QTIM second session. Clinical labels never enter fitting.

**Expected results.** We expect thickness centiles near-uniform on held-out HCP sites (KS < 0.10) but not on OASIS-3 or OpenNeuro sites (KS > 0.15), with FA and MD extreme rates above 15% even between HCP cohorts; 25 controls sufficing for thickness but >= 50 for diffusion; rho > 0.9 across references yet kappa < 0.6 for extreme flags; MD counts adding delta-AUC >= 0.03 for CDR >= 0.5 that shrinks when OASIS-3 leaves the reference; and centile ICC >= 0.8 with a standard error of >= 8 centile points.

**Significance.** The audit would tell clinicians and trialists how model- and site-dependent an individual centile is, with released curves and adaptation tools.

### Which diffusion model deserves a lifespan chart? Centile curves for DKI, NODDI and DTI on the same multi-shell brains, graded by age sensitivity, deviation reliability and protocol transportability

*Project: [`dmri-microstructure-charts`](projects/dmri-microstructure-charts/)*

**Background.** Diffusion MRI gained lifespan normative charts in 2025-2026, but only for diffusion tensor metrics, with FA peaking near 29 years. Multi-shell acquisitions support diffusion kurtosis imaging and NODDI, but no centile charts exist for them, no chart has been built with several models on identical subjects, the reliability of an individual's deviation score has never been measured, and identifiability critiques imply that multi-compartment parameters, and hence z-scores, may move with acquisition protocol.

**Objective.** We will fit lifespan centile charts for DKI (MK, AK, RK), NODDI (ICVF, ODI, ISOVF) and DTI metrics on the same subjects and grade each model on what a chart is for: age sensitivity, test-retest reliability of z-scores, transportability across protocols and sensitivity to clinical deviation.

**Methods.** Reference sample: HCP Young Adult (about 1,065 with dMRI, b = 1000/2000/3000, 45-subject retest), HCP Aging and Development (about 1,200 and 650, b = 1500/3000), Cam-CAN (about 650, b = 1000/2000) and IXI (about 550, single shell), ages 5-100; OpenNeuro multi-shell datasets with patient groups serve only for deviation validation. DTI and DKI by weighted linear least squares, NODDI by AMICO. Tract means on the ENIGMA-DTI skeleton and TractSeg bundles. A location-scale B-spline model with sex and site terms yields centiles, z-scores and bootstrap ages of peak; 10-fold cross-validation checks that held-out z-scores have mean 0, SD 1 and no age trend. Criteria: R2 explained by age (age-permuted null); ICC(2,1) of z-scores in the HCP retest; paired delta-z and Cohen's d between full HCP data and shell/direction subsets emulating Cam-CAN, UK Biobank and HCP-Aging schemes; deviation AUC in clinical cohorts; calibration of single-shell DTI z-scores to NODDI/DKI z-scores. FDR across tracts within metric, with pre-registered primary tracts.

**Expected results.** We expect ICVF and MK to exceed FA in age-explained variance in most tracts and ISOVF to show the strongest late-life acceleration; z-score ICC of about 0.8-0.9 for DTI, 0.7-0.85 for ICVF and 0.6-0.8 for ODI and MK; DTI from b = 1000 to be protocol-stable (d below 0.2) while ICVF shifts by d of 0.3-0.6 when the high shell changes; ICVF to peak a decade later than FA and ODI to reach its minimum earlier; and DTI z-scores to predict ICVF (R2 about 0.5-0.7) but not ODI (R2 below 0.3).

**Significance.** Public DKI and NODDI chart files, a transportability atlas and a model leaderboard tell users which microstructural deviations can be trusted and pooled across protocols.

### Scanner Upgrades as Natural Experiments: Regression Discontinuity in Time for Counterfactual Atrophy Trajectories and a Clinically Anchored Harmonization Benchmark in OASIS-3 and ADNI

*Project: [`scanner-upgrade-discontinuity`](projects/scanner-upgrade-discontinuity/)*

**Background.** Longitudinal MRI cohorts outlive their scanners. OASIS-3 spans four Siemens platforms and ADNI moved from 1.5T to 3T with most sites changing hardware. Traveling-subject studies show upgrade-induced shifts of several percent in subcortical volumes, worth years of normal atrophy, yet harmonization methods are validated on batch detectability or paired scans rather than on whether the atrophy rates they produce remain unbiased and clinically predictive.

**Objective.** We will estimate the level shift and slope change at each scanner transition inside the cohorts themselves with a regression-discontinuity-in-time design, produce counterfactual no-upgrade trajectories, and benchmark harmonization methods by whether harmonized hippocampal atrophy rates still predict CDR progression.

**Methods.** From OASIS-3 (about 1,378 participants, 2,842 MR sessions, FreeSurfer 5.3) and ADNI (about 2,000 participants, UCSF FreeSurfer tables) we will build transition tables with two sessions per scanner side, using a subject-anchored event-time design in OASIS-3 and calendar-time site cutoffs in ADNI. The estimator is local-linear regression on both sides of the cutoff with a triangular kernel, subject fixed effects, cross-validated bandwidth, donut exclusions, placebo cutoffs, covariate-continuity and density tests, and cluster-robust and bootstrap intervals. The 69 OASIS-3 Trio-to-mMR traveling subjects and ADNI's dual-field-strength subset give paired ground truth. Harmonization arms are scanner covariate, cross-sectional ComBat, longitudinal ComBat, ComBat-GAM and an RD-anchored correction, fitted on cognitively normal reference subjects or inside cross-validation folds, with the outcome never entering harmonization. Arms are graded by residual discontinuity, slope bias against matched single-scanner controls, preserved age and CDR-by-time effects, and AUC, C-index and calibration for CDR progression. Simulated cohorts calibrate bias and coverage; hippocampal volume is the primary endpoint with Holm correction within feature groups.

**Expected results.** We expect hippocampal shifts of 1 to 3%, equivalent to 1 to 3 years of aging, with level shifts dominating for volumes; cohort estimates agreeing with traveling-subject offsets for volumes; longitudinal ComBat and the RD-anchored correction preserving prediction while cross-sectional ComBat shrinks true atrophy in subjects with unbalanced follow-up.

**Significance.** The released offsets, trajectories and package let any cohort with a scanner change grade harmonization by clinical validity.

### Do Sex-Stratified Brain-Age Models Make the Delta Fairer, More Reliable or More Informative? A Pre-Registered Factorial Audit Across Five Lifespan Cohorts with Ground-Truth Simulation

*Project: [`sex-stratified-brain-age`](projects/sex-stratified-brain-age/)*

**Background.** Brain-age models handle sex inconsistently: many ignore it, some add it as a covariate, and consortium pipelines train separate models per sex, after which sex differences in delta are reported as biology. Head size differs by about 10% between sexes, and a recent 14-cohort longitudinal analysis found sex differences in healthy structural ageing to be small. Fairness audits have documented sex gaps in error, but no study has compared strategies on the same features, folds and cohorts or asked whether a strategy manufactures a sex difference in delta.

**Objective.** We will test whether pooled, pooled-with-sex-covariate, sex-stratified and pooled-with-sex-specific-bias-correction strategies, crossed with five head-size corrections, differ in sex gap in error, spurious sex effect on delta, test-retest reliability, cross-cohort transport and association with cognition, CDR and vascular risk.

**Methods.** FreeSurfer tables with Euler-number quality control will be assembled from IXI (about 580), Cam-CAN (about 650), HCP Young Adult (about 1,100, with retest), HCP-Aging (about 1,200), OASIS-3 (about 1,300 subjects, 2,800 sessions), SALD (494) and DLBS (about 300). Head-size corrections are fitted on training folds. Ridge, kernel ridge and gradient-boosting estimators use ten-fold subject- and family-grouped cross-validation over ten repeats, with age-bias correction and ComBat fitted within folds and whole cohorts held out for transport. Outcomes are MAE by sex with cluster bootstrap, delta regressed on sex and age, partial associations with outcomes in cohort random-intercept mixed models, ICC on repeat scans, and transport MAE by sex; a pooled model trained on a random half separates strategy from sample size. A simulation with configurable sex differences in head size and ageing slope identifies which strategy recovers the true delta-outcome coefficient without a spurious sex effect. Nulls are age-permuted models and null-simulated cohorts; two primary contrasts are pre-registered.

**Expected results.** We expect pooled models without sex to show the largest error gap (0.3 to 0.8 years) and, with uncorrected volumes, a spurious positive delta shift of 0.5 to 1.5 years in women reproduced by the head-size-only simulation; stratified models to equalise error at the cost of 0.2 to 0.5 years MAE, lower ICC and worse transport; and no strategy to improve delta-outcome associations by more than 10%.

**Significance.** The audit and simulation show when reported sex differences in brain age are artefacts and give a defensible default strategy.

### Do Growing White-Matter Hyperintensities Amplify Amyloid's Effect on Cognitive Decline? A Time-Varying, Additive-Scale and Segmentation-Robust Test in OASIS-3 with ADNI Replication

*Project: [`wmh-amyloid-interaction`](projects/wmh-amyloid-interaction/)*

**Background.** White-matter hyperintensities (WMH) co-occur with amyloid in most older adults, and whether they act additively or synergistically determines whether vascular prevention alters the course of amyloid-positive individuals. Existing evidence relies on baseline WMH product terms in cohorts of a few hundred. No study has modelled the concurrent WMH trajectory as a time-varying modifier of amyloid's effect, reported additive-scale interaction with intervals, or shown the estimate survives the segmentation tool and scanner generation. OASIS-3 holds FLAIR across roughly 2,842 MR sessions with Centiloids and annual CDR over up to 15 years but no WMH table.

**Objective.** We will test whether WMH level and growth modify amyloid's effect on cognitive slopes and CDR progression in OASIS-3, quantify it on multiplicative and additive scales, localise it by compartment, and replicate in ADNI.

**Methods.** FLAIR sessions from about 1,378 OASIS-3 participants will be matched to amyloid PET within 365 days (PiB 16.4 and AV45 20.6 Centiloid thresholds) and clinical visits within 180 days. WMH will be segmented with LST-LPA, BIANCA, SAMSEG-lesion and a deep-learning tool calibrated on the 60 MICCAI challenge masks, and split into periventricular and deep compartments. Linear mixed models with random slopes will estimate the time by amyloid by WMH(t) term for a global composite, episodic memory and CDR-SB, adjusting for demographics, APOE e4 and scanner generation; person-period hazard models for CDR progression will yield RERI, attributable proportion and synergy index with subject bootstrap intervals; joint longitudinal-survival models will link WMH trajectories to hazard, and counterfactual mediation will test amyloid to WMH growth to decline. A pre-registered primary specification sits within a specification curve over tools, compartment rules, amyloid coding and outcomes. Nulls include amyloid permutation within age-sex strata. ADNI (about 1,500 with UC Davis WMH volumes and Centiloids) is used only after the OASIS-3 analysis is frozen.

**Expected results.** We expect a negative three-way coefficient concentrated in episodic memory; RERI of 0.5-1.5 with attributable proportion of 20-40%; WMH growth as the stronger modifier, with baseline WMH understating it by at least 30%; interaction in periventricular and posterior WMH; mediation under 10%; and coefficients within each other's intervals across tools and eras.

**Significance.** The results would establish whether vascular burden multiplies amyloid's harm on the scale that matters for prevention.

### Quality-control exclusion as a hidden multiverse: linking crowdsourced MRIQC metrics to OpenNeuro metadata for age-conditional norms and repository-wide sensitivity of group effects

*Project: [`openneuro-mriqc-audit`](projects/openneuro-mriqc-audit/)*

**Background.** MRIQC image quality metrics (IQMs) are the standard for MRI quality control and are uploaded by default to a public WebAPI holding hundreds of thousands of records. Threshold-based exclusion removes children and patients disproportionately, but this has only been shown within single cohorts. Published reference IQM values do not condition on age, and WebAPI records have never been linked back to the datasets and participants they came from.

**Objective.** We will join WebAPI IQMs to OpenNeuro participant metadata, build age-conditional normative IQM charts by scanner, quantify differential exclusion of clinical and pediatric participants, and measure how much group effect sizes move under the QC rules studies actually use.

**Methods.** WebAPI T1w and BOLD records (more than 100,000, de-duplicated by MD5 and version) will be linked to OpenNeuro (more than 1,000 datasets, more than 40,000 participants) by matching provenance MD5 sums to git-annex MD5E keys, then to participant age, sex and group via a versioned two-rater vocabulary. Quantile-regression normative models (B-spline in age; manufacturer, field strength and MRIQC version) will give 5th to 95th centiles, validated by cross-validated coverage. A multiverse engine will cross motion rules (mean FD above 0.2, 0.25 and 0.5 mm), T1w rules (CJV above 0.45; per-dataset worst 10 percent) and combined and centile rules, reporting per cell the retained n, patient-versus-control exclusion risk ratio and Hedges' g for within-dataset group contrasts on IQM-independent outcomes. Risk ratios are pooled by random-effects meta-analysis with Benjamini-Hochberg correction; the g range is tested against a permutation null that shuffles rule assignment within dataset, separating fewer subjects from different subjects. IQMs recomputed locally on 50 datasets bound version heterogeneity.

**Expected results.** We expect at least 30 percent of OpenNeuro images to have a recoverable WebAPI record; IQM centiles strongly non-monotonic in age, so a fixed FD threshold excludes over 25 percent of children but under 5 percent of young adults; exclusion risk ratios above 1.3 in most clinical datasets, largest for neurodevelopmental groups; a between-rule g range above 0.2 in over half of eligible datasets; and relative rules that reduce differential exclusion without shrinking that range.

**Significance.** The audit makes quality thresholds age- and scanner-relative, releases a repository-scale IQM reference chart, and shows how much of the literature rests on an unreported analytic choice.

### Who Does Automated MRI Quality Control Exclude? A Corpus-Wide Audit of Exclusion by Age, Sex and Clinical Group Across OpenNeuro

*Project: [`mriqc-scannability-equity`](projects/mriqc-scannability-equity/)*

**Background.** Automated QC with MRIQC metrics and motion thresholds is routine, yet image quality is not randomly distributed: children, older adults and clinical populations move more, and motion biases morphometry. Single-cohort analyses in ABCD and UK Biobank show exclusions structured by race, income, age and clinical status, but each uses one rule and mostly fMRI motion, and none asks whether structural metrics such as CNR and CJV track normal biology rather than artefact.

**Objective.** We will estimate each participant's probability of surviving standard QC as a function of age, sex and diagnostic group with dataset as a random effect across hundreds of OpenNeuro datasets, quantify how much of the age structure reflects metric–biology coupling, measure exclusion elasticity to threshold choice, and report the representational shift QC imposes on the corpus.

**Methods.** From more than 1,000 OpenNeuro datasets we will harmonise participants.tsv age, sex and group fields into coarse classes with a reviewed keyword dictionary, targeting roughly 10,000–20,000 T1w and 8,000 BOLD subjects with demographics. Exclusion will be flagged under the MRIQC classifier, literature thresholds (mean FD 0.2–0.5 mm, tSNR, CJV, CNR, EFC, SNR) and a dataset-relative worst-10% rule. Binomial GLMs with an age spline, sex, group, modality and dataset effects and cluster-robust standard errors give odds ratios; a null permuting participant rows within dataset breaks participant–metric links while preserving design. Within QC-passed controls we will residualise metrics on age within dataset and re-derive exclusions. Elasticity is the slope of group exclusion fraction over a threshold sweep; representation shift uses age quantiles and KL divergence between acquired and QC-passed distributions. Three primary hypotheses are pre-registered; robustness is checked per MRIQC version, field strength and year.

**Expected results.** We expect a U-shaped age curve with exclusion odds at least twofold higher at ages 8 and 80 than at 25, within-dataset odds ratios of 1.3–2.0 for neurological and 1.2–1.5 for psychiatric groups, modestly higher male odds, at least 30% of the structural age effect removed by biology adjustment, threshold-driven swings above 20 percentage points for older and neurological groups, and a QC-passed corpus with 10–25% fewer participants under 10 and over 65.

**Significance.** A scannability calculator, biology-adjusted thresholds and a citable representation report would make QC-driven selection a quantified and correctable step.

### Do Brain-MRI Foundation Models Survive Clinical Data? A Quality- and Pathology-Stratified External Validation

*Project: [`mri-foundation-model-clinical-validation`](projects/mri-foundation-model-clinical-validation/)*

**Background.** Brain-MRI foundation models such as BrainSegFounder and SAM-Med3D promise segmentation without task-specific training, but their reported gains come from curated, quality-filtered challenge sets, and the one community evaluation on clinical data keeps its test set private. Artefact robustness has been studied for classical pipelines, not for segmentation foundation models, and none has seen paired scans with and without deliberate motion.

**Objective.** We will build an open, reproducible benchmark that reports foundation-model performance as a function of measured image quality, lesion load, age and prompt regime against task-specific nnU-Net and contrast-agnostic SynthSeg, and locate the label budget at which a from-scratch model catches up.

**Methods.** We will harmonise MR-ART (148 adults, three motion levels each), MindBoggle-101, CANDI (103 children), Hammers (30 adults), ATLAS v2.0 (655 public stroke masks), ISLES 2022, Shifts 2.0 MS lesions, BraTS 2023 adult glioma and BraTS-Africa, OASIS-3, OASIS-4 and M4Raw 0.3 T into BIDS, recording MRIQC metrics (CJV, CNR, EFC, SNR) for every input. Released weights will be evaluated frozen, few-shot fine-tuned at 5–100 labelled cases, and zero-shot with oracle versus automatic prompts, against nnU-Net from scratch at the same budgets and SynthSeg and FastSurfer as no-training references, on fixed subject-level splits. Controlled noise, k-space motion, bias field and ghosting give dose-response curves validated against real MR-ART motion. The primary endpoint is the subject-level paired Dice difference from nnU-Net at 50 cases within quality tertiles, by Wilcoxon signed-rank with Holm correction and 2,000 bootstraps; a mixed model with subject and dataset intercepts estimates quality slopes; TOST margins are 0.02 Dice for anatomy and 0.05 for lesions. Pretraining overlap will be ledgered and analysed separately; label-shuffled curves and an untrained-encoder probe give the floor.

**Expected results.** We expect frozen foundation models to lose at least 0.10 Dice per SD of worsening CJV against at most 0.05 for SynthSeg, Dice drops of at least 0.15 next to stroke lesions and 0.10 on BraTS-Africa relative to adult BraTS, nnU-Net parity at roughly 25–50 lesion and 10–25 anatomy cases, deficits of 0.05–0.10 outside the pretraining age range, and at least 0.10 Dice lost without oracle prompts.

**Significance.** Either outcome is decision-relevant for hospitals, and the released harness, predictions and quality-tertile leaderboard let any future model be added and stratified.


## Neuroimaging: fMRI, connectomics & imaging transcriptomics

### Why connectome-based cognition prediction fails for under-represented groups: a causal decomposition and group-robust training audit in adults

*Project: [`connectome-prediction-fairness`](projects/connectome-prediction-fairness/)*

**Background.** Connectome-based prediction models are trained on demographically skewed samples and predict cognition less accurately for African American than for White participants in HCP and ABCD. Recent ABCD benchmarks show that balanced subsampling and weighting can remove the accuracy gap in children. What remains unknown is why the gap exists: no study partitions it into sample size, head motion, label reliability and genuine shift in the brain-behaviour mapping; gaps are reported as correlation differences rather than error structure; and whether fixes work in adults and transport across datasets is untested.

**Objective.** To measure subgroup gaps in error structure for adult connectome-based cognition prediction, decompose them causally, and test whether group-aware training closes them without harming anyone.

**Methods.** In HCP S1200 (1,003 subjects), Schaefer-400 Fisher-z connectivity (79,800 edges) will predict CogTotalComp_Unadj with ridge and CPM under family-aware, subgroup-stratified 10 x 5-fold cross-validation, with age, sex and motion residualised inside folds and race never residualised. Per-group Pearson r, RMSE, mean residual, calibration slope and residual variance will be compared for race, ethnicity, sex, SES and race x sex cells, with family-cluster bootstrap CIs and a 5,000-permutation family-level label null; shuffled race labels serve as a negative control. The decomposition will retrain the majority model at the minority's n (50 draws), estimate motion mediation of absolute error, evaluate on motion-matched subsamples and compute attenuation ceilings from subgroup-specific test-retest ICCs, averaging over all component orderings. ERM, inverse-frequency weighting, balanced subsampling and group-DRO ridge will be compared on worst-group gap and overall r, then transported to AOMIC and HBN or NKI. The primary endpoint is Holm-corrected across five schemes.

**Expected results.** We anticipate an r gap of at least 0.10 for Black/African American, Hispanic/Latino and low-SES participants; systematic over-prediction toward the majority mean; motion mediating 10-40% and matched-n reproducing under half of the gap; and balanced weighting and group-DRO halving the worst-group gap at no more than 0.02 loss in overall r while leaving mis-calibration unless groups are calibrated separately.

**Significance.** Each outcome implies a different remedy, in acquisition, recruitment, measurement or training, and the released fairness-card template gives brain-wide association studies a reporting standard.

### A reliability-corrected twin atlas of cortical functional organisation: shared and distinct genetic influences on gradients, structure-function coupling and dynamic states

*Project: [`connectome-heritability-gradients`](projects/connectome-heritability-gradients/)*

**Background.** Twin studies in the Human Connectome Project have shown that functional connectivity edges, gradient loadings, regional structure-function coupling and dynamic-state occupancy are each heritable, and gradient and gradient-coupling heritability have recently been mapped. These estimates come from different subsets, parcellations and estimators, so heritability maps have never been compared region by region under one pipeline; bivariate genetic correlations between feature families are unreported; heritability has not been corrected for the test-retest reliability ceiling; and transcriptomic comparisons of the heritability topography have not used both spatial and random-gene-set nulls.

**Objective.** To estimate, in the same HCP twins under one reliability-corrected twin model, the heritability topographies of gradient loadings, eccentricity, SC-FC coupling and dynamic-state occupancy, their shared genetic variance, pipeline stability and transcriptomic correlates.

**Methods.** From HCP S1200 resting-state and diffusion data in a Schaefer-400 parcellation, we will compute per-subject Procrustes-aligned diffusion-map gradients, eccentricity, SC-FC coupling from SIFT2-weighted tractography, and k-means dynamic-state fractional occupancy. Twin pairs (approximately 130-150 MZ and 70-90 DZ) will be analysed with Falconer screens and maximum-likelihood ACE with nested AE/CE/E likelihood-ratio tests, covariates residualised, and 1,000 pair-resampled bootstrap CIs. ICC(2,1) from the 45-subject retest set will bound and disattenuate h2. Bivariate Cholesky ACE will confirm genetic correlations passing FDR. A pre-registered grid of 72 pipeline variants will test map stability. Heritability maps will be compared with Allen Human Brain Atlas expression PC1 and cell-type marker sets using 10,000 spin permutations and 10,000 random gene sets. ABCD twins (about 400-450 pairs) will replicate the atlas.

**Expected results.** We expect gradient and coupling heritability maps to correlate only moderately, reliability correction to move at least 20% of regions across tertiles, weak genetic correlations except in heteromodal association cortex, heritable state occupancy largely independent of static gradients, gradient heritability to track expression PC1 beyond both nulls, and corrected maps to be stable across pipelines.

**Significance.** A single-model, reliability-corrected atlas would establish whether the heritable architecture of cortical organisation is shared or distinct across feature families, released as parcel-level tables only.

### Pipeline-robustness scores for published task-fMRI findings: a dataset-scale multiverse across OpenNeuro with a fragility model

*Project: [`fmri-pipeline-multiverse`](projects/fmri-pipeline-multiverse/)*

**Background.** NARPS showed that seventy analysis teams disagree on one dataset, and single-dataset multiverses, resting-state confound benchmarks and multiverse inference methods have since matured. What remains unmeasured is whether the same multiverse, run on many task-fMRI datasets, separates robust findings from fragile ones, and whether fragility is predictable from dataset characteristics. OpenNeuro now ships fMRIPrep derivatives for many task datasets, making a dataset-scale multiverse feasible without re-preprocessing.

**Objective.** We will summarise each published task-fMRI finding as a specification curve with a pipeline-robustness score (PRS), rank the analytic choices driving between-specification variance, quantify fMRIPrep version drift, and model which dataset features make a finding fragile.

**Methods.** We will run a pre-registered 360-specification grid (smoothing 0/4/8 mm; six confound strategies including global signal regression, aCompCor and FD scrubbing; three HRF models; two high-pass cutoffs; Schaefer-200/400 parcellations; OLS or AR(1)) on parcel time series from at least 12 findings in at least 5 datasets: NARPS ds001734 (108 subjects), UCLA CNP ds000030 (272), AOMIC PIOP1 ds002785 (216) and PIOP2 ds002790 (226). Regions of interest are fixed from the source papers. Per finding we will compute the PRS, sign consistency and an eta-squared variance decomposition, with sign-flip nulls (5,000 permutations recomputing the whole grid) and subject-bootstrap intervals; null-contrast multiverses (odd versus even trials) calibrate the score to alpha. A version arm re-runs fMRIPrep 20.2.7 and 24.x on 30-subject subsets. A leave-one-finding-out ridge model with a dataset random intercept will predict PRS from sample size, design type, TR, mean framewise displacement and trials per condition, fit only after all scores are frozen, and PRS will be compared with NARPS team agreement.

**Expected results.** We expect PRS to span below 0.3 to above 0.9, with sensory and working-memory contrasts above 0.8 and NARPS value contrasts below 0.5; confound strategy and HRF model to explain more variance than smoothing or parcellation; fMRIPrep version to account for under 10 percent of specification variance; and PRS to rank the NARPS hypotheses in the order of team agreement (Spearman rho above 0.7).

**Significance.** The output is a reusable PRS table for classic contrasts, a runner for future datasets, and the first empirical guidance on which design features protect a task-fMRI finding from analytic flexibility.

### Decomposing Functional-Connectivity Prediction of Cognition Into Motion Artifact, Motion Trait and Brain Signal

*Project: [`motion-causal-decomposition`](projects/motion-causal-decomposition/)*

**Background.** Head motion corrupts resting-state functional connectivity in ways that survive denoising, and it is also a stable, heritable trait correlated with age, attention, IQ and body mass. Existing responses censor, regress or report residual correlations, and recent trait-level impact scores detect over- or under-estimation, but none states how much of a prediction flows through an artifact path versus a trait path. Negative-control exposures, instrumental variables and sensitivity bounds from causal epidemiology have not been applied to motion.

**Objective.** We will decompose the covariance between predicted and observed cognition into an artifact-mediated component, a trait-confounded component and a motion-independent remainder, with identifying assumptions stated and tested, across cohorts with different motion regimes.

**Methods.** In HCP S1200 (about 1,000 subjects with four resting runs over two days), ABIDE I and II (about 2,200 participants) and ABCD (about 7,000 children with usable rest), we will compute per-run framewise displacement and Fisher-z connectivity. Ridge and connectome-based predictive models trained in family-grouped or site-stratified outer folds will be applied to each run of held-out subjects. Artifact sensitivity, the within-subject change in predicted cognition per unit of run-level motion, will be estimated with subject fixed effects and by two-stage least squares with run order and session as instruments (first-stage F above 10). The covariance is split into an artifact term, a trait term identified by motion in acquisitions that cannot corrupt the connectivity used (diffusion-scan movement, other-day rest), and the clean remainder, with 2,000 family- or site-block bootstraps. Nulls include label permutation, motion-matched permutation, a placebo instrument and a synthetic negative-control outcome; robustness values and E-values bound residual confounding; four denoising pipelines are summarised by specification curve.

**Expected results.** We expect negative artifact sensitivity ordered ABCD > ABIDE > HCP; artifact and trait shares of 10–25% and 10–20% for fluid cognition in HCP with a surviving clean component, and combined shares above 50% in ABCD; other-acquisition motion coefficients at 40–70% of same-run coefficients; and more than 30% of the ABIDE group difference to be motion-mediated.

**Significance.** The output is a one-table decomposition (total r, artifact, trait and clean shares, robustness value) that any connectivity-prediction paper can add in place of binary motion exclusion.

### An atlas-robustness index for connectome-based phenotype prediction: do the findings survive a change of parcellation?

*Project: [`cross-atlas-prediction-stability`](projects/cross-atlas-prediction-stability/)*

**Background.** Every connectome-based prediction starts with a parcellation choice, and benchmarks show that this choice changes accuracy modestly and alters individual-difference estimates. Existing comparisons stop at accuracy: because atlases have different node sets, the edges, nodes and networks that carry a prediction are never compared in a common space, no per-phenotype measure tells readers which findings are atlas-robust, the inflation from choosing the best of many atlases post hoc is unquantified, and results come from HCP alone.

**Objective.** To quantify, per phenotype, how much connectome-based predictions and their findings depend on parcellation, to estimate post-hoc atlas-selection inflation, and to test whether a robustness index replicates across datasets.

**Methods.** Resting-state data from HCP S1200 (about 1,000 subjects) and AOMIC ID1000, PIOP1 and PIOP2 (about 900, 200 and 200) will be parcellated with 12-15 atlases spanning Schaefer 100-1000, Glasser, Gordon, Brainnetome, AAL, Craddock, DiFuMo and individualised parcels. Ridge and CPM with family-grouped nested cross-validation (20 repeats) will predict pre-specified phenotypes: age, sex, cognition composites, processing speed and NEO-FFI traits in HCP; age, sex, IQ, education and NEO-FFI in AOMIC. Haufe-transformed patterns will be projected to fsLR vertices and Yeo networks. The atlas-robustness index combines normalised accuracy dispersion, pairwise Spearman concordance of out-of-fold subject predictions, and vertex-level finding concordance tested against parcel-permutation and spin nulls, with bootstrap CIs over subjects and a permuted-phenotype null. Selection inflation will be estimated as best-of-atlases minus nested atlas selection and atlas-ensemble accuracy across repeats and sample sizes of 100 to 800; a mixed model will partition variance between atlas family and resolution; simulations with a ground-truth parcellation will validate the index.

**Expected results.** We expect accuracy ranges below 0.05 in r for age and sex and 0.05-0.15 for cognition, subject-level concordance above 0.8 for age and sex but below 0.4 for low-reliability traits, network-level concordance above 0.7 even where vertex maps diverge, family mattering more than resolution, best-of inflation of 0.03-0.08 in r at n = 400 that ensembles remove, and index rank correlation above 0.7 between HCP and AOMIC.

**Significance.** The index and consensus maps would let readers separate atlas-robust findings from atlas artefacts and support reporting atlas-ensemble predictions.

### Resting-State Blood-Arrival Lag and HRF Timing as Vascular-Aging Biomarkers: ASL Validation, Normative Charts and the Hemodynamic Share of the Functional-Connectivity Age Effect

*Project: [`fmri-hemodynamic-aging`](projects/fmri-hemodynamic-aging/)*

**Background.** With age, cerebral blood transit slows and the hemodynamic response becomes smaller and later, so hemodynamic timing is both a candidate cerebrovascular biomarker available from any resting scan and a confound of functional-connectivity (FC) aging effects. Systemic low-frequency-oscillation lag mapping and resting-state HRF deconvolution capture this timing, yet neither has been validated against an independent transit-time measurement at cohort scale, no normative charts exist for hemodynamic timing, and the hemodynamic share of the FC-age effect has not been estimated with both correction strategies on the same data.

**Objective.** We will establish test-retest reliability of lag and HRF timing, validate lag maps against multi-delay pCASL arterial transit time (ATT) in the same sessions, fit sex-specific normative charts and a hemodynamic-age model in HCP-Aging, test associations with vascular risk, WMH and amyloid in OASIS-3, and quantify FC-age attenuation after hemodynamic correction.

**Methods.** HCP-Aging (about 1,200 participants aged 36-100+, four resting runs at TR 0.8 s, multi-delay pCASL) will be processed from the minimally preprocessed volumes without ICA-FIX. Lag maps use three-pass cross-correlation in 0.009-0.15 Hz; HRF timing from point-process deconvolution. Reliability is ICC(2,1) between same-visit sessions; lag-ATT agreement uses spatial correlation with 10,000-rotation spin tests. Heteroscedastic spline normative models by sex yield centiles; ridge regression on about 800 timing features with participant-grouped 10-fold CV and label-shuffled permutation nulls yields a hemodynamic-age delta regressed on vascular risk factors with Holm correction. OASIS-3 (about 1,378 participants, FLAIR WMH volume, Centiloid, CDR) is processed with fMRIPrep and harmonised by ComBat-GAM, with models fitted only on HCP-Aging. Edge-wise FC age slopes are compared across confound-regressed, sLFO-regressed and HRF-deconvolved series with participant bootstrap and mediation (2,000 resamples).

**Expected results.** We expect ICC of at least 0.6 for parcel lag and dispersion; within-participant lag-ATT spatial r of 0.3-0.6; WMH explaining more delta variance (partial R-squared at least 0.05) than Centiloid (below 0.02); 20-40% attenuation of FC-age slopes; and a median absolute z-shift under 0.3 on transfer to cognitively normal OASIS-3 participants.

**Significance.** A validated, normative hemodynamic-age measure from routine resting fMRI would give a free vascular-aging marker and correct a confound in FC aging research.

### Lifespan centile charts of breath-hold cerebrovascular reactivity amplitude and delay, anchored by test-retest reliability and gas-free bias

*Project: [`cvr-normative-maps`](projects/cvr-normative-maps/)*

**Background.** Cerebrovascular reactivity (CVR), the BOLD response per unit change in end-tidal CO2, rises through childhood, falls with age and confounds every BOLD comparison across age groups. Reference atlases come from a few dozen healthy young adults, developmental work in NKI-Rockland children stops at 18, and adult curves come from gas studies of fewer than 100 people. No lifespan centile model exists, and the test-retest reliability, minimal detectable change and bias of gas-free regressors have not been measured at the precision needed for individual deviation scores.

**Objective.** We will build the first lifespan centile charts (ages 6-85) of breath-hold CVR amplitude and delay, quantify their reliability and gas-free bias, and test whether vascular risk shifts an individual's centile and whether resting-state proxies reproduce them.

**Methods.** Reliability and bias will be established in EuskalIBUR (OpenNeuro ds003192; 7 participants, 10 weekly breath-hold sessions with PetCO2): ICC(2,1), within-subject CoV and MDC95 for grey-matter and parcel amplitude and delay with PetCO2, boxcar and RVT regressors, with Bland-Altman analysis of gas-free regressors. Charts will be fitted in NKI-Rockland enhanced (more than 1,000 participants, 6-85 years, breath-hold task with respiration belt, resting-state, vitals and history) after fMRIPrep and a lag-optimised voxelwise GLM. A sex-specific B-spline location-scale model with protocol, motion and compliance covariates yields centiles and z-scores. NKI is split into age-stratified discovery and replication halves; centiles are estimated on discovery only, and probit-transformed centiles are regressed on hypertension, BMI, smoking and diabetes on the replication half (Holm across four factors). Nulls: phase-randomised BOLD and age permutation. Resting-state proxies (CO2-band ALFF, sLFO and RVT-response amplitudes) are compared with breath-hold amplitude within subject.

**Expected results.** We expect amplitude ICC above 0.7 with PetCO2 and above 0.5 with the boxcar, delay ICC of 0.4-0.6, and MDC95 below 30% of the between-subject SD; an inverted-U amplitude curve peaking in the early 20s and delay rising after 40; hypertension lowering the amplitude centile by at least 0.5 SD; proxies correlating with breath-hold amplitude at r = 0.4-0.6; and the CVR centile attenuating age effects on task BOLD by 20-50%.

**Significance.** Open centile tables with measured reliability turn CVR into an individual-level deviation score for vascular ageing and give BOLD studies a vascular covariate.

### How Reproducible Are Laminar fMRI Profiles? Variance Decomposition, a Specification-Curve Multiverse and SNR Degradation Across Open 7T Datasets

*Project: [`layer-fmri-7t-reproducibility`](projects/layer-fmri-7t-reproducibility/)*

**Background.** Laminar fMRI at 7T has produced landmark claims of layer-specific feedforward and feedback signals, but gradient-echo BOLD is biased toward superficial layers by draining veins and depth depends on equidistant versus equivolume layering. Replications test one paradigm in one lab. No study has decomposed profile variance into lab, subject, session and pipeline components, reported analysis choices as a specification curve, or estimated how laminar claims degrade toward 3T-like sensitivity.

**Objective.** We will quantify within-lab reliability and cross-lab variance of cortical-depth profiles, map the analysis multiverse of the key laminar contrasts, estimate the tSNR at which profile-shape classification fails, and validate a chance-corrected reproducibility index.

**Methods.** Open 7T datasets will be pooled by paradigm class (M1 tapping, V1 stimulation, dlPFC working memory): the Kenshu whole-brain VASO+BOLD dataset (one participant, six sessions; OpenNeuro ds003216), V1 VASO (ds001547), Donders Repository GE-BOLD V1 datasets (~20-30 participants each), the 21-participant MPI CBS dlPFC replication and 5-15 further OpenNeuro datasets. Harmonised preprocessing, 5x upsampling, equidistant and equivolume layering (3-20 bins, LayNii as reference), canonical and FIR GLMs, anatomical versus independent-run activation ROIs, and three deveining strategies define >= 200 specifications. Reliability uses ICC(2,1) per bin and Lin's CCC across sessions with bootstrap CIs; nested mixed models estimate lab, subject and residual variance; the specification curve's sign-consistency share is compared with a null curve from 500 within-run condition-label permutations. SNR degradation adds noise to target tSNR and resamples to 1.0-1.2 mm (200 realisations per level), with the 80% accuracy point from a logistic fit against held-out references. Deveining parameters are fixed from the literature and phantoms.

**Expected results.** We anticipate within-subject CCC > 0.8 for VASO in M1 and V1; between-lab variance exceeding between-subject variance for GE-BOLD but not VASO; the published sign preserved in < 60% of GE-BOLD superficial-versus-deep specifications versus > 80% for the VASO M1 double peak; equivolume layering shifting peak depth by >= 1 bin in curved ROIs; and shape classification falling below 80% at roughly one-third of 7T tSNR.

**Significance.** The study gives the field its first cross-lab reproducibility numbers for laminar profiles, a reusable index, and guidance on which claims can be attempted at 3T and at what cost in runs.

### How much of a PET binding-potential result is the kinetic model? A specification-curve and test-retest audit across OpenNeuro PET-BIDS datasets

*Project: [`openneuro-pet-kinetic-multiverse`](projects/openneuro-pet-kinetic-multiverse/)*

**Background.** Reference-tissue binding potential (BP_ND) depends on rarely justified choices: SRTM, SRTM2, Logan reference, MRTM2 or SUVR; reference-region delineation; t*; frame weighting; scan truncation; and k2' handling. A single-site multiverse of about 1,000 preprocessing pipelines on one [11C]DASB dataset showed that pipeline choices change reliability and conclusions, but the kinetic model was fixed and the data closed. The OpenNeuro PET portal now lists about 32 PET-BIDS datasets, roughly a third dynamic with reference-region tracers and several with test-retest sessions, but no cross-dataset multiverse exists.

**Objective.** We will run a pre-registered kinetic-modelling multiverse on every dynamic reference-region dataset on OpenNeuro and quantify the BP_ND variance due to analytic choices, which specifications maximise test-retest reliability, how much scan time can be cut, and whether published contrasts survive.

**Methods.** Regional time-activity curves will be extracted with PETPrep using cerebellar grey, whole and eroded cerebellar references. Open implementations of SRTM, SRTM2, Logan with and without k2', MRTM, MRTM2 and SUVR will be validated by recovering simulated parameters within 5 percent and by concordance with kinfitr on real data. About 2,000 specifications over roughly 300 scans and 20 regions will be summarised by nested ANOVA with eta-squared per factor and subject-resampled bootstrap CIs, ICC(2,1) and within-subject CV per specification in test-retest datasets, and the fraction of specifications reproducing each published contrast's sign and significance, with joint inference by permuting group labels or ages within dataset. Population k2' is estimated leave-one-subject-out; t* is fixed per specification, never per subject.

**Expected results.** We expect 10 to 30 percent of total BP_ND variance to be attributable to specification factors, with model family and t* dominant except where cerebellar off-target signal makes the reference region dominant; Logan and SUVR to show BP-dependent bias; SRTM2 or MRTM2 with population k2' and frame weighting to be most reliable, beating the default by more than 0.05 ICC in at least one tracer; published signs reproduced in over 90 percent but significance in under 60 percent of specifications; and 60-minute scans to suffice only for fast tracers.

**Significance.** The released specification table and reliability curves let PET studies justify pipelines by evidence and place kinetic-model variance beside the known preprocessing variance.

### Which imaging-transcriptomics findings survive their nulls? A factorial benchmark of spatial and gene-set null models with a calibrated reporting protocol

*Project: [`imaging-transcriptomics-nulls`](projects/imaging-transcriptomics-nulls/)*

**Background.** Imaging transcriptomics correlates cortical maps with Allen Human Brain Atlas expression and interprets the correlated genes through category enrichment. Two corrections are now accepted: spatial nulls for autocorrelated maps and ensemble gene-set nulls, since some categories are enriched for almost any smooth map. Most classic findings predate both or used one, spatial nulls have been benchmarked on map-map correlations rather than enrichment, ensemble nulls were shown mainly in mouse, a recent single-case preprint shows a spin-guarded enrichment dissolving under complementary nulls, and no calibrated rule says which nulls to report.

**Objective.** We will re-test ten canonical human map-transcriptome findings under a factorial of null and processing choices, calibrate decision rules by false-positive rate and power on planted signals, and apply the resulting protocol to the HCP T1w/T2w myelin map and HCP-Aging age-effect maps.

**Methods.** Expression will be processed with abagen (six donors, 3,702 samples, about 20,000 genes) at three gene-filter levels onto Glasser 360, Schaefer and Desikan parcellations; benchmark maps and gene lists will be reconstructed from open data or supplements. Five spatial null families (two spin variants, variogram surrogates, Moran spectral randomisation, eigenstrapping), validated against neuromaps and BrainSMASH, and two gene-set nulls (size-matched random genes and an ensemble null on the same surrogates) yield a 1,800-cell factorial with 5,000 surrogates per cell, summarised as survival heatmaps. Planted gene-set maps with autocorrelation-matched noise will score five decision rules on false-positive rate and power. Donors are the unit for expression uncertainty (leave-one-donor-out); left-hemisphere and mirrored analyses are reported separately; BH-FDR applies across categories.

**Expected results.** We expect spatial null families to disagree by more than tenfold in p for at least a quarter of moderately correlated genes; at least half of the published enrichment claims to lose significance under the ensemble null while map-level correlations survive; parcellation resolution to flip at least three findings; the rule requiring both nulls to hold false positives at or below 5 percent with at least 80 percent power at r of 0.3 and 360 parcels; and the myelin-hierarchy association to survive all nulls while only myelination categories persist.

**Significance.** The survival matrix, calibrated protocol and released surrogate caches give the field an evidence-based standard for transcriptomic decoding.

### Do human imaging-transcriptomics associations replicate in the mouse? A MERFISH-based cross-species replication panel with matched spatial and gene-ensemble nulls

*Project: [`cross-species-imaging-transcriptomics`](projects/cross-species-imaging-transcriptomics/)*

**Background.** Imaging transcriptomics correlates MRI maps with regional gene expression from the Allen Human Brain Atlas, and mouse expression is routinely invoked to interpret such associations mechanistically. Cross-species work so far relies on the single-animal Allen ISH atlas, compares expression to expression rather than testing specific published gene-map associations, and rarely applies spatial-autocorrelation and gene-ensemble nulls in both species. The Allen Brain Cell Atlas now provides about 4 million MERFISH cells (500-gene panel) and about 9 million cells (1,122 genes) with cell-type labels, so a mouse regional map can be split into cell-type composition and within-type expression.

**Objective.** We will test what fraction of a pre-registered panel of about 40 published human AHBA-versus-MRI associations replicates in the mouse when the mouse side is measured with MERFISH, and whether replicating associations are carried by cell-type composition or by within-type expression.

**Methods.** MERFISH cells will be aggregated to the 213 CCF summary structures and each gene's regional profile decomposed by ridge regression on subclass proportions into composition and residual components. Human expression will come from abagen on a parcellation matched in region count. For each human map with a mouse analogue we will compute per-gene Spearman associations in both species. Nulls: 3-D variogram-matched surrogates for mouse, spin or eigenstrapping surrogates for human cortex, and expression- and autocorrelation-matched random gene sets, at least 5,000 surrogates per test. An association replicates only if it passes both null types in both species (conjunction p, BH-FDR q = 0.05 across the panel). The panel is fixed before the mouse side is computed; ISH-versus-MERFISH agreement and split-animal replication define the noise ceiling; 1,000 random genes calibrate false-positive rates per null.

**Expected results.** We expect fewer than half of the panel to replicate, with sensorimotor-dominant maps replicating more often than association-cortex and disorder maps; MERFISH maps to agree better with AHBA orthologs than ISH maps; at least two thirds of replicating associations to be carried by composition; and enrichment p-values from the 500-gene panel with a genome-wide background to be inflated by more than an order of magnitude relative to a panel-aware null.

**Significance.** A replication table annotated by modality and composition will tell the field which human associations survive a cross-species test and which are composition effects.

### Do Published Brain-Map Correlations Survive the Null Model? A Registered Re-Analysis of 50 Cortical Map-to-Map Associations Under Six Spatial Null Families

*Project: [`spatial-null-replication-audit`](projects/spatial-null-replication-audit/)*

**Background.** Correlating two smooth cortical maps requires a spatial null model: spin tests, variogram-matched surrogates, Moran spectral randomisation or eigenstrapping. These nulls disagree, and recent simulations show that spin and variogram surrogates, the two most used, fail to control the false-positive rate for strongly autocorrelated maps. Nobody has returned to the published map-to-map literature to ask how many claims depended on that miscalibration.

**Objective.** We will re-analyse 50 published cortical map-to-map correlations under a matrix of spatial nulls, quantify per-claim null robustness, and explain flips by each null's false-positive rate on smoothness-matched simulations.

**Methods.** Claims will be selected by a pre-registered PubMed and Europe PMC search (2018-2025) requiring public maps, a scalar correlation as headline test and a stated parcellation and null; 50 eligible claims will be sampled by citation count within year, stratified by null used. Maps will be fetched from neuromaps, NeuroVault, the Allen Human Brain Atlas, ENIGMA and HCP S1200, and the reported r reproduced within 0.05 before audit. Each claim will be tested with 10,000 surrogates under naive permutation, nearest-neighbour, one-to-one and projection-corrected spin, Moran spectral randomisation, variogram matching and eigenstrapping, cross-checked against reference packages. Per claim we will record a null-robustness index (fraction of null families with p below 0.05) and the surrogate autocorrelation deviation. A calibration experiment will simulate 1,000 independent Gaussian random field pairs at each claim's autocorrelation length and parcellation to measure each null's false-positive rate with Wilson intervals. Synthetic positive and negative controls bound the procedure, a sweep spans 100 to 1,000 Schaefer parcels, and cluster-robust logistic meta-regression relates flips to autocorrelation length, original null, parcel count and reported p.

**Expected results.** We expect at least 90% of claims to replicate under their original null but only 60-75% under all corrected nulls, with losses concentrated in claims with above-median autocorrelation length and reported p between 0.001 and 0.05; variogram-tested claims to flip more often under eigenstrapping; and flipped claims to show a smoothness-matched false-positive rate above 0.10 for their original null.

**Significance.** The audit turns a methods debate into an empirical account of the literature, with a reusable claim registry and reporting checklist.

### Anatomically Matched Null Lesions for Lesion Network Mapping: Ground-Truth Simulations with a Volume Confound, a Prior-Leakage Statistic, and Structural Versus Functional Mapping in the Aphasia Recovery Cohort

*Project: [`lesion-network-mapping-nulls`](projects/lesion-network-mapping-nulls/)*

**Background.** Lesion network mapping seeds lesions in a normative connectome, but stroke lesions follow arterial territories, so any lesion set yields a structured map. Recent work showed that label permutation controls type-I error in null studies and that bias maps are cohort-specific and connectome-independent. The null families have not been compared against ground truth under realistic lesion anatomy with a volume confound; no null resamples real lesions matched on volume and territory; no statistic reports how much of a map is the cohort's lesion prior; and functional and structural mapping have not met identical nulls on open data.

**Objective.** We will measure type-I error, power and specificity of six null families under a known network with and without a volume term, introduce territory- and volume-matched resampling from the ATLAS v2.0 pool, define a prior-leakage statistic, and compare functional and structural mapping in the Aphasia Recovery Cohort.

**Methods.** Lesion masks from ATLAS v2.0 (655 public masks) and the Aphasia Recovery Cohort (ARC; 228 chronic left-hemisphere strokes with WAB-R scores) are resampled to MNI 2 mm with volume, hemisphere and arterial territory. Functional maps use voxel-to-parcel connectivity from 100 HCP-YA subjects; disconnection maps use the HCP-842 template. Symptoms are simulated as beta times lesion-to-network connectivity plus gamma times log volume plus noise (n = 50, 100, 228; 500 studies per condition); each null (label permutation, random spheres, shuffled locations, matched resampling, connectome bootstrap, spatial surrogates) is scored by empirical FPR (Wilson CIs) at max-statistic FWER alpha = 0.05, power and Dice. Prior leakage is the R-squared of an observed map on the expected map under matched resampling. Tolerances are fixed in simulation before ARC behaviour is touched.

**Expected results.** We expect label permutation to exceed nominal error when unmodelled volume drives symptoms, random spheres to inflate false positives at hubs, and matched resampling to hold near-nominal error in both cases at modest power cost; prior leakage above 50% for sensitivity-style ARC maps but below 30% for adjusted regression maps; structural disconnection to gain >= 0.15 precision over functional mapping; and covariate adjustment to make permutation and matched-resampling inferences converge (Jaccard > 0.7).

**Significance.** A ground-truth-anchored, open comparison would say which null to believe when lesion volume confounds symptoms, and give every published map a leakage score.


## Neuromodulation & biophysical modelling

### Morphological Variance Versus Field Variance: Stimulation Thresholds Across Hundreds of Real Cortical Reconstructions

*Project: [`morphology-dependent-stimulation`](projects/morphology-dependent-stimulation/)*

**Background.** Individualized brain-stimulation dosing assumes that the neurons receiving an electric field are interchangeable enough for field magnitude to dominate response variance. Multiscale models rest on a small canonical set of neurons, and recent uniform-field simulations showed that within-type morphological variability blurs cell-type specificity. Whether that variability is large relative to the between-subject spread of E-field dose, on real reconstructions with laboratory modelled, is untested.

**Objective.** We will quantify the within-class spread of activation threshold across several hundred real human and mouse cortical reconstructions, compare it with the between-subject spread of E-field dose that individualized modelling corrects, and test whether the morphological ranking of susceptibility transfers across modalities.

**Methods.** We will retrieve 300–800 CNG-standardised pyramidal and interneuron SWC reconstructions from NeuroMorpho.Org plus about 100 human and 100 mouse Allen Cell Types reconstructions, after clone deduplication. A passive cable model under the quasi-uniform approximation yields a linear polarization sensitivity matrix per cell, swept over 128 field directions and DC, tACS, TMS and DBS waveforms, weighted by achievable directions from the SimNIBS ernie head model. About 40 stratified cells re-simulated in NEURON with active channels will calibrate the surrogate. Mixed models of log threshold on species, class, size and layer with an archive random intercept will partition variance; the human–mouse ratio uses a cluster bootstrap over archives. Nulls include direction shuffling, within-archive species permutation and size-matched resampling; six confirmatory hypotheses are Benjamini–Hochberg corrected at q = 0.05.

**Expected results.** We expect the archive-adjusted interquartile range of log threshold within human pyramidal cells to exceed 0.2 dex, larger than the roughly 1.5–2-fold between-subject IQR of E-field dose reported by head modelling; pyramidal-versus-interneuron probability of superiority below 0.70; Spearman ρ above 0.9 between tDCS and TMS rankings; larger anisotropy for pyramidal cells; and lower human than mouse thresholds only if the effect survives the archive intercept.

**Significance.** The result is a variance budget, not a specificity test: it states where the canonical model neuron sits in the real distribution, whether one susceptibility index generalizes across devices, and whether field-based individualization alone can explain who responds.

### Dose Confounds Connectivity: A Spatial-Null Audit and Joint Connectivity-by-Electric-Field Model of Stimulation Targeting Against Measured TMS Responses

*Project: [`tms-target-connectivity`](projects/tms-target-connectivity/)*

**Background.** Network mapping predicts where transcranial magnetic stimulation (TMS) should be applied from normative connectivity to a reference region, while field modelling shows that dose at a cortical target varies with scalp-to-cortex distance and gyral geometry. Because both are spatially structured, a connectivity-effect correlation can be produced in part by dose, yet network-mapping studies rarely measure the field or use spatially constrained nulls. Joint connectivity and field targeting has recently been proposed; unanswered are how much of the published literature survives spatial nulls, how much is attributable to dose, and whether individualised connectomes help once dose is held constant.

**Objective.** We will audit reproduced connectivity-effect correlations under spatial nulls, split connectivity's contribution into dose-shared and unique parts, and test a field-weighted connectivity score against measured TMS responses.

**Methods.** The HCP S1200 group-average functional connectome (n = 1,003) and tractography-derived structural connectomes will supply normative connectivity on the HCP-MMP1 parcellation, with Schaefer parcellations as sensitivity checks. For OpenNeuro ds004024 (13 participants, TMS-EEG with anatomical and diffusion MRI) and ds005498 (TMS-fMRI), SimNIBS fields will be simulated per subject at recorded coil placements, parcellated and normalised to a constant M1 field at motor threshold. Outcomes are TMS-evoked potential amplitudes by latency and parcel-wise BOLD responses. Connectivity-only, network-dose and local-dose scores plus distance baselines will enter ridge regression evaluated leave-one-subject-out and leave-one-dataset-out, with any threshold fitted inside training folds. Map comparisons will use 10,000 variogram-matched and 10,000 spin surrogates cross-checked against reference packages, plus dose-preserving and site permutations, with Benjamini-Hochberg control.

**Expected results.** We expect at least 30% of nominally significant correlations to fail variogram nulls, field to correlate with seed connectivity (absolute rho above 0.2), the joint score to improve held-out Spearman rho over connectivity-only with a bootstrap CI excluding zero, a thresholded weighting to fit best at 40-100 V/m, and individualised connectomes not to beat the normative one at 30 minutes of fMRI. Whether connectivity retains a unique increment after partialling dose is open.

**Significance.** The audit determines whether a widely used targeting rationale needs re-derivation and releases dose-corrected scores and field maps.

### Gain or Shape? A Pooled Subject-Level Re-Analysis of Individual Electric Field and Behavioural Response to Transcranial Electrical Stimulation Under Montage and Spatial Null Models

*Project: [`tes-dose-individualization`](projects/tes-dose-individualization/)*

**Background.** Transcranial electrical stimulation delivers a fixed 1-2 mA current regardless of head anatomy, yet finite-element models show two- to three-fold variability in cortical electric field (E-field) for the same montage. Individual E-field has been linked to outcomes in single studies, and demographic predictors of peak E-field are now established for dose standardisation. Two problems remain. E-field maps from one montage are spatially autocorrelated and nearly identical in shape across subjects, differing mainly in gain, so vertex-wise E-field-to-behaviour maps tested by parametric or naive permutation methods can be a montage artefact, and none has used spatial nulls. Single studies of 10-40 participants cannot separate whether dose magnitude or field shape drives behaviour.

**Objective.** We will pool subject-level E-field and behavioural data across independent open tDCS and tACS datasets to test whether individual target E-field predicts behavioural response, whether shape adds anything beyond global gain, and how many map-level findings survive spatial and montage nulls.

**Methods.** A programmatic OpenNeuro registry will identify tES datasets with a behavioural outcome and, where present, a T1-weighted image (for example ds003670 and ds006126); ROI definitions, sign conventions and models will be pre-registered on OSF before any simulation. SimNIBS 4 head models will be built per subject, or the MNI152 template for datasets without T1 (flagged), simulated at 1 mA and scaled to the applied current. Dose metrics include target ROI E-field, focality and global grey-matter mean magnitude (gain). Behavioural effects (active minus sham, standardised) will be modelled per dataset on dose with age and sex, pooled by DerSimonian-Laird random effects with I-squared, prediction intervals and leave-one-dataset-out sensitivity. Nested mixed models test target dose against gain plus residual shape. Vertex-wise maps on fsaverage will be evaluated with 5,000 spin rotations and 5,000 subject-label permutations that keep each map intact, with a max-statistic and equivalence tests at r = 0.2.

**Expected results.** With 150-250 pooled subjects we have 80% power for r of 0.2-0.25 and expect a positive pooled dose slope, no added value of residual shape once gain is removed, and fewer than 25% of naively significant vertices surviving spatial and montage nulls.

**Significance.** The study will either establish E-field gain as the usable dose variable for individualised tES or show that E-field-behaviour correlations are null-model artefacts.

### How much of the deep-brain-stimulation activation volume is neuronal? Threshold distributions and variance decomposition over hundreds of empirical axon reconstructions

*Project: [`dbs-vta-population-variability`](projects/dbs-vta-population-variability/)*

**Background.** Connectomic and clinical DBS analyses rely on the volume of tissue activated, computed by thresholding a field or by pathway activation models on streamlines. Lead-DBS now ships probabilistic activation metrics that incorporate axon-morphology variability. That variability comes from a few idealised or generative axons rather than empirical reconstructions, and no study has formally attributed threshold uncertainty to morphology versus fibre diameter, placement, orientation and conductor model, or provided a surrogate that avoids per-axon biophysical simulation.

**Objective.** We will compute activation-threshold distributions for hundreds of empirically reconstructed axons in Lead-DBS lead fields, decompose threshold and activation-radius variance with Sobol indices, test branch-point effects, benchmark the resulting probabilistic activation volume against Lead-DBS's metrics and the 0.2 V/mm isosurface, and fit an activating-function surrogate.

**Methods.** Morphologies: about 300 axon-bearing NeuroMorpho.org reconstructions (cortex, pallidum, STN, thalamus) plus about 50 whole-brain MouseLight and SEU-ALLEN layer-5 pyramidal-tract neurons. Lead fields for Medtronic, Boston Scientific and Abbott leads use point-source contacts in a 0.2 S/m medium with encapsulation and anisotropy proxies. Thresholds are found by bisection in a CRRSS myelinated-axon model, calibrated against MRG axons in NEURON on 50 morphologies. The factorial sweep covers diameter (2-10 um), distance (0.5-6 mm), five orientations, conductor model and pulse width (60/90/120 us). Estimands: threshold CV and log-normal versus normal fit (AIC); first-order and total Sobol indices with cluster bootstrap by archive; 50%, 5% and 95% activation-probability radii; median threshold ratio for axons with a branch point or terminal within 1.5 mm of the contact; a surrogate from activating-function statistics under leave-morphology-out validation. Nulls: re-straightened paths of equal length and permuted morphology labels.

**Expected results.** We expect threshold CV of at least 25% with a right-skewed heavy tail; diameter to dominate variance at all distances and morphology to exceed the conductor model within 3 mm; the 50% radius to differ from the 0.2 V/mm radius by more than 0.5 mm with a 5-95% band above 1 mm; a branch-point median ratio of 0.8 or less; and surrogate R2 of at least 0.8.

**Significance.** A released threshold table and surrogate let Lead-DBS-style pipelines report activation with credible bands and direct modelling effort to where uncertainty lives.

### Type-Resolved Population Models of Retinal Ganglion Cell Activation by Epiretinal and Subretinal Prostheses: Threshold Distributions, Morphological Determinants and the Selectivity Ceiling

*Project: [`retinal-prosthesis-morphology-models`](projects/retinal-prosthesis-morphology-models/)*

**Background.** Retinal prostheses activate ganglion cells without type selectivity and with unwanted axon-bundle activation. Biophysical models have clarified the roles of the axon initial segment (AIS), soma-to-axon geometry and pulse waveform, but every study has used one to about a dozen morphologies. How much thresholds vary within and between real morphological types, and what that implies for attainable selectivity, is unquantified.

**Objective.** We will derive per-type activation-threshold distributions from hundreds of typed reconstructions under epiretinal and subretinal fields, model their morphological determinants, rank AIS-geometry uncertainty against morphological variability, and estimate a selectivity ceiling validated against ex vivo multi-electrode threshold data.

**Methods.** Reconstructions from the Eyewire museum (about 400 cells, 47 types), Sümbül et al. (about 380 cells) and NeuroMorpho.org (about 1,500 retinal ganglion cells) will be compartmentalised into 10 micrometre cylinders with synthesised axons toward the optic disc and AIS start and length as design factors. A Fohlmeister-Miller multicompartment model with region-specific channel densities will compute propagated-spike thresholds by bisection under disk-electrode fields (epiretinal and subretinal placements) and pulses from 50 microseconds to 4 milliseconds, with a NEURON calibration subset of 50 cells. Statistics comprise per-type spread, mixed models of log-threshold with type and archive random intercepts, bootstrap variance fractions for AIS versus morphology, pairwise selectivity AUC and activation-site classification. Nulls shuffle type labels and randomise electrode offsets. Empirical comparison fits one global scale on ON parasol cells and tests the remaining types by Kolmogorov-Smirnov against published primate and human data.

**Expected results.** We expect within-type coefficients of variation of at least 20% with between-to-within variance ratios between 0.5 and 2; soma diameter and AIS distance explaining at least 40% of between-cell variance, with dendritic field mattering mainly for subretinal placement; AIS uncertainty exceeding same-type cell replacement; pairwise AUC below 0.8 for short pulses; distal-axon activation above 30% epiretinally and below 10% subretinally; and reproduction of the ON/OFF parasol versus midget ordering.

**Significance.** The open per-cell threshold table and fitted determinant model provide a realistic population prior for stimulus optimisation and a quantitative bound on type-selective retinal stimulation.

### How Much Axon Initial Segment Plasticity Does a Dendritic Tree Demand? Population-Scale Simulation Across Real NeuroMorpho.org Reconstructions

*Project: [`ais-plasticity-real-morphologies`](projects/ais-plasticity-real-morphologies/)*

**Background.** Neurons relocate or resize the axon initial segment (AIS) to regulate excitability, with effects that depend on the somatodendritic tree. Resistive-coupling theory predicts that somatic threshold depends on soma-AIS axial resistance and AIS sodium conductance but little on dendritic load, whereas somatic spike amplitude does depend on load. These predictions have been tested on a handful of morphologies; the AIS geometry required to hold a set point across the real diversity of neuronal shapes has never been computed, and whether measured AIS-dendrite covariation matches a homeostatic prediction is unquantified.

**Objective.** We will quantify, for 500-2,000 quality-controlled NeuroMorpho.org reconstructions across five cell classes, the AIS relocation or elongation each tree demands to hold threshold, rheobase or somatic spike amplitude at a class-specific set point, test whether resistive-coupling theory predicts the simulations, and compare predicted covariation with published AIS geometry.

**Methods.** Passive frequency-domain solutions of each tree will yield input conductance, effective capacitance at 500-2,000 Hz and somatic transfer impedance, validated against analytic ball-and-stick solutions. A standard axon with variable AIS start, length and sodium density will be attached in NEURON with three ModelDB channel sets; threshold, rheobase, amplitude, maximal dV/dt and onset rapidness will be measured by grid and bisection. Set points are defined beforehand from Allen Cell Types electrophysiology, independent of the simulated morphologies. Mixed models with archive as random intercept will test load independence of threshold, log-linear scaling of demand with load (cluster bootstrap over archives), theory-versus-simulation R^2, agreement of predicted and measured covariation slopes, and morphology-versus-channel-set variance decomposition. Nulls include permuted morphology-load pairing, ball-and-stick controls spanning the same load range, and a constant-AIS reference model; shrinkage metadata enter as covariates.

**Expected results.** We expect threshold to vary by less than 3 mV over a tenfold load range while rheobase varies more than fivefold, demand to scale log-linearly with load at the theoretical slope, R^2 above 0.8 for point-like AIS, measured covariation within the predicted interval, and morphology to explain more demand variance than channel set.

**Significance.** The study turns a single-neuron theory into a population-scale, falsifiable account of AIS plasticity and releases per-neuron demand tables.


## EEG, sleep & seizures

### Leakage, scorers and age shift: an audit of scalp-EEG seizure detectors trained on CHB-MIT and transferred to adult and neonatal cohorts

*Project: [`cross-dataset-seizure-generalization`](projects/cross-dataset-seizure-generalization/)*

**Background.** Scalp-EEG seizure detectors routinely report above 95% performance on CHB-MIT, yet the 2025 SzCORE challenge reached an event-level F1 of 0.43 on continuous EEG. Window-level splits leak subject identity, and foundation-model benchmarks document cross-cohort drops without explaining them. Nobody has quantified leakage inflation for the same model families under one event scorer, tested whether rankings depend on the scorer, separated acquisition shift from age shift, or evaluated label-free domain adaptation under event scoring.

**Objective.** To audit how much CHB-MIT performance is inflated by split regime and model capacity, whether scorers reorder models, how much of the cross-cohort drop is acquisition versus age shift, and whether foundation models and domain adaptation close the gap.

**Methods.** CHB-MIT (23 pediatric subjects, about 980 hours, 198 seizures) is the training source; Siena (14 adults, 47 seizures), TUSZ v2.0.3 (mixed ages) and the Helsinki neonatal corpus (79 neonates) are zero-shot targets. All EDFs will be harmonised to an 18-pair longitudinal bipolar montage at 256 Hz. Model families are spectral and Riemannian tangent-space logistic regression, a compact CNN, and BIOT, LaBraM, EEGPT and CBraMod as linear probes and fine-tunes. Window-, record- and patient-wise CHB-MIT splits precede cross-cohort evaluation, with thresholds frozen on CHB-MIT validation folds and target labels used only for scoring. Every model is scored with OVLP, TAES and SzCORE, with SzCORE F1 and sensitivity at 1 false positive per 24 hours as primary endpoints, subject-level bootstrap CIs, Kendall's tau between scorer rankings, Spearman correlation of log-MMD between background windows and F1 drop, and Holm correction within hypothesis families. Riemannian re-centering, Euclidean alignment, CORAL and AdaBN will use unlabelled target background only; TUH pretraining overlap is flagged.

**Expected results.** We expect record-wise minus patient-wise inflation above 0.15 F1 for CNN and foundation-model families and below 0.10 for the spectral baseline, scorer rank correlations below 0.8 on at least one cohort, harmonisation halving the Siena drop but not the Helsinki drop, log-MMD correlating above 0.6 with loss, fine-tuned foundation models reducing but not removing the drop, and re-centering helping on adult targets while raising neonatal false alarms.

**Significance.** The result is a fixed-split, fixed-scorer cross-dataset leaderboard that any detector can enter, and evidence on which reported gains reflect memorised patient identity.

### Does the Heart Add Anything the Brain Does Not Already Say? A Patient-Independent Seizure Forecasting Benchmark on Open Simultaneous EEG and ECG

*Project: [`seizure-forecasting-eeg-ecg`](projects/seizure-forecasting-eeg-ecg/)*

**Background.** ECG is montage-free and tolerated for weeks, and pre-ictal heart-rate-variability (HRV) changes are widely reported. Most such results are patient-specific, drawn from private data with tens of seizures, evaluated at horizons of seconds that make them early detection, and rarely tested against seizure-time surrogate nulls. SeizeIT2 now provides hundreds of open focal seizures with simultaneous behind-the-ear EEG and ECG.

**Objective.** We will build a patient-independent forecasting benchmark pairing scalp EEG with ECG and quantify how much pre-ictal HRV information adds to EEG, how much of it is circadian or vigilance confounding, and whether ECG models transfer across cohorts better than montage-bound EEG models.

**Methods.** SeizeIT2 (125 patients, about 11,600 hours, 886 focal seizures) is the primary corpus and Siena (14 adults, 47 seizures, full 10-20 montage) the transportability cohort. R-peak detection and RR cleaning yield HRV features in 5-minute windows with 1-minute steps, gated by a signal-quality index; EEG features are spectral and Hjorth descriptors, with a matched four-channel condition for Siena. Pre-ictal windows are defined by a seizure prediction horizon of 5 minutes and occurrence period of 30 minutes, with a horizon grid; post-ictal periods are excluded and inter-ictal windows lie at least 2 hours from any seizure. Logistic regression, fusion and gradient-boosting models are trained with subject-grouped five-fold cross-validation, a subject-leakage assertion and firing-power alarms. The primary endpoint is improvement over chance: sensitivity for lead seizures minus the analytical chance sensitivity at the achieved time-in-warning, tested per patient against at least 200 constrained seizure-time surrogates and at population level by Wilcoxon signed-rank, with paired patient-level bootstrap for modality contrasts and Holm correction. ECG features are residualised on clock time, an EEG vigilance proxy and accelerometry.

**Expected results.** We expect ECG-only discriminability to beat the surrogate null in 20 to 40% of SeizeIT2 patients; fusion to add at least 0.05 improvement over chance in that subset and under 0.02 elsewhere; residualisation to remove at least half of the ECG excess over chance; smaller cross-cohort loss for ECG than EEG; and no modality distinguishable from null at horizons of 30 minutes or more in most patients.

**Significance.** The benchmark separates genuine forecasting from early detection and tells device designers what ECG adds once confounds are removed.

### Age as a Continuous Domain Variable in EEG Seizure Detection: Transfer Curves, Embedding Drift and Age-Conditioned Normative Normalisation Across TUSZ, CHB-MIT and Siena

*Project: [`eeg-age-domain-shift`](projects/eeg-age-domain-shift/)*

**Background.** Scalp EEG changes systematically across the lifespan in the same spectral dimensions that seizure detectors use, so a model trained at one age sees a different background distribution at another. Pediatric-to-adult transfer has been studied only as a dichotomy, with adult-trained detectors falling to near chance on CHB-MIT, and cross-dataset benchmarks and foundation-model probes never stratify by age. Age-conditioned normative modelling of background EEG has not been used as a label-free pre-processing step.

**Objective.** We will estimate how seizure-detection performance decays with log-age distance between training and test populations, decompose the CHB-MIT to TUSZ gap into age and site components, test whether foundation-model embedding drift predicts the loss, and evaluate age-conditioned normative z-scoring as a cheap mitigation.

**Methods.** Ages will be parsed from TUSZ v2.0.3 EDF headers (about 1,600 h, 0-90+ years), CHB-MIT (23 subjects, 1.5-22 years, about 198 seizures), Siena (14 adults, 20-71 years, 47 seizures), and recordings harmonised to an 18-pair bipolar montage at 256 Hz in 4-s windows. Seven age bins from infancy to over 65 years define a within-site ladder in TUSZ. For logistic regression, gradient boosting, a compact CNN and frozen BIOT/LaBraM/EEGPT/CBraMod linear probes, we will train on each bin and test on every other, and regress window-level AUROC on log-age distance with a same-site indicator in a mixed model with training-bin random intercept. A normative model of background features as smooth functions of log-age, fitted on TUAB (about 2,300 recordings) and non-seizure TUSZ training windows, will z-score each window given subject age, against per-record standardisation. MMD and a ridge age probe quantify embedding drift. Subject-level bootstrap (2,000 draws), a permuted-age null that should flatten the slope and Holm correction provide inference.

**Expected results.** We expect a negative decay slope with CI excluding zero for every family, steeper for high-capacity models; within-TUSZ age decay explaining at least half of the CHB-MIT to adult-TUSZ drop; adult-to-child transfer losing more than the reverse; embedding age R-squared above 0.5 with MMD-loss Spearman rho above 0.6; normative z-scoring reducing the slope by at least 30%.

**Significance.** A transfer curve rather than a table, plus a normative EEG-background model for 0-90 years, tells developers what age distance costs and how pediatric datasets should be built.

### Interictal spike detectors across scalp and intracranial EEG: false-positive calibration on normal cortex, IFCN-criteria audits and seizure-onset-zone consequences

*Project: [`interictal-spike-detection-benchmark`](projects/interictal-spike-detection-benchmark/)*

**Background.** Interictal epileptiform discharges (IEDs) support the diagnosis of epilepsy and localise the epileptogenic zone, yet automated detectors are validated within one modality on epilepsy recordings, where every annotator miss counts as a false positive. A cross-modality benchmark exists for high-frequency oscillation detectors, not for IEDs. No detector paper reports false-positive rates on normal cortex, audits detections against the six IFCN criteria, stratifies by vigilance, or measures how detector choice alters spike-rate rankings of seizure-onset-zone (SOZ) channels.

**Objective.** We will build one event-level IED benchmark spanning scalp and intracranial EEG with normal-tissue false-positive calibration, an IFCN-criteria audit, vigilance stratification and SOZ consequences.

**Methods.** Corpora are TUEV (scalp annotations), the two-centre annotated sleep-iEEG dataset, Omni-iEEG (302 patients, 178 hours, more than 36,000 annotations), ds003876 (sleep and wake labels), ds003029 (SOZ channels and surgical outcome) and the MNI Open iEEG Atlas (106 patients, about 1,700 normal-cortex channels), all at 512 Hz bipolar. Detectors are an envelope detector, a matched filter, a morphology classifier and a 1-D CNN. Events are matched one-to-one within 100 ms; primary endpoints are event F1, sensitivity at 1 false positive per minute, and false positives per minute on MNI normal channels at the same operating point. Learned detectors use patient-grouped cross-validation; cross-modality runs test on the other modality without target tuning. Each detection is scored on features mapped to the IFCN criteria, and 200 criteria-rich false positives are adjudicated by an independent reader. Per-channel spike rates give SOZ AUROC per patient. Uncertainty uses 2,000-draw patient-level bootstraps; nulls shuffle annotation times and permute channel labels; Holm correction applies within families.

**Expected results.** We expect learned detectors to lose at least half their event F1 across modalities while the envelope detector loses under a quarter; normal-tissue false-positive rates to correlate weakly with precision on epilepsy corpora (rho below 0.3); true positives to satisfy at least four criteria and false positives at most two, over half of criteria-rich false positives being adjudicated as IEDs; detector rankings to change between NREM and wake; and SOZ AUROC to vary by over 0.10 in at least 30 percent of patients.

**Significance.** The benchmark, scorer and normal-tissue set make detector specificity and its clinical consequences measurable.

### An Open, Bias-Corrected Neonatal EEG Maturation Clock From Term-Range Cohorts

*Project: [`neonatal-eeg-maturation-clock`](projects/neonatal-eeg-maturation-clock/)*

**Background.** The neonatal EEG matures lawfully over weeks, so the gap between EEG-estimated and true postmenstrual age is a candidate biomarker of disrupted maturation. Automated estimators reach one- to two-week errors but are trained on private cohorts, none applies the regression-to-the-mean correction that adult brain-age work requires, no normative feature centiles exist, test-retest reliability is unreported, and hypoxic-ischaemic encephalopathy (HIE) background grade has never been placed on the same axis as EEG age.

**Objective.** We will build the first fully open neonatal EEG-age clock with released weights and fixed splits, correct its deviation for bias, publish normative feature centiles, and test whether the corrected deviation tracks HIE grade and seizure burden and whether it is a trait of the infant or a state of the recording.

**Methods.** We will use the Helsinki cohort (79 term neonates, 19-channel EEG at 256 Hz, three seizure annotators) and the Zenodo HIE background-grading cohort of one-hour expert-graded epochs, covering 35–45 weeks. After a common bipolar montage, filtering and artefact screening, we will extract band powers, spectral slope, range-EEG percentiles, interburst intervals, continuity and interhemispheric synchrony on seizure-free epochs. Ridge and gradient-boosting clocks will be fitted under subject-grouped five-fold CV with fixed folds; delta is corrected by regressing it on age within training folds only. Gaussian normative models in age give centiles and a multivariate out-of-distribution score. Outcomes use a Jonckheere–Terpstra trend across HIE grades with permutation p, partial Spearman correlation with consensus seizure burden adjusting for age, ICC(1) across epochs with an epoch-shuffle null, an age-permutation null, 2,000 subject bootstraps and Holm correction, with MAE per two-week bin. Transfer to grade-0/1 HIE recordings and reduction to eight channels test portability.

**Expected results.** We expect MAE at or below 1.5 weeks, a raw delta correlating with age at r below −0.3 that correction removes while reordering who looks delayed (rank correlation below 0.9), a monotone decrease of corrected delta with HIE grade with median delta at or below −2 weeks from grade 2, a partial correlation with seizure burden below −0.3, within-infant ICC above 0.6, and cross-cohort MAE at or below 2 weeks.

**Significance.** Released weights, centile charts and a command-line tool for EDF input would give neonatal units a reproducible maturational biomarker with honestly stated range limits.

### Beyond BIS: Testing Propofol-to-Sevoflurane Transfer of EEG Depth-of-Anaesthesia Models Against Pharmacological and Behavioural Targets in VitalDB

*Project: [`anesthesia-depth-cross-dataset`](projects/anesthesia-depth-cross-dataset/)*

**Background.** Cross-dataset depth-of-anaesthesia studies train EEG models to reproduce the Bispectral Index, a proprietary, smoothed and agent-dependent number, so agent transfer measured against BIS conflates EEG shift with the index's own agent bias. VitalDB records raw frontal EEG with target-controlled propofol effect-site concentration, end-tidal sevoflurane and clinical timestamps, and the Cambridge propofol-sedation set adds behavioural responsiveness, but no open-data study has used either as the reference.

**Objective.** We will test whether EEG models transfer between propofol and sevoflurane and to an external behaviourally labelled dataset when the target is pharmacological exposure or clinical events rather than BIS, and whether six interpretable spectral features beat a CNN on that transfer.

**Methods.** From VitalDB's 6,388 cases we will select adult general-anaesthesia cases with BIS/EEG waveforms and SQI of at least 50 for 60% of maintenance, defining a propofol TCI arm and a sevoflurane arm. EEG will be cut into 4-s epochs with artefact gating; features include band powers, SEF95, spectral entropy, aperiodic exponent, alpha peak, alpha/delta ratio and burst-suppression ratio. Targets are lag-aligned BIS, normalised exposure (propofol Ce over 3 ug/mL; end-tidal sevoflurane over age-adjusted MAC), phase labels from anaesthesia timestamps, and Cambridge responsiveness in 20 volunteers. Feature models, a six-feature ridge model and a 1-D CNN will be trained with case-grouped 5-fold CV, then evaluated cross-arm without target-side tuning and on Cambridge frontal channels never used for fitting. The primary metric is prediction probability Pk per case with jackknife SE; secondary metrics are RMSE and concordance against BIS, calibration slope, AUROC for responsiveness and LOC/ROC timing error. Paired case-level bootstrap compares models and targets with Holm correction; mixed models test age-decade effects; nulls are case-label permutation and within-case circular time shifts.

**Expected results.** We expect BIS-reproduction models to lose at least 2 RMSE points across agents with half the loss explained by BIS-exposure relationships alone, exposure-target models to lose under 0.05 Pk across agents, the six-feature model to match the CNN within agent and exceed it out of distribution, Pk to decline with age with age inputs closing half the gap, and AUROC above 0.85 on Cambridge.

**Significance.** The study reframes agent transfer as a question about the reference standard and releases an open evaluation kit for VitalDB.

### Lifespan Normative Charts of Slow-Oscillation-Spindle Coupling Precision: Cross-Cohort Transportability and Incident Cardiovascular Disease in 12,000 Polysomnograms

*Project: [`sleep-spindle-aging-biomarker`](projects/sleep-spindle-aging-biomarker/)*

**Background.** Spindle timing relative to the slow-oscillation up-state predicts memory consolidation and decays with age, but nearly all coupling evidence comes from single-laboratory high-density EEG studies of fewer than 150 participants. Spindle density has lifespan norms from 11,630 NSRR participants and has been linked to incident coronary disease in SHHS. Whether coupling metrics survive two-channel clinical polysomnography, how they vary by age and sex, and whether models built on them transport across cohorts and montages is unknown.

**Objective.** We will build sex-specific lifespan centile charts of coupling precision and test whether coupling is a more transportable predictor of age, cognition and incident cardiovascular disease than spindle density.

**Methods.** One harmonized YASA pipeline will process Sleep-EDF Expanded (197 PSGs, 25-101 y), SHHS1 and SHHS2 (5,793 and 2,651 PSGs), MESA Sleep (2,056), MrOS Sleep (2,911 and 1,026), CFS (730 PSGs, 6-88 y) and open high-density datasets, with one canonical derivation per cohort resampled to 100 Hz. Per night we will compute coupled-spindle fraction, preferred phase, mean resultant length (MRL), Tort modulation index and an MRL z-score against random-position surrogates. Normative curves will use B-spline quantile regression at the 5th to 95th centiles with a GAMLSS sensitivity fit. Leave-one-cohort-out prediction of age and processing speed will compare coupling, density and combined features, with harmonization estimated on training cohorts only, subjects never split across folds, and paired subject-level bootstrap of absolute-error differences. Cox models in SHHS1, replicated in MESA, will relate MRL z-scores to adjudicated incident CVD adjusting for spindle density, AHI and cardiovascular risk factors, with Fine-Gray competing-risk analysis.

**Expected results.** We expect coupling precision to decline after roughly 40 years with a standardized slope ratio above 1.5 relative to spindle density; a leave-one-cohort-out MAE increase below 20% for coupling models versus above 35% for density models; within-subject cross-derivation correlation above 0.7 for coupling but below 0.5 for density; associations with Digit Symbol Coding and Trails B beyond density in MESA and MrOS; and a hazard ratio near 1.15 per SD of lower coupling for incident CVD.

**Significance.** The study will deliver an open centile calculator for thalamocortical timing, a transportability benchmark for sleep-EEG biomarkers, and the first test of whether coupling carries cardiovascular information beyond spindle count.

### Automated Sleep-Staging Error as Differential Misclassification: Propagating and Correcting N3 Bias in Cardiovascular Outcome Models Across NSRR Cohorts

*Project: [`sleep-staging-in-disease`](projects/sleep-staging-in-disease/)*

**Background.** Automated sleep stagers increasingly score epidemiological cohorts, after which N3 percentage enters outcome models as an exposure. Staging error is not uniform: recent multi-cohort evaluations show N3 performance degrades with apnoea-hypopnoea index (AHI). Because AHI, arousals, age and heart failure also cause cardiovascular outcomes, this is differential misclassification, which can bias hazard ratios in either direction. Regression calibration and SIMEX address this problem but have not been applied to AI-derived sleep exposures, and no study has reported what automated staging does to a published epidemiological estimate.

**Objective.** We will characterise where and for whom automated N3 errors occur, test whether they are differential with respect to cardiovascular risk, quantify the bias in hazard ratios and evaluate corrections using human scoring as a validation instrument.

**Methods.** Pretrained U-Sleep and YASA will be applied unchanged to SHHS1 (5,804 PSGs), SHHS2 (2,651), MESA Sleep (2,056), MrOS Sleep (about 2,900) and CFS (730), retaining per-epoch posteriors; a feature-based stager trained on Sleep-EDF Expanded (197 healthy recordings) versus SHHS1 will test source-domain effects. We will report per-stage sensitivity across AHI strata, error odds ratios within two epochs of scored respiratory events and arousals, and cluster-robust logistic regression of misclassification among true-N3 epochs on AHI, arousal index, age, sex, heart failure and beta-blocker use. Cox and logistic models for incident hypertension, CVD and all-cause mortality will be fitted with human, automated argmax and posterior-expected N3%, then corrected by regression calibration on a pre-registered 15% human-scored subset, SIMEX and multiple imputation. Splits are by participant, U-Sleep results on its NSRR training cohorts are labelled in-domain, and a classical-error simulation null and within-stratum label shuffling define non-differential references.

**Expected results.** We expect N3 sensitivity at least 0.15 lower at AHI 30 or more than below 5, with N3-to-N2 confusion clustered near respiratory events; an odds ratio above 1.2 for heart failure after AHI adjustment; automated exposure shifting the hazard ratio per 10-point N3% beyond the human-scored bootstrap interval; and corrected estimates recovering the human-scored value within its CI.

**Significance.** The study supplies a general recipe for AI-derived exposures in cohort studies, showing when staging bias changes an inference and how a small human-scored subset repairs it.

### Species or recording scale? A harmonised, scale-free slow-oscillation and spindle coupling pipeline applied to rodent LFP, human intracranial EEG and human scalp EEG

*Project: [`cross-species-spindle-coupling`](projects/cross-species-spindle-coupling/)*

**Background.** Nesting of sleep spindles in the slow-oscillation (SO) up-state is the central mechanistic claim of active systems consolidation and is assumed to translate from rodents to humans. Yet rodent evidence comes from depth LFP and human evidence mostly from scalp EEG, SO detectors use 0.5-4 Hz in rodents but 0.16-1.25 Hz in humans, and detector agreement is only moderate even within humans. Reported species differences therefore mix biology with recording scale and detector convention.

**Objective.** We will apply one detector with percentile-based thresholds and an individualised spindle band to rodent intracortical LFP, human intracranial EEG and human scalp PSG, to determine which canonical cross-species claims about SO-spindle coupling survive constant settings and how much of the apparent species difference is attributable to recording scale and detector choice.

**Methods.** Data: DANDI:000041 (rat frontal cortex, 11 rats), DANDI:000978 (rat prefrontal cortex, 8 rats), DANDI:000166 (mouse V1 laminar probes), the MNI Open iEEG Atlas (more than 100 patients) and Sleep-EDF Expanded (197 PSGs, ages 25-101), all resampled to 200 Hz. The spindle band is set from the aperiodic-corrected NREM spectrum; SOs are zero-crossing cycles in the top 25% of peak-to-peak amplitude; spindles use Hilbert-envelope percentile thresholds with duration limits in cycles. Metrics: mean SO phase at the spindle peak, event-count-matched mean resultant length (MVL), Tort's modulation index and offsets in SO cycles. A 24-specification multiverse (3 SO bands x 2 spindle-band rules x 2 threshold rules x 2 duration rules) yields specification curves. Mixed models with subject as a random effect, Holm within pre-registered families; nulls from event-count-matched random phases, circularly shifted spindle trains and wake-segment sham coupling.

**Expected results.** We expect the mean coupling phase to lie within 30 degrees of the SO peak in all five sources, with between-species spread smaller than between-detector spread within a species; MVL to be higher in depth recordings than scalp regardless of species; duration in cycles and offset in SO cycles to be conserved while seconds-based values diverge; the SO band to be the largest multiverse driver; and laminar phase shifts to vanish under local bipolar or CSD referencing.

**Significance.** The study separates biology from measurement in a foundational cross-species claim and delivers a reusable harmonised pipeline and a cross-species event catalogue.

### How Many Calibration Trials Does Transfer Learning Save? A Leave-One-Dataset-Out, Mixed-Effects Benchmark of Alignment, Fine-Tuning and EEG Foundation Models for Motor-Imagery BCIs

*Project: [`bci-decoder-transfer-benchmark`](projects/bci-decoder-transfer-benchmark/)*

**Background.** Motor-imagery brain-computer interfaces need per-user calibration, and transfer learning through covariance alignment, deep fine-tuning and EEG foundation models promises to shorten it. Existing benchmarks compare decoders at full calibration within a dataset, alignment methods have been compared on a handful of datasets, and average rankings hide per-subject heterogeneity. No benchmark uses every other dataset as the source pool with target calibration trials as the primary axis, none compares these families under identical budgets and montages, and negative-transfer rates are unreported.

**Objective.** We will estimate calibration savings, zero-shot floors, negative-transfer rates and their moderators for alignment, fine-tuning and frozen foundation-model embeddings in a leave-one-dataset-out design across about 18 open MOABB datasets and PhysioNet EEGMMIDB (109 subjects).

**Methods.** All datasets will be preprocessed identically (8-30 Hz, 128 Hz, 0.5-2.5 s post-cue) in three montage arms: native channels, a common 8-channel motor set, and a 32-channel interpolated montage. Transfer arms are target-only training, Euclidean alignment, Riemannian re-centering with MDM, tangent-space regression, Riemannian Procrustes analysis, EEGNet/ShallowConvNet fine-tuning, and frozen LaBraM/CBraMod embeddings with a linear probe. For each target subject, budgets of 0 to 80 trials and all trials are sampled stratified by class in ten draws and evaluated on held-out trials never in the source pool; learning curves yield trials-to-90%-of-asymptote. Alignment references use only calibration trials, and targets overlapping foundation-model pre-training corpora are excluded from those contrasts. A mixed model of balanced accuracy on method by log budget with random intercepts for dataset and subject, Holm-corrected contrasts per budget, Wilson intervals for negative-transfer rates, REML meta-regression on dataset moderators, and label-permutation and shuffled-source nulls complete the analysis.

**Expected results.** We expect Riemannian re-centering from a pooled source to reach 90% of asymptotic accuracy with at least 50% fewer target trials, zero-shot gains shrinking with montage overlap, no advantage of frozen foundation models below 20 trials, 15-30% of subjects harmed at low budgets, predictably from covariance distance and baseline accuracy, and moderators explaining over 30% of between-dataset variance.

**Significance.** A versioned, budget-keyed leaderboard gives BCI developers a realistic estimate of what transfer buys.

### Would You Have Been Rewarded? A Specification-Curve Re-Analysis of the Feedback Signal in Open EEG Neurofeedback Datasets

*Project: [`neurofeedback-multiverse`](projects/neurofeedback-multiverse/)*

**Background.** In EEG neurofeedback the number a participant sees is the product of at least seven online engineering choices that papers rarely report in full, and about a third of participants are labelled non-learners. Multiverse analyses show that EEG preprocessing choices change conclusions for ERPs and decoding, but no study has recomputed the closed-loop feedback signal itself, the reward decisions it drove, or the learner labels derived from it across a specification space, and no estimate exists of how much delivered feedback was artifact.

**Objective.** We will recompute the feedback signal of open neurofeedback datasets under every defensible pipeline and quantify how many reward decisions would have flipped, how stable learner status is, how much feedback variance artifact regressors explain, and whether feedback–target contingency predicts learning.

**Methods.** We will use OpenNeuro ds002336 and ds002338 (10 and 20 participants; EEG-fMRI motor-imagery neurofeedback with shipped online scores) and ds005846 and ds005878 (parietal-alpha down-regulation in immersive VR). After fixed pre-steps (MR-artifact correction, resampling, block timing from events.tsv), a pre-registered space of reference, band, estimator, window, normalisation, artifact handling and smoothing yields 2,160 specifications, with each documented online pipeline flagged as one point. Reward decisions will be compared pairwise across specifications and decomposed by factor; learner labels by pairwise kappa and unstable fraction with participant bootstraps. Ocular, muscular and broadband regressors give feedback specificity; contingency between delivered feedback and a Laplacian, EOG-regressed offline estimate is related to learning slope in mixed models with dataset effects. A joint sign-flip specification-curve test per dataset and band, with beta as control band, block-order shuffling as slope null and zero-gain synthetic sessions as false-positive check, supports inference.

**Expected results.** We expect median reward-decision agreement below 0.80, at least 25% of participants changing learner status with mean kappa below 0.6, artifact regressors explaining at least 20% of feedback variance under recording-reference pipelines, a stable sign but more than twofold magnitude variation of the learning slope, contingency predicting learning at rho above 0.3, and lower agreement for down- than up-regulation protocols.

**Significance.** The result bounds the dose of contingent reinforcement actually delivered, with a tool any laboratory can run on raw data.


## Spikes, Neuropixels & human single units

### Is representational drift ordered thalamus, cortex, hippocampus? A quality-controlled Neuropixels benchmark across three open datasets

*Project: [`neuropixels-representational-drift`](projects/neuropixels-representational-drift/)*

**Background.** Neural responses to identical stimuli change over minutes to weeks even when behaviour is stable. Theory predicts it should be smaller near sensory input, yet no study has compared drift across cortex, thalamus and hippocampal formation in the same animals under matched stimuli. Existing estimates also ignore unit-quality confounds, which differ between areas and can masquerade as biology.

**Objective.** We will test whether within-session drift is ordered thalamus (LGd, LP) < visual cortex < hippocampal formation (CA1, DG, SUB) in simultaneously recorded populations, how much of the naive area difference is explained by unit quality, and whether linear population readouts are shielded from single-unit drift differently across areas.

**Methods.** We will stream NWB sessions from the Allen Visual Coding Neuropixels dataset (58 sessions, about 100k units), Allen Visual Behavior Neuropixels (about 150 sessions) and the IBL Brain-wide Map (459 sessions, 139 mice, about 76k good units). Per session and area group we will compute population-vector correlation versus time lag, RDM stability, per-unit tuning correlation and a cross-time decoder drift index (train early, test late, relative to within-block cross-validated accuracy). Every statistic is reported against a time-shuffle null and a Poisson rate-matched surrogate that has no drift by construction. Area contrasts use session-by-area as the unit, mixed models with session and mouse random intercepts and quality covariates, plus quality-matched unit resampling; Holm correction across area pairs; session-level bootstrap CIs. Cross-session geometry compares RDMs across different mice, so no unit-level leakage is possible.

**Expected results.** We expect null-corrected drift ordered thalamus < cortex < hippocampus, at least 30 percent of the raw area difference to vanish after quality stratification with a residual that survives the mixed model, cross-time decoders to lose under 15 percent relative accuracy in cortex and thalamus but over 25 percent in hippocampus, and stimulus dependence (movies > gratings) in cortex but not thalamus. In IBL, early-to-late generalisation of stimulus-side and choice decoders should be highest in sensory thalamus and superior colliculus.

**Significance.** This is the first cross-area, cross-dataset drift benchmark with explicit confound modelling and shared nulls, separating a brain-wide property from a spike-sorting artefact.

### Which brain-wide decoding claims replicate across laboratories? Lab-as-random-effect reproducibility of task-variable decoding in the IBL Brain-wide Map

*Project: [`ibl-brainwide-decoding`](projects/ibl-brainwide-decoding/)*

**Background.** The International Brain Laboratory (IBL) Brain-wide Map reports per-region decoding of choice, stimulus, block prior and reward from about 700 Neuropixels insertions from 12 labs, and a companion study found comparable decodability across labs at one repeated trajectory in five regions. Whether that holds for about 280 regions is unknown: no analysis has partitioned between-session variance in decoding scores into lab, subject and unit-yield components, tested how many claims rest on one lab, or asked whether visual-region rankings transfer to the Allen survey. Foundation-model benchmarks on these data address transfer to unseen animals, not labs.

**Objective.** We will flag each BWM decoding claim as robust across labs, lab-dominated or yield-dominated, and test whether visual-region information rankings survive a change of protocol and consortium.

**Methods.** Using the 2025 re-sorted release (459 sessions, 699 insertions, 139 mice, 12 labs) we will assign good units to Beryl regions, bin spikes at 20 ms in target-specific windows, and fit L2-regularised logistic and ridge decoders per insertion and region with the penalty chosen inside contiguous-block cross-validation folds. Nulls are trial-label shuffles, IBL pseudo-sessions and imposter sessions, giving null-corrected scores. Regions with at least three insertions from at least two labs enter random-effects meta-analysis and mixed models with lab as group and subject as a variance component, yielding lab intraclass correlations with bootstrap intervals, leave-one-lab-out fragility flags, and covariates for unit yield, QC, drift, trial count and behaviour. Subsampling to matched unit counts tests whether lab differences persist. Stimulus decoders harmonised over the first 250 ms will be trained and tested in both directions between IBL and Allen Visual Coding (58 sessions) for 11 visual, thalamic and hippocampal regions. Region lists and model formulae will be pre-registered; BH-FDR applies across regions.

**Expected results.** We expect lab ICC below 0.2 for most regions with significant choice or stimulus decoding; fragile regions to be enriched for fewer than 10 good units per insertion; lab variance to fall by at least half at matched yield; block-prior decoding to be the least reproducible target; and visual-region rank order to be preserved between datasets (Spearman rho above 0.7) while absolute scores differ.

**Significance.** The per-region reproducibility table and yield-corrected brain map give the field a calibrated prior on which encoding claims to build on.

### Do Cortical Layers Occupy Distinct Population Subspaces? Laminar Latent Geometry in Mouse Visual Cortex with Depth-Shuffled and Unit-Matched Nulls

*Project: [`laminar-latent-dynamics`](projects/laminar-latent-dynamics/)*

**Background.** Population analyses of mouse visual cortex typically pool layers into one area and describe a single low-dimensional state. Recent macaque work reports inter-laminar communication in V1 and its modulation by spatial context. Whether, within one mouse column, layers share latents or occupy distinct subspaces, which layer's variance is stimulus- versus movement-driven, and whether inter-laminar interaction is low-rank has not been tested at scale; no laminar Neuropixels study reports sensitivity to layer-assignment error.

**Objective.** We will estimate the overlap between layer-specific latent subspaces, partition each layer's latent variance into stimulus and behavioural components, measure communication dimensionality between layers, and test state, hierarchy and assignment-robustness effects across three datasets.

**Methods.** From the Allen Visual Coding Neuropixels survey (58 sessions), the Visual Behavior survey (~150 sessions, active change detection) and DANDI:000166 (chronic laminar V1 recordings), L4 will be anchored at the earliest flash-evoked CSD sink and layers assigned from thickness priors, with CCF labels, spike-power landmarks and +/- 50 um jitter as sensitivity arms. Fifty-ms square-root-transformed counts yield cross-validated factor-analysis dimensionality per layer, principal-angle overlap between layer subspaces, reduced-rank regression communication rank, and ridge encoding models with stimulus, running, pupil and face-motion regressors, all cross-validated in contiguous time blocks. Nulls: depth shuffling (1,000 permutations per column), unit-count-matched subsampling (100 repeats) and a split-unit ceiling. Mixed-effects models with session as a random effect, Holm correction within hypothesis families and a trend test across hierarchy scores for six visual areas.

**Expected results.** We expect partial segregation: overlap below the depth-shuffle null for every pair and lowest for L4 versus L5/L6; stimulus-unique variance highest in L4 and behaviour-unique variance highest in deep layers and L2/3; communication rank below within-layer dimensionality and smallest for L4 to L2/3; higher overlap during locomotion and spontaneous activity; segregation decreasing up the hierarchy; and conclusions stable across assignment arms.

**Significance.** A replicated, null-controlled account of laminar subspace geometry would tell modellers whether a column is one dynamical system or several coupled ones, and set a reporting standard for laminar Neuropixels analyses.

### Decomposing the two-photon versus Neuropixels tuning gap into sampling, measurement and analysis with the Allen Brain Observatory

*Project: [`cross-modality-tuning-reproducibility`](projects/cross-modality-tuning-reproducibility/)*

**Background.** The Allen Institute's matched-stimulus comparison of two-photon imaging and Neuropixels found that stimulus preferences agree across modalities while responsiveness and selectivity differ. The contributions of sampling (layer, cell class, firing rate), measurement (indicator kinetics, nonlinearity, deconvolution) and analysis (metric definitions, inclusion criteria) have never been estimated in one framework, agreement has not been compared with test-retest reliability, and the Visual Behavior datasets, which repeat the design in an active task, have not been used to test transfer.

**Objective.** To decompose the modality gap in tuning statistics into sampling, measurement and analysis terms with a reliability ceiling, test whether the decomposition transfers from passive viewing to an active task, and identify modality-invariant metrics.

**Methods.** From Allen Visual Coding 2-photon (about 1,300 experiments, 60,000 neurons) and Visual Coding Neuropixels (58 sessions, 40,000 units), we will compute responsiveness, OSI/DSI, preferences and lifetime sparseness from dF/F, Allen events, deconvolved rates and spike counts. Propensity weights on layer, cell class and firing rate will estimate the sampling term. A GCaMP6 forward model with supralinear amplification applied to Neuropixels spikes, followed by the identical 2-photon pipeline, will estimate the measurement term; a factorial multiverse of signal type, neuropil rule, QC, metric and response window will estimate the analysis term. The gap will be attributed sequentially, averaged over all orderings with bootstrap CIs. Split-half and test-retest reliabilities will disattenuate per-area agreement. Distribution differences use Wasserstein distance against within-modality bootstrap nulls, and forward-model parameters and invariance thresholds fixed on Visual Coding will be applied unchanged to Visual Behavior 2-photon and Neuropixels.

**Expected results.** We expect preference distributions to agree within the null while magnitude distributions differ; sampling reweighting and forward modelling to each close at least 30% of the responsiveness gap; fraction-selective claims to shift by over 15 points across the multiverse; disattenuated agreement above 0.8 for preferences and 0.5 for magnitudes; and reliability-normalised, deconvolution-based metrics to meet the 0.2 SD invariance criterion.

**Significance.** The study would turn a documented discrepancy into an attributed budget and a checklist of tuning statistics that survive a change of recording modality.

### How much of the concept cell is the criterion? A multiverse and cross-dataset audit of selectivity in open human single-neuron recordings

*Project: [`human-concept-cell-reproducibility`](projects/human-concept-cell-reproducibility/)*

**Background.** Concept cells, medial temporal lobe (MTL) neurons responding selectively to a specific person, object or place, anchor the dominant single-neuron account of human memory, and reviews cite prevalences such as a third of MTL units. Each figure rests on one laboratory's selection statistic, alpha, window, null model and spike sorter. Open NWB datasets on DANDI now allow several published criteria to be applied to the same neurons, yet no study has done so, used a null that preserves rate drift and trial blocking, reported cross-validated selectivity, modelled non-independence of units within wires, sessions and patients, or tested whether sorting quality predicts selection.

**Objective.** We will quantify how much reported concept-cell prevalence, selectivity and regional distribution depend on criterion, null model and sorting quality rather than on the brain.

**Methods.** DANDI:000004 (recognition memory, about 1,500 units), DANDI:000469 (working memory; 1,809 units, 41 sessions, 21 patients, MTL and medial frontal cortex) and an open object-recognition corpus will be loaded through a harmonised NWB loader. Five criteria will be applied to every unit: one-way ANOVA, Kruskal-Wallis, Holm-corrected bin-wise rank-sum against baseline, a response-strength index (z at least 3, ratio at least 2) and a Poisson GLM with a drift spline. Each will be evaluated under label-permutation, circular-shift and block-permutation nulls with at least 1,000 permutations per unit, after calibrating false-positive rates on synthetic sessions with matched trial counts, drift and block structure. Selection bias will be measured by split-half cross-validation across trials. Prevalence will be estimated with logistic mixed models carrying session, patient and wire random intercepts and sorting-quality covariates, with cluster-bootstrap intervals, Cochran's Q and criterion-by-region interaction tests. Windows are fixed from the dataset papers.

**Expected results.** We expect MTL prevalence in the same dataset to vary more than twofold across criteria; drift-preserving nulls to remove at least 20 percent of nominal detections in blocked designs; cross-validated selectivity to be at least 30 percent smaller than in-sample; the MTL-over-frontal gradient to survive while the hippocampus-amygdala ordering depends on criterion; and selected units to have higher SNR.

**Significance.** The audit delivers calibrated false-positive rates per criterion and a harmonised units table with all criterion flags, turning concept-cell prevalence into a reproducible quantity.

### Cell-type composition or within-type expression? Decomposing gene-gradient predictions of neural timescales and tuning across the mouse visual hierarchy

*Project: [`gene-gradients-neural-timescales`](projects/gene-gradients-neural-timescales/)*

**Background.** Intrinsic neural timescales lengthen along cortical hierarchies, and Allen ISH expression patterns have recently been shown to predict single-neuron timescales brain-wide in one Neuropixels dataset. That work used one modality and one target, and did not ask whether the prediction reflects regional differences in cell-type proportions or molecular differences within cell types. Cell-resolved ABC Atlas MERFISH data and three open Neuropixels surveys now allow the two to be separated.

**Objective.** We will test which gene families predict which tuning properties, decompose each prediction into a cell-type-composition part and a within-type residual, test whether it holds within cortical layers, and replicate a frozen model across surveys.

**Methods.** Unit-level targets will be computed from Allen Visual Coding Neuropixels (58 sessions, about 100,000 units, CSD-based layers): intrinsic timescale from spike-count autocorrelation with three estimators, plus latency, adaptation, temporal-frequency preference, orientation selectivity and receptive-field size. Gene maps come from Allen ISH expression energy and from MERFISH mean expression and subclass proportions per area and per area-by-layer. PLS and ridge models with nested leave-one-area-out cross-validation predict area medians. The decomposition fits target on subclass proportions, then residuals on genes, reporting out-of-sample R^2 for each step. Nulls comprise 3-D variogram surrogates over CCF centroids, hierarchy-stratified permutations and gene-ensemble nulls matched on expression level and Moran's I, with at least 5,000 surrogates and BH-FDR. Recovering the Harris hierarchy score from expression (R^2 above 0.5) is the positive control. After pre-registration the frozen model is applied once to Visual Behavior Neuropixels (about 150 sessions) and IBL visual-area units.

**Expected results.** We expect one PLS component to carry timescale and latency while temporal-frequency preference and adaptation load on a second component enriched for HCN and Kv genes; composition to explain at least 60 percent of cross-area timescale variance with within-type expression adding under 10 percent; the association to weaken by at least half within layers except L5; replication R^2 within 0.1 of in-sample; and ensemble nulls to leave fewer than 3 of more than 10 naive enrichment hits.

**Significance.** The study determines whether "genes predict timescales" is a statement about cell-type proportions or molecular tuning, which bears on how human transcriptomic-gradient results should be read.

### Dendritic Morphology Versus Channel Composition as Determinants of Extracellular Spike Waveforms on High-Density Probes

*Project: [`morphology-to-spike-waveform`](projects/morphology-to-spike-waveform/)*

**Background.** Extracellular spike shape depends on transmembrane currents and on the morphology carrying return currents, and high-density probes record footprints across tens of sites. Opto-tagged waveform classifiers reach high accuracy but cannot explain why a class has its footprint, and many opto-tagged interneurons are not narrow-spiking. Simulations showing that morphology shapes the waveform used a few cells and one probe, so no population-scale variance partition between morphology and channels exists.

**Objective.** We will quantify, for each waveform feature, the variance attributable to dendritic morphology versus ion-channel densities, identify the morphological features that carry it, test probe-geometry dependence, validate simulated class footprints against opto-tagged units, and derive morphology-conditioned priors.

**Methods.** Using roughly 600 mouse and 150 human Allen Cell Types reconstructions with perisomatic and all-active models, and 9,200 all-active model ensembles for 230 cells, we will run a factorial design crossing each morphology with its own and transplanted same- and different-class channel sets, dropping transplants that fail to spike. NEURON compartment currents will be projected through a line-source forward model onto Neuropixels 1.0, 2.0 and Ultra geometries over a 10–100 µm placement grid. Features include trough-to-peak, half-width, footprint, decay exponent, asymmetry and propagation velocity. A sum-of-squares decomposition with bootstrap CIs over cells yields eta² for morphology, channels, interaction and placement; ridge and gradient-boosted regression from morphology to waveform use grouped five-fold CV by cell with a shuffled-feature control. Simulated class distributions will be compared with opto-tagged Pvalb, Sst and Vip units from Allen Visual Coding and Neuropixels Ultra recordings by energy distance and permutation tests, with BH-FDR across features.

**Expected results.** We expect spatial features to be more than 60% morphology-driven and temporal features more than 60% channel-driven, dendritic surface area within 100 µm and stem count to predict footprint with cross-validated R² above 0.5, morphology-explained fractions to rise from NP1.0 to NP Ultra, simulated footprint medians to match opto-tagged medians within 10 µm, and priors to add at least five points of class accuracy.

**Significance.** A per-probe morphology-to-waveform lookup table would give Neuropixels laboratories mechanistic priors for cell-type identification and a plausibility check on sorted units.


## Cellular morphology, connectivity & atlases

### Batch-Corrected, Phylogenetically Aware Dendritic Scaling Laws Across the NeuroMorpho.Org Corpus

*Project: [`neuromorpho-scaling-laws`](projects/neuromorpho-scaling-laws/)*

**Background.** Wiring-optimality theory predicts that total dendritic length scales with branch-point number and spanning volume as a power law with a 2/3 exponent for three-dimensional arbors and 1/2 for planar ones, confirmed on about 6,000 curated reconstructions. The original tests did not model the tracing laboratory as a random effect, two-species comparisons are largely single-consortium, confounding lab and species, no neuromorphology study has used phylogenetic comparative methods, and a reconstruction's effective dimensionality, which sets the predicted exponent, is itself a protocol property.

**Objective.** We will estimate allometric exponents across the full NeuroMorpho.Org corpus with archive as a random effect and reconstruction protocol covariates, quantify how much apparent between-species variance is absorbed by provenance, and test for phylogenetic signal in batch-adjusted species residuals.

**Methods.** We will harvest metadata and L-Measure morphometry for more than 250,000 reconstructions from roughly 1,000 laboratories and over 100 species, restrict to complete dendritic reconstructions with at least 20 branch points, remove duplicates, and recompute length, branch points, hull volume, Sholl profiles and PCA-based effective dimensionality from CNG SWC files. Mixed models of log length on log branch points, log volume, dimensionality, batch covariates, species, region and class with an archive random intercept yield exponents tested by Wald and equivalence tests (±0.05), with leave-lab-out validation. Species attenuation is one minus the ratio of adjusted to naive species-effect variance, identified by multi-species labs and multi-lab species, with a null permuting species labels across archives. Batch-adjusted species means will be regressed on brain mass by weighted PGLS under a TimeTree-derived Brownian covariance with Pagel's lambda and a tip-shuffle null. Region, class, axon and within-lab software contrasts are BH-corrected.

**Expected results.** We expect pooled exponents to recover 2/3 and 1/3 for three-dimensional and 1/2 and 1/2 for near-planar arbors, at least 30% of naive between-species variance to be absorbed by provenance, lambda above zero with a PGLS slope differing from OLS, conserved exponents for principal cells with deviations for interneurons and axons, and reduced z-extent in shrinkage-uncorrected reconstructions.

**Significance.** A citable table of provenance bias magnitudes and a corpus-wide batch-adjusted morphometrics release would let cross-species claims be stated net of who traced the cell.

### Graph Neural Networks on Dendritic Trees Predict Intrinsic Electrophysiology and Transfer From Mouse to Human

*Project: [`morphology-to-electrophysiology`](projects/morphology-to-electrophysiology/)*

**Background.** The Allen Cell Types Database pairs standardised patch-clamp features with SWC reconstructions and, through Patch-seq, transcriptomic types for mouse and human cells. Existing predictive work uses hand-crafted morphometrics on mouse cells only, graph representations are evaluated on cell-type classification, not regression, cross-species transfer exists only for electrophysiology-to-transcriptomics mapping, and biophysical models have never served as a predictive baseline.

**Objective.** We will test whether a graph neural network on the SWC tree regresses intrinsic electrophysiological features, whether the mapping carries information within transcriptomic types, whether it transports from mouse to human, and whether it beats type-average biophysical models on each cell's own morphology.

**Methods.** We will assemble on the order of 2,000 mouse and 400 human Allen cells with electrophysiology, several hundred of each with reconstructions, with input resistance, sag, time constant, rheobase, upstroke/downstroke ratio, adaptation, f-I slope and AP half-width recomputed from NWB sweeps. Each reconstruction becomes a graph with 15 node features; a four-layer GraphSAGE regressor with five-seed ensembles will be trained with rotation, jitter and subtree-dropout augmentation and optional self-supervised pre-training on NeuroMorpho graphs. Baselines are ridge on 25 morphometrics, a size-only and a type-mean model, all under GroupKFold by donor or animal with inner-fold tuning. Transfer is evaluated zero-shot and with head fine-tuning on 10, 25 and 50 human cells over 20 subsets; the biophysical baseline simulates the long-square protocol in NEURON with type-average parameters on each morphology. Label permutation (500 per target), 1,000 cell-level bootstraps and Benjamini–Hochberg correction support inference.

**Expected results.** We expect out-of-fold R² above 0.3 for passive features, exceeding ridge by more than 0.05, with active features below 0.2; within-type partial R² above 0.1 for passive features only; zero-shot retention of more than half the within-species R² for input resistance and time constant after size normalisation but not for sag or AP width; biophysical models matching the GNN on mouse yet transferring worse to human; and seed-stable attributions on proximal dendrites.

**Significance.** A validated mapping with stated transfer limits would let electrophysiology be imputed for reconstruction-only and EM-derived cells and show which human neuron properties are geometric rather than channel-driven.

### How Well Do Bulk-Tracer Projection Maps Predict Single-Axon Targets? A Brain-Wide Concordance Benchmark with a Sampling-Corrected Projection Heterogeneity Index

*Project: [`mesoscale-vs-single-axon-connectivity`](projects/mesoscale-vs-single-axon-connectivity/)*

**Background.** The Allen Mouse Brain Connectivity Atlas parameterises most mouse network models, but bulk tracing averages over hundreds of neurons and cannot say whether targets are reached by the same axons or by parallel subpopulations. Single-neuron reconstructions registered to CCFv3 show idiosyncratic target subsets, and one study compared bulk and single-neuron callosal projections. No brain-wide per-region predictive benchmark exists, no heterogeneity metric has an independent-sampling null or rarefaction, and the consequences for network models are unquantified.

**Objective.** We will measure, for each source region with >= 10 reconstructed neurons, how well the bulk projection vector predicts individual axons' targets, whether neurons cluster into motifs beyond random sampling, how many neurons recover the bulk map, and how single-neuron weights alter simulated dynamics.

**Methods.** Bulk vectors are normalised projection volumes averaged over wild-type Allen experiments (injection fraction >= 0.5) per structure, split ipsi/contra. Axons from MouseLight (>1,000 neurons), SEU-ALLEN (1,741 and later releases) and ION projectomes are mapped to CCFv3 at 25 um, collapsed to summary structures, weighted by axon length or terminal count and binarised. Per neuron we compute Jaccard, AUROC and precision at 5 with the bulk vector as score; per region, pairwise and nearest-neighbour Jaccard distance, rarefied, with neuron-level bootstrap CIs. The primary null draws each neuron's target count independently with probabilities proportional to the bulk vector; label permutation and wrong-region bulk vectors give floor and specificity checks. Subsampling curves give n_50 per region; Virtual Brain simulations compare bulk, single-neuron and heterogeneity-scaled weights by normalised FC difference and hub-rank Kendall tau.

**Expected results.** We expect mean per-neuron AUROC > 0.7 with precision at 5 below 0.5; nearest-neighbour distances below the independent-sampling null (z < -3) in most cortical regions, larger deficits for L5 ET and L2/3 IT neurons than thalamic relays, and rare z > 3 regions at known anti-correlated pathways; rho > 0.8 with 30-100 pooled neurons, scaling with heterogeneity; FC differences > 0.2 and hub-rank tau < 0.8 for heterogeneous regions; and lower terminal- than length-based concordance near white matter.

**Significance.** A per-region table of predictability and null-corrected heterogeneity would tell modellers where the mesoscale connectome can be trusted as an edge weight and where it hides parallel channels.

### Reconstruction Software Leaves a Fingerprint: Provenance Detectability and Harmonisation in NeuroMorpho.Org

*Project: [`neuromorpho-software-fingerprinting`](projects/neuromorpho-software-fingerprinting/)*

**Background.** NeuroMorpho.Org curates more than 250,000 reconstructions from about 1,000 laboratories, and analyses built on it assume that geometry reflects the neuron rather than the tracing pipeline. Laboratories are known to differ more than anatomical classes, but no archive-wide estimate exists of how identifiable software or laboratory is from SWC geometry alone, no separation of sampling artefacts from morphometric distortion, and no harmonisation framework of the kind standard in single-cell genomics.

**Objective.** We will quantify provenance leakage as a detectability index, decompose it into removable sampling signal and residual morphometric signal, calibrate it on BigNeuron, where the same images were traced by dozens of algorithms, audit its inflation of cell-type classification accuracy, and test whether harmonisation removes it without erasing biology.

**Methods.** We will harvest NeuroMorpho metadata and CNG SWC files stratified by software and archive with caps, extracting morphometrics and sampling-fingerprint features (inter-node spacing, coordinate precision, radius uniqueness, z-step quantisation, collinearity). Gradient-boosting classifiers under GroupKFold by archive with nested tuning will predict software, and archive within software, reporting balanced accuracy and a detectability index against 500 archive-level label permutations. A resampling arm interpolates trees to 1 µm spacing, rounds coordinates and drops radii. On BigNeuron, an algorithm classifier with image-level grouping will be transferred to NeuroMorpho subsets with matching software labels. Cell-type tasks within one species and region will be evaluated under random versus leave-archive-out CV. ComBat with software or archive as batch and biology as covariates, plus a kBET-style mixing test, will be judged by residual detectability and by preservation of apical/basal, species and class effect sizes in the single-pipeline Allen Cell Types reference.

**Expected results.** We expect a software detectability index above 0.3 and above-chance archive prediction within software, a fall of more than 50% after resampling with signal remaining above null, above-chance transfer from BigNeuron, a cell-type accuracy drop under leave-archive-out CV correlated with detectability, and near-null residual detectability after ComBat with preserved reference contrasts.

**Significance.** A per-record provenance risk score and a validated harmonisation recipe would make lab-held-out validation and batch correction standard practice for downstream users of archived morphologies.

### A method-annotated database of dendritic spine densities mined from open-access literature, with meta-regression of methodological variance and calibration against dense electron-microscopy connectomes

*Project: [`dendritic-spine-metadata-mining`](projects/dendritic-spine-metadata-mining/)*

**Background.** Spine density is among the most reported neuroanatomical measures, yet it depends strongly on method: Golgi impregnation hides spines behind the shaft, fluorescence microscopy is diffraction-limited, and serial EM resolves all spines. Disease meta-analyses pool studies without modelling method effects, and method comparisons exist only within single papers. Dense EM connectomes (MICrONS, H01, Kasthuri) now provide exhaustive counts. How often NeuroMorpho.org reconstructions encode spines is unknown.

**Objective.** We will build a method-annotated database of control-condition spine densities, estimate how much between-study variance is methodological, calibrate light-microscopy values against EM counts on matched cells, and audit spine annotation across NeuroMorpho.org.

**Methods.** Europe PMC open-access full text will be searched; a regex grammar will extract spine-density values, units, dispersion and context (species, region, cell type, compartment, method, imaging modality, shrinkage correction, distance from soma, condition) with sentence-level provenance. Two curators will annotate 200 random statements for precision, recall and kappa. The meta-regression models log density on method, imaging, shrinkage, species, region, compartment, distance bin and year with a lab random effect, reporting method contrasts as log ratios with 95% CIs, tau2, I2 and between-study R2; a permutation null reassigns method labels within species-by-region strata. EM calibration will count spine synapses per um of skeleton on EM-reconstructed L2/3 and L5 pyramidal dendrites by compartment, compared with meta-regression predictions with bootstrap CIs over cells. Every NeuroMorpho SWC file (250k+ reconstructions) will be scanned for custom type codes and short terminal segments, cross-tabulated by archive, software, species and year.

**Expected results.** We expect method, imaging and shrinkage correction to explain at least 30% of between-study variance; Golgi densities to be 1.3-2.0 times lower than fluorescence values and lower still than EM; literature values to underestimate exhaustive EM counts, with a smaller gap for two-photon and STED studies; fewer than 5% of NeuroMorpho reconstructions to carry spine annotations; and reporting of compartment, shrinkage and animal-level n to remain below 50% of papers.

**Significance.** A released, provenance-linked spine-density table with method-adjusted normative values gives disease meta-analyses calibrated priors and quantifies how much of the spine literature is measurement rather than biology.

### Are dendritic disease signatures reproducible across laboratories? A within-archive meta-analysis of Alzheimer's-model, aging and epilepsy reconstructions on NeuroMorpho.org with electrotonic propagation

*Project: [`disease-morphology-signatures`](projects/disease-morphology-signatures/)*

**Background.** Dendritic changes in Alzheimer's-model mice, aged primates and epileptic granule cells are each argued to change excitability, yet almost every claim rests on one laboratory, one protocol and 10-40 cells per group. NeuroMorpho.org curates experiment condition, archive, software and shrinkage correction for more than 1,300 conditions, and most disease datasets were deposited with their own controls, so contrasts can be estimated within archive and pooled. No cross-archive disease meta-analysis of reconstructions exists, the AD signature has not been tested against normal aging, and effect sizes have not been propagated to electrotonic quantities.

**Objective.** We will estimate lab-effect-corrected pooled effect sizes of AD models, aging and epilepsy on dendritic morphometrics, test whether the AD signature is separable from aging, and propagate pooled effects through passive cable models.

**Methods.** Condition-labelled records will be harvested through the NeuroMorpho API and mapped to a version-controlled vocabulary, with same-archive controls matched on species, region and cell class; at least 5 cases and 5 controls per archive contrast. Twelve morphometrics (length, bifurcations, tips, branch order, extents, hull volume, Sholl measures, diameter) will be recomputed from SWC files with one definition for every archive. Hedges' g per archive, condition and morphometric will be pooled by REML random effects with Hartung-Knapp-Sidik-Jonkman intervals, Q, I2 and tau2, and meta-regressed on species, region, cell class, age, shrinkage, software and n. Condition signatures will be compared by cosine similarity against 1,000 within-archive label permutations. Passive cable models will yield input resistance and tip-to-soma attenuation, pooled identically. Egger regression and leave-one-archive-out influence follow; one primary morphometric per hypothesis, Benjamini-Hochberg across the rest.

**Expected results.** We expect a negative pooled g for total length and branch points in AD models with the CI excluding zero and I2 above 50%; AD and aging signatures more similar than the permutation null but differing at distal Sholl radii; opposite-signed epilepsy effects in granule and pyramidal cells; input resistance changed by more than 10% in AD but not epilepsy models; and larger effects in smaller and shrinkage-uncorrected datasets.

**Significance.** The study provides the first reproducibility and funnel-plot evidence for disease morphology and electrotonic effect sizes that carry laboratory heterogeneity into functional predictions.

### Does Interneuron Morphology Alone Carry Transcriptomic Identity Across Species? A Mouse-to-Human Patch-seq Transfer Benchmark Separating Shape from Laminar Position

*Project: [`interneuron-morphology-ttype-transfer`](projects/interneuron-morphology-ttype-transfer/)*

**Background.** Patch-seq links morphology to transcriptomic type in mouse visual and motor cortex and human middle temporal gyrus; a 2026 study transferred electrophysiology-to-subclass classifiers from mouse to human, but no study has trained a morphology-only subclass classifier in one species and tested it in the other under a harmonised taxonomy, separated positional (depth, layer) from shape information, or asked which representation survives the scale and laboratory shifts that break transfer.

**Objective.** We will quantify how much harmonised subclass identity (Pvalb, Sst, Vip, Lamp5, Sncg/PAX6) is recoverable from morphology alone within species, how much of it is laminar position, and how well it transfers from mouse to human, comparing morphometrics, depth-normalised density maps, persistence summaries, scale-normalised versions and a self-supervised graph embedding.

**Methods.** We will assemble ~2,000 reconstructions from the Allen mouse V1 Patch-seq set (~500 with morphology), the Tolias-lab mouse M1 mini-atlas (~1,300) and the Allen human MTG interneuron set (hundreds), mapped to five subclasses via the cross-species consensus taxonomy. Logistic-regression and random-forest classifiers will be evaluated by stratified 5-fold CV within dataset, leave-dataset-out across the three sets, and train-on-mouse, test-on-human with per-dataset standardisation and CORAL alignment. The primary metric is balanced accuracy over subclasses, with macro-F1 and calibrated log-loss. Nulls: within-dataset label permutation (1,000 draws), permutation of training labels for transfer, and a position-only (depth plus layer) floor. Ablations remove positional features and compare absolute with scale-normalised representations; the two mouse datasets bound the laboratory shift; dendrite-only and dendrite-plus-axon subsets are analysed separately.

**Expected results.** We expect within-species balanced accuracy above 0.6 in mouse and 0.5 in human, a mouse-to-human drop of at least 0.15 that alignment halves, a within-species cost under 0.1 when position is removed but a changed transfer gap, smaller loss between mouse datasets than between species, and a representation ranking that is not shift-invariant (Kendall's tau < 0.5), with learned embeddings winning within species and normalised shape features transferring best.

**Significance.** The study delivers the morphology counterpart of cross-species electrophysiology transfer, a decomposition of positional versus shape information, and a public feature table for ~2,000 Patch-seq interneurons in one taxonomy.

### Multi-Site, Multi-Modal Validation of Connectome-Based Whole-Mouse-Brain Models Against 17-Site Resting-State fMRI and Widefield Calcium Imaging Under Spatially Informed Nulls

*Project: [`virtual-mouse-brain-validation`](projects/virtual-mouse-brain-validation/)*

**Background.** Virtual-Mouse-Brain models build whole-brain dynamics from the Allen mesoscale connectome and have been validated against functional connectivity (FC) from one or two laboratories, usually under anaesthesia with a single connectome construction. Multi-centre mouse fMRI and awake widefield calcium imaging now exist in the same Common Coordinate Framework. No study has asked whether model fit is a property of the model or of one lab's acquisition, whether models fitting BOLD FC also fit calcium FC, how much spatial embedding alone explains, or which construction choices matter.

**Objective.** We will fit one family of connectome-based models across 17 fMRI sites and awake widefield sessions, benchmark fits against degree- and distance-preserving null connectomes, and quantify how construction and model choice affect fit.

**Methods.** Connectomes will be built from the Allen Mouse Brain Connectivity Atlas as a multiverse over density versus strength, contralateral handling, symmetrisation, thresholds of 5-50% and a MouseLight single-axon alternative. Empirical FC will come from OpenNeuro ds001720 (255 subjects, 17 sites) preprocessed with RABIES to CCF space, and from resting widefield GCaMP sessions on DANDI registered to the Allen dorsal map, with FC dynamics (FCD) from windows. Models are a linear stochastic model with analytic covariance, Hopf oscillators and optionally Wilson-Cowan, passed through a Balloon-Windkessel transform for BOLD or a calcium kernel for widefield. Global coupling will be selected leave-one-site-out and leave-one-session-out. Degree-preserving rewiring and distance-preserving weight permutation (500 realisations each) will be simulated through the same pipeline; site effects enter mixed models on temporal SNR, rankings are compared by Kendall's tau, multiverse variance is decomposed by ANOVA, and FCD fit uses Kolmogorov-Smirnov distance.

**Expected results.** We expect site-selected coupling to transfer with fit loss under 0.05 in at least 12 of 17 sites; widefield-tuned models to predict BOLD FC above the spatial null with connectome rankings concordant across modalities (tau above 0.5); the distance-preserving null to reproduce at least 30% of explained FC variance, with tracer-specific gains in homotopic edges; construction to explain more fit variance than model choice; and only nonlinear models to reproduce awake FCD.

**Significance.** The study delivers a public connectome-by-model-by-site benchmark and settles how much mouse structure-function modelling rests on acquisition, construction and geometry.

### Which cell types explain regional MRI contrast in the mouse brain? Forward modelling of counted MERFISH cell-type densities with spatial nulls and cross-atlas replication

*Project: [`celltype-composition-mri-contrast`](projects/celltype-composition-mri-contrast/)*

**Background.** The mapping from cellular composition to MRI contrast rests on histology of a few regions or lesion models. Earlier brain-wide work related the cortical T1w:T2w ratio to ISH-inferred densities, and recent deep-learning models run the inverse direction, from MRI to cell type, without reporting which cell types explain which contrast. The Allen Brain Cell Atlas now provides about 4 million MERFISH-segmented cells with class and subclass labels registered to CCFv3, alongside multi-contrast MRI atlases in the same space. No forward, interpretable model has related counted densities to multiple contrasts with spatial autocorrelation controlled, counts compared with gene programs, or replication across ex vivo and in vivo atlases.

**Objective.** To estimate how much between-region variance in T1w:T2w, magnetisation-transfer, T2*, FA and MD contrast is explained by each counted cell class, whether gene programs add to counts, and whether the answer replicates across atlases.

**Methods.** We will assign every ABC Atlas cell to a CCF structure, compute per-structure densities (corrected for MERFISH section coverage, structures with at least 200 cells) for up to 12 aggregated classes plus subclasses, and extract per-structure contrast from the DSURQE, AMBMC and in vivo DTI atlases after ANTs registration. Standardised ridge and PLS regressions across roughly 300 structures will use leave-one-region-out and leave-one-major-division-out cross-validation, LMG relative-importance decomposition with bootstrap CIs, and nested comparison of density-only versus density-plus-gene-program models (myelin and iron genes). Significance will come from 1,000 Moran spectral randomisation surrogates and variogram-matched surrogates built on structure centroids. Importance rankings will be compared across atlases by Kendall's tau and Lin's concordance; voxel-level 50 um models will quantify the resolution limit.

**Expected results.** We expect oligodendrocyte density to carry over 40% of relative importance for myelin-sensitive contrasts with neuronal subclasses adding under 10%, neuron density to dominate MD and neurite indices, gene programs to add over 10% R^2 for T2* but not T1w:T2w, cross-atlas tau above 0.6, and structure-level models to outperform voxel-level ones.

**Significance.** The result is a cell-type-resolved forward model of mouse MRI contrast and a public table aligning regional densities with contrasts for interpreting MRI phenotypes with spatially valid inference.

### Which brain cell types, in which anatomical domain, carry the heritability of imaging-derived phenotypes? A spatially resolved, resolution-matched enrichment study

*Project: [`gwas-celltype-enrichment`](projects/gwas-celltype-enrichment/)*

**Background.** Cell-type enrichment of GWAS heritability is routine for psychiatric disorders, and new methods map trait-associated cells in spatial transcriptomics. Imaging-derived phenotypes (IDPs) have been tested only against developmental cell types or bulk tissue. No study has used adult whole-brain taxonomies with CCF-registered MERFISH cells to ask whether regional IDP genetics localise to the homologous region's cell types, how much resolution IDP GWAS can support, or how estimators agree for imaging traits.

**Objective.** We will map which adult cell types and spatial domains carry the heritability of about 200 UK Biobank BIG40 IDPs, test anatomical concordance as a falsifiable prediction, quantify where enrichment saturates in mouse and human, and report estimator concordance.

**Methods.** Specificity references will be built from the Allen Brain Cell Atlas whole-mouse-brain taxonomy (about 4 million cells; 34 classes, 338 subclasses, 1,201 supertypes, 5,322 clusters), its CCFv3-registered MERFISH dataset (about 4 million cells, imputed genome-wide), and the Siletti human atlas (3.3 million nuclei; 31 superclusters, 461 clusters) via one-to-one orthologs. GWAS inputs are BIG40 discovery (n about 22,000) and replication (n about 11,000) statistics for cortical thickness, surface area, subcortical volume and DTI FA/MD, with schizophrenia and Alzheimer's as positive controls and height as a negative control. Estimators are MAGMA gene-property analysis, stratified LD-score regression conditioned on baseline-LD, expression-matched EWCE bootstraps, per-cell scDRS-style scoring and a Cauchy combination. The anatomical test compares in-homolog against out-of-homolog domain-restricted specificity with a paired Wilcoxon test and a composition-preserving label-permutation null (1,000 permutations), using a pre-registered homology table for primary cortex, hippocampus, thalamus and striatum. Enrichments are BH-corrected per estimator and level and replicated at p below 0.05 with the same sign.

**Expected results.** We expect thickness and area IDPs to enrich in glutamatergic subclasses and DTI IDPs in the oligodendrocyte lineage; stronger enrichment from homologous-region cells; IDP signal to plateau at subclass or supertype level while schizophrenia gains at cluster level; mouse-human agreement at subclass but not cluster level; and lower estimator concordance for IDPs.

**Significance.** The result is a brain-wide, spatially resolved map of the cellular basis of imaging heritability and a calibrated statement of the resolution it can support.


## ICU, EHR & clinical decision support

### Where and why ICU prediction models break: shift decomposition, calibration and subgroup parity across MIMIC-IV, eICU-CRD, HiRID and AmsterdamUMCdb

*Project: [`icu-model-transportability`](projects/icu-model-transportability/)*

**Background.** Every major open ICU database now has a benchmark, and cross-database drops for sepsis, kidney injury and mortality models are well documented. Transfer studies report one discrimination delta and stop: the gap is not attributed to covariate, label or concept shift, calibration is seldom reported although it governs alerts, subgroup performance is rarely examined, and the cost of restoring calibration with local labels is unknown. The closest sepsis work covers three databases without decomposing the loss.

**Objective.** We will decompose the cross-site gap of sepsis, acute kidney injury and 48-hour mortality models into covariate, label and concept components for every ordered site pair, characterise calibration and subgroup parity externally, and measure how many labelled target stays recalibration needs.

**Methods.** MIMIC-IV v3.1 (about 94,000 ICU stays), eICU-CRD v2.0 (about 200,000), HiRID v1.1.1 (about 34,000) and AmsterdamUMCdb v1.0.2 (about 23,000) will be mapped onto a 38-concept ontology whose ricu and YAIB ids are verified against each database's item dictionary. YAIB-compatible cohorts (adults, first stay, prediction at 24 hours) carry 48-hour mortality, KDIGO AKI and Sepsis-3 labels. Logistic regression and LightGBM models tuned by patient-grouped cross-validation are evaluated on a source hold-out and on each other site. Covariate shift is estimated by importance-weighted source evaluation from a domain classifier, label shift by black-box shift estimation, and concept shift as the remainder, with Shapley averaging and bootstrap intervals. Calibration is reported as intercept, slope and ECE; parity by sex, age band and, in US databases, race, with permutation nulls. Intercept-only, Platt, temperature and isotonic recalibration is evaluated for 25 to 2,000 target stays, pooled and group-wise. A negative control decomposes two halves of one site; a positive control recovers an induced shift.

**Expected results.** We expect label shift to dominate calibration-intercept drift, with intercepts tracking the log prevalence ratio (rho above 0.7); concept shift to dominate sepsis discrimination loss and covariate shift the AKI loss; subgroup calibration gaps to widen externally; about 100 labelled stays to remove at least 80 percent of calibration-in-the-large error, slope needing 500 and isotonic 2,000; and only group-wise recalibration to close subgroup gaps.

**Significance.** A per-task, per-site-pair table of shift components and recalibration budgets tells hospitals what a published model needs before deployment.

### A Fully Crossed, Implementation-Traceable Multiverse of Sepsis-3 Operationalisations on MIMIC-IV and eICU-CRD: Cohort Geometry, Model-Ranking Instability, Definition Transfer and a Reportable Core Set

*Project: [`sepsis-definition-multiverse`](projects/sepsis-definition-multiverse/)*

**Background.** Sepsis-3 leaves the analyst to choose the suspected-infection rule, SOFA imputation, baseline, window and onset convention, and public implementations each choose differently. A 2026 variance-components analysis established that suspected-infection rules dominate label variability and shift early-warning AUROC. Unmeasured are the geometry of the resulting cohorts, whether model rankings survive a change of definition, how models trained under one definition behave under another, how treatment-effect estimates move, and which definitions a benchmark should report.

**Objective.** We will run a fully crossed grid of 288 Sepsis-3 specifications, each dimension traceable to a published implementation, identically on MIMIC-IV v3.1 and eICU-CRD v2.0, and carry the variation through to cohort overlap, onset timing, benchmark rankings, definition transfer, time-to-antibiotics estimates and a recommended core set.

**Methods.** Hourly concept tables for about 94,000 MIMIC-IV ICU stays and 200,000 eICU stays (208 hospitals) will be extracted with DuckDB. Grid dimensions cover SOFA missingness handling, culture-antibiotic pairing rules and antibiotic lists, four baselines, two onset conventions and admission-relative filters. Reference implementations are reproduced as grid points and accepted only when prevalence matches published values within 2%. Cohort geometry comprises Jaccard overlap, onset shifts and admission-boundary crossings. Early-warning models (logistic regression, gradient-boosted trees, optional GRU) predict onset within 6 and 12 hours on patient-level 70/15/15 splits fixed across all specifications, yielding AUROC, AUPRC and calibration with bootstrap intervals, Kendall's W of model rankings across specifications, and a train-under-A, test-under-B transfer matrix. A target-trial emulation with cloning, censoring and weighting estimates the time-to-antibiotics association per specification. A shuffled-onset specification is the negative control; six hypotheses are pre-registered with FDR control.

**Expected results.** We expect cohort size to vary more than two-fold and mortality by more than 5 points; median onset to shift more than 6 hours between conventions with over 20% of patients changing admission status; model rankings unstable across the grid (Kendall's W below 0.7); and the time-to-antibiotics association changing sign or significance with the onset convention.

**Significance.** The grid and label-emitting code give sepsis benchmarks a reportable core set in place of one silently chosen definition.

### Circadian label bias in ICU benchmarks: how the hospital clock shapes outcome labels and prediction performance in MIMIC-IV and eICU-CRD

*Project: [`circadian-label-bias-icu`](projects/circadian-label-bias-icu/)*

**Background.** ICU outcomes carry time-of-day structure that is not physiological: discharges follow morning rounds, laboratory draws are batched before dawn and charting is shift-structured. Standard benchmark labels (48-hour mortality, next-24-hour discharge, Sepsis-3 onset from antibiotic and culture times, KDIGO acute kidney injury from creatinine draws) are all defined relative to these workflow timestamps. No study has audited label timing itself, measured how much benchmark performance is clock rather than physiology, or proposed evaluation that the clock cannot inflate.

**Objective.** To quantify phase-locking of label-defining events to the hospital clock, decompose benchmark performance into clock and physiological components, and propose a clock-robust evaluation protocol.

**Methods.** Using MIMIC-IV v3.1 (about 94,000 ICU stays) and eICU-CRD v2.0 (about 200,000 stays across 208 hospitals), we will extract every label-defining event with its clock time, weekday and storetime. Circular statistics (mean resultant length, Rayleigh tests, shift-change phase-locking with permutation p-values) will describe each label, with hospital-level estimates in eICU related to hospital characteristics. Standard tasks will be trained with logistic regression and gradient-boosted trees on physiology-only, physiology-plus-clock and clock-only feature sets under fixed patient-level 70/15/15 splits, reporting AUROC, AUPRC, Brier score and calibration stratified by prediction hour and admission hour, with 1,000-resample paired bootstrap CIs. A phase-randomisation null will add a random offset in [0, 24) hours to each stay's event times, recompute windowed labels 200 times and measure label-flip rates and metric change. Documentation delays from storetime minus charttime will deconvolve hourly hazard curves; tests are BH-FDR corrected.

**Expected results.** We expect mean resultant lengths above 0.3 for discharge and death and above 0.5 for creatinine-defined AKI, clock-only AUROC above 0.6 for discharge, label flips exceeding 10% for mortality and 25% for discharge under phase randomisation with AUROC shifts larger than typical architecture differences, hospital-dependent phase-locking that impairs transfer, and under-predicted mortality for night admissions that pooled recalibration does not fix.

**Significance.** The audit would show that part of every published ICU benchmark score is the hospital clock and would provide hour-stratified, phase-randomised evaluation as a reporting standard.

### How Much of ICU Outcome Prediction Is the Clinician's Ordering Behaviour? Attributing, Localising and Stress-Testing the Laboratory Ordering Channel in MIMIC-IV and eICU

*Project: [`lab-ordering-information-leak`](projects/lab-ordering-information-leak/)*

**Background.** ICU labs are ordered when clinicians are worried, so the presence, timing and urgency of orders carry prognostic information that partly encodes the clinician's own forecast. Masks improve prediction, ordering-only models rival vitals, and observation-process features worsen external calibration. Missing are a quantitative attribution of how much benchmark performance is ordering, an anatomy of where it lives, and a stress test under ordering-policy change.

**Objective.** We will compute the share of predictive information attributable to ordering versus values, localise it among routine, off-schedule, STAT and ordered-but-not-resulted orders, relate mask reliance to degradation under cross-hospital, cross-era and synthetic thinning shifts, compare order-agnostic training strategies, and test whether the mask channel encodes insurance, language and race.

**Methods.** Using MIMIC-IV v3.1 (~94k ICU stays; lab priority, order and result times) and eICU-CRD v2.0 (~200k stays, 208 hospitals), adults on a first ICU stay of >= 24 h will be predicted at 24 h for 48-h mortality, in-hospital mortality, 48-h AKI and ICU LOS > 7 days. Value and mask tables over 26 labs feed logistic regression and gradient-boosted trees; four fits per fold give an exact two-player Shapley split of held-out log-likelihood gain and eight fits localise the mask share. Shift experiments train on eICU ordering-intensity tertiles and MIMIC-IV 2008-2013 eras and test elsewhere, with thinning curves at 100-10% of test-time orders. Order-level dropout, schedule standardisation, intensity reweighting and a joint observation-process model are compared on internal cost versus external gain. Validation uses subject-level and hospital-grouped splits, stay- and hospital-level bootstrap CIs and Benjamini-Hochberg across tasks; controls are within-stay permutation of order times, mask permutation across stays matched on order counts, and a semi-synthetic generator with known ordering informativeness.

**Expected results.** We expect at least 25% of the 48-h mortality gain to sit in the ordering channel, concentrated in STAT and off-schedule orders in the last 6 h; cross-hospital and cross-era AUROC drops correlating with mask share (Spearman rho > 0.6); order dropout halving degradation for <= 0.01 internal AUROC; and mask-only models predicting insurance and language with AUROC above 0.60.

**Significance.** A per-task table of ordering-channel shares and a validated recipe for order-agnostic training would give ICU modelling groups a leak budget and a transfer-robust default.

### Can We Trust Off-Policy Estimates of Ventilator Policies? A Ground-Truth-Anchored Reliability Audit of Offline Reinforcement Learning Across MIMIC-IV, eICU-CRD and HiRID

*Project: [`ventilation-policy-offline-rl`](projects/ventilation-policy-offline-rl/)*

**Background.** Offline reinforcement learning for mechanical ventilation, from VentAI and DeepVent to recent Koopman-constrained and uncertainty-guided conservative methods, reports off-policy evaluation (OPE) values from importance sampling, fitted Q-evaluation or model-based returns as evidence that learned policies beat clinicians. None can say whether those numbers are right, because observational data contain no ground truth, and importance-sampling estimates with effective sample sizes of a few dozen are fragile to the behaviour-policy model. No reliability audit exists for ventilation with cross-site quasi-ground-truth and explicit guideline constraints.

**Objective.** We will measure the bias, coverage and ranking reliability of standard OPE estimators for ventilator policies against two ground truths, and quantify the price of ARDSNet constraints.

**Methods.** Adults ventilated invasively for at least 24 hours in MIMIC-IV v3.1 (about 20,000 ventilated stays), eICU-CRD (about 30,000 across 208 hospitals) and HiRID (34,000 ICU stays) will be harmonised into 4-hour-bin MDPs with 27 tidal-volume, PEEP and FiO2 actions plus weaning and terminal survival reward. A behaviour-cloned policy of site B will be treated as the target, its value estimated on site A by WIS, PDIS, doubly-robust and FQE estimators, and compared with B's observed survival transported by inverse-probability-of-site weighting, across six site pairs and 20 perturbed policies. A calibrated tabular MDP fitted to MIMIC-IV with exact values will test estimators under controlled overlap violation, horizon and reward design. Tabular CQL and IQL policies will be trained with and without hard ARDSNet constraints (tidal volume 4-8 mL/kg, plateau at or below 30 cmH2O, PEEP/FiO2 tables). Every estimate carries a 1,000-draw bootstrap CI and effective sample size (ESS); the held-out site is never used for tuning, and a target equal to the behaviour policy must return the observed mean.

**Expected results.** We expect WIS and PDIS bias above 5 percentage points of survival with coverage below 60% once target and behaviour policies differ on more than 30% of state-action pairs; rank agreement above 0.7 only at ESS above 200 per 10,000 trajectories; unconstrained CQL gains shrinking by at least half under a held-out-site behaviour model; and constrained policies losing under 20% of estimated gain while doubling ESS.

**Significance.** The audit supplies the first ground-truth-anchored reliability statement for ventilation RL and pre-specified criteria for any better-than-clinician claim.

### Inferring Rather Than Charting Respiratory Mechanics: State-Space Estimation of Elastance and Resistance from Ventilator Charting in HiRID, MIMIC-IV and eICU and the Transportability of Driving-Pressure Risk

*Project: [`respiratory-mechanics-estimation`](projects/respiratory-mechanics-estimation/)*

**Background.** Driving pressure and mechanical power guide lung-protective ventilation, yet retrospective associations with mortality in open ICU databases rely on charted plateau pressure, documented in a minority of stays and preferentially in sicker patients and volume-controlled modes. Waveform-based mechanics estimators have never been applied to the sparse charted values in HiRID, MIMIC-IV and eICU.

**Objective.** We will estimate elastance and resistance trajectories with uncertainty for every ventilated hour from charted pressures, volumes and rates, validate the estimator against ground-truth lungs, quantify the selection bias of charted plateau pressure, and test whether driving-pressure risk is invariant across databases once mechanics are inferred uniformly.

**Methods.** Adult stays with six or more hours of invasive ventilation in HiRID 1.1.1 (about 34,000 admissions), MIMIC-IV 3.x (about 94,000 ICU stays) and eICU-CRD 2.0 (about 200,000 stays, 208 hospitals) will be mapped to canonical variables. A Kalman filter with Rauch-Tung-Striebel smoothing will treat elastance and resistance as a random walk with gap-scaled process noise, observing the single-compartment relations for peak and plateau pressure. Validation uses the Google Brain artificial-lung dataset (about 75,000 breaths with known mechanics), a 2026 daily patient mechanics dataset and Bland-Altman agreement with charted values. Cox models of 28-day mortality on time-weighted driving pressure, mechanical power and mechanics will be adjusted for demographics, severity, oxygenation and ventilator settings, with inverse-probability weighting for plateau charting, random-effects pooling with I-squared, leave-one-database-out calibration, hospital-level cross-validation in eICU and no post-outcome data. Nulls permute mechanics across stays, and a synthetic charting simulator checks that the estimator does not manufacture associations.

**Expected results.** We expect median absolute relative error below 15% for elastance and resistance on ground truth and driving-pressure agreement within 4 cmH2O; a significant plateau-charted versus full-cohort interaction; heterogeneity below 50% for inferred associations and above 50% for charted ones; reproducible 72-hour elastance phenotypes (adjusted Rand index above 0.6); and residual-based detection of spontaneous effort with AUROC above 0.8.

**Significance.** The work delivers a uniform mechanics layer for three open ICU databases and tests whether cross-database differences in driving-pressure risk reflect documentation or biology.

### Sedation-Aware Multi-State Landmark Models for Dynamic ICU Delirium Risk: Quantifying Sedation-Policy Leakage and Informative Non-Assessment in MIMIC-IV and eICU

*Project: [`dynamic-delirium-risk`](projects/dynamic-delirium-risk/)*

**Background.** Dynamic ICU delirium models that update every 12 h from EHR time series report high discrimination, but they code comatose (RASS -4/-5) and unscreened windows as delirium-free or drop them, and their strongest features are sedative doses and RASS values that are partly treatment decisions. Deeper sedation then lowers the label rate, so sedatives look protective, and a high score may reflect a plan to lighten sedation rather than vulnerability.

**Objective.** We will build a multi-state landmark framework (normal, delirium, coma-unassessable, unscreened, discharged, dead) with sedation as a time-varying exposure, quantify sedation-policy leakage, correct for informative non-assessment, and test transportability from MIMIC-IV to eICU-CRD conditional on screening density.

**Methods.** From MIMIC-IV v3.1 (about 94,000 ICU stays) we will parse CAM-ICU and RASS chart items into 12-h window states and build landmark datasets from 24 h onward under three label schemes: multi-state, binary with coma/unscreened excluded, and binary with coma coded negative. Cause-specific logistic, multinomial and LightGBM landmark models will use physiology, sedation-policy and assessment-process feature blocks; transition-intensity Poisson regressions will estimate sedative-specific effects; a screening model supplies inverse-probability-of-assessment weights. Splits are by subject with temporal validation; the frozen model is applied to eICU-CRD (about 200,000 stays, 208 hospitals) with per-hospital recalibration. Cluster bootstrap over patients (1,000 resamples), a permuted-window-order control and a null simulator in which sedation has no true effect provide inference.

**Expected results.** We expect the label scheme to shift AUROC of identical features by at least 0.05 and flip the sign of at least one sedative coefficient; removing sedation-policy features to erase at least 30% of the dynamic model's gain over an admission-only model; benzodiazepines to raise normal-to-delirium and coma-to-delirium intensities while dexmedetomidine does not; a prior positive screen to raise the odds of assessment above 1.5; and the largest external loss in the lowest screening-density tertile of eICU hospitals, with calibration gaps by age, sex, race and language wider under binary-negative coding.

**Significance.** The study separates delirium risk from knowledge of sedation plans and screening behaviour, yielding lower but transportable discrimination and sedative effects consistent with randomised trials.

### Braden-scale trajectories and their observation process for dynamic pressure-injury prediction in MIMIC-IV: a label multiverse for a database without present-on-admission flags

*Project: [`nursing-assessment-prediction`](projects/nursing-assessment-prediction/)*

**Background.** Hospital-acquired pressure injuries (HAPI) are a nursing-sensitive quality indicator, and Braden risk is re-scored every shift. Machine-learning HAPI models beat the static Braden score but use admission-time features and single-centre labels. MIMIC-IV records every Braden subscale, skin documentation, prevention and ICD codes but no present-on-admission indicator, so existing MIMIC-IV models on ICD labels cannot separate hospital-acquired from imported injuries, ignore the Braden trajectory, and ignore that nurses assess more often when worried.

**Objective.** We will quantify how HAPI prevalence and cohort membership change across label definitions in MIMIC-IV, test whether landmark prediction from Braden trajectories and the assessment process beats the admission Braden total, and evaluate performance in a treatment-aware way.

**Methods.** Using MIMIC-IV v3.1 (about 94k ICU stays), we will build six HAPI definitions (ICD, ICD stage 2 or higher, nursing-documented first injury after 24 h or 48 h, ICD and nursing combined) and report prevalence, agreement and the fraction of ICD-coded injuries documented within 24 h. A landmark supermodel (pooled logistic regression with landmark-day interactions) at daily landmarks to day 14 will use last, minimum and 48-h slope of Braden total, subscales, assessments per 24 h, hours since last assessment, RASS, prevention and static covariates to predict nursing-documented HAPI within 72 h. Baselines are the admission Braden total and an admission-time model. Patient-level 70/15/15 splits keep all landmarks of a patient together; features use only pre-landmark data and label-defining skin items are excluded. Metrics are time-dependent AUROC, calibration slope and net benefit with 1,000 stay-level bootstraps, Holm-corrected primary tests, within-stay permutation of observation-process features, a label-permutation null and a negative-control outcome.

**Expected results.** We expect prevalence to vary at least threefold across definitions with moderate ICD-nursing agreement, at least 30 percent of ICD-coded stays to show probable present-on-admission injury, the landmark model to exceed the admission Braden score by at least 0.05 AUROC at days 2 to 7 with a calibration slope closer to 1, assessment frequency to add independent information, and lower discrimination after documented prevention.

**Significance.** The label audit serves every group using MIMIC-IV for pressure-injury work, and the treatment-aware evaluation shows how nursing assessments should be used as signals, not snapshots.

### An Open External-Evaluation Benchmark for Bayesian Vancomycin Forecasting on MIMIC-IV: Model Disagreement, Dosing-Record Provenance, Practice Drift and Subgroup Error

*Project: [`bayesian-pk-real-world-dosing`](projects/bayesian-pk-real-world-dosing/)*

**Background.** AUC-guided precision dosing of vancomycin depends on which published population-pharmacokinetic model a Bayesian tool embeds, and external evaluations show large differences between models. That evidence comes from curated or proprietary datasets; what a hospital tool actually receives is EHR data with mislabelled troughs and missing infusion durations. MIMIC-IV exposes that layer, but existing MIMIC-IV vancomycin studies use raw levels without reconstructing dosing histories, and no two groups evaluate models on the same data.

**Objective.** We will build an open, reproducible benchmark of published vancomycin and gentamicin models for Bayesian forecasting on MIMIC-IV and quantify how model choice, dose-record source, practice drift and patient subgroup propagate into AUC24 estimates.

**Methods.** Adult ICU courses with at least one intravenous vancomycin dose and one level will be assembled from inputevents, emar and prescriptions, with trough status inferred from the dosing schedule and weight, creatinine, CRRT and KDIGO AKI joined per course. A versioned library of at least eight ICU vancomycin and two gentamicin models, each with a checker reproducing a published simulation, will drive MAP-Bayesian forecasting and OFV-weighted model averaging. Forecasts are strictly sequential. Metrics are rBias and rRMSE for next-level prediction a priori and after one to three levels, AUC24 disagreement across models, and attainment of AUC24 400-600 mg*h/L, with patient-level cluster bootstrap CIs. AUC24 will be recomputed per dose source and a planted administration-time-error experiment will quantify sensitivity. Priors refitted on anchor-year groups 2008-2013 and 2017-2022 will test temporal drift. Subgroup error by sex, race, BMI class and renal function will be reported, including CKD-EPI 2009 versus 2021. Shuffled dose times serve as a positive control and a simulation with the cohort's sampling schedule checks identifiability; day-2 AUC24 with propagated uncertainty will be compared with first trough for predicting AKI.

**Expected results.** We expect between-model AUC24 spread above 1.25 in over 30% of courses at the first level, dose-source choice changing AUC24 by more than 10% in at least 15% of courses, model averaging lowering rRMSE with the largest gains in AKI, CRRT and obesity subgroups, a measurable temporal validation gap, and subgroup differences in bias.

**Significance.** Anyone with PhysioNet credentials can rerun the benchmark and add a model, giving the field a shared yardstick for real-world dosing tools.

### Outcome-Anchored Mis-Triage by Language and Insurance, Text-Augmented Local Triage Models and Era Drift in MIMIC-IV-ED

*Project: [`ed-triage-mimic-ed`](projects/ed-triage-mimic-ed/)*

**Background.** About one-third of ED encounters are mis-triaged under the Emergency Severity Index (ESI), but the large cohort behind that estimate is not public. The MIMIC-IV-ED benchmark uses structured features, random splits, no text and no fairness analysis. Recent MIMIC-IV-ED work audits ESI assigned by large language models, or runs local small language models for triage prediction, which we take as the baseline; none anchors nurse ESI to outcomes by language or insurance or measures drift across eras.

**Objective.** We will audit outcome-anchored under- and over-triage in MIMIC-IV-ED by race, sex, age, language and insurance, test whether locally run chief-complaint text models reduce the subgroup under-triage gap at the nurses' over-triage rate, and measure how ESI mis-triage and model calibration drift across the 2008-2022 eras.

**Methods.** Reproducing the published cohort (441,437 adult visits) with DuckDB, we will define under-triage as ESI 3-5 followed by a critical outcome (ICU transfer or death within 12 h) and over-triage as ESI 1-2 discharged without admission, ICU or death, then fit adjusted logistic models adjusted for vitals, pain, arrival mode, complaint category and era, with E-values. Models are logistic regression and gradient boosting on structured features, plus TF-IDF and a locally fine-tuned clinical transformer on normalised complaints; demographic variables are audit-only. Primary validation is temporal: train on eras through 2016, validate on 2017-2019, test on 2020-2022, with rolling-origin drift curves and a recalibration arm. Equalised-odds gaps for nurse ESI and each model are compared at the same overall over-triage rate. Nulls are permutation of group labels within ESI strata and a model trained on ESI rather than outcomes; bootstrap CIs use 2,000 patient-clustered resamples with Holm correction.

**Expected results.** We expect at least 60% of under-triaged critical outcomes to sit in ESI 3; adjusted under-triage odds above 1.15 for non-English-speaking and Black patients; text to cut the absolute language under-triage gap by at least 30% at matched over-triage; AUROC stable across eras (change below 0.02) while calibration intercepts drift by more than 0.1 logits; and outcome-trained models to show smaller TPR gaps across language groups than ESI.

**Significance.** A reproducible mis-triage audit with decision curves at deployable operating points shows whether a local text model could safely re-prioritise the ESI-3 pool without importing the disparities encoded in nurse acuity labels.

### Does Triage Documentation Mediate Emergency-Care Disparities for Non-English-Preferring Patients? A MIMIC-IV-ED Study Using Granular Preferred-Language Groups

*Project: [`ed-language-disparity`](projects/ed-language-disparity/)*

**Background.** Limited English proficiency is linked to ED revisits and delays, and Emergency Severity Index (ESI) triage depends on a brief history that a language barrier degrades. The mechanism usually invoked, that less information at triage leads to mis-triage and delay, has never been measured on open data. Prior MIMIC-IV-ED and single-centre work used a binary language flag and reported associations only. MIMIC-IV v3 now records a standardised primary language.

**Objective.** We will test whether non-English-preferring (NEP) patients receive different ED processes and outcomes across granular language groups, whether chief-complaint informativeness mediates those differences, whether the ESI under-triages NEP patients at the same acuity, whether effects persist within ethnicity and shrink when an interpreter is documented, and whether triage ML models inherit the gap.

**Methods.** From MIMIC-IV-ED v2.2 (about 425,000 stays, 205,000 patients), we will build a visit table with modal preferred language per subject (nine language groups plus Unknown), outcomes (admission, length of stay, time to first medication, 72-h return, ICU transfer within 12 h or in-hospital death), chief-complaint features (token count, complaint count, vagueness, barrier markers, missing pain score) and a local regex for interpreter mentions in discharge notes. We will fit adjusted logistic and quantile regressions with cluster-robust errors by subject, critical-outcome rates within ESI level with Wilson intervals and ESI AUROC per group, exact matching on ESI and demographics, and natural direct and indirect effects with 1,000 cluster-bootstrap resamples. Benchmark triage models retrained with patient-level splits and without language as a feature will be compared on sensitivity at a 10% alert rate. Triage temperature is the negative-control outcome.

**Expected results.** We expect higher adjusted admission odds at ESI 3-5, longer stays and later first analgesia for NEP visits; shorter and vaguer complaints, largest for languages other than Spanish; a critical-outcome risk ratio above 1.2 at ESI 3 and ESI AUROC lower by at least 0.02; informativeness mediating at least 20% of under-triage; persistence within Hispanic/Latino patients; smaller gaps with documented interpreters; and lower model sensitivity for NEP visits that widens with text features.

**Significance.** Quantifying a documentation pathway to under-triage points to interpreter access and triage prompts as interventions and shows what deployed triage models inherit.

### Reports That Arrive After the Patient Has Left: Radiology Report Turnaround, Emergency Department Disposition and 72-Hour Returns in MIMIC-IV

*Project: [`radiology-delay-ed-disposition`](projects/radiology-delay-ed-disposition/)*

**Background.** Radiology report turnaround is a standard operational metric linked to emergency department (ED) crowding in single-centre studies, yet its patient-level consequences, including disposition before the final read, discharge with a pending report and addenda after departure, have not been quantified on open data. MIMIC-IV-Note records both the charting time and the signed storage time of about 2.3 million radiology reports.

**Objective.** We will measure how often ED disposition precedes the final report, whether turnaround prolongs post-imaging dwell and admission, and whether post-exit finalisation and addenda predict 72-hour ED return.

**Methods.** Reports from MIMIC-IV-Note v2.2 will be joined to radiology_detail for exam names and addendum links, classified by modality with rules validated on 300 hand-checked names, and linked to about 425,000 stays in MIMIC-IV-ED v2.2 by charting time within the stay. Turnaround equals storetime minus charttime after quarantining implausible values and auto-stored reports; MIMIC-CXR-JPG StudyTime will validate charttime and supply CheXpert positive-finding counts as severity. Post-imaging dwell, admission, ICU within 24 hours and 72-hour return will be modelled with log-linear and logistic regression adjusted for acuity, demographics, arrival mode, chief complaint, modality, exam count, severity and timing, with subject-clustered standard errors. Because date shifting preserves only time of day, quasi-experimental leverage will come from two-stage least squares with hour-block by modality and weekend instruments, and from regression discontinuity in time at the 07:00 shift boundary. Arrival-to-exam time and triage temperature are negative-control outcomes, and storetime permuted within modality by hour is the null. Primary tests are CT abdomen/pelvis dwell, pending-report 72-hour return and night turnaround, Holm-corrected.

**Expected results.** We expect disposition to precede the final report in at least 15% of imaged stays and over 30% of overnight chest radiographs, each hour of CT turnaround to add at least 20 minutes of dwell with concordant instrumental estimates, higher admission odds at ESI 3, and adjusted odds of 72-hour return of at least 1.15 after post-exit finalisation and 1.5 after a post-discharge addendum, with documented verbal communication removing the dwell penalty.

**Significance.** This will be the first patient-level open-data quantification of report timing as a flow and safety variable, and the storetime-versus-charttime audit is reusable by the MIMIC benchmark community.

### Why preoperative risk models fail to travel: shift decomposition, calibration-first evaluation and a cohort-definition multiverse across INSPIRE, MOVER and the MIMIC-IV surgical cohort

*Project: [`perioperative-outcome-transportability`](projects/perioperative-outcome-transportability/)*

**Background.** Preoperative risk models reach AUROC 0.85 to 0.95 inside one health system, and a first INSPIRE-to-MOVER transfer study showed that preoperative models travel worse than intraoperative ones and that calibration can collapse across continents. That evidence covers two sites, does not decompose why preoperative features fail, has no subgroup analysis, and treats VitalDB and INSPIRE as separate sources although both come from Seoul National University Hospital.

**Objective.** We will add a third, differently coded site, decompose transfer failure into covariate, label and concept shift for three outcomes, quantify the labelled sample a hospital needs to recalibrate a foreign model, and show how much the surgical-cohort definition changes the answer.

**Methods.** Sites are INSPIRE (about 130k operations, Korea), MOVER (83,468 surgeries, Irvine) and MIMIC-IV v3.1 (Boston), whose surgical cohort will be defined six ways from PACU transfers, ICD procedure codes and surgical services. Outcomes are 30-day in-hospital mortality, unplanned ICU admission within 24 h and KDIGO AKI within 7 days. Logistic regression and LightGBM on a harmonised preoperative ontology (12 labs, demographics, emergency flag, surgery group, Charlson) will be tuned by subject-grouped cross-validation on the source and evaluated on all six site pairs. Covariate shift is estimated by importance weighting, label shift by black-box shift estimation and concept shift as the remainder, Shapley-averaged with 1,000 subject-level bootstraps; random halves of one site must return components near zero and synthetic label shift must be recovered. Calibration intercept, slope and net benefit are reported with few-shot recalibration curves for 50 to 5,000 target encounters and subgroup metrics by sex and age; 54 tests are Benjamini-Hochberg controlled, and INSPIRE year strata provide within-site temporal comparisons.

**Expected results.** We expect label shift to dominate intercept error and concept shift discrimination loss for mortality, concept shift to dominate ICU admission, and covariate shift in creatinine availability to dominate AKI; slopes below 1 with intercept error tracking log prevalence ratio; 200 target encounters to remove at least 80 percent of intercept error; transported AUROC to vary by at least 0.05 across MIMIC-IV cohort definitions; and within-INSPIRE temporal drift to be at least half the geographic gap.

**Significance.** The results turn external validation into a recipe: what fails, why, and what a receiving hospital must label before deployment.

### Does Synthetic EHR Training Data Transport? Decomposing the Synthetic Gap and the Site Gap from MIMIC-IV to eICU Across Rule-Based, Copula, GAN and Diffusion Generators

*Project: [`synthetic-ehr-transportability`](projects/synthetic-ehr-transportability/)*

**Background.** Synthetic electronic health records are promoted for privacy-safe sharing and prototyping, and their utility is almost always scored by training on synthetic data and testing on a held-out slice of the same real database. That tests whether the source distribution is preserved, not whether a model built on synthetic data survives deployment elsewhere. Existing benchmarks cover single databases or cross-era transfer and do not decompose transport loss. Learned generators also copy the source's measurement process (which labs are ordered, when, and how missingness looks), the features known to drive cross-site shift, so they may yield models that transport worse than real source data.

**Objective.** We will test whether synthetic training data amplifies cross-site loss, attribute any amplification to copied observation-process features, and determine how many target records restore parity.

**Methods.** A 2x2 design crosses training source (real MIMIC-IV versus synthetic) with evaluation site (MIMIC-IV holdout versus eICU-CRD, 208 hospitals, about 200,000 ICU stays) for in-hospital mortality and 30-day readmission on a shared 26-concept encounter ontology that includes per-lab measurement counts and missingness indicators. Generators span Synthea (rule-based, never sees MIMIC), a Gaussian copula, CTGAN, TVAE and tabular diffusion, plus negative and positive control generators, five seeds each. MIMIC-IV patients are split 70/30 by subject; eICU serves only for evaluation. Logistic regression and gradient boosting are scored by AUROC, Brier, calibration intercept and slope, and ECE; the synthetic gap, site gap and their interaction (amplification) carry 1,000-resample bootstrap intervals, and a split-source null must give a site gap near zero. Analyses are repeated without the observation-process block, by subgroup, per eICU hospital, and with intercept-slope recalibration on 100 to 2,500 target records; fidelity and privacy metrics are reported alongside.

**Expected results.** We expect positive amplification for learned generators in AUROC and calibration with intervals excluding zero, a smaller site gap but larger synthetic gap for Synthea, at least a 50% drop in amplification without observation-process features, fidelity metrics that track source but not target utility, larger subgroup gaps, and 500 target records closing most of the calibration but not the discrimination gap.

**Significance.** Current fidelity metrics could not certify synthetic data for model building at a new site; the design gives a transport-aware alternative.

### From Methods Section to Executable Cohort: Local Open-Weight LLMs, a Typed Cohort DSL and a Multiverse Audit of MIMIC Cohort Irreproducibility

*Project: [`llm-cohort-extraction-reproducibility`](projects/llm-cohort-extraction-reproducibility/)*

**Background.** MIMIC underpins thousands of papers whose cohorts are defined in prose. Johnson, Pollard and Mark hand-reproduced 28 mortality-prediction papers and found cohort sizes off by more than 25% in half the experiments, without analysing which criteria drive the gaps. Text-to-SQL benchmarks over MIMIC use synthetic questions, trial-criteria parsers target registries, and PhysioNet policy bars sending credentialed data to third-party services. Whether a local model can turn a paper into an executable, auditable definition, and whether failures are the model's or the paper's, is unknown.

**Objective.** We will benchmark local open-weight models that read only paper text and emit definitions in a typed cohort DSL, compile them deterministically to SQL, measure reproduction of reported cohort sizes, attribute irreproducibility to under-specification versus extraction error via a multiverse over flagged ambiguities, and compare the DSL with model-written SQL.

**Methods.** The corpus is ~60 MIMIC-IV papers (2020-2025) from a fixed PubMed rule plus the 28 MIMIC-III papers of Johnson et al. Each DSL criterion (age, first-stay rules, LOS thresholds, care units, ICD prefixes) carries its source sentence, an ambiguity flag and alternatives; a compiler applies MIMIC-Code conventions on MIMIC-IV v3.1 (~94k ICU stays) and MIMIC-III v1.4. Models span 7-9B, 14-32B and 70B classes at full and 4-bit precision, with five seeds for self-consistency. The primary metric is the fraction within 25% of reported n (Wilson CIs); secondary metrics are median absolute relative error, criterion-level F1 on a 20-paper dual-annotated gold subset, multiverse coverage and execution-failure rate. Comparisons are paired by paper (McNemar, Wilcoxon), Holm-corrected. The model never sees MIMIC rows. Controls: methods text from a different paper must fail to reproduce n; a criteria-removed ablation gives marginal effects.

**Expected results.** We expect the best local model (>= 30B) to reach >= 60% within 25%, matching the human rate; >= 70% of reported sizes to fall inside the multiverse range, with first-stay definition, LOS scope and age above 89 as the dominant ambiguities; < 5% execution failures for the DSL versus > 20% for free SQL; F1 plateauing near 30B with < 3 points lost to 4-bit quantisation; and outcome prevalence shifts above 2 points in >= 30% of papers.

**Significance.** The work would separate reporting failures from extraction failures, give reviewers a checklist of criteria to demand, and offer credentialed groups a policy-compliant auditing tool.


## Cardiovascular signals, ECG, echo & wearables

### Does cuffless blood pressure track change? Bolus-locked evaluation of physiological and deep-learning PPG/ECG models under subject-independent splits in ICU and operating-room waveforms

*Project: [`cuffless-bp-mimic-waveform`](projects/cuffless-bp-mimic-waveform/)*

**Background.** Subject-independent benchmarks on MIMIC-III and VitalDB have established that calibration-free cuffless blood-pressure models fail AAMI/ISO limits, and change-point studies show that models miss statistically detected transitions. Untested is whether any model tracks a causally anchored BP change: a documented vasoactive bolus, where physiology predicts that pulse transit time (PTT) should respond to alpha-agonists but not to beta-agonists that alter the pre-ejection period. How fast a single-point calibration decays and how large leakage inflation is per model class are also unquantified.

**Objective.** We will measure bolus-locked delta-SBP tracking, calibration-interval decay and split-design leakage effect sizes for physics-based PTT models, a feature ridge model, a 1D-CNN and a PTT-plus-CNN hybrid under one subject-independent pipeline across ICU and operating-room waveforms.

**Methods.** Data: the MIMIC-III Waveform Database Matched Subset (10,282 patients), the MIMIC-IV Waveform Database (198 patients) linked to credentialed MIMIC-IV inputevents pushes of phenylephrine, norepinephrine, epinephrine and vasopressin, and VitalDB (6,388 surgical cases). Signals are cut into 10-s windows accepted only if ECG, PPG and ABP quality indices pass. Models: Moens-Korteweg SBP = a + b ln(PAT), ridge on beat features, a 1D-CNN on raw ECG+PPG, and the hybrid. Splits: subject-level GroupKFold with an automated leakage audit, time-blocked splits for calibration decay, cross-domain ICU-to-OR transfer, and deliberately leaky segment-level splits. Bolus analysis: median SBP in [-3, -0.5] versus [+1, +5] min, within-subject correlation of predicted and true deltas and a mixed-effects slope, by drug class. Floors: population- and subject-mean predictors; nulls: subject-shuffled PPG and time-reversed PAT; cluster bootstrap by subject; Benjamini-Hochberg across comparisons.

**Expected results.** We expect PTT-based models to track phenylephrine and norepinephrine responses with within-subject r above 0.5 while the pure CNN stays below 0.3, and PTT tracking to drop after ephedrine; one-point calibration to hold IEEE grade B for at most 30 min and exceed grade D within 2 h in the ICU; and moving from segment-level to subject-level splits to inflate SBP MAE by at least 6 mmHg for the CNN but under 2 mmHg for the PTT model.

**Significance.** Evaluating causally anchored change tracking rather than level estimation identifies the one use case where cuffless models may be clinically useful and gives device validators concrete decay and leakage numbers.

### Open pulse-contour stroke-volume algorithms validated against oesophageal Doppler, thermodilution and echocardiography across the operating room and the ICU

*Project: [`pulse-contour-cardiac-output`](projects/pulse-contour-cardiac-output/)*

**Background.** Pulse-contour analysis estimates stroke volume from the arterial waveform, but commercial algorithms are proprietary, validation studies are small single-setting comparisons against one reference, and percentage errors exceed 40 percent in unstable patients. Deep-learning models are likewise single-centre and closed. VitalDB provides 500 Hz arterial waveforms with simultaneous oesophageal Doppler and FloTrac/EV1000 stroke volume, and MIMIC-III links arterial waveforms to thermodilution and about 45,000 echo reports, but no open algorithm suite has been validated against all three references.

**Objective.** We will validate open pulse-contour estimators against three references in two settings, quantify trending and absolute accuracy under explicit calibration regimes, and measure how damping, arrhythmia and sampling rate moderate error.

**Methods.** Estimators are Liljestrand-Zander, Herd, systolic area, a two-element Windkessel and a fitted impedance correction on slope-sum-detected beats. Calibration regimes are uncalibrated, one-point, 10-minute OLS and hourly, using past reference values only. In VitalDB (about 6,300 cases, references in hundreds) estimates are aligned to CardioQ and EV1000 in 20-s windows; in MIMIC-III (10,282 patients, 125 Hz) to thermodilution within 5 min and to echo LVOT-VTI stroke volume within 30 to 60 min. Hyper-parameters and impedance coefficients are fitted on a 60 percent case-level VitalDB split and frozen for the hold-out and all MIMIC arms. Agreement uses repeated-measures Bland-Altman, percentage error, four-quadrant concordance and polar statistics with 1,000 case-level bootstraps, Benjamini-Hochberg across estimator-reference-regime cells, a shuffled-reference null, a constant-SV trending floor and a Windkessel simulator with known stroke volume as positive control.

**Expected results.** We expect calibrated Windkessel and systolic-area estimators to reach percentage error below 30 percent against CardioQ in stable segments but above 40 percent during vasopressor boluses or high stroke-volume variation, concordance above 90 percent for changes over 10 percent, operating-room calibration to transfer to the ICU with a bias explained by heart rate and MAP, underdamping to inflate systolic-area estimates by over 15 percent, 125 Hz sampling to cost under 3 points, and echo agreement to decay with alignment gap.

**Significance.** An open three-reference, two-setting benchmark gives clinicians and device makers a public yardstick for pulse-contour monitoring where it matters.

### Recorded Fast-Flush Tests as Free Labels: Prevalence, Flush-Free Detection and Decision Impact of Arterial-Line Damping in Open ICU and Operating-Room Waveforms

*Project: [`arterial-line-damping-detection`](projects/arterial-line-damping-detection/)*

**Background.** An arterial catheter-transducer system is a second-order system whose natural frequency and damping coefficient determine whether systolic and diastolic pressures are overshot or blunted; mean pressure is barely affected, so harm concentrates in decisions that use pulse shape. Prevalence is known only from single-centre series with study-team flush tests, detectors were trained on a few dozen hand-labelled patients, and open signal-quality pipelines discard the recorded fast-flush tests that objectively encode the line's dynamic response.

**Objective.** We will mine recorded fast-flush tests in MIMIC-III, MIMIC-IV and VitalDB arterial waveforms to label under- and over-damping at scale, train and externally test a flush-free detector, and quantify how damped periods distort hypotension flags, NIBP agreement, vasopressor titration and waveform-based models.

**Methods.** Flush events are plateaus above 250 mmHg lasting at least 0.2 s; natural frequency, damping coefficient and release time are identified by fitting the second-order step-down response (grid search plus Nelder-Mead, r^2 of at least 0.9 required), then classified by Gardner adequacy rules. Synthetic Windkessel records with embedded flushes serve as positive controls and flush-free segments as negative controls. Labels propagate forward in time only. Per-beat and per-window features (normalised dP/dt max, systolic width, notch prominence, overshoot, rise time, spectral peakiness) train a classifier with record-grouped 5-fold CV on the MIMIC-III matched subset (22,317 records, 10,282 patients), excluding windows containing flushes, then tested on VitalDB (500 Hz and resampled to 125 Hz) and MIMIC-IV. Decision impact uses enrichment ratios for SBP below 90, MAP below 65 and SBP above 160 minutes, Bland-Altman NIBP-ABP bias per class, vasopressor rate-change ratios, mixed-effects logistic models with record random intercepts and severity covariates, and 1,000-record cluster bootstraps with Benjamini-Hochberg correction.

**Expected results.** We expect at least 80% of records over 12 h to contain a fittable flush, over 25% of labelled line-hours to be inadequately damped, AUROC of at least 0.90 within and 0.85 across databases, enrichment above 1.3 for systolic flags with MAP flags near 1.0, systolic NIBP-ABP bias beyond 10 mmHg by class, and cuffless-BP and pulse-contour errors at least 50% larger in damped segments.

**Significance.** The detector and labelled cohort would make line quality an explicit variable in pressure-based ICU research and decision support.

### Pulse-Pressure Variation in the Wild: Automated Waveform PPV, Real Fluid Boluses and a Regression-to-the-Mean Null in MIMIC Arterial Lines

*Project: [`fluid-responsiveness-waveforms`](projects/fluid-responsiveness-waveforms/)*

**Background.** Pulse-pressure variation (PPV) predicts fluid responsiveness with pooled AUC near 0.87, but validity requires controlled ventilation with tidal volume of at least 8 mL/kg, no spontaneous effort, regular rhythm and a closed chest, and point-prevalence data show few ICU patients meet these conditions at any moment. Validation cohorts select patients who do, so no real-world estimate exists of PPV performance when prerequisites fail. Bolus-response studies never control regression to the mean, although boluses are given when pressure is low and pressure tends to rise anyway.

**Objective.** We will compute PPV and systolic-pressure variation automatically on arterial waveforms before real ICU boluses, estimate their predictive value for pressure response overall and within the prerequisite-satisfying subset, quantify the response expected under no treatment with matched sham windows, and test the effect of co-interventions and outcome definitions.

**Methods.** From the MIMIC-III Waveform Database Matched Subset (22,317 records, 10,282 patients) and MIMIC-III Clinical v1.4 MetaVision stays, boluses are crystalloid or colloid inputevents of at least 250 mL in 30 min or less with no other bolus nearby. PPV and SPV are computed over 8-s windows stepped by 2 s. Response labels are a MAP rise of at least 10% and a pulse-pressure rise of at least 15% between the 15-min pre-window and 30-min post-window; vasopressor rate changes are flagged; prerequisite flags come from charted ventilator settings and rhythm. Sham windows matched on pre-window MAP within the same patient give the untreated response rate. The algorithm is first validated on VitalDB (6,388 cases) against device SVV by Bland-Altman analysis. AUROC with patient-cluster bootstrap, mixed-effects dose-response models, post-window PPV as a causal-direction negative control, and cross-patient permutation for chance complete the design.

**Expected results.** We expect AUROC of 0.60-0.70 in unselected boluses and at least 0.80 with a narrower gray zone when all prerequisites hold; sham response rates at least half the crude bolus response rate; co-intervention exclusion changing responder rates by more than 20% and AUROC by at least 0.03; MAP-based and PP-based labels agreeing with kappa below 0.6 with PPV predicting the PP label better; and bias under 2 points against device values.

**Significance.** This audit quantifies how a guideline index behaves under real ICU conditions and provides an open, repeatable pipeline for later MIMIC-IV waveform releases.

### Decomposing Cross-Population Generalisation of 12-Lead ECG Classifiers into Label-Mapping, Acquisition and Population Components Across Four Countries

*Project: [`ecg-cross-dataset-generalization`](projects/ecg-cross-dataset-generalization/)*

**Background.** Deep 12-lead ECG classifiers reach cardiologist-level AUROC within a dataset, but their drop on other sources ranges from negligible to large and has never been explained, because sources differ at once in label vocabulary (SCP-ECG, SNOMED-CT, free-text classes), acquisition (400 vs 500 Hz, filters, vendors) and population. Recent benchmarks report aggregate scores only, without label-mapping sensitivity, subgroup parity or outcome validation.

**Objective.** We will build a population-shift benchmark across seven public datasets from Germany, China, the USA and Brazil, decompose the cross-source AUROC drop into label-mapping, acquisition and population components, compare open ECG foundation models with a supervised baseline on the same grid, and test whether transfer quality predicts prognostic utility in MIMIC-IV-ECG.

**Methods.** PTB-XL (21,799 ECGs), Chapman-Shaoxing and Ningbo (45,152), Georgia (10,344), CPSC-2018 and CPSC-Extra (10,330), CODE-15% (345,779) and MIMIC-IV-ECG (about 800,000) will be harmonised to 12 x 5000 arrays at 500 Hz and mapped to 12 classes under strict, lenient and superclass policies. A handcrafted-feature logistic model, a 1D-ResNet-18 and frozen and fine-tuned ECG-FM, HuBERT-ECG and ECGFounder will run on a 5-source x 5-target grid with counterfactual conditions (harmonised mapping, acquisition harmonisation, age/sex density-ratio reweighting) whose Shapley-averaged differences give the three components. Evaluation uses macro AUROC, agreement-restricted AUROC, simulated 5-15% label flips, per-class kappa between policies, and AUROC and calibration gaps by sex and age band with patient-level bootstrap (2,000 resamples). Splits are patient-level; nulls permute source labels within age/sex strata. Surviving models are probed on MIMIC-IV-ECG for 1-year mortality (Harrell's C) and troponin-T elevation within 24 h of the first ECG per hospitalisation.

**Expected results.** Label mapping will explain the largest share of the drop for conduction and repolarisation classes and population for AF and hypertrophy; population drop will exceed device drop after harmonisation; fine-tuned foundation models will cut the mean drop by at least 30% without narrowing parity gaps, which widen most on CODE-15%; transfer AUROC will correlate with mortality C-index (Spearman rho above 0.5).

**Significance.** An open leaderboard, versioned mapping table and decomposition code make generalisation claims attributable to causes and tie diagnostic transfer to hard outcomes.

### Where Do ECG Foundation Models Encode Age, Sex and Race? Layer-Wise Probing, Concept Erasure and the Link to Diagnostic Inequity

*Project: [`ecg-foundation-model-probing`](projects/ecg-foundation-model-probing/)*

**Background.** Open 12-lead foundation models (ECG-FM, HuBERT-ECG, ECGFounder) report that their embeddings encode demographics, framed as robustness. Those reports examine only the final embedding, give no layer-resolved account of where demographic information enters, and never test interventionally whether removing it helps or hurts. No study maps leakage across several public foundation models or links the erasable component to subgroup fairness of downstream diagnosis.

**Objective.** We will quantify how strongly and at which depth frozen ECG foundation-model embeddings encode age, sex and self-reported race, apply linear concept erasure to measure the diagnostic cost of removing that information, and test whether labels whose embeddings leak more demographics show larger subgroup performance gaps.

**Methods.** PTB-XL (21,799 ECGs, official folds) and MIMIC-IV-ECG (about 800,000 ECGs, 160,000 patients, self-reported race via MIMIC-IV linkage; split by subject) will be harmonised to 12 x 5000 arrays at 500 Hz and encoded by each frozen model, retaining per-block hidden states. For each model, layer and attribute we will fit linear and small MLP probes with nested cross-validation, minimum-description-length probes reporting codelength in bits, and random-label control tasks for selectivity. LEACE closed-form erasure and iterative INLP, fitted on training patients only, will produce erased embeddings on which attribute decodability and per-label diagnostic linear-probe AUROC are re-measured. Subgroup AUROC and equalised-odds gaps by age band, sex and race will use patient-level stratified bootstrap (2,000 resamples); label-wise leakage will be correlated with label-wise gap. Permuted-attribute probes give chance decodability, Benjamini-Hochberg controls the false discovery rate within each family.

**Expected results.** We expect sex AUROC above 0.95, age MAE below 8 years and race macro-AUROC above 0.7 for at least one model; sex and age information rising to the final layers with a compressible code while race is diffuse; sex erasable with under 2% mean diagnostic AUROC loss except for labels with genuine sex dependence such as LVH; a positive Spearman correlation (rho above 0.3) between demographic decodability and subgroup gap.

**Significance.** A layer-wise leakage atlas, erasure-versus-utility curves and a leakage-to-gap correlation would separate legitimate physiological covariance from shortcut and give auditors a concrete tool for fair, private ECG representations.

### How far can adult-trained 12-lead ECG models be pushed into childhood? Age-continuous generalization decay, shift decomposition and label-efficient adaptation on ZZU-pECG

*Project: [`pediatric-ecg-generalization`](projects/pediatric-ecg-generalization/)*

**Background.** The pediatric ECG is not a small adult ECG: rates, intervals, amplitudes and T-wave polarity change from neonate to adolescent, so adult-trained models face both covariate shift and an age-dependent waveform-to-label mapping. Recent adult-to-pediatric transfer work reports failure and proposes new pretraining, but no study charts where transfer breaks as a continuous function of age, decomposes the loss into covariate, prior and concept shift, or measures how many pediatric labels adaptation needs.

**Objective.** We will chart the age-resolved generalization decay of adult-trained ECG models on the open ZZU-pECG database, decompose the infant-versus-adolescent gap, and find the most label-efficient adaptation on labels comparable across age.

**Methods.** Adult sources are PTB-XL (21,799 ECGs), CODE-15% (345,779 ECGs) and the PhysioNet/CinC 2021 training set (about 88k ECGs); the pediatric target is the 12-lead subset of ZZU-pECG (12,334 of 14,190 records from 11,643 children aged 0 to 14). Labels are harmonised to shared rhythm and conduction classes (sinus tachycardia and bradycardia, bundle branch block, first-degree AV block, ectopy). A 1D-ResNet and a handcrafted-feature baseline trained only on adults will be evaluated zero-shot; AUROC will be fitted as a smooth function of age with bootstrap bands, an age-permutation null and a matched-rate placebo subsampling adolescents to the infant rate distribution. The infant-adolescent gap will be Shapley-decomposed using renormalisation to Rijnbeek age norms, reweighting and recalibration counterfactuals. Adaptation methods (recalibration, age-conditioned standardisation, linear probe, full fine-tuning) will be compared on log-spaced pediatric label budgets against a pediatric-only model, with child-level disjoint splits, normalisation fitted on source or pediatric-train only, and Benjamini-Hochberg correction.

**Expected results.** We expect AUROC to decline monotonically toward younger ages and be worst in infants; covariate renormalisation to remove most of the gap for rate-defined labels, leaving a residual concept-shift gap for morphology labels; age-conditioned normalisation with a linear probe to be the most label-efficient route to within 2 AUROC points of the pediatric-only ceiling, with full fine-tuning winning only beyond 2 to 3k labels; and confident errors to spike in infants for repolarisation outputs.

**Significance.** The decay curve and label-efficiency results tell hospitals with few pediatric ECGs which adult models can be reused, at what ages, and at what cost.

### Dose-Resolved and Time-Resolved QTc Prolongation from Administration-Level Electronic Health Records: A Within-Patient Multiverse Analysis of MIMIC-IV-ECG

*Project: [`qt-prolongation-dose-response`](projects/qt-prolongation-dose-response/)*

**Background.** Regulatory assessment of drug-induced QT prolongation rests on concentration-QTc modelling in small thorough-QT studies of healthy volunteers, while CredibleMeds and FAERS supply risk categories and disproportionality signals rather than effect sizes. No public study estimates, per drug, the within-patient QTc change per unit administered dose and its time-course at the doses used in routine inpatient care.

**Objective.** We will estimate real-world dose-response slopes and time-to-peak of QTc change for a pre-registered list of QT-prolonging drugs, map how fragile these estimates are across analyst choices, and test whether they recapitulate CredibleMeds categories and FAERS torsades signals.

**Methods.** We will link the MIMIC-IV v3.1 medication-administration record (emar, emar_detail; ICU inputevents for infusions), normalised to RxNorm ingredients and canonical doses, to approximately 800,000 12-lead ECGs in MIMIC-IV-ECG v1.0. Each administration will be paired with a drug-naive baseline ECG within 48 hours and post-dose ECGs within 24 hours. Machine-measured QT will be compared with re-delineated QT under Bazett, Fridericia, Framingham, Hodges and a cohort-fitted population correction. The primary estimand is the within-patient change in QTc, modelled per drug with mixed effects (dose, spline in time-since-dose, serum potassium, age, sex, patient random intercept). A multiverse over correction formula, measurement source, electrolyte adjustment and exposure windows will produce specification curves with permutation nulls from doses shuffled within drug episodes. Negative-control drugs, the PR interval as a negative-control outcome, a time-reversal placebo and a naive between-patient model will bound residual confounding. Slopes will be rank-correlated with CredibleMeds ordinal categories and shrunk FAERS reporting odds ratios from openFDA, with Benjamini-Hochberg control.

**Expected results.** Known-risk drugs will show positive dose slopes and negative-control drugs will not; time-to-peak will exceed zero for oral agents; Bazett will inflate slopes for rate-changing drugs relative to Fridericia and the fitted correction, with rankings more stable than absolute slopes; slopes will correlate with CredibleMeds categories (Spearman rho above 0.4), and discordant drugs will be tabulated.

**Significance.** This study delivers a per-drug real-world dose-response table anchored to the regulatory 10 ms threshold and a transparent audit of how correction and measurement choices shape pharmacoepidemiologic QT estimates.

### Do Conformal Ejection-Fraction Intervals Survive Dataset Shift? Coverage, Weighted Recovery and Abstention Across Adult, Pediatric and Multi-View Echocardiography

*Project: [`echonet-ef-uncertainty`](projects/echonet-ef-uncertainty/)*

**Background.** Video networks estimate left-ventricular ejection fraction (LVEF) from apical 4-chamber clips with MAE near 4%, but they give point estimates without calibrated uncertainty. Anatomically constrained EF is already reported to transfer with better accuracy across datasets, yet no study measures whether intervals with finite-sample guarantees keep their coverage when the model meets a new population, view or vendor, whether interval width flags clinically hard studies, or whether uncertainty grows on wrong-view inputs.

**Objective.** We will quantify the empirical coverage of split-conformal LVEF intervals calibrated on adult A4C echoes under adult-to-pediatric, A4C-to-A2C and vendor/site shift, test whether likelihood-ratio weighted conformal prediction restores coverage, ask whether adaptive widths identify poor-quality and rhythm-irregular studies, and test abstention on wrong-view inputs.

**Methods.** An R(2+1)D-18 regressor with a five-seed ensemble scale, and a DeepLabV3 segmentation route with Simpson's EF as a secondary comparator on interval validity only, will be trained on EchoNet-Dynamic train (10,030 videos) and calibrated on its official validation split (1,288 videos), never on test. Nominal 80/90/95% intervals from split, locally adaptive and weighted conformal scores will be evaluated on EchoNet-Dynamic test (1,277), EchoNet-Pediatric A4C (about 3,000) and PSAX (about 4,000), and CAMUS A4C and A2C (500 patients each, cross-fitted), with weights from a logistic domain classifier on image and embedding descriptors. We report coverage with Clopper-Pearson intervals, width, worst-slab coverage and a 200-study target-calibrated arm; width AUROC is computed for CAMUS quality labels and beat-to-beat EF variability. TMED-2 (5,261 images) and EchoNet-LVH PLAX clips serve as wrong-view inputs. Random versus learned weights are the null; the primary contrast (adult-to-pediatric A4C at 90%) is pre-registered with Holm correction.

**Expected results.** We expect coverage below 80% for pediatric A4C, pediatric PSAX and CAMUS A2C while CAMUS A4C stays within 85-92%; weighting to restore at least 88% coverage at under 1.5x width for vendor and quality shifts but not for pediatric transfer without local calibration; width AUROC of at least 0.70 for poor versus good quality; and wrong-view widths above the 95th in-distribution percentile in at least 90% of cases for the segmentation route.

**Significance.** Coverage under shift, not in-distribution calibration, decides whether an EF uncertainty estimate is safe to deploy.

### Echocardiographic View Classification Under Scanner-Vendor Shift: A Leave-One-Source-Out Benchmark with Acquisition Decomposition and Clinically Weighted Confusions

*Project: [`echo-view-vendor-robustness`](projects/echo-view-vendor-robustness/)*

**Background.** Every echocardiographic measurement is view-specific, so a view classifier that silently mislabels views on a new scanner corrupts ejection fraction and strain. View models are usually trained and tested on one institution and one vendor. Existing view benchmarks evaluate methods, but no study isolates the vendor axis for view classification, decomposes the drop, or maps which acquisition differences induce dangerous confusions.

**Objective.** We will measure per-view degradation of view classifiers across public sources that differ in vendor and site, decompose it into acquisition versus anatomy with controlled counterfactual renderings, characterise clinically consequential confusions, compare label-free adaptation strategies, and quantify downstream EF error.

**Methods.** Frames from EchoNet-Dynamic (10,030 GE A4C clips), CAMUS (500 patients, GE, A4C/A2C), TMED-2 (599 studies, 5,261 labelled images), TTE47 and a view-labelled sample of MIMIC-IV-ECHO (about 7,000 Philips/GE DICOM studies) will be sector-masked, resized and mapped to a canonical view set. A handcrafted-feature logistic baseline, a 2D CNN and frozen echo-foundation-model features with a linear head will be evaluated on a leave-one-source-out grid with study-level splits. Histogram matching, sector re-masking, frame-rate resampling and gamma jitter serve as counterfactual transforms whose Shapley-averaged contributions decompose the macro-F1 drop; a clinically weighted error up-weights A4C/A2C and apical/parasternal swaps. Instance normalisation, test-time BatchNorm adaptation and histogram matching will be compared without target labels. Nulls include a shuffled-vendor test and a permutation confusion-matrix null; study-level bootstrap (2,000 resamples) and Benjamini-Hochberg correction provide inference.

**Expected results.** We expect the largest drops for A4C versus A2C and PSAX levels and larger drops across vendors than within one; intensity plus geometry matching to remove most of the drop for coarse views; cross-vendor confusion mass to concentrate on dangerous swaps so the weighted error widens more than raw accuracy; frozen foundation features with instance normalisation to give the best label-free recovery; and realistic confusion rates to inflate downstream EF MAE measurably.

**Significance.** The benchmark and decomposition tell developers which acquisition differences to normalise and where a view classifier must abstain before it can be trusted across scanners.

### A physiology-anchored cardiac digital twin learned from unpaired open ECG and echocardiography corpora and tested on paired ICU data

*Project: [`cardiac-digital-twin-ecg-echo`](projects/cardiac-digital-twin-ecg-echo/)*

**Background.** AI-ECG screens for low ejection fraction, and joint ECG-echo representation models now learn shared embeddings, but all require paired data from one institution and expose no interpretable state on which counterfactuals can be run or calibration audited. The largest open corpora, PTB-XL (21,799 ECGs) and EchoNet-Dynamic (10,030 echo videos), never share a patient, and whether they can be fused into one cardiac state that transports to paired data from a third institution has not been tested.

**Objective.** To learn a low-dimensional, physiologically constrained cardiac state (EF, EDV, LV mass, heart rate, QRS duration) from unpaired ECG and echo corpora, and to test its transport, calibration, counterfactual consistency and discordance detection on paired ECG-echo data.

**Methods.** ECG features (QRS duration, axis, Sokolow-Lyon voltage, a ResNet embedding) will be extracted from PTB-XL; echo labels and an R2+1D encoder come from EchoNet-Dynamic and EchoNet-LVH. Forward models from state to each modality, simulation-based regression from ECG to state, and one-dimensional optimal-transport alignment of the ECG-derived EF marginal to the echo cohort will produce the twin, first linear-Gaussian, then conditional generative. Evaluation will use paired MIMIC-IV-ECG and MIMIC-IV-ECHO (ECG within 7 days of the echo, patient-level splits, never used for tuning) and EchoNext (82,543 pairs, 36,286 patients), against a paired supervised ECG-to-EF regressor and a contrastive model retrained on the same MIMIC pairs. Metrics are EF MAE, AUROC for EF < 40%, predictive-interval coverage and discordance AUROC, with 2,000-resample patient-level cluster bootstrap, a pre-specified non-inferiority margin of 0.05 AUROC, a label-permutation null for counterfactual direction and a prior-shift sensitivity analysis.

**Expected results.** We anticipate EF MAE at or below 9 points and AUROC at or above 0.85 for EF < 40% without any paired training sample; a smaller domain-shift drop than for the contrastive embedding; physiologically correct counterfactual ECG changes in at least 80% of interventions; posterior width ranking discordant pairs with AUROC at or above 0.70, enriched for long acquisition delays and conduction disease; and quantified bias when the EF prior is mis-specified.

**Significance.** An unpaired, interpretable twin would show that open, non-overlapping corpora suffice for cross-modal cardiac inference; failure on structural but not functional axes would itself be informative.

### Does glucose forecasting break when the insulin regimen changes? Cross-regimen transfer, controller-induced predictability and data artefacts in open CGM datasets

*Project: [`cgm-forecasting-regimen-shift`](projects/cgm-forecasting-regimen-shift/)*

**Background.** Short-horizon glucose forecasting from continuous glucose monitoring underpins hypoglycaemia alarms, and the field benchmarks on OhioT1DM, an open-loop pump cohort. Therapy has shifted to automated insulin delivery (AID), where the forecast target is the output of a control loop that dampens variance. Existing benchmarks pool datasets across regimens, but none frames insulin regimen as a distribution-shift axis, reports clinical warning metrics across it, separates "AID is easier to forecast" from genuine transfer, or audits the OpenAPS Data Commons for artefacts that inflate accuracy.

**Objective.** To quantify accuracy and hypoglycaemia-warning degradation when forecasters cross between open-loop and AID regimens, decompose the change into marginal shift and controller-induced predictability, audit DIY-loop data quality, and test regimen-aware normalisation as a mitigation.

**Methods.** We will harmonise OhioT1DM (12 people, 8 weeks, open-loop), DiaTrend (54 patients, mixed devices), AZT1D (25 patients on Tandem Control-IQ) and the OpenAPS Data Commons (over 100 users, 46,000+ days) onto a 5-minute grid. Persistence, ARIMA, ridge and a small LSTM will produce 30- and 60-minute forecasts in a leave-one-regimen-out transfer grid with participant-level splits. Primary endpoints are hypoglycaemia (< 70 mg/dL) warning sensitivity at a fixed false-alarm rate, median lead time and Clarke zone A+B proportion, with participant-level bootstrap CIs; RMSE and MAE are secondary. Nulls include shuffled-participant transfer and a variance-matched surrogate rescaling open-loop glucose to AID variance. A feedback decomposition will regress AID glucose change on recent CGM and insulin to estimate the loop-explained fraction, and the audit will flag gaps, interpolation, calibration jumps and algorithm-version mix, re-evaluating under artefact-filtered inclusion.

**Expected results.** We expect open-loop-to-AID transfer to improve absolute RMSE yet show systematic bias and lost warning sensitivity and lead time, AID-to-open-loop transfer to degrade markedly, a substantial loop-explained fraction of AID variance, inflated accuracy under naive inclusion of interpolated OpenAPS segments, and regimen-aware normalisation to recover calibration but not lead time.

**Significance.** The study delivers a cross-regimen benchmark with clinical endpoints and an artefact-filtered evaluation protocol, and tests whether forecasters validated on open-loop data remain safe under automated delivery.

### GaitShift-PD: a leave-one-device-out benchmark for Parkinson's gait models across sensors and body placements with free-living wrist validation

*Project: [`gait-wearable-cross-device`](projects/gait-wearable-cross-device/)*

**Background.** Wearable gait analysis is a mature digital-biomarker area in Parkinson's disease (PD), yet published models are trained and tested on one device and protocol. Cross-dataset work exists only for freezing of gait, where lab performance collapses in daily living. No benchmark holds out a whole device or placement family for PD detection, severity or gait-timing regression, none ablates harmonisation choices, and UK Biobank self-supervised wrist backbones are untested on non-wrist placements and PD tasks.

**Objective.** We will quantify the deployment gap of hand-crafted, deep and self-supervised accelerometry models when sensor, placement, protocol or population changes, identify which harmonisation and feature choices recover it, and validate surviving models on free-living wrist data.

**Methods.** Eight open datasets enter a fixed harmonisation pipeline: PhysioNet gaitpdb (93 PD, 73 controls, force plates), gaitndd (64, footswitches), LTMM (71, lower back), PADS (469, smartwatch), Daphnet (10 PD), the Kaggle TLVMC FoG data (about 100), WearGait-PD and mPower (over 10,000, smartphone). Signals are resampled to 50 Hz, gravity-filtered and mapped to orientation-invariant channels in 5 s windows; force-plate and footswitch data enter through stride-timing features. Tasks are PD versus control, severity strata, FoG detection and stride-time variability. Models span feature-based classifiers, a 1-D CNN, a fine-tuned UK Biobank self-supervised backbone, and domain-generalisation baselines. The primary comparison is within-dataset GroupKFold balanced accuracy against leave-one-dataset-out accuracy, typed by shift, with 2,000-resample subject-level bootstraps, Holm correction, label-permutation chance levels and a dataset-identity probe. Lab-trained severity models will then be applied to PPMI Verily Study Watch measures (343 participants, about 485 days each) aggregated over 14 to 90 days and related to MDS-UPDRS-III.

**Expected results.** We expect leave-one-dataset-out accuracy at least 15 points below in-distribution accuracy, largest for wrist-to-lower-body shifts; harmonisation to recover at least 30 percent of the gap; the self-supervised backbone to transfer between wrist datasets only; timing features to transfer across force plates and IMUs (R^2 at least 0.5) where amplitude features do not; and free-living severity correlations (rho at least 0.3) only after multi-week aggregation.

**Significance.** GaitShift-PD delivers a reusable benchmark and a quantitative account of how much PD gait accuracy survives a change of hardware.


## Pharmacovigilance & regulatory science (openFDA)

### Denominator-Aware and Bias-Adjusted Signal Detection in FAERS: Sex-Specific Reporting Rates, Stimulated Reporting After FDA Safety Communications and Reporter-Type Effects

*Project: [`faers-reporting-bias`](projects/faers-reporting-bias/)*

**Background.** FAERS disproportionality analyses treat report counts as proportional to harm, yet three biases are rarely modelled jointly. FAERS has no exposure denominators, so a sex-specific reporting odds ratio (ROR) cannot separate biology from prescribing and reporting behaviour; FDA Drug Safety Communications (DSCs) alter who reports what; and consumer or lawyer reports can flip drug rankings relative to physician reports. Sex-stratified analyses have become common without denominators, interaction tests or adjustment.

**Objective.** We will estimate denominator-aware sex- and age-specific reporting rates, meta-analyse stimulated reporting across all DSCs since 2010 with a placebo-date null, build a bias-adjusted sex-stratified disproportionality screen with a formal drug-by-sex interaction test, and measure how many published sex-specific FAERS signals survive it.

**Methods.** US FAERS reports received 2010-2024 will be flattened by drug, MedDRA PT, sex, age band, reporter type and date. Denominators come from MEPS prescribed-medicine files 2018-2023 (survey-weighted persons with at least one fill by sex and age) and Medicare Part D beneficiaries aged 65 and over for about 450 ingredients. Reports per 10,000 exposed with exact Poisson intervals yield female:male reporting-rate ratios. For each DSC, negative-binomial segmented regression with an all-FAERS offset estimates level and slope changes for the named drug-event pair and the drug's other events by reporter type, calibrated against 200 placebo dates and pooled by random-effects meta-analysis. A report-level logistic model with drug, reporter, post-DSC window, sex and drug-by-sex terms or an exposure-offset Poisson model give adjusted RORs with likelihood-ratio interaction tests under Benjamini-Hochberg at 5%. Permuted sex labels and negative-control pairs check false-positive rates; a pre-registered list of 2023-2026 papers claiming sex-specific signals is re-estimated.

**Expected results.** We expect the adjusted female:male ratio to fall below the crude ratio for most drugs, with at least 30% of drugs with crude ratios above 1.5 no longer differing from 1; a median post-DSC level change above 1.3 for the named event and near 1 for other events; HCP-only restriction removing at least 25% of ROR signals; and an order of magnitude fewer sex-specific signals than "female signal, male null" logic implies.

**Significance.** The framework separates "women report more" from "women are harmed more" and gives a denominator-aware standard for sex-stratified pharmacovigilance.

### How Often Is a FAERS Disproportionality Signal Real? A Positive-Predictive-Value Map Against Lab-Defined Outcomes in MIMIC-IV Using Active-Comparator Target-Trial Emulation

*Project: [`faers-signal-ehr-validation`](projects/faers-signal-ehr-validation/)*

**Background.** Disproportionality statistics on FAERS generate thousands of signals per year. Their accuracy is judged against reference sets such as OMOP and EU-ADR that were built from labels and literature largely informed by the same spontaneous reports, so the evaluation is partly circular. Studies combining spontaneous-report and EHR scores fuse two noisy detectors rather than validating one against a measured outcome. MIMIC-IV holds administrations, serial labs and machine-measured ECG intervals but has not been used for this.

**Objective.** We will estimate, drug by drug and outcome by outcome, the positive predictive value of FAERS signals against calibrated active-comparator new-user effect estimates for seven laboratory- or ECG-defined outcomes in MIMIC-IV, identify reporting features that predict false positives, and release an EHR-derived reference set.

**Methods.** For every ingredient with at least 100 suspect reports in openFDA and each outcome (hyperkalaemia, hyponatraemia, acute kidney injury, hepatocellular injury, thrombocytopenia, neutropenia, QTc prolongation), we will compute PRR, ROR, IC025 and EB05 with an MGPS prior fitted by marginal likelihood. In MIMIC-IV v3.1 and MIMIC-IV-ECG (about 800,000 ECGs), incident outcomes require a normal baseline within 7 days and a threshold crossing within 7-14 days. Target-trial emulation compares first inpatient administration of drug A versus a same-indication comparator after a 180-day washout, with propensity-score 1:1 caliper matching on demographics, comorbidity, baseline labs and co-medications. Empirical calibration with 20-40 pre-registered negative-control ingredients per outcome gives calibrated p-values; truth is a calibrated risk ratio of at least 1.5 with p below 0.05. PPV, sensitivity and AUROC are estimated over roughly 150-300 adequately powered pairs with Wilson intervals and pair-level bootstrap; shuffled signals define chance.

**Expected results.** We expect PPV above 0.5 for hyperkalaemia and acute kidney injury and below 0.3 for hepatotoxicity and QTc prolongation; EB05 and IC025 to outperform PRR/ROR at matched sensitivity with Spearman rho above 0.4 between signal rank and EHR risk ratio; signals driven by non-professional reporters, early post-approval reports or concomitant roles to be confirmed less often (odds ratio below 0.5); at least 20% of strong EHR-confirmed harms to lack any FAERS signal.

**Significance.** A measured-outcome PPV map and a label-independent reference set replace circular benchmarks for spontaneous-report signal detection.

### A Database-Wide Map of Unlabeled FAERS Signals and the Lag from Signal Emergence to US Label Update, Benchmarked Against FDA's Quarterly Potential Signals

*Project: [`unlabeled-adverse-event-mining`](projects/unlabeled-adverse-event-mining/)*

**Background.** FAERS signal detection exists to find risks not yet on the label, yet standard disproportionality workflows do not know what the label says, so labelled reactions dominate published signals and new ones are buried. The time between when a signal became detectable and when the label named it is rarely measured; audits of FDA's 603 potential signals from 2008 to 2019 start the clock at FDA's listing rather than at emergence. Extraction of labelled reactions from Structured Product Labels is now feasible, but no study has joined a validated labelled-reaction set to a dated FAERS scan across all drugs.

**Objective.** We will build a database-wide unlabeled-signal map for US prescription ingredients, date the emergence of every sustained signal, measure the lag to the first label naming it, and evaluate the scan against FDA's quarterly potential-signals list.

**Methods.** Prescription labels from openFDA (about 45,000 set ids) will be parsed by section and affirmed adverse-reaction terms mapped to MedDRA preferred terms with a dictionary matcher handling negation and hierarchy, validated against 100 manually annotated labels and the 200-label Demner-Fushman corpus. FAERS data from 2004 to 2025 (about 20 million reports, suspect drugs only) will yield cumulative quarterly 2x2 tables for roughly 600 ingredients with at least 1,000 reports; a sustained signal requires IC025 above 0 with at least three reports for two consecutive quarters, with ROR and PRR as sensitivity criteria, BH-FDR q-values, and false-emergence rates calibrated on OMOP and EU-ADR negative controls. Signals are flagged unlabeled after exact, synonym and HLT-sibling matching. For labelled pairs, the first label version naming the term will be located in DailyMed history and FDA Safety-related Labeling Changes; lags will be analysed by Kaplan-Meier with right censoring, label-first pairs excluded, and Cox regression clustered by drug. Detection rate and lead time will be computed against FDA entries from 2008 to 2025 with thresholds fixed a priori.

**Expected results.** We expect at least 25% of sustained signals to be unlabeled, more for drugs approved over 15 years ago; a median signal-to-label lag above 24 months, shorter for designated medical events; over 30% of signals for labelled events to emerge only after labelling; detection of at least 60% of FDA entries with a median lead of two or more quarters; and extractor F1 of at least 0.85.

**Significance.** The study provides a public unlabeled-signal map and a reproducible benchmark for future signal-detection methods.

### From FAERS Trajectory to Boxed Warning: Dynamic Landmark Prediction of FDA Safety-Related Labeling Changes with Lead-Time Estimation and a Notoriety-Bias Audit

*Project: [`label-change-prediction`](projects/label-change-prediction/)*

**Background.** Most post-approval safety labeling changes (SLCs) cite spontaneous reports, and the FDA SrLC database together with DailyMed SPL version histories now dates the quarter in which an adverse event entered a label's Boxed Warning or Warnings and Precautions section. Earlier prediction of label changes was cross-sectional, could not say when a signal became predictive, and risked using reports stimulated by the change itself.

**Objective.** We will build a dated pair-level label-event benchmark and a landmark survival model that, from FAERS reports received up to each landmark quarter, predicts an SLC within 8 quarters, estimate the lead time between first sustained signal and labeling, and quantify how much cross-sectional designs are inflated by notoriety bias.

**Methods.** SrLC exports and DailyMed LOINC-section diffs will yield (drug, MedDRA PT, section, quarter) events, with ~2,000 boxed and W&P rows dual-annotated and pairs already labelled at the first available SPL excluded. Quarterly openFDA counts give cumulative 2x2 tables per pair; BCPNN IC and IC025, 4- and 8-quarter IC slopes, consecutive positive quarters, report volume, serious and healthcare-professional shares, drug age and a class-warning flag are computed with nothing dated after the landmark. Landmarks every second quarter from 2008Q1 to 2023Q4 define risk sets of pairs with >= 3 reports and no prior event. A pooled logistic hazard model with landmark interactions (primary) and gradient boosting (secondary) are compared with static-IC025 and drug-age baselines under 5-fold CV grouped by ingredient and a temporal split at 2019Q4. The primary metric is IPCW time-dependent AUC at 8 quarters, plus 2 and 4, with Brier score and calibration slope; the null AUC comes from 999 permutations of event quarters within ingredient.

**Expected results.** We anticipate time-dependent AUC >= 0.75, exceeding static IC025 by >= 0.05 and drug age by >= 0.15; a median lead of >= 4 quarters for post-marketing-driven SLCs versus near zero for trial-driven ones; higher predictability for W&P additions than for boxed warnings; a class-warning hazard ratio >= 2; a post/pre reporting ratio >= 1.5 with leaky designs inflating AUC by >= 0.05; and temporal AUC within 0.05 on 2020-2024 landmarks.

**Significance.** The dated event table and strict-cutoff evaluation would turn pharmacovigilance prediction into a dynamic, leakage-safe benchmark and show how much earlier FAERS could have called a labeling change.

### A Sex-Stratified Drug-Drug Interaction Signal Atlas from FAERS 2015-2026 Benchmarked Against Clinical Reference Sets and Reporter-Flagged Interactions

*Project: [`faers-ddi-signals`](projects/faers-ddi-signals/)*

**Background.** Spontaneous-report drug-drug interaction (DDI) detection has a mature toolkit: the interaction reporting odds ratio (IOR) and the shrinkage observed-to-expected ratio Omega. Mechanism-informed Bayesian DDI signalling in FAERS has been reported. Yet every DDI resource pools sexes although women file more reports and differ in CYP3A4 activity and QT interval, the reporter-assigned "interacting" role code that FAERS already contains has never been used as a label, and the no-interaction model is rarely tested.

**Objective.** We will build a sex-stratified DDI signal atlas from FAERS 2015-2026 with a formal three-way sex interaction test and same-sex backgrounds, benchmark Omega, IOR, RERI and a hierarchical mechanism-pooled shrinkage prior against clinical reference sets on modern FAERS, and use the interacting role code as a native positive label.

**Methods.** openFDA reports received 2015-01 to 2026-06 (about 12 million) will be deduplicated by CASEID/CASEVERSION. A 300-drug panel extended by ONC, CredibleMeds and FDA CYP tables gives pairs co-reported at least 20 times are crossed with MedDRA PTs having at least 100 reports. Report-level eight-cell tables, pooled and per sex, yield Omega under additive, multiplicative, independence and maximum models, IOR with likelihood-ratio tests, RERI and a ratio-of-IORs sex interaction; log-IOR is shrunk toward mechanism-group means by empirical-Bayes pooling. Evaluation uses AUROC, average precision and precision at k against ONC and mechanism cross-products with explicit negatives from unrelated classes, pair-level bootstrap and Benjamini-Hochberg within families. Nulls permute sex labels within pair-event strata and drug-B labels across reports; indication-stratified backgrounds and topical negative-control pairs test confounding. Reporting follows READUS-PV.

**Expected results.** We expect the mechanism-pooled posterior to gain at least 0.05 AUROC over Omega025 and IOR bounds, mostly for pairs with fewer than 20 co-reports; Kendall tau below 0.8 between no-interaction models; recall of at least 0.6 for reporter-flagged pairs at 5% FDR, enriched for pharmacokinetic mechanisms; at least 5% of significant signals with a sex ratio of IORs beyond 1.5 or 0.67, stronger QT-class interactions in women and none for statin myopathy; and same-sex backgrounds disagreeing with pooled estimates in at least 10% of signals.

**Significance.** The atlas is a sex-resolved successor to pooled DDI resources and shows which statistics and backgrounds practitioners should trust.

### Exposure, reporting propensity or biology? Decomposing the female predominance of GLP-1 receptor agonist adverse-event reports in FAERS

*Project: [`glp1-sex-stratified-pv`](projects/glp1-sex-stratified-pv/)*

**Background.** Every FAERS analysis of GLP-1 receptor agonists (GLP-1 RAs) notes that 65 to 75 percent of reports come from women, and recent disproportionality studies call for sex-stratified work. None converts counts into rates per user, adjusts sex-stratified signals for indication, uses sex-specific backgrounds, screens the full MedDRA vocabulary with multiplicity control, or compares compounded with branded products. A raw female share cannot distinguish who takes the drug, who reports, and who is harmed.

**Objective.** We will decompose the female excess in GLP-1 RA reports into exposure, reporting propensity and residual sex-specific disproportionality, and produce a preferred-term-wide atlas of sex differences with indication adjustment, active comparators and label expectedness.

**Methods.** From openFDA and FAERS quarterly files (about 0.5 million exposed reports, 2018 to mid-2026, deduplicated) we will identify reports for seven GLP-1 RAs, classify indication from free text with brand fallback (validated on 500 hand-checked reports), flag compounded products, and tag four comparator classes. Survey-weighted sex- and age-specific user counts from MEPS and NHANES yield reporting rates per 10,000 users with Poisson intervals and an age-standardised propensity ratio. For every preferred term with at least three reports per sex we will compute female and male reporting odds ratios against same-sex backgrounds (a 10 percent FAERS sample or each comparator class), a likelihood-ratio test for drug-by-sex interaction, and Benjamini-Hochberg control; weighted logistic models and Mantel-Haenszel pooling adjust for indication and age band. SPL labels classify each term as labelled or unlabelled by brand. Nulls permute sex within exposure, indication and age strata; negative-control terms are pre-specified; sensitivity analyses cover suspect-only roles, consumer reports and the 2023 media peak. Reporting follows READUS-PV.

**Expected results.** We expect most of the raw female excess to be exposure, with a propensity ratio of 1.1 to 1.3 and an age-standardised rate ratio below 1.5; at least 5 percent of signals to show a sex interaction, female-enriched in alopecia, cholelithiasis and psychiatric terms and male-enriched in pancreatitis and acute kidney injury; at least a third of crude differences to attenuate after indication adjustment; and compounded-product reports to carry more medication-error terms.

**Significance.** The atlas replaces descriptive female shares with a decomposition that pharmacovigilance and prescribing guidance can act on.

### Machine-readable labelled age floors turn FAERS into an age-off-label pharmacovigilance system for children

*Project: [`pediatric-offlabel-signals`](projects/pediatric-offlabel-signals/)*

**Background.** Off-label prescribing to children is common and linked to adverse reactions in hospital cohorts, but spontaneous-report studies infer off-label use informally, drug by drug, because on-label status is not coded in FAERS. Yet onset age is recorded to the day for most paediatric reports, and every SPL states its paediatric age floor in stereotyped phrasings. Hundreds of BPCA/PREA labelling changes, each converting an off-label age range into a labelled one, are natural experiments that reporting data have never been used to evaluate.

**Objective.** We will build a machine-readable age-floor resource from SPL text, classify every paediatric FAERS report as on-label-age, below-floor or no-paediatric-labelling, and test whether below-floor reports are more serious and carry different, more often unlabelled, events.

**Methods.** From openFDA drug/label (about 50k prescription labels) a regex extractor will derive the minimum established age from pediatric_use and indications text, validated against 300 held-out hand-labelled labels and the FDA Pediatric Labeling Changes table (about 900 rows). Paediatric FAERS reports 2015 to 2026 (about 1.5 million) will be classified per suspect drug. For each drug and preferred term we will compute Mantel-Haenszel reporting odds ratios within 1-year age strata for below-floor and on-label reports and their ratio, Benjamini-Hochberg across terms per drug, and enrichment in medication-error and unlabelled terms. Seriousness and death will be modelled by logistic regression with 1-year age, sex, reporter type and drug effects. Segmented Poisson interrupted time series 36 months around each floor-lowering labelling change will be pooled by random-effects meta-analysis. Nulls include permuted label class within age-by-drug strata, placebo change dates and negative-control drugs.

**Expected results.** We expect at least 90 percent exact agreement of extracted floors with hand labels; 25 to 40 percent of paediatric reports to be below-floor or without paediatric labelling, highest in infants; adjusted odds of serious outcome of at least 1.3 for below-floor reports; at least 10 percent of drugs to show a below-floor excess term enriched for dosing errors and unlabelled events; a fall in below-floor share after labelling changes; and a downward bias of at least 5 points when reports with only an age group are excluded.

**Significance.** The floor table is a reusable public resource, and the design lets regulators monitor age-off-label harm at scale and measure what labelling changes achieve.

### Launch-Aligned, Label-Controlled Comparison of Biosimilar and Originator Adverse-Event Profiles in FAERS Across the 2023-2025 Subcutaneous Immunology Wave

*Project: [`biosimilar-ae-profiles`](projects/biosimilar-ae-profiles/)*

**Background.** FAERS comparisons of biosimilars with originators are confounded by design: a new biosimilar is compared against an originator's entire reporting history despite the early post-launch reporting peak, labels are identical by construction, reporter mix differs, and INN-only reports cannot be attributed. Existing analyses cover oncology antibodies to 2022 and suffix use in biosimilar reports. The 2023-2025 wave of self-injected immunology biosimilars (ten adalimumab products with different autoinjectors, plus ustekinumab, denosumab, tocilizumab and aflibercept) has not been compared, and device, injection-site and nocebo-type events are usually discarded.

**Objective.** We will test whether biosimilar and originator adverse-event profiles differ once launch timing, label expectedness and reporter type are controlled, with device and effectiveness panels as secondary endpoints, and report INN-only fractions per family and quarter as the denominator bounding every comparison.

**Methods.** FAERS reports from 2015 to 2026 for catalogued families will be extracted, deduplicated by case identifier, and attributed by the hierarchy biosimilar brand, suffixed INN, originator brand, INN-only, with conflicts flagged. Each biosimilar's first 24 months from its Purple Book launch will be compared with the originator in the same calendar months and with sibling biosimilars at equal months since launch. Profile difference is Jensen-Shannon divergence on the top-200 preferred-term vocabulary with report-level permutation, the null being random halves of the originator; per-term reporting odds ratios use Benjamini-Hochberg and reporter stratification, with product-by-reporter interaction models for effectiveness, injection-site, device and substitution panels. SPL sections classify terms as labelled or unlabelled. Launch curves yield early/late ratios by interchangeability with permuted launch dates as null; filgrastim families serve as negative controls; 200 reports per product-window is the minimum.

**Expected results.** We expect calendar-aligned divergence within the originator split null for at least 70% of biosimilars, exceptions concentrated in device and injection-site panels, effectiveness and substitution enrichment carried by consumer rather than physician reports, unlabelled fractions differing by under 5 points among clinician reports, and sibling adalimumab products differing in device panels.

**Significance.** The design separates molecule from device, reporter and launch effects, giving regulators a template for biosimilar surveillance.

### Do drug shortages leave a footprint in FAERS? A staggered difference-in-differences analysis of medication-error reporting for shortage drugs and their substitutes, 2012-2025

*Project: [`drug-shortage-ae-patterns`](projects/drug-shortage-ae-patterns/)*

**Background.** US drug shortages reached a record in early 2024, and substitution of concentration, vial size, route or molecule is exactly where handling errors occur. The closest precedent, a synthetic-control study of the 2017 heparin shortage, found medication-error reports rose by roughly 150% for heparin and 110% for enoxaparin, but covered one episode and one hand-picked substitute. FAERS codes medication errors as MedDRA terms across all drugs on a monthly axis, and openFDA now exposes the FDA Drug Shortage Database; staggered-adoption estimators have not been applied to pharmacovigilance panels.

**Objective.** We will estimate whether shortage onset raises the rate of medication-error and dosing-error reports for the index drug and for its therapeutic substitutes across US shortage episodes from 2012-2025, whether effects reverse at resolution, and what modifies them.

**Methods.** Exposure: all openFDA shortage records collapsed to one episode per ingredient, harmonised with RxNorm, with the ASHP list as a sensitivity list. Substitutes share Established Pharmacologic Class and route; never-shortage controls are matched 2:1 on class and 2012-2015 report volume. Outcomes from openFDA count queries: monthly primary-suspect reports per ingredient with medication-error, dosing-error, supply-issue and negative-control terms, and serious reports. Primary estimator: Callaway-Sant'Anna group-time ATTs on log error rates (offset by all reports of the drug) with not-yet-treated controls, aggregated over horizons 0-6 and 7-12 months and around resolution; secondary: Poisson event study with two-way fixed effects. Inference by drug-cluster bootstrap, placebo permutation of onset dates and joint Wald tests on leads -12 to -2. Robustness: FAERS quarterly-file rebuild, exclusion of pandemic months and GLP-1 agonists, and a Medicaid prescription denominator. Minimum baseline of 20 reports per month.

**Expected results.** We expect an index-drug IRR of at least 1.2 over months 0-6 with null pre-trends; substitute IRRs of 1.1-1.5, larger when few substitutes exist; larger effects for sterile injectables, narrow-therapeutic-index drugs and manufacturing-driven shortages; return toward baseline within 6 months of resolution; negative-control IRRs near 1; and shortage-period error reports no less serious.

**Significance.** A population-scale causal estimate of shortage-induced error reporting quantifies a harm that policy debates currently infer from anecdote, and the pre-registered design offers a credible template for external-exposure pharmacovigilance.

### Ingredient-level dietary supplement safety signals from CAERS: DSLD linkage, FAERS replication and reference-set validation

*Project: [`caers-supplement-signals`](projects/caers-supplement-signals/)*

**Background.** Dietary supplements are regulated as foods without pre-market safety review and account for a growing share of drug-induced liver injury. The FDA's CFSAN Adverse Event Reporting System (CAERS) records supplement events with verbatim product names but no ingredient coding, so existing analyses, including a 2026 product-name-level disproportionality study of 2004-2025 reports, cannot aggregate one botanical sold under many brands or decompose multi-ingredient products. No CAERS method has been validated against a reference set, compared with the supplement reports inside FAERS, or modelled around the 2007 mandatory serious-reporting rule.

**Objective.** To build and validate an ingredient-level signal resource from CAERS, test cross-system replication in FAERS, and quantify how reporting rules and media episodes shape the data.

**Methods.** We will extract all CAERS reports 2004-2026 (about 5 x 10^4 supplement reports, industry code 54, Suspect role), deduplicate manufacturer and consumer versions, and link product strings to ingredients through a curated lexicon and the NIH DSLD (about 150,000 labels), validating precision on 300 hand-checked products. For every ingredient-PT pair with a >= 3 we will compute ROR, PRR and BCPNN IC with BH correction against supplement-only and all-CAERS backgrounds, alongside a product-level screen. Performance will be evaluated as AUROC against LiverTox/DILIN hepatotoxins versus nutrient and probiotic controls, with bootstrap CIs and shuffled-label nulls. The lexicon will be applied to FAERS verbatim drug names to compute Spearman concordance of log-RORs and CUSUM first-alarm dates. Segmented quasi-Poisson interrupted time series at December 2007 and around the Hydroxycut (2009) and OxyElite Pro (2013) episodes will use placebo break dates; tainted-product listings will be matched to pre-listing reports; NHANES supplement-use prevalence will supply denominators.

**Expected results.** We expect linkage to assign an ingredient to at least 70% of supplement reports, ingredient-level screening to recover known hepatotoxins that product-level screening splits across brands, ingredient-level AUROC to exceed product-level AUROC by at least 0.10, moderate CAERS-FAERS concordance, a level increase in serious but not non-serious reports after 2007, and pre-listing signals for a quarter of adulterated products.

**Significance.** This would convert CAERS into an ingredient-level pharmacovigilance resource with an open signal table and linkage code, giving regulators and hepatologists testable hypotheses.

### Dogs and Cats as Pharmacovigilance Sentinels: Cross-Species Concordance and Lead-Lag of Adverse-Event Signals Between openFDA Veterinary and Human Reporting Systems

*Project: [`animal-drug-cross-species-signals`](projects/animal-drug-cross-species-signals/)*

**Background.** The FDA Center for Veterinary Medicine database holds about 1.3 million adverse-event reports, mostly in dogs and cats, for many ingredients also used in humans and reported to FAERS. Animal-human toxicity concordance has been quantified only for controlled preclinical studies; whether spontaneous reports in companion animals carry information about human risks has not been tested, and no open crosswalk links VeDDRA to MedDRA.

**Objective.** We will quantify ingredient-wide concordance, discordance and lead-lag of disproportionality signals between canine, feline and human reports at a harmonised organ-system level, release the crosswalk, and test whether known species-specific pharmacology is recovered as discordance.

**Methods.** All CVM reports will be harvested and flattened; FAERS counts per ingredient, preferred term and quarter will come from primary-suspect count queries. The shared set comprises RxNorm-normalised ingredients with at least 100 reports in each system (about 150 expected). VeDDRA and MedDRA terms will be mapped to 22 organ-system buckets by a rule-based classifier refined with official hierarchies; two reviewers blind to signal values will check the top 300 VeDDRA terms. BCPNN information components will be computed per species and cell against each system's own totals. Concordance is Spearman rho on IC and Cohen's kappa on IC025 flags with ingredient-cluster bootstrap and 999 ingredient-label permutations; pre-registered discordant controls (cat-acetaminophen, dog-ivermectin, dog-NSAID, cat-permethrin) and concordant controls (gabapentin, insulin, anticoagulants, levothyroxine) will be scored in a two-by-two table. Lead-lag uses the first sustained IC025 above zero per quarter in a mixed model with mechanism match and marketing order; mg/kg exposure from dose and weight fields will be related to seriousness with ingredient random intercepts. Benjamini-Hochberg applies within families; robustness excludes manufacturer-only, lack-of-efficacy and pre-VeDDRA-v3 reports.

**Expected results.** We expect rho of at least 0.3 and kappa above 0.2, higher concordance in pharmacodynamic than in immune or idiosyncratic organ systems, recovery of the pre-specified discordant pairs, animal signals preceding human signals by a median of at least four quarters where the animal product was marketed first, and mg/kg dose ratio predicting seriousness.

**Significance.** The study tests whether companion-animal pharmacovigilance is a usable sentinel for human safety and supplies the first open VeDDRA-MedDRA crosswalk.

### Predicate-chain depth, recalled ancestry and early MAUDE signals as predictors of FDA device recall: a landmark survival analysis of 510(k) and PMA devices

*Project: [`device-recall-prediction`](projects/device-recall-prediction/)*

**Background.** Most moderate-risk devices reach the US market through 510(k) substantial equivalence to predicates that were themselves cleared on predicates, and post-market safety depends on the passive MAUDE system. Among 156 Class I-recalled 510(k) devices from 2017-2021, 44.1% cited predicates that had been Class I recalled; AI-enabled devices are recalled early; and a deep-learning model on the 45,398-device citation network predicts recall but gives no interpretable hazard estimates and uses no MAUDE inputs.

**Objective.** We will estimate interpretable hazard ratios for predicate depth and time-anchored recalled ancestry across all product codes, test whether the first 6-12 months of MAUDE reports predict subsequent Class I/II recall in a landmark design, and whether the two signals are complementary, with cardiovascular, orthopaedic, insulin-pump/CGM and AI-enabled subgroups.

**Methods.** Cohort: all 510(k) clearances and original PMAs from 2003-2020 (openFDA), follow-up to mid-2025. Outcome: first Class I/II recall linked by K/PMA number. Predicate chains will be extracted from 510(k) summary PDFs, assembled into a DAG and validated on 300 random edges (target precision 0.95); features include depth and ancestors recalled before versus after clearance. MAUDE reports will be linked via GUDID or fuzzy firm-brand matching; landmark features at 6 and 12 months comprise report counts, event mix, problem-code diversity, slope and narrative TF-IDF/SVD components fitted on training years only. Models: Cox proportional hazards stratified by product code with clearance-year splines and applicant-clustered SEs, plus Fine-Gray and gradient-boosted survival, against network-only, counts-only and regulatory baselines. Training on 2003-2014 clearances, validation 2015-2017, test 2018-2020; Harrell's C, time-dependent AUC and top-5% watch-list yield. Nulls: permuted predicate edges within product code and placebo landmark windows after recall.

**Expected results.** We expect hazard to rise per predicate generation (HR above 1.05), pre-clearance recalled ancestry to carry HR of at least 1.5 for Class I recall with a smaller post-clearance association, a 12-month MAUDE model to reach C-index of at least 0.70 with narrative text adding at least 0.03, and AI-enabled devices to show shallow chains but higher early hazard with sparser, less specific reports.

**Significance.** Interpretable, time-anchored hazard estimates and a temporally validated early-warning score give regulators a surveillance prioritisation tool and inform predicate-selection guidance.

### From First Serious MAUDE Report to Recall: A Left-Truncated Survival Analysis of the Signal-to-Recall Lag for FDA-Regulated Medical Devices

*Project: [`recall-lag-survival`](projects/recall-lag-survival/)*

**Background.** Recall determinants have been studied from the pre-market side, late manufacturer reporting to MAUDE has been measured, and classifiers predict recall status from report narratives. No published study treats the time between the first adverse-event signal in MAUDE and recall initiation as the outcome, decomposes it into reporting, detection and regulatory components, or asks how much lead time a simple surveillance statistic would have given.

**Objective.** We will model the lag from first serious MAUDE report to recall initiation with explicit truncation and censoring, decompose it into its components, compare strata by recall class, pathway, root cause and firm size, test for change after the 2019 end of Alternative Summary Reporting, and quantify the lead time of a Poisson CUSUM at a fixed false-alarm rate.

**Methods.** Using openFDA only, device keys will be product code by normalised firm name, with submission numbers as a secondary key and manual adjudication of 200 recalled keys. Monthly serious-report counts per key from device/event will be joined to device/recall and device/enforcement (about 100,000 records each). The origin is the first death or injury report or the first CUSUM crossing; the event is event_date_initiated; keys are right-censored at the data cut and left-truncated when the origin precedes reliable recall records, and reports after initiation never contribute to signals. We will fit Kaplan-Meier curves with entry times, firm-clustered Cox models, accelerated failure time models per component, segmented regression with a 2019 Q3 break and Newey-West errors, and lead-time versus false-alarm curves with thresholds tuned on panels disjoint from those evaluated. Nulls permute recall dates within panel and simulate Poisson streams without change points; reports received in the 90 days after initiation form a negative control. Lag, pathway and firm-size hypotheses are pre-registered, with Benjamini-Hochberg control across strata.

**Expected results.** We expect a median lag above 12 months for Class II recalls and shorter lags for Class I, PMA devices (hazard ratio above 1.2 versus 510(k)) and software-related recalls; larger firms faster from signal to initiation but slower from event to receipt; shorter lags after 2019 beyond trend; and a CUSUM flagging at least half of eventually recalled keys six or more months before initiation at no more than one false alarm per 100 device-key-years.

**Significance.** The study locates where recall delay lies and gives surveillance designers an actionable lead-time figure.

### What Breaks When AI-Enabled Medical Devices Fail: Linking the FDA AI Device List to MAUDE Narratives with an AI-Specific Failure-Mode Taxonomy and Denominator-Adjusted Comparator Rates

*Project: [`ai-device-failure-taxonomy`](projects/ai-device-failure-taxonomy/)*

**Background.** The FDA list of AI-enabled medical devices exceeds 1,300 entries, yet post-market evidence consists of counts of adverse-event reports and recalls per device. Existing MAUDE analyses of these devices are small manual classifications or tallies without a narrative taxonomy, device-year denominators or non-AI comparators, and FDA problem codes cannot express distribution shift or integration failure.

**Objective.** We will build and release a versioned linkage between the FDA AI-enabled device list and MAUDE, classify report narratives with an eight-category AI-specific failure-mode taxonomy, and estimate denominator-adjusted reporting rates against non-AI devices in the same product codes and authorisation years.

**Methods.** Using openFDA device/event, device/510k, device/pma, device/recall and GUDID, we will link submission numbers to product codes and applicants and match MAUDE reports by product code plus brand or manufacturer similarity in three tiers, adjudicating 300 pairs manually to report precision and recall with Wilson intervals. Narratives will receive rule-based weak labels with evidence spans (incorrect output; no or delayed output; input quality; integration; version/update; hardware; user/workflow; cybersecurity), a TF-IDF classifier, and double human annotation of 500 narratives with per-category kappa; brand tokens are removed before classification. Negative-binomial models with device-year offsets and covariates (panel, pathway, class, firm size) estimate incidence rate ratios per category against comparators cleared within two years; a self-controlled 180-day pre/post design around successive clearances tests update-related regressions; agreement between narrative labels and FDA product_problems codes is measured with kappa. Nulls come from shuffling AI status within product-code-by-year strata and applying the taxonomy to non-AI narratives; reporter stratification and post-recall exclusions address reporting bias.

**Expected results.** We expect algorithm-output failures to account for fewer than 40% of linked AI-device reports, incidence rate ratios above 1.5 for incorrect-output and integration categories with similar hardware rates, higher reporting in the 180 days after a new clearance, kappa below 0.4 between codes and narratives for incorrect output, and more user-facility and voluntary reports for AI devices.

**Significance.** The linkage table, taxonomy and annotation guideline give regulators a reusable, denominator-aware basis for AI device surveillance that structured codes cannot provide.


## Medical imaging: CT, CXR & musculoskeletal

### Predicting Radiologist Disagreement in Lung Nodule Screening: Multi-Rater Calibration, Second-Read Triage and Lung-RADS Consequences in LIDC-IDRI

*Project: [`lidc-nodule-uncertainty`](projects/lidc-nodule-uncertainty/)*

**Background.** In LIDC-IDRI four radiologists independently rated every nodule's malignancy without forced consensus, and they disagree in most cases. The field collapses this to a majority label and treats disagreement as noise, while distributional annotation models such as the Probabilistic U-Net and PHiSeg have stayed in segmentation. Nobody has asked whether disagreement is predictable from the image and evaluated it as a triage signal, compared soft-label with majority-vote training on a common probabilistic reference, or translated reader variance into Lung-RADS disagreement.

**Objective.** We will treat inter-rater disagreement as a prediction target, calibrate malignancy models on the readers' label distribution, and quantify the effect of both on Lung-RADS assignment, second-read triage yield and false-positive rate at fixed sensitivity.

**Methods.** From LIDC-IDRI (1,018 cases; ~2,600 nodules with >= 3 readers), each nodule carries every reader's rating, diameter and contour; nodules with fewer readers are reported separately. Shape, attenuation, boundary-sharpness and texture features (fixed 25 HU bin width, 1 mm isotropic) feed gradient-boosting and logistic models. Soft-label training uses weighted positive and negative rows (exact soft cross-entropy) and is scored against a common reference soft label on identical folds. A disagreement model is evaluated by recall, precision and lift at 5-50% review budgets against threshold-proximity (distance to 6, 8 and 15 mm) baselines. Lung-RADS v2022 categories are assigned from each reader's own diameter. Validation is 5-fold GroupKFold by scan, scan-level cluster bootstrap (2,000 replicates), Benjamini-Hochberg over H1-H7, and nulls from within-scan label permutation, feature permutation and reader-identity permutation. Duke DLCS (1,613 volumes, 2,487 nodules) is scored once with frozen models.

**Expected results.** We expect out-of-fold Spearman rho > 0.4 and AUC > 0.70 for boundary-crossing disagreement, exceeding threshold proximity by > 0.05 AUC; >= 30% of such disagreements recalled at a 10% review budget; >= 30% lower expected calibration error from soft labels with no AUC loss, with the split-3 convention best; different Lung-RADS categories from different readers in >= 10% of nodules; and >= 15% fewer false positives at 95% sensitivity when the top disagreement decile is deferred.

**Significance.** Reframing disagreement as a target turns an unused LIDC measurement into a screening-programme quantity and extends multi-rater modelling from masks to grading and management.

### Not Missing at Random: A Multi-Source Audit of Differential Label Noise in Report-Derived Chest Radiograph Labels and Its Effect on Benchmarks and Fairness Claims

*Project: [`radiology-report-weak-supervision-audit`](projects/radiology-report-weak-supervision-audit/)*

**Background.** Chest radiograph AI is trained and evaluated on labels extracted from reports by CheXpert, NegBio, CheXbert and local LLMs. Small re-reads show frequent disagreement with expert image labels, and a recent cardiomegaly case study attributed the gap mainly to non-mention rather than negation. Whether that disagreement is differential, and how much of published rankings and underdiagnosis gaps it generates, is unmeasured at scale.

**Objective.** We will build per-finding, per-labeler noise matrices against expert image labels, test whether it is differential by view, portable acquisition, care setting and demographics, and quantify its consequences for fairness metrics and model rankings by calibrated simulation and re-evaluation.

**Methods.** Expert labels from Chest ImaGenome (500 gold studies), REFLACX (about 3,000 images) and MS-CXR on MIMIC-CXR v2.1, plus CheXpert validation and test sets (700 studies), VinDr-CXR (18,000 three-radiologist images), the NIH adjudicated subset (about 4,000 images), will be mapped to the 14 CheXpert findings. Reports will be labelled with CheXpert, NegBio, CheXbert, a transparent rule-based labeler and an on-premises LLM labeler with a frozen prompt. Sensitivity, specificity, kappa and PABAK will be estimated under both non-mention conventions with patient-clustered bootstrap intervals. Differential noise will be modelled by logistic regression of disagreement on view, setting, demographics and report-structure features with cluster-robust errors, Holm correction, E-values, and permutation of subgroup labels within setting as the null. Measured subgroup-specific noise rates will be injected into true labels to compute the false-negative-rate gap a perfect classifier would display. At least five public models, screened for training overlap at study level, will be scored on expert versus report labels with Kendall tau on rankings.

**Expected results.** We expect sensitivity below 0.6 for cardiomegaly, atelectasis and effusion with specificity above 0.9; odds of non-mention false negatives above 2 for ICU portable films and above 3 for comparison-style reports, with adjusted demographic odds ratios near unity; simulations reproducing 30 to 60% of reported no-finding gaps; rankings with tau below 0.7 for at least three findings; and LLM labelers reducing negation errors without altering the differential structure.

**Significance.** The audit yields reusable noise matrices, a noise-injection simulator and a reporting checklist that separate label artefacts from genuine model bias in weak-label benchmarks.

### Uncertainty that means something: competing-risk knee osteoarthritis progression models in the OAI whose uncertainty tracks reader disagreement and transports to MOST

*Project: [`oai-progression-uncertainty`](projects/oai-progression-uncertainty/)*

**Background.** Deep learning on Osteoarthritis Initiative (OAI) images predicts Kellgren-Lawrence (KL) grade, progression and total knee replacement (TKR), and recent work has transferred OAI-trained models to the MOST cohort and reported conformal sets for KL and MOAKS targets. Progression is still a binary label at a fixed horizon, ignoring TKR and death as competing events; uncertainty is a single scalar; and grade noise (inter-reader kappa about 0.5 to 0.7) is never linked to model uncertainty, although OAI holds repeat and adjudicated readings.

**Objective.** We will build cause-specific progression models with decomposed uncertainty and test whether epistemic uncertainty concentrates where readers disagree, whether conformal coverage holds within strata, and how much MOST calibration data restores coverage after transport.

**Methods.** From OAI (4,796 participants) we will construct a knee-level cohort with baseline KL 0 to 3 and visits to 96 months; events are KL or medial JSN increase, TKR and death, censored at the last completed visit. Deep ensembles (M = 5) of image encoders with a discrete-time competing-risk head, clinical covariates, an MRI arm using OAI-ZIB cartilage masks (507 scans) and Fine-Gray references are trained with 5-fold participant-grouped cross-validation and one locked hold-out fold. Uncertainty is split into aleatoric (expected entropy) and epistemic (mutual information) parts; Mondrian conformal sets target 90 percent coverage. Evaluation uses cause-specific C-index, IPCW Brier at 48 and 96 months, subgroup coverage by sex, race, BMI class and site (Holm-corrected), 2,000 participant-level bootstraps, permuted reader-disagreement labels and label-shuffled models. Frozen models are applied once to MOST baseline radiographs (about 3,000 participants), recalibrating conformal thresholds on 100 to 1,000 knees.

**Expected results.** We expect competing-risk rankings to diverge from binary 48-month classifiers (rank correlation below 0.9) with better Brier scores, epistemic uncertainty to predict reader disagreement with AUROC of at least 0.65 while aleatoric uncertainty adds nothing, coverage differences across strata under 5 points, MOST discrimination to fall by less than 0.05 while coverage drops below nominal until recalibrated on a few hundred knees, and MRI to add at least 0.03 C-index for progression only.

**Significance.** This reframes OA progression modelling around the outcome's real structure and gives a reusable validity test for imaging uncertainty: does it track human disagreement and survive transport?


## Methods, meta-science & benchmarks

### leakscan: A Subject-Aware Static Leakage Detector and a Code-Level Audit of Leakage Prevalence and Metric Inflation in Biomedical Machine Learning

*Project: [`leakage-detector-biomedical-ml`](projects/leakage-detector-biomedical-ml/)*

**Background.** Leakage drives irreproducible machine-learning claims, and in biosignal studies the dominant form is subject-level: windows from one person land in both training and test sets. Paper-reading audits find this common, but methods text under-describes splitting. Existing static detectors find row overlap, test-set reuse and preprocessing leakage, but none models grouping structure, flags resampling before the split, or has been tested on EEG, ECG or EHR code.

**Objective.** We will validate an AST-based detector for group-level and preprocessing leakage, estimate prevalence from public code linked to CHB-MIT, PTB-XL, MIMIC, Bonn, DEAP and Sleep-EDF, test whether flagged repositories report inflated metrics, and confirm inflation by re-execution.

**Methods.** leakscan parses scripts and notebooks into ordered events (splits, transformer and selector fits, resamplers, estimator fits, evaluations) with the variables they touch, infers subject and window structure from identifiers, and applies rules such as split-without-groups, window-level-random-split, preprocess-before-split and test-set-in-training. Validity is precision, recall and F1 with Wilson CIs on a mutation corpus (20 clean pipelines with templated injections) and 100 repositories annotated by two blinded annotators (adjudicated, Cohen's kappa), against the Yang et al. notebook analysis (McNemar) and a keyword baseline. Corpora of 300-1,500 repositories per dataset come from GitHub and Semantic Scholar; prevalence is modelled by logistic regression on dataset, year, notebook versus script and paper linkage. Inflation is tested by regressing the headline metric on the leakage flag, adjusted for year, model family and evaluation unit, with a flag-permutation null. Twenty flagged CHB-MIT and ten PTB-XL repositories are re-run with only the splitter made subject-wise; paired drops are bootstrapped. Thresholds are frozen on the mutation corpus.

**Expected results.** We expect precision >= 0.80 and recall >= 0.70 for the core rules, lower on notebooks; group-unaware window splits in >= 40% of CHB-MIT repositories but < 20% for PTB-XL and < 30% for MIMIC; odds ratios > 2 for preprocessing co-occurrence and for notebooks; a >= 5-point higher median metric in flagged CHB-MIT papers; >= 10-point drops on re-execution in >= 75% of cases; and prevalence falling after 2022 but staying > 25%.

**Significance.** This would be the first leakage prevalence estimate derived from code rather than prose, tying a detector to the inflated numbers the field cites and giving reviewers a tool.
