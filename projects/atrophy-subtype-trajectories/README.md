# atrophy-subtype-trajectories — Do cross-sectional atrophy subtypes and stages hold up longitudinally, and what do they look like when amyloid is absent?

Event-based / SuStaIn-style subtype-and-stage models of regional atrophy fitted on OASIS-3 and ADNI, audited against each subject's own follow-up MRIs (stage monotonicity, subtype stability, coupling to CDR change), with dedicated models for the amyloid-negative cognitively impaired ("SNAP"/non-AD) group and white-matter-hyperintensity (FLAIR) burden as an explicit event.

## Status / difficulty / timeline / compute

- Status: design + starter code (longitudinal table builder, control-referenced z-scoring, compact z-score EBM with EM subtyping, longitudinal consistency metrics with permutation nulls; pySuStaIn hook).
- Difficulty: MSc/early-PhD, 8-12 months. The modelling is CPU-only; the time goes into data curation (FLAIR/WMH segmentation, amyloid status over time) and the validation design.
- Compute: FreeSurfer outputs are provided by OASIS-3 (v5.3) and ADNI (UCSF); WMH segmentation from T1+FLAIR (FreeSurfer 7 `samseg --lesion`, LST-AI or BIANCA) costs 20-40 CPU-min per session. pySuStaIn with ~30 events and ~1,000 subjects takes hours per subtype count on a 16-32-core node (10-fold CV for model selection: a few days). The built-in compact EBM runs in minutes and is used for prototyping and null models.

## Background

SuStaIn (Young et al., 2018, Nature Communications) infers disease subtypes that differ in the order in which regions become abnormal, and stages each person along their subtype's sequence, from cross-sectional data. It has produced the widely cited "typical / limbic-predominant / hippocampal-sparing / cortical" atrophy subtypes in AD and tau-PET subtypes (Vogel et al., 2021, Nature Medicine). The core assumption is that a cross-sectional ordering of abnormality is a temporal ordering. That assumption is testable when subjects have repeated scans: their stage should not go down, their subtype should not change, and stage change should track clinical change. OASIS-3 (LaMontagne et al., 2019) provides exactly this: many subjects with 2-6 MR sessions over up to ~15 years, amyloid PET (PiB/AV45, Centiloid), FLAIR, and CDR at each visit.

## The research gap

What has been done:

- Young et al., 2018: SuStaIn on ADNI MRI; longitudinal consistency shown descriptively.
- Archetti et al., 2021, Frontiers in Big Data ("Inter-cohort validation of SuStaIn model for AD"): subtypes replicate across cohorts (test set of 767 subjects from OASIS, PharmaCog and ViTA).
- "How reproducible are data-driven subtypes of Alzheimer's disease?", Journal of Alzheimer's Disease, 2025: 5,444 subjects from ANMerge, OASIS and ADNI; reproducibility of subtypes across cohorts and with/without controls. Cross-sectional only.
- Baumeister et al., 2025, Alzheimer's & Dementia (DZNE): a fast pipeline that retrains SuStaIn in DELCODE (n = 813), ADNI (n = 2,117) and A4/LEARN, reporting two subtypes (limbic-predominant, hippocampal-sparing) with strong within-subject subtype stability and monotonically increasing stages at follow-up. This is the closest prior work: it establishes longitudinal consistency in amyloid-defined AD samples.
- "Subtyping and staging of Alzheimer's disease from routine structural MRI" (2026) extends to clinical-grade scans.
- A4 pre-randomization SuStaIn (three subtypes in cognitively unimpaired A+; Journal of Prevention of Alzheimer's Disease, 2024).
- WMH progression subtypes with SuStaIn in 9,179 stroke patients + UK Biobank (Nature Communications, 2025: fronto-parietal, radial, temporo-occipital subtypes). Atrophy was not modelled jointly.
- SuStaIn in non-AD conditions: corticobasal syndrome (Brain Communications, 2025), ALS deformation-based morphometry (Imaging Neuroscience, 2025), ALS-FTD spectrum (Translational Neurodegeneration, 2023), epilepsy (2023).

What is specifically missing:

1. Longitudinal validity with a proper null and stratified by amyloid status. Reports of "monotonically increasing stages" are given without a within-subject session-order permutation null (with many ties at early stages, 60-70% non-decreasing pairs can arise by chance) and without a fair single-region comparator (does the stage track CDR-SB change better than hippocampal volume alone?). Baumeister et al. tested AD (amyloid-positive or clinically defined) samples; OASIS-3 has not been used for a longitudinal audit.
2. The amyloid-negative cognitively impaired group. AD subtype models are trained on A+ or clinical AD and A- impaired individuals are excluded or forced into AD sequences. A-CI (a substantial fraction of memory-clinic patients; heterogeneous non-AD pathologies such as LATE/TDP-43, vascular, PART, argyrophilic grain; Jack et al., 2012 defined SNAP; Nelson et al., 2019 defined LATE) has no dedicated data-driven subtype-and-stage model with longitudinal validation. A limbic-predominant, hippocampal-sparing/vascular or WMH-led sequence in A-CI would be a testable, clinically meaningful result, and so would the finding that A-CI stages do not progress monotonically (i.e. that cross-sectional heterogeneity there is not a progression axis).
3. WMH as an event in atrophy models. WMH subtypes exist (stroke) and atrophy subtypes exist (AD), but a joint atrophy + WMH model in an aging/AD cohort with FLAIR, with the question "does vascular burden define its own subtype or does it just add a stage offset?", has not been reported. OASIS-3's FLAIR + PET + longitudinal design is uniquely suited.
4. Subtype-assignment uncertainty as a function of stage. Early-stage subjects are assigned subtypes with high posterior entropy; longitudinal instability should be reported conditional on baseline stage and posterior confidence rather than as a single percentage.

## Research questions / hypotheses

1. RQ1 (stage monotonicity). Fit the model on baseline sessions only; stage follow-up sessions with the frozen model. H1: in A+ CI, ≥ 80% of consecutive session pairs are non-decreasing in stage, exceeding the session-order permutation null by ≥ 20 percentage points; mean stage change ≥ 1 event per 2 years.
2. RQ2 (subtype stability). H2: ≥ 80% of subjects keep their modal subtype at follow-up when baseline posterior confidence ≥ 0.7 and stage ≥ 4; stability is < 60% for stage ≤ 2 (near-zero z-scores are uninformative).
3. RQ3 (A-CI model). Fit a separate model on A-CI (CDR ≥ 0.5, all amyloid PET negative, latest PET used). H3: A-CI yields a limbic/medial-temporal-first subtype (LATE-like) and a WMH-led subtype; monotonicity is lower than in A+ CI but above the null for the limbic subtype.
4. RQ4 (WMH as event). H4: adding WMH volume (z-scored against A-CN controls) as an event improves cross-validated log-likelihood (CVIC) and increases longitudinal consistency in A-CI more than in A+ CI.
5. RQ5 (clinical coupling). H5: within-subject Δstage explains ΔCDR-SB better (mixed-model likelihood, marginal R²) than Δhippocampal-volume z alone in A+ CI; in A-CI the difference is smaller.
6. RQ6 (replication). Repeat RQ1-RQ5 in ADNI (FLAIR available from ADNI-GO onwards; UC Davis WMH volumes) and report subtype correspondence (sequence Kendall τ) between cohorts.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| OASIS-3 | Longitudinal T1 + FLAIR, FreeSurfer 5.3 (aseg/aparc), PiB/AV45 PET Centiloid (PUP), CDR / CDR-SB / MMSE per visit, APOE | ~1,300 participants, ~2,800 MR sessions (≈ 60% with ≥ 2 sessions), ~1,500 amyloid PET | Free registration + DUA on NITRC-IR (XNAT) | https://sites.wustl.edu/oasisbrains/ , https://www.nitrc.org/projects/oasis3/ |
| ADNI (GO/2/3/4) | Longitudinal T1 + FLAIR, UCSF FreeSurfer tables, UC Berkeley Centiloid tables, UC Davis WMH volumes, ADNIMERGE (CDR-SB, MMSE, diagnosis) | ~2,000 participants with amyloid PET; several thousand longitudinal MR sessions | Application via LONI IDA | https://adni.loni.usc.edu/ |
| OASIS-4 (optional) | Clinical memory-clinic cohort, T1 + FreeSurfer, CSF subset (Aβ42/40 for A status), no PET | 663 individuals | NITRC DUA | https://www.nitrc.org/projects/oasis4/ |
| pySuStaIn | Reference implementation (z-score, mixture and ordinal models) | software | open source (MIT) | https://github.com/ucl-pond/pySuStaIn |

Amyloid status over time: a subject is A+ from the first Centiloid-positive PET onward (PiB ≥ 16.4 CL, AV45 ≥ 20.6 CL per OASIS-3 documentation; sensitivity analyses at ±5 CL); A- sessions are those preceded and followed (within 2 years) by negative PET only. A-CI requires CDR ≥ 0.5 at the session; "SNAP" additionally requires neurodegeneration (hippocampal z ≤ -1 vs A-CN controls), reported as a sub-analysis.

## Methods

1. Timeline construction (`atrophy_subtypes.longitudinal`): parse OASIS ids to (subject, day); attach to each MR session the nearest clinical visit (±180 d), the amyloid status by the "once positive, always positive" rule, FreeSurfer regional volumes/thickness, eTIV, and WMH volume (FreeSurfer `WM-hypointensities` as the T1-only proxy; SAMSEG-lesion/LST-AI from T1+FLAIR as the primary measure). Define groups A-CN, A+CN, A+CI, A-CI; build consecutive-session pairs with Δyears.
2. Control-referenced z-scores (`atrophy_subtypes.zscoring`): regress each feature on age, sex and eTIV in A-CN controls (baseline sessions), z-score all sessions against the control residual SD, flip sign so that higher z = more abnormal (volumes/thickness negative, ventricles/WMH positive).
3. Features: 13-20 regions as in the AD SuStaIn literature (hippocampus, amygdala, entorhinal, parahippocampal, fusiform, middle/inferior temporal, precuneus, posterior cingulate, inferior parietal, supramarginal, superior frontal, insula, caudate, putamen, thalamus, lateral ventricles, WMH), left/right averaged; z-thresholds 1, 2, 3 (z_max 5) as in Young et al., 2018.
4. Models (`atrophy_subtypes.ebm`): compact z-score EBM with piecewise-linear expected trajectories and a uniform stage prior (Fonteijn et al., 2012; Young et al., 2018), sequence optimization by greedy event moves with restarts, EM over subtype memberships. Used for rapid iteration, simulations and nulls. Final models with pySuStaIn (`ZscoreSustain`, 1e5-1e6 MCMC iterations, 10-fold CVIC for K = 1..5). Models are fitted on baseline sessions only.
5. Longitudinal audit (`atrophy_subtypes.longitudinal_metrics`): stage monotonicity with a within-subject session-order permutation null; subtype stability vs. baseline stage and posterior confidence, with the marginal-frequency null Σ f_k²; mixed models (statsmodels MixedLM) of CDR-SB on stage with subject random intercepts, compared with hippocampal z; Holm correction over subtypes/groups.
6. Simulations: generate longitudinal data from known sequences (`ebm.simulate`) with realistic noise and interval distributions to calibrate what monotonicity fraction a correct model achieves; this gives an effect-size benchmark rather than an arbitrary "80%".

Libraries: numpy, pandas, scipy, scikit-learn, statsmodels; pySuStaIn (optional), FreeSurfer 7 / LST-AI for WMH; nibabel.

## Evaluation and statistics

- Model selection: CVIC (10-fold cross-validated log-likelihood) across K; subtype sequence uncertainty from MCMC positional variance (pySuStaIn) or restarts (compact EBM).
- Longitudinal consistency: fraction of non-decreasing stage transitions, mean Δstage/year, Spearman(Δstage, Δyears); 1,000 within-subject permutations of session order for the null; effect size = observed − null mean, with bootstrap CI over subjects.
- Subtype stability: agreement of modal subtype across sessions, stratified by baseline stage tertile and posterior confidence; Cohen's κ vs Σ f_k² null.
- Clinical coupling: MixedLM `CDRSB ~ stage + age + sex + (1|subject)` vs `CDRSB ~ hipp_z + ...`; compare by AIC and marginal R²; likelihood-ratio for adding stage on top of hippocampal z.
- Leakage prevention: control model (z-scoring) and EBM fitted on baseline sessions of the training subjects only; follow-up sessions never enter model fitting; ADNI is a held-out replication cohort.
- Multiple comparisons: Holm within each family (subtypes × groups); primary endpoint is the A+ CI monotonicity vs null.
- Null models: session-order permutation; label-shuffled subtype null; simulated-from-model benchmark.

## Publishable angle

Headline: "Cross-sectional atrophy stages advance monotonically in amyloid-positive impairment (x% non-decreasing vs y% null), but in amyloid-negative impairment the inferred 'stages' are largely non-progressive, except for a limbic-first subtype; white-matter hyperintensity burden forms its own subtype rather than a stage offset." Either outcome for the A-CI group is informative for how SNAP/non-AD patients are characterized in MRI-based staging. 

Target venues: Brain Communications; NeuroImage: Clinical; Alzheimer's & Dementia; Imaging Neuroscience. Methods-only spin-off (permutation nulls for disease progression models) to IPMI/MICCAI or Medical Image Analysis.

Follow-ups: tau-PET (OASIS-3 has AV1451 on a subset) to test whether A-CI limbic subtype is tau-negative (LATE-like); plasma p-tau217/NfL where available; ordinal SuStaIn on visual ratings; OASIS-4 CSF subset as a memory-clinic replication.

## Risks, confounds and mitigations

- Scanner/field-strength changes within subjects across sessions (OASIS-3 has 1.5T legacy and 3T sessions): restrict primary analysis to same-scanner pairs; include scanner in the control regression; sensitivity with ComBat.
- FreeSurfer version (5.3 in OASIS-3 vs 6/7 in ADNI): fit z-scores per cohort against that cohort's controls; report sequence agreement, not pooled fits.
- FLAIR availability is incomplete in OASIS-3: WMH analyses on the FLAIR subset; T1 `WM-hypointensities` as a proxy with an agreement check on the overlap.
- Amyloid status drift (converters): the once-positive rule and a ±2-year window; converters analysed as a separate group.
- Practice/regression effects in CDR-SB: use ≥ 1-year intervals; mixed models with random slopes where identifiable.
- Small A-CI sample (OASIS-3 A-CI with ≥ 2 sessions may be ~100-150): pre-specify K ≤ 3 for that group, pool with ADNI A-CI for a sensitivity analysis, and report CIs.
- Attrition bias (sicker subjects drop out): inverse-probability-of-follow-up weights as a sensitivity analysis.

## Milestones

- [ ] DUAs (OASIS-3; ADNI application). Download tables, FreeSurfer stats, FLAIR (OASIS-3) and PET Centiloid tables.
- [ ] Timeline builder + amyloid-status-over-time rules; group counts table (sessions, subjects, intervals).
- [ ] WMH segmentation on FLAIR subset; agreement with T1 proxy.
- [ ] Control z-scoring; simulation benchmark for monotonicity/stability.
- [ ] Compact EBM prototypes; pySuStaIn fits (K = 1..5) on baseline A+ CI, A-CI, and joint atrophy + WMH.
- [ ] Longitudinal audit (RQ1, RQ2, RQ5) with nulls; A-CI results (RQ3, RQ4).
- [ ] ADNI replication (RQ6).
- [ ] Manuscript + code/model release (sequences, z-score reference parameters; no participant data).

## Ethics / data-use notes

- OASIS-3/4 require the NITRC DUA; cite LaMontagne et al., 2019 and acknowledge OASIS grants. ADNI requires an approved application, the ADNI acknowledgement text and Data and Publications Committee review before submission. Data may not be redistributed.
- Never commit participant-level data or derived tables that could re-identify (dates, exact ages with rare combinations); `data/` and `outputs/` are ignored. Participant-level data must not be sent to third-party APIs.
- Subtype labels are research constructs; the manuscript should avoid implying clinical diagnostic use of an individual's subtype/stage without prospective validation.
