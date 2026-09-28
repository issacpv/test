# lifespan-normative-models — Diffusion-MRI normative charts and a centile transportability audit

Harmonized lifespan normative models (GAMLSS-style quantile regression, optional PCNtoolkit) for cortical thickness, subcortical volumes and tract-wise diffusion metrics (FA/MD) built on the protocol-harmonized HCP Lifespan cohorts (HCP-Development 5-21 y, HCP Young Adult 22-37 y, HCP-Aging 36-100+ y) plus OASIS-3 and open OpenNeuro cross-sectional datasets, with the central question: how far do an individual's centile scores travel? (leave-one-site-out, reference-model agreement, small-sample site adaptation, test-retest), and whether diffusion centiles add clinical deviation detection in OASIS-3 beyond morphometry.

## Status / difficulty / timeline / compute

- Status: design + starter code (multi-dataset phenotype harmonizer, quantile/heteroscedastic normative fitting with spline age basis and site effects, centile scoring and extreme-deviation counts, leave-one-site-out site-sensitivity evaluation).
- Difficulty: PhD chapter or strong MSc with a data-engineering partner; 9-14 months. The modelling is fast; the diffusion processing across ~4,000 subjects is the long pole.
- Compute: DTI fitting + tract segmentation (TractSeg or atlas-based) ≈ 5-15 min per subject on CPU, minutes on GPU; ~4,000 subjects → 1-3 weeks on a 32-core node. Storage: HCP preprocessed dMRI is 1-2 GB per subject; fetch the b = 1000 shell only or process in streaming batches (the downloader supports per-subject sampling). FreeSurfer outputs are provided for HCP-YA, HCP-A/D (Lifespan 2.0 release) and OASIS-3. Normative fitting and the audit run on a laptop.

## Background

Normative modelling turns a brain measure into a centile relative to a reference population conditioned on age and sex (Marquand et al., 2016, Biological Psychiatry; 2019, Molecular Psychiatry). Lifespan reference charts now exist for morphometry: BrainChart (Bethlehem et al., 2022, Nature; > 100,000 scans, GAMLSS), PCNtoolkit lifespan cortical thickness/subcortical models (Rutherford et al., 2022, eLife; ~58,000 scans, warped Bayesian linear regression), CentileBrain (Ge et al., 2024, Lancet Digital Health), and longitudinal extensions (Di Biase et al., 2023, PNAS). The promise is that a clinician or trialist can score a new individual against an open reference. Whether that promise holds depends on transportability: site/protocol effects, the reference model chosen, and how many local controls are needed to adapt.

Diffusion MRI (FA/MD) is the modality most sensitive to white-matter aging and vascular injury, yet normative charts for it lagged behind morphometry until 2024-2026.

## The research gap

What has been done (2023-2026 check):

- "Lifespan normative models of white matter fractional anisotropy: applications to early psychosis" (bioRxiv 2024; Biological Psychiatry, 2025): HBR models of tract FA across the lifespan, applied to HCP Early Psychosis.
- "Lifespan normative modeling of brain microstructure" (bioRxiv 2024, revised 2026): 19 datasets, N ≈ 54,000, ages 4-91, FA/MD/AD/RD (U-shaped diffusivity minima at 37-53 y), HCP Baby/D/YA/A + UK Biobank among the cohorts.
- "Distributional and centile calibration of diffusion tensor imaging" (bioRxiv, 2026): calibration of DTI centiles.
- "The influence of sample size and covariate distributions on neuroanatomical normative models" (eLife reviewed preprint 2025/2026): OASIS-3 cortical thickness/subcortical volumes; reference subsamples from 5 to 600 controls; UK Biobank pre-training with transfer; age alignment matters more than raw n. Morphometry only.
- Bayer et al., 2022 (Human Brain Mapping) and Fortin et al., 2017 (NeuroImage) on retrospective harmonization (ComBat) for diffusion and thickness.
- Practical guides (2025) and federated/distributed GAMLSS (dGAMLSS, Bioinformatics, 2025).

So diffusion lifespan curves exist. What is specifically missing:

1. A transportability audit of centiles across reference models. Nobody has reported, for the same individuals, how much centiles (and extreme-deviation flags) disagree between BrainChart-style GAMLSS curves, PCNtoolkit HBR/BLR curves and locally fitted quantile models, for morphometry and for diffusion. If κ for "extreme deviation" flags between two respectable references is 0.4, the clinical reading of a centile is model-dependent, which matters for any use in trials.
2. Leave-one-site-out transport with realistic adaptation budgets. The eLife study varied reference size; the open question for a user of a public chart is: with 0/10/25/50 local controls, how far are held-out-site centiles from uniform (KS statistic), what is the false extreme-deviation rate, and how does it differ between thickness (well studied) and FA/MD (protocol-sensitive: b-value, resolution, multi-shell vs single-shell)?
3. Protocol-harmonized backbone vs heterogeneous pooling. HCP-D/YA/A share HCP-style multi-shell acquisitions across 5-100 years; OASIS-3 DTI is single-shell b ≈ 1000. Fitting on the harmonized backbone and testing transfer to OASIS-3 (and to OpenNeuro sites: CNP ds000030, MPI-LEMON ds000221, AOMIC ds003097) isolates protocol shift from age-range shift.
4. Clinical transfer in aging/AD. Diffusion normative deviations have been applied to psychosis; in OASIS-3 the question "do FA/MD extreme deviations discriminate CDR ≥ 0.5 vs 0, and A+ vs A- cognitively normal, beyond thickness/volume deviations?" has not been reported, nor how much that discrimination depends on whether the OASIS-3 site is part of the reference.
5. Individual-level reliability of centiles (test-retest: HCP-YA retest subset; QTIM ds004169 two sessions for T1) is rarely reported alongside the charts.
6. Open release of fitted reference curves (parameters, not data) with a scoring function, including site-adaptation utilities.

Verification note on OpenNeuro: ds004169 is the Queensland Twin IMaging (QTIM) dataset (1,202 healthy twins/siblings, T1w, two sessions, `family_id` for non-independence). It is a healthy external site with test-retest, not a clinical dataset. The verified open clinical dataset with T1 + 64-direction DWI is UCLA CNP (ds000030: 138 controls, 58 schizophrenia, 49 bipolar, 45 ADHD). Aging/dementia clinical transfer uses OASIS-3 (DUA), not OpenNeuro.

## Research questions / hypotheses

1. RQ1 (transport without adaptation). Fit on all sites but one; score the held-out healthy site. H1: thickness centiles are near-uniform (KS < 0.10, extreme rate 5-8%) for HCP sites but not for OASIS-3/OpenNeuro sites (KS > 0.15); FA/MD centiles fail even between HCP-D/YA/A when acquisition differs (extreme rate > 15%).
2. RQ2 (adaptation budget). H2: shift-scale adaptation with 25 local controls restores extreme rates to ≤ 8% for thickness; FA/MD need ≥ 50 controls or a site-specific variance term.
3. RQ3 (reference-model agreement). For OASIS-3 and CNP individuals, compare centiles from (a) BrainChart curves, (b) PCNtoolkit lifespan models, (c) local quantile models. H3: Spearman ρ > 0.9 for centiles but κ < 0.6 for extreme-deviation flags (threshold effects dominate).
4. RQ4 (clinical value of diffusion centiles). H4: MD extreme-deviation counts add ΔAUC ≥ 0.03 over thickness/volume counts for CDR ≥ 0.5 vs 0 in OASIS-3; no added value for A+ vs A- among CDR 0; the gain shrinks when OASIS-3 is held out of the reference (site-effect sensitivity of clinical detection).
5. RQ5 (reliability). H5: test-retest ICC of centiles ≥ 0.8 for regional thickness and tract FA, ≥ 0.7 for MD; the SE of a centile is ≥ 8 centile points, which bounds how "extreme" a single scan can be declared.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| HCP Young Adult (S1200) | 3T T1/T2, FreeSurfer stats, multi-shell dMRI (b = 1000/2000/3000), 45-subject retest; open data give age in bins (22-25, 26-30, 31-35, 36+), exact age needs Restricted Data | ~1,100 subjects, ~1,050 with dMRI | Free registration at ConnectomeDB + Open Access Data Use Terms; AWS S3 (`s3://hcp-openaccess`) with credentials from ConnectomeDB; Restricted Data by application | https://db.humanconnectome.org , https://www.humanconnectome.org/study/hcp-young-adult |
| HCP-Development (HCP-D) | Lifespan 2.0 release: T1/T2, FreeSurfer, multi-shell dMRI, `interview_age` in months | ~1,300 participants, 5-21 y | NIMH Data Archive (NDA) account + Data Use Certification; download with `nda-tools` (`downloadcmd`) | https://nda.nih.gov/ (HCP-D collection 2846), https://www.humanconnectome.org/study/hcp-lifespan-development |
| HCP-Aging (HCP-A) | Lifespan 2.0 release: same modalities | ~1,200 participants, 36-100+ y | NDA account + DUC (collection 2847) | https://www.humanconnectome.org/study/hcp-lifespan-aging |
| OASIS-3 | T1, FreeSurfer 5.3, DTI (single-shell) subset, CDR, amyloid PET (Centiloid) | ~1,300 participants; DTI on several hundred sessions | NITRC-IR DUA (free) | https://www.nitrc.org/projects/oasis3/ |
| OpenNeuro ds004169 (QTIM) | Healthy twins/siblings, T1w, two sessions, age (rounded), sex, `family_id` | 1,202 subjects | Open (CC0) | https://openneuro.org/datasets/ds004169 |
| OpenNeuro ds000030 (UCLA CNP) | T1 + 64-dir DWI; controls, schizophrenia, bipolar, ADHD | 272 subjects | Open | https://openneuro.org/datasets/ds000030 |
| OpenNeuro ds000221 (MPI-LEMON) | Healthy adults 20-80 y, T1 + DWI | ~228 subjects | Open | https://openneuro.org/datasets/ds000221 |
| OpenNeuro ds003097 (AOMIC-ID1000) | Healthy young adults, T1 + DWI | ~928 subjects | Open | https://openneuro.org/datasets/ds003097 |
| BrainChart reference curves | GAMLSS models/centile tables for GMV, WMV, sGMV, ventricles, mean thickness, total SA | software/curves | Open (GitHub) | https://github.com/brainchart/Lifespan |
| PCNtoolkit lifespan models | Pretrained BLR models for Destrieux cortical thickness and subcortical volumes; adaptation scripts | software/models | Open (GitHub) | https://github.com/predictive-clinical-neuroscience/braincharts |

All cohorts contribute only healthy controls to the reference (HCP cohorts by design; OASIS-3 CDR 0 and amyloid-negative; CNP controls; QTIM/LEMON/AOMIC all healthy). Clinical groups (OASIS-3 CDR ≥ 0.5, A+ CDR 0; CNP patient groups) are scored only.

## Methods

1. Phenotype harmonization (`lifespan_norm.harmonize`): adapters for HCP-YA unrestricted CSV (age bins → midpoints with an `age_is_binned` flag; exact ages substituted when Restricted Data are available), NDA `ndar_subject01`-style tables (`interview_age` months → years), OASIS-3 (`ageAtEntry + days/365.25`), BIDS `participants.tsv`. Output schema: `subject_id, session_id, dataset, site, scanner, age, sex, age_is_binned, group, diagnosis`. FreeSurfer `aparcstats2table`/`asegstats2table` outputs and tract tables are attached by subject/session.
2. Image-derived features: FreeSurfer thickness (68 DK regions) and subcortical volumes (eTIV-normalized); DTI from the b ≈ 1000 shell (HCP: extract shell; OASIS-3/OpenNeuro: native) with DIPY/FSL `dtifit`; tract-mean FA/MD in TractSeg bundles (or JHU-ICBM atlas after ANTs registration) — one pipeline for all cohorts, provenance recorded.
3. Normative models (`lifespan_norm.normative`): (a) quantile regression (statsmodels `QuantReg`) at 11 quantiles with a natural cubic spline in age (knots at age quantiles), sex and site fixed effects, with quantile rearrangement to enforce monotonicity — a GAMLSS-like empirical alternative that needs no distributional assumption; (b) heteroscedastic Gaussian model (spline mean + log-variance regression) with shift-scale site adaptation; (c) optional PCNtoolkit HBR/BLR and BrainChart curves via their published scripts for the reference-model agreement analysis.
4. Centile scoring (`lifespan_norm.centiles`): centile by interpolation across fitted quantiles, z = Φ⁻¹(centile), extreme deviations at |z| > 1.96 and 2.58, per-subject counts (negative/positive), group contrasts with Mann-Whitney and BH-FDR, AUC of deviation counts for clinical labels, uniformity checks (KS), test-retest ICC.
5. Site-sensitivity (`lifespan_norm.site_sensitivity`): leave-one-site-out with adaptation budgets n ∈ {0, 10, 25, 50, 100} (repeated random control subsets), reporting KS, extreme rates and mean |Δcentile| against in-sample scoring; agreement between reference models (Spearman, κ on extreme flags); clinical AUC in-reference vs out-of-reference.
6. Release: fitted quantile parameters per feature (CSV/JSON), scoring script, adaptation utility; no participant data.

## Evaluation and statistics

- Uniformity of held-out healthy centiles: KS statistic vs U(0,1) with bootstrap CI over subjects; extreme rate with Wilson CI; expected 5% at |z| > 1.96.
- Adaptation: paired comparison of KS/extreme rate across budgets (repeated subsets; report medians and IQR).
- Reference agreement: Spearman ρ of centiles; Cohen's κ for extreme flags; Bland-Altman of z.
- Clinical detection: AUC with DeLong CI; ΔAUC (diffusion + morphometry vs morphometry) via bootstrap; permutation of diagnosis labels as the null.
- Multiple comparisons: BH-FDR across features within each family (thickness, volumes, FA, MD); primary endpoints pre-specified: (i) held-out extreme rate for FA at n = 0 vs n = 25 adaptation; (ii) ΔAUC for MD counts in CDR ≥ 0.5 vs 0.
- Leakage: models fitted on healthy controls only; adaptation controls excluded from evaluation sets; clinical labels never enter fitting; family structure (QTIM `family_id`, HCP-YA families) respected by grouping in any resampling.
- Age-bin handling for HCP-YA: sensitivity analysis excluding HCP-YA or using restricted exact ages; report the effect of bin midpoints on 22-37 y curves.

## Publishable angle

Headline: "Diffusion centiles are far less transportable than morphometric centiles: without local adaptation, FA/MD extreme-deviation rates at a held-out site are x% instead of 5%, and two respectable references disagree on which individuals are extreme (κ = y); with 25-50 local controls diffusion centiles become usable and MD deviations add z AUC to CDR-based impairment detection in OASIS-3." Plus an open release of the reference curves and scoring code.

Target venues: Imaging Neuroscience; NeuroImage; Human Brain Mapping; Nature Communications or Nature Neuroscience (if the audit spans all modalities and references); OHBM abstract early.

Follow-ups: NODDI/multi-shell metrics from HCP (not transportable to single-shell sites by construction → quantify); longitudinal centile change in OASIS-3 (rate-of-change charts); federated fitting (dGAMLSS) with sites that cannot share data; PCNtoolkit HBR with site as a random effect vs fixed effect comparison.

## Risks, confounds and mitigations

- HCP-YA age bins: use restricted exact ages where granted; otherwise bin midpoints with the flag and a sensitivity analysis.
- Protocol shift (multi-shell vs single-shell, resolution 1.25 mm vs 2 mm+): use the b = 1000 shell for all; report a within-subject comparison where both exist (none public: state limitation); treat site as protocol.
- Family structure (HCP-YA twins; QTIM): grouped resampling; HCP-YA family IDs are restricted → use one subject per family when restricted data are unavailable (random pick), or accept mild dependence and say so.
- FreeSurfer versions differ (5.3 OASIS-3, 6/7 HCP): site fixed effects absorb version; re-run FreeSurfer 7 on OASIS-3 for a subset to quantify.
- Healthy control definition differs (self-report vs CDR 0 + A-): document; sensitivity with OASIS-3 CDR 0 regardless of amyloid.
- Quantile crossing and tail extrapolation: rearrangement; tails beyond 1st/99th percentile capped and flagged.
- Sex/site imbalance in some sites: report per-site n by sex and age decade; exclude cells with n < 10.
- Data volume: only stats/derivatives need to persist; raw dMRI processed in batches and deleted.

## Milestones

- [ ] Access: ConnectomeDB (HCP-YA + S3 credentials), NDA DUC for HCP-A/D, NITRC DUA for OASIS-3; OpenNeuro downloads.
- [ ] Harmonized phenotype table with per-site counts; FreeSurfer feature tables.
- [ ] Diffusion pipeline (b = 1000 shell, DTI, tract means) on all cohorts; QC (FA outliers, motion).
- [ ] Normative fits (thickness, volumes, FA, MD); release-ready parameter files.
- [ ] RQ1-RQ2: leave-one-site-out and adaptation budgets.
- [ ] RQ3: BrainChart and PCNtoolkit scoring of OASIS-3/CNP; agreement analysis.
- [ ] RQ4: clinical detection in OASIS-3 (CDR, amyloid) in- vs out-of-reference.
- [ ] RQ5: test-retest reliability (HCP-YA retest, QTIM sessions).
- [ ] Manuscript, open curves + scoring code.

## Ethics / data-use notes

- HCP-YA: Open Access Data Use Terms (no re-identification, no redistribution of data; acknowledge the WU-Minn HCP consortium and grants). Restricted Data (exact age, family structure) require separate approval and must never be shared or released in any form, including derived tables that reveal them.
- HCP-A/HCP-D via NDA: Data Use Certification; NDA data may not be redistributed; publications must acknowledge NDA and the Lifespan HCP grants.
- OASIS-3: NITRC DUA; cite LaMontagne et al., 2019.
- OpenNeuro datasets are CC0/PDDL; still, respect the `family_id` non-independence and do not attempt re-identification.
- The released reference curves contain only fitted parameters/centile tables (aggregate), which is permitted; no participant-level data leave the environment and none go to third-party APIs.
- `data/` and `outputs/` are git-ignored.
