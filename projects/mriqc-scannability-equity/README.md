# mriqc-scannability-equity — Who does automated MRI quality control exclude? A corpus-wide audit of MRIQC-based exclusion across OpenNeuro by age, sex and clinical group

Join MRIQC image-quality metrics (IQMs) with `participants.tsv` demographics across hundreds of OpenNeuro datasets to estimate each person's "scannability" (probability of surviving standard QC), quantify how exclusion is structured by age, sex and diagnosis after accounting for dataset, show that several IQMs track biology (atrophy, paediatric contrast) rather than artefact, and measure the representational shift that QC imposes on the open-data corpus.

## Status / difficulty / timeline / compute

- Status: design + starter code (participants.tsv harmonizer across BIDS conventions, IQM loader with configurable exclusion rules, mixed-effects exclusion models with scannability index and exclusion elasticity, representational-shift metrics, a corpus simulator for tests).
- Difficulty: MSc-level (6-9 months). The statistics are standard; the effort is in corpus curation and compute.
- Compute: MRIQC on ~10,000-20,000 T1w and ~8,000 BOLD runs at ~5-15 CPU-minutes each ⇒ 1,500-4,000 CPU-hours (a small cluster or cloud spot instances for a week). Where datasets already ship `derivatives/mriqc`, reuse them (record MRIQC version). Downstream models run on a laptop.
- Related project in this repository: `openneuro-mriqc-audit` (dataset-level IQM audit). This folder is self-contained and participant-level.

## Background

Automated QC is now routine: MRIQC (Esteban et al., 2017, PLoS ONE) computes ~60 anatomical and ~40 functional IQMs and ships a classifier for accept/exclude; fMRIPrep, XCP-D and most large studies apply motion thresholds (mean framewise displacement, fraction of high-motion volumes; Power et al., 2012, NeuroImage; Satterthwaite et al., 2012, NeuroImage), and structural pipelines exclude on Euler number or CNR (Rosen et al., 2018, NeuroImage). Head motion and image quality are not randomly distributed: children, older adults and clinical populations move more, and motion itself biases morphometry (Reuter et al., 2015, NeuroImage; Alexander-Bloch et al., 2016, Human Brain Mapping; Pardoe et al., 2016, NeuroImage). QC therefore acts as a selection mechanism.

Recent single-cohort work has made the equity consequences explicit. In ABCD, Black, Hispanic and Asian youth were more likely to be excluded for motion at baseline and participants from lower-income and lower-parental-education households were over-represented among exclusions (Cosgrove et al., 2022, Brain Imaging and Behavior; "Retention and data exclusion challenges for representative longitudinal neuroimaging", bioRxiv 2025). In UK Biobank task fMRI, strict thresholds would remove most participants, with removals structured by age, BMI and motion-associated clinical conditions ("The State of Motion: demographic and representational consequences of head-motion exclusion", bioRxiv 2026). A 2025 analysis of functional QC and bias found the same pattern for lower neighbourhood opportunity and trauma exposure. These studies each analyse one cohort with one QC rule and mostly fMRI motion.

## The research gap

What has been done:

- Cohort-specific exclusion-bias analyses: ABCD (Cosgrove et al., 2022; bioRxiv 2025), UK Biobank (bioRxiv 2026), HBN/HBCD-style developmental cohorts (2025). All are single-cohort, mostly fMRI-motion-based, and cannot separate dataset design (protocol, site, population) from participant characteristics.
- MRIQC's own validation (Esteban et al., 2017; Esteban et al., 2019, Scientific Data — crowdsourced IQMs and expert ratings) established cross-site generalization of the classifier but did not examine demographic structure of its decisions; the MRIQC Web-API holds > 100,000 IQM records but no demographics.
- Structural QC studies (Rosen et al., 2018; Bethlehem et al., 2022, Nature brain charts) apply Euler-number thresholds uniformly across the lifespan without testing whether the threshold has age-dependent exclusion rates.

What is specifically missing:

1. A corpus-level estimate. OpenNeuro hosts > 1,000 datasets with `participants.tsv` (age, sex, and often `group`/`diagnosis`). Nobody has estimated P(excluded | age, sex, group) with dataset as a random effect, i.e., pooled across hundreds of protocols so that the participant-level effect is separable from dataset-level design.
2. Biology-vs-artefact confounding of IQMs. CNR, CJV, SNR and EFC depend on grey/white-matter contrast and tissue volumes, which change with age (paediatric myelination, adult atrophy) and disease. A threshold on CNR excludes atrophic brains for being atrophic. No study has quantified how much of the age-structure of exclusions is due to IQM-biology coupling versus true artefact (motion, ghosting), nor proposed biology-adjusted thresholds.
3. Threshold sensitivity ("exclusion elasticity"): how quickly the exclusion fraction of each group rises as the threshold tightens; groups with steep elasticity are the ones whose representation depends on an arbitrary analyst decision.
4. Representational consequences on the corpus: the effective age and diagnostic distribution of "QC-passed OpenNeuro" versus "acquired OpenNeuro" — the population that reused-data papers actually study.
5. Structural (T1w) exclusion is under-studied compared with fMRI motion; both modalities should be analysed with the same participants.

## Research questions / hypotheses

1. RQ1 (age). Across datasets, does P(exclusion) follow a U-shape in age for both T1w and BOLD? H1: yes; exclusion odds at age 8 and age 80 are ≥ 2× those at age 25 under the MRIQC classifier and under common threshold rules, after dataset random effects.
2. RQ2 (clinical group). Are psychiatric/neurological groups excluded more than controls *within the same dataset*? H2: odds ratio 1.3-2.0 for neurological (stroke, epilepsy, dementia, movement disorders) and 1.2-1.5 for psychiatric groups; the effect is larger for BOLD than T1w.
3. RQ3 (sex). Is there a residual sex difference after age and group? H3: males have modestly higher exclusion odds (OR 1.1-1.3), consistent with motion literature, and this is larger in children.
4. RQ4 (biology-IQM coupling). Within QC-passed controls, do CNR/CJV/EFC vary with age? H4: CNR declines and CJV rises with age beyond 60 and in children under 10 independently of motion IQMs; residualizing IQMs on age within dataset reduces the age-structure of T1w exclusions by ≥ 30%.
5. RQ5 (elasticity). Which groups have the steepest exclusion elasticity? H5: older adults and neurological groups; threshold choices within the range used in the literature (mean FD 0.2-0.5 mm) change their exclusion fraction by > 20 percentage points.
6. RQ6 (representation). Does QC shift the corpus age distribution? H6: the QC-passed corpus has a narrower age IQR and fewer participants > 65 y and < 10 y (each by 10-25%) than the acquired corpus.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| OpenNeuro (all BIDS datasets with T1w and/or BOLD and a participants.tsv) | `participants.tsv` (age, sex, group/diagnosis), raw T1w/BOLD for MRIQC, existing `derivatives/mriqc` where present, dataset metadata (scanner, field strength, year) | > 1,000 datasets; ~40,000 participants; ~10-20k usable T1w and ~8k BOLD subjects with demographics | Open (CC0); GraphQL API, `openneuro-py`, DataLad, or S3 (`s3://openneuro.org`) | https://openneuro.org , https://github.com/OpenNeuroDatasets |
| MRIQC Web-API | Crowdsourced IQMs (no demographics) for normative IQM distributions and version drift | > 100,000 records | Open REST API | https://mriqc.nimh.nih.gov/ |
| UCLA CNP LA5c (OpenNeuro ds000030) | Worked example: controls, schizophrenia, bipolar, ADHD with T1w and multiple BOLD tasks | 272 participants | Open | https://openneuro.org/datasets/ds000030 |
| ABCD (optional replication) | Baseline T1w/rs-fMRI IQMs, demographics, income/education | ~11,800 | NDA DUA | https://nda.nih.gov/abcd |
| Healthy Brain Network (optional) | Paediatric clinical sample, phenotypes, raw MRI | > 3,000 | Data usage agreement (lite) | https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/ |

Group harmonization: `participants.tsv` uses many conventions (`group`, `diagnosis`, `dx`, `condition`, `patient`, free text). `scannability.participants` maps them to coarse classes (control, psychiatric, neurological, neurodevelopmental, other-clinical, unknown) with a curated keyword dictionary that is exported for review; a random 10% of mappings are manually checked.

## Methods

1. Corpus construction (`scripts/download_data.py`): query the OpenNeuro GraphQL API for datasets with anat/func modalities; fetch `participants.tsv`, `participants.json`, `dataset_description.json` and any `derivatives/mriqc/group_*.tsv`; store a dataset index with scanner and year.
2. IQMs: run MRIQC (Docker/Apptainer, version pinned) per dataset on T1w and BOLD; collect `group_T1w.tsv` and `group_bold.tsv`; run `mriqc_clf` for the classifier prediction on T1w.
3. Harmonization (`scannability.participants`): parse age (numbers, ranges like `20-25`, `89+`, months), sex (M/F/male/female/1/2), group keywords; attach dataset ID.
4. Exclusion rules (`scannability.iqm`): (a) MRIQC classifier; (b) literature thresholds (BOLD: mean FD > 0.2/0.3/0.5 mm, fraction FD > 0.2 mm above 20%, tSNR low tail; T1w: CJV, CNR, EFC, SNR dataset-relative percentiles and absolute cut-offs); (c) dataset-relative "worst 10%" rule. IQMs are z-scored within dataset for the relative rules.
5. Models (`scannability.models`): binomial GLM with natural-cubic-spline age, sex, coarse group, modality, dataset fixed effects (and a mixed-model cross-check with `BinomialBayesMixedGLM`), cluster-robust SEs by dataset. Scannability index = predicted pass probability at a reference dataset. Exclusion elasticity = slope of group exclusion fraction over a threshold sweep. Biology adjustment: residualize IQMs on the age spline (within dataset, QC-passed controls only), re-derive exclusions, compare age-structure.
6. Representation (`scannability.models.representation_shift`): age quantiles, share of < 10 y and > 65 y, share of clinical groups, KL divergence between acquired and QC-passed distributions, per rule.
7. Replication in ABCD/HBN (optional) with the same code.

Libraries: pandas, numpy, scipy, statsmodels, scikit-learn, patsy; `openneuro-py`/`datalad` for downloads; MRIQC (container) for IQMs.

## Evaluation and statistics

- Effects reported as odds ratios with 95% cluster-robust CIs; age effects as predicted exclusion probability curves with pointwise CIs.
- Dataset heterogeneity: random-slope sensitivity analysis (age by dataset) and leave-one-dataset-out influence checks.
- Multiple comparisons: three primary hypotheses (age U-shape, clinical-group OR, sex OR) pre-registered; the rest exploratory with Holm correction within families.
- Nulls: permute participant rows within dataset (breaks participant-IQM links while preserving dataset design) to obtain the null distribution of each OR; compare observed ORs against it.
- Robustness: results per MRIQC version (IQMs drift across versions), per field strength, per acquisition year, and with the dataset-relative rules vs absolute thresholds.
- Reporting: STROBE-style flow of datasets/participants; release the harmonized participant-IQM table (CC0 inputs) with dataset IDs.

## Publishable angle

Headline: "Standard QC removes older adults, children and neurological patients at 2-3× the rate of young controls even within the same dataset; a third of the age effect in structural QC is the metric reacting to normal biology, not to artefact; and the QC-passed open-data corpus is measurably younger and healthier than the acquired one." Deliverables: a scannability calculator, biology-adjusted IQM thresholds, and a corpus-level representation report that data reusers can cite.

Target venues: NeuroImage; Imaging Neuroscience; Scientific Data (for the harmonized table + audit); Nature Human Behaviour (short report if the equity result is strong); OHBM abstract.

Follow-ups: does QC-driven selection change reported effect sizes in reused OpenNeuro datasets (re-analysis of published group differences with and without QC)? Prospective acquisition guidance (protocol features that reduce differential exclusion); inverse-probability-of-QC weighting as a correction.

## Risks, confounds and mitigations

- Missing or inconsistent demographics: many datasets lack `group`; report the fraction mapped and run analyses on the subset with complete data; sensitivity analysis with multiple imputation by dataset.
- Dataset-level confounding (clinical datasets use different protocols): dataset fixed/random effects; within-dataset contrasts only for the group effect.
- IQM version drift: pin MRIQC version for new runs; treat shipped derivatives as a separate stratum.
- Selection before upload: authors may already have removed bad scans before sharing; this biases exclusion rates downwards and makes the estimates conservative; check `sourcedata`/README statements where available.
- Age parsing errors (months vs years, ranges): unit heuristics with dataset-level review; exclude datasets whose ages cannot be resolved.
- Compute cost: prioritize datasets by size and demographic completeness; the design is valid with a random sample of datasets (report the sampling frame).

## Milestones

- [ ] Dataset index from OpenNeuro API; participants tables harmonized; group-mapping dictionary reviewed.
- [ ] MRIQC runs (or shipped derivatives) for the top-N datasets; IQM table with version stamps.
- [ ] Exclusion flags under all rules; descriptive exclusion rates by age decade, sex, group, modality.
- [ ] Primary models (age spline, group OR, sex OR) with dataset effects and permutation nulls.
- [ ] Biology-IQM coupling analysis and biology-adjusted thresholds.
- [ ] Elasticity sweeps and representation-shift report.
- [ ] Optional ABCD/HBN replication; manuscript; harmonized table release.

## Ethics / data-use notes

- OpenNeuro data are CC0, but `participants.tsv` can contain quasi-identifiers; do not attempt re-identification and do not enrich with external sources at the individual level.
- ABCD and HBN are under DUAs: no redistribution of participant-level data; aggregate results only; never commit data.
- Coarse diagnostic mapping must not be used to make claims about individuals; report mapping uncertainty.
- Do not send participant-level data to third-party LLM or cloud APIs; keyword mapping is done locally.
