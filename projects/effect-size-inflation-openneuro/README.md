# effect-size-inflation-openneuro

**How inflated are the effect sizes claimed for OpenNeuro datasets? A winner's-curse audit that joins every OpenNeuro dataset's sample size (via the GraphQL API) to the peak effects reported in its associated papers, applies selection-corrected estimators, and measures inflation directly with cross-validated re-analysis of fMRIPrep'd task datasets.**

## Status / difficulty / timeline / compute

- Status: design + starter code (working OpenNeuro GraphQL client with GitHub-mirror fallback, winner's-curse estimators with tests, circular-selection inflation estimator). No data is shipped.
- Difficulty: MSc-level for the meta-research half (API + estimators + curated effect table); early-PhD level if the re-analysis half (fMRIPrep derivatives, cross-validated peak effects) is included. 6-9 months.
- Compute: laptop for the meta-research half. Re-analysis half: one workstation (16 cores, 64 GB RAM, ~2 TB disk for 20-30 fMRIPrep'd task datasets pulled from OpenNeuro Derivatives; no GPU).

## Background

Two mechanisms inflate published neuroimaging effects. The first is *selection across studies*: a study with a small n only reaches significance when its estimate happens to be large, so published estimates are biased upwards (Ioannidis, 2008, Epidemiology; Button et al., 2013, Nat Rev Neurosci; Gelman & Carlin, 2014, Perspect Psychol Sci "Type M" error). The second is *selection within a study*: reporting the effect at the peak voxel/cluster that was chosen because it was the largest ("double dipping", Kriegeskorte et al., 2009, Nat Neurosci; Vul et al., 2009, Perspect Psychol Sci). Marek et al. (2022, Nature) showed for brain-wide association studies that with n < 100 the largest observed correlations are almost entirely sampling noise and that replication requires thousands of participants; Poldrack et al. (2017, Nat Rev Neurosci) and Szucs & Ioannidis (2020, NeuroImage) documented that median fMRI sample sizes stayed in the 20s for two decades.

OpenNeuro is the largest open repository of raw BIDS neuroimaging data. Its own resource paper (Markiewicz et al., 2021, eLife) reports a median of 23 participants per dataset. Crucially, most OpenNeuro datasets are linked to the paper that first analysed them (`DatasetDOI`, `ReferencesAndLinks` in `dataset_description.json`), and hundreds now have public fMRIPrep derivatives (OpenNeuro Derivatives). That makes OpenNeuro a natural population for an empirical winner's-curse study: for each dataset we know n exactly, we can read the reported effect, and for a subset we can recompute the effect ourselves with an unbiased (cross-validated) estimator on the very same data.

## The research gap

What has been done (2013-2026):

- Sample-size surveys: Button et al. (2013); Poldrack et al. (2017); Szucs & Ioannidis (2020, NeuroImage) on highly cited studies 1990-2018; Markiewicz et al. (2021, eLife) for OpenNeuro. These describe n, not effect inflation.
- BWAS-specific: Marek et al. (2022, Nature) quantified inflation of brain-behaviour correlations as a function of n using ABCD/HCP/UKB subsampling; Spisak, Bingel & Wager (2023, Nature) showed multivariate BWAS can succeed at smaller n. Both concern resting-state/structural correlations with behaviour, not task-activation effects, and neither uses the published literature as the unit of analysis.
- A 2026 review, "Effect sizes in human functional neuroimaging", argues that standard mass-univariate procedures give an inflated picture of effect sizes and that common sample sizes are too small for many brain-behaviour relationships; it is a review, not an empirical audit of repository-linked papers.
- Winner's-curse estimators are mature in statistical genetics: conditional-likelihood MLE (Zhong & Prentice, 2008, Biostatistics; Ghosh, Zou & Wright, 2008, Am J Hum Genet), bootstrap corrections (Sun & Bull, 2005; Faye et al., 2011), and empirical Bayes; a 2023 PLOS Genetics review ("Review and further developments in statistical corrections for winner's curse in genetic association studies") compares them. They have not been applied to a repository-linked corpus of neuroimaging effects.
- The cross-validated / split-half estimate of the circular peak effect is standard advice (Kriegeskorte 2009) but has not been applied at scale across many public datasets to measure how big the within-study inflation actually is in practice.

What is specifically missing (our angle):

1. A **repository-anchored corpus**: every OpenNeuro dataset with a linked paper, its exact n (from the BIDS snapshot, not from the paper), and the peak effect(s) the paper reports. Nobody has built this table; it is the denominator that sample-size surveys lack.
2. **Selection-corrected effect estimates per paper** (conditional MLE, empirical Bayes) and the implied Type M exaggeration ratio, tabulated by modality, task family, year and n.
3. **Direct measurement of both inflation mechanisms on the same data**: for datasets with fMRIPrep derivatives, recompute (a) the peak effect in-sample (circular) and (b) the cross-validated peak effect, and compare both with the paper's number. This separates within-study circularity from across-study selection, which no previous work does.
4. A **calibrated expectation** for replication: given the corpus, what fraction of reported effects would be expected to replicate at the same n, and what n would be needed for 80% power on the *corrected* effect?

## Research questions / hypotheses

1. **RQ1 (corpus).** What is the joint distribution of n and reported peak effect (Cohen's d or r) across OpenNeuro-linked papers? H1: reported |d| is negatively correlated with n (Spearman rho < -0.3), the signature of selection, and the funnel (Egger) intercept is significantly positive.
2. **RQ2 (corrected effects).** After conditional-MLE correction at the study's own threshold, what is the median shrinkage? H2: median corrected effect is < 60% of the reported effect for n <= 25 and > 85% for n >= 100.
3. **RQ3 (within- vs across-study inflation).** In the re-analysis subset, what is the ratio of in-sample peak d to cross-validated peak d? H3: the circular/cross-validated ratio exceeds 1.5 at n <= 25 and falls below 1.2 at n >= 80; the paper-reported effect exceeds the cross-validated effect by more than the circular estimate does (because papers also select among contrasts/ROIs).
4. **RQ4 (replication expectation).** H4: at the original n, the expected replication probability (power at the corrected effect) has a median below 0.5; the median n needed for 80% power on the corrected effect is > 3x the original n.
5. **RQ5 (moderators).** H5: inflation is larger for cognitive/affective task contrasts than for sensorimotor localisers, and smaller for datasets whose papers pre-registered ROIs.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| OpenNeuro dataset index (GraphQL API) | n per dataset (`summary.subjects`), modalities, tasks, `DatasetDOI`, `ReferencesAndLinks` | ~1,300+ datasets (2026), metadata only (MB) | Open, no key | https://openneuro.org/crn/graphql |
| OpenNeuroDatasets GitHub mirrors | fallback `participants.tsv` / `dataset_description.json` per dataset when GraphQL is unavailable | one git repo per dataset | Open | https://github.com/OpenNeuroDatasets |
| OpenNeuro Derivatives (fMRIPrep / MRIQC outputs) | re-analysis subset: subject-level task GLMs from preprocessed BOLD | hundreds of datasets; tens of GB per dataset | Open (S3 / DataLad) | https://github.com/OpenNeuroDerivatives |
| NeuroVault | unthresholded group statistic maps for linked papers (whole-brain effect distribution, peak locations) | thousands of collections | Open REST API | https://neurovault.org/api/ |
| PubMed / PMC OA (E-utilities) | full text of linked open-access papers for effect extraction | text | Open | https://www.ncbi.nlm.nih.gov/books/NBK25501/ |
| Large-n reference effects (e.g., HCP S1200 task contrasts, UK Biobank task fMRI summaries) | "true effect" anchors for Type M calculations on common contrasts (faces, n-back, MID, stop-signal) | group maps | HCP: free registration; UKB: application | https://db.humanconnectome.org ; https://www.ukbiobank.ac.uk |

No participant-level data leaves the repositories; only summary metadata and public derivatives are used.

## Methods

Pipeline (each step maps to a module in `src/es_inflation/`):

1. **Dataset index** (`openneuro_client.py`, `scripts/download_data.py`). Paginate `datasets(first, after)` on the GraphQL endpoint; keep `id`, `latestSnapshot.summary.subjects` (n), `modalities`, `tasks`, `description.DatasetDOI`, `ReferencesAndLinks`, `Authors`, snapshot dates. If the endpoint is unreachable, enumerate `ds000001..ds00xxxx` on the GitHub mirror and count rows in `participants.tsv`. Parse DOIs from the free-text references with a regex. Output: `data/openneuro/datasets.csv`.
2. **Paper linkage and effect extraction.** Resolve DOIs to PubMed/PMC; for OA papers, pull full text; two annotators extract, per paper, the *primary* contrast, the reported peak statistic (t, z, F or r), its df/n, the threshold used (voxel-wise p, cluster-forming p, FWE/FDR), whether the peak was chosen a priori (ROI) or by search, and the reported n. Inter-annotator agreement (Cohen's kappa) on 50 papers before scaling. Convert statistics to Cohen's d (one-sample: d = t/sqrt(n); two-sample: d = t*sqrt(1/n1 + 1/n2)) and to r (r = sqrt(t^2/(t^2 + df))) with standard errors (`effects.py`).
3. **Selection-corrected estimators** (`winners_curse.py`). Model: reported effect d_hat ~ N(d, se^2), observed only if |d_hat/se| > c where c is the study's own threshold (voxel-wise p converted to z; for cluster-level thresholds we use the cluster-forming z as the conservative choice). Estimators: (i) analytical expected reported effect E[d_hat | selected] and Type M / Type S errors (Gelman & Carlin); (ii) conditional MLE with profile-likelihood CI (Zhong & Prentice, 2008); (iii) normal-normal empirical Bayes shrinkage across the corpus, stratified by task family; (iv) a simulated literature with the corpus's n distribution to check calibration of (ii)-(iii).
4. **Re-analysis subset** (`effects.py`). Select 20-30 task-fMRI datasets with fMRIPrep derivatives, n >= 16, a clearly defined primary contrast and a NeuroVault or in-paper peak. Fit first-level GLMs (Nilearn) from the fMRIPrep confounds-regressed BOLD; then (a) in-sample peak d (select the voxel with max t, report d there); (b) K-fold cross-validated peak d (select on K-1 folds, estimate on the held-out fold, average over folds; `cross_validated_peak_effect`); (c) split-half version for comparability with the older literature. Compare with the paper's peak d.
5. **Replication expectation.** For each paper, power at the corrected effect for the original n and the n needed for 80% power (`required_n`).
6. **Moderator analysis.** Mixed-effects meta-regression of log(inflation ratio) on log(n), task family, year, ROI-vs-search, threshold type, with paper as random effect (statsmodels MixedLM).

Tools: requests (GraphQL), pandas, numpy/scipy, statsmodels, Nilearn (GLMs, only for the re-analysis subset), DataLad/openneuro-py for derivatives, NeuroVault REST API.

### Worked example of the correction

A one-sample task contrast reported as peak t(19) = 4.2 with n = 20 at a voxel-wise p < 0.001 threshold gives d_hat = 4.2 / sqrt(20) = 0.94 with se ~ 0.26. The selection threshold is c = z_{0.9995} = 3.29, so only |d_hat| > 0.86 could have been reported. Under the truncated-normal model the conditional MLE (`winners_curse.conditional_mle`) shrinks the estimate to roughly d ~ 0.6, with a profile-likelihood CI that extends well below 0.3; the Type M exaggeration ratio at d = 0.6 and n = 20 is about 1.5 (`type_m_error`), and 80% power on the corrected effect needs n ~ 25-30 (`required_n`), versus n > 90 if the true effect were 0.3. The corpus table repeats this calculation for every paper, and the re-analysis subset tests whether the circular/cross-validated ratio measured on the data agrees with the model-based shrinkage.

### Annotation table

`data/papers/effects_annotated.csv` has one row per reported primary effect with the columns in `effects.ANNOTATION_COLUMNS`: `dataset_id, doi, contrast, design (one_sample | paired | two_sample | correlation), stat_type (t | z | F | r), stat_value, df, n1, n2, threshold_type (voxel_fwe | voxel_fdr | voxel_unc | cluster_fwe | roi_apriori), threshold_p, peak_selected_by_search (0/1), annotator`. `effects.annotation_to_effect` converts each row into `d, se_d, threshold_z`. Two annotators complete the first 50 rows independently; disagreements on `stat_value`, `n1` or `threshold_type` are adjudicated before the remaining rows are split.

## Evaluation & statistics

- Unit of analysis: one primary effect per paper (pre-specified as the first reported whole-brain peak for the paper's main contrast). Sensitivity: all reported peaks with paper as a random effect.
- Corrected-effect estimators are validated on simulated literatures matched to the corpus's n distribution and threshold mix (bias and coverage of the profile-likelihood CI at 95%).
- Funnel asymmetry: Egger regression of d_hat/se on 1/se (`funnel_asymmetry`), with the caveat that heterogeneity in true effects also produces asymmetry; we therefore stratify by task family and report both.
- The re-analysis comparisons use paired statistics within dataset (Wilcoxon on log ratios) and bootstrap CIs over datasets.
- Leakage prevention in the cross-validated estimator: peak selection and effect estimation use disjoint subjects; confound regression and smoothing are done per subject, not on the group.
- Multiple comparisons: the primary tests are H1-H4 (four pre-registered tests, Holm-corrected); moderator analyses are exploratory and reported with FDR.
- Nulls: (i) label-permuted contrasts in the re-analysis subset should give circular/cross-validated ratios that scale as predicted by the null distribution of the maximum; (ii) a synthetic "no selection" literature must give corrected effects unbiased at zero.

## Publishable angle

Headline result: "Across N OpenNeuro-linked papers with median n = 23, published peak effects are inflated by a median factor of X after selection correction; direct re-analysis of M datasets attributes Y% of that inflation to within-study peak selection and the remainder to publication selection; only Z% of effects would be expected to replicate at their original n."

Target venues: NeuroImage; Imaging Neuroscience; Nature Human Behaviour (meta-research); PLOS Biology (meta-research collection); OHBM (abstract).

Follow-ups: (a) a living dashboard that updates as OpenNeuro grows; (b) the same audit for EEG/MEG datasets on OpenNeuro; (c) using the corpus as an empirical prior for Bayesian analysis of new small-n studies.

## Related project

`openneuro-mriqc-audit` in this repository also indexes OpenNeuro but for image-quality metrics; this project is independent and only shares the idea of treating OpenNeuro as a population.

## Risks, confounds & mitigations

- **Thresholds vary and are often under-reported.** We record the threshold type and run sensitivity analyses over plausible c; cluster-level inference gets the cluster-forming z as a conservative bound.
- **Heterogeneous designs (one-sample, paired, two-sample, correlations).** Effect conversions are design-specific (`effects.py`) and designs are a stratification variable.
- **True heterogeneity mimics selection in funnel plots.** Egger is descriptive only; the corrected-effect estimators do not rely on funnel symmetry.
- **The paper's n may differ from the BIDS n** (exclusions). We record both and use the paper's analysed n for corrections; the difference itself is an interesting secondary outcome.
- **Re-analysis pipelines differ from the original** (software, smoothing). We report the *ratio* circular/cross-validated within our own pipeline, which is pipeline-invariant to first order, and treat the comparison to the paper's number as secondary.
- **API availability.** The GraphQL endpoint is occasionally down; the client retries with backoff and falls back to the GitHub mirrors.

## Milestones

- [ ] Pull the full OpenNeuro index; verify n against `participants.tsv` on 50 random datasets.
- [ ] Resolve DOIs; identify OA papers; annotate 50 papers for agreement, then the full corpus.
- [ ] Corrected-effect table with Type M/S, conditional MLE, EB; calibration on simulated literatures.
- [ ] Select the re-analysis subset; run first-level GLMs from fMRIPrep derivatives.
- [ ] Cross-validated vs circular peak effects; comparison with reported effects.
- [ ] Replication expectation and required n; moderator meta-regression.
- [ ] Pre-registration (OSF) of H1-H4 before effect extraction is complete; manuscript.

## Ethics / data-use notes

- All inputs are public metadata, public derivatives, and published papers; no consent issues beyond the datasets' own licences (most are CC0/PDDL).
- Respect API etiquette: page sizes <= 100, backoff on 429/5xx, cache responses in `data/` (git-ignored).
- Never commit data or derivatives; `data/` and `outputs/` are git-ignored.
- Report per-paper results in aggregate form in the main text; a per-paper supplementary table is appropriate for meta-research but should be framed as a property of small-n designs, not of authors.
