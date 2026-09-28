# OpenNeuro image-quality audit: QC exclusion as a hidden multiverse

**One sentence:** Use the MRIQC WebAPI's crowdsourced image-quality metrics, linked to OpenNeuro dataset and participant metadata, to quantify how much reported group differences and usable sample sizes across OpenNeuro would change under standard QC exclusion rules, and to publish a normative IQM reference chart by age and scanner.

| | |
|---|---|
| Status | proposal + starter code (API clients, MD5 linkage, normative models, multiverse engine) |
| Difficulty / timeline | MSc-level for the descriptive audit; 6-9 months to the full multiverse with re-analysed datasets |
| Compute | laptop for the API/statistics part (tables of 10⁵-10⁶ rows). Optional: MRIQC runs on a stratified sample of OpenNeuro datasets (~10-20 min per T1w on 4 cores; a few hundred CPU-hours total) |
| Package | `src/mriqc_audit` |

## Background

Automated image quality metrics (IQMs) from MRIQC (Esteban et al., 2017, *PLoS ONE*) are the de-facto standard for structural and functional MRI quality control, and MRIQC uploads them by default to a public database, the MRIQC WebAPI (Esteban et al., 2019, *Sci Data*), which now holds hundreds of thousands of records with scanner metadata. Separately, OpenNeuro (Markiewicz et al., 2021, *eLife*) hosts > 1,000 BIDS datasets with machine-readable `participants.tsv` (age, sex, group) and curated study-level metadata, and offers a GraphQL API.

Head motion and other quality problems bias morphometry (Reuter et al., 2015, *NeuroImage*; Alexander-Bloch et al., 2016, *Hum Brain Mapp*; Rosen et al., 2018, *NeuroImage*) and functional connectivity (Power et al., 2012; Satterthwaite et al., 2012; Van Dijk et al., 2012, all *NeuroImage*), and the usual remedy, threshold-based exclusion, removes patients and children disproportionately (Pardoe et al., 2016, *NeuroImage*; Nebel et al., 2022, *NeuroImage*). Comparisons of QC pipelines in large pediatric cohorts (e.g. the 2023 *NeuroImage* study comparing QC approaches across three pediatric datasets, and Elyounssi and colleagues' 2023 work on bias in automated developmental MRI analyses) show that different rules exclude different children and that the excluded differ systematically in age, symptoms and IQ. A 2025 preprint proposes propensity-score matching on scan quality as an alternative, and a 2026 *Diagnostics* paper reports 21 % vs 12 % exclusion in autistic vs control participants in ABIDE-II at mean FD > 0.2 mm.

## The research gap

**What exists.** (i) The WebAPI paper described the database and its normative distributions in 2019 and a 2025 paper ("Development of reference Image Quality Metrics for quantitative MRI research using MRIQC") derives reference IQM values, but neither conditions on participant age or links records to the datasets and populations they came from. (ii) QC-exclusion bias has been shown within single cohorts (ABCD, PNC, ABIDE, HBN) using in-house ratings. (iii) Multiverse/specification-curve thinking has been applied to preprocessing (Botvinik-Nezer et al., 2020, *Nature*; Dafflon et al., 2022, *Nat Commun*) but not to QC exclusion across many independent datasets.

**What is missing (2023-2026 literature checked Sept 2026).**
1. No one has joined the crowdsourced IQMs to OpenNeuro metadata. This is feasible because `provenance.md5sum` in each WebAPI record is the MD5 of the input file bytes and OpenNeuro's git-annex keys are MD5E, so a shallow `git clone` (symlinks only, no images) links IQM records to dataset, subject and hence age/sex/group (see `openneuro_client.annex_md5_index`).
2. There is no age-conditional normative chart for IQMs. Rules like FD > 0.2 mm are applied identically to 6-year-olds and 60-year-olds on 1.5T and 3T scanners; a centile chart by age x field strength x manufacturer x MRIQC version makes "poor quality" a relative rather than absolute judgement.
3. Nobody has quantified, across the whole repository, how much sample size and group effect sizes move under the set of QC rules that studies actually use, i.e. the size of the QC multiverse and how much of it is driven by differential exclusion of clinical or pediatric participants.

## Research questions and hypotheses

1. **Coverage.** H1: at least 30 % of OpenNeuro T1w/BOLD images have a WebAPI record recoverable by MD5 linkage; coverage is higher for datasets published after MRIQC-on-OpenNeuro was introduced and for 3T Siemens data.
2. **Age dependence.** H2: after adjusting for scanner descriptors and MRIQC version, IQM centiles are strongly non-monotonic in age (worse in < 12 y and > 65 y), so that a fixed FD or CJV threshold excludes > 25 % of children but < 5 % of young adults.
3. **Differential exclusion.** H3: in datasets with a clinical `group` column, exclusion risk ratios (patients vs controls) exceed 1.3 for a majority of datasets under mean FD > 0.2 mm, and are larger for neurodevelopmental than for adult psychiatric groups.
4. **Hidden multiverse.** H4: across common rule sets (FD 0.2/0.25/0.5 mm, fd_perc 20 %, CJV 0.45, per-dataset worst-10 %, combined), the between-cell range of Hedges' g for a given group difference (e.g. global grey-matter volume from FreeSurfer-derived outcomes, or the dataset's own reported contrast where re-analysable) exceeds 0.2 in more than half of eligible datasets, and sign consistency is < 90 %.
5. **Relative vs absolute rules.** H5: per-dataset relative rules (worst 10 %) yield smaller differential-exclusion RRs than absolute rules but do not reduce the effect-size range, i.e. the multiverse is not fixed by relative thresholds.
6. **Temporal drift.** H6: median IQMs improve with acquisition year (proxied by dataset creation date and scanner model), so studies pooling old and new datasets confound quality with cohort.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MRIQC WebAPI | IQMs + `bids_meta` scanner fields + MRIQC version | > 10⁵ T1w and BOLD records (2019 paper: > 100k; growing) | open, no auth (GET) | https://mriqc.nimh.nih.gov/ (API root `/api/v1/`) |
| OpenNeuro GraphQL | dataset listing, `participants.tsv`-derived age/sex/group, study metadata | > 1,000 datasets, > 40,000 participants | open (CC0 datasets) | https://openneuro.org/crn/graphql |
| OpenNeuroDatasets GitHub mirrors | git-annex MD5E keys for IQM-to-file linkage | one small git repo per dataset | open | https://github.com/OpenNeuroDatasets |
| OpenNeuro S3 | raw images for local MRIQC runs on a stratified sample | selective (`datalad get`) | open | s3://openneuro.org |
| ABIDE-II / ADHD-200 (optional) | external replication of differential exclusion | ~1,100 / ~900 subjects | open (INDI) | http://fcon_1000.projects.nitrc.org |

## Group-label vocabulary and QC rule set

`participants.tsv` group columns are free text. The audit maps them with an explicit, versioned vocabulary (two raters, disagreements to manual review):

| Binary class | Accepted labels (case-insensitive) | Domain tag |
|---|---|---|
| control | control, ctrl, hc, healthy, td, typical, cn, nc, comparison | – |
| patient | asd, autism | neurodevelopmental |
| patient | adhd | neurodevelopmental |
| patient | sz, scz, schizophrenia, psychosis, chr | psychiatric |
| patient | mdd, depression, bipolar, bd, anxiety, ptsd | psychiatric |
| patient | ad, mci, dementia, ftd | neurodegenerative |
| patient | pd, parkinson, ms, epilepsy, stroke, tbi | neurological |
| ambiguous | anything else (e.g. `group1`, `A`, drug arms) | excluded from H3/H4 |

QC rules enumerated by `qc_sensitivity.standard_rule_grid` (extendable):

| Axis | Rules | Source of threshold |
|---|---|---|
| BOLD motion | mean FD > 0.2, 0.25, 0.5 mm; FD > 0.2 mm in > 20 % of volumes | Power et al., 2012; Satterthwaite et al., 2012; MRIQC default `fd_thres` |
| T1w quality | CJV > 0.45; per-dataset worst 10 % CJV; per-dataset lowest 10 % SNR | Ganzetti et al., 2016; relative rules common in ABCD/HBN papers |
| Combined | motion AND T1w; motion OR T1w | typical multimodal exclusion |
| Normative | IQM centile > 90 or > 95 for age × scanner (from `normative.py`) | this project |

## Quick start (module API)

```python
from mriqc_audit import MRIQCClient, OpenNeuroClient, dedupe_records, annex_md5_index
from mriqc_audit.openneuro_client import link_iqms_to_openneuro
from mriqc_audit.normative import QuantileNormativeModel
from mriqc_audit.qc_sensitivity import QCRule, qc_multiverse, specification_summary

iqms = dedupe_records(MRIQCClient().fetch("T1w", max_pages=5))            # 5,000 records
on = OpenNeuroClient()
subjects = on.subject_metadata(max_datasets=100)                          # participantId/age/sex/group
idx = annex_md5_index("data/openneuro_git/ds000030")                      # after `git clone --depth 1`
linked = (link_iqms_to_openneuro(iqms, idx, "ds000030")
          .merge(subjects, left_on=["dataset_id", "bids_sub"], right_on=["dataset_id", "participantId"]))

qm = QuantileNormativeModel("cjv", cat_cols=("bids_meta.Manufacturer", "bids_meta.MagneticFieldStrength"),
                            log_transform=True).fit(linked)
linked["cjv_centile"] = qm.centile(linked)

grid = {"cjv": [None, QCRule("cjv", 0.45), QCRule("cjv", 0.9, relative=True, by="dataset_id")]}
res = qc_multiverse(linked, grid, group_col="group_binary", outcome_col="gm_vol", covariates=["age"])
print(specification_summary(res))
```

## Methods

1. **Harvest** (`mriqc_client.py`): page through `/api/v1/T1w` and `/api/v1/bold`; flatten records; de-duplicate by (md5, MRIQC version); keep version because IQM definitions changed between major releases.
2. **Metadata** (`openneuro_client.py`): list datasets with cursor pagination; pull `summary.subjectMetadata` and raw `participants.tsv`; parse `group` columns with a small vocabulary (control/HC/TD vs patient labels) and flag pediatric datasets (median age < 18).
3. **Linkage**: shallow-clone each dataset's git mirror, index MD5E keys, join to IQMs on `provenance.md5sum`, then to participants on the BIDS `sub-` entity. Report coverage (H1) and check for MD5 collisions across snapshots.
4. **Normative charts** (`normative.py`): `QuantileNormativeModel` (quantile regression, cubic B-spline in age, one-hot manufacturer/field strength/MRIQC major version) for 5/25/50/75/95th centiles; `LocationScaleModel` (log-IQM, spline mean and spline log-scale) for z-scores; calibration via empirical coverage. Fit separately for T1w (`cjv`, `cnr`, `snr_total`, `efc`, `fber`, `qi_1`) and BOLD (`fd_mean`, `fd_perc`, `dvars_std`, `tsnr`, `gsr_x`).
5. **Multiverse engine** (`qc_sensitivity.py`): `QCRule` objects (absolute or per-dataset relative), Cartesian product of rule axes, and per-cell retained n, exclusion risk ratio with CI, Hedges' g with CI, Welch p, and achieved power at the full-sample g. Outcomes: (a) within-dataset group contrasts on IQM-independent measures available from OpenNeuro derivatives (FreeSurfer or fMRIPrep outputs where published), (b) for a curated subset of ~30 datasets with a published group difference, re-computed contrasts.
6. **Specification curves and meta-summary**: per dataset, `specification_summary`; across datasets, mixed-effects meta-regression of g-range on pediatric fraction, clinical group type, scanner and year.

## Evaluation and statistics

- Normative-model validity: 5-fold CV coverage of centiles by age bin and scanner (nominal vs empirical, tolerance ±3 percentage points); comparison of quantile-regression vs location-scale models by pinball loss.
- Differential exclusion: log risk ratios with random-effects meta-analysis across datasets (DerSimonian-Laird), heterogeneity I², moderators (pediatric, clinical domain, field strength).
- Multiverse: the primary quantity is the per-dataset range and sign-consistency of g across cells; inference uses a permutation null in which QC-rule assignment is shuffled within dataset (does the observed range exceed what random subsampling of the same size produces?), separating "fewer subjects" from "different subjects".
- Multiple comparisons: Benjamini-Hochberg across datasets for the RR tests; the multiverse itself is descriptive, reported in full.
- Leakage/independence: repeated uploads of the same image (same MD5, same version) are collapsed; re-uploads across versions are analysed as version strata, not pooled; datasets contributing < 20 participants are excluded from effect-size cells.
- Robustness: repeat with IQMs recomputed locally (single MRIQC version) on a stratified sample of 50 datasets to bound version/upload heterogeneity.

## Publishable angle

**Headline result:** a repository-wide figure in which each OpenNeuro dataset is a point (pediatric fraction on x, differential-exclusion RR on y, size = g-range across QC rules), plus an age-by-scanner IQM centile chart released as a lookup table and a small `mriqc_audit.normative` API so that authors can report "FD at the 85th centile for age and scanner" instead of an absolute cut-off. Secondary: the linkage method itself (IQMs to participants without downloading images).

Target venues: *NeuroImage*, *Imaging Neuroscience*, *Scientific Data* (for the linked resource + charts), *Human Brain Mapping*; OHBM abstract for the linkage/coverage result.

Follow-ups: (i) age-conditional QC thresholds embedded in MRIQC reports; (ii) extend to diffusion (`dwi` IQMs when the WebAPI adds them) and PET; (iii) test propensity-matching on quality vs exclusion in the same multiverse; (iv) per-site drift monitoring for large consortia.

## Risks, confounds and mitigations

| Risk | Mitigation |
|---|---|
| MD5 linkage coverage low (re-compressed inputs, later snapshots) | index all snapshots via `git log --all`; supplement with local MRIQC runs; report coverage as a result, not a precondition |
| WebAPI records are not a random sample of OpenNeuro (self-selection by MRIQC users) | compare linked vs un-linked datasets on metadata; weight by inverse coverage |
| IQM definitions differ across MRIQC versions | stratify by major version; local re-run on a sample with one version |
| `participants.tsv` age/group heterogeneously coded (`n/a`, ranges, months) | explicit parsing vocabulary, manual curation of the top-200 datasets, sensitivity analysis |
| Group labels ambiguous (which is "patient"?) | curated mapping with two raters; drop datasets with ambiguous groups from H3/H4 |
| Outcomes for effect sizes not uniformly available | tier the multiverse: IQM-only (all datasets), derivative-based (datasets with published FreeSurfer/fMRIPrep outputs), re-analysed (curated subset) |
| Server load / rate limits | polite pagination with sleeps, append-only caches, resume by page |

## Milestones

- [ ] Harvest WebAPI T1w + BOLD; version and duplicate audit (week 1-2)
- [ ] OpenNeuro listing + participants; group/age parsing vocabulary (week 2-4)
- [ ] MD5 linkage on all datasets; coverage report (H1) (week 4-6)
- [ ] Normative charts with CV calibration (H2, H6) (month 2-3)
- [ ] Differential exclusion meta-analysis (H3) (month 3-4)
- [ ] Multiverse on derivative-based outcomes, curated re-analysis subset (H4, H5) (month 4-7)
- [ ] Local MRIQC re-run on stratified sample for robustness (month 5-7)
- [ ] Release linked table + centile lookup + manuscript (month 8-9)

## Ethics and data-use notes

- OpenNeuro datasets are CC0 and the WebAPI is public; participant-level rows are still kept out of version control and are only reported in aggregate.
- Some OpenNeuro datasets carry additional consent restrictions in their README; respect them before re-running MRIQC or re-uploading IQMs.
- The MD5 linkage re-associates hashed WebAPI records with dataset/subject labels that are already public on OpenNeuro; it adds no identifying information, but the linked table should not be redistributed with the hashed `subject_id` fields removed from context (keep the mapping reproducible, not published).
- No data go to third-party LLM APIs.
