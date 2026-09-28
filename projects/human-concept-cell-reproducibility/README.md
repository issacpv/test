# human-concept-cell-reproducibility

**A criterion-multiverse and cross-dataset reproducibility audit of "concept cells" in open human single-neuron recordings (DANDI/NWB): how much of the reported prevalence, selectivity strength and regional distribution of concept cells depends on the selection statistic, the null model and the spike-sorting pipeline rather than on the brain.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (NWB-agnostic session container, five selectivity criteria with drift-aware nulls, split-half selection-bias correction, hierarchical prevalence model, synthetic-session generator).
- Difficulty: MSc thesis to early-PhD chapter. Statistics-heavy; no GPU.
- Timeline: 6-9 months (1 month data harmonisation across NWB conventions, 2 months criterion multiverse, 2 months hierarchical modelling and nulls, 1-2 months writing).
- Compute: laptop. All open datasets together are a few tens of GB of NWB files; a single session (spike times + trial table) fits in memory. Permutation nulls over ~5k units x 5 criteria x 1k permutations run in hours on 8 cores.

## Background

Concept cells are medial temporal lobe (MTL) neurons that respond selectively and invariantly to a specific person, object or place (Quian Quiroga, Reddy, Kreiman, Koch & Fried, 2005, Nature; Quian Quiroga, 2012, Nature Reviews Neuroscience). They are the dominant single-neuron account of human declarative memory: sparse, explicit, rapidly formed associations (Ison, Quian Quiroga & Fried, 2015, Neuron; De Falco, Ison, Fried & Quian Quiroga, 2016, Nature Communications; Rey et al., 2020, Current Biology), and reviews continue to build on prevalence numbers such as "roughly a third of MTL units are concept cells" (Rutishauser, Reddy, Mormann and colleagues; The Architecture of Human Memory, 2021, Journal of Neuroscience; Arruda et al., 2026, Acta Physiologica).

Every one of those claims rests on a *selection statistic* applied to a few dozen trials per stimulus in a handful of patients. Groups use different tests (one-way ANOVA across stimulus identities in a fixed window; bin-wise rank-sum tests against baseline as in Mormann et al., 2008, Journal of Neuroscience; response-strength / z-score thresholds; Poisson-GLM approaches), different alphas, different windows, different spike sorters (OSort: Rutishauser, Schuman & Mamelak, 2006, Journal of Neuroscience Methods; Wave_clus: Chaure, Rey & Quian Quiroga, 2018, Journal of Neurophysiology; Combinato: Niediek et al., 2016, PLoS ONE) and different unit-inclusion rules. Until recently the raw data were unavailable, so these choices could not be compared on the same neurons.

That has changed. Rutishauser's lab released NWB datasets on DANDI: a new/old recognition-memory task (Chandravadia et al., 2020, Scientific Data; DANDI:000004) and a Sternberg working-memory task (Kyzar et al., 2024, Scientific Data; DANDI:000469, 1,809 units, 41 sessions, 21 patients, reporting 32.8% concept cells in MTL vs 5.4% in medial frontal cortex under their criterion). A human single-neuron dataset for object recognition was published in Scientific Data in 2024 as well. These are the first open, multi-session, multi-region, multi-task human single-neuron corpora, and they make a reproducibility audit feasible for the first time.

## The research gap

**What has been done**

- Prevalence and selectivity of concept cells have been reported per lab with a fixed in-house criterion (Quian Quiroga et al., 2005; Mormann et al., 2008; Rey et al., 2020; Kamiński et al., 2017, Nature Neuroscience; Kyzar et al., 2024).
- Sparseness of MTL coding was estimated analytically from response probabilities (Waydo, Kraskov, Quian Quiroga, Fried & Koch, 2006, Journal of Neuroscience; Quian Quiroga, Kreiman, Koch & Fried, 2008, Trends in Cognitive Sciences), again under one criterion.
- The dataset papers include technical validation that reproduces their own earlier proportions with their own pipeline (Chandravadia et al., 2020; Kyzar et al., 2024), plus spike-sorting quality metrics (Rutishauser et al., 2015, Nature Neuroscience).
- Circularity ("double dipping") in neural selectivity analyses is well characterised in general (Kriegeskorte, Simmons, Bellgowan & Baker, 2009, Nature Neuroscience) but has not been quantified for human concept-cell effect sizes specifically.

**What is missing (checked against 2023-2026 literature)**

1. No study applies *several* published concept-cell criteria to the *same* open recordings and reports the criterion x dataset x region interaction in prevalence. The 32.8% vs 5.4% MTL-vs-MFC contrast, for instance, has never been shown to be criterion-invariant.
2. No published concept-cell prevalence uses a null model that preserves slow firing-rate drift and trial blocking. Stimuli are typically presented in blocks or with repeats clustered in time; non-stationary units then pass ANOVA-type tests at inflated rates. A circular-shift / block-permutation null is standard elsewhere (e.g. place-cell analysis) but absent here.
3. Reported selectivity strengths (best-vs-rest firing-rate ratios, "response magnitude") are selection-biased because the preferred stimulus is chosen and quantified on the same trials. No split-half, cross-validated effect size for human concept cells has been published.
4. Non-independence is ignored: units on the same microwire and sessions within the same patient are treated as independent Bernoulli draws when computing proportions and their CIs. Hierarchical (unit-in-wire-in-session-in-patient) prevalence estimates do not exist.
5. Whether spike-sorting quality (SNR, isolation distance, ISI violations, waveform stability across the session; all available in the NWB files) predicts "concept-cell" status has not been reported. If it does, prevalence partly reflects sorting, not coding.

## Research questions / hypotheses

1. **H1 (criterion dependence).** Across five criteria (ANOVA, Kruskal-Wallis, bin-wise rank-sum vs baseline with Holm correction, response-strength index at a fixed z-threshold, Poisson GLM likelihood-ratio test), MTL concept-cell prevalence in the same dataset varies by more than a factor of two. Test: prevalence per criterion with session-level cluster-bootstrap CIs; Cochran's Q across criteria; pairwise unit-level agreement (Cohen's kappa, Jaccard).
2. **H2 (drift inflation).** Replacing the i.i.d. trial-shuffle null with a circular-shift null that preserves the trial order reduces the number of units passing at alpha = 0.05 by at least 20% in blocked designs, and the reduction is larger in units with a significant linear rate trend. Test: paired comparison of per-unit p-values under the two nulls; regression of the p-value change on a drift index.
3. **H3 (selection bias).** Split-half cross-validated selectivity indices are at least 30% smaller than in-sample indices, and the shrinkage is largest for units with few trials per stimulus. Test: paired in-sample vs cross-validated index with unit-level bootstrap; regression of shrinkage on trials-per-stimulus.
4. **H4 (regional gradient survives).** After hierarchical modelling and criterion harmonisation, the MTL > MFC prevalence gradient (Kyzar et al., 2024) remains, but the hippocampus-vs-amygdala ordering is criterion-dependent. Test: mixed-effects logistic model, region x criterion interaction, likelihood-ratio tests.
5. **H5 (sorting-quality confound).** Units labelled concept cells have higher SNR and lower ISI-violation rates than non-selected units, and adjusting prevalence for quality metrics changes regional comparisons. Test: logistic mixed model with quality covariates; sensitivity analysis excluding units below quality thresholds.
6. **H6 (cross-task, within-patient replication).** For patients recorded in more than one task or session with overlapping stimulus sets, the per-stimulus preference of a "concept cell" replicates above chance across sessions. Test: permutation of stimulus labels across sessions (only feasible where stimuli overlap; report feasibility).

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| DANDI:000004 (Chandravadia et al., 2020, Sci Data) | Human MTL single units during a new/old recognition-memory task with images of five visual categories; spike times, waveforms, trial table, electrode locations, sorting-quality metrics; NWB | ~1.5k units, tens of sessions and patients (see dandiset page for exact counts) | Open (no registration) | https://dandiarchive.org/dandiset/000004 |
| DANDI:000469 (Kyzar et al., 2024, Sci Data) | Sternberg working-memory task; MTL + medial frontal cortex units; encoding/maintenance/probe epochs; NWB | 1,809 units, 41 sessions, 21 patients | Open | https://dandiarchive.org/dandiset/000469 |
| Human single-neuron dataset for object recognition (Sci Data, 2024) | Single units during viewing of natural object images; used as a third, differently-collected corpus for criterion transfer | see paper | Open (NWB; follow the paper's data-availability statement) | https://www.nature.com/articles/s41597-024-04265-1 |
| Other human single-unit dandisets found by searching DANDI for "human" + "single unit" | Additional MTL/frontal recordings as they appear (the downloader enumerates them) | varies | Open | https://dandiarchive.org/ |
| Rutishauser lab release code (recogmem / workingmem NWB repositories) | Reference implementations of the labs' own selection criteria, used as the anchor criterion | code | Open | https://github.com/rutishauserlab/workingmem-release-NWB |

No credentialed data are needed. All datasets are de-identified epilepsy-monitoring recordings released under DANDI's terms.

## Methods

1. **Harmonised session loader** (`src/concept_cells/nwb_loader.py`): read NWB `units` (spike times, electrode, quality columns) and `trials` (stimulus identity/category, onset times, epoch boundaries) with `pynwb`, streamed from DANDI via `remfile`/`fsspec` or from local files, into a plain `SessionData` container (spike times per unit, trial table as a DataFrame, region labels). The container is what all downstream code consumes, so datasets with different NWB column names are mapped once in a per-dandiset config.
2. **Trial-window firing rates**: per unit, spike counts in the analysis window (default 0.2-1.0 s after stimulus onset, configurable per task) and a baseline window (-1.0 to 0 s), plus a bin-wise matrix (100 ms bins, 50 ms steps) for bin-wise tests.
3. **Criterion multiverse** (`src/concept_cells/selectivity.py`):
   - C1: one-way ANOVA on window rates across stimulus identities, alpha = 0.05 (Rutishauser-lab-style).
   - C2: Kruskal-Wallis, alpha = 0.05.
   - C3: bin-wise rank-sum vs baseline for each stimulus, Holm-corrected over bins, unit selected if any stimulus passes (Mormann-style).
   - C4: response-strength index: best-stimulus mean rate vs pooled others, z >= 3 and rate ratio >= 2.
   - C5: Poisson GLM likelihood-ratio test of a stimulus factor with a slow-drift spline covariate (trial index), so drift is modelled rather than ignored.
   Each criterion returns per-unit statistics, p-values, preferred stimulus and a selection flag.
4. **Nulls**: (a) i.i.d. label permutation; (b) circular shift of the stimulus-label sequence relative to the response sequence (preserves both autocorrelation structures); (c) block permutation when the task has explicit blocks. Empirical p-values with 1,000+ permutations per unit.
5. **Selection-bias correction**: split trials into odd/even halves; choose the preferred stimulus on one half, quantify selectivity index on the other, average both directions; repeat over random splits.
6. **Hierarchical prevalence** (`src/concept_cells/prevalence.py`): unit-level Bernoulli outcomes with logistic mixed effects (session and patient random intercepts; electrode/wire random intercept where available; statsmodels `BinomialBayesMixedGLM` or cluster-robust GLM), plus a beta-binomial per-session model and session-level cluster bootstrap for CIs; heterogeneity across datasets via Cochran's Q and I^2; agreement matrices between criteria.
7. **Sorting-quality covariates**: pull SNR, isolation distance, ISI-violation fraction and waveform-stability columns from the NWB `units` table where present; include in the mixed model.
8. **Synthetic sessions** (`src/concept_cells/synthetic.py`): Poisson/gamma spike trains with known selective units, controllable rate drift, blocked stimulus order and wire-level shared noise, used to calibrate false-positive rates of every criterion under every null before touching real data.
9. **Tools**: `pynwb`, `dandi`, `remfile`, `numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn` (only for permutation utilities), `matplotlib`.

## Evaluation & statistics

- Primary estimands: prevalence per (dataset, region, criterion, null) with 95% cluster-bootstrap CIs; criterion x region interaction from the mixed model; cross-validated selectivity index and its shrinkage relative to in-sample.
- False-positive calibration: every criterion/null pair is run on synthetic null sessions with matched trial counts, drift and block structure; the empirical alpha is reported next to the real-data prevalence. A criterion with empirical alpha > 0.07 at nominal 0.05 under realistic drift is flagged.
- Leakage prevention: preferred-stimulus choice and quantification always use disjoint trials; nothing is tuned on real data before the pre-registered analysis; all per-dataset window/epoch choices are fixed from the dataset papers.
- Multiple comparisons: within-unit bin-wise tests are Holm-corrected; across units, prevalence is reported at nominal alpha (as in the literature) *and* after Benjamini-Hochberg at q = 0.05, because the field's numbers are nominal-alpha numbers and the comparison itself is a result.
- Non-independence: cluster bootstrap by session (and by patient) for every proportion; ICC of "selected" status within wires reported.
- Heterogeneity: Cochran's Q, I^2 and prediction intervals for prevalence across datasets, computed per criterion.
- Sensitivity: window length (0.2-1.0 vs 0.1-1.0 vs 0.3-1.5 s), baseline definition, minimum trials per stimulus (5/8/10), unit-quality thresholds, inclusion of multi-units.

## Publishable angle

- **Headline**: "Concept-cell prevalence in human MTL ranges from X% to Y% across published criteria on identical neurons; drift-preserving nulls remove Z% of nominal detections; cross-validated selectivity is W% lower than reported; the MTL > frontal gradient survives, the hippocampus-amygdala ordering does not." A short table of calibrated false-positive rates per criterion is itself citable.
- Target venues: *Journal of Neuroscience* (methods/audit), *eLife* (Research Advance on the Kyzar/Chandravadia releases), *Journal of Neuroscience Methods*, *Scientific Data* (if the harmonised units table with multiverse labels is released as a resource), *PLoS Computational Biology*.
- Follow-ups: extend to invariance tests (same concept across modalities) where datasets include text/audio; population-level decoding as a criterion-free alternative; apply the same audit to "time cells" and "memory-selective" units in the same datasets; a community "concept-cell detector" package with calibrated defaults.

## Risks, confounds & mitigations

- **Different tasks, different stimuli**: prevalence is not comparable across tasks in absolute terms. Mitigation: the primary estimand is the *criterion effect within dataset*; cross-dataset comparisons use the mixed model with dataset as a random effect and report heterogeneity rather than pooling.
- **Few trials per stimulus** (often 5-10): split-half estimates are noisy. Mitigation: repeated random splits; report shrinkage as a function of trial count; simulations to show the estimator is unbiased at those counts.
- **NWB heterogeneity**: column names and epoch definitions differ between dandisets. Mitigation: per-dandiset mapping config with unit tests; log every unmapped column.
- **Spike-sorter choice cannot be varied** (the releases contain sorted units, not raw broadband in all cases). Mitigation: use provided quality metrics as covariates; where raw data exist, re-sort a subset with Wave_clus/Combinato as a sensitivity analysis.
- **Non-stationarity is real biology too** (attention, fatigue). Mitigation: report both nulls; treat drift-robust prevalence as a lower bound, not the "truth".
- **Small number of patients** (~20-40 per dataset): patient-level random effects have few levels. Mitigation: report patient-level bootstrap and Bayesian mixed models with weakly informative priors; avoid over-interpreting variance components.

## Milestones

- [ ] Enumerate human single-unit dandisets; write per-dandiset NWB mapping configs; load all sessions into `SessionData` and verify unit/trial counts against the papers.
- [ ] Reproduce each dataset's own concept-cell proportion with its own criterion (anchor result).
- [ ] Synthetic calibration: empirical alpha for all criterion x null pairs under drift/block scenarios.
- [ ] Criterion multiverse on real data; agreement matrices; prevalence table with cluster-bootstrap CIs.
- [ ] Split-half selection-bias analysis; shrinkage vs trials-per-stimulus.
- [ ] Hierarchical prevalence model with region, criterion, quality covariates; heterogeneity across datasets.
- [ ] Sensitivity analyses (windows, thresholds, multi-units); pre-registered vs exploratory tables.
- [ ] Release harmonised units table (unit id, dataset, region, quality metrics, all criterion flags) + code; write manuscript.

## Ethics / data-use notes

- All datasets are de-identified human intracranial recordings released on DANDI under their stated licenses (typically CC-BY-4.0); cite the dataset papers and the dandiset DOIs.
- Do not attempt re-identification; do not combine with external patient-level information.
- Never commit NWB files or derived per-trial tables that could be considered raw data; commit only aggregate tables and code.
- If any restricted human dataset is later added (e.g. data available on request), keep it out of this repository and out of third-party cloud/LLM services unless the data-use agreement explicitly allows it.

## Related projects (kept self-contained here)

- `neuropixels-representational-drift` (drift-aware nulls for single-unit analyses in rodent data).
- `ibl-brainwide-decoding` (population-level decoding as a criterion-free alternative to single-unit selection).
