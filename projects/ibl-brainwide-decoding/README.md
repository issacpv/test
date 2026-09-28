# ibl-brainwide-decoding

**Which brain-wide decoding claims replicate across laboratories?** A cross-region, cross-lab benchmark
of task-variable decoding (choice, stimulus, block prior, reward) on the International Brain Laboratory
Brain-wide Map (~700 Neuropixels insertions, 12 labs, 139 mice) with *lab-as-random-effect* hierarchical
models, unit-yield/quality dependence, and a cross-dataset transfer test against the Allen Visual Coding
Neuropixels survey.

## Status / difficulty / timeline / compute

- Status: design + starter code. No results yet.
- Difficulty: MSc-level for the core benchmark (decoders + mixed models), PhD-level if the
  latent-dynamics transfer part is pursued in depth. 6–9 months.
- Compute: the IBL BWM spike-sorted data are ~3–4 TB in total but per-session spike/cluster tables
  are ~100–500 MB; a 16-core workstation with 64 GB RAM and 2 TB disk suffices. Ridge/logistic decoders
  with nested CV over ~700 insertions × ~280 regions × 4 targets × 200 pseudo-sessions ≈ 10^3–10^4
  CPU-hours; use a SLURM cluster or spread across days. No GPU needed unless LFADS-style models are added.

## Background

The IBL released a brain-wide map of neural activity in mice performing a standardised visual
decision task (International Brain Laboratory et al., 2025, *Nature*), with a companion paper on
brain-wide representations of the block prior (Findling et al., 2025, *Nature*). Both report per-region
decoding of task variables with pseudo-session / imposter-session nulls (Harris, 2020, bioRxiv,
"Nonsense correlations in neuroscience") and combine sessions with Fisher's method. The consortium also
ran an explicit reproducibility study at one repeated insertion trajectory (10 labs, 121 replicates):
region identity was more decodable than lab identity and task-variable decodability was comparable
across labs (International Brain Laboratory, Banga et al., 2024, *eLife* RP100840), but targeting error
and some single-unit metrics varied substantially across labs.

That leaves an obvious scaling question. The reproducibility paper covered five regions on one
trajectory; the BWM covers ~280 regions with each region typically recorded in ≥ 2 labs by design. Nobody
has yet asked, region by region across the whole brain: how much of the between-session variance in
"region X encodes choice" is lab variance, how many region-level claims are driven by a single lab, and
how much of the lab effect is explained by unit yield and quality rather than biology. Meanwhile
foundation-model work on the same data (NEDS, Zhang et al., 2025, *ICML*; POYO, Azabou et al., 2023,
*NeurIPS*; the "universal translator" of Zhang et al., 2024, *NeurIPS*; BrainWideBench, arXiv:2609.22064,
2026) benchmarks *transfer to unseen animals*, not lab reproducibility, and does not compare against the
Allen Visual Coding survey (Siegle et al., 2021, *Nature*), the other large public Neuropixels resource.

## The research gap

**Done:** brain-wide per-region decoding with pseudo-session nulls (IBL 2025); lab reproducibility at
one trajectory (Banga et al., 2024); multi-session decoders and transfer benchmarks (NEDS 2025;
BrainWideBench 2026); mixed-effects statistics recommended for nested neuroscience data (Aarts et al.,
2014, *Nat Neurosci*; Yu et al., 2022, *Neuron*) but rarely applied to decoding scores.

**Missing (this project):**

1. A brain-wide *variance partition* of decoding performance: score ~ region + (1 | lab) + (1 | subject
   within lab) + covariates(unit yield, QC pass rate, drift, trial count), with region-specific lab ICCs
   and leave-one-lab-out re-tests. Output: a list of regions whose BWM decoding claim is (a) robust
   across labs, (b) lab-dominated, (c) yield-dominated.
2. A *unit-yield curve* per region (decoding vs. number of units by subsampling) to separate "region
   carries no information" from "insertion caught too few units"; test whether lab differences vanish
   at matched yield.
3. A *cross-dataset transfer* test for visual regions (VISp/VISl/VISal/VISam/VISpm/VISrl, LGd, LP, CA1,
   DG, SCs): stimulus-evoked decoders trained on IBL (Gabor contrast) evaluated on Allen Visual Coding
   (full-field gratings / flashes) and vice versa after feature harmonisation, to estimate how much the
   region ranking of visual information is protocol-specific.
4. Whether *latent-dynamics* preprocessing (GPFA-style factor analysis, Yu et al., 2009, *J Neurophysiol*)
   reduces the lab component of variance relative to raw binned counts, and whether latent dimensionality
   needed for decoding is itself reproducible across labs.

If a 2026 IBL paper reports lab-level variance components brain-wide, sharpen towards (2)–(4): the
yield-matched comparison and the cross-dataset ranking stability remain untested.

## Research questions / hypotheses

1. **H1.** For most regions with significant choice/stimulus decoding in the BWM, the lab ICC of the
   null-corrected decoding score is < 0.2 (lab explains < 20 % of between-session variance). Test:
   mixed model per region; bootstrap CI on ICC; report the fraction of regions with ICC > 0.2.
2. **H2.** Regions whose decoding claim disappears in leave-one-lab-out re-analysis are enriched for
   low mean unit yield (< 10 good units per insertion) — logistic regression of "fragile" on yield and
   n_labs.
3. **H3.** Lab differences in decoding score are reduced by ≥ 50 % after subsampling to matched unit
   counts (compare lab variance components at raw vs. matched yield).
4. **H4.** The rank order of visual-region stimulus decodability (VISp > LGd > VISl ... > CA1) is
   preserved between IBL and Allen datasets (Spearman ρ > 0.7 across ≥ 10 regions), while the absolute
   scores are not (protocol effect).
5. **H5.** Block-prior decoding (probabilityLeft) is the least reproducible target across labs (largest
   lab ICC), because it relies on slow signals sensitive to drift and session length.
6. **H6.** Factor-analysis latents (10–20 dims) give lower lab ICC than raw counts at equal decoding
   accuracy.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| IBL Brain-wide Map (2025 release) | Spike-sorted Neuropixels (spikes.times/clusters, clusters.metrics, channels.brainLocationIds_ccf_2017 → Beryl regions), trials (choice, contrast, probabilityLeft, feedbackType, stimOn/feedback times), session→lab/subject | 459 sessions, 699 insertions, 139 mice, 12 labs; ~3 TB total, ~1 GB per session subset | Open (public Alyx; no account: user `intbrainlab`, password `international`) | https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html ; ONE API `openalyx.internationalbrainlab.org` |
| IBL BWM on DANDI 000409 | Same data as NWB (for pynwb users / no ONE) | ~3 TB | Open | https://dandiarchive.org/dandiset/000409 |
| IBL BWM on AWS | S3 mirror `s3://ibl-brain-wide-map-public` (no-sign-request) | ~3 TB | Open | https://registry.opendata.aws/ibl-brain-wide-map/ |
| IBL repeated-site subset | Reproducibility positive control (10 labs, 1 trajectory) | ~80 insertions | Open | same ONE server; `brainwidemap`/`reproducible_ephys` code |
| Allen Visual Coding – Neuropixels | 58 sessions × 6 probes; units with `ecephys_structure_acronym`; stimulus tables (gratings, flashes, natural scenes); passive viewing | ~1.5 TB NWB; ~2 GB per session | Open | DANDI 000021; `allensdk` EcephysProjectCache; `s3://allen-brain-observatory/visual-coding-neuropixels/` |
| Allen CCF 2017 / Beryl parcellation | Region assignment and hierarchy | small | Open | `pip install iblatlas` ; `BrainRegions().acronym2acronym(..., mapping='Beryl')` |
| IBL behaviour-only sessions (optional) | Extra trials for the imposter-session null (Findling et al. 2025) | thousands of sessions | Open | ONE API |

## Methods

1. **Loading & region assignment** (`ibl_bwm.loaders`) — `ONE` client on the public Alyx; the BWM
   session/insertion table via `brainwidemap.bwm_query` (columns include `eid, pid, subject, lab,
   probe_name`); `SpikeSortingLoader` to get spikes/clusters/channels; map channel CCF acronyms to
   Beryl regions with `iblatlas`; keep units with IBL `label == 1` (good) plus a relaxed QC set for the
   yield-quality analysis; NWB/DANDI path with `pynwb` for the Allen data.
2. **Trial-aligned features** (`ibl_bwm.binning`) — bin spikes in [−0.1, +0.4] s around stimulus
   onset for stimulus decoding, [−0.1, +0.1] s around first movement for choice, the 0.4 s before
   stimulus for the block prior, and [0, 0.2] s after feedback for reward; 20 ms bins; per-unit
   √-transform; region × trial × time tensors.
3. **Decoders** (`ibl_bwm.decoding`) — L2-regularised logistic regression (choice, stimulus side,
   reward) and ridge (contrast, probabilityLeft) with 5-fold CV nested over the penalty, trained per
   (session, region), score = balanced accuracy or R². Nulls: (a) trial-label shuffle, (b) IBL
   pseudo-sessions generated from the block-structure generative process (probabilityLeft ∈ {0.2, 0.8},
   block lengths ~ truncated exponential 20–100 trials), (c) imposter sessions using other sessions'
   behaviour. Report null-corrected score and one-sided p.
4. **Hierarchical meta-analysis** (`ibl_bwm.hierarchical`) — DerSimonian–Laird random-effects
   per region; `statsmodels` MixedLM with lab as group and subject as variance component; ICCs with
   bootstrap CIs; leave-one-lab-out fragility flag; yield-matched subsampling.
5. **Latents** — `sklearn.decomposition.FactorAnalysis` on binned counts per session as a GPFA-lite
   (smoothed factors), decode from latents; optional LFADS via `lfads-torch` for a subset.
6. **Cross-dataset** — harmonise to "stimulus present vs. absent" and "contrast/spatial-frequency
   level" targets over the first 250 ms after onset; train/test both ways; region ranking correlation.
7. **Baselines** — IBL's published per-region decoding results (paper-brain-wide-map repo) as the
   reference against which fragility is judged.

## Evaluation & statistics

- Unit of analysis: (insertion, region) with ≥ 10 good units and ≥ 150 trials; regions with ≥ 3
  insertions from ≥ 2 labs enter the mixed model.
- Leakage prevention: CV folds are contiguous trial blocks (block prior is autocorrelated); the
  penalty is chosen inside the training folds; pseudo-sessions use the same fold structure.
- Multiple comparisons: BH-FDR over regions per target; Bonferroni across the 4 targets for headline
  counts.
- Effect sizes: null-corrected score (observed − pseudo-session mean) in SD units of the null; lab ICC;
  I² and τ² from the random-effects meta-analysis.
- Robustness: repeat with relaxed QC units; with 50/100 ms bins; with 3 alternative alignment windows;
  with imposter vs. pseudo-session nulls.
- Pre-registration of the region list, unit criteria and mixed-model formula on OSF before running.

## Publishable angle

Headline: "Brain-wide decoding claims are largely reproducible across labs, except for N regions
whose apparent encoding of choice/prior is driven by one lab or by unit yield" — with a public
per-region reproducibility table and a yield-corrected brain map. Complement: visual-region information
rankings that transfer between two protocols and consortia.

Venues: *eLife* (reproducibility series), *Nature Communications*, *PLOS Computational Biology*,
*Neurons, Behavior, Data analysis and Theory*; the benchmark component fits *NeurIPS Datasets &
Benchmarks*.

Follow-ups: extend to Steinmetz et al. (2019, *Nature*) recordings and to the IBL video/pose data;
per-lab histology-error simulation; a "reproducibility prior" for future multi-lab studies.

## Risks, confounds & mitigations

| Risk | Mitigation |
|---|---|
| Lab confounded with region coverage (labs targeted different areas) | Only regions recorded by ≥ 2 labs; include insertion-trajectory covariate; leave-one-lab-out |
| Lab confounded with time (data collected at different periods / spike-sorting versions) | Use the uniform 2025 re-sorted release; add sorting version as covariate if mixed |
| Behavioural differences across labs drive decodability | Include per-session psychometric slope and reaction time as covariates; decode within matched-performance sessions |
| Pseudo-session null mis-specified for choice (choice is not exchangeable) | Use imposter sessions for choice/reward; shuffle within block for stimulus |
| Allen passive viewing vs IBL task: different states | Restrict to stimulus-evoked window and to quiescent trials; report as protocol effect, not a failure |
| Region mis-assignment from histology error | Beryl-level regions; sensitivity with channels within 100 µm of a boundary removed |
| Computational budget | Cache binned tensors per session; run pseudo-sessions in parallel; start with 20 regions |

## Milestones

- [ ] `scripts/download_data.py --sample`: pull the BWM session table with lab metadata and 3 sessions; 1 Allen session.
- [ ] Binning + region assignment for all BWM insertions; cache per session (parquet/npz).
- [ ] Decoders + pseudo-session nulls for 4 targets, all (insertion, region) pairs.
- [ ] Mixed models / ICC per region; leave-one-lab-out; fragility table (H1, H2, H5).
- [ ] Yield-matched subsampling (H3).
- [ ] Factor-analysis latents (H6).
- [ ] Allen harmonisation and cross-dataset ranking (H4).
- [ ] Pre-registered report → manuscript; release per-region table and code.

## Ethics / data-use notes

- All data are open animal-research datasets released under CC-BY (IBL) / Allen Institute terms of
  use; cite the IBL 2025 Nature papers and Siegle et al. 2021. No new animal work is involved.
- Do not redistribute the IBL data; link to the ONE/DANDI sources. Never commit downloaded data
  (`data/` is git-ignored).
- Report per-lab results in aggregate or with lab codes consistent with the IBL's own publications;
  the aim is methodological, not to rank laboratories.
