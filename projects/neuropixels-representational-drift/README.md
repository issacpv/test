# Representational drift across cortex, thalamus and hippocampus: a cross-dataset Neuropixels benchmark with unit-quality confounds modeled

**One-sentence pitch.** Use the Allen Visual Coding and Visual Behavior Neuropixels datasets and the IBL Brain-wide Map to ask whether representational drift of visual and decision codes differs systematically between cortex, thalamus and hippocampus, whether population readouts stay stable while single units drift, and how much of the apparent drift is explained by spike-sorting/unit-tracking quality rather than biology.

## Status / difficulty / timeline / compute

- Status: design + starter code; no data committed.
- Difficulty: MSc-to-PhD; 6-9 months for the within-session/cross-area paper, 12+ months to add the IBL decision-code arm and cross-session geometry.
- Compute: NWB streaming from DANDI/S3 means no bulk download is required; a laptop handles single sessions, a 32-64 GB RAM workstation handles pooled analyses over ~200 sessions. No GPU needed (linear decoders).

## Background

Neural responses to identical stimuli change over minutes to weeks even when behavior is stable ("representational drift"): in hippocampus (Ziv et al., 2013, *Nat. Neurosci.*), posterior parietal cortex (Driscoll et al., 2017, *Cell*), piriform cortex (Schoonover et al., 2021, *Nature*) and visual cortex (Deitch, Rubin & Ziv, 2021, *Curr. Biol.*; Marks & Goard, 2021, *Nat. Commun.*). Theory proposes drift as a consequence of ongoing plasticity with implicit regularization (Micou & O'Leary, 2023, *Curr. Opin. Neurobiol.*; Masset, Qin & Zavatone-Veth, 2022, *Biol. Cybern.*), and that it should be smaller in circuits close to sensory input or motor output. Whether drift is a cortical peculiarity or a brain-wide property, and whether downstream readouts are shielded from it (Rule et al., 2020, *eLife*; Gallego et al., 2020, *Nat. Neurosci.*), remains contested.

The Allen Neuropixels datasets record simultaneously from visual cortex, thalamus (LGd, LP) and hippocampal formation (CA1, CA3, DG, SUB) in the same mice while identical stimuli are repeated across the session (Siegle et al., 2021, *Nature*), and the IBL Brain-wide Map (International Brain Laboratory, 2025, *Nature*) covers most of the brain during a standardized decision task across 12 labs. These are the right resources for a *cross-area, cross-dataset* drift benchmark, and because they were spike-sorted with documented quality metrics, unit-quality confounds can be modeled explicitly rather than ignored.

## The research gap

**What has been done.**

- Deitch, Rubin & Ziv (2021, *Curr. Biol.*) quantified drift in the Allen Brain Observatory (mainly two-photon; Neuropixels used for within-session analyses) across visual areas and reported that stability did not follow the cortical hierarchy; Aitken, Garrett, Olsen & Mihalas (2022, *PLoS Comput. Biol.*) described the geometry of drift in the Visual Behavior two-photon data and in ANNs.
- Marks & Goard (2021) showed stimulus-dependent drift in V1 (natural movies drift, gratings less so); Sadeh & Clopath (2022, *eLife*) showed behavioral variability can masquerade as drift.
- Chronic Neuropixels tracking (Steinmetz et al., 2021, *Science*) and UnitMatch (van Beest et al., 2024, *Nat. Methods*) established that visual-cortex tuning is largely stable over weeks when units are tracked by waveform, whereas striatal responses change during learning; "Hippocampal representations drift in stable multisensory environments" (*Nature*, 2025) confirmed hippocampal drift with chronic probes; Khatib et al. (2023, *Neuron*) showed within-day CA1 drift scales with active experience.
- "Temporal coding carries more stable cortical visual representations" (*Nat. Commun.*, 2025) showed that fine-timescale codes drift less than rate codes in cortex.
- The IBL papers (2025, *Nature*) document that slow probe drift induces spurious trial-to-trial correlations and provide brain-wide decoding of task variables, but do not analyze representational drift per se.

**What is specifically missing.**

1. **No cross-area comparison with subcortical structures in the same animals under matched stimuli and matched unit quality.** The thalamus (LGd/LP) and hippocampal formation are recorded in the same Allen sessions as V1/LM/AL/PM/AM/RL, but drift has only been reported for cortex (and LGN in passing). The hypothesis "drift is lower closer to sensory input (thalamus) and higher in hippocampus" is untested with simultaneous recordings.
2. **Unit-quality confounds are not modeled.** Within-session drift estimates depend on presence ratio, amplitude cutoff, ISI violations and probe motion (`max_drift`, `cumulative_drift` in Allen metrics; `slidingRP`, `noise_cutoff` in IBL). Areas differ in these metrics (hippocampal pyramidal units have larger amplitudes and better isolation than thalamic units), so any area difference in drift is confounded unless quality is stratified or entered as a covariate. No published drift comparison does this.
3. **Population-readout stability vs single-unit drift across areas.** Cross-time decoder generalization (train on early repeats, test on late) relative to within-time accuracy has been reported for cortex and PPC, not compared across cortex/thalamus/hippocampus, and not for the IBL decision code (stimulus side, choice, prior block) across ~200 regions.
4. **No cross-dataset replication.** Visual Coding (passive), Visual Behavior (active change detection, consecutive-day sessions) and IBL (decision task) differ in behavior, sorter and lab; a drift benchmark that reports the same metric across all three, with the same nulls, does not exist.
5. **Rate-matched nulls.** Much of the "drift" measured with correlation-based metrics is the expected decline from Poisson variability when firing rates are low; per-unit rate-matched surrogates and time-shuffle nulls must be reported alongside the observed drift, which is rarely done.

## Research questions / hypotheses

1. **H1 (area ordering).** Within-session drift of natural-movie representations (decline of population-vector correlation between repeats with time lag, corrected by the time-shuffle null) is ordered thalamus (LGd, LP) < visual cortex < hippocampal formation (CA1/DG/SUB), in the same sessions.
2. **H2 (quality confound).** At least 30% of the raw area difference in drift disappears after (a) restricting to units passing Allen default quality criteria and (b) stratifying by presence ratio and amplitude; the residual difference remains significant in a mixed model with session as random effect.
3. **H3 (readout stability).** Cross-time decoding (movie-frame or image identity) trained on the first third of repeats and tested on the last third loses < 15% relative accuracy in cortex and thalamus but > 25% in hippocampus, and the population decoder loses less than the median single-unit tuning correlation would predict.
4. **H4 (stimulus dependence).** Drift is larger for natural movies than for drifting gratings in cortex (replicating Marks & Goard) and this stimulus dependence is absent in thalamus.
5. **H5 (IBL decision code).** Cross-time generalization of stimulus-side and choice decoders (early vs late trials within session) is region-dependent, with sensory thalamus and superior colliculus more stable than frontal cortex and hippocampus, after controlling for probe drift metrics.
6. **H6 (cross-session geometry).** RDM similarity between sessions of different mice (same stimuli) is higher for thalamus than for hippocampus, i.e., subcortical codes are more stereotyped across animals as well as more stable within animals.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Visual Coding - Neuropixels | 58 sessions (Brain Observatory 1.1 and Functional Connectivity stimulus sets), ~100k units across 6 visual areas + LGd/LP + hippocampal formation; natural movies, drifting/static gratings, repeats across the session | ~2-3 GB per session NWB (~150 GB total) | Open (DANDI, AWS S3, AllenSDK) | https://dandiarchive.org/dandiset/000021 , https://dandiarchive.org/dandiset/000022 , https://allensdk.readthedocs.io |
| Allen Visual Behavior - Neuropixels | ~150 sessions (2 consecutive days per mouse: familiar vs novel image sets), change-detection task + passive replay; V1/LM/AL/PM/AM/RL + LGd/LP + CA1/CA3/DG/SUB | ~2-4 GB per session | Open (AWS S3 via AllenSDK `VisualBehaviorNeuropixelsProjectCache`) | https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html |
| IBL Brain-wide Map (2025 release) | 459 sessions, 699 insertions, 139 mice, 12 labs; ~76k good units in ~280 regions; trials with stimulus side, contrast, choice, block prior | ~ TB scale on S3; per-session streaming | Open (ONE API / AWS Open Data `ibl-brain-wide-map`; DANDI mirror) | https://registry.opendata.aws/ibl-brain-wide-map/ , https://docs.internationalbrainlab.org |
| Allen CCF region hierarchy | area -> group mapping (cortex / thalamus / hippocampal formation) | small | Open | AllenSDK / `ccf_structure_tree` |

## Methods

1. **Loading (`npx_drift.loaders`).** Stream NWB files from DANDI (`dandi` API -> S3 URL -> `remfile` + `h5py` + `pynwb`), or from the AllenSDK cache. Extract the units table with quality metrics, spike times, and all stimulus/trial interval tables into one `SessionData` object. IBL sessions are loaded via ONE (`spikes.times`, `clusters.metrics`, `trials`).
2. **Unit quality (`npx_drift.quality`).** Allen default criteria (`isi_violations < 0.5`, `amplitude_cutoff < 0.1`, `presence_ratio > 0.95`); IBL `label == 1`. Compute per-unit stability covariates over the session (firing-rate CV across 4 blocks, presence ratio in 60-s bins, amplitude slope) and map CCF acronyms to area groups.
3. **Response matrices (`npx_drift.responses`).** Bin spikes in stimulus-aligned windows: natural-movie frames (30 frames/s -> 33 ms bins, or 10 frames pooled), drifting-grating trials (2 s), Visual Behavior image flashes (250 ms), IBL trials (stimulus-locked 0-200 ms). Trials are labeled with condition (frame/orientation/image/side) and their absolute time, then assigned to time blocks (early/middle/late) or to repeat index.
4. **Drift metrics (`npx_drift.drift_metrics`).** (a) Population-vector correlation between condition-averaged responses in blocks *i* and *j* as a function of |i-j| (slope = drift rate); (b) RDM stability (Spearman between block RDMs); (c) per-unit tuning-curve correlation early vs late; (d) cross-time decoder generalization: multinomial logistic regression trained on block *i*, tested on block *j*, expressed relative to within-block cross-validated accuracy (`drift index = 1 - acc_cross / acc_within`); (e) a "readout shielding" score = single-unit drift index minus population drift index.
5. **Nulls (`npx_drift.nulls`).** Time-shuffle (permute trial-to-block assignment; destroys temporal structure, keeps tuning and rate), Poisson rate-matched surrogates (no drift by construction; quantifies the correlation decline expected from finite counts), and circular spike-train shifts (destroys stimulus locking, keeps autocorrelation). Report observed minus null-mean and permutation p-values.
6. **Confound model.** Mixed-effects regression: `drift ~ area_group * stimulus + presence_ratio + amplitude_cutoff + isi_violations + log(firing_rate) + max_drift + (1 | session) + (1 | mouse)`; alternatively, quality-matched resampling of units across areas (propensity-style matching on quality metrics) before computing area contrasts.
7. **Cross-dataset harmonization.** Same metrics and nulls for all three datasets; stimuli mapped to a common "condition" abstraction; sorter differences (Kilosort 2 vs 2.5 vs pyKilosort) recorded as a covariate.

## Evaluation & statistics

- Unit of analysis: session x area group (not unit) for area contrasts; units nested in sessions nested in mice; mixed models (statsmodels `MixedLM` or `pymer4`) with random intercepts for session and mouse.
- Effect sizes with 95% CIs from session-level bootstrap; permutation tests against the time-shuffle null for every drift statistic; Poisson-surrogate-corrected drift reported alongside raw drift.
- Multiple comparisons: Holm correction across area-pair contrasts within each metric; H1-H6 pre-registered.
- Decoder evaluation: stratified k-fold within block for `acc_within`; equal trial counts across blocks; class-balanced accuracy; chance level from label shuffles.
- Leakage prevention: decoders never see test-block trials during training or hyperparameter choice (regularization fixed a priori, C = 1 with standardization); cross-session analyses use different mice, so no unit-level leakage is possible.
- Robustness: repeat with only high-quality units, with quality-matched units, with firing-rate-matched units, and with 3 vs 5 time blocks.

## Publishable angle

- **Headline.** "Within-session representational drift is ordered thalamus < cortex < hippocampus in simultaneously recorded populations; about a third of the naive area difference is a unit-quality artefact; and linear population readouts are shielded from single-unit drift in cortex and thalamus but not in hippocampus." Replicated in passive (Visual Coding), active (Visual Behavior) and decision (IBL) datasets with shared code and nulls.
- **Venues.** *Nature Communications* / *eLife* (systems neuroscience with methods emphasis); *PLoS Computational Biology* (benchmark + confound modeling); *Journal of Neuroscience*; *NeurIPS Datasets & Benchmarks* for the benchmark release.
- **Follow-ups.** Temporal-code vs rate-code drift across areas; drift vs behavioral state (pupil, running) using the Allen running/pupil signals; UnitMatch-tracked cross-day drift in the IBL repeated-site and chronic datasets; linking drift to layer/cell-type (waveform duration, optotagging in Visual Behavior).

## Risks, confounds & mitigations

- **Probe motion mimics drift.** Mitigation: `max_drift`/`cumulative_drift` covariates; exclude sessions with large motion; compare drift computed on amplitude-stable units only.
- **Different insertions on consecutive days (Visual Behavior)** prevent unit tracking. Mitigation: cross-session claims are population-geometry claims only; state this explicitly.
- **Low firing rates in thalamus/hippocampus inflate correlation-based drift.** Mitigation: Poisson rate-matched nulls; rate-matched unit subsampling.
- **Behavioral state changes (arousal, running) across the session.** Mitigation: include running speed and pupil diameter as trial covariates; repeat analyses on state-matched trials.
- **Stimulus adaptation vs drift.** Mitigation: separate monotonic rate decline (adaptation) from tuning change (drift) by analyzing normalized tuning curves and by using blocks separated by other stimuli.
- **IBL heterogeneity (labs, sorters).** Mitigation: lab as random effect; use the repeated-site subset for the cleanest cross-lab replication.
- **Streaming performance from DANDI.** Mitigation: cache units tables and binned responses locally (`data/cache/*.npz`); `--sample` mode in the downloader.

## Milestones

- [ ] `scripts/download_data.py --sample` works: list dandisets, stream one Visual Coding session, cache units + stimulus tables.
- [ ] Area-group mapping validated against the CCF structure tree; per-session unit counts per group table.
- [ ] Within-session natural-movie drift (PV correlation vs lag) for all Visual Coding sessions; time-shuffle + Poisson nulls.
- [ ] Quality-stratified and mixed-model area contrasts (H1, H2).
- [ ] Cross-time decoders per area group (H3); stimulus dependence (H4).
- [ ] Visual Behavior replication (image identity, active vs passive blocks).
- [ ] IBL decision-code arm: stimulus-side / choice decoders early vs late per region (H5).
- [ ] Cross-session RDM geometry across mice (H6).
- [ ] Benchmark release: metrics, nulls, per-session results tables, figures; manuscript.

## Ethics / data-use notes

- All three datasets are openly licensed animal-research data (Allen Institute terms of use; IBL CC-BY). Cite the dataset papers and the DANDI/ONE identifiers used.
- No data are committed; `data/` and `*.nwb` are git-ignored. Cached derived tables (unit metrics, binned counts) live under `data/cache/` and are also ignored.
- Report sorter versions and quality thresholds exactly; they change the answer.
