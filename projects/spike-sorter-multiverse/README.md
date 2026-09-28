# Spike-sorter multiverse: how much of a Neuropixels "finding" is the sorter?

**One-sentence pitch.** Re-sort the same open raw Neuropixels AP-band recordings (Allen Visual Coding raw `spike_band.dat` on S3, IBL raw `.ap.cbin`, DANDI:000034) with Kilosort 2.5 / 3 / 4, SpyKING CIRCUS 2, MountainSort 5 and a SpikeInterface consensus, cross the sorter with the curation/QC rule, and measure how much the *scientific conclusions* drawn from those recordings (fraction of visually responsive units, orientation/direction selectivity distributions, within-session representational drift, noise-correlation strength, waveform-based cell-type proportions) move across the multiverse, relative to how much they move across mice.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: PhD-scale computationally, MSc-scale conceptually. 9-12 months. The bottleneck is GPU time for sorting, not analysis.
- Compute: a CUDA GPU (>= 16 GB) for Kilosort 2.5/3/4; each 1 h of 384-channel data takes ~0.5-2 GPU-hours per sorter. A 40-min segment from 12 probes x 5 sorters is ~60-120 GPU-hours. Storage: each Allen probe's full-session `spike_band.dat` is ~197 GB (verified on S3), but only time windows are needed (HTTP byte-range reads; 40 min = ~55 GB per probe). Plan 2-4 TB of scratch.
- Related projects in this repo: `neuropixels-representational-drift` (drift metrics on the Allen processed units - this project asks whether those metrics depend on the sorter) and `ibl-brainwide-decoding` (IBL data access). This project is self-contained.

## Background

Every Neuropixels dataset released with "units" has been through one spike sorter, one drift-correction setting and one curation rule; downstream analyses inherit those choices silently. Sorters disagree substantially on real data: Buccino et al. (2020, *eLife*; SpikeInterface) found that only a minority of units were found by all sorters on the same dense-probe recordings, and SpikeForest (Magland et al., 2020, *eLife*) showed accuracy varies by sorter and recording type even with ground truth. Kilosort4 (Pachitariu et al., 2024, *Nat Methods*) improved on Kilosort 1-3 in simulations; drift correction interacts with sorting (Steinmetz et al., 2021, *Science*; Garcia et al., 2024, *eNeuro*). A detailed biophysical simulation of a cortical column (bioRxiv 2024, doi 10.1101/2024.12.04.626805) reports that sorters preferentially retain high-firing neurons and that mean firing rates per layer are over-estimated several-fold relative to ground truth. Yet the claims that matter to neuroscience - what fraction of neurons in area X are tuned, how stable representations are across time, how strong noise correlations are - are almost never reported with a sorter sensitivity analysis.

The Allen Institute released the raw AP band of the Visual Coding Neuropixels survey (Siegle et al., 2021, *Nature*; 58 sessions, ~6 probes each, sorted with Kilosort 2 at the time), and the IBL brain-wide map (DANDI:000409; International Brain Laboratory, 2023, bioRxiv) provides raw data with its own sorter. These make a *real-data* multiverse possible for the first time at scale.

## The research gap

**What has been done.**

- Sorter agreement and ground-truth benchmarks: SpikeInterface (Buccino et al., 2020) with a consensus method; SpikeForest (Magland et al., 2020); Kilosort4 benchmarks (Pachitariu et al., 2024); drift benchmark (Garcia et al., 2024); the DANDI:000034 recordings used in the SpikeInterface paper are public (mouse, 6 files, ~74 GB).
- Reproducible pipelines: the Allen Institute for Neural Dynamics pipeline (bioRxiv 2025, doi 10.1101/2025.11.12.687966; eLife reviewed preprint) containerises sorters and includes a benchmarking module; SpikeInterface ships container images per sorter version.
- Cross-lab reproducibility with one sorter: the IBL reproducibility study (International Brain Laboratory et al., 2022, bioRxiv; eLife reviewed preprint) held the sorter fixed and varied labs.
- Simulated bias: the cortical-column simulation above quantifies selection and assignment biases against ground truth but cannot speak to real tuning or drift.
- Representational drift with fixed sorting: e.g., Deitch, Rubin & Ziv (2021, *Curr Biol*) on Allen 2-photon; Marks & Goard (2021, *Nat Commun*); Schoonover et al. (2021, *Nature*) with tetrodes; Rule, O'Leary & Harvey (2019, *Curr Opin Neurobiol*) review causes; none varies the sorter.

**What is specifically missing.**

1. **A sorter x curation multiverse on real recordings with downstream *claims* as the outcome.** Agreement scores tell us units differ; they do not tell us whether the OSI distribution of V1, the fraction of responsive LGN units, or the drift index changes enough to alter a conclusion.
2. **Variance decomposition: sorter vs curation vs mouse.** Whether the between-sorter variance of a claim is smaller or larger than the between-animal variance is the number a reviewer needs; it has not been reported.
3. **Which units carry the disagreement.** Units matched across all sorters vs "orphans" found by one sorter: if orphans are systematically low-amplitude, low-rate, or drifting units, then sorter choice acts as a hidden selection filter on tuning and stability estimates. Consensus-only analyses (as in SpikeInterface) discard exactly the units that differ.
4. **Drift as a sorter artefact.** Within-session representational drift (natural-movie responses repeated ~1-2 h apart in Visual Coding) could partly reflect sorter-specific drift correction failures (unit splitting/merging over time). Sorter-dependence of the drift index, and its relation to per-unit presence ratio and amplitude drift, is untested.
5. **Curation rules interact with sorters.** Allen's default QC (ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9) was tuned on Kilosort 2 output; applying the same thresholds to Kilosort4 or SpyKING CIRCUS 2 output changes yields differently. The sorter x QC interaction has not been quantified.

## Research questions / hypotheses

1. **H1 (agreement).** On the same 40-min segments, pairwise unit agreement (SpikeInterface score >= 0.5) between Kilosort 2.5, 3, 4, SpyKING CIRCUS 2 and MountainSort 5 is 40-70% of the smaller unit count; consensus units (>= 3 sorters) are 30-50% of any single sorter's QC-passing units.
2. **H2 (selection).** Orphan units (found by only one sorter) have lower amplitude (median < 60 uV), lower firing rate and lower presence ratio than consensus units; excluding them raises the median OSI in V1 by > 0.05 and the fraction of responsive units by > 10 percentage points.
3. **H3 (claims move).** Across the sorter x QC grid, the between-specification standard deviation of (a) the fraction of visually responsive V1 units, (b) median OSI/DSI, (c) mean noise correlation, and (d) the within-session drift index is at least half the between-mouse standard deviation of the same quantity for at least two of the four claims; the sorter main effect explains more variance than QC for (a) and (d), QC more than sorter for (b).
4. **H4 (drift is partly sorter-specific).** The within-session drift index is largest for sorters without drift correction (Kilosort 2.5 with correction disabled, MountainSort 5) and smallest for Kilosort 4 / consensus; per-unit drift correlates with amplitude drift and presence ratio (Spearman rho > 0.3) more strongly for the former.
5. **H5 (cell-type proportions).** The fraction of narrow-spiking (putative fast-spiking) units per area varies by < 5 percentage points across sorters after QC, i.e., waveform-based cell-type proportions are sorter-robust, whereas responsiveness fractions are not.
6. **H6 (ground-truth anchor).** On DANDI:000034 and hybrid ground-truth recordings (real data with injected templates via SpikeInterface's hybrid generator), the sorters that retain the most low-amplitude units also inflate estimated responsiveness the least, tying H2 to a known bias direction.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Visual Coding Neuropixels raw data (S3 `allen-brain-observatory`, prefix `visual-coding-neuropixels/raw-data/<session>/<probe>/`) | `spike_band.dat` (int16, 384 ch, 30 kHz; ~197 GB/probe/session), `lfp_band.dat`, `channel_states.npy`, `event_timestamps.npy`; select V1/LM/LGN probes from 8-12 `brain_observatory_1.1` sessions | ~2-3 TB for 12 probes (windows only: ~0.7 TB) | Open (AWS Open Data, no credentials) | https://registry.opendata.aws/allen-brain-observatory/ |
| Allen Visual Coding Neuropixels processed cache (S3, prefix `visual-coding-neuropixels/ecephys-cache/`) | `sessions.csv`, `probes.csv`, `channels.csv`, `units.csv`, per-session NWB (stimulus tables, original Kilosort 2 units and QC metrics) - the "reference universe" | ~100 GB for all sessions; CSVs < 200 MB | Open | same bucket; `allensdk` EcephysProjectCache |
| DANDI:000034 - recordings from Buccino et al. 2020 (SpikeInterface) | Raw mouse recordings with existing multi-sorter agreement analysis; anchor for H1/H6 | 6 files, ~74 GB | Open | https://dandiarchive.org/dandiset/000034 |
| IBL Brain-wide Map (DANDI:000409; raw `.ap.cbin` via IBL ONE/AWS) | Raw AP data for probes in visual cortex during the IBL task; second lab/rig for generalisation | 2,048 NWB, ~50 TB (use a handful of probes) | Open (DANDI; raw via `ONE` public credentials) | https://dandiarchive.org/dandiset/000409 ; https://int-brain-lab.github.io/ONE/ |
| Sorter containers | Kilosort 2.5/3 (MATLAB, via SpikeInterface docker images), Kilosort 4 (Python), SpyKING CIRCUS 2, MountainSort 5, Tridesclous 2 | - | Open source | https://spikeinterface.readthedocs.io/ |

## Methods

1. **Segments (`sortverse.raw_io`).** For each selected probe, read a fixed window (e.g., 40 min spanning the drifting-gratings block and the first and last natural-movie-1 repeats) directly from S3 with HTTP byte ranges (`byte_range_for_window`), and write a local int16 binary plus a Neuropixels 1.0 geometry file (`neuropixels1_geometry`). Same windows for every sorter.
2. **Sorting.** SpikeInterface >= 0.101 with pinned container images: Kilosort 2.5 (with and without drift correction), Kilosort 3, Kilosort 4 (default), SpyKING CIRCUS 2, MountainSort 5. Common preprocessing (bandpass 300-6000 Hz, common median reference, bad-channel removal) applied *before* every sorter so the multiverse isolates the sorting step. Waveforms/templates and amplitudes extracted with SpikeInterface for all outputs.
3. **Matching and consensus (`sortverse.agreement`).** Pairwise agreement scores (coincident spikes within 0.4 ms) with Hungarian assignment; consensus units by >= k sorters; orphan labelling; QC metrics (ISI violation ratio, presence ratio, amplitude cutoff; Hill, Mehta & Kleinfeld, 2011, *J Neurosci*) computed identically for every sorter; two QC rules (Allen default vs lenient).
4. **Downstream claims (`sortverse.downstream`).** Per unit: responsiveness (permutation test on evoked vs baseline counts), OSI/DSI from drifting gratings (vector method; Mazurek, Kager & Van Hooser, 2014, *Front Neural Circuits*), waveform duration (narrow vs broad), noise correlations; per population: fraction responsive, median OSI, mean noise correlation, within-session representational drift (population-vector similarity between early and late natural-movie repeats, relative to within-block split-half reliability), per-unit amplitude drift.
5. **Multiverse (`sortverse.multiverse`).** Specification grid sorter x QC-rule x drift-correction x unit-set (all / consensus / orphans); variance decomposition (eta-squared from a factorial ANOVA with mouse as a factor), specification curves, claim-stability bootstrap.
6. **Ground-truth anchors.** DANDI:000034 recordings and SpikeInterface hybrid recordings (templates injected into real background) to relate agreement patterns to known accuracy.

## Evaluation & statistics

- Unit of analysis for claims: probe-area-segment (e.g., V1 on probe C in session S). Mixed-effects models with mouse as random effect; sorter and QC rule as fixed factors; report eta-squared with bootstrap CIs (resampling sessions).
- Agreement: pairwise agreement matrices per probe; consensus counts; distributions of amplitude/rate/presence for consensus vs orphan units (Mann-Whitney with cluster bootstrap over probes).
- Drift: drift index with a within-block split-half reference (drift is meaningful only if between-block similarity is below within-block reliability); per-unit drift vs amplitude drift by Spearman with probe-level bootstrap.
- Nulls: (a) spike-time jitter null for agreement (jittering by 5 ms should destroy matches); (b) label-shuffle null for OSI/responsiveness; (c) *sorter-swap* null: assign units randomly to "sorters" to calibrate the expected between-sorter variance from sampling alone.
- Multiple comparisons: H1-H6 pre-registered; Holm within family; claims reported as effect sizes with CIs rather than p-values.
- Leakage/reproducibility: identical raw windows and preprocessing for all sorters; container digests, sorter versions and random seeds recorded in `outputs/manifest.json`; no manual curation.

## Publishable angle

- **Headline result.** "On identical Neuropixels recordings, the choice of spike sorter moves the fraction of visually responsive V1 units by X points and the within-session drift index by Y, comparable to the between-animal spread; orphan units are low-amplitude and less tuned, so sorter choice acts as a selection filter. Waveform-based cell-type proportions are robust; drift and responsiveness are not." Delivered with a reusable multiverse harness that any lab can run on its own recordings.
- **Venues.** *eLife* or *Journal of Neuroscience* (methods-heavy systems neuroscience); *Nature Methods* / *Nature Communications* brief communication if the effect sizes are large; *Journal of Neural Engineering* or *eNeuro* for the harness; *NeurIPS Datasets & Benchmarks* for a released multi-sorter unit catalogue.
- **Follow-ups.** (i) Extend to across-day drift with chronic Neuropixels 2.0 datasets; (ii) sorter-robust estimators (analyses on consensus units with orphan-aware reweighting); (iii) the same design for calcium-imaging segmentation (Suite2p vs CaImAn) - the imaging analogue.

## Risks, confounds & mitigations

- **GPU cost and sorter failures.** Mitigation: fixed 40-min windows; run sorters in containers; treat crashes as data (report per-sorter failure rate).
- **MATLAB dependency for Kilosort 2.5/3.** Mitigation: SpikeInterface docker images with MATLAB runtime; if unavailable, use pykilosort 2.5 and note the substitution.
- **Preprocessing differences hidden inside sorters** (Kilosort's own filtering/whitening). Mitigation: pass pre-filtered data and disable internal filtering where possible; report both.
- **Allen's reference universe was sorted with an older Kilosort 2 on full sessions** - windowed re-sorting is not identical. Mitigation: include the released units as a specification but do not treat them as ground truth; compare on the same windows.
- **Drift correction confounded with sorter.** Mitigation: run Kilosort 2.5 with correction on/off and Kilosort 4 with `nblocks=0` vs default to separate the factor.
- **Small number of mice.** Between-mouse variance estimated from 8-12 sessions is noisy; report CIs and add IBL probes as a second cohort.

## Milestones

- [ ] Build the S3 manifest of raw probes (`scripts/download_data.py --list`) and select 12 probes (V1/LM/LGN) from `brain_observatory_1.1` sessions.
- [ ] Byte-range extraction of fixed windows; geometry files; `--sample` end-to-end test on a 2-s slice.
- [ ] SpikeInterface pipeline with pinned containers; run all sorters on one probe; agreement matrices.
- [ ] QC metrics and consensus/orphan labelling for all outputs.
- [ ] Downstream claims per specification; variance decomposition and specification curves.
- [ ] Drift analysis with drift-correction factor (H4).
- [ ] Ground-truth anchors (DANDI:000034, hybrid recordings) (H6).
- [ ] IBL replication on 4-6 probes.
- [ ] Pre-registration after the single-probe pilot; manuscript; release harness + unit catalogue.

## Ethics / data-use notes

- Allen Brain Observatory data are released under the Allen Institute Terms of Use (non-commercial research; cite Siegle et al., 2021, and the dataset). IBL and DANDI data are CC-BY; cite the dandisets and papers. No new animal experiments.
- Raw recordings are large; do not commit any data or sorter outputs (`.gitignore`); commit manifests, container digests and aggregate tables only.
