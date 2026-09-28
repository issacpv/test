# Cross-species spindle / slow-oscillation coupling: separating species from recording scale with one harmonised pipeline

**One-sentence pitch.** Apply a single, scale-free slow-oscillation (SO) / spindle detection and coupling pipeline to open rodent intracortical LFP (DANDI), human intracranial EEG (MNI Open iEEG Atlas) and human scalp PSG (Sleep-EDF, NSRR), to determine which of the canonical cross-species claims about SO-spindle coupling (preferred SO phase near the up-state, coupling strength, spindle-frequency scaling) survive when detector settings are held constant, and how much of the reported species difference is actually a recording-scale (depth LFP vs scalp) and detector-convention difference.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc/early-PhD; 6-9 months. Mostly signal processing and careful statistics; no training of models.
- Compute: CPU only. The rodent NWB files are large (DANDI:000041 ~155 GB; DANDI:000978 ~323 GB; DANDI:000166 ~787 GB) but only a few LFP channels per session are needed, so stream/slice with `pynwb`/`h5py` over `fsspec` rather than downloading whole files. Human PSG: Sleep-EDF ~8 GB; NSRR cohorts per DUA. Storage: 0.5-1 TB if sessions are cached locally.
- Related project in this repo: `sleep-spindle-aging-biomarker` (human spindle metrics vs age in NSRR). This project is self-contained.

## Background

SO-spindle coupling is the central mechanistic claim of the active-systems-consolidation model: spindles nested in the SO up-state (and ripples nested in spindle troughs) time hippocampal-cortical information transfer (Siapas & Wilson, 1998, *Neuron*; Mölle et al., 2002, *J Neurosci*; Staresina et al., 2015, *Nat Neurosci*; Maingret et al., 2016, *Nat Neurosci*; Latchoumane et al., 2017, *Neuron*). The rodent-to-human translation is explicit in the literature (Mölle et al., 2009, *Eur J Neurosci*, compared SO-spindle grouping in rats and humans; Helfrich et al., 2018, *Neuron*, and Muehlroth et al., 2019, *Sci Rep*, showed that the *precision* of coupling degrades with ageing and predicts memory in humans; Hahn et al., 2020, *eLife*, in development). Yet spindle physiology differs across species in ways that interact with detection: rodent spindles are shorter and faster (often 10-20 Hz in mice, ~10-16 Hz in rats) than human spindles (11-16 Hz, with slow/fast subtypes), rodent SOs are typically detected in the 0.5-4 Hz "delta" range while human SO detectors use 0.16-1.25 Hz, and rodent signals are depth LFPs whereas human evidence is mostly scalp EEG (Fernandez & Lüthi, 2020, *Physiol Rev*). Spindle detector agreement is only moderate even within humans (Warby et al., 2014, *Nat Methods*; Lacourse et al., 2019, *J Neurosci Methods*), and coupling metrics depend on the SO band used to define phase.

## The research gap

**What has been done.**

- Rodent SO-spindle-ripple coupling on open data: Watson et al. (2016, *Neuron*) recorded rat frontal cortex across sleep with silicon probes (now DANDI:000041) and Levenstein, Buzsáki & Rinzel (2019, *Nat Commun*) re-analysed NREM dynamics on those data; Kim, Gulati & Ganguly (2019, *Cell*) distinguished SOs from delta waves by their opposite effects on consolidation in rats.
- Human intracranial coupling: Staresina et al. (2015); Andrillon et al. (2011, *J Neurosci*) and Nir et al. (2011, *Neuron*) characterised local spindles and slow waves in human depth recordings; a normative sleep iEEG atlas exists (Frauscher et al., 2018, *Brain*; von Ellenrieder et al., 2020, *Ann Neurol*).
- Human scalp coupling and ageing/development: Helfrich et al. (2018), Muehlroth et al. (2019), Hahn et al. (2020); phase-precession of coupling along the anterior-posterior axis was reported in 2025 (bioRxiv).
- Methods: aperiodic-corrected spectral parameterisation (Donoghue et al., 2020, *Nat Neurosci*) to individualise the spindle band; Tort's modulation index (Tort et al., 2010, *J Neurophysiol*); event-locked circular statistics.

**What is specifically missing.**

1. **No study has run the *same* detector with scale-free (percentile-based) thresholds and an *individualised* spindle band across rodent LFP, human iEEG and human scalp EEG.** Cross-species statements therefore mix species differences with detector-convention differences (fixed band edges, absolute microvolt thresholds, seconds-based duration limits).
2. **Species vs recording scale is confounded.** Rodent evidence is intracortical; human evidence is mostly scalp. Human iEEG sits between them. A three-way comparison (rodent depth LFP, human depth iEEG, human scalp) can attribute differences to species or to recording scale. This has not been done with open data.
3. **Scale-free coupling descriptors are missing.** Coupling offset is reported in milliseconds, although the SO period differs across species; spindle duration is reported in seconds, although spindle frequency differs. Reporting offsets in *SO cycles* and durations in *spindle cycles* is trivial but has not been systematically applied.
4. **Detector multiverse.** Whether the "preferred phase near the SO peak" result is robust to SO band (0.16-1.25 vs 0.3-2 vs 0.5-4 Hz), spindle band (fixed vs individualised), and threshold rule (SD- vs percentile-based) has not been quantified within one dataset, let alone across species.
5. **Layer/region dependence in rodents vs region dependence in human iEEG.** Whether coupling phase varies with cortical depth (available in DANDI:000166 mouse V1 laminar recordings and DANDI:000041 rat frontal probes) as it varies with region in human iEEG is untested.

## Research questions / hypotheses

1. **H1 (phase conservation).** With identical scale-free detection, the mean SO phase of spindle-power maxima is within +/- 30 degrees of the SO peak (up-state) in rat frontal cortex (DANDI:000041), rat PFC (DANDI:000978), mouse V1 (DANDI:000166), human frontal iEEG (MNI atlas) and human scalp (Sleep-EDF/NSRR); the between-species spread of mean phases is smaller than the between-detector spread within any one species.
2. **H2 (coupling strength scales with recording depth, not species).** The mean resultant length (MVL) of spindle-peak phases is higher in depth recordings (rodent LFP, human iEEG) than in human scalp EEG after event-count matching; rodent vs human iEEG differ by less than iEEG vs scalp.
3. **H3 (temporal scaling).** Spindle duration in cycles (5-15 cycles) and SO-spindle offset in SO cycles (spindle peak 0.1-0.3 cycles after the SO trough) are conserved across species, whereas the same quantities in seconds differ by the ratio of spindle/SO frequencies.
4. **H4 (detector multiverse).** Across a grid of 3 SO bands x 2 spindle-band rules x 2 threshold rules, the sign of the "spindles prefer the up-state" result is stable, but the estimated MVL varies by > 2x and the mean phase by > 45 degrees for at least one species; the largest single driver is the SO band.
5. **H5 (depth/region dependence).** In laminar rodent recordings, coupling phase shifts by more than 30 degrees between superficial and deep channels (consistent with the polarity reversal of the SO), and in human iEEG coupling phase differs between frontal and temporal/parietal contacts; both are removed by referencing phase to the local CSD or bipolar SO rather than a single channel.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| DANDI:000041 - Watson et al. 2016, "Network homeostasis and state dynamics of neocortical sleep" | Rat frontal cortex silicon-probe LFP during natural sleep, 11 rats, 22 sessions, sleep-state labels | 22 NWB, ~155 GB | Open (CC-BY) | https://dandiarchive.org/dandiset/000041 |
| DANDI:000978 - "Single Day W-Track Learning" (Shin; Frank lab) | Rat dorsal CA1 + prefrontal LFP and units with interleaved sleep epochs (8 rats, 17 epochs/animal) | 9 NWB, ~323 GB | Open | https://dandiarchive.org/dandiset/000978 |
| DANDI:000166 - Senzai, Fernandez-Ruiz & Buzsáki 2019, mouse V1 laminar | Mouse V1 across all layers (silicon probes) with CSD-defined laminar landmarks; sessions include natural sleep | 19 NWB, ~787 GB | Open | https://dandiarchive.org/dandiset/000166 |
| DANDI:000044 - Grosmark & Buzsáki 2016 | Rat CA1 with long pre/post-task rest (sleep) periods; ripple-spindle-SO triads | 8 NWB, ~66 GB | Open | https://dandiarchive.org/dandiset/000044 |
| MNI Open iEEG Atlas (Frauscher et al., 2018; von Ellenrieder et al., 2020) | Normal-region intracranial EEG segments in wake and sleep stages (NREM, REM) from >100 epilepsy patients, with MNI coordinates and region labels | ~1-min segments per channel (GB scale) | Open after accepting the site's terms of use | https://mni-open-ieegatlas.research.mcgill.ca/ |
| Sleep-EDF Expanded (PhysioNet) | 197 human scalp PSGs (Fpz-Cz, Pz-Oz at 100 Hz) with hypnograms; healthy 25-101 y | ~8 GB | Open | https://physionet.org/content/sleep-edfx/1.0.0/ |
| NSRR CFS / MESA (optional) | Human scalp PSG at 128-256 Hz with AASM staging; larger age range and higher sampling rate than Sleep-EDF | ~200-600 GB | DUA (NSRR) | https://sleepdata.org/datasets/cfs ; https://sleepdata.org/datasets/mesa |
| MASS SS2 (optional) | Human PSG with expert spindle annotations for detector calibration | ~20 GB | Free registration + agreement | http://ceams-carsm.ca/mass/ |

## Methods

1. **Ingest.** LFP `ElectricalSeries` from NWB (`pynwb` or `h5py` + `fsspec` streaming) with electrode depth/region metadata; brain-state intervals from the NWB (where provided) or re-derived with a crude spectral-ratio scorer (`xspindle.harmonize.crude_nrem_mask`) for sessions without labels; human PSG via `mne`/`pyedflib` with hypnograms. All signals resampled to 200 Hz (`harmonize.resample_to`).
2. **Individualised bands (`xspindle.spectrum`).** NREM Welch spectrum -> aperiodic (1/f) fit in log-log -> residual -> spindle-band peak found in a wide species-agnostic search window (8-18 Hz); the SO band is fixed but varied in the multiverse. Recordings without a sigma peak (residual peak < 0.1 log10 units) are flagged and analysed separately.
3. **Detection (`xspindle.detect`).** SOs: zero-crossing cycles in the SO band with duration limits in seconds and a *percentile* peak-to-peak amplitude criterion (top 25% of candidate cycles), so no microvolt thresholds are needed. Spindles: Hilbert envelope in the individualised band (peak +/- 2 Hz), percentile threshold on the NREM envelope, duration limits expressed in *cycles* (5-30 cycles of the individual spindle frequency), event merging.
4. **Coupling (`xspindle.coupling`).** SO phase by Hilbert transform of the SO-band signal; each spindle peak within +/- 1 SO period of an SO trough contributes one phase; mean phase and MVL with Rayleigh test; an event-count-matched surrogate distribution (random NREM phases) gives a z-scored MVL; Tort modulation index between SO phase and spindle-band amplitude as a continuous alternative; offsets in SO cycles.
5. **Harmonised metrics and multiverse (`xspindle.harmonize`).** One function returns the full metric vector for a signal under a preset; presets encode `CONVENTIONAL` per-species settings from the literature and a single `HARMONIZED` setting; `detector_multiverse` runs the grid.
6. **Depth analysis.** For laminar probes, coupling is computed per channel against (a) a single reference channel's SO phase and (b) the local bipolar/CSD SO phase (H5).

## Evaluation & statistics

- Units of analysis: recording (session x channel-group). Species-level comparisons use mixed models with subject as a random effect (`statsmodels` MixedLM on circular quantities after unwrapping around the grand mean, and on log MVL).
- Circular statistics: Rayleigh test per recording; Watson-Williams / circular ANOVA (via bootstrap) for between-group differences in mean phase; MVL compared only after subsampling to a common event count (MVL is biased upward for small n).
- Nulls: (a) event-count-matched random-phase surrogates (per recording); (b) time-shifted spindle trains (circular shift by 5-20 s) to preserve autocorrelation; (c) *sham* coupling from wake segments where SO-spindle coupling should be absent.
- Multiverse: full factorial grid (3 SO bands x 2 spindle-band rules x 2 threshold rules x 2 duration rules = 24 specifications) per recording; report specification curves (Simonsohn, Simmons & Nelson, 2020, *Nat Hum Behav*) and variance components of MVL/phase attributable to each factor vs species.
- Multiple comparisons: hypotheses H1-H5 are pre-registered families; Holm within family.
- Reproducibility: fixed asset lists (dandiset version IDs), fixed channel selections written to `data/manifests/`, and pinned software.

## Publishable angle

- **Headline result.** "When detected identically, spindles couple to the SO up-state at the same phase in rats, mice and humans; the apparent species differences in coupling strength are explained by recording scale (depth vs scalp) and detector conventions, and scale-free descriptors (cycles rather than seconds) make the temporal organisation invariant across species." Plus a reusable harmonised pipeline and a detector multiverse that tells the field which settings drive the variability.
- **Venues.** *Sleep* or *Journal of Sleep Research* (sleep-physiology audience); *Journal of Neuroscience* or *eNeuro* (systems neuroscience, cross-species); *Journal of Neuroscience Methods* for the harmonised pipeline/multiverse; *Scientific Data* for a harmonised event catalogue across species.
- **Follow-ups.** (i) Ripple-spindle nesting with the hippocampal channels in DANDI:000978/000044 vs human hippocampal iEEG; (ii) ageing: add aged-rodent datasets as they appear on DANDI and compare with the NSRR ageing gradient (link to `sleep-spindle-aging-biomarker`); (iii) closed-loop stimulation timing recommendations that are species-consistent (cf. Ngo et al., 2013, *Neuron*).

## Risks, confounds & mitigations

- **State labelling differs** (human 30-s AASM epochs vs rodent continuous state segmentation). Mitigation: use NREM-only segments longer than 60 s in all species; sensitivity analysis with the crude spectral scorer applied to humans too.
- **Reference/polarity.** Depth LFP polarity depends on layer; scalp SO polarity is negative-up by convention. Mitigation: define the SO cycle by the trough of the *surface-negative / deep-positive* deflection consistently; test H5 with local CSD phase; report results for both polarities.
- **Anaesthesia/epilepsy.** MNI atlas channels come from epilepsy patients (normal regions only). Mitigation: use only channels labelled normal; exclude segments with interictal spikes using a simple spike detector; compare frontal contacts to scalp Fpz-Cz.
- **Sampling rates and filters** (Sleep-EDF at 100 Hz limits spindle-band fidelity). Mitigation: 200 Hz common rate; verify that detector counts do not depend on the original rate using NSRR 256 Hz data downsampled to 100 Hz.
- **Rodent spindle definition is contested** (10-15 vs 10-20 Hz; "spindle-like" events in delta). Mitigation: individualised band from the spectrum; report the fraction of recordings without a sigma peak as a result in itself.
- **Data volume.** Stream LFP channels rather than downloading whole NWB files; cache a fixed set of channels per session.

## Milestones

- [ ] Fetch dandiset asset lists and choose sessions/channels (`scripts/download_data.py --dataset dandi --dandiset 000041 --list`).
- [ ] Human scalp (Sleep-EDF) and MNI atlas ingestion; NREM segment extraction across all sources.
- [ ] Spectral individualisation: sigma-peak detection rate per species and recording scale.
- [ ] Detector calibration against MASS expert spindle annotations (optional) and against published rodent counts.
- [ ] Harmonised metrics for all recordings; H1-H3 tables and circular plots.
- [ ] Detector multiverse (24 specifications) and specification curves (H4).
- [ ] Laminar/region analysis (H5).
- [ ] Pre-registration after pilot on two sessions per source; manuscript; release of harmonised event catalogue.

## Ethics / data-use notes

- DANDI datasets are CC-BY: cite the dandiset DOIs and the primary papers. Do not redistribute NWB files.
- The MNI Open iEEG Atlas requires acceptance of its terms of use; data derive from patients and must be used as provided (no re-identification attempts). Cite Frauscher et al. (2018) and von Ellenrieder et al. (2020).
- NSRR data are under a DUA (see `data/README.md`); per-subject derived data are not committed and not sent to third-party services. Sleep-EDF is open but is still not committed.
- Animal data were collected under the original studies' approvals; no new animal work is involved.
