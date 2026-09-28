# Cross-modality tuning reproducibility: decomposing the two-photon vs Neuropixels gap into sampling, measurement and analysis

**One-sentence pitch.** Using the Allen Brain Observatory's matched-stimulus datasets (Visual Coding 2-photon and Neuropixels; Visual Behavior 2-photon and Neuropixels), quantify how much of the well-documented difference in visual-tuning estimates between calcium imaging and electrophysiology is due to *which cells are sampled* (depth, cell class, firing rate), *how activity is measured* (indicator kinetics and nonlinearity, deconvolution) and *how tuning is analysed* (metric definition, inclusion/QC choices), test whether the decomposition transfers from the passive to the active-task datasets, and derive reliability-adjusted, modality-invariant tuning metrics with pre-registered acceptance criteria.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc/PhD; 9-12 months. Analysis-only; the datasets are already processed (ROI traces, sorted units) and available on public S3 buckets.
- Compute: CPU workstation; the per-experiment NWB files are 0.5-3 GB (Visual Coding 2P ~1,300 experiments; Neuropixels 58 sessions; Visual Behavior 2P several hundred experiments; Visual Behavior Neuropixels ~150 sessions). Storage 0.5-2 TB depending on how many experiments are cached.
- Related project in this repo: `neuropixels-representational-drift` (Neuropixels within-session stability, one ingredient of the reliability ceiling here). This project is self-contained.

## Background

Calcium imaging and extracellular electrophysiology are the two workhorse population-recording methods, and both are used to make quantitative claims about tuning (fraction of orientation-selective neurons, selectivity indices, sparseness, preferred temporal frequency). The Allen Institute's Visual Coding programme recorded the *same* visual stimuli with 2-photon imaging (de Vries et al., 2020, *Nat Neurosci*) and Neuropixels (Siegle et al., 2021, *Nature*), and the direct comparison (Siegle et al., 2021, *eLife*, "Reconciling functional differences...") found consistent stimulus *preferences* across modalities but systematic differences in *responsiveness and selectivity*, partly attributable to missed/merged low-rate units in ephys and to indicator nonlinearities that sparsify and amplify bursts; a forward model of spikes -> fluorescence reduced but did not eliminate the gap. Simultaneous cell-attached + imaging recordings in GCaMP6 mice (Huang et al., 2021, *eLife*) and earlier comparisons (Wei et al., 2020, *PLoS Comput Biol*; Ledochowitsch et al., 2019, bioRxiv) established that single-spike detectability and burst amplification depend on indicator, cell type and firing rate. Spike inference from calcium is itself model-dependent (Theis et al., 2016, *Neuron*; Berens et al., 2018, *PLoS Comput Biol*; Pachitariu, Stringer & Harris, 2018, *J Neurosci*; Rupprecht et al., 2021, *Nat Neurosci*). The Visual Behavior datasets (2P: Garrett et al., 2023, bioRxiv; Neuropixels: Allen Institute MindScope, 2022) repeat the two-modality design in an active change-detection task with natural images, which nobody has used to test whether the Visual Coding conclusions generalise.

## The research gap

**What has been done.**

- The same-stimulus modality comparison on Visual Coding: Siegle et al. (2021, *eLife*) - preferences agree; responsiveness/selectivity differ; forward model partially reconciles; differences in dF/F vs deconvolved events noted.
- Ground-truth calibration of imaging against spikes: Huang et al. (2021); Chen et al. (2013, *Nature*) for GCaMP6 kinetics; spike-inference benchmarks (Theis et al., 2016; Berens et al., 2018; Rupprecht et al., 2021).
- Dataset descriptions and sharing: de Vries et al. (2020); Siegle et al. (2021, *Nature*); de Vries, Siegle & Koch (2023, *eLife*) on Allen data sharing; Visual Behavior 2P (Garrett et al., 2023) and Visual Behavior Neuropixels (dataset whitepaper).
- Representational drift and reliability within a modality: Deitch, Rubin & Ziv (2021, *Curr Biol*) on Visual Coding 2P; Marks & Goard (2021, *Nat Commun*).

**What is specifically missing.**

1. **A quantitative decomposition of the modality gap.** Siegle et al. reported the gap and one correction. The relative contributions of sampling (2P samples layer-specific Cre-line populations with all cells visible; Neuropixels samples all layers but favours high-rate units), measurement (indicator kinetics, nonlinearity, neuropil), and analysis (metric, QC/inclusion) have not been estimated within one framework, nor with a reliability ceiling.
2. **Reliability-adjusted agreement.** Tuning metrics are noisy; cross-modality agreement should be compared to within-modality test-retest reliability (2P: same cells across sessions A/B/C in Visual Coding containers; Neuropixels: repeated stimulus blocks within session). Disattenuated agreement (Spearman-Brown / correction for attenuation) has not been reported.
3. **Transfer to a second paradigm.** Whether the Visual Coding forward-model parameters and the observed gap replicate in Visual Behavior (active task, natural images, different Cre lines: Slc17a7, Sst, Vip) is untested; a decomposition that transfers is far more useful than one that fits a single dataset.
4. **A QC/inclusion multiverse across modalities.** 2P ROI filtering, neuropil correction, dF/F vs deconvolved events, and Neuropixels QC thresholds each shift metric distributions; the joint multiverse and its effect on "fraction selective" claims has not been quantified.
5. **Modality-invariant metrics.** Rank-based preference measures and reliability-normalised selectivity should be more invariant than raw OSI/DSI or responsiveness fractions; identifying such metrics, with pre-registered invariance criteria, is the constructive outcome.

## Research questions / hypotheses

1. **H1 (preferences vs magnitudes).** Preferred direction/orientation/TF/SF distributions per area agree across modalities (Wasserstein distance within the within-modality bootstrap null) in both Visual Coding and Visual Behavior, whereas responsiveness fractions and OSI/DSI/lifetime-sparseness distributions differ by more than the null in both.
2. **H2 (sampling).** Reweighting Neuropixels units to match the 2P depth (layer) and cell-class (Cre-line proxy via waveform/rate) distribution via propensity weights closes at least 30% of the responsiveness-fraction gap; reweighting 2P cells by estimated firing rate closes a further part.
3. **H3 (measurement).** Passing Neuropixels spike trains through a GCaMP6f/6s forward model with supralinear burst amplification and then applying the *same* 2P analysis pipeline (dF/F event detection or deconvolution) closes at least another 30% of the gap in lifetime sparseness and responsiveness; the fitted nonlinearity parameters are similar (within 20%) between Visual Coding and Visual Behavior fits.
4. **H4 (analysis).** Across a QC/inclusion x metric-definition multiverse, the fraction-selective claim varies by > 15 percentage points within each modality; deconvolved-event-based metrics are closer to ephys than dF/F-based metrics for every area.
5. **H5 (reliability ceiling).** After disattenuation with within-modality test-retest reliability, cross-modality agreement of per-area median OSI exceeds 0.8 for preferences and 0.5 for magnitudes; raw agreement is substantially lower.
6. **H6 (invariance).** A small set of metrics (rank-based preference, reliability-normalised selectivity, deconvolution-based sparseness) satisfies a pre-registered invariance criterion (|standardised modality difference| < 0.2 SD in both datasets), whereas raw responsiveness and OSI do not.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Visual Coding 2-photon (S3 `allen-brain-observatory/visual-coding-2p/`) | ~1,300 experiments, ~60,000 neurons, 6 areas, 14 Cre lines, sessions A/B/C per container with drifting/static gratings, natural scenes/movies; `ophys_experiment_analysis/*_analysis.h5` (tuning metrics), `cell_specimens.json`, per-experiment NWB (dF/F, events) | ~50 GB metrics; ~1 TB all NWB | Open (AWS Open Data; `allensdk` BrainObservatoryCache) | https://registry.opendata.aws/allen-brain-observatory/ |
| Allen Visual Coding Neuropixels (S3 `.../visual-coding-neuropixels/ecephys-cache/`; DANDI:000021/000022 mirrors) | 58 sessions, ~40,000 QC units, same stimulus set; `brain_observatory_1.1_analysis_metrics.csv`, `units.csv`, session NWB | ~100 GB NWB; CSVs < 200 MB | Open | same bucket; https://dandiarchive.org/dandiset/000021 |
| Allen Visual Behavior 2-photon (S3 `visual-behavior-ophys-data/visual-behavior-ophys/`) | `behavior_ophys_experiment_<id>.nwb`: Slc17a7/Sst/Vip lines, multi-plane, natural-image change detection with running/pupil/licking | hundreds of experiments, ~0.5-1 GB each | Open (`allensdk` VisualBehaviorOphysProjectCache) | https://registry.opendata.aws/allen-brain-observatory/ |
| Allen Visual Behavior Neuropixels (S3 `visual-behavior-neuropixels-data/visual-behavior-neuropixels/`) | `ecephys_session_<id>.nwb` + probe LFP; same task and images | ~150 sessions, 2-4 GB each | Open (`allensdk` VisualBehaviorNeuropixelsProjectCache) | as above |
| Simultaneous imaging + cell-attached ground truth (Huang et al., 2021, *eLife*) | Spike-to-fluorescence transfer functions per indicator/line for forward-model priors | GB scale | Open (released with the paper; see its data availability statement) | https://elifesciences.org/articles/51675 |

## Methods

1. **Common stimulus-locked responses (`xmodal.tuning`).** For every cell/unit: trial x condition response matrices for drifting gratings (8 directions x 5 TF in Visual Coding), static gratings, natural scenes/movies (Visual Coding) and natural images (Visual Behavior). 2P responses in three versions: mean dF/F, extracted events (Allen L0 events), and AR(1)-deconvolved rates; ephys responses as spike counts. Metrics: responsiveness (permutation), vector OSI/DSI and ratio OSI (Mazurek, Kager & Van Hooser, 2014; circular variance after Ringach, Shapley & Hawken, 2002), preferred TF/SF/direction, lifetime sparseness (Vinje & Gallant, 2000).
2. **Forward model (`xmodal.forward_model`).** Spikes -> calcium (double-exponential GCaMP6f/6s kernels from Chen et al., 2013) -> supralinear amplification and saturation -> dF/F with noise; parameters fitted so that the forward-modelled Neuropixels population matches the 2P population on *held-out* metrics (fit on Visual Coding, test on Visual Behavior); AR(1) deconvolution to compare like with like.
3. **Reliability (`xmodal.reliability`).** Split-half reliability per cell/unit and per metric (Spearman-Brown), 2P across-session reliability for matched cells (containers A/B/C), Neuropixels across-block reliability; disattenuated cross-modality agreement of per-area statistics.
4. **Decomposition and multiverse (`xmodal.multiverse`).** Distribution-shift statistics (KS, Wasserstein, quantile differences) between modalities; propensity-score reweighting on depth/layer, cell-class proxies and firing-rate proxies (sampling term); forward-model matching (measurement term); metric/QC grid (analysis term); the gap is decomposed sequentially with bootstrap CIs and order-sensitivity (Shapley-style averaging over orderings).
5. **Replication.** Full pipeline on Visual Coding, then frozen and applied to Visual Behavior; transfer of the forward-model parameters is a pre-registered test (H3).

## Evaluation & statistics

- Unit of analysis: area x layer x metric; experiments/sessions nested in mice. Mixed models with mouse random effects for magnitude comparisons; distribution comparisons via Wasserstein distance with a within-modality bootstrap null (resample experiments/sessions).
- Reliability-adjusted agreement: correlation of per-area medians across modalities, divided by the geometric mean of within-modality reliabilities; CIs by bootstrap over experiments.
- Gap decomposition: sequential attribution (sampling -> measurement -> analysis) averaged over all 6 orderings; report the share of the raw gap closed by each term with 95% bootstrap CIs.
- Multiverse: full factorial grid (2P: 3 signal types x 2 neuropil rules x 2 ROI QC levels; ephys: 3 QC rules) x 3 metric definitions; specification curves for "fraction selective" per area.
- Multiple comparisons: H1-H6 as pre-registered families; Holm within family; invariance criteria (H6) fixed before Visual Behavior is analysed.
- Leakage prevention: forward-model parameters fitted on Visual Coding only; Visual Behavior held out; metric thresholds fixed a priori (e.g., OSI > 0.5 "selective").
- Nulls: label-shuffled tuning (chance selectivity), within-modality split null for distribution distances, and a forward-model null with linear (non-amplifying) kernels.

## Publishable angle

- **Headline result.** "Of the two-photon vs Neuropixels difference in the fraction of selective V1 neurons, X% is sampling, Y% is measurement and Z% is analysis; the decomposition transfers from passive viewing to an active task, and reliability-normalised, deconvolution-based metrics are modality-invariant to within 0.2 SD." Plus a practical checklist for reporting tuning statistics that will survive a change of recording modality.
- **Venues.** *eLife* (natural home for the Siegle et al. follow-up); *Journal of Neuroscience* or *Cell Reports*; *PLoS Computational Biology* / *Neuron* (methods-forward if the invariant metrics are strong); *Scientific Data* for a harmonised cross-modality tuning table.
- **Follow-ups.** (i) Extend to functional connectivity/noise correlations across modalities; (ii) use the OpenScope 2P/Neuropixels datasets on DANDI as an external replication; (iii) cross-lab: compare the Allen Neuropixels tuning distributions with IBL passive-stimulus data after stimulus harmonisation.

## Risks, confounds & mitigations

- **Stimulus-set differences between datasets** (Visual Coding gratings vs Visual Behavior natural images). Mitigation: within-dataset comparisons only; cross-dataset tests concern the *decomposition*, not raw metrics.
- **Cre-line sampling in 2P vs unlabeled ephys populations.** Mitigation: restrict primary comparison to excitatory lines (Slc17a7/Emx1/Cux2/Rorb/Rbp4/Ntsr1 by layer) vs regular-spiking ephys units; report inhibitory lines separately.
- **Depth/layer registration** differs (imaging depth vs probe depth). Mitigation: coarse layer bins; sensitivity analysis on bin edges.
- **Forward-model over-fitting.** Mitigation: fit on Visual Coding, evaluate on Visual Behavior; parameter priors from Huang et al. (2021).
- **Deconvolution algorithm choice** is itself a specification. Mitigation: include AR(1) deconvolution, Allen events and dF/F as multiverse arms; report all.
- **Ephys undersampling of low-rate cells cannot be reweighted if they are absent.** Mitigation: report the reweighting's effective sample size; treat residual gap after reweighting as a lower bound on the measurement term.

## Multiverse factors and matching covariates (pre-specified)

| Modality | Factor | Levels |
|---|---|---|
| 2-photon | signal | mean dF/F; Allen L0 events; AR(1)-deconvolved rate (`xmodal.forward_model.deconvolve_ar1`, gamma from the indicator decay) |
| 2-photon | neuropil correction | Allen default (fitted r); fixed r = 0.7 |
| 2-photon | ROI inclusion | Allen default filters; additional SNR filter (top 80% by event SNR) |
| Neuropixels | curation | Allen default (ISI ratio < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9); lenient (1.0 / 0.3 / 0.5); none |
| both | metric | vector OSI/DSI; ratio OSI/DSI; reliability-normalised OSI (`reliability_normalised_selectivity`); lifetime sparseness; responsiveness (permutation p < 0.05) |
| both | response window | 0-0.5 s and 0-2 s after onset (drifting gratings); 0-0.25 s and 0-0.75 s (natural images) |

Sampling covariates for propensity reweighting (`xmodal.multiverse.propensity_weights`): cortical depth binned to L2/3,
L4, L5, L6; excitatory vs inhibitory proxy (Cre line for 2P, waveform trough-to-peak duration for Neuropixels); and a
firing-rate proxy (spike rate for Neuropixels, deconvolved event rate for 2P). Effective sample sizes and standardised
mean differences before/after reweighting are reported with every reweighted estimate.

Forward-model parameters fitted on Visual Coding and frozen before Visual Behavior: indicator kernel (GCaMP6f for
Ai93/Ai148 lines, GCaMP6s for Ai94/Ai162; approximate single-spike time constants from Chen et al., 2013), supralinearity
exponent gamma in {1, 1.25, 1.5, 2}, saturation c_sat in {inf, 5, 2}, amplitude and noise from matched dF/F statistics.

## Pre-specified outputs

1. Table 1: cells/units per area x layer x modality x dataset under each inclusion arm, with effective sample sizes after reweighting.
2. Figure 1: distributions of preferences (direction, TF, SF) and magnitudes (OSI, DSI, sparseness, responsiveness) by modality, with Wasserstein distances and within-modality bootstrap nulls (H1).
3. Figure 2: sequential gap decomposition into sampling, measurement and analysis with ordering-averaged shares and bootstrap CIs (`sequential_gap_decomposition`; H2-H4).
4. Figure 3: raw vs disattenuated cross-modality agreement of per-area statistics against the reliability ceiling (H5).
5. Figure 4: invariance check per metric in Visual Coding and, held out, in Visual Behavior (H6); specification curves for "fraction selective".
6. Supplement: forward-model fits and transfer; deconvolution-method sensitivity; inhibitory-line analyses.

## Repository layout and quick start

```
src/xmodal/   tuning.py (condition means, OSI/DSI, preferences, sparseness, responsiveness)
              forward_model.py (GCaMP kernels, nonlinearity, dF/F, event responses, AR(1) deconvolution, fitting)
              reliability.py (split-half, test-retest, Spearman-Brown, disattenuation, bootstrap)
              multiverse.py (distribution shift, propensity weights, balance, gap decomposition, grids, invariance)
scripts/download_data.py   tests/test_xmodal.py   data/README.md
```

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q
python scripts/download_data.py --dataset vcnpx --sample     # cache CSVs + released Neuropixels tuning metrics
python scripts/download_data.py --dataset vc2p --sample      # 2P cell_specimens.json + 5 analysis files
python - <<'EOF'
import pandas as pd
from xmodal.multiverse import distribution_shift
npx = pd.read_csv("data/visual-coding-neuropixels/brain_observatory_1.1_analysis_metrics.csv")
print(npx.columns[:20].tolist())     # locate g_osi_dg / g_dsi_dg / pref_tf_dg / lifetime_sparseness_dg
EOF
```

## Milestones

- [ ] Download metric tables and 10 pilot experiments per dataset/modality (`scripts/download_data.py --sample`).
- [ ] Harmonised response matrices and metrics for Visual Coding 2P (three signal types) and Neuropixels.
- [ ] Reliability estimates (split-half, across-session/block); raw and disattenuated agreement (H1, H5).
- [ ] Sampling reweighting (H2); forward-model fit (H3) on Visual Coding.
- [ ] QC/metric multiverse and specification curves (H4).
- [ ] Freeze pipeline and parameters; pre-register invariance criteria; run on Visual Behavior (H3, H6).
- [ ] Manuscript; release harmonised tuning tables and the decomposition code.

## Ethics / data-use notes

- Allen Brain Observatory data: Allen Institute Terms of Use (non-commercial research; cite de Vries et al., 2020; Siegle et al., 2021; the Visual Behavior datasets). No new animal work.
- Do not commit data or per-cell tables derived from NWB files (`.gitignore`); commit aggregate results and manifests only.
