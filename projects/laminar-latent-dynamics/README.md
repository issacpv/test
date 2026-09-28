# Laminar latent dynamics: do cortical layers occupy distinct population subspaces?

**One-sentence pitch.** Use CSD-anchored layer assignment on the Allen Visual Coding and Visual Behavior Neuropixels surveys (plus the Senzai et al. laminar V1 dataset on DANDI) to test whether L2/3, L4, L5 and L6 populations within one cortical column share a single low-dimensional latent state or occupy distinct subspaces, how much of each layer's latent variance is stimulus- versus movement-driven, and whether inter-laminar "communication subspaces" are lower-dimensional than within-layer activity - with unit-count-matched and depth-shuffled nulls that existing laminar analyses lack.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc/PhD; 9-12 months. Statistically careful but computationally light.
- Compute: CPU; the Allen processed data are per-session NWB files (~2-3 GB per session file plus ~2 GB per probe LFP file; verified sizes on S3) and the Visual Coding cache CSVs. Storage: 150-300 GB for the sessions used; a workstation with 64 GB RAM is comfortable.
- Related projects in this repo: `neuropixels-representational-drift` (same Allen Neuropixels data, different question) and `ibl-brainwide-decoding`. This project is self-contained.

## Background

Cortical layers differ in inputs, outputs and cell types (Harris & Shepherd, 2015, *Nat Neurosci*; Adesnik & Naka, 2018, *Neuron*), and in mouse V1 they differ in tuning and state modulation (Niell & Stryker, 2008, *J Neurosci*; 2010, *Neuron*). Population-level analyses of cortex, however, usually pool all layers into one "area" and describe a single latent state (e.g., Stringer et al., 2019, *Science*; Musall et al., 2019, *Nat Neurosci*). Laminar-resolved population work exists in anaesthetised/awake rodent auditory cortex (Sakata & Harris, 2009, *Neuron*) and in macaque V1/V2 with communication-subspace methods (Semedo et al., 2019, *Neuron*), and the Allen Institute has used CSD-aligned Neuropixels to study inter-areal, layer-specific signal flow (Jia et al., 2022, *Neuron*; Siegle et al., 2021, *Nature*). Senzai, Fernandez-Ruiz & Buzsáki (2019, *Neuron*) defined physiological laminar landmarks in mouse V1 (depth of maximal spike power, CSD sink-source pattern) and characterised inter-laminar interactions; their data are open (DANDI:000166).

What is still unknown is the *geometry* of population activity across layers of one column: whether the low-dimensional latents that dominate V1 activity (Stringer et al., 2019) are shared by all layers or are layer-specific, which layer's latents carry stimulus information versus locomotion/pupil-related variance, and whether the between-layer interaction lives in a communication subspace of lower dimension than each layer's own activity - the within-area analogue of Semedo et al. Dimensionality-reduction tools to answer this exist (factor analysis with cross-validated dimensionality, Williamson et al., 2016, *PLoS Comput Biol*; GPFA, Yu et al., 2009, *J Neurophysiol*; reduced-rank regression; DLAG, Gokcen et al., 2022, *Nat Comput Sci*), and the Allen surveys provide hundreds of laminar-spanning V1 penetrations with simultaneous running and pupil data.

## The research gap

**What has been done.**

- Layer assignment on Neuropixels: CSD sink identification (Mitzdorf, 1985, *Physiol Rev*; iCSD, Pettersen et al., 2006, *J Neurosci Methods*); the Allen SDK provides flash-evoked CSD per probe; Senzai et al. (2019) provide physiological landmarks; Jia et al. (2022) aligned Allen probes to L4 by CSD to study laminar signal flow between areas.
- Population geometry in mouse V1 without layers: Stringer et al. (2019, *Science*; 2019, *Nature*) on dimensionality and behaviour-related variance; Musall et al. (2019) on movement-dominated single-trial dynamics.
- Inter-areal communication subspaces: Semedo et al. (2019); reviews of methods (Semedo et al., 2020, *Curr Opin Neurobiol*); recent macaque laminar work reports a structurally stable inter-laminar communication subspace in V1 (2024-2025 preprints), and 2025 studies apply supervised latent models separately to L4 and L2/3 populations.

**What is specifically missing.**

1. **Within-column laminar subspace geometry in mouse, at scale.** No study has estimated, across dozens of columns, the principal angles / overlap between the latent subspaces of different layers, nor tested whether they are more aligned than expected from random partitions of the same units (depth-shuffled null) with unit-count matching.
2. **Layer-resolved variance partitioning.** Which layer's latent variance is stimulus-driven vs running/pupil-driven has not been quantified with a common encoding model across layers (the Stringer/Musall results are area-level).
3. **Communication dimensionality between layers vs within layers.** Whether L4 -> L2/3 -> L5 interactions occupy a low-rank communication subspace (as between areas) or share the full within-layer dimensionality is untested in mouse.
4. **Robustness to layer-assignment error.** Layer labels from average-CCF templates are approximate (as noted by the Allen team); results should be shown as a function of assignment method (CSD sink vs CCF vs spike-power landmark) and boundary jitter. No laminar Neuropixels paper reports this sensitivity.
5. **Replication across datasets/paradigms.** Visual Coding (passive), Visual Behavior (active change detection) and DANDI:000166 (different lab, chronic recordings) allow a rare three-way replication of laminar population claims.

## Research questions / hypotheses

1. **H1 (shared vs distinct latents).** After unit-count matching, the mean overlap (cos^2 of principal angles) between the top-k latent subspaces of two layers (k from cross-validated FA) is lower for L4 vs L5/L6 than for L2/3 vs L4, and lower than the depth-shuffled null for every pair; i.e., layers are partially but not fully segregated.
2. **H2 (stimulus vs movement).** In an encoding model with stimulus and behavioural (running, pupil, face-motion) regressors, the share of latent variance uniquely explained by stimulus is highest in L4 and the share uniquely explained by behaviour is highest in L5/L6 and L2/3; the ordering holds in Visual Coding and Visual Behavior.
3. **H3 (communication subspace).** Reduced-rank regression between layers reaches 95% of full-rank predictive R^2 at a rank smaller than the cross-validated FA dimensionality of either layer (a within-column communication subspace); rank is smallest for L4 -> L2/3.
4. **H4 (state dependence).** Between-layer subspace overlap increases during locomotion relative to stationary periods, and during spontaneous (gray-screen) activity relative to natural-movie epochs.
5. **H5 (hierarchy).** Laminar segregation (1 - overlap) is larger in V1 than in higher visual areas (LM, AL, PM, AM, RL) in Visual Coding, consistent with weaker laminar specificity up the hierarchy.
6. **H6 (robustness).** H1-H3 conclusions are unchanged when layer boundaries are jittered by +/- 50 um and when CCF-based labels are substituted for CSD-based labels; results degrade gracefully with unit-count subsampling.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Visual Coding Neuropixels (S3 `allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/`) | 58 sessions (`brain_observatory_1.1`, `functional_connectivity`), ~6 probes each; `units.csv`, `channels.csv` (vertical positions, CCF coordinates, structure acronyms), `probes.csv` (`surface_channel_index`), per-probe LFP NWB with flash-evoked CSD, session NWB with stimulus tables, running speed, pupil | ~2.9 GB/session NWB + ~2 GB/probe LFP NWB; CSVs < 200 MB | Open (AWS Open Data; `allensdk` EcephysProjectCache) | https://registry.opendata.aws/allen-brain-observatory/ |
| Allen Visual Behavior Neuropixels (S3 `visual-behavior-neuropixels-data`) | `ecephys_session_<id>.nwb` + `probe_probeX_lfp.nwb` per session; active change-detection task with running, pupil, licking | ~150 sessions; 2-4 GB per session | Open (`allensdk` VisualBehaviorNeuropixelsProjectCache) | https://registry.opendata.aws/allen-brain-observatory/ |
| DANDI:000166 - Senzai, Fernandez-Ruiz & Buzsáki 2019, "Layer-specific physiological features and interlaminar interactions in the primary visual cortex of the mouse" | Chronic laminar silicon-probe recordings across all V1 layers with physiological laminar landmarks; independent replication | 19 NWB, ~787 GB | Open | https://dandiarchive.org/dandiset/000166 |
| Allen CCFv3 (via `allensdk` reference space) | Average-template layer annotation for the CCF-based assignment arm of H6 | ~1 GB | Open | https://allensdk.readthedocs.io/ |

## Methods

1. **Tables and depths (`lamlat.io_allen`).** Load `units.csv`/`channels.csv`/`probes.csv`; keep units in cortical visual areas (`ecephys_structure_acronym` in VISp, VISl, VISal, VISpm, VISam, VISrl); compute depth from the cortical surface per probe (`surface_channel_index`, channel `probe_vertical_position`).
2. **Layer assignment (`lamlat.layers`).** Flash-evoked CSD per probe (Allen-provided, or recomputed from the LFP NWB with `compute_csd`); L4 anchored at the earliest strong current sink after flash onset (`find_l4_sink`); layer boundaries from the L4 centre using mouse V1 thickness priors (`assign_layers`; L1 100, L2/3 200, L4 150, L5 250, L6 300 um by default), with alternative arms: CCF labels and the spike-power landmark of Senzai et al.
3. **Population activity (`lamlat.latent`).** 50-ms binned spike counts per layer (sqrt-transformed) for natural movies, drifting gratings and spontaneous epochs; cross-validated FA dimensionality per layer; shared/private variance; principal angles and overlap between layer subspaces; reduced-rank regression between layers with cross-validated rank selection (communication dimensionality).
4. **Variance partitioning (`lamlat.partition`).** Ridge encoding models with stimulus one-hot/lagged regressors and behavioural regressors (running, pupil, face motion from the NWB); unique/shared R^2 by nested models (Musall-style) per unit and per latent dimension.
5. **Nulls and matching (`lamlat.latent`).** Depth-shuffled null (permute depths among units, recompute layer-pair statistics), unit-count-matched subsampling (equal n per layer, repeated), and boundary-jitter sensitivity.
6. **Replication.** Same pipeline on Visual Behavior sessions (active task) and on DANDI:000166 (Senzai landmarks used directly).

## Evaluation & statistics

- Unit of analysis: probe insertion (column) x stimulus epoch. Mixed-effects models with session (mouse) as random effect for H1-H5; layer pair as fixed effect.
- Nulls: (a) depth-shuffle null per column (1,000 permutations) with z-scores; (b) unit-count-matched subsampling (100 repeats, n = min layer count, minimum 15 units per layer or the column is excluded); (c) *split-unit* null: split a single layer randomly into two halves and compute overlap - the ceiling against which between-layer overlap is compared.
- Dimensionality: FA log-likelihood cross-validation (5-fold over time bins in contiguous blocks to respect autocorrelation); RRR rank by the same blocked CV.
- Multiple comparisons: H1-H6 as pre-registered families; Holm within family; hierarchy analyses (H5) use a single trend test across areas ordered by anatomical hierarchy score (Siegle et al., 2021).
- Leakage prevention: encoding models cross-validated in contiguous time blocks; behavioural regressors lagged to avoid using future information; layer assignment done blind to activity statistics (CSD only).
- Reporting: overlap matrices per column, specification-style plots across assignment methods, and effect sizes with CIs.

## Publishable angle

- **Headline result.** "Across >100 laminar V1 penetrations, layers share a common low-dimensional state but retain layer-specific subspaces (overlap X vs shuffle null Y); L4 latents are stimulus-dominated, deep-layer latents are movement-dominated, and inter-laminar interactions occupy a communication subspace of rank r << within-layer dimensionality. The pattern weakens up the visual hierarchy and replicates in an active task and in an independent laboratory's chronic recordings."
- **Venues.** *Neuron* or *Nature Neuroscience* (if the replication across three datasets is strong); *eLife* / *Journal of Neuroscience* (systems); *PLoS Computational Biology* for the methods-focused version (nulls, matching, assignment sensitivity).
- **Follow-ups.** (i) Cell-type-resolved latents using the Cre-line optotagging sessions in Visual Coding (Sst, Pvalb, Vip); (ii) laminar latents across areas simultaneously (extending Jia et al., 2022, to subspace geometry); (iii) time-resolved (DLAG) directionality between layers.

## Risks, confounds & mitigations

- **Layer assignment error** (probe angle, CSD ambiguity, CCF template mismatch). Mitigation: three assignment arms; boundary jitter; report the fraction of probes with a clean L4 sink; exclude probes without one.
- **Unequal unit counts per layer** (L4/L6 often sparser). Mitigation: unit-count matching and split-unit ceilings; minimum-unit thresholds.
- **Non-stationarity / drift** biasing shared variance across layers. Mitigation: blocked CV; include time as a regressor; compare early vs late halves.
- **Behavioural regressors differ across datasets** (no face camera in some sessions). Mitigation: running + pupil as the common core; face motion as an add-on.
- **Overlap metrics depend on k.** Mitigation: report overlap as a function of k and use CV-selected k per layer.
- **Multiple probes in one area within a session** are not independent. Mitigation: session random effect; probe nested in session.

## Layer-assignment priors and inclusion criteria (pre-specified)

| Layer | Default thickness prior (um, mouse V1; `lamlat.layers.MOUSE_V1_THICKNESS_UM`) | Anchor |
|---|---|---|
| L1 | 100 | above the L2/3 top |
| L2/3 | 200 | above the L4 top |
| L4 | 150 | centred on the earliest flash-evoked CSD sink at 20-80 ms, 100-700 um below the surface (`find_l4_sink`) |
| L5 | 250 | below the L4 bottom |
| L6 | 300 | below the L5 bottom |

Priors can be replaced per area; boundary jitter (+/- 50 um), CCF labels and the spike-power landmark of Senzai et al.
(2019) are the sensitivity arms of H6. A column enters the analysis when: a clean L4 sink exists (sink magnitude > 3 SD
of the pre-stimulus CSD at the same depth), at least three layers have >= 15 QC-passing units, and >= 20 min of the
relevant stimulus epoch (natural movies, drifting gratings, spontaneous) are available. Bin width 50 ms; counts are
square-root transformed; behavioural regressors are running speed (lags 0-500 ms), pupil area and, where present,
face-motion energy.

## Pre-specified outputs

1. Figure 1: CSD sink depth and latency distributions, unit counts per layer per column, agreement between assignment arms.
2. Figure 2: layer x layer overlap matrices per area against the depth-shuffle null and the split-unit ceiling, unit-count matched (H1).
3. Figure 3: unique stimulus and unique behaviour variance by layer (median across units and across latent dimensions) in both Allen datasets (H2).
4. Figure 4: RRR cross-validated R^2 vs rank and communication dimensionality per layer pair (H3); locomotion and stimulus-epoch dependence of overlap (H4).
5. Figure 5: laminar segregation vs anatomical hierarchy score across VISp, VISl, VISal, VISpm, VISam, VISrl (H5).
6. Supplement: robustness to boundary jitter and assignment method (H6); replication in DANDI:000166.

## Repository layout and quick start

```
src/lamlat/   io_allen.py (cache tables, depth from surface, unit selection, LFP NWB reader)
              layers.py (CSD, L4 sink, layer boundaries and assignment, jitter)
              latent.py (count matrices, FA dimensionality, overlaps, RRR, depth-shuffle null, matching)
              partition.py (ridge encoding models, unique/shared variance, layer summaries)
scripts/download_data.py   tests/test_lamlat.py   data/README.md
```

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q
python scripts/download_data.py --cache                 # sessions/probes/channels/units CSVs from S3
python - <<'EOF'
from pathlib import Path
from lamlat.io_allen import load_cache_tables, unit_depths_from_surface, select_units
t = load_cache_tables(Path("data/allen/visual-coding/cache"))
u = select_units(unit_depths_from_surface(t["units"], t["channels"], t["probes"]), session_id=715093703)
print(u.groupby("structure")["depth_um"].describe())
EOF
```

## Milestones

- [ ] Download cache CSVs and 5 pilot sessions (`scripts/download_data.py --cache --sessions 5`); compute depths.
- [ ] CSD-based L4 anchoring for all V1 probes; QC figure of sink latency/depth distributions.
- [ ] FA dimensionality and subspace overlaps per column with nulls (H1) on Visual Coding V1.
- [ ] Encoding models and variance partitioning by layer (H2).
- [ ] RRR communication dimensionality (H3); state dependence (H4).
- [ ] Hierarchy analysis across areas (H5); assignment-robustness arm (H6).
- [ ] Replication on Visual Behavior and DANDI:000166.
- [ ] Pre-registration after the pilot; manuscript and code release.

## Ethics / data-use notes

- Allen Brain Observatory data: Allen Institute Terms of Use (cite Siegle et al., 2021, and the Visual Behavior Neuropixels dataset); DANDI:000166 is CC-BY (cite Senzai et al., 2019). No new animal work.
- Do not commit data (`.gitignore`); commit manifests and aggregate results only.
