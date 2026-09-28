# morphology-to-electrophysiology

**Do graph neural networks on the dendritic tree learn a morphology → intrinsic-electrophysiology mapping that transfers from mouse to human and respects transcriptomic cell types — and do they beat biophysical model fitting as a "structure-function" baseline?**

## Status / difficulty / timeline / compute

- Status: proposal + starter code (allensdk fetcher, SWC → graph converter, own ephys feature extractor with a simulator, ridge baseline with grouped CV and transfer metrics, plain-PyTorch GNN).
- Difficulty: MSc/PhD-chapter level; requires comfort with allensdk/NWB, PyTorch, and single-cell electrophysiology.
- Timeline: 8-12 months (2 months data assembly + QC, 2 months features/baselines, 3 months GNN + transfer, 2 months biophysical comparison, 1-2 months writing).
- Compute: CPU is fine for the ridge/morphometric baselines; one mid-range GPU (8-16 GB) for GNN training on a few thousand graphs (each graph 500-5,000 nodes); biophysical model simulations (NEURON) need ~100-1,000 CPU-hours if you refit rather than reuse the Allen models.

## Background

The link between dendritic shape and firing is one of the oldest structure-function questions in cellular neuroscience: Mainen & Sejnowski (1996, Nature) showed that the same channel complement in different reconstructed morphologies produces different firing patterns. The Allen Cell Types Database (Gouwens et al., 2019, Nat Neurosci) provides, for the same cells, standardised patch-clamp protocols, extracted electrophysiological features (input resistance, sag, membrane time constant, rheobase, AP up/down-stroke ratio, adaptation, f-I slope), SWC reconstructions and, for mouse, the Cre driver line; Patch-seq extensions add transcriptomic (t-, MET-) types for mouse GABAergic (Gouwens et al., 2020, Cell), mouse glutamatergic and human cells (Berg et al., 2021, Nature; Lee et al., 2021, eLife; Chartrand et al., 2023, Science; Lee et al., 2023, Science). Human supragranular pyramidal cells differ from mouse in size, compartmentalisation and h-channel-dependent membrane properties (Kalmbach et al., 2018, Neuron; Beaulieu-Laroche et al., 2018, Cell; Deitcher et al., 2017, Cereb Cortex), so the mouse → human question is not trivial.

Modelling of the same data exists in two flavours: (i) biophysically detailed single-neuron models fitted per cell (perisomatic: Gouwens et al., 2018, Nat Commun; all-active: Nandi et al., 2022, Cell Reports — the latter explicitly links models to morphology and transcriptomics), and (ii) statistical learning on hand-crafted morphometrics or on graph/point-cloud representations of the SWC tree (Laturnus, Kobak & Berens, 2020, Neuroinformatics; Kanari et al., 2018, Neuroinformatics; GraphDINO, Weis et al., 2021, arXiv; GICLMorph, 2026).

## The research gap

**What has been done**

- A bioRxiv 2020 preprint ("Electrophysiology prediction of single neurons based on their morphology") trained fully connected networks on Allen *hand-crafted* morphometrics to predict multiple Allen ephys features with predictive distributions — mouse only, no graph representation, no cross-species test, no comparison with biophysical models.
- Graph/self-supervised morphology representations (GraphDINO; GICLMorph 2026; a 2025 contrastive morphology framework) are evaluated on *cell-type classification*, not on regression of intrinsic electrophysiology.
- Cross-species transfer with Allen Patch-seq data has been done for **electrophysiology → transcriptomic subclass** (Schwider & Ramezani, 2026, Neuroinformatics: attention-BiLSTM on IPFX feature families; mouse pre-training improves human 4-class macro-F1). Morphology is not an input in that work.
- Biophysical all-active models (Nandi et al., 2022) show that model parameters cluster by transcriptomic type and that morphology is needed for some features, but they were not used as a *predictive baseline* against learned models, nor tested on human cells.
- The morphology-ephys relationship in human vs. mouse has been characterised descriptively (Kalmbach 2018; Berg 2021; Chartrand 2023) rather than as a transportable predictive mapping.

**What is missing (verified by 2023-2026 searches)**

1. No published GNN that operates directly on the SWC tree to *regress* intrinsic ephys features (R_in, sag, tau, rheobase, AP width, adaptation, f-I slope), with proper donor/specimen-grouped validation.
2. No test of whether a morphology → ephys mapping learned on mouse transports to human cells (zero-shot and with few-shot fine-tuning), and which features transfer (passive, geometry-dominated features such as R_in/tau expected to transfer; active features such as AP width/adaptation expected not to).
3. No test of whether prediction residuals are structured by transcriptomic type (i.e. whether "morphology explains ephys *within* t-types" or only *between* t-types), which is the key to knowing whether morphology carries independent information.
4. No head-to-head against the biophysical baseline: using the Allen perisomatic/all-active models as a "physics" predictor (simulate the same protocol on the same morphology with type-average channel densities) vs. the learned predictor.

## Research questions / hypotheses

1. **H1 (morphology carries ephys information).** A GNN on the SWC graph predicts passive features (R_in, tau, sag) with out-of-fold R² > 0.3 on mouse Allen cells, beating ridge on 25 hand-crafted morphometrics by > 0.05 R². Active features (AP width, adaptation, f-I slope) are predicted worse (R² < 0.2).
2. **H2 (within-type information).** After removing t-type (or Cre-line/dendrite-type) means, morphology still explains a significant share of within-type ephys variance for passive features (partial R² > 0.1, permutation p < 0.05), but not for active features.
3. **H3 (mouse → human transfer).** Zero-shot transfer of the mouse model to human cells retains > 50% of within-species R² for R_in and tau after size normalisation but fails (R² ≤ 0) for sag and AP width; fine-tuning the head on ≤ 50 human cells closes most of the gap for passive features only.
4. **H4 (biophysical vs. learned).** Type-average biophysical models simulated on each cell's own morphology predict R_in/tau/rheobase at least as well as the GNN on mouse, but the GNN transfers to human better than the mouse-fitted biophysical parameters (because human channel densities differ, Kalmbach 2018).
5. **H5 (which parts of the tree matter).** Gradient-based node attributions concentrate on proximal apical/basal segments for R_in/tau and on the soma/axon-initial region for AP features; attributions are stable across random seeds (rank correlation > 0.6).

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Cell Types Database (mouse VISp + human neocortex) | Ephys feature table, NWB sweeps (long-square protocol), SWC reconstructions, Cre line / dendrite type / layer | on the order of 2,000 mouse and 400 human cells with ephys; hundreds of each with reconstructions (script prints exact counts) | Open (allensdk, no registration) | https://celltypes.brain-map.org , https://allensdk.readthedocs.io |
| Allen Patch-seq (mouse VISp GABAergic, Gouwens 2020; mouse glutamatergic; human neocortex, Berg 2021 / Chartrand 2023 / Lee 2023) | ephys + morphology + t-type / MET-type labels | thousands of cells, several hundred with reconstructions | Open (brain-map.org downloads; NWB on DANDI Archive) | https://portal.brain-map.org/explore/classes/multimodal-characterization , https://dandiarchive.org |
| Allen biophysical models (perisomatic, Gouwens 2018; all-active, Nandi 2022) | Fitted channel parameters and model code per cell; simulate on other morphologies | hundreds of models | Open (allensdk `BiophysicalApi`, NEURON) | https://allensdk.readthedocs.io/en/latest/biophysical_models.html |
| NeuroMorpho.org (mouse + human neocortex) | Morphology-only reconstructions for self-supervised GNN pre-training and for scaling the morphology encoder | tens of thousands | Open (REST API) | https://neuromorpho.org/api/ |

## Quick start

```bash
cd projects/morphology-to-electrophysiology
pip install -r requirements.txt                 # allensdk/torch are only needed for data access / the GNN
python -m pytest tests -q                       # synthetic tests: graph builder, ephys extractor, ridge, transfer
python scripts/download_data.py --sample        # 10 mouse + 5 human reconstructed cells + feature tables
python - <<'EOF'
import glob, pandas as pd, numpy as np
from morph2ephys.allen_fetch import load_cached_table, coarse_type_labels, PASSIVE_TARGETS
from morph2ephys.swc_graph import read_swc, morphometric_vector
from morph2ephys.baseline import multi_target_baseline, within_type_partial_r2
tab = load_cached_table("data/allen")
feats = {int(p.split("specimen_")[1].split("/")[0]): morphometric_vector(read_swc(p)) for p in glob.glob("data/allen/specimen_*/reconstruction.swc")}
X = pd.DataFrame.from_dict(feats, orient="index")
tab = tab.set_index("id").loc[X.index]
print(multi_target_baseline(X, tab[list(PASSIVE_TARGETS)], groups=tab["donor_id"].fillna(tab.index.to_series())))
print(within_type_partial_r2(X.to_numpy(), tab["rin"].to_numpy(), coarse_type_labels(tab), tab["donor_id"].fillna(tab.index.to_series())))
EOF
```

## Analysis tables

| Table | Grain | Key columns |
|---|---|---|
| `cells_with_features` | cell (specimen) | id, species, structure_layer_name, structure_area_abbrev, dendrite_type, transgenic_line, donor_id, reconstruction_type, rin, sag, tau, rheobase, ud_ratio, adaptation, fi_slope, vrest, latency, avg_isi, allen_morph_* |
| `ephys_recomputed` | cell | ap_width (ms, half-height at rheobase sweep), rin, sag, tau, rheobase, adaptation, fi_slope from `ephys_features.py`, plus Allen-table values for Bland-Altman |
| `morphometrics` | cell | 25 features from `morphometric_vector` (dend/basal/apical length, n_bif, n_tips, max_branch_order, max_path_dist, hull_volume, soma_radius, tortuosity, sholl_peak, ...) |
| `graphs` | cell | `.npz` with x (n_nodes × 15), edge_index, pos, node_type, globals |
| `patchseq_labels` | cell | t_type, met_type, subclass, dataset (mouse VISp GABA / mouse glut / human) |
| `predictions` | cell × target × model × experiment | y, yhat, fold, train_species, test_species, shots (few-shot), seed |
| `metrics` | target × model × experiment | r2, spearman, rmse, bootstrap CI, permutation p, transfer_gap, partial_r2_within_type, r2_type_only |

## Methods

1. **Assembly** (`scripts/download_data.py`, `src/morph2ephys/allen_fetch.py`): cells with reconstruction; join ephys features (`input_resistance_mohm`, `sag`, `tau`, `threshold_i_long_square` (rheobase), `upstroke_downstroke_ratio_long_square`, `adaptation`, `f_i_curve_slope`, `vrest`), morphology features, metadata (species, layer, dendrite type, Cre line, donor). AP half-width is recomputed from NWB sweeps with `ipfx`/own extractor (`ephys_features.py`) at the rheobase sweep.
2. **Graph construction** (`swc_graph.py`): nodes = SWC points (optionally resampled to 5 µm), edges = parent-child (both directions); node features = one-hot compartment type, radius, path distance, Euclidean distance, branch order, segment length, bifurcation/tip flags, unit direction; global scalars = soma radius, total lengths. Augmentation: rotation about the pial axis, small jitter, random subtree dropout (simulates cut dendrites).
3. **Baselines** (`baseline.py`): ridge on 25 morphometrics with nested GroupKFold by donor; multi-output; permutation null; within-type partial R² (residualise on t-type / Cre line / dendrite type).
4. **GNN** (`gnn.py`): 4-layer GraphSAGE-style mean aggregation (plain PyTorch, torch_geometric optional), global mean + max pooling, MLP head for all targets jointly (z-scored per species), Huber loss; early stopping on donor-grouped validation fold; ensembles of 5 seeds; optional self-supervised pre-training on NeuroMorpho graphs (GraphDINO-style) before regression fine-tuning.
5. **Transfer**: (a) zero-shot mouse → human; (b) human-only training; (c) mouse pre-train + human fine-tune of the head with 10/25/50 cells; (d) size-normalised inputs (divide by total dendritic length / soma-pia distance) to separate scale from shape effects.
6. **Biophysical baseline**: for each mouse cell with an all-active/perisomatic model of the same t-type or Cre line, simulate the long-square protocol in NEURON on the *target* cell's morphology with type-average parameters; extract the same features with `ipfx`. Compare prediction errors to the GNN on matched cells. For human, use mouse type-average parameters (tests the "channel densities differ" hypothesis).
7. **Attribution**: integrated gradients on node features, aggregated per compartment and per Sholl shell.

## Evaluation & statistics

- Primary metrics: out-of-fold R², Spearman rho and RMSE per target; transfer gap = R²_within − R²_transfer; few-shot curves with 20 random human subsets.
- Validation: GroupKFold by donor (human) / by mouse and Cre line (mouse) so no cell of the same animal appears in train and test; hyperparameters tuned in inner folds only; NeuroMorpho pre-training set filtered for overlap with Allen cells by NeuroMorpho archive name and neuron name.
- Nulls: permutation of ephys labels across cells (per target, 500 permutations); "size-only" null model (predict from total length and soma radius alone); type-only null (predict type mean).
- Multiple comparisons: Benjamini-Hochberg across targets × models; report per-target CIs from 1,000 cell-level bootstraps of the OOF predictions.
- Ephys QC: use Allen sweep QC flags; exclude cells with `vrest` > −50 mV or bridge/seal issues; feature extraction reproducibility checked against the Allen table (Bland-Altman) before using own extractor.
- Reconstruction QC: dendrite-only vs. dendrite+axon; cut-dendrite fraction (from the `reconstruction_type`/`apical` intactness flags) used as a covariate.

## Publishable angle

- **Headline**: "Dendritic tree GNNs recover passive electrophysiology from morphology and transfer mouse → human for geometry-dominated features but not for active ones; the mapping is largely between-type, with a measurable within-type component for R_in and tau; learned models transfer across species better than mouse-fitted biophysical parameters."
- Target venues: *PLoS Computational Biology*, *eLife*, *Journal of Neuroscience* (Systems/Circuits), *NeurIPS/ICLR workshop* (AI for Science) for the methods part, *Neuroinformatics*.
- Follow-ups: extend to Patch-seq MET-types as a multi-task target; use the encoder for EM-morphology (MICrONS) cells to impute ephys; active-learning selection of human cells to record.

## Risks, confounds & mitigations

- **Small human n with reconstructions** (~100-200): use few-shot design with repeated random subsets; report learning curves; pool Cell Types + Patch-seq human cells with dataset as a covariate.
- **Reconstruction truncation** in slices (cut apical tufts in human 350 µm slices) biases morphometrics: subtree-dropout augmentation; include intactness covariates; sensitivity analysis on cells with intact apical dendrite.
- **Recording-condition differences** (temperature, ACSF, holding) between mouse and human and between Cell Types vs. Patch-seq protocols: harmonise to a single protocol family (long square); include dataset/protocol indicator; do not transfer across protocol families.
- **Type confounding**: morphology predicts type, type predicts ephys; hence the within-type analysis (H2) is primary rather than raw R².
- **GNN overfitting** on a few hundred graphs: strong augmentation, ensembling, self-supervised pre-training on NeuroMorpho, and report ridge baselines with the same CV.
- **Biophysical baseline cost**: reuse existing Allen models; only simulate the long-square protocol at 3-5 amplitudes per cell.

## Milestones

- [ ] Assemble mouse/human tables with SWC + ephys features; QC; exact counts documented.
- [ ] Own ephys extractor validated against the Allen feature table (Bland-Altman on 50 cells).
- [ ] Ridge baseline with donor-grouped CV; permutation nulls; within-type partial R².
- [ ] SWC → graph pipeline and augmentation; GNN trained on mouse, OOF metrics vs. ridge (H1, H2).
- [ ] Transfer experiments mouse → human, zero/few-shot (H3).
- [ ] Biophysical baseline on matched cells (H4).
- [ ] Attribution analysis (H5).
- [ ] Preprint, code and model release.

## Repository layout

```
README.md                          this document
requirements.txt                   dependencies (allensdk, torch optional at import time)
data/README.md                     acquisition: allensdk Cell Types, Patch-seq/DANDI, biophysical models, NeuroMorpho
scripts/download_data.py           Allen cells + feature tables + SWC (+NWB); NeuroMorpho neocortical SWCs
src/morph2ephys/allen_fetch.py     AllenCellTypesFetcher (tables, SWC, NWB, long-square sweeps), target definitions
src/morph2ephys/swc_graph.py       SWC -> NeuronGraph (15 node features, bidirectional edges), augmentations, 25 morphometrics
src/morph2ephys/ephys_features.py  own long-square extractor (rin, sag, tau, rheobase, AP width, adaptation, f-I) + LIF-with-sag simulator
src/morph2ephys/baseline.py        ridge with GroupKFold, permutation null, transfer gap, few-shot curve, within-type partial R2
src/morph2ephys/gnn.py             plain-PyTorch GraphSAGE regressor, training loop with early stopping, fine-tuning, attributions
tests/test_morph2ephys.py          synthetic graph/ephys/regression tests (torch not required)
```

## Ethics / data-use notes

- Allen Institute data are released under the Allen Institute Terms of Use (citation required; non-commercial restrictions for some assets). Human tissue in the Allen datasets is de-identified neurosurgical tissue collected with consent; only aggregate donor metadata (age, sex, region, condition) is available and must not be used for re-identification.
- No credentialed datasets are used. Do not commit NWB/SWC files or caches; `data/` is git-ignored.
- If Patch-seq data are downloaded from DANDI, cite the dandiset and the source publication.
