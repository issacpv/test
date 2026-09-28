# cross-atlas-prediction-stability — An atlas-robustness index for connectome-based phenotype prediction: do the *findings* survive a change of parcellation?

Quantify, per phenotype, how much connectome-based predictions (ridge, CPM) and their edge-level/network-level findings depend on the choice of parcellation (Schaefer 100-1000, Glasser MMP, Gordon, Brainnetome, AAL, Craddock, DiFuMo, individualised parcels), by projecting predictive patterns from every atlas into a common vertex space, measuring subject-level prediction concordance, and estimating the accuracy inflation created by picking the best atlas post hoc — across HCP and open OpenNeuro resting-state datasets.

## Status / difficulty / timeline / compute

- Status: design + starter code (parcellation of vertex/voxel series, FC features, ridge/CPM with grouped nested CV, Haufe patterns, projection of node/edge weights to a common space, network aggregation, robustness metrics, post-hoc selection inflation, simulator with a ground-truth parcellation).
- Difficulty: MSc thesis (HCP + AOMIC; 6-9 months) to PhD chapter (adding individualised parcellations, ABCD/NKI and test-retest).
- Compute: parcellating ~1,000 HCP dense series with ~15 atlases and running ridge/CPM with nested CV and permutations is ~1-3 days on a 16-core workstation; storage for dense CIFTI series ~1-2 TB (or use the HCP_PTN1200 release plus parcellated series generated once with `wb_command`). No GPU.

## Background

Every connectome-based prediction begins with a decision that is rarely justified: the parcellation. Atlases differ in resolution (100-1000 nodes), construction (functional gradients, multimodal boundaries, anatomy, clustering), space (surface vs volume) and population. Benchmarks show that this decision changes accuracy modestly (Dadi et al., 2019, NeuroImage; Arslan et al., 2018, NeuroImage; Messé, 2020, NeuroImage for structure-function coupling), that it changes individual-difference estimates meaningfully (Bryce et al., 2021, NeuroImage: "an overlooked decision point"), and that it is one of several processing choices that yield inconsistent connectomes (Luppi et al., 2024, Nature Communications). Reproducibility work on brain-wide association studies (Marek et al., 2022, Nature) and on analytic variability (Botvinik-Nezer et al., 2020, Nature) shows that when the space of reasonable choices is wide, reported effects are selected from it, and the "vibration of effects" (Patel, Burford and Ioannidis, 2015, Journal of Clinical Epidemiology) becomes part of the result.

## The research gap

What has been done:

- Accuracy comparisons across atlases for demographics and cognition: Dadi 2019 (functional atlases with soft parcellations do best; ~few-percent differences), a 2024 Human Brain Mapping study ("Impact of brain parcellation on prediction performance in models of cognition and demographics") reporting that low-resolution functional parcellations generalise for age, sex, executive function and other NIH Toolbox measures, with different atlases favoured in adult vs developmental samples, and a 2022 bioRxiv comparison of 35 parcellations for age prediction ("Choice of parcellation atlas might not be too critical for connectomic analysis") finding differences ≤ 0.12 years of MAE.
- Feature scaling and feature-type comparisons for brain-behaviour prediction ("Comparing and scaling fMRI features for brain-behavior prediction", Imaging Neuroscience, 2025).
- Individualised parcellations improve prediction slightly (Kong et al., 2021, Cerebral Cortex) and test-retest reliability of parcels has been characterised.
- Spatial null models for comparing brain maps (Alexander-Bloch et al., 2018, NeuroImage; Markello and Misic, 2021, NeuroImage).

What is specifically missing:

1. Studies compare *accuracy* across atlases; almost none compare the *findings* — the edges, nodes or networks that carry the prediction. Because atlases have different node sets, feature maps are never compared in a common space. Projecting Haufe-transformed weights to the vertex/voxel level makes every atlas's finding comparable.
2. No per-phenotype robustness measure. Whether an atlas matters plausibly depends on the phenotype's reliability and effect size (age and sex vs. fluid intelligence vs. personality), and no study reports a robustness index that lets a reader know which results to trust.
3. No estimate of the accuracy inflation from post-hoc atlas choice (selecting the best of 10-15 atlases on the same data), nor of the sample size at which nested selection stops paying.
4. Subject-level concordance is unreported: two atlases can give the same r with different subjects mis-predicted, which matters for individual-level use.
5. Cross-dataset replication is missing: results come from HCP alone; the open AOMIC datasets (different scanner, TR and populations) and test-retest data (CoRR) are unused for atlas questions.

## Research questions / hypotheses

1. RQ1 (accuracy dispersion). Across 12-15 atlases, what is the range of out-of-sample r per phenotype? H1: for age and sex the range is < 0.05; for fluid cognition 0.05-0.15; for personality and psychopathology scores the range covers zero.
2. RQ2 (subject-level concordance). What is the mean pairwise Spearman correlation of subjects' out-of-fold predictions across atlases? H2: > 0.8 for age/sex; 0.4-0.7 for cognition; < 0.4 for low-reliability phenotypes, i.e. different atlases mis-predict different people.
3. RQ3 (finding concordance). After projection to fsLR vertices, how similar are predictive patterns across atlases (Pearson r, against a parcel-permutation and spin null)? H3: network-level profiles (Yeo-7/17) are concordant (r > 0.7) even when vertex-level maps are not (r ≈ 0.3-0.5); anatomical atlases (AAL) diverge most from functional ones.
4. RQ4 (resolution vs construction). Is the variance in accuracy and findings explained more by resolution (100 → 1000 nodes) or by atlas family (Schaefer vs Glasser vs Gordon vs Brainnetome)? H4: family explains more than resolution above ~200 nodes.
5. RQ5 (selection inflation). How much does "best-of-atlases" selection inflate r relative to nested selection or to an atlas-ensemble prediction? H5: inflation 0.03-0.08 in r at n = 400; ensembles (average of predictions across atlases) match or beat the best single atlas and remove the selection problem.
6. RQ6 (transportability). Do atlas rankings and the robustness index replicate between HCP and AOMIC (ID1000, PIOP1/2) and across CoRR test-retest sessions? H6: the robustness index replicates (rank correlation > 0.7 across datasets) even where absolute accuracies do not.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| HCP Young Adult S1200 | Dense resting-state CIFTI series (`*_Atlas_MSMAll_hp2000_clean.dtseries.nii`, 4 runs), HCP_PTN1200 parcellated series, NIH Toolbox and personality (unrestricted), family structure (restricted) | ~1,000 subjects | Free registration (+ Restricted Data application for family IDs) | https://db.humanconnectome.org |
| AOMIC ID1000 / PIOP1 / PIOP2 (OpenNeuro) | Resting-state and task fMRI with fMRIPrep derivatives, demographics, IQ (Raven / IST), personality, education | ID1000: ~900; PIOP1: ~200; PIOP2: ~200 | Open (OpenNeuro, CC0) | https://openneuro.org/datasets/ds003097 , ds002785 , ds002790 ; https://nilab-uva.github.io/AOMIC.github.io/ |
| CoRR | Test-retest resting sessions (atlas robustness of *reliability*) | ~1,600 subjects | Open (INDI) | http://fcon_1000.projects.nitrc.org/indi/CoRR/ |
| NKI-Rockland Sample (optional) | Lifespan resting fMRI with cognitive batteries | ~1,000 | Free registration (INDI) | http://fcon_1000.projects.nitrc.org/indi/enhanced/ |
| Atlases | Schaefer 100-1000 (7/17 networks), Craddock, DiFuMo, AAL, Harvard-Oxford (nilearn fetchers); Glasser MMP1.0 and Gordon 333 (BALSA); Brainnetome 246 (registration); Yeo-7/17 network labels | small | Open / free registration | https://nilearn.github.io , https://balsa.wustl.edu , https://atlas.brainnetome.org |

## Methods

1. Parcellation (`atlas_stability.parcellate`): mean time series per label from vertex/voxel series (surface atlases on fsLR-32k via `wb_command -cifti-parcellate` in production; the same operation is implemented on arrays here), Fisher-z FC, vectorisation; projection of node-level quantities to the common vertex space (`node_to_space`) and aggregation to Yeo networks by majority vote; a random contiguous parcellation generator for nulls and simulations.
2. Prediction (`atlas_stability.predict`): ridge (dual form, inner grouped CV for alpha) and CPM (Shen et al., 2017, Nature Protocols) with family-grouped outer folds, 20 repeats; out-of-fold predictions per subject; Haufe-transformed edge patterns per fold, node strength per fold.
3. Robustness (`atlas_stability.robustness`): accuracy dispersion (range, SD, coefficient of variation across atlases), subject-level prediction concordance (pairwise Spearman), vertex-level finding concordance (pairwise Pearson after projection, against a parcel-permutation null and, in production, spin nulls with `neuromaps`), network-profile concordance, and the atlas-robustness index ARI ∈ [0, 1] = mean of (1 − normalised dispersion), prediction concordance and finding concordance. Post-hoc selection inflation: best-of-atlases vs nested atlas selection vs atlas-ensemble, from a folds × atlases accuracy matrix.
4. Multiverse factors: atlas (family × resolution), feature (Fisher-z, partial correlation), GSR on/off, and model (ridge/CPM), analysed with a mixed model of accuracy on factors (subject folds as random effects) and specification curves.
5. Simulation (`atlas_stability.simulate`): vertex series with a ground-truth regional structure and a phenotype-dependent coupling between two regions; used to verify that the ARI is high when atlases respect the true regions and low when they split them.

Libraries: numpy, scipy, pandas, scikit-learn; nilearn (atlas fetchers, signal extraction), nibabel (CIFTI/GIFTI), Connectome Workbench (`wb_command`), neuromaps (spin nulls), optional `brainspace`.

## Evaluation and statistics

- Accuracy: out-of-fold Pearson r and R² (with 95 % CI from repeats); chance via 1,000 label permutations per atlas (family structure respected by permuting within exchangeability blocks).
- Concordance metrics with CIs from fold repeats; finding-concordance nulls: parcel permutation (preserves atlas geometry) and spin rotations.
- ARI reported per phenotype with a bootstrap CI over subjects and a null from permuted phenotypes (ARI is not defined for unpredictable phenotypes; report only when r > chance in ≥ 50 % of atlases).
- Selection inflation: paired difference between best-of and nested accuracies across repeats; sample-size curve (n = 100, 200, 400, 800).
- Leakage: atlases and features fixed before CV; alpha and CPM thresholds inside folds; atlas selection inside folds for the nested arm; family-grouped folds.
- Multiple comparisons: phenotypes pre-specified (HCP: age, sex, fluid, crystallised, total cognition, processing speed, NEO-FFI; AOMIC: age, sex, IQ, education, NEO); FDR across phenotypes for hypothesis tests.

## Publishable angle

Headline: "Atlas choice changes accuracy by < 0.05 in r for most phenotypes but changes *which subjects and which networks* carry the prediction; a per-phenotype atlas-robustness index separates findings that survive parcellation from those that do not, and post-hoc atlas selection inflates r by ≈ 0.05 at n = 400." Deliverables: the ARI table for ~10 phenotypes in two datasets, vertex-level consensus maps of predictive patterns (across atlases), and a recommendation to report atlas-ensemble predictions.

Target venues: NeuroImage, Imaging Neuroscience, Network Neuroscience, Human Brain Mapping; Nature Communications if the multi-dataset replication and individualised-parcellation arm are complete.

Follow-ups: extend to structural connectomes (parcellation drives tractography endpoints even more), to task-fMRI activation-based prediction, and to fairness (does atlas choice interact with demographic subgroup accuracy, cf. `connectome-prediction-fairness`).

## Risks, confounds and mitigations

- Node count and edge count scale with resolution, changing regularisation: alpha tuned per atlas; also report accuracy after PCA to a fixed number of components.
- Volume vs surface atlases differ in registration: run all atlases on both spaces where available (Schaefer has both); treat space as a factor.
- Finding-concordance at vertex level is inflated by atlas geometry: parcel-permutation and spin nulls; report z-scores relative to nulls.
- Phenotype reliability limits everything: use HCP retest to estimate reliability and report ARI against it; simulate the expected ARI at that reliability.
- Individualised parcellations need dense data: restrict this arm to HCP.
- Family structure and site (AOMIC has one site; HCP twins): grouped folds; exchangeability blocks for permutations.

## Milestones

- [ ] Access: HCP registration (+ restricted), download AOMIC derivatives, fetch atlases (nilearn/BALSA/Brainnetome).
- [ ] Parcellated FC for all atlases on HCP (4 runs) and AOMIC; QC (motion, coverage).
- [ ] Simulation study of ARI and selection inflation; pre-register phenotypes and atlases.
- [ ] RQ1-RQ3 on HCP; vertex-projection maps; network profiles.
- [ ] RQ4-RQ5: resolution vs family mixed model; nested selection and ensembles; sample-size curve.
- [ ] RQ6: AOMIC replication; CoRR reliability arm; individualised parcellation arm.
- [ ] Release ARI tables, consensus maps and code; write-up.

## Atlas set, phenotypes and key variables

| Atlas (family) | Resolutions | Space | Access |
|---|---|---|---|
| Schaefer 2018 (functional gradient) | 100, 200, 400, 600, 800, 1000 (7/17 networks) | fsLR + MNI | open (nilearn / CBIG) |
| Glasser MMP1.0 (multimodal) | 360 | fsLR | free registration (BALSA) |
| Gordon 2016 (boundary mapping) | 333 | fsLR + MNI | open / BALSA |
| Brainnetome (connectivity-based) | 246 | MNI + fsLR | free registration |
| AAL, Harvard-Oxford (anatomical) | 116 / 96 | MNI | open (nilearn) |
| Craddock 2012 (clustering) | 100-950 | MNI | open (nilearn) |
| DiFuMo (soft dictionary) | 64-1024 | MNI | open (nilearn) |
| Individualised MS-HBM parcels (HCP only) | 400 | fsLR | derived |

Phenotypes (pre-specified): HCP `Age_in_Yrs` (restricted), `Gender`, `CogFluidComp_Unadj`, `CogCrystalComp_Unadj`, `CogTotalComp_Unadj`, `ProcSpeed_Unadj`, NEO-FFI `NEOFAC_N/E/O/A/C`; AOMIC `age`, `sex`, `IQ` (Raven / IST), `education_level`, NEO-FFI. Subject-level variables: `subject`, `family_id` (HCP) / `site`, mean FD per run (QC covariate), `fc_<atlas>` edge vectors per atlas, out-of-fold `yhat_<atlas>`, per-fold Haufe patterns, Yeo-7 network assignment per node.

## Starter code map

| Module / function | What it does |
|---|---|
| `atlas_stability.parcellate.parcellate_timeseries`, `fisher_z_fc`, `vectorize_upper` | array-level parcellation and FC features |
| `parcellate.node_to_space`, `edge_weights_to_node_strength`, `majority_network_assignment`, `aggregate_nodes_to_networks` | projection of findings to the common vertex space and to networks |
| `parcellate.random_contiguous_parcellation` | random atlases for nulls and simulations |
| `atlas_stability.predict.ridge_cv_predict`, `cpm_cv_predict`, `prediction_accuracy`, `haufe_pattern` | grouped nested CV with out-of-fold predictions and per-fold patterns |
| `atlas_stability.robustness.accuracy_dispersion`, `prediction_concordance`, `finding_concordance` | the three ARI components (finding concordance with a parcel-permutation null) |
| `robustness.atlas_robustness_index`, `posthoc_selection_inflation` | ARI and best-of vs nested vs ensemble accuracy |
| `atlas_stability.simulate.simulate_vertex_dataset` | vertex series with a ground-truth parcellation and phenotype-coupled edge |
| `tests/test_atlas_stability.py` | parcellation/projection arithmetic, true atlas beats a mismatched atlas, ARI in [0, 1], selection inflation > 0 under the null |

Quick start:

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/download_data.py atlases --out data/atlases        # nilearn fetchers
python scripts/download_data.py aomic --dataset ds002785 --sample 5   # open OpenNeuro derivatives
```

## Ethics / data-use notes

- HCP open-access terms; restricted data (family IDs) never redistributed; AOMIC is CC0 but participants must not be re-identified; CoRR/NKI per INDI terms.
- No data are sent to third-party APIs; `data/` and `outputs/` are git-ignored; only group-level maps and tables are released.
