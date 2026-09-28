# virtual-mouse-brain-validation

**A multimodal, multi-site validation of connectome-based whole-mouse-brain models: fit Virtual-Mouse-Brain-style models built from the Allen mesoscale connectome and score them against open resting-state fMRI from 17 acquisition sites and against widefield calcium imaging of dorsal cortex in the same Common Coordinate Framework, across a connectome-construction multiverse and with spatially-informed null models.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (connectome normalisation/thresholding utilities, analytic linear (Ornstein-Uhlenbeck) and Hopf whole-brain models, FC/FCD metrics, degree- and distance-preserving nulls, parcellation of fMRI volumes and widefield frames).
- Difficulty: PhD-chapter level (dynamical modelling + neuroimaging data engineering). No GPU needed; CPU parallelism helps for parameter sweeps.
- Timeline: 9-12 months (2 months data assembly and preprocessing, 1 month connectome multiverse, 2-3 months model fitting and sweeps, 2 months nulls/statistics, 2 months writing).
- Compute: a 16-32 core workstation. The linear model has an analytic FC (one Lyapunov solve per parameter setting, milliseconds); Hopf/Wilson-Cowan simulations of ~200 regions for 10 min at 1 ms take ~1 min each; a full multiverse (12 connectomes x 3 models x 50 couplings x 20 noise seeds) is ~a day. Storage ~300 GB for the raw fMRI and widefield data.

## Background

The Virtual Mouse Brain (TVMB; Melozzi et al., 2017, *eNeuro*) brought The Virtual Brain (Sanz-Leon et al., 2015, *NeuroImage*) to the mouse by building whole-brain models from the Allen Mouse Brain Connectivity Atlas (Oh et al., 2014, *Nature*), later refined by regionalised (Knox et al., 2019, *Network Neuroscience*) and voxel-level (Coletta et al., 2020, *Science Advances*) connectome models. Melozzi et al. (2019, *PNAS*) showed that individual diffusion-MRI-informed structural connectomes, combined with Allen tracer directionality, predict individual mouse functional connectivity (FC) better than population connectomes. Mouse resting-state fMRI has since become a multi-centre, reproducible measurement (Grandjean et al., 2020, *NeuroImage*, 17 sites; awake mouse fMRI dynamics, Gutierrez-Barragan et al., 2022, *Current Biology*; standardised preprocessing with RABIES, Desrosiers-Gregoire et al., 2024, *Nature Communications*; rat consensus protocol, Grandjean et al., 2023, *Nature Neuroscience*). In parallel, widefield calcium imaging offers a direct, awake, cortex-wide neural readout (Musall et al., 2019, *Nature Neuroscience*; MacDowell & Buschman, 2020, *Current Biology*), with a multimodal widefield + behaviour dataset published on DANDI in 2025 (*Scientific Data*, 2025).

## The research gap

**What has been done**

- TVMB-style models validated against fMRI FC from one or two labs, usually under anaesthesia, with one connectome construction (Melozzi et al., 2017, 2019).
- Structure-function coupling in the mouse analysed statistically (Stafford et al., 2014, *PNAS*; Sethi et al., 2017, *Chaos*; Coletta et al., 2020) rather than with generative dynamical models.
- Whole-cortex models fitted to widefield calcium imaging exist in isolation, and whole-brain models fitted to fMRI exist in isolation; none has been scored on *both* readouts for the same connectome.
- Connectome construction choices (connection density vs. strength, ipsi/contra handling, thresholding, regionalisation, tracer vs. single-axon MouseLight; see the related project `mesoscale-vs-single-axon-connectivity`) are known to change graph statistics, but their effect on *simulated dynamics and empirical fit* has not been quantified.

**What is missing (checked against 2023-2026 literature)**

1. No study has fitted the same connectome-based model family to fMRI FC from many acquisition sites and reported how the optimal global coupling and the model-empirical fit vary across sites (i.e. whether "the mouse model fit" is a property of the model or of one lab's acquisition).
2. No study has validated whole-brain mouse models against an awake, direct neural readout (widefield calcium) in the same CCF space as the fMRI, to ask whether models that fit BOLD FC also fit calcium FC after accounting for the hemodynamic transform.
3. Model fit has not been benchmarked against spatially-informed nulls (connectomes rewired while preserving degree and the distance-dependence of connection weights), so it is unknown how much of the "connectome explains FC" result is due to spatial embedding alone.
4. The connectome multiverse has not been carried through to dynamics: which construction choices matter for fit, and do tracer-derived and single-axon-derived connectomes differ in predicted FC?

This project is self-contained; it reuses the connectome-comparison ideas of `mesoscale-vs-single-axon-connectivity` only as an input factor.

## Research questions / hypotheses

1. **H1 (site generalisation).** The global coupling that maximises FC fit on one site's data transfers to other sites with a loss in fit of less than 0.05 (Pearson r of upper-triangular FC) for at least 12 of 17 sites; sites with the lowest temporal SNR show the largest loss. Test: leave-one-site-out coupling selection; mixed model of fit on tSNR with site as random effect.
2. **H2 (cross-modality).** Models tuned on widefield calcium FC (dorsal cortical regions) predict BOLD FC of the same regions better than chance and better than the spatial null, but worse than models tuned on BOLD; the ranking of connectomes by fit is concordant between modalities (Kendall's tau > 0.5).
3. **H3 (nulls).** At least 30% of the empirical FC variance explained by the tracer connectome is also explained by the distance-preserving null connectome; the tracer-specific gain is largest for long-range and interhemispheric (homotopic) connections.
4. **H4 (multiverse).** Connectome construction explains more variance in fit than the choice of dynamical model (linear vs. Hopf vs. Wilson-Cowan); normalised connection density with contralateral projections included is the best-performing construction, and single-axon-derived connectomes fit worse than tracer-derived ones at equal density.
5. **H5 (dynamics beyond FC).** Only nonlinear models reproduce the functional-connectivity-dynamics (FCD) distribution of awake widefield data; the linear model matches static FC but not FCD (KS distance).

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Mouse Brain Connectivity Atlas | Regionalised normalised connection density/strength matrices (ipsi + contra) via allensdk `MouseConnectivityCache`; CCFv3 annotation | ~2,000 tracer experiments; 200-400 regions per hemisphere depending on level | Open (allensdk / API) | https://connectivity.brain-map.org , https://allensdk.readthedocs.io |
| Mouse_rest_multicentre (Grandjean et al., 2020) | Resting-state fMRI from 17 sites, ~15 scans per site, multiple anaesthesia protocols; BIDS | 255 subjects | Open (OpenNeuro ds001720; openneuro-py / datalad / AWS S3) | https://openneuro.org/datasets/ds001720 |
| Widefield calcium imaging multimodal dataset (*Scientific Data*, 2025) | Resting-state and task widefield GCaMP frames of dorsal cortex registered to the Allen dorsal map | tens of sessions, several mice | Open (DANDI; `dandi` CLI - see data/README.md for locating the dandiset) | https://dandiarchive.org |
| MouseLight single-neuron projections (Janelia) | Single-axon-derived region-to-region connectivity as an alternative connectome | >1,000 neurons | Open (ml-neuronbrowser downloads) | https://ml-neuronbrowser.janelia.org |
| Allen CCFv3 (Wang et al., 2020, *Cell*) | Annotation volume, region hierarchy, dorsal cortical flat map | atlas files | Open | https://atlas.brain-map.org |
| RABIES preprocessing (Desrosiers-Gregoire et al., 2024) | Standardised rodent fMRI preprocessing and confound regression to CCF space | software | Open | https://github.com/CoBrALab/RABIES |

No credentialed data are involved.

## Methods

1. **Connectome multiverse** (`src/vmb_validation/connectome.py`). From allensdk: structure-level matrices at the "summary structures" level, ipsilateral and contralateral, as normalised connection density (NCD) and normalised connection strength; symmetrisation (max / mean / none); thresholding by density (5-50%) and by injection-count confidence; log-transform; optional MouseLight-derived matrix. Each construction is a named factor level.
2. **Empirical FC** (`src/vmb_validation/parcellate.py`). fMRI: RABIES preprocessing to CCF, parcellation to the same region list (mean time series), band-pass 0.01-0.1 Hz, FC = Pearson correlation; FCD from 60 s sliding windows. Widefield: hemodynamic-corrected dF/F frames (dataset-provided), registered to the Allen dorsal map, parcellated to dorsal cortical regions, FC and FCD as above (with calcium-appropriate windows).
3. **Models** (`src/vmb_validation/models.py`). (a) Linear stochastic model dx = (-x + G W x) dt + noise with analytic covariance from the Lyapunov equation (fast fitting; Galan-style); (b) Hopf/Stuart-Landau oscillators near bifurcation with heterogeneous intrinsic frequencies (Deco and colleagues' formulation), simulated with Euler-Maruyama; (c) optional Wilson-Cowan / reduced Wong-Wang via TVB. BOLD: Balloon-Windkessel (TVB implementation) for fMRI comparisons; for widefield, a calcium kernel (exponential rise/decay) instead.
4. **Fitting**. Sweep global coupling G (and Hopf bifurcation parameter) to maximise FC similarity; report optimum and fit; leave-one-site-out for H1.
5. **Nulls** (`src/vmb_validation/nulls.py`). Degree-preserving rewiring (Maslov-Sneppen) for directed weighted graphs; distance-preserving weight permutation within distance bins (geometric null); random weight shuffle; each null is simulated with the same pipeline.
6. **Statistics** (`src/vmb_validation/fc_metrics.py`). FC similarity (Pearson on upper triangle, homotopic FC correlation, FC-SC coupling); FCD KS distance; mixed models over sites; Kendall's tau over connectome rankings.
7. **Tools**: `allensdk`, `numpy`, `scipy`, `pandas`, `nibabel`, `statsmodels`, `tvb-library` (optional), `openneuro-py`/`datalad`, `dandi`, `RABIES` (container).

## Evaluation & statistics

- Primary estimands: FC fit r per (connectome, model, site/session) with the coupling selected out-of-sample; fit gain over the spatial null; FCD KS distance.
- Validation scheme: leave-one-site-out (fMRI) and leave-one-session-out (widefield) for coupling selection; no parameter is tuned on the data it is scored on.
- Nulls: spatial/degree nulls as above (500 realisations each; simulated, not just correlated); for the FC-fit statistic, the null distribution is the fit obtained with null connectomes.
- Multiple comparisons: five pre-registered hypotheses; multiverse results summarised as variance components (ANOVA over factors) rather than per-cell p-values.
- Confounds: anaesthesia protocol (site-level covariate), motion/tSNR, global signal regression (both with/without), hemispheric asymmetry from injection-side conventions (mirror-averaged connectomes as sensitivity), hemodynamic model parameters (sweep).
- Reproducibility: every multiverse cell is a deterministic function of (construction, model, seed); results cached as parquet.

## Publishable angle

- **Headline**: "Connectome-based mouse brain models generalise across 17 fMRI sites with a coupling loss of X, fit awake widefield calcium FC at r = Y, and Z% of their explanatory power is reproduced by a distance-preserving null; connectome construction matters more than the choice of dynamical model." A public benchmark table (connectome x model x site) and the code to extend it.
- Target venues: *PLoS Computational Biology*, *NeuroImage*, *Network Neuroscience*, *eLife*.
- Follow-ups: individual-animal models with dMRI (Melozzi 2019 design) on the multi-site data; perturbation validation using chemogenetic/optogenetic fMRI datasets on OpenNeuro (e.g. optogenetic stimulation datasets) - predicting stimulation-evoked FC changes from the fitted model; extension to rat with the StandardRat consensus data.

## Risks, confounds & mitigations

- **Widefield covers only dorsal cortex**: the cross-modality comparison is restricted to the ~20-30 dorsal cortical regions; whole-brain models are still simulated in full and scored on the overlap.
- **Hemodynamic vs. calcium transforms** are different filters; compare after transform-specific forward models and also on transform-invariant statistics (FC rank correlations).
- **Anaesthesia heterogeneity across sites** is a feature (H1) but also a confound; report per-protocol subgroup results.
- **Registration of widefield to the Allen dorsal map** is approximate: use the dataset's provided registration and test sensitivity to region erosion.
- **Model degeneracy**: multiple (G, noise) combinations give similar FC; report fit surfaces and identifiability, not just optima.
- **Data volume**: use RABIES outputs at 200 um and store only parcellated time series.

## Milestones

- [ ] Connectome multiverse table built from allensdk (with MouseLight alternative); graph statistics per construction.
- [ ] fMRI multi-site data preprocessed (RABIES) and parcellated; site QC table (tSNR, motion).
- [ ] Widefield sessions parcellated to the dorsal map; FC/FCD per session.
- [ ] Linear model analytic fits across the multiverse (fast pass); H3 nulls.
- [ ] Hopf model sweeps; H4 variance decomposition; H5 FCD.
- [ ] Leave-one-site-out generalisation (H1); cross-modality (H2).
- [ ] Preprint, benchmark table, code release.

## Quick start

```bash
cd projects/virtual-mouse-brain-validation
pip install -r requirements.txt
python scripts/download_data.py --ccf --resolution 100     # CCF annotation (plain HTTP)
python scripts/download_data.py --allen --sample           # connectome matrices (needs allensdk)
python scripts/download_data.py --openneuro --sample       # one subject of ds001720 (needs openneuro-py)
PYTHONPATH=src pytest -q                                   # analytic-vs-simulated FC, Hopf, nulls, parcellation
```

Fast multiverse pass with the linear model (analytic FC, no simulation):

```python
import numpy as np
from vmb_validation.connectome import load_npz_connectome, bilateral, apply, multiverse, spectral_normalise
from vmb_validation.models import linear_fc, max_stable_coupling, coupling_sweep
from vmb_validation.fc_metrics import fc_similarity, fit_gain_over_null
from vmb_validation.nulls import distance_preserving_permutation

ipsi = load_npz_connectome("data/allen/connectome_ncd_ipsi.npz")["matrix"]
contra = load_npz_connectome("data/allen/connectome_ncd_contra.npz")["matrix"]
fc_emp = np.load("data/parcellated/fc_site01_mean.npy")          # empirical FC on the same region list
coords = np.load("data/allen/centroids_bilateral.npy")
rng = np.random.default_rng(0)
for c in multiverse():
    W = spectral_normalise(apply(bilateral(ipsi, contra), c))
    gs = np.linspace(0.05, 0.95, 19) * max_stable_coupling(W)
    obs = coupling_sweep(W, fc_emp, gs, fc_similarity, model="linear")
    null = [coupling_sweep(distance_preserving_permutation(W, coords, rng), fc_emp, gs, fc_similarity)["best_score"]
            for _ in range(50)]
    print(c.name(), obs["best_G"], obs["best_score"], fit_gain_over_null(obs["best_score"], null))
```

## Pre-registered analysis table

| # | Unit of analysis | Primary outcome | Estimand / test | Decision rule | Confirmatory / exploratory |
|---|---|---|---|---|---|
| H1 | site (17) | FC fit r with coupling selected on other sites | leave-one-site-out; mixed model of fit loss on tSNR | loss < 0.05 in >= 12 sites | confirmatory |
| H2 | session (widefield) x site (fMRI), dorsal regions | FC fit r across modalities | fit of widefield-tuned model on BOLD vs. spatial null; Kendall's tau of connectome ranking | fit > null (p < 0.01); tau > 0.5 | confirmatory |
| H3 | connectome construction | fit gain over distance-preserving null | 500 null connectomes simulated | null explains >= 30% of variance explained; gain largest for homotopic edges | confirmatory |
| H4 | multiverse cell (construction x model) | FC fit r | variance components (ANOVA) | construction share > model share | confirmatory |
| H5 | session | FCD KS distance | linear vs. Hopf vs. Wilson-Cowan | only nonlinear models reach KS < 0.2 | confirmatory |
| S1 | site | optimal coupling vs. anaesthesia protocol | descriptive | - | exploratory |

## Key references

- Melozzi F, Woodman MM, Jirsa VK, Bernard C (2017) The Virtual Mouse Brain: a computational neuroinformatics platform to study whole mouse brain dynamics. *eNeuro*.
- Melozzi F et al. (2019) Individual structural features constrain the mouse functional connectome. *PNAS*.
- Sanz-Leon P, Knock SA, Spiegler A, Jirsa VK (2015) Mathematical framework for large-scale brain network modeling in The Virtual Brain. *NeuroImage*.
- Oh SW et al. (2014) A mesoscale connectome of the mouse brain. *Nature*.
- Knox JE et al. (2019) High-resolution data-driven model of the mouse connectome. *Network Neurosci*.
- Coletta L et al. (2020) Network structure of the mouse brain connectome with voxel resolution. *Sci Adv*.
- Harris JA et al. (2019) Hierarchical organization of cortical and thalamic connectivity. *Nature*.
- Grandjean J et al. (2020) Common functional networks in the mouse brain revealed by multi-centre resting-state fMRI analysis. *NeuroImage* (OpenNeuro ds001720).
- Grandjean J et al. (2023) A consensus protocol for functional connectivity analysis in the rat brain. *Nat Neurosci*.
- Desrosiers-Gregoire G et al. (2024) A standardized image processing and data quality platform for rodent fMRI. *Nat Commun* (RABIES).
- Gutierrez-Barragan D et al. (2022) Unique spatiotemporal fMRI dynamics in the awake mouse brain. *Curr Biol*.
- Musall S, Kaufman MT, Juavinett AL, Gluf S, Churchland AK (2019) Single-trial neural dynamics are dominated by richly varied movements. *Nat Neurosci*.
- MacDowell CJ, Buschman TJ (2020) Low-dimensional spatiotemporal dynamics underlie cortex-wide neural activity. *Curr Biol*.
- Stafford JM et al. (2014) Large-scale topology and the default mode network in the mouse connectome. *PNAS*.
- Sethi SS, Zerbi V, Wenderoth N, Fornito A, Fulcher BD (2017) Structural-connectome topology relates to regional BOLD signal dynamics in the mouse brain. *Chaos*.
- Burt JB, Helmer M, Shinn M, Anticevic A, Murray JD (2020) Generative modeling of brain maps with spatial autocorrelation. *NeuroImage*.
- Wang Q et al. (2020) The Allen Mouse Brain Common Coordinate Framework: a 3D reference atlas. *Cell*.

## Ethics / data-use notes

- All animal data were collected under the original institutions' approvals; this project performs secondary analysis only.
- OpenNeuro data are CC0; DANDI datasets carry their own licence (check the dandiset metadata); Allen data are under the Allen Institute terms of use. Cite every dataset paper.
- Do not commit raw or preprocessed imaging data; `data/` and `outputs/` are git-ignored; keep parcellated time series only.
