# layer-fmri-7t-reproducibility — How reproducible are laminar fMRI profiles across labs, contrasts and analysis choices?

A quantitative reproducibility study of cortical-depth (laminar) fMRI profiles built only from open 7T datasets: variance decomposition of profile shape into lab, subject, session and analysis-pipeline components; a specification-curve multiverse over layering, ROI, smoothing, deveining and GLM choices; and an SNR-degradation experiment that uses real 7T data to estimate what laminar claims would survive at 3T-like sensitivity.

## Status / difficulty / timeline / compute

- Status: design + starter code (equidistant and equivolume depth computation on distance maps, laminar profile extraction and normalization, draining-vein forward/inverse model, profile similarity, variance components, chance-corrected reproducibility index, SNR degradation curves).
- Difficulty: MSc-to-early-PhD, 6-9 months. Requires comfort with sub-millimetre preprocessing (LayNii, ANTs/SPM) and mixed models.
- Compute: CPU only. Sub-mm preprocessing at 5× upsampling is memory-hungry (16-32 GB RAM per run) but not slow; the whole multiverse over ~10 datasets × 5 pipelines × 10 layer schemes is a few CPU-days. Storage: 300-500 GB.

## Background

Laminar fMRI at 7T resolves feedforward (middle-layer) from feedback (superficial/deep-layer) signals and has produced landmark claims: layer-specific input/output in M1 with VASO (Huber et al., 2017, Neuron), deep-layer top-down effects in V1 (Kok et al., 2016, Current Biology; Lawrence, Norris and de Lange, 2019, eLife; Muckli et al., 2015, Current Biology), depth-dependent tuning in auditory cortex (De Martino et al., 2015, PNAS), and layer-dependent working-memory activity in dlPFC (Finn et al., 2019, Nature Neuroscience). Two well-known confounds complicate interpretation: gradient-echo BOLD is biased toward superficial layers by ascending veins (Polimeni et al., 2010, NeuroImage; Kashyap et al., 2018, NeuroImage), and the depth coordinate itself depends on whether layers are defined equidistantly or by equal volume (Waehnert et al., 2014, NeuroImage; LayNii, Huber et al., 2021, NeuroImage). Biophysical models describe the draining-vein leakage (Markuerkiaga, Barth and Norris, 2016, NeuroImage; Havlicek and Uludağ, 2020, NeuroImage). Layer-resolved fMRI at 3T has been demonstrated with EEG-fMRI (Scheeringa et al., 2016, PNAS) but remains rare; reviews summarise the field (Norris and Polimeni, 2019, NeuroImage; a 2024 Psychoradiology review of ultra-high-field fMRI).

## The research gap

What has been done (2020-2026):

- Single-lab replications: a pre-registered, fully automated replication of Finn et al. (2019) in 21 participants at 7T (MPI CBS Leipzig, "Laminar fMRI in the human prefrontal cortex at 7T: a replication"); reports that laminar patterns reproduce across pulse sequences, days and subjects (Proceedings B, 2020, "Robust functional mapping of layer-selective responses in human lateral geniculate nucleus"/visual system); multi-contrast laminar fMRI (BOLD, VASO, and others) differentiating excitation and inhibition (Imaging Neuroscience, 2024); new contrasts such as magnetization-transfer laminar fMRI (NeuroImage, 2026); VASO in hippocampus and FFA (2025-2026 preprints).
- Open data: whole-brain layer-fMRI VASO+BOLD in one participant over 6 days (Kenshu dataset, OpenNeuro ds003216; Huber et al., 2023), V1 VASO (ds001547), and a growing set of OpenNeuro/Donders repository datasets.

What is specifically missing:

1. No cross-dataset, cross-lab quantitative reproducibility of laminar profile shape. Existing replications test one paradigm in one lab. Nobody has decomposed profile variance into lab, subject, session and pipeline components, nor asked whether between-lab variance exceeds between-subject variance for GE-BOLD versus VASO.
2. No analysis multiverse. Layering scheme (equidistant vs equivolume), number of depth bins (3-20), ROI definition (anatomical vs activation-based, the latter circular), smoothing, deveining strategy (none, linear depth detrend, spatial deconvolution) and GLM (canonical vs FIR) are each known to matter, but their joint effect on the sign and size of "superficial vs deep" contrasts has never been reported as a specification curve.
3. No empirical estimate of how laminar claims degrade toward 3T-like SNR. With so few open 3T sub-mm datasets, the honest approach is to degrade real 7T data (noise addition to target tSNR, resampling to 1.0-1.2 mm) and measure when profile-shape classification (double peak in M1, deep-layer feedback in V1, superficial WM effect in dlPFC) fails; this also yields "minimum runs at 3T" estimates.
4. No chance-corrected reproducibility index for laminar profiles that the community could report routinely.

## Research questions / hypotheses

1. RQ1 (within-lab reliability). Across sessions (Kenshu: 6 days; ds001547 runs) what are ICC(2,1) per depth bin and the concordance (Lin's CCC) of normalized profiles? H1: CCC > 0.8 within subject across days for VASO in M1/V1; GE-BOLD CCC is higher (superficial bias is itself reproducible) but shape-specific contrasts (deep vs middle) are less reliable.
2. RQ2 (cross-lab variance). For matched paradigm classes (finger tapping/M1, visual stimulation/V1, working memory/dlPFC) pooled across open datasets, what fraction of profile variance is between-lab? H2: lab > subject variance for GE-BOLD profiles; lab ≈ subject for VASO.
3. RQ3 (multiverse). Across ≥ 200 specifications (layering × bins × ROI × smoothing × deveining × GLM), what share preserves the published sign of the key laminar contrast? H3: < 60% for GE-BOLD superficial-vs-deep contrasts; > 80% for the VASO M1 double-peak; equivolume vs equidistant changes peak depth by ≥ 1 bin in curved ROIs.
4. RQ4 (SNR degradation). At what tSNR / voxel size does classification accuracy of the profile shape (against the full-SNR reference) fall below 80%? H4: accuracy drops below 80% at roughly one-third of 7T tSNR for GE-BOLD (i.e. 3T at 0.8 mm would need ~3-4× the runs) and earlier for VASO.
5. RQ5 (index validity). The chance-corrected reproducibility index (mean pairwise CCC minus its depth-shuffled null, rescaled) is unbiased with respect to the number of bins and SNR in simulations, and orders datasets consistently with ICC.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| Kenshu whole-brain layer-fMRI (Huber et al., 2023; OpenNeuro ds003216) | VASO + BOLD, 7T, one participant, 6 sessions, HCP audio-movie; whole-brain depth profiles, session reliability | 1 participant × 6 days (~100 GB) | Open (CC0) | https://openneuro.org/datasets/ds003216 |
| Layer VASO in the visual system (Huber; OpenNeuro ds001547) | High-resolution VASO/BOLD in V1, 7T; visual-stimulation profiles | few participants | Open | https://openneuro.org/datasets/ds001547 |
| Additional OpenNeuro laminar datasets | Any dataset returned by searching OpenNeuro for "laminar", "layer", "VASO", "cortical depth" (motor, visual, prefrontal paradigms; multiple labs/vendors) | 5-15 datasets | Open | https://openneuro.org/ |
| Donders Repository laminar datasets (Kok et al., 2016; Lawrence et al., 2019) | 7T GE-BOLD V1 with top-down paradigms | ~20-30 participants each | Free registration / data-use terms on the Donders Repository | https://data.donders.ru.nl/ |
| MPI CBS replication of Finn et al. (2019) | 7T dlPFC working-memory VASO/BOLD, 21 participants, pre-registered pipeline | 21 participants | Check availability (project page states data sharing); otherwise request from authors | https://www.cbs.mpg.de/neurophysics/finn-et-al-replication |
| Finn et al. (2019) dlPFC data | Original working-memory layer data (NIH) | ~10 participants | Shared by the authors (check OpenNeuro / paper data statement) | https://openneuro.org/ |

The script `scripts/download_data.py --list` prints the curated accession list and `--search` queries OpenNeuro for the keywords above so that new datasets are picked up.

## Methods

1. Preprocessing (per dataset, harmonised): motion correction (SPM realign or ANTs) without spatial smoothing; 5× in-plane upsampling of the functional grid for layering; VASO BOLD-correction (dynamic division) where applicable; T1-EPI or MP2RAGE segmentation (FreeSurfer/CAT12 + manual correction in the ROI); GM/WM/CSF boundary masks in EPI space.
2. Layering (`layerfmri_repro.layering`): equidistant depth from WM and pial distance maps; equivolume depth using the local-curvature annulus solution (Waehnert et al., 2014) with curvature from the segmentation; 3/6/10/20 bins. Production: LayNii `LN2_LAYERS` (equidist and equivol) as the reference implementation; our module reproduces both on phantoms and small ROIs.
3. Activation: GLM (nilearn/AFNI 3dDeconvolve) with canonical and FIR models; contrast maps per run.
4. Profiles (`layerfmri_repro.profiles`): mean contrast per bin in the ROI; normalization (z across depth, or peak-normalised); superficial-bias slope; double-peak detection; deveining via (a) none, (b) linear depth detrend, (c) spatial deconvolution with a leaky draining model (inverse of the forward model in the same module; parameters from Markuerkiaga et al., 2016).
5. ROIs: anatomical (M1 hand knob, V1 calcarine, dlPFC from HCP-MMP) vs activation-based with independent runs (to avoid circularity).
6. Reproducibility (`layerfmri_repro.reproducibility`): ICC per bin; CCC across acquisitions; nested variance components (lab / subject-within-lab / residual) with statsmodels MixedLM; chance-corrected reproducibility index; SNR degradation with noise injection and resampling.
7. Multiverse: enumerate specifications, fit each, plot the specification curve for the key contrast and the sign-consistency share.

Libraries: numpy, scipy, pandas, statsmodels, scikit-learn; nibabel, nilearn; LayNii (C++), ANTsPy, FreeSurfer/CAT12; AFNI (optional).

## Evaluation and statistics

- Reliability: ICC(2,1) with bootstrap CIs (over runs/sessions); CCC for profile shapes; Bland-Altman for peak depth.
- Variance components: MixedLM with lab random intercept and subject-within-lab variance component per bin and for the summary contrast; likelihood-ratio tests; report proportions with bootstrap CIs.
- Multiverse: specification curve with the median effect and the share of specifications with the published sign; permutation of condition labels within run (500×) to obtain the null specification curve (Simonsohn et al., 2020, Nature Human Behaviour).
- SNR degradation: accuracy vs tSNR curves with 200 noise realisations per level; logistic fit for the 80% point.
- Leakage prevention: activation ROIs from independent runs; degradation reference profiles from held-out sessions; no tuning of deveining parameters on the datasets being evaluated (parameters fixed from the literature or from the phantom).
- Multiple comparisons: three pre-registered primary outcomes (RQ2 lab-vs-subject variance ratio; RQ3 sign-consistency share; RQ4 80%-accuracy tSNR); the rest exploratory.

## Publishable angle

Headline: "Laminar profile shape is reproducible within a lab, but for GE-BOLD the between-lab variance exceeds between-subject variance and fewer than 60% of reasonable analysis specifications preserve the sign of superficial-vs-deep contrasts; VASO M1 double peaks are the most robust laminar finding, and 3T-like SNR would require several-fold more data." Plus a reusable reproducibility index and a public specification-curve toolkit.

Target venues: Imaging Neuroscience; NeuroImage; Human Brain Mapping; Magnetic Resonance in Medicine (for the SNR/3T component).

Follow-ups: (i) a living meta-analysis of laminar profiles hosted with the layer-fMRI community; (ii) prospective multi-site travelling-subject laminar study design informed by the variance components; (iii) extension to layer-resolved connectivity (Huber et al., 2021, Progress in Neurobiology) and to the lag/vascular questions in `fmri-hemodynamic-aging`.

## Risks, confounds and mitigations

- Few open datasets and heterogeneous paradigms: restrict cross-lab comparisons to matched paradigm classes; pre-specify inclusion; the Kenshu single-subject data serve reliability, not generalization.
- Segmentation and sub-mm registration errors dominate at 0.7-0.8 mm: manual QC in the ROI, report depth-shift sensitivity (±1 bin), and use both anatomical and functional ROIs.
- Vendor/sequence differences (VASO variants, 3D-EPI vs 2D) are confounded with lab; report them as descriptors, not causal factors.
- Degradation emulates thermal-noise limits only, not 3T-specific physiology (lower BOLD contrast, different vein sensitivity); state this clearly and calibrate the tSNR targets from published 3T sub-mm values.
- Circular ROI definitions in some original studies: we quantify the effect rather than judge the papers.
- Data-use terms of the Donders Repository and author-shared data may restrict redistribution; only derived profiles are released.

## Milestones

- [ ] Curate open laminar datasets (OpenNeuro search + Donders); document paradigms, contrasts, voxel sizes.
- [ ] Harmonised preprocessing and layering on Kenshu and ds001547; validate our layering against LayNii on the annulus phantom and on real ROIs.
- [ ] Within-lab reliability (RQ1).
- [ ] Cross-dataset pooling by paradigm class; variance components (RQ2).
- [ ] Multiverse specification curve (RQ3).
- [ ] SNR degradation (RQ4) and index validation (RQ5).
- [ ] Release profiles, index code and specification-curve notebooks; manuscript.

## Ethics / data-use notes

- All datasets are open or under repository data-use terms; cite each dataset and its DOI; do not redistribute raw data from repositories that forbid it (Donders, author-shared).
- No participant-level data are committed; `data/` and `outputs/` are ignored by git.
- No third-party LLM services are involved.
