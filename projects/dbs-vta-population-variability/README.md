# dbs-vta-population-variability

**How much of the uncertainty in the deep-brain-stimulation "volume of tissue activated" comes from the neurons themselves? A population model that replaces the single idealised axon with hundreds of real axonal reconstructions (mouse whole-brain single-neuron axons, human and primate cortical reconstructions from NeuroMorpho.org) placed in Lead-DBS lead fields, with a variance decomposition of activation thresholds over morphology, fibre diameter, placement and volume-conductor assumptions.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (lead geometries for Medtronic 3389/3387, Boston Scientific Cartesia and Abbott directional leads with point-source contact fields; CRRSS/Sweeney myelinated-axon model along arbitrary 3-D paths with bisection thresholds; SWC axon-path extraction, descriptors and random placement; probabilistic VTA radius and factorial variance decomposition).
- Difficulty: MSc thesis (starter) to PhD chapter (with FEM fields and MRG axons in NEURON). No GPU.
- Timeline: 6-9 months for the starter design (analytic fields, CRRSS axons); 12-18 months with OSS-DBS/Lead-DBS FEM fields, MRG axons and patient-specific validation.
- Compute: laptop for the analytic-field population sweeps (a threshold is ~0.2 s); a workstation (16+ cores) for 10^4-10^5 axon x lead x setting combinations with FEM potentials; NEURON only for the MRG calibration subset.

## Background

Clinical DBS programming and connectomic DBS analyses rely on the VTA: the region in which axons are assumed to fire in response to the stimulus. Lead-DBS v3 (Neudorfer et al., 2023, NeuroImage) computes it by thresholding a FEM electric field (default 0.2 V/mm) or by pathway activation modelling (OSS-DBS; Butenko et al., 2020, PLoS Comput Biol) with MRG-type axon models (McIntyre, Richardson & Grill, 2002, J Neurophysiol) placed on atlas streamlines (Petersen et al., 2019, Neuron). Methodological comparisons (Duffley et al., 2019, J Neural Eng; Gunalan, Howell & McIntyre, 2018, NeuroImage; Howell, Gunalan & McIntyre, 2019, Neuromodulation) show that different volume-conductor and axon-model choices shift single-axon thresholds by tens of percent, whereas population activation summaries move less. All of these use idealised straight or streamline-following axons of a single or a few diameters, so the biological variability of axonal geometry - branching, collaterals, tortuosity, terminal arbors - enters only through hand-built generative models (Bingham & McIntyre, 2022, J Neurophysiol, hyperdirect-pathway arbors grown from human histology; Anderson et al., 2018, Brain Stimul).

Meanwhile, whole-brain single-neuron axon reconstructions now exist at scale: the Janelia MouseLight project (Winnubst et al., 2019, Cell; > 1,000 neurons) and the SEU-ALLEN/BICCN full-morphology dataset (Peng et al., 2021, Nature; 1,741 neurons), many of them layer-5 pyramidal-tract neurons with subthalamic and thalamic collaterals, plus thousands of cortical, pallidal and thalamic reconstructions with axons on NeuroMorpho.org (Ascoli et al., 2007, J Neurosci; Akram et al., 2018, Sci Data) and human cortical neurons in the Allen Cell Types Database. Axon diameter distributions in human white matter are also measured (Liewald et al., 2014, Biol Cybern) and have been used to weight DBS field impact (2021, arXiv:2102.09956). Nobody has put the two together: a *population* of empirically shaped axons in DBS fields, with the resulting threshold distribution and a formal attribution of VTA uncertainty to morphology vs. diameter vs. placement vs. conductor model.

## The research gap

**What has been done**

- Single-axon and streamline-based pathway activation models (McIntyre et al., 2004, Clin Neurophysiol; Butson & McIntyre, 2006, Clin Neurophysiol; Gunalan et al., 2018; Butenko et al., 2020; Lead-DBS/OSS-DBS pipelines).
- Sensitivity of VTA/PAM to conductor model, encapsulation, anisotropy and diameter (Duffley et al., 2019; Howell & McIntyre, 2016, J Neural Eng; an in-vivo evaluation of DBS computational modelling methodologies, 2025) - reported ranges of roughly -24% to +47% for individual axon thresholds and about +/-10% for population activation.
- Generative arborisation models of the hyperdirect pathway in the STN region (Bingham & McIntyre, 2022) showing that terminal-arbor geometry substantially changes excitability.
- Optimisation of directional DBS under uncertainty (2025, arXiv:2506.13452) treats uncertainty in fields, not in neurons.
- Related morphology-population work for tES/TMS (Aberra et al., 2018, J Neural Eng; 2020, Brain Stimul) used ~5 Blue Brain morphologies per type, not empirical populations, and uniform fields rather than DBS point fields.

**What is missing (checked against 2023-2026 literature)**

1. Activation-threshold *distributions* from hundreds of empirical axon morphologies (not generative models) in realistic DBS lead fields, per lead type and contact configuration.
2. A variance decomposition (Sobol/ANOVA) of thresholds and of VTA radius into morphology, fibre diameter, placement/orientation, conductor/encapsulation model and axon-model factors, so that modelling effort can be spent where uncertainty actually lives.
3. A "probabilistic VTA": activation probability as a function of position with credible bands, compared with the deterministic 0.2 V/mm isosurface and with PAM on straight streamlines, for the same stimulation settings.
4. A morphology-informed surrogate (activating-function statistics of real axons -> threshold) that Lead-DBS-style pipelines could use without running NEURON per axon.

## Research questions / hypotheses

1. **H1 (morphological spread).** For a fixed fibre diameter, lead, contact and distance, activation thresholds across empirically shaped axons (branching, tortuous, with collaterals) have a coefficient of variation >= 25%, and the threshold distribution is right-skewed with a heavy tail of hard-to-activate morphologies (log-normal fit preferred to normal by AIC).
2. **H2 (variance attribution).** In a factorial design (morphology x diameter x distance x orientation x conductor model x pulse width), morphology explains more threshold variance than the conductor model (encapsulation/anisotropy scaling) at distances < 3 mm, whereas diameter dominates at all distances; first-order Sobol indices with bootstrap CIs.
3. **H3 (probabilistic vs. deterministic VTA).** The 50% activation-probability radius differs from the 0.2 V/mm isoline radius by more than 0.5 mm for at least one common setting (e.g. 3 mA, 60 us, monopolar), and the 5-95% band exceeds 1 mm.
4. **H4 (branch points and terminals).** Thresholds are lower for axons with a branch point or terminal within 1.5 mm of the active contact (ratio of medians <= 0.8), replicating Bingham & McIntyre (2022) with empirical arbors and extending it to pallidal/thalamic axons.
5. **H5 (surrogate).** A regression from activating-function summary statistics of the axon path (max, spatial extent above 0, number of local maxima, distance of max to a branch point) predicts log-threshold with R^2 >= 0.8 across morphologies and leads, in leave-morphology-out CV.
6. **H6 (directional leads).** Segmented contacts reduce the morphology-induced spread of thresholds in the targeted direction but increase it laterally, relative to ring-mode stimulation (variance ratio test).

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| NeuroMorpho.org (REST API v1) | Reconstructions with `structural_domains` including axon: mouse/rat/monkey/human neocortex, globus pallidus, subthalamic nucleus, thalamus; CNG.swc files + metadata (species, region, cell type, archive, software) | ~250k total; tens of thousands with axons | Open | https://neuromorpho.org/api/ ; https://neuromorpho.org/apiReference.html |
| Janelia MouseLight | Whole-brain single-neuron axon reconstructions (JSON/SWC via NeuronBrowser; many deposited to NeuroMorpho) | > 1,000 neurons | Open | https://ml-neuronbrowser.janelia.org |
| SEU-ALLEN / BICCN full morphologies (Peng et al., 2021) | Whole-brain mouse neurons incl. L5 PT with STN collaterals (SWC; mirrored on NeuroMorpho / BICCN portal) | 1,741 neurons | Open | https://neuromorpho.org ; https://www.biccn.org |
| Allen Cell Types Database (human & mouse) | Human cortical reconstructions with dendrites (axon initial segments) via `allensdk` | ~600 mouse, ~100 human reconstructions | Open | https://celltypes.brain-map.org |
| Lead-DBS electrode models | Lead geometries (contact length/spacing/diameter, segment angles) in `templates/electrode_models`; used to parametrise `dbs_popvta.lead_field.LEADS`; optional exported E-field/potential NIfTI from Lead-DBS/OSS-DBS | - | Open (GPL) | https://github.com/netstim/leaddbs |
| Petersen et al. (2019) pathway atlas (Lead-DBS bundled) | Streamlines for straight-axon baselines and for placing arbors along pathways | ~10 pathways | Open (with Lead-DBS) | https://github.com/netstim/leaddbs |
| Axon diameter distributions (Liewald et al., 2014) | Diameter priors for human white matter | table | Open (paper) | doi:10.1007/s00422-014-0626-2 |

No credentialed data are involved.

## Methods

1. **Morphology harvest** (`scripts/download_data.py`): NeuroMorpho `/api/neuron/select` with `structural_domains` containing axon and region in {neocortex, globus pallidus, subthalamic nucleus, thalamus}; CNG.swc download; MouseLight JSON->SWC conversion; Allen human cells via `allensdk`. `--sample` fetches metadata for a few species/regions and ~10 SWC files.
2. **Axon paths** (`src/dbs_popvta/morph_paths.py`): extract root-to-terminal axon paths and the full axon tree; descriptors (length, tortuosity, curvature, bends, branch-point and terminal positions); scaling of rodent axons to human calibre (diameter is a model factor, geometry is scaled by a factor in {1, 1.5, 2} as a sensitivity axis); random rigid placement at controlled perpendicular distance and orientation relative to the lead.
3. **Lead fields** (`lead_field.py`): four lead geometries; contacts discretised into surface point sources in a homogeneous, isotropic medium (default 0.2 S/m) with optional encapsulation scaling; current- or voltage-controlled; monopolar ring or segmented configurations. Upgrade path: load OSS-DBS/Lead-DBS FEM potentials (NIfTI) and interpolate (`FieldFromNifti`).
4. **Axon model** (`axon_model.py`): CRRSS/Sweeney mammalian node kinetics on nodes of Ranvier spaced 100 x fibre diameter along the resampled path, passive internodes (McNeal-type), implicit-Euler tree solver; extracellular potentials enter through the discrete activating current; thresholds by bisection for cathodic monophasic and charge-balanced biphasic pulses (60/90/120 us). Calibration subset in NEURON with MRG axons (Lead-DBS/OSS-DBS axon files) to map CRRSS thresholds to MRG thresholds.
5. **Population sweeps** (`population.py`): factorial design over morphology (n = 200+), diameter (2, 3.5, 5.7, 7.3, 10 um; and Liewald-weighted sampling), distance (0.5-6 mm), orientation (random, 5 draws), conductor model (nominal, +encapsulation 0.1 S/m x 0.5 mm, anisotropy proxy), pulse width; thresholds stored as long tables.
6. **Probabilistic VTA** (`activation_probability_curve`, `probabilistic_vta_radius`): P(activation | amplitude, r) and the 50%/5%/95% radii; deterministic comparators: 0.2 V/mm isoline (`efield_isoline_radius`) and straight-fibre PAM.
7. **Variance decomposition** (`variance_decomposition`): ANOVA/Sobol first-order and total indices on log-threshold and on VTA radius with bootstrap CIs.
8. **Surrogate** (H5): activating-function statistics -> log-threshold (ridge / gradient boosting), leave-morphology-out CV.
9. **Tools**: `numpy`, `scipy`, `pandas`, `scikit-learn`, `statsmodels`; optional `nibabel`, `allensdk`, `neuron`, `SALib`.

## Evaluation & statistics

- Primary estimands: threshold CV and log-normal parameters per lead/diameter/distance (H1); first-order and total Sobol indices with 1,000-resample bootstrap CIs (H2); differences in radii with CIs from morphology bootstrap (H3).
- Validation: (a) straight-fibre thresholds vs. published strength-distance curves for CRRSS/MRG axons; (b) CRRSS-vs-MRG calibration on 50 morphologies in NEURON (Bland-Altman); (c) optional comparison with intraoperative/evoked-potential thresholds from public reports (qualitative).
- Leakage / independence: morphologies from the same archive/lab are treated as clustered (archive as random effect; cluster bootstrap); the surrogate uses leave-morphology-out CV and is also tested across species.
- Multiple comparisons: H1-H3 confirmatory; lead x setting contrasts with Benjamini-Hochberg.
- Nulls: (i) shuffled geometry (path re-straightened with same length) to isolate the effect of tortuosity/branching; (ii) permutation of morphology labels within design cells for Sobol index significance.
- Sensitivity: node spacing rule (100 x D vs. MRG table), node length, 37 C kinetics, pulse polarity, voltage vs. current control, homogeneous vs. FEM fields.

## Publishable angle

- **Headline**: "With empirically shaped axons, DBS activation thresholds at a given distance vary by X-fold; morphology accounts for Y% of threshold variance (more than the conductor model at < 3 mm), the 50% activation radius differs from the 0.2 V/mm VTA by Z mm, and a four-statistic activating-function surrogate predicts thresholds with R^2 = W." A public table of per-morphology thresholds and the surrogate coefficients is a reusable resource for Lead-DBS-style pipelines.
- Target venues: *Journal of Neural Engineering*; *Brain Stimulation*; *NeuroImage*; *PLoS Computational Biology*; *IEEE TNSRE*.
- Follow-ups: patient-specific fields (Lead-DBS cohorts with shared VTAs); sweet-spot mapping with probabilistic VTAs; extension to spinal cord and vagus nerve stimulation; using the surrogate for closed-loop programming optimisation.

## Risks, confounds & mitigations

- **Species and scale mismatch**: mouse axons are smaller and shorter than human. Mitigation: geometry scaling factor as a design axis; human/primate cortical axons from NeuroMorpho and Allen as a cross-check; report sensitivity.
- **Truncated axons in slice reconstructions**: use `physical_Integrity` metadata; prefer whole-brain (MouseLight/SEU-ALLEN) for arbor completeness; treat cut ends as non-terminals in H4.
- **Diameter unknown in SWC**: fibre diameter is a model factor, not read from SWC radius (unreliable); Liewald-weighted sampling for population summaries.
- **Homogeneous field approximation**: analytic fields ignore tissue heterogeneity/anisotropy. Mitigation: FEM-field upgrade path and conductor-model factor in the Sobol design.
- **CRRSS vs. MRG kinetics**: CRRSS thresholds are known to be lower and less diameter-sensitive than MRG. Mitigation: NEURON calibration subset; report relative (not absolute) spreads as primary.
- **Archive/lab batch effects** in morphologies: cluster bootstrap and archive random effects.
- **Compute blow-up** with FEM x 10^5 axons: use the surrogate for screening and full simulation for a stratified subset.

## Milestones

- [ ] Metadata harvest; morphology inclusion rules; ~300 axons across cortex/GP/STN/thalamus + 50 whole-brain PT neurons.
- [ ] Straight-fibre validation of the CRRSS implementation against published strength-distance curves.
- [ ] Factorial threshold sweeps for four leads; long table released.
- [ ] H1-H2: threshold distributions and Sobol indices.
- [ ] Probabilistic VTA vs. 0.2 V/mm and PAM (H3); branch/terminal effects (H4).
- [ ] Surrogate model (H5); directional-lead analysis (H6).
- [ ] NEURON/MRG calibration on a subset; FEM-field replication with OSS-DBS exports.
- [ ] Preprint; code + threshold tables + surrogate release.

## Ethics / data-use notes

- All morphology data are open; cite NeuroMorpho.org, the depositing labs (each record carries `reference_pmid`/`reference_doi`), MouseLight, BICCN/SEU-ALLEN and the Allen Institute as required by their terms.
- Human reconstructions are de-identified surgical/post-mortem tissue curated by the depositing institutions; no patient data are used in the starter design. If patient-specific Lead-DBS fields are added later, use only data shared under appropriate consent/DUA and never commit them.
- Do not commit SWC files, field volumes or threshold tables (`data/` and `outputs/` are git-ignored).

## Related projects

- `morphology-dependent-stimulation` (uniform-field tES/TMS polarisation model over NeuroMorpho populations; this project is the DBS point-field, myelinated-axon, threshold-distribution counterpart and stays self-contained).
- `neuromorpho-scaling-laws` (archive/lab batch-effect modelling reused here for cluster bootstraps).
