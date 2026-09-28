# retinal-prosthesis-morphology-models

**Type-resolved population models of retinal ganglion cell activation by epiretinal and subretinal prostheses: hundreds of typed RGC reconstructions (Eyewire, Sümbül/Sanes, NeuroMorpho.org) run through a biophysical model to derive per-type threshold distributions, their morphological determinants, the attainable type selectivity, and the dominant modelling uncertainty (axon initial segment geometry) - validated against ex vivo multi-electrode threshold distributions.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (SWC compartmentaliser with morphometrics and synthetic axon/AIS generator, disk/point epiretinal-subretinal electrode fields, Fohlmeister-Miller-type multicompartment RGC model with implicit tree solver and bisection thresholds, type-level statistics, morphological-determinant mixed models, selectivity and empirical-comparison utilities).
- Difficulty: MSc thesis (starter, numpy model) to PhD chapter (NEURON/full FM kinetics, degenerate-retina fields, human RGCs). No GPU.
- Timeline: 6-9 months starter; 12-15 months with NEURON calibration, FEM retina fields and validation against published MEA data.
- Compute: laptop for ~1,000 cells x few settings with the numpy model (10-30 s per threshold at 1-2k compartments); a workstation for full factorial designs; NEURON for the calibration subset.

## Background

Epiretinal (Argus II-type, and current high-density MEA research systems) and subretinal (PRIMA, Alpha-AMS) prostheses restore crude vision by electrically activating surviving retinal ganglion cells (RGCs) or the inner retinal network. Two problems dominate current research (Palanker and colleagues, Annu Rev Vis Sci, 2025 overview): (i) unintended activation of passing axon bundles producing elongated percepts (Beyeler et al., 2019, Sci Rep, axon-map model in `pulse2percept`), and (ii) the lack of cell-type selectivity - ON and OFF cells, and the ~40 morphological RGC types, are activated together. Cellular-resolution epiretinal stimulation in primate and human ex vivo retina (Grosberg et al., 2017, J Neurophysiol; Madugula et al., 2022, J Neural Eng) shows that thresholds differ between cells and types and provides empirical threshold distributions that models should reproduce.

Biophysical RGC models (Fohlmeister & Miller, 1997, J Neurophysiol; Rattay & Resatz, 2004, IEEE TBME; Werginz, Fried & Rattay, 2014, Neuroscience; Guo et al., 2016, J Neural Eng; Kameneva et al., 2016, J Neural Eng; Paknahad et al., 2020, IEEE TNSRE; 2021, Sci Rep; Rattay et al., 2023, J Neural Eng) established the roles of the sodium-channel band / AIS, the soma-to-axon geometry, pulse waveform and frequency in activation and selectivity. Werginz, Raghuram & Fried (2020, J Neural Eng) related thresholds of alpha RGCs to their measured morphology (AIS length and distance, soma size). Esler et al. (2018, PLoS ONE) used NeuroMorpho RGC reconstructions to derive axon-orientation statistics. Every one of these studies used from one to about a dozen morphologies; the population-level question - how much thresholds vary *within* and *between* the actual morphological types, and what that implies for selectivity - remains open, even though ~800 typed mouse RGC reconstructions are public (Bae et al., 2018, Cell, Eyewire museum, ~400 cells in 47 types; Sümbül et al., 2014, Nat Commun, ~380 cells; Coombs et al., 2006, Neuroscience) and the field is moving to conditional-generative optimisation of stimuli (2024, arXiv:2403.04884) that needs a realistic response model.

## The research gap

**What has been done**

- Single- or few-morphology biophysical models of epiretinal/subretinal stimulation with detailed treatment of the AIS/SOCB, stimulus polarity, pulse duration and frequency (Rattay & Resatz 2004; Werginz et al. 2014; Guo et al. 2016; Kameneva et al. 2016; Paknahad et al. 2020, 2021; Rattay et al. 2023).
- Morphology-threshold relationships in a small set of alpha cells with measured AIS (Werginz, Raghuram & Fried, 2020) and AIS scaling with soma size (Raghuram, Werginz & Fried, 2019, Front Cell Neurosci).
- Population responses of *identical* model cells at different positions (Tsai et al., 2012, PLoS ONE) and multiscale retina-plus-electrode models (Loizos et al., 2016/2018, IEEE TNSRE) with a handful of cell models.
- Phenomenological axon-map percept models fitted to patients (Beyeler et al., 2019; `pulse2percept`).
- Empirical threshold distributions across ON/OFF parasol and midget cells from high-density MEAs in primate and human retina (Grosberg et al., 2017; Madugula et al., 2022).
- A 2025 IEEE TNSRE study on RGC stimulation parameters (doi:10.1109/TNSRE.2025.3627290) continues the parameter-optimisation line with limited morphological sampling.

**What is missing (checked against 2023-2026 literature)**

1. Per-type activation-threshold distributions from hundreds of real, typed RGC reconstructions under epiretinal and subretinal fields, with within-type spread quantified.
2. A statistical model of the morphological determinants of threshold (soma size, dendritic field, stratification depth, dendritic length, axon trajectory relative to the electrode, AIS distance/length) across types, with type as a random effect.
3. An explicit treatment of AIS geometry uncertainty (unknown for nearly all reconstructions) as a propagated uncertainty, and the ranking of that uncertainty against morphological variability and electrode geometry.
4. A quantitative selectivity ceiling: given within-type variability, how well can any waveform separate types, and does the model reproduce the empirical ON/OFF parasol vs midget threshold ordering and spread of Grosberg 2017 / Madugula 2022?

## Research questions / hypotheses

1. **H1 (within-type spread).** For a 100 um epiretinal disk 30 um above the inner limiting membrane and a 100 us cathodic-first biphasic pulse, thresholds within a morphological type have CV >= 20%, and between-type differences of medians (e.g. alpha vs small-field types) are of the same order as within-type spread (variance ratio between/within in [0.5, 2]).
2. **H2 (morphological determinants).** In a mixed model of log-threshold with type random intercepts, soma diameter (negative coefficient) and AIS distance from the soma (positive when the electrode is above the soma) explain >= 40% of the between-cell variance; dendritic field diameter explains less than 10% for epiretinal but more for subretinal stimulation (network-free direct activation of dendrites).
3. **H3 (AIS uncertainty dominates).** Varying AIS distance (20-60 um) and length (20-40 um) within published ranges changes a cell's threshold more (median |delta log threshold|) than replacing the cell by another cell of the same type; first-order variance fractions AIS > morphology within type.
4. **H4 (selectivity ceiling).** For any pair of types, the separability of threshold distributions (AUC of the pairwise comparison) is < 0.8 for short pulses, and increases for long pulses (> 1 ms) only for pairs differing in soma size; the empirical ON/OFF parasol vs midget ordering (Grosberg 2017) is reproduced in sign.
5. **H5 (axon-bundle activation).** With synthetic axon trajectories toward the optic disc (Beyeler-style axon map), the fraction of cells activated via their distal axon rather than their AIS exceeds 30% for electrodes over the nerve-fibre layer, and drops below 10% with subretinal placement.
6. **H6 (scaling to human).** After scaling mouse morphologies to human soma/dendritic dimensions (or using the few human RGC reconstructions available), the predicted threshold distribution matches the Madugula 2022 distribution up to a global scale factor (KS test on standardised thresholds, p > 0.05).

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| Eyewire Museum (Bae et al., 2018) | Mouse RGC reconstructions with 47 morphological types, stratification profiles, soma positions (SWC/OBJ export) | ~400 cells | Open | https://museum.eyewire.org |
| Sümbül et al. (2014) mouse RGC reconstructions | Typed (genetic lines) SWC with IPL depth registration; deposited to NeuroMorpho | ~380 cells | Open | https://neuromorpho.org (search cell type "ganglion", species mouse) |
| NeuroMorpho.org retinal ganglion cells (REST API) | All species' RGC reconstructions (mouse, rat, cat, rabbit, primate, human where available) with metadata (archive, software, integrity) | ~1,500 | Open | https://neuromorpho.org/api/ |
| Allen / other human RGC reconstructions (sparse) | Human RGC morphologies where deposited | tens | Open | https://neuromorpho.org |
| Grosberg et al. (2017); Madugula et al. (2022) threshold data | Empirical threshold distributions per type (tables/figures in papers; raw data on request) | hundreds of cells | Open (paper) / request | doi:10.1152/jn.00750.2016 ; doi:10.1088/1741-2552/aca5b5 |
| `pulse2percept` axon map | Axon trajectories toward the optic disc for synthetic axons | - | Open (BSD) | https://github.com/pulse2percept/pulse2percept |
| Rattay / Werginz / Paknahad published model parameters | Channel densities, AIS ranges, conductivities | - | Open (papers; ModelDB entries) | https://modeldb.science |

No credentialed data are involved.

## Methods

1. **Harvest** (`scripts/download_data.py`): NeuroMorpho `/api/neuron/select` with `cell_type` containing "ganglion" and `brain_region` "retina"; CNG.swc download; Eyewire museum export (manual/URL list). `--sample` fetches metadata for 3 species and ~10 SWC files.
2. **Morphology processing** (`src/rgc_prosthesis/swc_morph.py`): compartmentalise SWC into <= 10 um cylinders; morphometrics (soma diameter, dendritic field diameter via convex hull, total length, branch points, stratification depth and spread in IPL z, asymmetry); axon synthesis when absent (hillock -> AIS at 20-60 um -> distal axon in the nerve-fibre layer toward the optic disc, direction from the axon map or a fixed angle); AIS parameters are explicit design factors.
3. **Electrode fields** (`electrode.py`): disk electrode on a semi-infinite homogeneous medium (Wiley-Webster closed form) or point source; epiretinal (above the NFL) and subretinal (below the photoreceptor layer) placements by z offset; pulse waveforms (cathodic/anodic monophasic, charge-balanced biphasic, 50 us - 4 ms). Upgrade path: layered-retina/FEM potentials via NIfTI or COMSOL export.
4. **Biophysical model** (`rgc_model.py`): Fohlmeister-Miller Na/K/leak kinetics with region-specific densities (dendrite, soma, hillock, AIS, distal axon), balanced leak reversal for a stable rest, implicit-Euler tree solver with extracellular potentials as activating currents, propagated-spike detection at the distal axon, bisection thresholds. Calibration subset in NEURON with the full five-channel FM model (ModelDB) and the Werginz 2020 parameters.
5. **Population sweeps**: cells x AIS configurations (3 x 3) x electrode geometry (disk 50/100/200 um; epiretinal 20/50/100 um heights; subretinal) x pulse (50, 100, 500, 1000, 4000 us) x electrode lateral offset (over soma, over dendrites, over axon).
6. **Statistics** (`population_stats.py`): per-type summaries; mixed models of log-threshold (statsmodels MixedLM, type random intercept; archive random effect for NeuroMorpho cells); variance fractions for AIS vs morphology; pairwise selectivity AUC; comparison with empirical distributions after a global scale (KS, QQ slope); activation-site classification (AIS vs distal axon vs dendrite).
7. **Tools**: `numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn`; optional `neuron`, `pulse2percept`, `nibabel`.

## Evaluation & statistics

- Primary estimands: within-type CV and between/within variance ratio (H1), mixed-model coefficients with 95% CIs and marginal/conditional R^2 (H2), AIS-vs-morphology variance fractions with bootstrap CIs (H3), pairwise AUC matrix (H4).
- Validation: (a) straight-axon and single-cell thresholds vs published values (Rattay 2004; Werginz 2014/2020) for the same geometry; (b) numpy-model vs NEURON-FM calibration (Bland-Altman on log threshold) for 50 cells; (c) ordering and spread vs empirical MEA data (H4, H6).
- Independence / leakage: cells from the same dataset/archive are clustered (random effects; cluster bootstrap); the empirical comparison uses a single global scale factor fitted on one type and tested on the others.
- Multiple comparisons: H1-H4 confirmatory in fixed order; determinants beyond the pre-specified set and all pairwise type contrasts use Benjamini-Hochberg.
- Nulls: shuffle type labels across cells (between-type differences); re-place each cell's electrode at random lateral offsets (site effects); straightened dendrites of equal length (geometry vs size).
- Sensitivity: channel densities +/-30%, conductivity 0.1-1 S/m, degenerate-retina thickness changes, temperature, dt.

## Publishable angle

- **Headline**: "Across N typed RGC reconstructions, epiretinal activation thresholds vary X-fold within a type; soma size and AIS geometry explain Y% of the variance while dendritic morphology matters only for subretinal stimulation; AIS uncertainty alone exceeds within-type morphological variability; and the resulting selectivity ceiling (AUC < Z for short pulses) matches the ON/OFF parasol-midget separation measured ex vivo." Deliverable: an open per-cell threshold table and fitted determinant model usable as a realistic response prior for stimulus-optimisation pipelines.
- Target venues: *Journal of Neural Engineering*; *IEEE TNSRE*; *PLoS Computational Biology*; *Journal of Neuroscience* (if the empirical validation is strong); *Frontiers in Neuroscience* (Neuroprosthetics).
- Follow-ups: network-mediated (bipolar) activation for subretinal devices with bipolar cell reconstructions; degenerate-retina remodelling (Loizos-style multiscale); human-retina-specific models with Madugula-type data; closed-loop stimulus optimisation with the population prior.

## Risks, confounds & mitigations

- **AIS unknown** in reconstructions: treated as an explicit uncertainty factor (H3), calibrated on the alpha-cell data of Werginz 2020 / Raghuram 2019.
- **Missing axons / truncated dendrites**: axon synthesis with documented assumptions; `physical_Integrity` metadata; sensitivity to axon direction.
- **Reduced channel set** (Na/K/leak in the starter): calibrate against the full FM model in NEURON; report relative spreads as primary.
- **Homogeneous field**: layered retina and electrode-tissue distance strongly affect absolute thresholds; use relative measures, and the FEM upgrade path.
- **Species mismatch** (mouse morphologies vs primate/human data): scaling as a design axis; human/primate NeuroMorpho cells where available.
- **Batch effects** across reconstruction datasets: dataset random effects; cluster bootstrap.
- **Compute**: 1k cells x ~100 settings x 10 bisection steps is ~10^6 simulations; use the linear activating-function pre-screen to narrow bisection brackets and parallelise.

## Milestones

- [ ] Harvest Eyewire + NeuroMorpho RGCs; type labels harmonised; morphometrics table.
- [ ] Model validation on single-cell literature cases; NEURON calibration subset.
- [ ] Population sweep v1 (one epiretinal, one subretinal geometry, 3 pulses, 3 AIS configs).
- [ ] H1-H3 analyses; determinant model release.
- [ ] Selectivity ceiling (H4); axon-bundle activation (H5).
- [ ] Empirical comparison (H6) with published threshold distributions.
- [ ] Preprint; code + threshold table + fitted model release.

## Ethics / data-use notes

- All morphology data are open; cite Eyewire/Bae et al. 2018, Sümbül et al. 2014, NeuroMorpho.org and depositing labs (`reference_pmid`/`reference_doi`) as required.
- No human subjects or animals are involved in this computational work; empirical comparison data come from published studies (cite; request raw data through the authors' stated channels).
- Do not commit SWC files, field volumes or threshold tables (`data/` and `outputs/` are git-ignored).

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py --sample          # NeuroMorpho RGC metadata + 10 SWC files
PYTHONPATH=src pytest -q tests                    # synthetic tests (no network, no NEURON)
```

```python
import glob
import pandas as pd
from rgc_prosthesis import (load_swc, compartmentalize, morphometrics, synthesize_axon, Electrode, Pulse,
                            RGCCable, threshold_current, type_threshold_summary, morphological_determinants)
from rgc_prosthesis.swc_morph import orient_z

rows = []
for f in glob.glob("data/neuromorpho/swc/*.CNG.swc"):
    swc = orient_z(load_swc(f))
    m = morphometrics(swc)
    for ais_start in (20.0, 40.0, 60.0):                           # AIS geometry as a design factor
        cell = RGCCable(compartmentalize(synthesize_axon(swc, ais_start_um=ais_start), max_len_um=10.0))
        cell.equilibrate()
        ve = Electrode.epiretinal(height_um=30, radius_um=50).unit_potential(cell.comp.xyz)
        thr = threshold_current(cell, ve, Pulse(0.1, "biphasic"))
        rows.append({"cell_id": f, "cell_type": "TODO from metadata", "ais_start_um": ais_start,
                     "threshold_uA": thr, **m})
df = pd.DataFrame(rows)
print(type_threshold_summary(df))
print(morphological_determinants(df, ["soma_diam_um", "dend_field_diam_um", "strat_depth_um", "ais_start_um"])["table"])
```

## Pre-specified operational definitions

| Item | Definition (frozen before the sweeps) |
|---|---|
| Cell set | Eyewire (type-labelled, 47 types), Sümbül 2014 (genetic lines) and NeuroMorpho RGCs with >= 500 um dendritic length and a soma; species and dataset recorded |
| Orientation | z flipped so dendrites are at positive z (IPL) relative to the soma; axon synthesised in the NFL at soma_z - 10 um toward the optic disc (axon map or fixed direction) |
| AIS design | start in {20, 40, 60} um, length in {20, 30, 40} um, diameter 1.0 um; hillock 2.0 um; distal axon 0.7 um, 1.5 mm |
| Compartments | <= 10 um cylinders; soma one equivalent cylinder (L = d) |
| Kinetics | FM97 Na/K/leak; densities (mS/cm^2) dendrite 25/12, soma 80/18, hillock 150/25, AIS 400/40, axon 100/25; g_L 0.15, balanced E_L; 5-channel FM in NEURON for calibration |
| Electrodes | epiretinal disk radius {25, 50, 100} um at heights {20, 50, 100} um; subretinal disk radius 50 um at depth 150 um; sigma 0.3 S/m (0.1-1.0 sensitivity) |
| Lateral offsets | over soma (0), over dendritic field centroid, over the axon 200 um from the soma |
| Pulses | cathodic-first biphasic 100 us (primary); 50, 500, 1000, 4000 us; monophasic cathodic/anodic |
| Threshold | smallest current with V > 0 mV at the distal axon end (propagated spike); geometric bracket + bisection to 3% |
| Activation site | region of the first compartment crossing 0 mV (AIS / hillock / distal axon / soma / dendrite) |
| Primary statistics | per-type median, IQR, geometric SD; between/within log-variance ratio; MixedLM of log threshold with type (and dataset) random intercepts |
| Selectivity | pairwise AUC = P(threshold_a < threshold_b); selective window = fraction of A activated at the 10th percentile of B |
| Empirical comparison | global scale fitted on ON parasol; KS and QQ slope on the remaining types (Grosberg 2017; Madugula 2022) |

## Related projects

- `morphology-dependent-stimulation` (uniform-field polarisation over NeuroMorpho populations; this project uses focal electrode fields, active RGC kinetics and typed retinal populations and stays self-contained).
- `dbs-vta-population-variability` (same population-variance philosophy for myelinated axons in DBS fields).
