# dmri-microstructure-charts — Which diffusion model deserves a lifespan chart? Model-comparison normative curves for DKI, NODDI and DTI on the same brains

Build lifespan centile charts for higher-order and multi-compartment diffusion metrics (DKI mean/axial/radial kurtosis; NODDI neurite density, orientation dispersion and free-water fraction; optionally SMT and fixel metrics) fitted on the *same* multi-shell subjects as DTI, and grade the models on what a chart is for: age sensitivity, test-retest reliability of individual deviations, transportability across acquisition protocols (emulated by shell subsampling), and sensitivity to clinical deviation.

Related project in this repository: `lifespan-normative-models` (volume/thickness charts, PCNtoolkit/BrainChart workflow). This project is self-contained and microstructure-only.

## Status / difficulty / timeline / compute

- Status: design + starter code (gradient-table/shell tools with protocol emulation, weighted linear DTI and DKI fitters with mean/axial/radial kurtosis, a location-scale spline normative model with centiles, z-scores and age-of-peak bootstrap, and model-comparison metrics: age sensitivity, ICC of z-scores, protocol-transfer bias, deviation AUC, leaderboard).
- Difficulty: MSc thesis (HCP-YA + Cam-CAN + IXI arm) to PhD chapter (adding HCP-Aging/-Development and clinical deviation cohorts). 6-9 months for the first paper.
- Compute: DTI/DKI fits are minutes per subject (DIPY or the linear fitters here). NODDI with AMICO is ~5-10 min per subject on CPU; ~2,500 multi-shell subjects across HCP-YA, HCP-A, HCP-D and Cam-CAN is a few hundred CPU-hours. Tract-level extraction (TractSeg or JHU/ENIGMA-DTI skeleton) adds a similar amount. No GPU required; storage ~2-4 TB for raw multi-shell data.

## Background

Normative "brain charts" (Bethlehem et al., 2022, Nature; Rutherford et al., 2022, eLife) turned volumetric MRI into centile scores that are interpretable at the individual level. Diffusion MRI has followed in 2025-2026, but only for the diffusion tensor: an ENIGMA-led model ("Lifespan normative modeling of brain microstructure", Nature Communications, 2026) fitted hierarchical Bayesian regression to DTI metrics from 19 datasets (N = 54,583, 4-91 years; FA peaks at 29 y, RD/MD/AD minima at 37/43/53 y), a GAMLSS model of FA across the HCP lifespan cohorts and UK Biobank ("Lifespan normative models of white matter fractional anisotropy", Biological Psychiatry, 2025; > 25,000 individuals, 0-100 y) states that non-tensor models are future work, and site-harmonising lifespan reference curves for regional DTI metrics were published in 2025.

The tensor is the least specific model available. Multi-shell acquisitions (HCP-YA: b = 1000/2000/3000; HCP-Aging/-Development: b = 1500/3000; Cam-CAN: b = 1000/2000; UK Biobank: b = 1000/2000) support diffusion kurtosis imaging (Jensen et al., 2005, Magnetic Resonance in Medicine), NODDI (Zhang et al., 2012, NeuroImage; fast fitting with AMICO, Daducci et al., 2015, NeuroImage), the spherical mean technique (Kaden et al., 2016, NeuroImage) and fixel-based measures (Raffelt et al., 2017, NeuroImage). Age effects on these metrics are established in cross-sectional studies (Cox et al., 2016, Nature Communications, UKB N = 3,513; Lawrence et al., 2021, Brain Imaging and Behavior, UKB N = 15,628; Slater et al., 2019, Human Brain Mapping; Beck et al., 2021, NeuroImage; Coutu et al., 2014, NeuroImage for DKI) — older age goes with lower neurite density and higher orientation dispersion in nearly every tract — but there are no centile charts, no individual-deviation reliability estimates and no head-to-head of models on the same subjects.

## The research gap

What has been done:

- Lifespan DTI charts at scale (ENIGMA HBR model, 2026; Biological Psychiatry GAMLSS FA model, 2025; 2025 regional reference curves) with public model files.
- Cross-sectional age effects for NODDI/DKI/WMTI in single cohorts (UKB, Cam-CAN, Norwegian lifespan sample), typically as regression slopes or quadratic fits, without centiles or deviation scores.
- Model-degeneracy and identifiability critiques of multi-compartment models (Jelescu et al., 2016, NMR in Biomedicine; Novikov et al., 2018, NMR in Biomedicine) that argue the parameters depend on protocol and constraints, which is exactly why transportability of a chart must be measured rather than assumed.
- Test-retest reliability of DKI/NODDI metrics in small samples, not of *normative deviations*.

What is specifically missing:

1. No lifespan centile charts for DKI or NODDI metrics, and no chart built with several models on identical subjects so that model differences are not confounded with sample differences.
2. No evaluation of charts by their intended use: the reliability of a person's z-score (HCP-YA retest, n = 45), the sensitivity of centiles to age per year of true aging, and how far a z-score moves when the protocol changes (b-values, number of directions, b = 0 count) — which can be emulated exactly by subsampling the HCP shells to the Cam-CAN, UKB and HCP-A schemes.
3. No comparison of the *age-of-peak / age-of-minimum* across models (FA at ~29 y in the ENIGMA model; does neurite density peak later, and orientation dispersion earlier?), with bootstrap CIs, which is a substantive neurobiological result about myelination vs axonal packing.
4. No deviation-based validation on clinical multi-shell data: which model's z-scores best separate patients from the reference (AUC) at fixed reliability.
5. Single-shell fallback: IXI and most OpenNeuro dMRI are single-shell; a chart is only useful if the DTI arm can be linked to the multi-shell arms (calibration of DTI z-scores against DKI/NODDI z-scores in the same subjects).

## Research questions / hypotheses

1. RQ1 (age sensitivity). Per tract, which metric has the largest fraction of variance explained by age (R²_age) and the steepest normalised slope after 40 y? H1: NODDI ICVF and DKI MK exceed FA in R²_age in the majority of JHU tracts; free-water fraction (ISOVF) has the strongest late-life acceleration.
2. RQ2 (reliability of deviations). What is the ICC of subject z-scores between HCP-YA test and retest sessions for each model? H2: DTI FA/MD z-scores ICC ≈ 0.8-0.9; NODDI ICVF ≈ 0.7-0.85; ODI ≈ 0.6-0.8; DKI MK ≈ 0.6-0.8 (kurtosis is noise-amplifying).
3. RQ3 (protocol transportability). When HCP-YA data are subsampled to the Cam-CAN (b = 1000/2000; 30/60 dirs), HCP-A (b = 1500/3000) and UKB (b = 1000/2000; 50/50 dirs) schemes, by how much do z-scores shift (mean Δz, Cohen's d)? H3: DTI metrics from b = 1000 only are stable (|d| < 0.2); NODDI ICVF shifts by |d| ≈ 0.3-0.6 when the high shell changes; DKI is most sensitive to the maximum b-value.
4. RQ4 (age of peak). Do the ages of peak/minimum differ across models? H4: ICVF peaks later (35-45 y) than FA (~29 y); ODI minimum occurs earlier (20-30 y); MK peaks with FA. Bootstrap CIs exclude equality for ICVF vs FA.
5. RQ5 (clinical deviation). In multi-shell clinical datasets (OpenNeuro multi-shell cohorts with patient groups; ADNI-3 multi-shell subset), which model's tract z-scores give the highest AUC for patient vs control and the highest fraction of subjects with |z| > 1.96 in expected tracts? H5: NODDI ISOVF and DKI MK outperform FA in neurodegeneration; no model outperforms MD in acute lesions.
6. RQ6 (single-shell calibration). How well can DTI z-scores from a single-shell subset predict NODDI/DKI z-scores in the same subjects? H6: R² ≈ 0.5-0.7 for ICVF, < 0.3 for ODI — a DTI-only chart cannot substitute for orientation dispersion.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| HCP Young Adult S1200 | Multi-shell dMRI (b = 1000/2000/3000, 90 dirs each, 1.25 mm), preprocessed `T1w/Diffusion/{data.nii.gz,bvals,bvecs}`; 45-subject retest | ~1,065 with dMRI; 22-37 y | Free registration (ConnectomeDB; S3 bucket with issued keys) | https://db.humanconnectome.org |
| HCP Aging / HCP Development (Lifespan) | Multi-shell dMRI (b = 1500/3000, 1.5 mm), same processing family | ~1,200 (36-100+ y) + ~650 (5-21 y) | Application via NIH NDA | https://nda.nih.gov/ccf , https://www.humanconnectome.org/study/hcp-lifespan-aging |
| Cam-CAN | Multi-shell dMRI (b = 1000 (30 dirs) / 2000 (60 dirs), 2 mm), 18-88 y | ~650 with dMRI | Data-use agreement (free) | https://cam-can.mrc-cbu.cam.ac.uk |
| IXI | Single-shell DTI (15 dirs, b = 1000), 20-86 y | ~550 with DTI | Open | https://brain-development.org/ixi-dataset/ |
| OpenNeuro multi-shell dMRI datasets | Datasets whose `*.bval` files contain ≥ 2 non-zero shells (discovered automatically; several include patient groups) | tens of datasets, 20-200 subjects each | Open (S3 `openneuro.org` bucket, anonymous) | https://openneuro.org |
| UK Biobank (optional) | Multi-shell dMRI (b = 1000/2000), NODDI/DTI IDPs already computed | ~50,000 | Application + fee | https://www.ukbiobank.ac.uk |
| ADNI-3 (optional, clinical deviation) | Multi-shell dMRI subset (b = 500/1000/2000 in ADNI-3 advanced protocol) with diagnosis | hundreds | Application via LONI IDA | https://adni.loni.usc.edu |
| ENIGMA-DTI / published DTI normative models | Public model files for cross-check of the DTI arm | small | Open (paper repositories) | see papers |

## Methods

1. Gradient tables and protocol emulation (`dmri_charts.shells`): read `bvals`/`bvecs`, identify shells with tolerance, and subsample shells and directions to emulate other protocols (Cam-CAN, UKB, HCP-A, single-shell) from HCP-YA data, including b = 0 count.
2. Model fitting (`dmri_charts.models`): weighted linear least squares DTI and DKI (kurtosis tensor with 15 unique elements, mean/axial/radial kurtosis via directional sampling) implemented here for transparency and testing; production fits with DIPY (`dipy.reconst.dki`), AMICO-NODDI (`amico`), SMT (`smt` toolbox) and MRtrix3 fixel pipeline. Quality control: MRIQC-style motion (eddy RMS), outlier slices, and DKI physical-plausibility constraints (K > 0).
3. Regional summaries: JHU-ICBM tract labels on the ENIGMA-DTI skeleton (mean per tract) and TractSeg bundle averages; both to check that the chart does not depend on the region definition.
4. Normative model (`dmri_charts.normative`): location-scale model with B-spline age terms for mean and log-variance, sex and site fixed effects (a GAMLSS-lite; the R `gamlss` fit with BCCG family and PCNtoolkit HBR are the production alternatives, with this implementation as the transparent reference). Outputs: centile curves, z-scores, and bootstrap CIs for the age of peak/minimum. Sites are harmonised by the model (site intercepts and variance) rather than by ComBat, so that the protocol-transfer experiment is not confounded by prior harmonisation.
5. Model comparison (`dmri_charts.compare`): per metric and tract — R²_age and normalised slope, ICC(2,1) of z-scores in the HCP retest subset, protocol-transfer bias (Δz mean, Cohen's d, correlation) between full and emulated protocols, deviation AUC in clinical datasets, and a rank-sum leaderboard.
6. Single-shell calibration: fit DTI on the b = 1000 shell only; regress NODDI/DKI z-scores on DTI z-scores; report R² and calibration slopes so that IXI/OpenNeuro single-shell data can be placed on the chart with stated uncertainty.

Libraries: numpy, scipy, pandas, statsmodels; DIPY, AMICO, MRtrix3, TractSeg, FSL (eddy/TBSS) for production; optional PCNtoolkit and R gamlss for the alternative normative fits.

## Evaluation and statistics

- Reference sample: HCP-YA + HCP-A + HCP-D + Cam-CAN healthy participants; exclusion by cohort-specific health criteria and QC; age range 5-100 y. Clinical datasets never enter the reference fit.
- Cross-validation: 10-fold CV for the normative fit (z-scores of held-out subjects; expect mean 0, SD 1, no age trend of z or of z²); calibration plots per decade and per cohort.
- Reliability: ICC(2,1) with 95 % CI (HCP retest, n = 45); minimal detectable change of z.
- Protocol transfer: paired subject-level Δz; bias and limits of agreement; Cohen's d; reported per tract and pooled.
- Age of peak: parametric bootstrap (1,000) of the fitted mean curve; differences between models tested by bootstrap CI of the difference.
- Multiple comparisons: tracts are the unit; FDR across tracts within a metric; pre-registered primary tracts (cingulum, fornix, corpus callosum genu/splenium, corticospinal tract).
- Nulls: age-permuted fits (R²_age null), and simulated multi-compartment signals with known ground truth to verify that the linear fitters are unbiased at the SNR of each protocol.

## Publishable angle

Headline: "NODDI neurite density and DKI mean kurtosis are more age-sensitive than FA but their normative deviations are less reliable and shift by up to d = 0.5 across common protocols; ICVF peaks a decade later than FA." Deliverables: public chart files (centile tables per tract, metric and sex) for DKI and NODDI, an "atlas of transportability" telling users which metrics can be pooled across protocols, and the model-comparison leaderboard.

Target venues: NeuroImage, Imaging Neuroscience, Human Brain Mapping, Magnetic Resonance in Medicine (protocol-transfer component), Nature Communications (if HCP-Lifespan + UKB are included).

Follow-ups: longitudinal validation with HCP-A follow-ups; extension to grey-matter NODDI; use of the charts as covariates in `lifespan-normative-models`-style multimodal deviation profiles.

## Risks, confounds and mitigations

- Model degeneracy and constraint choices (NODDI fixed intrinsic diffusivity; DKI upper b) change parameter values: fix constraints per protocol, report the sensitivity, and include the constrained-NODDI variant as an arm.
- Sample composition: HCP-YA is 22-37 y, so the middle of the curve comes from one cohort; site/cohort terms and CV across cohorts; leave-one-cohort-out fits.
- Resolution and SNR differences (1.25 vs 1.5 vs 2 mm) confound protocol emulation: emulate direction/shell subsets on HCP-YA (same voxels) and treat resolution separately with downsampled HCP data.
- Motion and age are correlated: include eddy motion as a covariate in a sensitivity fit.
- Registration/tract definitions drive tract means: two regional schemes (skeleton and bundles).

## Milestones

- [ ] Access: HCP-YA registration; HCP-A/D NDA application; Cam-CAN DUA; IXI and OpenNeuro discovery run.
- [ ] Preprocessing and QC of HCP-YA (+ retest); DTI/DKI/NODDI fits; tract summaries.
- [ ] Normative fits on HCP-YA (+ Cam-CAN, IXI) for all metrics; CV calibration; RQ1, RQ2.
- [ ] Protocol-emulation experiment on HCP-YA (RQ3); single-shell calibration (RQ6).
- [ ] Add HCP-A/D; age-of-peak analysis (RQ4).
- [ ] Clinical deviation validation on OpenNeuro multi-shell datasets (RQ5).
- [ ] Release chart files and leaderboard; write-up.

## Reference sample, metrics and tracts

| Cohort | Inclusion | Role |
|---|---|---|
| HCP-YA | dMRI QC pass, no neurological/psychiatric history flag, 22-37 y | reference core; protocol emulation; retest (n = 45) |
| HCP-Aging / HCP-Development | typical development/aging per HCP-Lifespan screening, 5-21 y and 36-100+ y | reference tails; age of peak |
| Cam-CAN | cognitively healthy (MMSE ≥ 25), 18-88 y | independent-protocol reference; site term |
| IXI | healthy volunteers, 20-86 y, single shell | single-shell calibration (RQ6) |
| OpenNeuro multi-shell clinical datasets | discovered by `openneuro-discover`; groups with a patient label | deviation validation (RQ5) |

Metrics per tract (JHU-ICBM skeleton, ENIGMA-DTI protocol; TractSeg bundles as the second scheme): DTI `FA, MD, AD, RD`; DKI `MK, AK, RK` (+ `KFA`); NODDI `ICVF (NDI), ODI, ISOVF`; optional SMT `intra-neurite fraction`, fixel `FD, FC, FDC`. Model covariates: `age`, `sex`, `site/cohort`, `eddy_motion` (sensitivity). Primary tracts: cingulum (cingulate), fornix, genu and splenium of the corpus callosum, corticospinal tract, superior longitudinal fasciculus, uncinate.

## Starter code map

| Module / function | What it does |
|---|---|
| `dmri_charts.shells.read_bvals_bvecs`, `identify_shells`, `protocol_summary` | gradient tables, shell detection, protocol description |
| `shells.subsample_protocol`, `SCHEMES` | emulate Cam-CAN / UKB / single-shell schemes from HCP data (farthest-point direction subsets) |
| `dmri_charts.models.fit_dti`, `fit_dki` | weighted linear DTI and DKI fits with FA/MD/AD/RD and MK/AK/RK |
| `models.simulate_multicompartment_signal`, `fibonacci_sphere` | stick+zeppelin+ball signals with Rician noise for validation |
| `dmri_charts.normative.NormativeModel` | spline location-scale model: `fit`, `predict`, `zscore`, `centiles`; `age_of_peak`, `bootstrap_age_of_peak`, `simulate_lifespan_dataset` |
| `dmri_charts.compare.age_explained_variance`, `icc_2_1`, `protocol_transfer_bias`, `deviation_auc`, `rank_models` | the four chart-quality criteria and the leaderboard |
| `tests/test_dmri_charts.py` | shell tools, DTI recovery of a known tensor, DKI kurtosis sign, normative calibration and age of peak, comparison metrics |

Quick start:

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/download_data.py openneuro-discover --out data/openneuro --max-datasets 100   # open, no credentials
```

## Ethics / data-use notes

- HCP data-use terms (open access and, for HCP-A/D, NDA); Cam-CAN DUA; IXI CC BY-SA 3.0; OpenNeuro CC0 — cite each dataset's DOI.
- Charts are released as aggregate centile tables only; no subject-level data leaves the analysis environment; no data are sent to third-party APIs.
- Clinical datasets are used only for validation; no diagnostic claims for individuals.
