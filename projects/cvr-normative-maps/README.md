# cvr-normative-maps — Lifespan centile charts of breath-hold cerebrovascular reactivity amplitude and delay from open data

Build the first normative (centile) charts of BOLD cerebrovascular reactivity (CVR) amplitude and delay across ages 6-85 from the largest open breath-hold fMRI sample (NKI-Rockland enhanced), anchor their reliability and gas-free bias with densely repeated end-tidal-CO2 breath-hold data (EuskalIBUR, OpenNeuro ds003192), and test whether age, sex and vascular risk shift an individual's CVR centile and whether resting-state CVR proxies can reproduce the charts without any task.

## Status / difficulty / timeline / compute

- Status: design + starter code (end-tidal CO2 and RVT extraction, breath-hold compliance scoring, lag-optimized voxelwise CVR amplitude/delay GLM, heteroscedastic spline normative model, reliability metrics).
- Difficulty: MSc-level, 6-9 months (reliability/bias paper on EuskalIBUR in ~3 months; NKI normative paper in ~6).
- Compute: fMRIPrep ~1.5-2 CPU-hours per NKI session (breath-hold + rest + T1); ~1,000-1,500 sessions → 2,000-3,000 CPU-hours on a cluster. The CVR fit itself is vectorized (seconds per run). Storage: ~500 GB if only the breath-hold, rest and T1w series are pulled from S3; EuskalIBUR is ~60 GB. No GPU.

## Background

CVR is the vasodilatory capacity of the cerebral vasculature, usually measured as the BOLD change per mmHg of end-tidal CO2 (PetCO2) during hypercapnia (Liu, De Vis and Lu, 2019, NeuroImage). It falls with age (Lu et al., 2011, Cerebral Cortex), rises through childhood (Thomason et al., 2005, NeuroImage; "Cerebrovascular reactivity increases across development in multiple networks", Human Brain Mapping, 2024, using NKI-Rockland children), is reduced in small-vessel disease and steno-occlusive disease, and confounds every BOLD comparison across age groups (Handwerker et al., 2007, Human Brain Mapping; Tsvetanov, Henson and Rowe, 2021, Phil Trans R Soc B). Gas-controlled CVR needs a RespirAct-type device; breath-holding is the practical alternative and, with end-tidal CO2 recorded through a nasal cannula, yields amplitude in %BOLD/mmHg and delay in seconds even with imperfect task performance (Bright and Murphy, 2013, NeuroImage; Moia et al., 2021, NeuroImage; Stickland et al., 2021, NeuroImage; Zvolanek et al., 2023, NeuroImage, comparing PetCO2, RVT and average GM signal as regressors; Pinto et al., 2021, Frontiers in Physiology, for gas-free methods). Clinically, CVR maps are read against a "reference atlas" of healthy adults (Sobczyk et al., 2015, Journal of Cerebral Blood Flow & Metabolism; 46 subjects, mean ± SD per voxel, z-score maps).

## The research gap

What has been done (2023-2026):

- Methodological refinements for breath-hold CVR amplitude and delay when PetCO2 quality is low (Imaging Neuroscience, 2025, "Quantitative mapping of cerebrovascular reactivity amplitude and delay with breath-hold BOLD fMRI when end-tidal CO2 quality is low"), comparison of signal models for breath-hold CVR (Imaging Neuroscience, 2025), and amplitude/lag estimation from resting-state and breath-hold data (JCBFM, 2024).
- Developmental CVR trajectories in the NKI-Rockland longitudinal child cohort (Human Brain Mapping, 2024), showing network-specific increases from 6 to 18 years.
- A bioRxiv 2023 study on the "significance and limited influence" of CVR on age and sex effects in task and resting BOLD.
- Reference atlases and z-score maps for gas-controlled CVR (Sobczyk et al., 2015) and, in 2024, a normative CVR dataset release.

What is specifically missing:

1. No lifespan centile charts (normative model with age-varying mean and variance, by sex) for CVR amplitude and delay, regionally or globally, in either breath-hold or gas-controlled data. Reference atlases assume a single healthy-young distribution; the developmental NKI work stops at 18; adult-lifespan curves come from N < 100 gas studies.
2. NKI-Rockland enhanced has a breath-hold task with a respiration belt in > 1,000 participants aged 6-85, plus resting-state scans, T1w, vitals and medical history; the adult part of this resource has never been used to chart CVR or to test vascular-risk effects on CVR at scale.
3. Reliability of CVR delay, minimal detectable change and the bias of gas-free regressors (task boxcar, RVT) against PetCO2 are unknown at the precision needed for individual deviation scores; EuskalIBUR (10 weekly sessions per subject, multi-echo, PetCO2) is the ideal open benchmark and has been used for denoising papers but not for a reliability/MDC/bias report.
4. Nobody has tested whether resting-state CVR proxies (CO2-band spontaneous fluctuations, sLFO amplitude, RVT responses; Liu et al., 2017, NeuroImage; Pinto et al., 2021) reproduce breath-hold CVR centiles within subjects across the lifespan; NKI has both scans in the same session.
5. The proportion of the age effect on BOLD amplitude that an individual CVR centile explains has not been estimated on a lifespan sample.

## Research questions / hypotheses

1. RQ1 (reliability, EuskalIBUR). ICC(2,1) and minimal detectable change (MDC95) of GM and parcel-level CVR amplitude and delay across 10 sessions with (a) PetCO2, (b) task boxcar ⊗ HRF and (c) RVT regressors. H1: amplitude ICC > 0.7 with PetCO2 and > 0.5 with the boxcar; delay ICC 0.4-0.6; MDC95 of GM amplitude < 30% of the between-subject SD.
2. RQ2 (gas-free bias). Boxcar- and RVT-derived amplitudes relate linearly to PetCO2-derived amplitude with a subject-specific scale (R² > 0.6 across parcels) and delays agree within 1 s (Bland-Altman). H2: the scale factor is predictable from the belt-derived breath-hold depth, allowing a partial calibration in datasets without capnography.
3. RQ3 (normative curves, NKI). GM CVR amplitude follows an inverted-U (rise to the early 20s, then ~3-5% decline per decade), delay increases monotonically after 40, with regional heterogeneity (posterior circulation and deep territories decline earlier) and sex differences. Centiles are stable to protocol (15 s vs 18 s hold) after harmonization.
4. RQ4 (vascular risk, NKI). Hypertension, BMI, smoking and diabetes lower the CVR-amplitude centile and raise the delay centile beyond age. H4: hypertension shifts amplitude by ≥ 0.5 SD; effects are strongest in deep/watershed territories.
5. RQ5 (resting-state proxies, NKI). Resting-state CVR proxies (CO2-band ALFF 0.02-0.04 Hz, sLFO amplitude, RVT-response amplitude) correlate with breath-hold amplitude at r = 0.4-0.6 (GM level) and reproduce the age curve shape.
6. RQ6 (confound accounting). Including the CVR centile as a covariate attenuates age effects on BOLD amplitude in the NKI task data by 20-50%.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| NKI-Rockland Sample enhanced (Nooner et al., 2012, Frontiers in Neuroscience) | Breath-hold task fMRI (TR 1.4 s, multiband; 15 s or 18 s holds × 7 blocks), respiration belt, resting-state (TR 0.645 s and 1.4 s), T1w; vitals (BP), BMI, medical history, medications, demographics | > 1,000 participants, ages 6-85; longitudinal subset | Imaging: public S3 (`fcp-indi`), no credentials. Phenotypes and physiological logs: free Data Use Agreement (NKI-RS DUA) | http://fcon_1000.projects.nitrc.org/indi/enhanced/ |
| EuskalIBUR (Moia et al., 2021, NeuroImage) | Multi-echo breath-hold fMRI with PetCO2 (nasal cannula), respiration, resting-state, T1w, 10 weekly sessions | 7 of 10 participants public (~70 sessions) | Open (OpenNeuro ds003192) | https://openneuro.org/datasets/ds003192 |
| Other OpenNeuro breath-hold / hypercapnia datasets | Optional external replication of protocol effects (search OpenNeuro for "breath-hold", "hypercapnia", "CO2") | varies | Open | https://openneuro.org/ |
| Arterial territory atlas (Liu et al., 2023, Scientific Data) | Territory-wise summaries (ACA/MCA/PCA, deep, watershed) | 1 atlas | Open | https://www.nitrc.org/ (search "arterial territories atlas") |

## Methods

1. Preprocessing. fMRIPrep 23.x (NKI, single-echo). EuskalIBUR multi-echo: tedana optimal combination and ME-ICA denoising with the Moia et al. (2021) strategy. Motion parameters and FD retained as confounds; no global signal regression (it removes CVR).
2. Physiology (`cvr_norm.physio`). End-tidal detection from the CO2 trace (exhalation peaks, prominence-gated, interpolated to a PetCO2 time course); RVT from the belt (Birn et al., 2006); breath-hold compliance = fraction of hold blocks where belt variance drops below 25% of the baseline; task boxcar and HRF convolution for datasets without capnography. Production alternatives: peakdet, phys2bids, phys2cvr.
3. CVR estimation (`cvr_norm.cvr_model`). Lag-optimized voxelwise GLM: regressor shifted over -9 to +9 s in 0.3 s steps, Legendre drift terms + motion confounds, amplitude in %BOLD per unit regressor (per mmHg with PetCO2), delay in seconds, R² and t maps. Cross-checked against phys2cvr and rapidtide's CVR mode.
4. Summaries. Schaefer-200 parcels, arterial territories, GM/WM medians; negative-CVR fraction; delay dispersion.
5. Normative model (`cvr_norm.normative`). Cubic B-spline mean and log-variance by sex with a protocol covariate; z-scores and centiles; GAMLSS/PCNtoolkit cross-check; ComBat-GAM for protocol (15 s vs 18 s hold) if needed.
6. Reliability (`cvr_norm.reliability`). ICC(2,1), within-subject CoV, MDC95, Dice of thresholded maps across EuskalIBUR sessions.
7. Resting-state proxies. CO2-band ALFF, rapidtide sLFO amplitude, RVT-response amplitude from the NKI resting scans; compared with breath-hold amplitude within session.

Libraries: numpy, scipy, pandas, statsmodels, scikit-learn; nibabel, nilearn; fMRIPrep, tedana; peakdet/phys2bids/phys2cvr (physiopy); rapidtide; PCNtoolkit (optional).

## Evaluation and statistics

- Reliability: ICC with 95% bootstrap CIs over subjects; MDC95 = 1.96 × √2 × SEM; Bland-Altman for regressor bias; all per parcel with FDR.
- Normative curves: GAM / spline models with sex-specific smooths and protocol, FD and compliance as covariates; knots by AIC; discovery/replication split-half of NKI (random, stratified by age decade) so that risk-factor associations (RQ4) are confirmed on held-out data.
- Vascular-risk models: linear models of the centile (probit-transformed) on risk factors adjusted for age, sex, FD, compliance; Holm across the 4 pre-registered factors.
- Proxies: within-subject correlations and centile agreement (weighted kappa on quintiles).
- Leakage prevention: centile parameters estimated on the NKI discovery half; EuskalIBUR used only for reliability/bias and for the calibration factor; no threshold tuning on replication data.
- Nulls: phase-randomized BOLD to obtain a false-CVR floor for R² thresholds; permutation of age labels for the curve-shape test.
- QC: compliance < 4/7 blocks, mean FD > 0.3 mm, or R² floor failures → excluded, with a sensitivity analysis including them.

## Publishable angle

Headline: "Open lifespan centile charts for breath-hold CVR amplitude and delay, with measured reliability and gas-free bias, show that vascular risk shifts an individual's CVR centile by half a standard deviation, and that resting-state proxies recover the age curve without a task." A companion reliability/MDC paper on EuskalIBUR is a natural first output.

Target venues: Imaging Neuroscience; NeuroImage; Journal of Cerebral Blood Flow & Metabolism; Human Brain Mapping; Frontiers in Physiology (for the reliability/bias piece).

Follow-ups: apply the charts to clinical breath-hold data (Moyamoya, small-vessel disease, sickle-cell); combine with resting-state blood-arrival lag (this repository's `fmri-hemodynamic-aging`) for a two-parameter vascular fingerprint; CVR-adjusted brain charts in `lifespan-normative-models`.

## Risks, confounds and mitigations

- Breath-hold compliance varies with age (children, older adults); belt-derived compliance is a covariate and an exclusion criterion, and lag-optimized fits reduce sensitivity to poor performance (Bright and Murphy, 2013).
- No capnography in NKI: amplitude is in %BOLD per unit regressor, not per mmHg; EuskalIBUR provides an empirical calibration and its uncertainty is propagated into the charts (reported as "boxcar-referenced" centiles).
- Two breath-hold protocols (15 s / 18 s) and scanner/sequence changes over the years of NKI acquisition: protocol covariate, harmonization, and a check that centiles do not differ by acquisition year.
- Deep breaths after holds cause motion; censor and model FD; report results with and without censoring.
- Negative CVR in WM/CSF and vascular steal: report signed maps, restrict centiles to GM, and provide territory-level values.
- Delay maps are noisy in low-CVR voxels: only voxels with R² above the null floor contribute to delay summaries.
- Age-related HRF timing changes bias the boxcar ⊗ canonical-HRF model: the lag search absorbs shifts, and an FIR model is run as a sensitivity check.

## Milestones

- [ ] Download EuskalIBUR (ds003192); tedana + physio processing; CVR fits with PetCO2, boxcar and RVT regressors.
- [ ] Reliability/MDC/bias analyses (RQ1-RQ2); write up methods paper.
- [ ] NKI DUA for phenotypes and physio; pull breath-hold + rest + T1w from S3; fMRIPrep.
- [ ] Compliance QC; CVR maps; parcel and territory summaries.
- [ ] Normative model on the discovery half; centile charts; replication of curve shape.
- [ ] Vascular-risk associations on the replication half (RQ4); resting-state proxies (RQ5); confound accounting (RQ6).
- [ ] Release charts (CSV + NIfTI) and code; manuscript.

## Ethics / data-use notes

- NKI-Rockland phenotypic and physiological data require the NKI-RS DUA; imaging on S3 is public but must be used under the same terms. EuskalIBUR is CC0.
- No participant-level data are committed; `data/` and `outputs/` are git-ignored. Charts are released as group-level curves only.
- No third-party LLM services are involved.
