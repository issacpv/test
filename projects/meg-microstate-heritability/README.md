# meg-microstate-heritability

**Are the temporal dynamics of resting-state MEG microstates heritable, and do they track cardiac autonomic tone? A reliability-ceilinged twin analysis of HCP MEG (with its co-recorded ECG), benchmarked against spectral power and with head-position similarity treated as a confound rather than ignored.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (GFP-peak modified k-means microstate segmentation, back-fitting and smoothing, temporal parameters incl. transition matrices and DFA exponent; R-peak detection and HRV; ACE twin models, Falconer estimates, twin-identification accuracy, simulation-based power; synthetic generators).
- Difficulty: MSc thesis (sensor-space + heritability) to PhD chapter (source-space, cardiac coupling, fMRI comparison).
- Timeline: 6-9 months (1 month HCP access + MEG preprocessing, 2 months microstates + reliability, 1 month HRV, 2 months twin modelling and nulls, 1-2 months writing).
- Compute: workstation (16 cores, 64 GB). HCP MEG resting data for 95 subjects x 3 runs is ~100-200 GB raw; preprocessed sensor-space runs are far smaller. Source reconstruction is the only expensive step (hours per subject on CPU).

## Background

EEG microstates are quasi-stable scalp-potential topographies lasting ~60-120 ms that recur in a small set of classes (Lehmann, Ozaki & Pal, 1987, EEG and Clinical Neurophysiology; review: Michel & Koenig, 2018, NeuroImage). Their sequence shows scale-free temporal structure (Van de Ville, Britz & Michel, 2010, PNAS), maps onto fMRI resting-state networks (Britz, Van de Ville & Michel, 2010, NeuroImage), changes with development and sex (Tomescu et al., 2018, Developmental Cognitive Neuroscience) and is a candidate endophenotype for schizophrenia on the basis of first-degree-relative studies (da Cruz et al., 2020, Nature Communications). MEG microstates have been characterised at sensor and source level (Tait & Zhang, 2022, NeuroImage; "MEG microstates: an investigation of underlying brain sources", 2024, Brain Topography).

"Endophenotype" is a genetic claim, but the heritability of microstate *dynamics* (durations, occurrence, transition structure, long-range dependence) has not been estimated in a classical twin design. What is known: resting EEG spectral power is highly heritable (meta-analysis: van Beijsterveldt & van Baal, 2002, Biological Psychology); in HCP MEG twins, power spectra are heritable and connectivity less so (Colclough et al., 2017, eLife); graph-theoretical properties of HCP MEG networks are heritable enough to identify MZ twins ("Genetic fingerprinting with heritable phenotypes of the resting-state brain network topology", 2024; 89 HCP MEG subjects: 17 MZ pairs, 12 DZ pairs, 4 sibling pairs, 23 singletons); fMRI HMM state dynamics in HCP are heritable (Vidaurre et al., 2017, PNAS). Heart-rate variability is moderately heritable (Kupper et al., 2004, Circulation), and HCP MEG runs include an ECG channel, so cardiac autonomic tone is measurable in the same sessions.

## The research gap

**What has been done**

- Heritability of MEG power and static connectivity in HCP twins (Colclough et al., 2017, eLife); heritability and twin-fingerprinting of MEG graph metrics (2024 study above).
- Heritability of fMRI temporal state dynamics (Vidaurre et al., 2017, PNAS) and of the functional connectome (multiple HCP papers).
- EEG microstate abnormalities in patients and unaffected relatives (da Cruz et al., 2020), taken as evidence of endophenotype status without a twin-based heritability estimate.
- Test-retest reliability of EEG microstate parameters (Khanna, Pascual-Leone & Farzan, 2014, PLoS ONE); nothing equivalent for MEG microstates across HCP's three runs.

**What is missing (checked against 2023-2026 literature)**

1. No twin-design heritability estimates for MEG (or EEG) microstate durations, occurrence, coverage, transition probabilities or the DFA/Hurst exponent of the microstate sequence.
2. No reliability ceiling: heritability cannot exceed test-retest reliability, yet MEG microstate parameter reliability across runs (same session) is unreported. Without it, "low heritability" and "unreliable measure" are indistinguishable.
3. An MEG-specific confound is unaddressed: sensor-space topographies depend on head position and head shape relative to the fixed helmet. MZ twins have more similar head shapes than DZ twins, so sensor-space heritability of *any* topographic measure may be inflated. Source-space microstates and head-position covariates can separate this.
4. Cardiac coupling: whether microstate transition probabilities are modulated by cardiac phase, and whether individual differences in microstate dynamics covary with HRV (RMSSD, HF power), has not been tested in a genetically informative sample; a cross-twin cross-trait design can ask whether any covariance is familial.
5. The HCP MEG sample is small for twin modelling (29 twin pairs). The literature nonetheless reports point estimates without power curves. A simulation-based power analysis and twin-identification accuracy (which is well powered at this n) are missing from the MEG dynamics literature.

## Research questions / hypotheses

1. **H1 (reliability).** Microstate mean duration, occurrence and coverage have within-session run-to-run ICC > 0.6; transition probabilities and the DFA exponent have ICC < 0.5. Test: ICC(2,1) across the 3 resting runs with bootstrap CIs.
2. **H2 (heritability, sensor space).** MZ intraclass correlations exceed DZ correlations for mean duration and occurrence of the dominant states, with Falconer h^2 > 0.4 and ACE likelihood-ratio support for A > 0. Test: ACE vs CE vs E models; permutation of pair labels within zygosity for the MZ-DZ difference.
3. **H3 (head-shape confound).** The MZ > DZ similarity of sensor-space microstate *templates* (spatial correlation of maps) is partly explained by head-position/head-shape similarity, and shrinks in source space. Test: partial twin correlations controlling for head-position distance; sensor vs source comparison.
4. **H4 (positive control).** Occipital alpha power heritability in the same runs reproduces published HCP MEG estimates (h^2 ~ 0.6-0.8), confirming the pipeline. Test: same ACE machinery on log alpha power.
5. **H5 (cardiac coupling).** Microstate transition probability is non-uniform across the cardiac cycle (systole vs diastole), and subjects with higher RMSSD show longer mean durations. Test: cardiac-phase histograms with phase-shuffled nulls; phenotypic correlation with HRV; cross-twin cross-trait correlation.
6. **H6 (twin identification).** A nearest-neighbour classifier on microstate-parameter vectors identifies the MZ co-twin above chance (chance = 1/(N-1)), and identification accuracy exceeds that obtained from head-position features alone. Test: identification accuracy with permutation null.
7. **H7 (power).** Given the observed reliabilities and the HCP MEG twin structure, simulation shows the minimum detectable h^2 at 80% power; parameters below that threshold are reported as "not resolvable" rather than "not heritable".

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| HCP Young Adult S1200, MEG | 3 resting-state runs (~6 min each, eyes open) per subject; 4D Neuroimaging 248-magnetometer system; co-recorded ECG/EOG channels; HCP-preprocessed sensor data and head models; head-position information | 95 subjects with MEG (89 used in the 2024 twin study) | Free registration + Open Access Data Use Terms (ConnectomeDB / AWS S3) | https://db.humanconnectome.org |
| HCP S1200 Restricted Data | Zygosity (genotype-confirmed), family IDs, age in years | csv | Restricted Access application (separate approval) | https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-restricted-data-use-terms |
| HCP S1200 resting fMRI (secondary) | Co-activation-pattern "fMRI microstates" for a well-powered heritability comparison | ~1,000 subjects, hundreds of twin pairs | same as above | https://db.humanconnectome.org |
| Cam-CAN MEG (secondary, unrelated adults) | Resting MEG with ECG across the adult lifespan; replication of templates and HRV coupling, age effects | ~650 subjects | Application / data-access agreement | https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/ |
| OMEGA (Open MEG Archive; secondary) | Resting MEG (CTF) with physiological channels; replication of microstate templates across MEG systems | hundreds of recordings | Registration | https://www.mcgill.ca/bic/resources/omega |

## Methods

1. **Data** (`scripts/download_data.py`): HCP MEG resting runs, both the HCP-preprocessed sensor-space output and the unprocessed 4D files (for the ECG channel), head models and head-position records, via AWS S3 with HCP-issued credentials; restricted zygosity table via ConnectomeDB after approval.
2. **Preprocessing**: MNE-Python; use HCP's cleaned sensor data (bad channels/segments, ICA removal of cardiac/ocular components) as the primary input; 1-40 Hz band-pass; downsample to 250 Hz. Secondary: reproduce cleaning from unprocessed data to check that ICA removal of the cardiac component does not remove cardiac-locked neural effects (H5 uses ECG timing only).
3. **Sensor-space microstates** (`src/meg_microstates/microstates.py`): GFP, GFP-peak maps, modified k-means (Pascual-Marqui, Michel & Lehmann, 1995, IEEE TBME) with polarity handling as a documented option, 4-7 states selected by cross-validation criterion and GEV; group templates from all subjects; back-fitting per subject with temporal smoothing (minimum 20 ms); parameters: mean duration, occurrence, coverage, transition matrix, entropy rate, DFA exponent of the state-indicator random walk.
4. **Head-position handling**: extract head-center coordinates and coil positions per run; compute pairwise head-position/head-shape distances between subjects; include as covariates and as a twin-similarity confound (H3).
5. **Source-space microstates**: LCMV beamformer or MNE on HCP head models, source maps at GFP peaks, clustering in source space (as in Tait & Zhang, 2022), same parameters.
6. **HRV** (`src/meg_microstates/hrv.py`): R-peak detection on the ECG channel, ectopic-beat cleaning, RMSSD, SDNN, pNN50, LF/HF; cardiac-phase histograms of microstate transitions.
7. **Twin modelling** (`src/meg_microstates/heritability.py`): double-entry ICC by zygosity; Falconer estimates; ACE/AE/CE/E maximum-likelihood fits with LRTs; pair-level bootstrap CIs; family-structure-preserving permutations; reliability-adjusted upper bounds; twin-identification accuracy; simulation-based power curves for the actual pair counts. Cross-check with OpenMx or SOLAR-Eclipse when available.
8. **Positive controls and comparisons**: alpha-power heritability; graph-metric heritability from the 2024 paper's definitions; fMRI CAP dynamics heritability in the large twin sample (secondary).
9. **Tools**: `mne`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, optional `pycrostates` for cross-checking segmentation, `boto3`/AWS CLI for data.

## Evaluation & statistics

- Reliability first: ICC(2,1) over 3 runs for every parameter; parameters with ICC < 0.4 are reported but not interpreted genetically.
- Heritability: ACE with LRT (A dropped; C dropped), reported with bootstrap CIs and the reliability ceiling; Falconer estimates alongside; permutation nulls that shuffle pair membership within zygosity (1,000+ permutations).
- Multiple comparisons: Benjamini-Hochberg across microstate parameters within each family (durations, occurrences, transitions); primary pre-registered outcomes are mean duration and occurrence of the state with the largest coverage, plus the DFA exponent.
- Confounds: age, sex, head-position distance, run order, recording day; MZ/DZ groups compared on these; covariate-adjusted residuals used for twin modelling.
- Cardiac coupling nulls: phase-shuffled and circularly-shifted R-peak trains; false-discovery control over states x phase bins.
- Leakage: group templates are computed with leave-one-family-out when used to back-fit a subject whose co-twin's data would otherwise shape the template (a subtle source of MZ similarity).
- Power: report the minimum h^2 detectable at 80% power for the observed pair counts and reliabilities (H7).

## Publishable angle

- **Headline**: "MEG microstate durations and occurrence are reliable (ICC ~0.7) and heritable (h^2 ~ X) in HCP twins, transition structure is not resolvable at this sample size, part of the sensor-space MZ similarity is head-shape, and microstate dynamics covary with vagal tone." Even a null result with power curves and reliability ceilings is publishable, because the endophenotype claim currently rests on family studies alone.
- Target venues: *NeuroImage*, *Human Brain Mapping*, *Brain Topography* (microstate methods community), *Psychophysiology* (cardiac coupling), *Cerebral Cortex*.
- Follow-ups: extend to EEG twin cohorts with larger n as they open (or via collaboration) using the same pre-registered pipeline; polygenic-score association of microstate parameters in large EEG biobanks; microstate-HRV coupling as an autonomic marker in clinical MEG.

## Risks, confounds & mitigations

- **Small twin sample**: 17 MZ + 12 DZ pairs cannot resolve modest h^2. Mitigation: power simulation reported up front; emphasis on ICC contrasts, twin identification (well powered) and reliability; pre-registered replication plan.
- **Head position / head shape**: explicitly modelled (H3); source-space arm.
- **ICA removal of cardiac artefact could remove heartbeat-evoked neural signal**: analyse cardiac coupling with and without the cardiac ICA component removed; report both.
- **Restricted-data approval delays**: zygosity is required; start the application in month 1; sensor pipeline and reliability analyses need no restricted data.
- **Polarity convention in MEG microstates**: run both polarity-invariant and polarity-sensitive segmentation; report agreement.
- **Template selection (number of states)**: fix by pre-registered criterion (cross-validation + GEV elbow), report sensitivity for 4-7 states.

## Milestones

- [ ] HCP registration, AWS credentials, Restricted Data application submitted; download MEG resting runs + ECG + head models for all MEG subjects.
- [ ] Sensor-space microstate pipeline; number-of-states selection; group templates; per-subject parameters for 3 runs; ICC table.
- [ ] Head-position extraction; head-similarity matrices; alpha-power positive control.
- [ ] HRV extraction; cardiac-phase coupling with nulls.
- [ ] Twin modelling (ICC, Falconer, ACE, permutations, twin identification, power curves) on covariate-adjusted parameters.
- [ ] Source-space microstates; sensor-vs-source heritability comparison.
- [ ] Secondary: fMRI CAP dynamics heritability; Cam-CAN/OMEGA template replication.
- [ ] Manuscript + code release (no restricted data).

## Ethics / data-use notes

- HCP Open Access data require accepting the WU-Minn HCP Consortium Open Access Data Use Terms; Restricted Data (zygosity, family structure, exact age) require separate approval and must never be redistributed, committed, or uploaded to third-party services. Publish only aggregate twin statistics; never list family IDs.
- ECG-derived HRV is health-related information; report only aggregates.
- Cam-CAN and OMEGA have their own agreements; cite as required.
- Never commit MEG/ECG data, head-shape digitisations or restricted CSVs; `data/` is git-ignored.

## Related projects (kept self-contained here)

- `connectome-heritability-gradients` (twin modelling of connectome features in HCP; shares the ACE machinery conceptually).
- `sleep-spindle-aging-biomarker` (electrophysiological biomarkers and reliability).
