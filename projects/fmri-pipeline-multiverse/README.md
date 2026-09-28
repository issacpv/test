# fMRI pipeline multiverse at dataset scale: specification curves, a pipeline-robustness score and a fragility model for published task-fMRI findings

**Pitch.** NARPS showed that 70 teams disagree on one dataset; this project turns the question around — for *many* OpenNeuro task-fMRI datasets with shipped fMRIPrep derivatives, run one automated multiverse of defensible pipeline choices, summarise every published finding as a specification curve with a pipeline-robustness score (PRS), and model which dataset characteristics (n, design, TR, motion, effect size) make a finding fragile.

| | |
|---|---|
| **Status** | Design + starter code; no results yet |
| **Difficulty / timeline** | MSc-level, 6–9 months for the parcel-level multiverse on 5–7 datasets; +3 months for the fMRIPrep-version arm |
| **Compute** | Parcel-level GLM multiverse: ~360 specs × ~700 subject-runs ≈ minutes-hours on a workstation (NumPy path). Voxel-wise arm (nilearn) for the 8 primary pipelines: ~1 day on 16 cores. fMRIPrep re-runs (version arm, 2 versions × 30 subjects × 2 datasets): 600–1,000 CPU-h on a cluster. Storage ~0.5 TB for preprocessed BOLD. |
| **Package** | `src/fmri_multiverse` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                                   # synthetic confounds/design/GLM/spec-curve tests
python scripts/download_data.py --check                     # which registry datasets ship fMRIPrep derivatives (GraphQL)
python scripts/download_data.py --dataset ds002785 --task workingmemory --bold --n-subjects 50
```

```python
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from fmri_multiverse import fetch, confounds, glm, speccurve

runs = fetch.find_runs("data/ds002785", task="workingmemory")            # BOLD + confounds + events per subject
specs = glm.enumerate_multiverse()                                        # 360 PipelineSpec objects
effects = {}                                                              # {spec.label: [per-subject ROI effect]}
for spec in specs:
    for r in runs:
        reg, mask = confounds.load_confounds_for(r.bold, spec.confounds)  # nilearn if present, else TSV fallback
        ts = np.load(f"work/{r.subject}_{spec.parcellation}.npy")         # (T, n_parcels) parcel time series
        res = glm.run_first_level_numpy(ts, pd.read_csv(r.events, sep="\t"), tr=2.0, spec=spec,
                                        confounds=reg, sample_mask=mask, contrasts={"wm": "active - passive"})
        effects.setdefault(spec.label, []).append(res["wm"]["effect"][ROI_INDEX])
E = np.array([effects[s.label] for s in specs])                          # (n_specs, n_subjects)
print(speccurve.multiverse_inference(E, n_perm=5000))                    # PRS, p_median, p_share, CIs
table = pd.DataFrame([s.as_dict() for s in specs]).assign(effect=E.mean(1), p=glm.second_level_onesample(E.T)["p"])
print(speccurve.robustness_score(table)); print(speccurve.variance_decomposition(table, "effect", list(glm.DEFAULT_GRID)))
```

Repository layout:

```
src/fmri_multiverse/     fetch (registry, GraphQL, S3, DataLad) · confounds · glm · speccurve
scripts/download_data.py --check / --sample / --dataset ... (openneuro-py, AWS CLI or DataLad)
data/README.md           download routes, sizes, parcellations, fMRIPrep re-run recipe, ground-truth results
tests/                   synthetic confounds table, planted GLM effects, null/signal multiverses
```

## Background

Task-fMRI analysis involves thousands of defensible choices (Carp, 2012, *Front Neurosci*). NARPS (Botvinik-Nezer et al., 2020, *Nature*) found substantial disagreement across 70 teams on the same data; Bowring et al. (2019, 2022, *HBM*) isolated software and pipeline sources of variability; Dafflon et al. (2022, *Nat Commun*) introduced *guided* multiverse exploration; Germani et al. (2025, *Sci Data*) released HCP Multi-Pipeline (24 pipelines × 1,080 subjects); Li et al. (2024, *Nat Hum Behav*) showed low inter-pipeline agreement in rest-fMRI connectivity that is masked until data reliability is high; Wang et al. (2024, *PLoS Comput Biol*) built a continuous benchmark of fMRIPrep confound strategies through Nilearn's `load_confounds`; and the multiverse of graph-based fMRI has been catalogued (Comet toolbox, *Imaging Neuroscience* 2025). Inference over multiverses has matured in psychology (Steegen et al., 2016; Simonsohn, Simmons & Nelson, 2020, *Nat Hum Behav*) and is now being adapted to neuroimaging ("Statistical inference for same-data meta-analysis in neuroimaging multiverse analyses", *Imaging Neuroscience* 2025).

OpenNeuro (Markiewicz et al., 2021, *eLife*) now hosts > 1,000 datasets and ships fMRIPrep (Esteban et al., 2019, *Nat Methods*) derivatives for many of them (in-dataset for NARPS, CNP, AOMIC; via the OpenNeuroDerivatives organisation for others). This makes an *automated, dataset-scale* multiverse feasible for the first time without re-preprocessing.

## The research gap

**Done.** Single-dataset multiverses (NARPS; HCP Multi-Pipeline; Dafflon 2022 on HBN/others); confound-strategy benchmarks for *resting-state* connectivity (Ciric 2017; Parkes 2018; Wang 2024); catalogues of choices (Comet); inference methods for one multiverse.

**Missing.**
1. **Across datasets.** No study has run the same multiverse on many task-fMRI datasets and asked which *findings* are robust and which datasets are fragile — every published multiverse is one dataset, one task.
2. **A per-finding robustness score with inference.** Specification-curve summaries with a null (sign-flip / bootstrap) have not been applied to published fMRI contrasts; NARPS reported team-level agreement, not a pipeline-level score for each hypothesis.
3. **Predicting fragility.** Whether fragility is predictable from dataset characteristics (n, block vs. event design, TR/multiband, mean FD, number of trials, baseline effect size) is unknown — this is the actionable output for study design.
4. **Task-fMRI confound strategies.** The Ciric/Wang benchmarks are resting-state; the effect of 36P/aCompCor/AROMA/scrubbing on *task* effect sizes across datasets has not been quantified.
5. **fMRIPrep version as a factor.** Version-to-version drift of fMRIPrep outputs (e.g. 1.x shipped derivatives vs. 20.2 LTS vs. 24.x) on downstream results has not been measured against the other multiverse dimensions.

## Research questions / hypotheses

1. **H1 (heterogeneous robustness).** Across ≥ 12 published findings from ≥ 5 datasets, PRS ranges from < 0.3 to > 0.9; canonical sensory/motor contrasts (faces > scrambled, working-memory load) have PRS > 0.8, whereas value/affect contrasts (NARPS gain/loss in vmPFC/amygdala) have PRS < 0.5.
2. **H2 (which choices matter).** Confound strategy (especially GSR and scrubbing) and HRF model explain more between-specification variance in effect size than smoothing or parcellation (eta² ranking; confounds > HRF > smoothing > parcellation > noise model).
3. **H3 (fragility predictors).** PRS increases with n and with block designs and decreases with mean FD and with event count < 30 per condition; a ridge model on dataset characteristics predicts PRS with leave-one-finding-out r > 0.5.
4. **H4 (version drift is small).** fMRIPrep version (shipped 1.x vs. 20.2.7 vs. 24.x) accounts for < 10 % of specification variance for the same confound strategy — smaller than any single analytic choice.
5. **H5 (multiverse inference agrees with NARPS).** For the nine NARPS hypotheses, the sign-flip multiverse p (`p_median`) and PRS rank-order the hypotheses in the same order as the fraction of NARPS teams reporting a significant result (Spearman ρ > 0.7).
6. **H6 (GSR flips signs).** Strategies with GSR change the sign of the median effect for at least one affective contrast (documented "anti-correlation" artefact), producing bimodal specification curves.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| ds001734 NARPS mixed gambles | 4 runs event-related; gain/loss parametric contrasts; nine hypotheses with 70-team results | 108 subjects | Open (CC0) | https://openneuro.org/datasets/ds001734 |
| ds000030 UCLA CNP | Stop-signal, BART, SCAP, task-switching (+ rest); fMRIPrep + MRIQC shipped | 272 (130 controls) | Open | https://openneuro.org/datasets/ds000030 |
| ds002785 AOMIC-PIOP1 | Working memory, emotion matching, faces, Stroop, anticipation; fMRIPrep 1.3.2 + physio | 216 | Open | https://openneuro.org/datasets/ds002785 |
| ds002790 AOMIC-PIOP2 | Working memory, emotion matching, stop-signal | 226 | Open | https://openneuro.org/datasets/ds002790 |
| ds003097 AOMIC-ID1000 | Movie watching (ISC arm only) | 928 | Open | https://openneuro.org/datasets/ds003097 |
| ds000117 Wakeman & Henson | Faces > scrambled (fMRI part) — **verify derivatives** | 16 | Open | https://openneuro.org/datasets/ds000117 |
| ds000228 Richardson pixar | Naturalistic ToM/pain events; children + adults (high motion) — **verify derivatives** | 155 | Open | https://openneuro.org/datasets/ds000228 |
| OpenNeuroDerivatives | fMRIPrep/MRIQC derivative datasets for further OpenNeuro datasets | ~1,000 repos | Open (DataLad) | https://github.com/OpenNeuroDerivatives |
| HCP Multi-Pipeline (external validation) | 24 pipelines × 5 contrasts group maps | 1,080 subjects | Open (public-data release) | Germani et al. 2025 |

Run `python scripts/download_data.py --check` to confirm derivative availability before committing compute; the registry (`fetch.DATASETS`) records which datasets were verified.

## Methods

1. **Fetch** (`fetch`): registry of datasets and published findings (task, contrast, a-priori ROI, source paper); GraphQL snapshot listing; download via openneuro-py / anonymous S3 / DataLad; `find_runs` pairs BOLD, confounds and events; `dataset_characteristics` reads sidecars for the fragility model.
2. **Multiverse grid** (`glm.enumerate_multiverse`, default 360 specs): smoothing {0, 4, 8 mm} × confounds {6HMP, 24HMP+8Phys, 36P (+GSR), aCompCor, 36P+scrubbing FD > 0.5} × HRF {SPM, SPM+derivative, Glover} × high-pass {0.008, 1/128 Hz} × parcellation {Schaefer-200, -400} × noise {OLS, AR(1)}. Optional dimensions: AROMA (where available), fMRIPrep version (shipped / 20.2.7 / 24.x re-runs on subsets), physio regressors (AOMIC).
3. **Confounds** (`confounds.select_confounds`): strategies implemented on the fMRIPrep TSV (derivatives/powers computed if missing in old versions), aCompCor mask selection from the JSON sidecar, Power-style segment removal; `nilearn_kwargs()` reproduces each strategy in `nilearn.interfaces.fmriprep.load_confounds` for the voxel-wise arm.
4. **First level** (`glm.run_first_level_numpy` on parcel time series; `run_first_level_nilearn` for voxel maps): own SPM/Glover HRFs, cosine drifts (or fMRIPrep cosines), scrubbing by row deletion after convolution, OLS or AR(1) prewhitening, contrast t-maps.
5. **Second level** (`glm.second_level_onesample`, `sign_flip_max_t`): group mean/t/Cohen's d per parcel and within the a-priori ROI; FWE by sign-flip max-T within a spec.
6. **Specification curves** (`speccurve`): per finding, effect in the a-priori ROI across all specs; PRS, sign consistency, analytic SNR, vibration; eta² variance decomposition across pipeline dimensions.
7. **Multiverse inference** (`multiverse_inference`): sign-flip null of the median effect and of the share of significant specs (Simonsohn 2020 adapted to one-sample designs); subject bootstrap CIs for PRS.
8. **Fragility model** (`fit_fragility_model`): leave-one-finding-out ridge from dataset/design features (n, TR, volumes, mean FD, % FD > 0.5, multiband, block vs event, events per run, conditions, |median effect|) to PRS; mixed model with dataset random intercept in statsmodels as a check.
9. **Validation against NARPS**: nine hypotheses vs. team-level agreement; against HCP Multi-Pipeline group maps for the 5 contrasts (Dice of thresholded maps across pipelines vs. our PRS).

**Findings registry** (`fetch.DATASETS`; one PRS per row; ROIs from the source papers, both atlas label and 6-mm sphere):

| Dataset | Task | Contrast | A-priori ROI | Source | Derivatives |
|---|---|---|---|---|---|
| ds001734 | MGT | gain (parametric) | vmPFC, ventral striatum (NARPS H1–H4) | Botvinik-Nezer 2020 | shipped (1.1.4) |
| ds001734 | MGT | loss (parametric) | amygdala, vmPFC (H5–H9) | Botvinik-Nezer 2020 | shipped (1.1.4) |
| ds000030 | stopsignal | STOP_SUCCESS − GO | right IFG, pre-SMA | Poldrack 2016 | shipped |
| ds000030 | bart | ACCEPT − REJECT | ventral striatum | Poldrack 2016 | shipped |
| ds000030 | scap | load 4 − load 1 | dlPFC, IPS | Poldrack 2016 | shipped |
| ds002785 | workingmemory | active − passive | dlPFC, IPS | Snoek 2021 | shipped (1.3.2) |
| ds002785 | emomatching | emotion − control | amygdala, fusiform | Snoek 2021 | shipped (1.3.2) |
| ds002785 | gstroop | incongruent − congruent | dACC | Snoek 2021 | shipped (1.3.2) |
| ds002790 | workingmemory | active − passive | dlPFC, IPS | Snoek 2021 | shipped (1.3.2) |
| ds002790 | stopsignal | stop − go | right IFG, pre-SMA | Snoek 2021 | shipped (1.3.2) |
| ds000117 | facerecognition | faces − scrambled | FFA, OFA | Wakeman & Henson 2015 | **verify** |
| ds000228 | pixar | mental − pain (reverse correlation) | TPJ, mPFC | Richardson 2018 | **verify** |
| ds003097 | moviewatching | inter-subject correlation (ISC arm) | visual, auditory | Snoek 2021 | shipped (1.3.2) |

Null-contrast controls (odd − even trials, or run 1 − run 2 of the same condition) are added for every dataset; their PRS calibrates the score (expected ≈ α).

## Evaluation & statistics

* **Unit of analysis:** finding (dataset × contrast × ROI); ≥ 12 findings from ≥ 5 datasets; ROIs pre-specified from the source papers (no ROI search).
* **Primary endpoints:** PRS with bootstrap CI; `p_median` and `p_share` from 5,000 sign flips; eta² per pipeline dimension with bootstrap CI.
* **Fragility model:** LOO r and RMSE; coefficients with bootstrap CIs; no p-values (n_findings small). Dataset-level random effects to avoid pseudo-replication of findings within a dataset.
* **Multiple comparisons:** within-spec FWE by max-T sign-flipping; across findings no correction needed for descriptive PRS; H5/H3 correlations tested once each.
* **Nulls:** sign-flip (one-sample), subject bootstrap, and a *null multiverse* built from a contrast that should be empty (e.g. odd vs even trials) — PRS should be ≈ α-level.
* **Leakage/circularity:** ROIs from the original papers only; grid pre-registered; no specification selected post hoc; the fragility model is fit only after all PRS values are frozen.
* **Robustness of the score itself:** PRS computed with α ∈ {0.05, 0.01, 0.001} and with FWE-corrected p; report all.

## Publishable angle

**Headline:** "Across N OpenNeuro datasets and M published findings, pipeline-robustness scores span the full range; confound strategy (GSR, scrubbing) is the single largest source of analytic variance in task effect sizes, fMRIPrep version is the smallest, and fragility is predictable from sample size, motion and trial count — providing a practical robustness benchmark for new studies." A pre-computed PRS table for classic contrasts and a reusable `fmri_multiverse` runner are secondary deliverables.

**Venues:** *Nature Human Behaviour* (follows NARPS/Dafflon lineage), *Imaging Neuroscience*, *NeuroImage*, *Scientific Data* (for the PRS resource + code); OHBM.

**Follow-ups:** extend to functional connectivity BWAS effects (link with Li et al. 2024); "living" multiverse that recomputes when new fMRIPrep versions or derivatives appear (continuous evaluation, à la Wang et al. 2024); Bayesian model averaging across specs.

## Risks, confounds & mitigations

| Risk | Mitigation |
|---|---|
| Shipped derivatives use very old fMRIPrep (1.1–1.3) with different confound naming | `select_confounds` handles old/new names and computes missing derivative/power columns; version arm quantifies drift |
| Not every dataset ships derivatives (ds000117/ds000228 unverified) | `--check` live query; OpenNeuroDerivatives fallback; drop or re-run subsets |
| Specification grid is itself a choice (garden of forking multiverses) | Pre-registered grid; sensitivity to removing any dimension; report eta² so readers can re-weight |
| Correlated specs inflate "share significant" | Sign-flip null recomputes the *whole* grid per permutation, preserving dependence |
| Few findings → weak fragility model | Treat as descriptive; expand with OpenNeuroDerivatives datasets; report CIs |
| Compute for voxel-wise arm | Parcel-level NumPy path for the full grid; voxel-wise only for 8 primary pipelines |
| ROI definitions from papers are imprecise | Use both an atlas label and an MNI sphere (6 mm) per finding; report agreement |
| Group heterogeneity (CNP patients, children in ds000228) | Controls-only primary analysis; group as covariate in sensitivity |

## Milestones

- [ ] Verify derivatives (`--check`), freeze dataset/finding registry with a-priori ROIs, pre-register grid on OSF
- [ ] Download confounds/events/BOLD for 5 core datasets; extract Schaefer-200/400 parcel time series
- [ ] Run parcel-level multiverse (360 specs) for all findings; store per-subject effects
- [ ] Specification curves, PRS, variance decomposition, sign-flip inference; null-contrast multiverse
- [ ] Voxel-wise arm (8 primary pipelines) for figures and ROI-sphere sensitivity
- [ ] fMRIPrep version arm (20.2.7 and 24.x re-runs on 30-subject subsets of NARPS and PIOP1)
- [ ] Fragility model; validation against NARPS team agreement and HCP Multi-Pipeline
- [ ] Release PRS table + code; manuscript

## Ethics / data-use notes

* All datasets are CC0 on OpenNeuro; cite each dataset paper and the OpenNeuro accession. Re-check individual dataset licences before redistribution of derived maps (NeuroVault upload is encouraged).
* No participant-level data are committed; DataLad/annex content stays outside git (see `.gitignore`).
* Derived outputs (parcel effects, PRS tables) contain no identifiers and can be shared.
* Do not upload subject-level imaging to third-party LLM/API services; only aggregate statistics leave the compute environment.
* Report software versions (fMRIPrep, nilearn, TemplateFlow atlases) and container digests for every arm — the project is about reproducibility.
