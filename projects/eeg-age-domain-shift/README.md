# Pediatric-to-adult EEG domain shift as a continuous function of age: transfer curves, embedding drift and age-conditioned normalization for seizure detection

**One-sentence pitch.** Treat patient age as a *continuous* domain variable rather than a pediatric/adult dichotomy: train seizure detectors on age-binned slices of TUSZ (0-90 y), CHB-MIT (1.5-22 y) and Siena (20-71 y), measure how detection performance decays with the (log-)age distance between training and test populations, test whether EEG foundation-model embeddings drift with age in a way that predicts that loss, and evaluate age-conditioned normative normalization of background EEG as a cheap, label-free way to flatten the transfer curve.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis (spectral/logistic transfer matrix, ~6 months) to PhD chapter (foundation-model embeddings, normative models, ~12 months).
- Compute: CPU is enough for spectral features, normative models and the full transfer matrix (TUSZ ~1,600 h at 256 Hz reduces to a few GB of window features). A single 16-24 GB GPU is needed to extract frozen embeddings from BIOT/LaBraM/EEGPT/CBraMod-class models and for optional fine-tuning. Storage ~150 GB.
- Related project: `cross-dataset-seizure-generalization` (leakage audit, montage harmonization, scorer sensitivity). This project is self-contained; it asks a different question (age as a dose variable) and shares no code.

## Background

Scalp EEG changes systematically across the lifespan: the posterior dominant rhythm rises from ~3-4 Hz in infancy to ~8 Hz by age 3 and ~10 Hz in adolescence, slow-wave power falls through childhood, and amplitude decreases with age (classical quantitative work: Matousek & Petersen, 1973; developmental spectra reviewed in Segalowitz, Santesso & Jetha, 2010, *Int. J. Psychophysiol.*). These are the same spectral dimensions that seizure detectors use, so a model trained on one age range sees a *different background distribution* at another age, and ictal-vs-background boundaries shift. Machine-learning "EEG brain age" models exploit exactly this (Sun et al., 2019, *Neurobiol. Aging*, sleep EEG; Vandenbosch et al., 2019, *NeuroImage*, children and adolescents; Engemann et al., 2022, *NeuroImage*, reusable M/EEG brain-age benchmark on TUAB and Cam-CAN), yet those models have never been turned around and used as *shift diagnostics* for a downstream clinical task.

The two most-used open seizure corpora sit at opposite ends of the age range: CHB-MIT is pediatric (Shoeb, 2009; ages 1.5-22 y) and Siena is adult (Detti et al., 2020, *Processes*; 20-71 y), while TUSZ (Shah et al., 2018, *Front. Neuroinform.*) spans neonates to > 90 y with age recorded in the EDF patient header, which makes it possible to build an *age ladder within one site* and to separate age shift from site shift.

## The research gap

**What has been done.**

- Pediatric-vs-adult transfer as a dichotomy: a 2024 *BioMedInformatics* study trained on the TUH pediatric/adolescent subset with age as an extra feature and reached ~99% accuracy internally but ~65% on CHB-MIT as an external test; recent reports (2025) describe near-chance balanced accuracy (~0.51) when adult-TUH-trained detectors are applied unchanged to CHB-MIT. Both treat age as a binary group.
- Cross-dataset benchmarks: EEG-FM-Bench (arXiv 2508.17742), EEG-Bench (arXiv 2512.08959) and RobustSeiz (arXiv 2609.04007, 2026) evaluate detectors across CHB-MIT / TUSZ / Siena / SeizeIT1 / Helsinki, and "What EEG foundation models encode: dataset identity ..." (arXiv 2607.24519, 2026) shows that foundation-model embeddings carry dataset identity. None stratifies by age or models performance as a function of age distance.
- EEG brain age: Engemann et al., 2022 (*NeuroImage*) benchmark filter-bank Riemannian and deep models for age prediction on TUAB and Cam-CAN; Sabbagh et al., 2020 (*NeuroImage*) give the covariance-based regression theory; Gemein et al., 2020 (*NeuroImage*) show age is a strong covariate in TUAB pathology decoding. Age prediction from *epilepsy* EEG and its use as a covariate for detector transfer are unexplored.
- Normative modelling: Marquand et al., 2016 (*Biol. Psychiatry*) and Rutherford et al., 2022 (*eLife*) establish age-conditioned normative models for neuroimaging; the equivalent for EEG background spectra as a pre-processing step for detectors has not been reported.

**What is specifically missing.**

1. **A transfer curve, not a transfer table.** How many AUROC/F1 points does each decade (or log-year) of age distance cost, does the cost depend on model capacity, and is it symmetric (adult -> child vs child -> adult)?
2. **Decomposition of the CHB-MIT <-> TUSZ gap into age and site.** Within-TUSZ age bins give the age-only component; CHB-MIT and Siena give the site + age component at matched ages.
3. **Embedding drift as a predictor of loss.** Whether frozen foundation-model embeddings encode age (probe R^2) and whether between-bin embedding distance (MMD, centroid distance) predicts transfer loss.
4. **Label-free mitigation.** Age-conditioned z-scoring of background spectral features using a normative model fitted on TUAB (which has age but no seizure labels) as a domain-adaptation step, compared with age-importance weighting and with plain per-record standardization.
5. **Data value vs age distance.** For a fixed target age bin, the marginal value of one extra age-matched training subject vs several far-age subjects, which speaks directly to how pediatric datasets should be built.

## Research questions / hypotheses

1. **H1 (decay).** Test AUROC of a detector trained on age bin *i* and tested on bin *j* decreases approximately linearly in |log(age_i + 1) - log(age_j + 1)| (bin medians); the fitted slope is negative with a 95% bootstrap CI excluding zero for every model family.
2. **H2 (capacity).** The decay slope is steeper for the gradient-boosting and foundation-model linear-probe families than for the spectral logistic baseline.
3. **H3 (age vs site).** Within-TUSZ age-only decay accounts for >= 50% of the CHB-MIT -> TUSZ-adult drop after montage harmonization; the residual (site) component is smaller than the age component.
4. **H4 (asymmetry).** Adult -> pediatric transfer loses more than pediatric -> adult at equal age distance (higher-amplitude, slower pediatric background overlaps ictal spectral signatures).
5. **H5 (embedding drift).** Frozen FM embeddings predict age with R^2 > 0.5 (subject-grouped ridge probe), and MMD between age-bin embedding clouds correlates with transfer loss (Spearman rho > 0.6 across (i, j, model) triplets).
6. **H6 (mitigation).** Age-conditioned normative z-scoring reduces the decay slope by >= 30% relative to per-record standardization; age-importance weighting reduces it by less.
7. **H7 (data value).** Adding k age-matched subjects to a far-age training set improves target AUROC more than adding 5k far-age subjects, for k in {2, 5, 10}.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| TUH EEG Seizure Corpus (TUSZ v2.0.3) | All splits; age parsed from the EDF patient field; seizure terms from `*.csv_bi`; within-site age ladder (0-90+ y) | ~1,600 h, ~70 GB | Free registration (TUH data-use form; rsync) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| TUH EEG Abnormal Corpus (TUAB) | Age-labelled background EEG (no seizure labels) for fitting the normative age model and the age-probe reference; same site as TUSZ | ~2,300 recordings | Free registration (same form) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| CHB-MIT Scalp EEG | 23 pediatric subjects, ages in `SUBJECT-INFO`, bipolar 256 Hz, ~198 seizures | ~980 h, ~42 GB | Open (PhysioNet) | https://physionet.org/content/chbmit/1.0.0/ |
| Siena Scalp EEG | 14 adults, ages in `subject_info.csv`, referential 512 Hz, 47 seizures | ~128 h, ~20 GB | Open (PhysioNet) | https://physionet.org/content/siena-scalp-eeg/1.0.0/ |
| Helsinki neonatal EEG | 79 term neonates (age ~0), 3 annotators; extreme end of the age ladder | ~74 h, ~4 GB | Open (Zenodo, Stevenson et al., 2019) | https://zenodo.org/ (search "A dataset of neonatal EEG recordings with seizure annotations") |
| Optional: FM checkpoints (BIOT, LaBraM, EEGPT, CBraMod) | Frozen embeddings for the drift analysis; note TUH-pretraining overlap | small | Open (GitHub releases) | see model repositories |

Age bins used throughout (years): [0, 1) neonate/infant, [1, 6) preschool, [6, 12) school-age, [12, 18) adolescent, [18, 40) young adult, [40, 65) middle age, [65, 120) older adult. Bin membership counts per corpus are produced by `scripts/download_data.py --build-manifests` (`manifests/ages.csv`).

## Methods

1. **Ages (`eegage.ages`).** Parse ages from TUH EDF headers (`Age:NN` in the local-patient field), CHB-MIT `SUBJECT-INFO`, Siena `subject_info.csv`; Helsinki is age 0. Records with missing age are excluded from the ladder but kept for site-level analyses. Log-age distance between bins uses bin medians.
2. **Harmonization.** Common 18-pair longitudinal bipolar montage, 256 Hz, 0.5-70 Hz band-pass, 50/60 Hz notch; 4-s windows with 2-s step; a window is positive if >= 50% overlaps a seizure term. (Implementation as in the sibling project; the modules here take harmonized arrays.)
3. **Features (`eegage.features`).** Per channel: log absolute and relative power in delta/theta/alpha/beta/gamma, spectral edge 95%, peak frequency in 3-13 Hz (posterior-dominant-rhythm proxy), line length, Hjorth mobility and complexity. Foundation-model embeddings (optional) are mean-pooled per window.
4. **Normative age model (`eegage.features.AgeNormativeScaler`).** For each feature, mean and log-SD are smooth polynomial functions of log-age fitted on *background* windows of the training corpus (TUAB + non-seizure TUSZ training windows); every window is then z-scored given its subject's age. Alternatives: per-record standardization (baseline), age-importance weighting (`eegage.drift.age_importance_weights`).
5. **Transfer matrix (`eegage.transfer.transfer_matrix`).** For each model family, train on each age bin (subject-wise, seizure-balanced) and test on every other bin; diagonal cells via subject-grouped 5-fold CV. Model families: standardized logistic regression, HistGradientBoosting, compact CNN (optional), FM linear probes (optional).
6. **Decay model (`eegage.transfer.fit_decay`).** Cell-level regression of performance on log-age distance with a same-site indicator; cell bootstrap (resampling test subjects within cells) for CIs; a mixed model (training bin as random intercept) in `statsmodels` for the paper.
7. **Drift diagnostics (`eegage.drift`).** RBF-MMD with permutation p-value and centroid cosine distance between bins, in feature space and in embedding space; subject-grouped ridge probe of age from embeddings; Spearman correlation of drift with loss.
8. **Data-value curves (`eegage.transfer.sample_value_curve`).** Starting from a far-age training set, add k age-matched vs 5k far-age subjects and record target AUROC.

## Evaluation & statistics

- Primary metric: window-level AUROC (robust to threshold transfer); secondary: event-level F1 and sensitivity at 1 FP/24 h using the SzCORE-style scorer of the sibling project, at a threshold fixed on the training bin.
- Uncertainty: subject-level bootstrap (2,000 draws) within each cell; paired bootstrap for model-family or normalization comparisons.
- Decay slope inference: cell-level bootstrap CI; mixed model with training-bin random intercept; capacity x distance interaction term for H2; asymmetry (H4) via a signed-distance term.
- Leakage prevention: subject ids from file paths; a subject never appears in more than one bin's training set and its own test cell; normative models are fitted on training subjects only; thresholds fixed on training bins.
- Nulls: age labels permuted across subjects (destroys the age structure while preserving everything else) -> the decay slope should vanish; embedding-drift/loss correlation tested against the same permutation null.
- Multiple comparisons: Holm-Bonferroni within each hypothesis family (H1-H7 are pre-registered families); the full transfer matrix is reported as a figure without cell-wise tests.

## Publishable angle

- **Headline result.** "Each log-year of age distance between training and test populations costs X AUROC points; adult-to-child transfer is Y times worse than the reverse; age explains Z% of the CHB-MIT -> TUSZ gap; and a label-free age-conditioned normative z-scoring recovers W% of the loss." Delivered as an age-stratified leaderboard and a normative EEG-background model for 0-90 y.
- **Venues.** *Clinical Neurophysiology* or *Epilepsia* (clinical); *NeuroImage* or *Journal of Neural Engineering* (methods, normative modelling); *NeurIPS Datasets & Benchmarks* / *MLHC* for the leaderboard.
- **Follow-ups.** Age-conditioned foundation models (age token); extension to abnormal-EEG classification (TUAB) and IED detection; combining with the sibling project's scorer-sensitivity analysis; sex- and medication-stratified curves.

## Risks, confounds & mitigations

- **Age is entangled with etiology and recording context** (neonatal HIE, pediatric epileptic encephalopathies, adult ICU EEG in TUSZ). Mitigation: report seizure-type composition per bin; sensitivity analysis restricted to focal seizures; Helsinki as a labelled extreme, not a fair target.
- **Sparse bins.** TUSZ has few subjects in some pediatric bins. Mitigation: merge adjacent bins when < 15 subjects; report subject and seizure counts per bin; bootstrap CIs.
- **Missing or wrong header ages.** Mitigation: exclude missing ages; flag implausible values (> 110 y, < 0); cross-check TUAB and TUSZ ages for shared patients.
- **Montage/sampling differences across corpora.** Mitigation: harmonized montage; within-TUSZ ladder as the site-controlled reference.
- **FM pretraining overlap with TUH.** Mitigation: mark contaminated cells; rely on CHB-MIT/Siena for clean external checks.
- **Window autocorrelation.** Mitigation: all inference at subject level.

## Milestones

- [ ] Register for TUH; download TUSZ and TUAB; download CHB-MIT, Siena, Helsinki.
- [ ] Build `records.csv`, `events.csv`, `ages.csv`; bin-count table per corpus.
- [ ] Harmonize montage/sampling; extract spectral features for all windows.
- [ ] Fit the normative age model on TUAB + TUSZ background; validate on held-out subjects (calibration of z-scores by age).
- [ ] Freeze age-bin subject folds (`splits/*.json`).
- [ ] Transfer matrices for logistic and gradient-boosting families; decay fits (H1, H2, H4).
- [ ] Age-vs-site decomposition with CHB-MIT and Siena (H3).
- [ ] Extract FM embeddings; drift diagnostics and age probes (H5).
- [ ] Normalization comparison (H6); data-value curves (H7).
- [ ] Release leaderboard + normative model; write paper.

## Ethics / data-use notes

- CHB-MIT and Siena (PhysioNet) and Helsinki (Zenodo) are de-identified and openly licensed; cite the dataset papers.
- TUSZ/TUAB require a signed TUH EEG data-use agreement; the data may not be redistributed. Credentials in `TUH_USERNAME` / `TUH_PASSWORD`, never in code or git.
- Ages are demographic data of patients; publish only binned counts and model outputs, never per-record ages joined to other identifiers.
- Do not send raw EEG from these corpora to third-party LLM/ML APIs; PhysioNet's responsible-use terms and the TUH agreement restrict third-party processing.
- Never commit data; `data/` and `*.edf` are git-ignored. Manifests containing only public ids, event times and age bins may be committed.
