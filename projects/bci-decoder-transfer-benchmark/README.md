# bci-decoder-transfer-benchmark

**A leave-one-dataset-out benchmark of calibration transfer for motor-imagery BCIs: how many target-subject calibration trials do Riemannian/Euclidean alignment, deep fine-tuning and EEG foundation models actually save, estimated with mixed-effects models across ~18 open MOABB datasets and PhysioNet EEGMMIDB.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (numpy/scipy SPD geometry with Euclidean and Riemannian alignment, tangent space, MDM classifier; calibration-budget learning-curve protocol; mixed-effects and negative-transfer statistics; synthetic multi-subject generator).
- Difficulty: MSc thesis (Riemannian arm only) to PhD chapter (with deep and foundation-model arms).
- Timeline: 6-12 months (1 month data + channel harmonisation, 2 months Riemannian/EA arms, 2-3 months deep/foundation arms, 2 months statistics and writing).
- Compute: Riemannian/EA arms run on a laptop (covariances of ~50k trials). Deep fine-tuning and foundation-model embeddings need one GPU (8-16 GB) for a few days; foundation-model *feature extraction* (frozen backbone + linear probe) is cheap and is the pre-registered primary deep arm.

## Background

Motor-imagery (MI) BCIs need per-user calibration because EEG covariance structure differs between people and sessions. Transfer learning promises to shorten calibration by re-using data from other subjects or datasets (Lotte et al., 2018, Journal of Neural Engineering; Wu, Jiang & Peng, 2022, Neural Networks). The two dominant families are (i) covariance-space alignment: Riemannian re-centering (Zanini et al., 2018, IEEE TBME), Riemannian Procrustes analysis (Rodrigues, Jutten & Congedo, 2019, IEEE TBME) and Euclidean alignment (He & Wu, 2020, IEEE TBME), built on the Riemannian MDM/tangent-space classifiers of Barachant, Bonnet, Congedo & Jutten (2012, IEEE TBME) and reviewed by Yger, Berar & Lotte (2017, IEEE TNSRE); and (ii) deep networks (Schirrmeister et al., 2017, Human Brain Mapping; Lawhern et al., 2018, Journal of Neural Engineering; review: Roy et al., 2019, Journal of Neural Engineering) with fine-tuning, and since 2021 self-supervised EEG foundation models (BENDR: Kostas, Aroca-Ouellette & Rudzicz, 2021, Frontiers in Human Neuroscience; LaBraM: Jiang et al., 2024, ICLR; CBraMod: Wang et al., 2025, ICLR).

MOABB (Jayaram & Barachant, 2018, Journal of Neural Engineering) made rigorous comparison possible, and its 2024 reproducibility study re-implemented 30 pipelines on 36 datasets (14 MI, 15 P300, 7 SSVEP), finding Riemannian pipelines robust and many deep-learning claims fragile (Chevallier et al., 2024, preprint: "The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark"). Those evaluations are within-session, cross-session and cross-subject *within* a dataset.

## The research gap

**What has been done**

- Within-dataset benchmarks of decoders at full calibration (Chevallier et al., 2024; "Benchmarking brain-computer interface algorithms: Riemannian approaches vs convolutional neural networks", 2024, Journal of Neural Engineering).
- Euclidean alignment with deep learning evaluated systematically for cross-subject transfer on a handful of MI datasets (Junqueira et al., 2024, Journal of Neural Engineering; "Revisiting Euclidean alignment for transfer learning in EEG-based BCIs", 2025, preprint; "Combining Euclidean alignment and data augmentation for BCI decoding", 2024, preprint).
- The "cross-dataset variability problem" for deep EEG decoders was named and demonstrated on a few datasets (Xu et al., 2020, Frontiers in Human Neuroscience).
- Riemannian re-centering with few target trials was shown to cut calibration in single-dataset studies ("Minimizing subject-dependent calibration for BCI with Riemannian transfer learning", 2021, preprint).
- Two 2026 preprints show that *average* rankings hide strong per-subject heterogeneity in MI decoding ("Subject-level heterogeneity in EEG motor imagery decoding: a large-scale benchmark and portfolio-based reduction of the search space"; "Average rankings mask per-subject optimality: a Friedman-Nemenyi ..." analysis), which argues for subject-level, mixed-effects reporting.
- Foundation models report downstream accuracies after full fine-tuning on a few datasets, typically without a calibration-budget axis and without a Riemannian baseline under the same budget.

**What is missing (checked against 2023-2026 literature)**

1. No leave-one-dataset-out benchmark in which the *source* pool is every other MI dataset and the primary axis is the *number of target calibration trials* (0, 5, 10, 20, 40, 80, all). "Calibration savings" (trials needed to reach a target accuracy, with vs without transfer) has never been estimated across many datasets.
2. No head-to-head of Riemannian alignment (RA, RPA), Euclidean alignment (EA), tangent-space transfer, deep fine-tuning and frozen foundation-model embeddings under identical calibration budgets, channel sets and preprocessing.
3. Effect sizes are reported as mean accuracies. A mixed-effects treatment with subject nested in dataset, method x budget interactions, and dataset-level moderators (channel count, montage overlap, sampling rate, trials per class, cue type, feedback, number of classes, session structure) is absent, as is a *negative-transfer rate* (fraction of subjects harmed by transfer) with confidence intervals.
4. Channel-set harmonisation is a hidden moderator: datasets range from 3 to 128 channels. Whether alignment gains survive restriction to a common 8-16 channel motor montage, or interpolation to a canonical montage, is unreported.
5. Session-level structure (longitudinal datasets such as Stieger et al., 2021, Sci Data, with 7-11 sessions per subject) allows separating within-subject drift from between-subject shift; no transfer benchmark exploits it.

## Research questions / hypotheses

1. **H1 (calibration savings).** Riemannian re-centering with a multi-dataset source pool reaches 90% of a subject's asymptotic within-subject accuracy with at least 50% fewer target trials than from-scratch calibration, on average across datasets. Test: per-subject learning curves; trials-to-90%-asymptote compared with a paired mixed model (log-trials outcome).
2. **H2 (zero-shot floor).** With zero target trials, RA/EA transfer from the pooled source is above chance in every dataset but its gain over chance shrinks with decreasing channel overlap with the source montage. Test: mixed model of zero-shot accuracy on montage-overlap moderator.
3. **H3 (deep vs Riemannian under budget).** At budgets <= 20 trials, frozen foundation-model embeddings + tangent-space/linear probe do not outperform RA + MDM; at >= 80 trials, fine-tuned deep models catch up. Test: method x budget interaction in the mixed model; Holm-corrected pairwise contrasts at each budget.
4. **H4 (negative transfer is common and predictable).** 15-30% of subjects are harmed by transfer at low budgets; harm is predicted by the Riemannian distance between the subject's covariance mean and the source mean, and by the subject's within-subject accuracy (low performers gain, high performers lose). Test: logistic mixed model of "harmed" on these predictors.
5. **H5 (moderators).** Dataset-level moderators explain > 30% of between-dataset variance in transfer gain, with trials-per-class and channel count the largest. Test: meta-regression of dataset-level gains on moderators with REML tau^2.
6. **H6 (drift vs shift).** In longitudinal datasets, transfer from a subject's own earlier sessions is worth more than transfer from other subjects at any budget, and the difference shrinks with alignment. Test: three-way comparison in Stieger2021 and Lee2019 (2 sessions).

## Datasets

All datasets are downloaded through MOABB (`moabb.datasets`) unless stated; subject counts are approximate and should be verified from `dataset.subject_list`.

| Name (MOABB class) | What is used | Size | Access | URL |
|---|---|---|---|---|
| BNCI2014_001 (BCI Comp. IV 2a; Tangermann et al., 2012, Front Neurosci) | 4-class MI, 22 ch, 2 sessions | 9 subjects | Open | http://bnci-horizon-2020.eu/database/data-sets |
| BNCI2014_004 (BCI Comp. IV 2b) | 2-class MI, 3 ch, 5 sessions | 9 subjects | Open | same |
| BNCI2014_002, BNCI2015_001, BNCI2015_004 | 2- to 5-class MI, 13-30 ch | 9-14 subjects each | Open | same |
| PhysionetMI / EEGMMIDB (Schalk et al., 2004, IEEE TBME; Goldberger et al., 2000, Circulation) | 64-ch BCI2000 MI, 160 Hz | 109 subjects | Open (no registration) | https://physionet.org/content/eegmmidb/1.0.0/ |
| Cho2017 (Cho et al., 2017, GigaScience) | 2-class MI, 64 ch | 52 subjects | Open | via MOABB / GigaDB |
| Lee2019_MI (Lee et al., 2019, GigaScience) | 2-class MI, 62 ch, 2 sessions | 54 subjects | Open | via MOABB / GigaDB |
| Schirrmeister2017 (High-Gamma Dataset) | 4-class MI, 128 ch | 14 subjects | Open | via MOABB |
| Shin2017A, Weibo2014, Zhou2016, AlexMI, GrosseWentrup2009, Ofner2017 | assorted MI paradigms, 14-128 ch | 4-29 subjects each | Open | via MOABB |
| Stieger2021 (Stieger et al., 2021, Sci Data) | 2/4-class MI, 64 ch, 7-11 longitudinal sessions | 62 subjects | Open | via MOABB / Figshare |
| Dreyer2023 (Dreyer et al., 2023, Sci Data) | 2-class MI with user-profile questionnaires, 27 ch | 87 subjects | Open | via MOABB / Zenodo |
| Liu2024 | 2-class MI, 29 ch | ~50 subjects | Open | via MOABB |
| (secondary) MOABB P300 and SSVEP datasets | Replication of the budget protocol in ERP/SSVEP paradigms | 22 datasets | Open | via MOABB |

None require credentials; some hosts (GigaDB, Zenodo, Figshare) are slow, so `download_data.py` caches and resumes.

## Methods

1. **Harmonised preprocessing** (MOABB `MotorImagery` paradigm): 8-30 Hz band-pass, resample to 128 Hz, epoch 0.5-2.5 s post-cue (dataset-specific offsets recorded in a config table), common-average reference, three montage arms: (a) native channels, (b) common 8-channel motor set (FC3, FC4, C3, Cz, C4, CP3, CP4, Pz where present), (c) spherical-spline interpolation to a 32-channel canonical montage.
2. **Covariance features** (`src/bci_transfer/covariance.py`): Ledoit-Wolf shrinkage covariances; SPD utilities (matrix sqrt/log/exp via eigendecomposition); AIRM (Karcher) and log-Euclidean means; tangent-space vectorisation; **EA** (whitening by the arithmetic mean covariance), **RA** (congruence by the inverse square root of the Riemannian mean); MDM classifier. Production runs use `pyriemann` (RPA, TS+LR) and the built-in code as a cross-check.
3. **Transfer arms**: (T0) no transfer, target-only; (T1) EA source pool + target EA with k trials; (T2) RA + MDM; (T3) RA + tangent space + logistic regression with source pool; (T4) RPA (re-centering, re-scaling, rotation with k labelled target trials); (T5) EEGNet/ShallowConvNet trained on the pool, fine-tuned with k target trials (braindecode); (T6) frozen LaBraM / CBraMod embeddings + linear probe / tangent space, k target trials; (T7) same with full fine-tuning at budgets >= 40.
4. **Calibration-budget protocol** (`src/bci_transfer/learning_curves.py`): for each target subject, budgets k in {0, 5, 10, 20, 40, 80, all}; k trials sampled stratified by class, 10 random draws per budget; evaluation on the subject's held-out trials (never in the source pool). Fit an inverse-power learning curve acc(k) = a - b k^(-c); derive trials-to-90%-asymptote and area under the curve.
5. **Leave-one-dataset-out**: the source pool is every other MI dataset (subject-level covariances or raw epochs for deep arms); the target dataset's subjects are never in the pool. Also run leave-one-subject-out within dataset as the conventional baseline.
6. **Statistics** (`src/bci_transfer/mixed_effects.py`): long table (dataset, subject, method, budget, draw, accuracy, chance level, kappa). Mixed model: accuracy ~ method * log(k+1) + moderators, random intercepts for dataset and subject-in-dataset (statsmodels `MixedLM` with variance components). Paired effect sizes (Cohen's d_z, Wilcoxon) per budget; negative-transfer rate with Wilson CI; dataset-level meta-regression (REML tau^2) on moderators.
7. **Synthetic validation** (`src/bci_transfer/synthetic.py`): two-class SPD data with subject-specific congruence transforms and controllable shift, to verify that alignment removes the planted shift and that the learning-curve estimator is unbiased.
8. **Tools**: `moabb`, `mne`, `pyriemann`, `scikit-learn`, `braindecode` + `torch` (deep arms), `statsmodels`, `numpy`, `scipy`, `pandas`.

## Evaluation & statistics

- Primary metric: balanced accuracy (and Cohen's kappa to compare across class counts); chance-corrected gain = (acc - chance)/(1 - chance).
- Primary estimands: (1) trials-to-90%-asymptote per subject, method vs T0 (paired, log scale); (2) method x budget accuracy contrasts from the mixed model; (3) negative-transfer rate per method and budget.
- Leakage prevention: source pools exclude the target dataset entirely; target evaluation trials are never used for alignment statistics (EA/RA reference means for the target are computed from the k calibration trials only, or, in the unsupervised variant, from the evaluation trials *without labels*, reported separately as "transductive"); hyper-parameters selected by nested CV on the source pool only; foundation-model pre-training corpora are checked for overlap with every target dataset (LaBraM/CBraMod pre-training lists include some MOABB datasets; those targets are flagged and excluded from the foundation-model contrasts).
- Multiple comparisons: Holm over method contrasts within each budget; the pre-registered primary comparison is T2 vs T0 and T6 vs T2 at k = 10.
- Nulls: label-permutation null for accuracy at each budget; a "random source" null where the source pool is replaced by label-shuffled data to check that gains come from class information, not from covariance normalisation alone (which EA/RA provide even unsupervised).
- Heterogeneity: subject-level Friedman tests per dataset; report the fraction of subjects for which each method is within 2 percentage points of the best (portfolio view), following the 2026 heterogeneity preprints.
- Sensitivity: montage arm (a/b/c), band (8-30 vs 4-40 Hz), epoch window, covariance shrinkage, class subsets (left vs right hand only).

## Publishable angle

- **Headline**: "Across N open MI datasets, Riemannian re-centering from a multi-dataset pool cuts the calibration needed to reach 90% of asymptotic accuracy from X to Y trials; frozen EEG foundation models do not beat it below 20 trials; Z% of subjects are harmed by transfer, predictable from covariance distance and baseline performance." A public, versioned leaderboard keyed by calibration budget.
- Target venues: *Journal of Neural Engineering*, *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, *NeuroImage* (methods), *NeurIPS Datasets & Benchmarks* track (benchmark + code release), *Frontiers in Human Neuroscience* (BCI section).
- Follow-ups: online (pseudo-real-time) validation of calibration savings; extension to P300/SSVEP; per-subject method selection (portfolio) trained on covariance-distance features; contribution of a `CrossDatasetEvaluation` with calibration budgets to MOABB.

## Risks, confounds & mitigations

- **Paradigm heterogeneity** (cue types, feedback, class definitions): pool only compatible classes (left vs right hand as the core task; feet/tongue as secondary), record paradigm covariates as moderators.
- **Channel mismatch**: three montage arms; report gains as a function of overlap.
- **Foundation-model pre-training leakage**: document pre-training corpora, exclude overlapping targets, and report both with/without.
- **Compute for deep arms**: frozen-embedding arm is primary; fine-tuning only at high budgets and on a dataset subset if needed.
- **Subject-level heterogeneity dominating averages**: mixed models, per-subject reporting, portfolio metrics.
- **MOABB version drift**: pin `moabb`, `mne`, `pyriemann` versions; store the preprocessing config hash with every result row.
- **Small datasets (4-9 subjects)** inflate dataset-level variance: weight the meta-regression by subjects; report leave-one-dataset-out sensitivity of the pooled estimate.

## Milestones

- [ ] Data pull for all MI datasets; channel harmonisation config; per-dataset epoch/cue table; chance levels.
- [ ] Riemannian/EA arms (T0-T4) with the calibration-budget protocol; synthetic validation.
- [ ] Learning-curve fits and trials-to-90% per subject; first mixed model; negative-transfer rates.
- [ ] Deep arms (T5) and frozen foundation-model arm (T6); leakage audit of pre-training corpora.
- [ ] Moderator meta-regression; longitudinal drift-vs-shift analysis (Stieger2021, Lee2019).
- [ ] Sensitivity analyses (montage, band, window); P300/SSVEP replication of the protocol.
- [ ] Public results table + code release; manuscript.

## Ethics / data-use notes

- All datasets are public, de-identified EEG released under open licenses (check each MOABB dataset's citation and license; some request citation of the original paper). No credentialed data.
- Dreyer2023 includes questionnaire data (user profiles); use only aggregate covariates and do not attempt re-identification.
- Never commit raw EEG (`.gdf`, `.edf`, `.mat`, `.fif`) or MOABB caches; commit only result tables and configs.
- Foundation-model weights have their own licenses; check redistribution terms before packaging embeddings.

## Related projects (kept self-contained here)

- `cross-dataset-seizure-generalization` and `ecg-cross-dataset-generalization` (same leave-one-dataset-out philosophy in other biosignals).
- `icu-model-transportability` (mixed-effects treatment of transportability).
