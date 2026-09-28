# Cross-dataset seizure generalization: a leakage audit and pediatric-to-adult transfer benchmark for scalp-EEG seizure detection

**One-sentence pitch.** Train scalp-EEG seizure detectors on CHB-MIT and evaluate them *zero-shot* on Siena, TUSZ and the Helsinki neonatal corpus under one event-based scorer, to quantify (i) how much reported performance is inflated by record-wise splitting, (ii) how much of the cross-cohort drop is explained by montage/sampling-rate shift versus age/etiology shift, and (iii) whether self-supervised EEG foundation models and unsupervised domain adaptation actually close the gap.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis to early-PhD scale. 6-9 months for the baseline + audit paper; 12 months if fine-tuning several foundation models.
- Compute: CPU is enough for the spectral/Riemannian baselines and the scoring/audit work (CHB-MIT is ~40 GB of EDF, ~980 h). One 16-24 GB GPU is needed to fine-tune BIOT/LaBraM/EEGPT-class models on TUSZ (~1,600 h). Storage: ~150 GB for all four corpora.

## Background

Automated seizure detection from scalp EEG is one of the oldest ML-in-neurology problems, yet a 2024 systematic review of clinical translation ("Clinical translation of machine learning algorithms for seizure detection in scalp EEG: systematic review", arXiv 2404.15332) found that most published detectors are evaluated with protocols that do not reflect deployment: random window splits, sample-level metrics, and a single dataset. The 2025 SzCORE seizure-detection challenge ("SzCORE as a benchmark: report from the seizure detection challenge 2025", arXiv 2505.18191) made the gap concrete: the best entrant on a private 4,360-hour continuous-EEG test set reached an event-level F1 of only 0.43 (sensitivity 0.37, precision 0.45 at ~1 FP/day), far below the >95% figures routinely reported on CHB-MIT.

Two things make CHB-MIT-trained models an ideal test bed for a *generalization audit*: the corpus is pediatric (ages 1.5-22 y), and it ships as fixed bipolar recordings at 256 Hz, whereas Siena is adult (20-71 y), referential, 512 Hz; TUSZ is mixed-age, referential ("-REF"/"-LE"), 250-512 Hz; and Helsinki is neonatal, 256 Hz, with three independent annotators. This lets age shift and acquisition shift be separated by design.

## The research gap

**What has been done.**

- Datasets: CHB-MIT (Shoeb, 2009, MIT PhD thesis; PhysioNet), Siena Scalp EEG (Detti et al., 2020, *Processes*; PhysioNet), TUH EEG Seizure Corpus (Shah et al., 2018, *Front. Neuroinform.*), Helsinki neonatal EEG with three-expert annotations (Stevenson et al., 2019, *Sci. Data*).
- Scoring: NEDC's OVLP and TAES scorers (Shah, Golmohammadi, Obeid & Picone, 2021, in *Biomedical Signal Processing*, Springer) and the SzCORE event scorer with 30 s pre-/60 s post-onset tolerance and FP/24 h (Dan et al., 2024, *Epilepsia*, "SzCORE: Seizure Community Open-source Research Evaluation framework").
- Leakage: Ali, Angelova & Karmakar (2024, *R. Soc. Open Sci.*) showed that random window splits on CHB-MIT leak subject identity and that subject-wise sensitivity falls to ~73-75%; "Generalization or mirage? Data leakage and reported performance..." (*BioData Mining*, 2025) reviews the same problem across EEG/ECG ML.
- Cross-dataset transfer: SeizureTransformer (Wu & Zhao, 2025, arXiv 2504.00336) trained on TUSZ (+Siena) and won the SzCORE 2025 challenge; EEG-FM-Bench (arXiv 2508.17742; ICML 2026) benchmarks seven EEG foundation models (BENDR, BIOT, LaBraM, EEGPT, CBraMod, CSBrain, REVE, plus NeuroLM/SleepFM) on seven seizure cohorts including TUSZ, Siena, CHB-MIT and Helsinki, reporting that FMs beat supervised baselines except on the cohort the supervised model was trained on, and that event-level Sens@FA-AUC ranges from ~0.50 (Helsinki) to ~0.74 (SeizeIT1). RobustSeiz (arXiv 2609.04007, 2026) standardizes CHB-MIT/TUSZ/Siena/SeizeIT1 into BIDS-EEG at 256 Hz and stress-tests detectors with synthetic noise/adversarial perturbations. "What EEG foundation models encode: dataset identity..." (arXiv 2607.24519, 2026) shows FM embeddings carry strong dataset-identity information.
- Foundation models: BENDR (Kostas, Aroca-Ouellette & Rudzicz, 2021, *Front. Hum. Neurosci.*), BIOT (Yang, Westover & Sun, 2023, *NeurIPS*), LaBraM (Jiang, Zhao & Lu, 2024, *ICLR*), EEGPT (Wang et al., 2024, *NeurIPS*), CBraMod (Wang et al., 2025, *ICLR*).

**What is specifically missing.**

1. **A leakage-inflation audit that holds the scorer constant.** Existing work compares record-wise vs subject-wise within one paper and one model. Nobody has quantified, for the *same* set of model families (spectral+logistic, Riemannian tangent-space, compact CNN, fine-tuned FM), the inflation window-wise -> record-wise -> patient-wise -> cross-cohort, under the same event-based scorer. The hypothesis that higher-capacity models are inflated *more* by identity leakage (because they can memorize patient signatures) is untested.
2. **Scorer sensitivity of leaderboards.** OVLP, TAES and SzCORE event scoring make different assumptions (any-overlap vs time-weighted vs tolerance windows + merging). Whether model *rankings* change with the scorer, and by how much, has not been reported across cohorts. EEG-FM-Bench uses one event metric; RobustSeiz reports sample and event metrics but only for one cohort at a time under perturbation.
3. **Attributing the cross-cohort drop.** EEG-FM-Bench and the SzCORE challenge document that performance drops across cohorts but do not decompose *why*. Montage/sampling-rate shift can be removed by harmonization; age shift cannot. The pediatric (CHB-MIT) -> adult (Siena/TUSZ) -> neonatal (Helsinki) ladder gives a natural gradient in age shift with acquisition shift held roughly fixed after harmonization. Distribution-shift diagnostics (MMD in feature space, PSD divergence, Riemannian distance between covariance means) have not been tested as *predictors* of the transfer loss.
4. **Does unsupervised domain adaptation help under event scoring?** Riemannian re-centering (Zanini et al., 2018, *IEEE TBME*), Euclidean alignment (He & Wu, 2020, *IEEE TBME*), CORAL, and AdaBN are cheap and label-free; they are standard in BCI but rarely evaluated for seizure detection with FP/24 h as the endpoint, where a re-centering step that shifts the background distribution can flood the alarm rate.

## Research questions / hypotheses

1. **H1 (leakage inflation).** For each model family, event-based F1 decreases monotonically window-wise > record-wise > patient-wise > cross-cohort on CHB-MIT; the record-wise minus patient-wise gap is > 0.15 F1 for the CNN/FM families and < 0.10 for the spectral+logistic baseline.
2. **H2 (scorer sensitivity).** Kendall's tau between model rankings under OVLP vs TAES vs SzCORE is < 0.8 on at least one target cohort; TAES penalizes models producing long over-extended events, SzCORE rewards them.
3. **H3 (acquisition vs age).** After montage harmonization to a common 18-pair bipolar set and resampling to 256 Hz, the CHB-MIT -> Siena drop shrinks by at least half, whereas the CHB-MIT -> Helsinki drop does not (age/etiology-dominated).
4. **H4 (shift diagnostics predict loss).** Across (source, target, model) triplets, log MMD between source and target background-window features correlates with the drop in event F1 (Spearman rho > 0.6).
5. **H5 (foundation models).** Fine-tuned FMs reduce the cross-cohort drop relative to the CNN baseline but do not eliminate it; linear probes on frozen FM embeddings are inflated by record-wise leakage at least as much as supervised CNNs (consistent with dataset-identity encoding).
6. **H6 (domain adaptation).** Riemannian re-centering on target background raises event sensitivity at matched FP/24 h on Siena/TUSZ, but *increases* FP/24 h on Helsinki unless the re-centering reference is restricted to artifact-free background.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| CHB-MIT Scalp EEG | 23 pediatric subjects (24 cases), bipolar 256 Hz, ~198 seizures; training source | ~980 h, ~42 GB | Open (PhysioNet, no login) | https://physionet.org/content/chbmit/1.0.0/ |
| Siena Scalp EEG | 14 adults, referential 10-20, 512 Hz, 47 seizures; adult target | ~128 h, ~20 GB | Open (PhysioNet) | https://physionet.org/content/siena-scalp-eeg/1.0.0/ |
| TUH EEG Seizure Corpus (TUSZ v2.0.3) | train/dev/eval, mixed ages, referential (AR/LE), 250-512 Hz, thousands of annotated seizures; mixed-age target and FM fine-tuning corpus | ~1,600 h, ~70 GB | Free registration (TUH EEG data-use form; rsync with password) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| Helsinki neonatal EEG | 79 term neonates, 19-ch referential 256 Hz, 3 expert annotators; neonatal target | ~74 h, ~4 GB | Open (Zenodo; Stevenson et al., 2019) | https://zenodo.org/ (search "A dataset of neonatal EEG recordings with seizure annotations") |
| Optional: SeizeIT1 / Epilepsiae | extra adult targets used in EEG-FM-Bench | varies | DUA / paid | see EEG-FM-Bench |

Ages: CHB-MIT 1.5-22 y; Siena 20-71 y; TUSZ neonatal to >90 y (age in patient metadata); Helsinki 0-1 month.

## Methods

1. **Harmonization (`xseizure.io_edf`).** Every EDF is mapped to a common 18-pair longitudinal bipolar ("double banana") set: FP1-F7, F7-T3, T3-T5, T5-O1, FP2-F8, F8-T4, T4-T6, T6-O2, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FZ-CZ, CZ-PZ. Channel names are normalized (old/new 10-20 aliases T7=T3, P7=T5, etc.; "EEG ", "-REF", "-LE" prefixes/suffixes), bipolar pairs are computed from referential data or reused if already bipolar, missing pairs are masked. Signals are resampled to 256 Hz (polyphase), band-passed 0.5-70 Hz, notch 50/60 Hz according to cohort.
2. **Windows and labels.** 4-s windows, 2-s step (baseline) or 10-s windows for FM tokenizers; a window is positive if >= 50% overlaps a seizure annotation. For Helsinki, consensus (all three annotators) is the primary reference, majority vote secondary.
3. **Splits (`xseizure.splits`).** Four regimes on CHB-MIT: window-wise random, record-wise random (records of a subject can straddle folds), patient-wise 5-fold balanced by seizure count, leave-one-patient-out. Cross-cohort: train on all CHB-MIT, evaluate on the full target cohort with no target labels used for training or threshold selection (thresholds are fixed on CHB-MIT patient-wise validation folds).
4. **Baselines (`xseizure.features`).** (a) Spectral: log band power (delta/theta/alpha/beta/gamma) + line length + Hjorth mobility/complexity per channel -> standardized logistic regression (class-balanced). (b) Riemannian: shrinkage covariance per window -> affine-invariant tangent space at the log-Euclidean mean -> logistic regression (pyriemann optional; numpy implementation included). (c) Compact CNN (EEGNet-style, torch; not included here). (d) Foundation models: BIOT, LaBraM, EEGPT, CBraMod checkpoints with a linear probe on frozen embeddings and with full fine-tuning on CHB-MIT only.
5. **Post-processing.** Probability smoothing (moving average over 5 windows), fixed threshold, minimum event duration 5 s, merge gaps < 10 s; identical for every model.
6. **Scoring (`xseizure.scoring`).** Sample-based metrics; OVLP; TAES (time-aligned, fractional credit); SzCORE event scoring (30 s pre / 60 s post tolerance, merge < 90 s, split > 5 min); FP/24 h; sensitivity at matched FP/24 h (1, 5, 10) via threshold sweep.
7. **Domain-shift diagnostics (`xseizure.domain_shift`).** MMD (RBF, median heuristic) with permutation test between source and target background features; per-channel PSD divergence (log-spectral distance and Jensen-Shannon on normalized PSDs); affine-invariant Riemannian distance between cohort covariance means.
8. **Domain adaptation.** Riemannian re-centering (whiten with target background mean), Euclidean alignment, CORAL on spectral features, AdaBN for CNN/FM. All use only unlabeled target background.

## Evaluation & statistics

- Primary endpoint: SzCORE event F1 and sensitivity at 1 FP/24 h on each target; secondary: OVLP and TAES F1, sample AUROC, FP/24 h.
- Uncertainty: subject-level bootstrap (resample subjects with replacement, 2,000 draws) for all cohort-level metrics; report 95% percentile CIs. Model comparisons via paired subject-level bootstrap of the difference.
- Leakage inflation: for each model family, delta F1 (record-wise minus patient-wise) with bootstrap CI over folds/subjects; test model-family x split-regime interaction with a linear mixed model (subject random effect).
- Scorer sensitivity: Kendall's tau of model rankings between scorers, with bootstrap CI; Bland-Altman of OVLP vs TAES F1.
- Shift-loss relationship: Spearman correlation between log-MMD and delta F1 over (source, target, model) triplets; partial correlation controlling for age gap.
- Multiple comparisons: Holm-Bonferroni within each hypothesis family (H1-H6 are pre-registered as families).
- Leakage prevention: subject IDs parsed from file paths (CHB-MIT chbXX; Siena PNxx; TUSZ 8-digit patient id; Helsinki eegN); `assert_no_subject_leakage` runs before every fit; threshold and post-processing hyperparameters are frozen on CHB-MIT validation folds and never touched on targets; no target labels are used anywhere except scoring.
- Nulls: label-shuffled models to confirm event scorers return F1 near the chance level implied by the FP rate; MMD permutation null.

## Publishable angle

- **Headline result.** "Reported CHB-MIT performance is inflated by X F1 points by record-wise splitting, more so for high-capacity models; after montage harmonization, half of the pediatric-to-adult drop disappears but the pediatric-to-neonatal drop does not, and MMD between background windows predicts the loss." Delivered as a standardized cross-dataset leaderboard (code + fixed splits + fixed scorers) that any new detector can be dropped into.
- **Venues.** *Epilepsia* or *Clinical Neurophysiology* (clinical audience, methods audit); *Journal of Neural Engineering* or *IEEE TBME* (methods); *NeurIPS Datasets & Benchmarks* track or the *SzCORE / AI-in-Epilepsy* challenge venue for the leaderboard; *Scientific Data* for the harmonized annotation/manifest release.
- **Follow-ups.** Add SeizeIT1/2 and the Epilepsiae corpus; extend the audit to seizure *prediction* (where leakage is worse); annotator-disagreement-aware scoring using the three Helsinki annotators; age-conditioned FMs (age as a token).

## Risks, confounds & mitigations

- **Annotation conventions differ** (CHB-MIT onset/offset in summary files; TUSZ CSV_bi term-based; Helsinki per-second per-annotator). Mitigation: one `events.csv` schema per record; consensus vs majority reported for Helsinki; a "boundary tolerance" sensitivity analysis via SzCORE tolerances.
- **Class imbalance and long records** produce FP/24 h dominated by a few artifact-laden records. Mitigation: report per-subject FP/24 h distributions, not only means; artifact-rate covariate.
- **Montage harmonization is lossy** (TUSZ has extra T1/T2/A1/A2 channels; CHB-MIT records vary in channel sets within a case). Mitigation: mask missing pairs; report performance with the intersection montage only; document per-record channel availability.
- **Foundation-model pretraining overlap**: several FMs were pretrained on TUH corpora (TUEG), so TUSZ is not a clean zero-shot target for them. Mitigation: mark contaminated (model, target) cells in the leaderboard; use Siena/Helsinki as the clean targets.
- **Threshold transfer**: a threshold chosen on CHB-MIT may be miscalibrated on targets. Mitigation: report sensitivity at matched FP/24 h from a full threshold sweep, plus the fixed-threshold operating point.
- **Neonatal EEG is qualitatively different** (discontinuity, tracé alternant). Mitigation: treat Helsinki as the extreme of the age ladder, not as a fair target; add the neonatal-specific CNN of "Scaling convolutional neural networks achieves expert-level seizure detection in neonatal EEG" (*npj Digit. Med.*, 2025) as a reference point.

## Milestones

- [ ] Download CHB-MIT, Siena, Helsinki; register for TUSZ (`scripts/download_data.py`).
- [ ] Build `events.csv` + `records.csv` manifests for all four corpora (subject id, age, fs, channel set, seizure events).
- [ ] Validate harmonization: per-record montage coverage table; PSD sanity plots per cohort.
- [ ] Implement the four split regimes; freeze fold assignments (`splits/*.json`).
- [ ] Run spectral and Riemannian baselines under all regimes; produce the leakage-inflation table.
- [ ] Implement CNN baseline; add FM linear probes and fine-tunes.
- [ ] Cross-cohort evaluation with all three scorers; ranking-stability analysis.
- [ ] Shift diagnostics vs loss regression; harmonized vs unharmonized ablation.
- [ ] Domain-adaptation experiments at matched FP/24 h.
- [ ] Release leaderboard repo with fixed manifests, splits and scorer tests; write paper.

## Ethics / data-use notes

- CHB-MIT, Siena (PhysioNet open) and Helsinki (Zenodo, CC-BY) are de-identified and openly licensed; cite the dataset papers and PhysioNet.
- TUSZ requires a signed TUH EEG data-use agreement; the data may not be redistributed. Keep credentials in environment variables (`TUH_USERNAME`, `TUH_PASSWORD`), never in code or git.
- Do not upload raw EEG from any of these corpora to third-party LLM/ML APIs; PhysioNet's responsible-use terms and the TUH agreement restrict redistribution and third-party processing.
- Never commit data; `data/` and `*.edf` are git-ignored. Only manifests with subject ids and event times (already public in the source datasets) may be committed.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py --dataset chbmit --sample      # open, ~0.5 GB
python scripts/download_data.py --dataset siena --sample
python scripts/download_data.py --build-manifests              # data/manifests/{records,events}.csv
pytest tests -q                                                # synthetic-data tests, no EEG needed
```

Minimal end-to-end sketch with the starter modules:

```python
from xseizure import io_edf, features, splits, scoring, domain_shift
rec = io_edf.load_edf_harmonized("data/chbmit/chb01/chb01_03.edf")        # 18-pair bipolar, 256 Hz
W = features.windows_from_array(rec.data, rec.fs, win_s=4, step_s=2)      # (n_win, 18, 1024)
y = features.window_labels(len(W), 4, 2, events=[(2996, 3036)])           # from chb01-summary.txt
X = features.spectral_features(W, rec.fs)
folds = splits.patient_wise_split(subject_ids, n_folds=5, weights=y)      # over the whole corpus
p = features.LogisticBaseline().fit(X[tr], y[tr]).predict_proba(X[te])
hyp = scoring.probabilities_to_events(features.smooth_probabilities(p), step_s=2, win_s=4)
print(scoring.score_all(ref_events, hyp, duration_s=rec.duration_s))      # ovlp / taes / szcore
print(domain_shift.shift_report(X_chbmit_bg, X_siena_bg, W_chbmit_bg, W_siena_bg, 256))
```

## Repository layout

```
README.md                 this document
requirements.txt
data/README.md            acquisition instructions (PhysioNet wget, TUH rsync, Zenodo API)
scripts/download_data.py  downloader + manifest builder
src/xseizure/
  io_edf.py               EDF loading, channel-name normalization, 18-pair bipolar harmonization, resampling
  splits.py               window-/record-/patient-wise and LOPO splitters, leakage assertions, inflation CI
  features.py             spectral + Hjorth features, covariance/tangent-space (Riemannian) features, re-centering, logistic baseline
  scoring.py              OVLP, TAES, SzCORE-style event scorers, sample metrics, post-processing, sensitivity@FP/24h
  domain_shift.py         MMD + permutation test, PSD divergence, Riemannian covariance shift
tests/test_xseizure.py    synthetic tests pinning scorer semantics and leakage behaviour
```

## Planned tables and figures

- Table 1: cohort descriptives (subjects, age range, hours, seizures, fs, montage, channel coverage of the 18 canonical pairs).
- Table 2: leakage-inflation matrix, rows = model family (spectral-LR, Riemannian-LR, CNN, FM-probe, FM-finetune), columns = split regime (window, record, patient, LOPO), cells = SzCORE F1 with subject-bootstrap CI.
- Table 3: cross-cohort leaderboard, rows = models, columns = target (Siena, TUSZ-dev, TUSZ-eval, Helsinki-consensus, Helsinki-majority), three scorers per cell; contaminated (pretraining-overlap) cells flagged.
- Table 4: shift diagnostics per (source, target): MMD, PSD JS divergence, covariance distance, age gap; Spearman with delta F1.
- Figure 1: sensitivity vs FP/24 h curves per target, harmonized vs unharmonized montage.
- Figure 2: ranking stability across scorers (bump chart) with Kendall tau.
- Figure 3: delta F1 vs log-MMD scatter with model family as marker.
- Supplementary: per-subject FP/24 h distributions; annotator-disagreement analysis on Helsinki.

## Key references

- Ali, Angelova & Karmakar (2024). Epileptic seizure detection using CHB-MIT dataset: the overlooked perspectives. *R. Soc. Open Sci.* 11:230601.
- Dan et al. (2024). SzCORE: Seizure Community Open-source Research Evaluation framework. *Epilepsia*.
- SzCORE as a benchmark: report from the seizure detection challenge 2025 (arXiv:2505.18191).
- Shah, Golmohammadi, Obeid & Picone (2021). Objective evaluation metrics for automatic classification of EEG events. In *Biomedical Signal Processing*, Springer.
- Shah et al. (2018). The Temple University Hospital Seizure Detection Corpus. *Front. Neuroinform.* 12:83.
- Stevenson et al. (2019). A dataset of neonatal EEG recordings with seizure annotations. *Sci. Data* 6:190039.
- Detti et al. (2020). EEG synchronization analysis for seizure prediction: a study on data of noninvasive recordings. *Processes* (Siena Scalp EEG Database).
- Shoeb (2009). Application of machine learning to epileptic seizure onset detection and treatment. MIT PhD thesis (CHB-MIT).
- Wu & Zhao (2025). SeizureTransformer (arXiv:2504.00336).
- EEG-FM-Bench (arXiv:2508.17742); RobustSeiz (arXiv:2609.04007); "What EEG foundation models encode: dataset identity" (arXiv:2607.24519).
- Kostas, Aroca-Ouellette & Rudzicz (2021). BENDR. *Front. Hum. Neurosci.*; Yang, Westover & Sun (2023). BIOT. *NeurIPS*; Jiang, Zhao & Lu (2024). LaBraM. *ICLR*; Wang et al. (2024). EEGPT. *NeurIPS*; Wang et al. (2025). CBraMod. *ICLR*.
- Zanini et al. (2018). Transfer learning: a Riemannian geometry framework with applications to BCI. *IEEE TBME*; He & Wu (2020). Transfer learning for BCIs: a Euclidean space data alignment approach. *IEEE TBME*.
