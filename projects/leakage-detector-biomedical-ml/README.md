# leakscan: a subject-aware static leakage detector and a code-level audit of biomedical ML papers

**One-sentence pitch.** Build an AST-based static analyser that recognises the leakage mechanisms specific to biomedical ML (subject/record identity shared across splits, sliding-window overlap, preprocessing/feature-selection/resampling fitted before the split, test-set-driven early stopping), validate it against mutation corpora and human annotation, and apply it to the public code of papers using CHB-MIT, PTB-XL, MIMIC, Bonn, DEAP and Sleep-EDF to estimate, for the first time from *code* rather than from paper text, how prevalent leakage is and how much it inflates the numbers the field cites.

## Status / difficulty / timeline / compute

- Status: design + working starter detector (this repo). The detector passes 22 unit tests on synthetic snippets and notebooks; the corpus builder talks to the GitHub and Semantic Scholar APIs.
- Difficulty: MSc thesis (software + meta-research) to early-PhD if the dynamic re-execution arm is included. The hard part is annotation and the statistics of the prevalence-performance relationship, not the parser.
- Timeline: 5-8 months (1 month detector hardening + mutation corpus, 2 months corpus building + annotation, 1 month audit + statistics, 1-2 months dynamic confirmation on CHB-MIT/PTB-XL, 1 month writing).
- Compute: a laptop for everything static. The dynamic confirmation (re-running ~20 repos under subject-wise splits) needs one GPU-day and the CHB-MIT (~42 GB) and PTB-XL (~3 GB) downloads.

## Background

Leakage, "the use of information in the training process that would not be available at prediction time", is the leading cause of irreproducible ML claims across 17 scientific fields (Kapoor & Narayanan, 2023, *Patterns*); the REFORMS reporting checklist (Kapoor et al., 2024, *Science Advances*) asks authors to state how splits respect dependence structure. In biosignal ML the dominant mechanism is *subject-level* leakage: windows, epochs or records from one person land in both train and test, and the model learns identity rather than pathology. Saeb et al. (2017, *GigaScience*) showed record-wise cross-validation over-estimates accuracy relative to subject-wise; Brookshire et al. (2024, *Frontiers in Neuroscience*) found the majority of translational DNN-EEG studies use segment-based hold-out and therefore over-estimate performance on new subjects; Rosenblatt et al. (2024, *Nature Communications*) quantified inflation from feature selection, site correction and family structure in connectome-based prediction; Ali, Angelova & Karmakar (2024, *R. Soc. Open Sci.*) showed CHB-MIT subject-wise sensitivity drops to ~73-75% from the >95% reported with random window splits; a 2025 *Nature Mental Health* paper showed leaked estimates propagate into meta-analyses; a 2026 *Frontiers in Neuroinformatics* paper documents "window leakage" in clinical EEG pipelines.

All of these audits share one limitation: they were done by reading papers, where splitting is under-described. The code is the ground truth, and the code is increasingly public.

## The research gap

**What has been done (static analysis side, 2022-2026):**

- Yang, Brower-Sinning, Lewis & Kästner (2022, *ASE*) built a static data-flow analysis for notebooks detecting three leakage classes: *overlap* (train/test contamination), *multi-test* (test set reused for selection) and *preprocessing* leakage; they found leakage in a large share of public notebooks.
- LeakageDetector (PyCharm plugin) and LeakageDetector 2.0 (2025, arXiv:2509.15971; VS Code, adds LLM-driven quick fixes) operationalise the same three classes for developers.
- NBLyzer implements leakage detection by abstract interpretation (*Science of Computer Programming*, 2025); a Datalog + structured-LLM-prompt approach (ACM, 2025/2026) reports F1 gains for preprocessing and overlap leakage.
- bioLeak (Korkmaz, 2026, arXiv:2604.10965; CRAN) is an R package for leakage-aware resampling and post-hoc audits of *fitted* models - a modelling framework, not a scanner of third-party code.

**What is missing:**

1. **No detector knows about groups.** Overlap analyses ask whether the *same rows* appear in train and test; subject leakage arises from *different rows of the same person*. Detecting it requires a notion of grouping structure (subject/patient/record identifiers, dataset provenance, windowing) that none of the existing tools model. Likewise none flags SMOTE-before-split, feature selection on the full data or `validation_data=(X_test, y_test)` early stopping as a distinct class.
2. **No evaluation on real biomedical repositories.** Existing tools were evaluated on Kaggle/GitHub notebooks in general; precision and recall on EEG/ECG/EHR code, where the data-loading idioms (`mne`, `wfdb`, `pyedflib`, per-subject EDF loops) are different, are unknown.
3. **Prevalence has never been estimated from code.** The paper-reading audits give the *reporting* rate, not the *leakage* rate; the two differ whenever the methods text is vague ("we used 10-fold CV").
4. **The inflation is not linked to the literature's numbers.** Whether flagged repositories report systematically higher accuracy/AUROC than group-aware ones (adjusting for model family, year, dataset) has not been tested, nor whether re-running flagged code under subject-wise splits reproduces the ~20-point drops seen in one-off studies.

## Research questions / hypotheses

1. **RQ1 (detector validity).** On a mutation corpus and on 100 human-annotated repositories, `leakscan` achieves precision >= 0.80 and recall >= 0.70 for `split-without-groups` and `preprocess-before-split`, and detects group leakage that overlap-only detectors (Yang et al., 2022) cannot by construction. *H1:* recall is lower for notebooks than for scripts (cross-cell state, magics) by >= 0.1.
2. **RQ2 (prevalence).** *H2:* >= 40% of public repositories linked to CHB-MIT papers use a random window-level split without group awareness; the proportion is lower for PTB-XL (official patient-stratified folds; < 20%) and for MIMIC (patient-level splits are idiomatic; < 30%).
3. **RQ3 (co-occurrence).** *H3:* preprocessing-before-split co-occurs with `split-without-groups` (odds ratio > 2); notebooks have higher leakage prevalence than script repositories (OR > 2).
4. **RQ4 (inflation).** *H4:* among CHB-MIT papers, repositories flagged for group leakage report a median headline metric >= 5 points higher than group-aware ones, after adjustment for year, model family (CNN/transformer vs feature-based) and evaluation unit (window vs event).
5. **RQ5 (dynamic confirmation).** *H5:* re-executing 20 flagged repositories under subject-wise splits (same code, only the splitter changed) reduces the reported metric by >= 10 points in >= 75% of cases; the drop is larger for higher-capacity models.
6. **RQ6 (temporal trend).** *H6:* prevalence declines after 2022 (REFORMS, journal policies) but remains > 25% in 2024-2025 CHB-MIT repositories.

## Datasets

| Dataset / source | What is used | Size | Access | URL |
|---|---|---|---|---|
| GitHub REST API (search/repositories, search/code) | Repositories containing dataset markers (`ptbxl_database.csv`, `chb01_03.edf`, `icustays`, ...) | ~300-1,500 repos per dataset marker | Open (token for code search; free) | https://docs.github.com/en/rest/search |
| Semantic Scholar Graph API | Papers citing the dataset descriptors (PTB-XL, MIMIC-IV, eICU, Helsinki) with abstracts and OA-PDF links | thousands of citing papers | Open (optional free API key) | https://api.semanticscholar.org/ |
| Public repositories (cloned, `--depth 1`) | Python scripts and notebooks; static analysis only | 50-500 MB per 100 repos | Open, per-repo licence | github.com |
| Manual annotation set | 100 repos stratified by dataset, two annotators | small | Produced in this project | `data/corpus/annotations/` |
| CHB-MIT Scalp EEG | Dynamic confirmation arm (re-run flagged repos with subject-wise splits) | ~42 GB, 23 subjects | Open (PhysioNet, no login) | https://physionet.org/content/chbmit/1.0.0/ |
| PTB-XL v1.0.3 | Dynamic confirmation arm; official folds as reference | 21,799 ECGs, ~3 GB | Open (PhysioNet) | https://physionet.org/content/ptb-xl/1.0.3/ |
| MIMIC-IV / MIMIC-III | Only for the *static* audit (code that references them); no data download required | - | Credentialed (PhysioNet) if re-execution is attempted | https://physionet.org/content/mimiciv/ |

## Methods

1. **Detector** (`leakscan.detector`). Parse each `.py`/`.ipynb` (notebooks concatenated with cell bookkeeping, `leakscan.notebooks`), collect *events* in source order - splits (`train_test_split`, `KFold(...).split`, `cross_val_*`, `GridSearchCV.fit`, `torch random_split`), transformer fits (`StandardScaler`, `PCA`, imputers, encoders, `fit_transform` on unknown objects), feature selectors (`SelectKBest`, `RFE`, ...), resamplers (`SMOTE` family), estimator fits, evaluations and metric calls - each with the *root variable names* they touch. Rules compare order and name overlap. Subject-level structure is inferred from a vocabulary of identifiers and string constants (`subject`, `patient`, `chb01`, `ptbxl`, `mimic`, `wfdb`, `mne`, `.edf`, ...) and grades severity (`high` >= 2 tokens, `medium` = 1, `low` = 0); `--subject-level` forces the assumption. Window vocabulary (`window`, `epoch`, `segment`, `stride`, `overlap`) triggers the `window-level-random-split` rule.
2. **Rules** (all implemented and tested): `split-without-groups`, `window-level-random-split`, `preprocess-before-split`, `feature-selection-before-split`, `resample-before-split`, `test-set-in-training` (positional test-named arrays in `fit`, `validation_data=`/`eval_set=` pointing at test sets, scalers fitted on test sets), `no-holdout-evaluation`, plus positive evidence `group-aware-split-present`, `pipeline-present`, and `mixed-splitting`.
3. **Corpus** (`leakscan.corpus`, `scripts/download_data.py`): GitHub repository/code search per dataset marker with pagination and rate-limit handling; Semantic Scholar citations of dataset descriptors mined for GitHub links (abstract now, OA-PDF text next); shallow clones. Inclusion: Python, references one of the target datasets, linked to a paper or preprint (for RQ4-6) or not (for RQ2-3).
4. **Ground truth**: (a) *mutation corpus* - 20 clean pipelines with templated injections of each leakage type; (b) *annotated sample* - 100 repositories, two blinded annotators, adjudication, Cohen's kappa.
5. **Baselines**: the Yang et al. (2022) notebook analysis (open source) run on the same notebooks for the three classes it covers; a keyword grep (`train_test_split` present) as the naive baseline.
6. **Paper metadata** (manual, `paper_repo_links.csv`): headline metric, dataset, model family, evaluation unit, split claimed in text, year, venue.
7. **Dynamic confirmation**: for 20 CHB-MIT and 10 PTB-XL repositories flagged `high`, minimally patch the split to `GroupKFold`/leave-one-subject-out (documented diff), re-run, record the metric drop; use the sibling projects' harmonised loaders where the original loaders fail.

Tools: Python `ast`, `pandas`, `requests`; `statsmodels` (optional) for the regressions; `pymupdf` (optional) for PDF text.

## Evaluation & statistics

- Detector: precision, recall, F1 per rule on the mutation corpus (exact labels) and on the annotated sample (adjudicated labels), with Wilson 95% CIs; McNemar test vs the Yang et al. tool on the classes both cover; error analysis by idiom (loops over subject files, `mne.Epochs`, `wfdb.rdrecord`, custom splitters).
- Prevalence: per dataset and per rule with Wilson CIs (`leakscan.report.prevalence_table`); logistic regression of `any_leakage` on dataset, year, notebook-vs-script, stars, and paper-linked-vs-not.
- Inflation (RQ4): linear model of reported metric on leakage flag + dataset + year + model family + evaluation unit, with robust SEs; sensitivity: restrict to papers whose text states the split, and to window-level accuracy only.
- Dynamic confirmation: paired (before/after) differences with bootstrap CIs; relation to model capacity (parameter count).
- Multiple comparisons: hypotheses H1-H6 pre-registered; Holm correction within RQ4/RQ5.
- Nulls: shuffling the leakage flag across repositories (permutation p for the inflation association); the mutation corpus's clean variants as the false-positive control.
- Leakage prevention in *our* analysis: annotators never see detector output; the 100-repo sample is drawn before the detector is finalised; detector thresholds are frozen on the mutation corpus only.

## Publishable angle

- **Headline.** "In N public repositories behind CHB-MIT/PTB-XL/MIMIC papers, X% contain subject-level leakage detectable from code; leaked repositories report metrics Y points higher, and re-running them with subject-wise splits erases most of the difference. A 600-line static analyser finds these patterns with precision P." The tool itself (pip-installable, CI-friendly, `--fail-on-findings`) is a deliverable journals and challenges can adopt.
- **Venues.** *Patterns* or *Nature Machine Intelligence* (meta-research + tool); *Journal of Biomedical Informatics* or *JAMIA* (informatics audience); *ICSE/ASE/FSE* (software-engineering track, detector evaluation); *NeurIPS Datasets & Benchmarks* (audit corpus + mutation benchmark).
- **Follow-ups.** Temporal leakage rules for EHR (label windows overlapping features; `shuffle=True` on time-indexed data); an R/MATLAB front end; a GitHub Action; extension to imaging (`nibabel` per-subject loops); a leakage-corrected meta-analysis of CHB-MIT seizure-detection performance.

## Risks, confounds & mitigations

- **False positives from vocabulary heuristics** (a "subject" column that is already handled by an upstream split; random splits on truly i.i.d. tabular data). Mitigation: severity grading, `mixed-splitting` rule, human adjudication for every published count, precision reported per rule.
- **False negatives from dynamic idioms** (custom splitter functions, splits done by file-name lists, `pandas.sample`). Mitigation: error analysis on the annotated sample; extend the vocabulary iteratively but freeze before the prevalence run; report recall.
- **Selection bias of the corpus** (papers with code are not all papers; GitHub search caps at 1,000 results). Mitigation: two independent discovery routes (GitHub markers, citation mining); report overlap; treat prevalence as "among papers with public code".
- **Reported metrics are heterogeneous** (window accuracy vs event F1 vs AUROC). Mitigation: stratify RQ4 by evaluation unit and metric; primary analysis on window-level accuracy for CHB-MIT where it dominates.
- **Re-execution failures** (missing dependencies, hard-coded paths). Mitigation: report the failure rate as a finding; patch only the splitter and paths; publish diffs.
- **Ethics of naming.** Aggregate reporting; repositories anonymised in tables; authors of re-executed repos contacted before publication.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests                                   # 24 tests: rules on snippets, notebooks, reports
PYTHONPATH=src python -m leakscan scan path/to/repo --md report.md         # scan scripts + notebooks
PYTHONPATH=src python -m leakscan scan analysis.ipynb --subject-level --fail-on-findings   # CI gate
python scripts/download_data.py --dataset chbmit --sample                  # GitHub repository discovery
PYTHONPATH=src python -m leakscan audit data/corpus/chbmit_repos.csv --out outputs/chbmit/findings.csv
```

Dogfooding note: run on the sibling `unified-biomedical-shift-benchmark`, the scanner found one real pattern (a scaler fitted on pooled data before `cross_val_predict` in a domain classifier; fixed there with a `Pipeline`) and one false positive (`fit` and `predict` in different methods of a wrapper class), which motivated function-scope tracking for the order-based rules. Both cases are now unit tests.

## Repository layout

```
src/leakscan/
  detector.py    AST event collector (splits, transformer/estimator fits, evaluations, metrics) + rules + severity grading
  notebooks.py   .ipynb -> source with cell/line map; magics blanked; per-cell fallback on SyntaxError
  report.py      long-format findings, per-repo rule flags, Wilson-CI prevalence tables, Markdown report
  corpus.py      GitHub search (pagination, rate limits), Semantic Scholar citations, GitHub-link mining, shallow clones
  __main__.py    CLI: scan / audit
scripts/download_data.py   corpus builder (repository search, code search, citation mining, cloning)
tests/test_detector.py     positive and negative snippets per rule; notebook ordering; report aggregation
data/README.md             corpus layout, annotation protocol, ground-truth construction
```

## Analysis plan

| RQ | Unit | Estimand / statistic | Data | Test / model |
|---|---|---|---|---|
| RQ1 validity | file, rule | precision, recall, F1 with Wilson CIs; error taxonomy | mutation corpus (exact labels); 100 annotated repos | McNemar vs Yang et al. (2022) on shared classes |
| RQ2 prevalence | repository | share flagged per rule and dataset, Wilson CIs | full corpus | logistic regression on dataset, year, notebook, stars, paper-linked |
| RQ3 co-occurrence | repository | OR of preprocessing leakage given group leakage; notebook vs script OR | full corpus | logistic regression; Fisher exact as check |
| RQ4 inflation | paper-linked repository | difference in reported headline metric, flagged vs not | `paper_repo_links.csv` | linear model with dataset, year, model family, evaluation unit; robust SEs; permutation of the flag |
| RQ5 dynamic confirmation | re-executed repository | metric before vs after switching to subject-wise splits | 20 CHB-MIT + 10 PTB-XL repos | paired bootstrap; relation to parameter count |
| RQ6 trend | repository x year | prevalence by year | full corpus | logistic regression with year spline; change-point at 2022 |

## Milestones

- [ ] Freeze rule set v1 and vocabulary; build the mutation corpus; precision/recall per rule.
- [ ] Corpus build: GitHub markers for 6 datasets + Semantic Scholar citations for 4 descriptors; de-duplicate; inclusion screening.
- [ ] Draw the 100-repo annotation sample; two-annotator labelling; kappa; adjudication.
- [ ] Run the audit on the full corpus; prevalence tables with CIs; comparison to the Yang et al. tool.
- [ ] Paper metadata extraction for paper-linked repos; RQ4 regression.
- [ ] Dynamic confirmation on 20 CHB-MIT + 10 PTB-XL repositories.
- [ ] Pre-registration (OSF) of H2-H6 before the full audit; manuscript; release tool v1.0 on PyPI with the mutation benchmark.

## Ethics / data-use notes

- Only public repositories are analysed, statically; no code is executed except in the explicit re-execution arm, in an isolated environment.
- MIMIC/eICU code is audited from its *source text* only; no credentialed data are downloaded for the static audit. If re-execution of MIMIC code is attempted it requires PhysioNet credentialing, and per PhysioNet's responsible-use policy neither data nor derived patient-level tables may be sent to third-party LLM APIs; this project uses no LLM components.
- CHB-MIT and PTB-XL are open; cite PhysioNet and the dataset papers.
- Repository identities are anonymised in published tables; `data/` is git-ignored; API tokens (`GITHUB_TOKEN`, `S2_API_KEY`) are read from the environment and never written to disk.
