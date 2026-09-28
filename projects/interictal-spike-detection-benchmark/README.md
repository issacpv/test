# Interictal spike detectors across scalp and intracranial EEG: a cross-modality benchmark with false-positive calibration on normal iEEG, IFCN-criteria audits and downstream seizure-onset-zone consequences

**One-sentence pitch.** Build one event-level benchmark for interictal epileptiform discharge (IED) detectors that spans scalp EEG (TUEV) and intracranial EEG (OpenNeuro / Omni-iEEG / the two-centre annotated sleep-iEEG dataset), adds two things no published benchmark reports (false-positive rate on *normal* intracranial channels from the MNI Open iEEG Atlas, and whether true and false positives satisfy the IFCN morphological criteria of Kural et al., 2020), stratifies by vigilance state, and quantifies how much detector choice changes spike-rate-based seizure-onset-zone (SOZ) rankings in patients with known SOZ and surgical outcome.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis (classical detectors + scoring + normal-atlas calibration, ~6 months) to PhD chapter (deep detectors, cross-modality transfer, downstream multiverse, ~12 months).
- Compute: classical detectors (envelope, matched filter, morphology classifier) run on CPU over all corpora in hours. A single 16-24 GB GPU for a SpikeNet-style 1-D CNN and for foundation-model probes. Storage ~100 GB (TUEV + OpenNeuro iEEG sets; Omni-iEEG is a re-packaging of OpenNeuro sources).
- Related project: `cross-dataset-seizure-generalization` (seizure detection, event scorers). IEDs are sub-second events and need their own scorer; nothing is shared.

## Background

IEDs (spikes, sharp waves, spike-and-wave complexes) are the interictal signature of epilepsy; their presence supports the diagnosis and their spatial distribution helps localize the epileptogenic zone. Human agreement is imperfect: Jing et al., 2020 (*JAMA Neurol.*, "Interrater reliability of experts in identifying interictal epileptiform discharges in EEGs") quantified substantial disagreement, and the IFCN clinical-validation study of Kural et al., 2020 (*Neurology*) showed that requiring >= 4 of 6 morphological criteria (sharp/spiky di- or triphasic morphology, duration different from background, asymmetry, after-going slow wave, background disruption, scalp field consistent with a cerebral source) yields high specificity. Automated detection has a long history (scalp: SpikeNet, Jing et al., 2020, *JAMA Neurol.*; intracranial: Janca et al., 2015, *Brain Topogr.*, envelope modelling; Barkmeier et al., 2012, *Clin. Neurophysiol.*, multi-channel iEEG detector) and is now embedded in commercial software whose real-world performance in a tertiary centre was recently compared against a clinical reference standard (2025/2026, PMC12858492).

Open data now allow this to be benchmarked properly: TUEV (TUH EEG Events Corpus) gives per-channel scalp annotations of spike/sharp-wave (SPSW), generalized and lateralized periodic discharges (GPED, PLED), eye movement, artefact and background; a two-centre intracranial sleep dataset with expert sub-second IED annotations was released in BIDS (*Sci. Data*, 2024, "Annotated interictal discharges in intracranial EEG sleep data recorded at two medical centers"); Omni-iEEG (arXiv 2602.16072, 2026) re-packages OpenNeuro pre-surgical iEEG from 302 patients (178 h) with > 36,000 expert annotations of pathological events; the Epilepsy-iEEG-Multicenter-Dataset (OpenNeuro ds003029; Li et al., 2021, *Nat. Neurosci.*) carries clinically annotated SOZ channels and surgical outcomes; and the MNI Open iEEG Atlas (Frauscher et al., 2018, *Brain*; sleep extension von Ellenrieder et al., 2020, *Sci. Data*) provides intracranial recordings from channels in *normal* cortex - the only open source of spike-free intracranial background.

## The research gap

**What has been done.**

- Scalp detectors and external validation: SpikeNet (Jing et al., 2020, *JAMA Neurol.*); Nhu et al., 2023 (*Int. J. Neural Syst.*) benchmarked InceptionTime and MiniRocket across TUEV and private data and found that models trained on private data held up on TUEV but TUEV-trained models did not generalize to private data; Anguelova et al., 2025 (*Epilepsia*) optimize external validation of deep IED detectors; Lin et al., 2025 (*BMC Med.*) report a prospective multi-centre validation of a multimodal IED detector; ProtoEEG-kNN (arXiv 2510.20846, 2025) gives an interpretable case-based detector; a 2024 transformer detector (*Biomed. Signal Process. Control*) and the 2025 video-EEG analysis review (arXiv 2503.19949) summarize the field.
- Intracranial detectors: Janca et al., 2015; Barkmeier et al., 2012; spikes vs HFOs as biomarkers (Roehri et al., 2018, *Ann. Neurol.*, "High-frequency oscillations are not better biomarkers of epileptogenic tissue than spikes"). The 2024 sleep-iEEG dataset paper reports ~94% precision and sensitivity for a model trained on its own annotations. Omni-iEEG (2026) provides a large-scale iEEG benchmark of pathological events.
- Criteria: Kural et al., 2020 (*Neurology*) validated the six IFCN criteria for human readers; they have not been used to audit machine detections.

**What is specifically missing.**

1. **Cross-modality evaluation.** No benchmark trains on scalp and tests on intracranial (or the reverse) after amplitude/frequency renormalization; the modality gap for classical vs learned detectors is unknown.
2. **False-positive calibration on normal tissue.** Benchmarks report precision on epilepsy recordings, where every annotator miss becomes a "false positive". The MNI atlas provides channels in normal cortex where any detection is a genuine false positive; FP/min there is a cleaner specificity endpoint that no detector paper reports.
3. **Morphology-criteria audit.** Whether detector true positives satisfy more IFCN criteria than false positives, and whether "false positives" that satisfy >= 4 criteria are annotator misses, is untested; it turns disagreement into an adjudicable quantity.
4. **Vigilance-state stratification.** IED rates rise in NREM sleep; detectors are evaluated on pooled data. The annotated sleep-iEEG dataset and the sleep/awake labels of ds003876 make state-stratified sensitivity and FP rates possible.
5. **Downstream consequences.** Spike rate per channel is used to rank candidate SOZ channels; the sensitivity of that ranking to detector choice (a hidden analytic multiverse) has never been measured against clinically annotated SOZ and outcome (ds003029).

## Research questions / hypotheses

1. **H1 (modality gap).** Learned detectors (morphology classifier, CNN) lose >= 50% of their in-modality event F1 when transferred scalp -> iEEG or iEEG -> scalp after renormalization; the envelope detector loses < 25%.
2. **H2 (FP calibration is not captured by F1).** Across detectors and operating points, FP/min on MNI normal channels correlates weakly (Spearman rho < 0.3) with precision on epilepsy corpora.
3. **H3 (criteria).** True positives satisfy a median of >= 4 IFCN criteria and false positives <= 2; the subset of "false positives" satisfying >= 4 criteria is judged as IEDs by an independent reader in > 50% of a 200-event adjudication sample.
4. **H4 (vigilance).** Sensitivity is higher in NREM than in wake for every detector, but FP/min differs by detector family (threshold detectors inflate FPs in NREM due to slow-wave sharpness), so detector rankings change between states (Kendall tau < 0.8).
5. **H5 (downstream multiverse).** Across detectors, the top-3 spike-rate channels overlap < 60% on average; SOZ AUROC from spike rates varies by > 0.10 across detectors in >= 30% of ds003029 patients, and the variation is larger in patients with poor surgical outcome.
6. **H6 (annotator ceiling).** Where multiple annotators exist, the best detector-vs-consensus F1 is within 0.05 of the inter-annotator F1.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| TUH EEG Events Corpus (TUEV) | Per-channel scalp annotations: SPSW, GPED, PLED, EYEM, ARTF, BCKG; scalp training/test corpus | tens of hours, hundreds of records (check corpus page for current version) | Free registration (TUH data-use form; rsync) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| Annotated interictal discharges in intracranial EEG sleep data (two centres) | Expert sub-second IED annotations during overnight sleep iEEG, BIDS; iEEG training/test and vigilance analysis | see paper (*Sci. Data*, 2024) | Open (BIDS; accession in the paper's Data Availability) | https://www.nature.com/articles/s41597-024-04187-y |
| Omni-iEEG | 302 patients, 178 h interictal pre-surgical iEEG, > 36k expert annotations of pathological events, reorganized from OpenNeuro in BIDS | ~100 GB | Open (see arXiv 2602.16072 for the repository) | https://arxiv.org/abs/2602.16072 |
| Epilepsy-iEEG-Multicenter-Dataset (ds003029) | Interictal/ictal iEEG with clinically annotated SOZ channels and surgical outcome; downstream SOZ analysis | ~ tens of GB | Open (OpenNeuro) | https://openneuro.org/datasets/ds003029 |
| Epilepsy-iEEG-Interictal-Multicenter-Dataset (ds003876) | Interictal iEEG with sleep/awake state annotated; vigilance stratification | ~ tens of GB | Open (OpenNeuro) | https://openneuro.org/datasets/ds003876 |
| MNI Open iEEG Atlas (wake + sleep) | Segments from channels in normal cortex (106 patients, ~1,700 channels, wake; sleep extension); false-positive calibration | small (minutes per channel) | Open (web download after accepting terms) | https://mni-open-ieegatlas.research.mcgill.ca/ |
| Bonn EEG (Andrzejak et al., 2001) | Sets N/F (interictal iEEG, non-epileptogenic vs epileptogenic zone) as an out-of-distribution *rate* comparison only (no event annotations) | 500 x 23.6 s segments | Open | https://www.upf.edu/web/ntsa/downloads |

Modality-harmonized sampling rate: 512 Hz (scalp resampled up, iEEG down) so that 20-70 ms spikes retain >= 10 samples; iEEG referenced bipolar along electrode shafts; scalp in the common bipolar montage.

## Methods

1. **Ingestion.** TUEV `*.rec`/`*.tse`-style per-channel annotations to one `events.csv` (record, channel, onset_s, offset_s, label, annotator); BIDS `events.tsv` for the iEEG sets; MNI atlas segments to `records.csv` with `is_normal = 1`; ds003029 `channels.tsv` SOZ flags and outcome from participants metadata.
2. **Scoring (`iedbench.events`).** Events are (channel, centre time, duration, score). Matching is greedy one-to-one within a tolerance (default +/-100 ms, per channel; a channel-agnostic variant for scalp field events). Outputs: precision, recall, F1, FP/min, sensitivity at fixed FP/min, threshold sweeps.
3. **Detectors (`iedbench.detectors`).** (a) Envelope detector (Janca-style): 10-60 Hz band-pass, Hilbert envelope, log-envelope z-score in a 5-s running window, threshold k with 20-200 ms duration constraints; (b) matched filter with a spike + slow-wave template, z-scored normalized cross-correlation; (c) morphology classifier: candidate local extrema -> IFCN-inspired features -> logistic regression; (d) SpikeNet-style 1-D CNN on 1-s windows (torch, optional); (e) foundation-model linear probe (optional; note TUH pretraining overlap).
4. **Morphology audit (`iedbench.morphology`).** For every detection: duration at half prominence, sharpness (normalized second derivative), rise/fall asymmetry, amplitude relative to background RMS, after-going slow wave, background disruption, and spatial field (correlated deflections on neighbouring channels). Each maps to one IFCN criterion; the count of satisfied criteria is compared between TP and FP.
5. **Normal-tissue calibration.** Run every detector on MNI atlas channels; report FP/min per detector per operating point; report the operating point at which FP/min on normal tissue <= 0.1.
6. **Vigilance stratification (`iedbench.downstream.vigilance_stratified_rates`).** Sensitivity and FP/min per state (wake, N2, N3, REM where staged).
7. **Downstream SOZ multiverse (`iedbench.downstream`).** Per patient in ds003029: spike rate per channel for every detector; AUROC of rate vs clinical SOZ; top-k overlap and Spearman agreement between detectors; association with surgical outcome.

## Evaluation & statistics

- Primary endpoints: event F1 at the detector's default operating point and sensitivity at 1 FP/min (epilepsy corpora); FP/min on MNI normal channels at the same operating point.
- Uncertainty: patient-level bootstrap (2,000 draws); paired bootstrap for detector comparisons; Poisson exact CIs for per-channel rates.
- Validation: patient-grouped 5-fold CV within each corpus for learned detectors; cross-modality runs train on one modality and test on the other with no target tuning; thresholds fixed on training folds.
- Leakage prevention: patient ids parsed from file paths / BIDS `sub-*`; candidate windows never straddle folds; template and morphology thresholds fitted on training patients only.
- Criteria audit statistics: Mann-Whitney on criteria counts TP vs FP; adjudication sample scored with Cohen's kappa against the second reader.
- Ranking stability: Kendall tau between detector rankings across states / corpora with bootstrap CI; SOZ AUROC variability via patient-level distributions.
- Nulls: annotation-time shuffling within record (destroys alignment, preserves rate) for the F1 chance level; channel-label permutation for SOZ AUROC.
- Multiple comparisons: Holm-Bonferroni within each hypothesis family; the detector x corpus table is reported in full without cell-wise tests.

## Publishable angle

- **Headline result.** "Detectors that look equivalent on F1 differ X-fold in false positives on normal cortex; true positives satisfy 4-5 IFCN criteria and false positives 1-2, and a third of 'false positives' are annotator misses; NREM inflates threshold-detector FPs; and detector choice alone moves spike-rate SOZ rankings enough to change the top channel in Y% of patients." Delivered as a public scorer, criteria-audit code and a normal-tissue calibration set.
- **Venues.** *Clinical Neurophysiology* or *Epilepsia* (clinical audit); *Journal of Neural Engineering* / *IEEE TBME* (methods); *NeurIPS Datasets & Benchmarks* (benchmark release); *Brain Communications* for the SOZ-multiverse analysis.
- **Follow-ups.** HFO detectors under the same framework (Roehri et al., 2018 debate); criteria-constrained training (penalize detections violating IFCN criteria); annotator-disagreement-aware labels; extension to Omni-iEEG's full event taxonomy.

## Risks, confounds & mitigations

- **Annotation completeness.** IED annotations are rarely exhaustive; unannotated true IEDs count as FPs. Mitigation: normal-tissue FP endpoint; criteria audit; adjudication sample.
- **Modality differences in amplitude and spectrum.** Mitigation: per-record robust scaling (median/MAD), identical band-pass, feature-level normalization; report both raw and normalized transfer.
- **Normal atlas channels are not perfectly normal** (patients with epilepsy, channels selected as outside the epileptogenic region). Mitigation: use the atlas authors' inclusion labels; report per-region FP rates; treat as a lower bound.
- **TUEV label taxonomy** (SPSW vs GPED/PLED) differs from iEEG "IED". Mitigation: primary target SPSW; secondary analysis with periodic discharges included.
- **Vigilance staging availability** differs between corpora. Mitigation: use only corpora with staged data for H4; a simple delta-power proxy elsewhere, clearly labelled.
- **Foundation-model pretraining overlap with TUH.** Mitigation: mark contaminated cells; use iEEG sets as clean targets.

## Milestones

- [ ] Register for TUEV; download OpenNeuro sets (ds003029, ds003876, the sleep-IED accession, Omni-iEEG); download the MNI atlas; Bonn.
- [ ] Build `records.csv`/`events.csv` with channel-level annotations; harmonize to 512 Hz bipolar.
- [ ] Validate scorer on synthetic and hand-checked segments; freeze tolerance and matching rules.
- [ ] Run envelope, matched-filter and morphology detectors within each corpus; threshold sweeps.
- [ ] Normal-tissue FP calibration table (H2).
- [ ] Criteria audit on TP/FP; 200-event adjudication with a second reader (H3).
- [ ] Cross-modality transfer (H1); CNN detector.
- [ ] Vigilance-stratified evaluation (H4).
- [ ] SOZ multiverse on ds003029 (H5); annotator ceiling (H6).
- [ ] Release benchmark and write paper.

## Ethics / data-use notes

- OpenNeuro datasets, the MNI atlas and Bonn are de-identified and openly licensed; cite each dataset paper and respect the licence terms stated on each repository page.
- TUEV requires a signed TUH EEG data-use agreement; credentials in `TUH_USERNAME` / `TUH_PASSWORD`, never in code or git; no redistribution.
- Adjudication by a second reader must use de-identified segments only; store reader labels without patient identifiers.
- Do not send raw EEG/iEEG to third-party LLM/ML APIs; the TUH agreement restricts third-party processing.
- Never commit data; `data/` and `*.edf` are git-ignored. Only event manifests with public ids and times may be committed.
