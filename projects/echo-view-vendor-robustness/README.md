# Echo-View-Shift: how much does echocardiographic view classification degrade across scanner vendors and sites?

**One-sentence pitch.** Build the first controlled cross-dataset benchmark of echocardiographic *view* classification under vendor/site shift - training on one source (e.g., TMED-2 or EchoNet-Dynamic frames) and testing on others (CAMUS, MIMIC-IV-ECHO, and the public TTE47 benchmark) - decompose the drop into acquisition (vendor/format/frame-rate) vs anatomy, and test which normalization and domain-generalization strategies recover view accuracy without target labels, with a view-confusion analysis that flags the clinically dangerous swaps (e.g., A4C<->A2C).

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data or DICOMs are shipped.
- Difficulty: MSc-to-early-PhD (medical image analysis + domain generalization + biostatistics).
- Timeline: 6-9 months (2 months harmonisation + view-label crosswalk + baselines, 2 months shift decomposition + DG methods, 2 months MIMIC-IV-ECHO credentialed evaluation, remainder writing).
- Compute: one 16-24 GB GPU trains a 2D CNN on single-frame view classification (hours) across the sources; frame extraction from videos is CPU/IO-bound. Disk: EchoNet-Dynamic ~7 GB, CAMUS ~3 GB, TMED-2 ~a few GB, MIMIC-IV-ECHO tens of GB (DICOM).

## Background

Automated echocardiography pipelines begin with **view classification** - deciding whether a clip is an apical 4-chamber (A4C), 2-chamber (A2C), parasternal long/short axis (PLAX/PSAX), etc. - because every downstream measurement (ejection fraction, strain, valve assessment) is view-specific. A view classifier that silently mislabels views on a new scanner corrupts everything downstream, yet view models are usually trained and tested on a single institution's images from a single vendor. Echocardiography is acquired on Philips, GE, Siemens, Canon and others, with different post-processing, sector geometry, frame rates and DICOM/AVI export - a large, mostly uncharacterised distribution shift.

Recent work confirms shift is real: EchoNet-trained segmentation drops 5-25% on Mayo POCUS and CAMUS (Foundation vs Domain-Specific Models, medRxiv 2023, 2023.09.19.23295772), and "Domain Shift in Echocardiography" (arXiv:2607.19643, 2026) quantifies and predicts cross-dataset LV-segmentation degradation. For *view classification* specifically, the new TTE47 benchmark (Naidoo et al., 2026, *Medical Image Analysis*, doi:10.1016/j.media.2026... , S1361841526000757) introduces 47 fine-grained views and reports leading TMED-2 performance without dataset-specific pretraining, and CardioBench (arXiv:2510.00520, 2025) aggregates eight datasets including a view-recognition task. But these evaluate *methods*; none isolates the *vendor* axis of the shift for view classification, nor decomposes it, nor maps which view confusions are induced by which vendor differences.

## The research gap

**What has been done (2023-2026):**

- **View classification datasets/methods**: TMED-2 (Huang et al., 2022, view + AS-severity), TTE47 (2026, 47-view benchmark with supervised contrastive learning), CardioBench (2025) view-recognition task, and view-primed foundation models (Nature 2025, s41586-025-09850-x).
- **Segmentation domain shift** is quantified (medRxiv 2023; arXiv:2607.19643, 2026) and echo foundation models (EchoFM, EchoCLIP, Echo-Vision-FM) claim generalisation but are benchmarked mostly on pooled or single-vendor test sets.
- **Robustness claims**: EchoNet-Measurements (JACC 2025) reports consistency across vendors *for measurement*, using screenshots/AVI from multiple vendors - encouraging, but not a controlled view-classification shift study with a vendor decomposition.

**What is specifically missing (the gap this project fills):**

1. No study isolates the **vendor/site axis** for view classification with a controlled leave-one-source-out grid across public datasets that differ in vendor/format (EchoNet=GE, Stanford AVI; CAMUS=GE, Belgium; TMED-2=Tufts; MIMIC-IV-ECHO=Philips/GE DICOM), reporting the drop *per view*.
2. No **decomposition** of the view-classification drop into acquisition components (sector geometry, frame rate, contrast/gamma, resolution) vs genuine anatomy, using controlled counterfactual renderings (re-scan-convert, re-window, re-sample frame rate).
3. No **clinically weighted confusion analysis** identifying which vendor differences induce the dangerous confusions (A4C<->A2C, PLAX<->apical) that would silently corrupt EF/strain, as opposed to harmless near-view confusions.
4. No head-to-head of **label-free domain-generalization / test-time adaptation** (histogram matching, instance/style normalization, test-time BN, foundation-model frozen features) on the same view-shift grid, measuring how much vendor robustness each buys without target labels.

This complements the sibling project `echonet-ef-uncertainty` (EF regression + uncertainty on EchoNet); here the object is the *upstream view classifier* and its vendor transportability.

## Research questions / hypotheses

1. **RQ1 (vendor drop).** Training view classification on source S and testing on target T (different vendor/site), how large is the per-view accuracy/macro-F1 drop vs in-source? *H1:* the drop is largest for geometrically similar views (A4C vs A2C) and for PSAX levels, and larger across vendors than within one vendor's studies.
2. **RQ2 (decomposition).** How much of the drop is removed by matching (a) intensity/contrast (histogram matching), (b) sector geometry/scan-conversion, (c) frame rate/temporal sampling? *H2:* intensity+geometry matching removes most of the drop for coarse views; residual reflects anatomy/labeling.
3. **RQ3 (dangerous confusions).** Are cross-vendor errors concentrated in clinically consequential view swaps? *H3:* yes - cross-vendor confusion mass on A4C<->A2C and apical<->parasternal exceeds within-source, and a clinically weighted error metric widens more than raw accuracy.
4. **RQ4 (label-free DG).** Which of {histogram matching, instance normalization, test-time BN adaptation, frozen echo-foundation-model features + linear head} best recovers target macro-F1 without target labels, at matched source performance? *H4:* frozen foundation-model features + instance normalization give the best label-free robustness; test-time BN helps most when target batch is homogeneous.
5. **RQ5 (downstream impact).** Does the view-classification drop translate into EF-pipeline error when a mislabeled view is fed to a view-specific EF model? *H5:* a measurable EF error is introduced by realistic view-confusion rates, quantified as MAE inflation.

## Datasets

| Dataset | What is used | Size | Vendor/site | Access | URL |
|---|---|---|---|---|---|
| EchoNet-Dynamic | A4C clips -> frames (single-view source + EF for RQ5) | 10,030 videos | GE, Stanford (AVI) | Free registration (DUA) | https://echonet.github.io/dynamic/ |
| CAMUS | A4C/A2C end-diastole/systole frames + view labels | 500 patients | GE, Univ. Hosp. St Etienne | Open (registration) | https://www.creatis.insa-lyon.fr/Challenge/camus/ |
| TMED-2 | Multi-view labelled 2D images (PLAX/PSAX/A2C/A4C) + AS severity | 599 studies, 5,261 images | Tufts | Free registration (DUA) | https://tmed.cs.tufts.edu/tmed_v2.html |
| TTE47 | 47 fine-grained view labels benchmark | public benchmark | multi | Open | https://thrive-centre.com/datasets/TTE47 |
| MIMIC-IV-ECHO v0.1 | DICOM echo studies, linked EHR; used as an independent-vendor target | ~7k studies, ~4,500 patients | Philips/GE (DICOM) | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimic-iv-echo/0.1/ |

Note: MIMIC-IV-ECHO ships DICOM without curated view labels; a subset must be view-labeled (a small clinician/annotation task) or pseudo-labeled with a validated classifier and confirmed on a sample. All other sources ship view labels.

## Methods

Pipeline (`src/echo_view/`):

1. **Frame extraction & harmonisation** (`frames.py`): read AVI/DICOM/PNG; extract representative frames (ED/ES where annotated, else uniform sampling); convert to grayscale; mask to the ultrasound sector; resize to a fixed 112x112 or 224x224; store provenance (source, vendor, frame rate).
2. **View-label crosswalk** (`views.py`): a canonical view set (A4C, A2C, A3C, PLAX, PSAX-{AV,MV,PM,apex}, subcostal, suprasternal, other) with per-dataset mappings; coarse (A4C/A2C/PLAX/PSAX/other) and fine (TTE47-aligned) granularities.
3. **Acquisition transforms** (`acquisition.py`): controlled counterfactuals - histogram/CDF matching to a reference vendor, sector re-masking/scan-conversion approximation, frame-rate/temporal resampling, gamma/contrast jitter - to build the decomposition and as augmentations.
4. **Models** (`models.py`): a 2D CNN (ResNet-ish, PyTorch, optional) view classifier and a handcrafted-feature + logistic baseline (sector shape descriptors, intensity histograms, structural-similarity to view templates) that runs CPU-only; frozen echo-foundation-model features + linear head where available.
5. **Shift diagnostics & decomposition** (`shift.py`): domain-classifier AUC (vendor discriminability), per-view drop, Shapley decomposition over acquisition transforms; clinically weighted confusion matrix.
6. **Label-free DG / TTA** (`adapt.py`): instance/style normalization, test-time BatchNorm statistics update, histogram matching at inference; measured on the leave-one-source-out grid.

Tools: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `pydicom` (DICOM), `opencv`/`imageio` (video), optional `torch`.

## Evaluation & statistics

- **Primary metric.** Macro-F1 and balanced accuracy over canonical views; per-view recall; a **clinically weighted error** that up-weights dangerous confusions (A4C<->A2C, apical<->parasternal).
- **Shift metric.** Drop = in-source minus target macro-F1; vendor-discriminability AUC; Shapley shares of the acquisition components.
- **Downstream metric (RQ5).** EF MAE inflation when EF is computed on frames fed through the (imperfect) view classifier vs on gold views.
- **Validation scheme.** Patient/study-level splits within each source; leave-one-source-out for the shift grid; DG/TTA fit only on source + unlabeled target (no target labels).
- **Leakage prevention.** No study appears in both train and test; the same patient's clips never split across folds; MIMIC-IV-ECHO pseudo-labels validated on a held-out human-labeled sample before use as target.
- **Multiple comparisons.** sources x views x methods: Benjamini-Hochberg FDR within each RQ; bootstrap CIs (study-level, 2000 resamples).
- **Nulls.** (i) Shuffle-vendor null for the vendor-discriminability claim. (ii) Within-vendor cross-site drop as the "no vendor shift" reference. (iii) A permutation confusion-matrix null to test whether dangerous confusions exceed chance given per-view accuracy.

## Publishable angle

- **Headline.** "Echocardiographic view classification loses X macro-F1 points across scanner vendors, the loss is dominated by intensity and sector-geometry differences and is concentrated in the clinically dangerous A4C/A2C confusion; instance normalization plus frozen foundation-model features recover most of it without any target labels, and residual view errors inflate downstream EF MAE by Y%."
- **Deliverables.** The vendor-shift view-classification benchmark (leave-one-source-out grid), the acquisition decomposition, the clinically weighted confusion analysis, and label-free DG baselines; the canonical view crosswalk.
- **Target venues.** *Medical Image Analysis*; *IEEE Transactions on Medical Imaging*; MICCAI / MIDL; *JACC: Cardiovascular Imaging* or *European Heart Journal - Cardiovascular Imaging* for the clinical framing.
- **Follow-ups.** Full-video (3D) view classification shift; adding vendor metadata from DICOM headers as an explicit factor; a released view-labeled MIMIC-IV-ECHO subset; joint view+quality control across vendors.

## Risks, confounds & mitigations

- **View label taxonomies differ** across datasets (coarse vs 47-view). Mitigation: map to a coarse canonical set for cross-dataset comparison; report fine-grained only within compatible sources; clinician-reviewed crosswalk.
- **Vendor is confounded with site/population.** Mitigation: within-vendor cross-site contrast (where available) isolates vendor from site; DICOM `Manufacturer` tag used where present (MIMIC-IV-ECHO) as ground truth vendor.
- **MIMIC-IV-ECHO lacks view labels.** Mitigation: human-label a stratified sample; validate pseudo-labels against it; report target results with the labeled-sample CI.
- **Frame selection bias** (ED/ES vs random). Mitigation: fixed sampling protocol; sensitivity to frame-selection strategy.
- **Foundation-model pretraining overlap** with test sources (EchoCLIP etc. trained on public echo). Mitigation: document pretraining data; treat overlapping targets as in-distribution.

## Milestones

- [ ] Register/download EchoNet-Dynamic, CAMUS, TMED-2; obtain TTE47; frame extraction + sector masking; tests.
- [ ] Canonical view crosswalk (coarse + fine); per-dataset label audit.
- [ ] Baseline view classifiers (handcrafted + CNN); in-source performance.
- [ ] Leave-one-source-out shift grid; per-view drops; vendor discriminability.
- [ ] Acquisition decomposition (Shapley over transforms); dangerous-confusion analysis.
- [ ] Label-free DG/TTA methods on the grid.
- [ ] PhysioNet credentialing; MIMIC-IV-ECHO target eval on a view-labeled sample.
- [ ] Downstream EF-impact (RQ5) using the sibling EF model; manuscript.

## Ethics / data-use notes

- EchoNet-Dynamic and TMED-2 require registration and a data-use agreement; CAMUS requires registration; follow each licence and cite the dataset papers.
- MIMIC-IV-ECHO is credentialed: complete PhysioNet CITI training and sign the DUA; DICOMs must be stored encrypted, never committed, and **never** sent to third-party LLM/API services except per PhysioNet's responsible-use policy.
- Credentials read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD` env vars only.
- Any manual view-labeling of MIMIC-IV-ECHO stays within the credentialed environment; released artefacts contain only de-identified derived labels/statistics, not images.
- Findings are reported to improve robustness and patient safety (dangerous-confusion analysis), not to rank vendors commercially.
- Related project in this repo: `echonet-ef-uncertainty` (downstream EF regression + uncertainty) - complementary; this project is self-contained.
