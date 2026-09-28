# mri-foundation-model-clinical-validation — Do brain-MRI foundation models survive real clinical data? A strict, quality- and pathology-stratified external validation on open scans

An open, reproducible benchmark that takes released brain-MRI foundation models (BrainSegFounder, SAM-Med3D / SAM-Med3D-Turbo, SAM-Brain3D and BrainIAC-style encoders, MedSAM as a 2D reference) and evaluates them — frozen, few-shot and fully fine-tuned — against task-specific nnU-Net and contrast-agnostic SynthSeg on public data that looks like clinical practice: paired clean/motion scans (MR-ART), stroke and MS lesions, low-quality glioma scans from Sub-Saharan Africa, memory-clinic scans, paediatric and very old brains, with performance modelled as a function of measured image quality and lesion load rather than reported as one Dice number.

## Status / difficulty / timeline / compute

- Status: design + starter code (synthetic degradation engine, segmentation metrics with paired statistics, quality/pathology stratification with robustness curves and learning-curve crossing points, model-adapter registry with a working baseline and CLI hooks).
- Difficulty: PhD-chapter / strong MSc with GPU access (9-12 months).
- Compute: inference for all models on ~3,000 volumes: ~100-200 GPU-hours (one 24-48 GB GPU). Few-shot and full fine-tuning across 5 label budgets × 4 tasks × 4 models: ~500-800 GPU-hours. nnU-Net from-scratch baselines: ~300 GPU-hours. Storage ~1.5 TB.
- Related projects in this repository: `openneuro-mriqc-audit` and `mriqc-scannability-equity` (IQMs used here as the quality axis). This folder is self-contained.

## Background

Foundation models for medical image segmentation promise generalization without task-specific training. BrainSegFounder (Cox et al., 2024, Medical Image Analysis) pretrains a Swin UNETR on > 42,000 UK Biobank scans and BraTS and reports gains on BraTS and ATLAS v2.0; SAM-Med3D (Wang et al., 2023, arXiv 2310.15161) adapts the Segment Anything paradigm (Kirillov et al., 2023, ICCV) to volumetric prompts across 247 categories; MedSAM (Ma et al., 2024, Nature Communications) does so in 2D; SAM-Brain3D (2025) initializes from SAM-Med3D-Turbo and trains on nine brain datasets with 14 MRI modalities; BrainIAC ("A generalizable foundation model for analysis of human brain MRI", 2026) is a self-supervised multiparametric encoder trained on ~49,000 scans; a 2025 systematic review ("Brain imaging foundation models, are we there yet?", arXiv 2506.13306) counts dozens more. Meanwhile SynthSeg (Billot et al., 2023, Medical Image Analysis) reaches contrast- and resolution-agnostic anatomy segmentation without pretraining on real images, and nnU-Net (Isensee et al., 2021, Nature Methods) remains the strongest task-specific baseline.

Two evaluation gaps persist. First, the benchmarks used to report foundation-model gains (BraTS, ATLAS) are curated, skull-stripped, resampled and quality-filtered; the scans that a hospital produces are not. Second, the one community effort that evaluates on clinical out-of-distribution data, the FOMO25 challenge (MICCAI 2025: infarct detection, meningioma segmentation, brain age on hidden multi-centre clinical data), keeps its test data private, so nobody can reproduce the ranking, stratify it by image quality, or add a new model later. Early evidence from zero-shot evaluations of SAM3/MedGemma on BraTS (2026) and from lightweight fine-tuning of SAM-Med3D for subcortical structures (Dice ≈ 0.70 on hippocampi under domain shift, 2025) suggests that direct performance on clinical data is not sufficient.

## The research gap

What has been done:

- Model papers evaluate on curated challenge sets (BrainSegFounder: BraTS, ATLAS v2.0; SAM-Med3D: aggregated public sets) with internal splits and single summary metrics.
- FOMO25 evaluates on hidden clinical data with few-shot fine-tuning, reporting rankings but not quality-stratified or pathology-stratified results, and not reproducible outside the challenge.
- Robustness-to-artefact studies exist for classical pipelines (FreeSurfer/SynthSeg on MR-ART; Nárai et al., 2022, Scientific Data, released the paired motion dataset for this purpose) and for image-enhancement foundation models (2024-2025), not for segmentation foundation models.
- Low-quality clinical glioma data (BraTS-Africa, part of BraTS 2023) has been used for tumour models, not for testing whether pretraining on high-quality UK Biobank scans transfers to it.

What is specifically missing:

1. A public, reproducible, *stratified* external validation: performance as a smooth function of measured quality (MRIQC IQMs: CJV, CNR, EFC, SNR) and of pathology load (lesion volume, tumour extent), with subject-level paired statistics, rather than a single mean Dice.
2. The paired-motion experiment: same subject, same session, with and without deliberate head motion (MR-ART, 148 adults, three motion levels). It isolates the artefact effect from population shift, and no foundation model has been evaluated on it.
3. The label-budget question that matters clinically: at how many labelled cases does a from-scratch nnU-Net match a frozen or few-shot-adapted foundation model, and does the answer change with image quality? ("crossing point" N*).
4. A pretraining-population test: BrainSegFounder's encoder saw healthy UK Biobank adults (40-70 y); performance on children (CANDI), the very old with atrophy (OASIS-3 > 85 y), memory-clinic scans (OASIS-4) and low-field / low-resolution data (M4Raw 0.3 T) tests whether "foundation" generalizes beyond its pretraining population.
5. Prompt honesty for SAM-style models: zero-shot results depend on oracle prompts derived from ground truth; reporting must separate oracle-prompt, automatic-prompt and prompt-free regimes.
6. Silver-vs-gold labels: much clinical validation uses FreeSurfer outputs as reference; quantifying the agreement ceiling on manually labelled sets (MindBoggle-101, CANDI, Hammers) is needed before silver labels are used at scale.

## Research questions / hypotheses

1. RQ1 (quality slope). Does Dice decline with worse IQMs more steeply for foundation models than for SynthSeg and nnU-Net? H1: frozen-encoder foundation models lose ≥ 0.10 Dice per 1 SD worsening in CJV in anatomy tasks, SynthSeg ≤ 0.05; fine-tuning closes half of the gap.
2. RQ2 (paired motion). On MR-ART, what is the within-subject Dice between motion-free and motion-corrupted segmentations, and the induced volume bias? H2: foundation models show larger volume bias in cortex (≥ 3%) than SynthSeg (≤ 1.5%) at the strongest motion level.
3. RQ3 (pathology). Does anatomy segmentation degrade near lesions, and do lesion models degrade with lesion load? H3: Dice for structures adjacent to stroke lesions drops by ≥ 0.15 for foundation models vs ≤ 0.08 for SynthSeg; lesion Dice for foundation models on BraTS-Africa is ≥ 0.10 below BraTS-adult.
4. RQ4 (label budget). Where is the crossing point N*? H4: nnU-Net from scratch matches few-shot foundation models at N* ≈ 25-50 labelled cases for lesions and ≈ 10-25 for anatomy; N* is larger on low-quality strata (the foundation-model advantage is largest where quality is poor — or it isn't, which is equally publishable).
5. RQ5 (population). Does performance on children and the very old fall below the pretraining-age range? H5: yes for BrainSegFounder-derived models (Dice −0.05 to −0.10 in CANDI and OASIS-3 > 85 y), not for SynthSeg.
6. RQ6 (prompts). How much of SAM-Med3D's zero-shot performance depends on oracle prompts? H6: automatic prompting (from a coarse SynthSeg or atlas prior) loses ≥ 0.10 Dice vs oracle points.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| MR-ART (OpenNeuro ds004173) | Paired T1w: clean + two levels of deliberate head motion, same session; expert quality ratings | 148 healthy adults, 3 scans each | Open (CC0) | https://openneuro.org/datasets/ds004173 |
| MindBoggle-101 | T1w with manually corrected cortical (DKT) labels | 101 subjects | Open (CC BY) | https://mindboggle.info/data.html , https://osf.io/nhtur/ |
| CANDI Share (schizophrenia bulletin set) | Paediatric psychiatric/control T1w with manual subcortical + cortical labels | 103 children/adolescents | Open (registration) | https://www.nitrc.org/projects/candi_share/ |
| Hammers adult atlases | T1w with 83-95 manually delineated regions | 30 adults | Free registration | https://brain-development.org/brain-atlases/ |
| ATLAS v2.0 | T1w with manual stroke-lesion masks, multi-site | 1,271 (655 with public masks) | INDI data-use agreement (lite) | https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html |
| ISLES 2022 | DWI/ADC/FLAIR with manual infarct masks | 400 cases (250 public train) | Open (Zenodo, CC BY) | https://isles22.grand-challenge.org/ , https://zenodo.org/record/7153326 |
| Shifts 2.0 (MS lesion track) | FLAIR/T1 with WML masks across sites, with explicit in/out-of-distribution splits | ~300 scans | Open (CC BY-NC-SA) | https://shifts.grand-challenge.org/ , https://github.com/Shifts-Project/shifts |
| BraTS 2023 (adult glioma + BraTS-Africa) | Multiparametric MRI with tumour sub-region labels; the Africa subset is lower-resolution/lower-quality clinical data | ~1,250 adult + ~60 Africa (training) | Synapse registration + data-use terms | https://www.synapse.org/brats2023 |
| OASIS-3 | T1w/FLAIR, FreeSurfer silver labels, age 42-95 (very old subset), CDR | ~1,300 participants | NITRC DUA | https://sites.wustl.edu/oasisbrains/ |
| OASIS-4 | Clinical memory-clinic T1w, heterogeneous protocols, FreeSurfer outputs | 663 participants | NITRC DUA | https://www.nitrc.org/projects/oasis4/ |
| M4Raw | 0.3 T multi-coil raw k-space with reconstructions (T1/T2/FLAIR) | 183 volunteers | Open (CC BY) | https://github.com/mylyu/M4Raw |
| FOMO-60K (optional pretraining reference) | Large public pretraining set released with FOMO25 (check current licence) | > 60,000 scans | Open (per challenge terms) | https://fomo25.github.io/ |

Models (weights): BrainSegFounder (https://github.com/lab-smile/BrainSegFounder), SAM-Med3D / -Turbo (https://github.com/uni-medical/SAM-Med3D), MedSAM (https://github.com/bowang-lab/MedSAM), SynthSeg (FreeSurfer ≥ 7.3 `mri_synthseg`), nnU-Net v2 (https://github.com/MIC-DKFZ/nnUNet), FastSurfer (https://github.com/Deep-MI/FastSurfer). Add SAM-Brain3D / BrainIAC / Brain-SAM if weights are released; the registry is model-agnostic.

## Methods

1. Data harmonization: BIDS layout for all sets; minimal preprocessing per model as specified by its authors (no extra denoising); record MRIQC IQMs for every input (`quality axis`); compute lesion loads from reference masks (`pathology axis`); define age strata.
2. Reference labels: manual where available (MindBoggle, CANDI, Hammers, ATLAS, ISLES, Shifts, BraTS); FreeSurfer 7 silver labels on OASIS-3/-4 and MR-ART clean scans, with the manual-vs-silver agreement ceiling estimated on MindBoggle/CANDI (`fm_clinical_val.metrics.silver_ceiling`).
3. Evaluation regimes: (a) frozen encoder + light decoder/linear probe trained on N ∈ {5, 10, 25, 50, 100} cases; (b) few-shot full fine-tuning with the same budgets; (c) zero-shot prompted (SAM-Med3D) with oracle vs automatic prompts; (d) nnU-Net from scratch at the same N; (e) SynthSeg/FastSurfer as no-training references. Fixed subject-level splits shared by all models; the validation data are never used for model selection of any kind.
4. Controlled degradations (`fm_clinical_val.degradations`): Rician noise, k-space motion (phase perturbation of random k-space lines), bias field, through-plane downsampling, ghosting, applied to clean MR-ART / MindBoggle scans at graded levels to obtain dose-response curves; validated against the real MR-ART motion levels.
5. Metrics (`fm_clinical_val.metrics`): Dice, 95% Hausdorff distance, average surface distance, absolute volume error, per structure; paired Wilcoxon and subject-level bootstrap CIs; test-retest style within-subject agreement for MR-ART.
6. Stratified analysis (`fm_clinical_val.stratify`): quality tertiles within dataset; mixed model Dice ~ model × quality + lesion load + age + (1 | subject) + (1 | dataset); robustness slopes; learning curves per model and the crossing point N* by fitting a saturating power law to Dice(N).
7. Reporting: per-task tables at each quality tertile; dose-response plots; N* table; prompt-regime table; TRIPOD+AI-style checklist adapted to segmentation; full code and predictions released.

Libraries: numpy, scipy, pandas, statsmodels, scikit-learn, nibabel; MONAI / PyTorch for adapters; nnU-Net v2; FreeSurfer for SynthSeg/FastSurfer; MRIQC container.

## Evaluation and statistics

- Primary endpoint: subject-level paired difference in Dice between each foundation-model regime and nnU-Net-from-scratch at N = 50, within each quality tertile; Wilcoxon signed-rank with Holm correction over tasks; bootstrap CIs (2,000 resamples over subjects).
- Quality slope: linear mixed model with random subject and dataset intercepts; slope contrasts between models with Wald tests; robustness to using the MRIQC classifier probability instead of individual IQMs.
- Non-inferiority: TOST with a pre-registered margin of 0.02 Dice for anatomy and 0.05 for lesions.
- Crossing point: bootstrap CI for N* from learning-curve fits.
- Leakage prevention: pretraining corpora overlap check — any subject/dataset in a model's pretraining set (e.g., BraTS in BrainSegFounder, public sets aggregated into SAM-Med3D) is flagged and analysed separately as "seen"; subject-level splits; no test-time selection.
- Nulls/sanity: label-shuffled learning curves; an untrained-encoder probe as the floor for "what does pretraining buy".

## Publishable angle

Headline: "On paired-motion, lesion-rich, paediatric and low-field open data, brain-MRI foundation models lose more accuracy per unit of image degradation than contrast-agnostic SynthSeg and match a from-scratch nnU-Net only below ~N* labelled cases; their advantage is confined to good-quality scans from the pretraining age range." Or its negation — either is decision-relevant for hospitals deciding whether to adopt foundation models. The reproducible stratified benchmark itself is a contribution (public predictions, harness, and a leaderboard that reports quality-tertile results instead of a single mean).

Target venues: Medical Image Analysis; IEEE Transactions on Medical Imaging; Radiology: Artificial Intelligence; MICCAI (benchmark/validation track) or MIDL; NeuroImage for the anatomy-focused part.

Follow-ups: extend to CT-derived brain segmentation; add uncertainty calibration under degradation; test image-enhancement foundation models as a pre-step.

## Risks, confounds and mitigations

- Weight availability and licence restrictions for some models: the registry is adapter-based; report what was runnable and why others were excluded.
- Pretraining-data overlap (BrainSegFounder used BraTS; SAM-Med3D aggregated many public sets): maintain an overlap ledger from the papers' data lists; analyse "seen" vs "unseen" separately.
- Silver-label noise on OASIS: estimate the ceiling on manually labelled sets; report Dice against manual labels wherever possible.
- Oracle-prompt optimism for SAM models: three prompt regimes reported side by side.
- Compute budget: prioritize MR-ART, MindBoggle, ATLAS, BraTS-Africa; the rest are extensions.
- Multiple models × tasks × budgets = many comparisons: a small number of pre-registered primary contrasts, the rest descriptive.

## Milestones

- [ ] Data access completed (OpenNeuro, MindBoggle, CANDI, ATLAS, ISLES, Shifts, BraTS 2023, OASIS-3/4, M4Raw) and BIDS-harmonized with IQMs.
- [ ] Model registry populated (BrainSegFounder, SAM-Med3D, SynthSeg, nnU-Net, FastSurfer) with unit-tested adapters.
- [ ] Silver-vs-manual ceiling estimated on MindBoggle/CANDI.
- [ ] Zero-shot / frozen results on all sets; MR-ART paired-motion analysis.
- [ ] Degradation dose-response curves validated against MR-ART.
- [ ] Few-shot and nnU-Net learning curves; crossing points.
- [ ] Mixed-model stratified analysis; pre-registered contrasts; manuscript; release predictions + harness.

## Ethics / data-use notes

- OASIS-3/-4 (NITRC DUA), ATLAS (INDI DUA), CANDI (registration), BraTS (Synapse terms) restrict redistribution; release only predictions and metrics for open-licensed inputs, never DUA-covered images or derived masks that could be re-identified.
- Do not send any participant-level image to third-party inference APIs; all models run locally.
- MR-ART participants consented to deliberate motion scans for methods research; cite Nárai et al. (2022) and follow the CC0 terms.
- Report performance disparities by age and by data source (e.g., BraTS-Africa vs adult BraTS) explicitly; do not average them away.
