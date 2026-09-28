# Ped-ECG-Gen: how far can adult-trained 12-lead ECG models be pushed into childhood?

**One-sentence pitch.** Quantify the *age-continuous* generalization decay of adult-trained deep ECG models as they are applied to progressively younger patients - using the new large open pediatric ECG database (ZZU-pECG) as the pediatric target and PTB-XL / CODE-15% / PhysioNet-2021 as adult sources - and test which adaptation strategy (age-conditioned normalization, pediatric fine-tuning, age-reweighting) recovers the most performance with the fewest pediatric labels, on the shared arrhythmia/conduction labels that are physiologically comparable across age.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are shipped.
- Difficulty: MSc-to-early-PhD (deep learning + pediatric cardiology domain knowledge + biostatistics).
- Timeline: 6-9 months (1-2 months harmonisation + age-normal ECG reference, 2 months adult baselines + zero-shot decay curve, 2 months adaptation methods + label-efficiency, remainder validation and writing).
- Compute: one 16-24 GB GPU trains a 1D-ResNet on ~350k adult ECGs (hours) and fine-tunes on ~12k pediatric ECGs (minutes-hours). Frozen-feature baselines run on CPU. Disk: ZZU-pECG ~a few GB; CODE-15% ~20 GB; PTB-XL ~3 GB.

## Background

The pediatric ECG is not a small adult ECG: normal ranges for heart rate, PR/QRS/QT durations, R/S amplitudes and T-wave polarity change dramatically from the neonate (right-ventricular dominance, rapid rates, T-wave inversions in right precordial leads) through adolescence toward adult morphology (Rijnbeek et al., 2001, Eur Heart J, pediatric normal limits; Davignon et al., 1980, Pediatr Cardiol). Adult-trained AI-ECG models therefore have a *shifting target*: the mapping from waveform to label is age-dependent, and the covariate distribution (rates, amplitudes) barely overlaps the adult training data for young children.

Most public ECG deep-learning resources are overwhelmingly adult (PTB-XL, PhysioNet/CinC 2020-2021, MIMIC-IV-ECG), and CODE-II explicitly *excludes* under-18s. Until recently there was no large, openly available pediatric 12-lead dataset with diagnostic labels; ZZU-pECG (Zhengzhou University Pediatric ECG Database; Nature *Scientific Data*, 2025, doi:10.1038/s41597-025-05225-z) changed this with 14,190 records from 11,643 children aged 0-14, providing both ECG diagnostic statements and ICD-10 disease labels. This makes an *age-resolved transportability study* possible for the first time on open data.

## The research gap

**What has been done (2023-2026):**

- **Pediatric AI-ECG within pediatric data**: strong models exist for pediatric LV dysfunction (Mayo pediatric AI-ECG, *Circulation* 2024, doi:10.1161/CIRCULATIONAHA.123.067750) and congenital heart disease screening (PMC12020324), trained and tested in children.
- **Adult-to-pediatric transfer is known to be hard**: recent work states adult models "failed to generalize" to children and proposes cross-modal or pediatric-dominant pretraining (e.g., "Knowledge-Guided Cross-Modal Fusion for Adult-to-Pediatric ECG Transfer", arXiv:2607.15928, 2026; and a pediatric-dominant foundation model, ECG-Fyler, medRxiv 2026 with 782k ECGs). A 2025 benchmark of deep learning for multi-label pediatric ECG classification (arXiv:2510.03780) uses ZZU-pECG but focuses on *within-pediatric* method comparison.
- **Deep survival across ages**: "Deep Survival Analysis from Adult and Pediatric ECGs" (arXiv:2406.17002, 2024) is multi-center but targets survival, not diagnostic transportability, and does not chart the decay as a smooth function of age.

**What is specifically missing (the gap this project fills):**

1. No study charts the **generalization decay as a continuous function of patient age** for adult-trained diagnostic models on open data - i.e., AUROC(age) from neonate to adolescent - with a null model and confidence bands, isolating *where* transfer breaks.
2. No study **decomposes** the pediatric drop into (a) covariate shift (rates/amplitudes outside adult support), (b) label-prior shift (different disease mix), and (c) genuine concept shift (age-dependent waveform-to-label mapping), using the same counterfactual toolkit (age-reweighting, amplitude/rate renormalisation) as for cross-country transfer.
3. No **label-efficiency curve** for pediatric adaptation: how many pediatric labels does age-conditioned normalization or fine-tuning need to match a pediatric-only model, per label - the practical question for a hospital with few pediatric ECGs.
4. No use of **age-normalized ECG features** (z-scoring intervals/amplitudes against published pediatric normal limits, Rijnbeek 2001) as a physiologically grounded domain-adaptation input, benchmarked against learned adaptation.

This is complementary to `ecg-cross-dataset-generalization` (cross-country/adult population shift) and `cross-dataset-seizure-generalization` (EEG age shift): here the shift axis is *pediatric age* and the target is a specific open pediatric dataset.

## Research questions / hypotheses

1. **RQ1 (decay curve).** For an adult-trained model on shared labels (e.g., sinus tachycardia/bradycardia, AF/atrial flutter where present, RBBB, LBBB/IVCD, first-degree AV block, PVC/PAC, sinus rhythm), how does test AUROC vary as a smooth function of pediatric age (0-14 y)? *H1:* AUROC declines monotonically toward younger ages and is worst in infants (<1 y), where right-precordial T-inversion and fast rates most violate the adult prior.
2. **RQ2 (shift decomposition).** How much of the infant-vs-adolescent gap is removed by (a) rate/amplitude renormalisation to pediatric norms, (b) age-reweighting the (unavailable) adult support, (c) recalibration? *H2:* covariate renormalisation removes most of the gap for rate-defined labels; residual gap for morphology labels reflects concept shift.
3. **RQ3 (adaptation & label efficiency).** Among {zero-shot, recalibration only, age-conditioned batch-norm/feature-standardisation, linear-probe on frozen adult features, full fine-tuning}, which reaches within 2 AUROC points of a pediatric-only model with the fewest pediatric labels? *H3:* age-conditioned normalization + linear probe is the most label-efficient; full fine-tuning wins only with >2-3k pediatric labels.
4. **RQ4 (physiological features vs learned).** Do age-normalized handcrafted features (intervals/amplitudes z-scored to Rijnbeek norms) transfer better zero-shot than raw deep features? *H4:* yes for interval/rate labels; deep features win after fine-tuning.
5. **RQ5 (subgroup safety).** Are there age bands where the adult model is *confidently wrong* (high-confidence errors) on benign age-normal variants (e.g., calling infant T-inversion "ischemia-like")? *H5:* confident-error rate spikes in infants for repolarisation-related outputs.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| ZZU-pECG (Zhengzhou U. Pediatric ECG) | Pediatric target; 12-lead waveforms, age 0-14, ECG statements + ICD-10 | 14,190 records, 11,643 children, 500 Hz, 5-120 s | Open (figshare, published in Sci Data 2025) | https://doi.org/10.1038/s41597-025-05225-z (data on figshare) |
| PTB-XL v1.0.3 (Germany, adult) | Adult source; SCP-ECG labels, age, sex | 21,799 ECGs, 500 Hz, 10 s | Open (PhysioNet) | https://physionet.org/content/ptb-xl/1.0.3/ |
| CODE-15% (Brazil, adult) | Adult source; 6 classes, age, sex | 345,779 ECGs, 400 Hz | Open (Zenodo) | https://zenodo.org/records/4916206 |
| PhysioNet/CinC 2021 training | Adult multi-source; SNOMED labels | ~88k ECGs, 500 Hz | Open | https://physionet.org/content/challenge-2021/1.0.3/ |
| Rijnbeek 2001 pediatric normal limits | Age-banded normal ranges for HR/PR/QRS/QT/amplitudes (reference table) | reference | Open (published tables) | Eur Heart J 2001;22:702-711 |

Note: ZZU-pECG includes 9-lead records (1,856) and 12-lead records (12,334); only 12-lead records are used for cross-dataset comparison. Adult sources are 500 Hz except CODE-15% (400 Hz, resampled).

## Methods

Pipeline (`src/ped_ecg/`):

1. **Harmonised loading** (`loaders.py`): convert every record to (12, 5000) at 500 Hz, standard lead order, fixed 10 s crop/pad; CODE-15% resampled 400->500 Hz; per-record z-scoring. Parse age (years, plus an infant/child/adolescent band) and sex.
2. **Label harmonisation** (`labels.py`): a mapping from ZZU-pECG statements/ICD-10 and adult SCP/SNOMED codes to a shared set of *age-comparable* labels (rhythm and conduction classes that are defined the same way across age); age-specific-only pediatric labels (e.g., congenital) are held out as pediatric-only targets.
3. **Age-normalised features** (`pediatric_norms.py`): implement Rijnbeek 2001 age-banded normal limits; compute z-scores of measured HR/PR/QRS/QT and key amplitudes against the age-appropriate norm, as physiologically grounded adaptation inputs.
4. **Models** (`models.py`): adult-trained 1D-ResNet (PyTorch, optional) and a handcrafted-feature + logistic baseline; adaptation variants - recalibration (Platt/temperature), age-conditioned feature standardisation, frozen-feature linear probe, full fine-tune.
5. **Decay curve & decomposition** (`decay.py`): fit AUROC as a function of age (LOESS/spline with bootstrap bands); Shapley decomposition of the infant-vs-adolescent gap into covariate / prior / concept components using renormalisation and reweighting counterfactuals; confident-error analysis.
6. **Label-efficiency** (`adaptation.py`): learning curves of each adaptation method vs number of pediatric labels (log-spaced), compared to a pediatric-only model trained from scratch.

Tools: `wfdb`, `numpy`, `scipy`, `pandas`, `scikit-learn`, optional `torch`; statistics with `numpy`/`scipy`.

## Evaluation & statistics

- **Primary metric.** Macro AUROC on shared labels; the headline is the AUROC-vs-age curve with bootstrap confidence bands and an age-permutation null.
- **Decomposition metric.** Fraction of the infant-adolescent gap attributable to each component (Shapley-averaged over operation orderings), with CIs.
- **Label-efficiency metric.** Pediatric labels needed to reach within 2 AUROC points of the pediatric-only ceiling, per method and per label.
- **Validation scheme.** Patient-level splits in every dataset (ZZU-pECG by child ID; PTB-XL official folds; CODE-15% by patient_id). Zero-shot: no pediatric data touch adult training. Adaptation: pediatric train/val/test disjoint by child.
- **Leakage prevention.** No child appears in both adaptation-train and test; normalization statistics and calibration temperatures fit on the source or the pediatric-train only.
- **Multiple comparisons.** labels x age bands x methods: Benjamini-Hochberg FDR within each RQ; effect sizes with CIs primary.
- **Nulls.** (i) Age-permutation null for the decay curve (shuffle ages, refit curve). (ii) Label-shuffle model for chance transfer. (iii) A "matched-rate" placebo: subsample adolescents to the infant rate distribution to check that the decay is not purely a rate artefact.

## Publishable angle

- **Headline.** "Adult-trained 12-lead ECG models degrade smoothly toward infancy, with the drop dominated by covariate (rate/amplitude) shift for rhythm labels and by genuine age-dependent concept shift for repolarisation morphology; age-conditioned normalization recovers most of the loss with an order of magnitude fewer pediatric labels than full fine-tuning."
- **Deliverables.** The AUROC-vs-age transportability curve per label; the shift decomposition; label-efficiency curves for adaptation; an age-normalized-feature adaptation baseline; open code and the harmonised label map.
- **Target venues.** *npj Digital Medicine*; *Journal of Electrocardiology* / *Journal of the American College of Cardiology: Advances*; *IEEE JBHI*; ML4H / CHIL. Pediatric-cardiology venues (*Pediatric Cardiology*) for the clinical framing.
- **Follow-ups.** Neonatal (<1 mo) extension if a neonatal set becomes available; adding pediatric-dominant foundation models (ECG-Fyler) as a comparator; congenital-heart-disease detection as a pediatric-only target; prospective calibration in a children's hospital.

## Risks, confounds & mitigations

- **Label-scheme mismatch** between ZZU-pECG (Chinese guidelines + ICD-10) and adult SCP/SNOMED. Mitigation: restrict primary analysis to age-comparable rhythm/conduction labels; explicit crosswalk with clinician review; report per-label kappa.
- **Device/acquisition confound** (MedEx MECG-300 in ZZU vs Western carts). Mitigation: spectral fingerprints and a same-device sanity check; treat device as a nuisance in the decomposition, acknowledge partial confounding with country.
- **Age is confounded with disease prevalence** (the pediatric cohort is hospitalised). Mitigation: label-prior reweighting; report prevalence per age band; AUPRC alongside AUROC.
- **Small counts for rare pediatric labels / young infants.** Mitigation: pool age bands where N is low; report N and CIs; drop labels with insufficient positives.
- **ZZU records vary 5-120 s and include 9-lead.** Mitigation: standard 10 s window, 12-lead only for cross-dataset comparison; sensitivity to window length.

## Milestones

- [ ] Obtain ZZU-pECG (figshare) and adult sources; harmonised loaders; tests.
- [ ] Implement Rijnbeek age-normal limits and z-score features.
- [ ] Freeze age-comparable label crosswalk; per-label kappa; data card.
- [ ] Adult baselines (handcrafted + 1D-ResNet); zero-shot decay curve with bands.
- [ ] Shift decomposition; confident-error analysis by age band.
- [ ] Adaptation methods + label-efficiency curves; pediatric-only ceiling.
- [ ] Statistics, nulls, manuscript.

## Ethics / data-use notes

- ZZU-pECG is openly released (Sci Data 2025) under its stated licence; cite the dataset paper and follow any figshare terms. It is de-identified pediatric data - handle with care, do not attempt re-identification.
- PTB-XL, CODE-15%, PhysioNet-2021 are open; cite original papers.
- No credentialed data are required for the core study; if MIMIC-IV-ECG is added as an adult comparator, follow PhysioNet credentialing/DUA and never send data to third-party LLM/API services except per policy.
- Pediatric findings are reported to characterise model safety across age, not to deploy adult models in children; the confident-error analysis is included precisely to flag unsafe transfer.
- Related projects in this repo: `ecg-cross-dataset-generalization`, `cross-dataset-seizure-generalization` - complementary shift studies; this project is self-contained.
