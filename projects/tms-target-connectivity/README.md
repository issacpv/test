# Connectivity × E-Field Target Scoring for Non-Invasive Brain Stimulation

**Stimulation network mapping predicts effects from normative connectivity while treating dose as if it were constant. This project rebuilds target scoring as a joint connectivity × E-field model, and audits how much of the published connectivity-effect literature survives a properly spatially constrained null.**

## Status / difficulty / timeline / compute

| | |
|---|---|
| Status | Design complete; starter code runs end-to-end on synthetic connectomes and field maps |
| Difficulty | Late-MSc to PhD. Needs neuroimaging data engineering (CIFTI, BIDS, FEM meshes) plus real statistical care |
| Timeline | 9–14 months. The normative-connectome arm is tractable in 6; the individualized arm adds 4–6 |
| Compute | Group-average arm: a laptop with 32 GB RAM and ~200 GB disk. Individualized arm: a workstation or small cluster, ~2–5 TB disk, and ~100–500 CPU-hours for per-subject SimNIBS head models (`charm` is ~1–2 h per subject) |
| Storage | 200 GB (normative only) to ~3 TB (per-subject HCP rfMRI + diffusion + head models) |

## Background

Two literatures aim at the same clinical question — *where should we put the coil?* — and barely speak to each other.

**Network mapping.** Fox et al. (2012, *Biol Psychiatry*) observed that the antidepressant efficacy of a dorsolateral prefrontal TMS site tracks its resting-state functional connectivity with the subgenual cingulate, measured in a normative healthy cohort rather than in the patient. Weigand et al. (2018, *Biol Psychiatry*) prospectively validated the relationship, and Cash et al. (2021, *Biol Psychiatry*) turned it into individualized targeting. The framework generalized: lesions, stimulation sites and focal atrophy all localize to shared networks (Siddiqi et al., 2020, *Am J Psychiatry*, "Distinct symptom-specific treatment targets for circuit-based neuromodulation"; reviewed in Siddiqi, Kording, Parvizi & Fox, 2022, *Nat Rev Neurosci*). Work through 2025–2026 continues in this vein, including causal network mapping applied to preclinical stimulation (*Neuropsychopharmacology* 2025, doi:10.1038/s41386-025-02153-9) and connectome modelling of the white-matter routes from DLPFC to subgenual cingulate (*Nature Neuroscience* 2026, doi:10.1038/s41593-026-02248-6).

**E-field modelling.** In parallel, FEM tools made individualized dose computable: SimNIBS (Thielscher, Antunes & Saturnino, 2015, *EMBC*; segmentation pipeline in Puonti et al., 2020, *NeuroImage*) and ROAST (Huang, Datta, Bikson & Parra, 2019, *J Neural Eng*). The consequence is now well documented: at fixed stimulator output, the field delivered to a fixed cortical target varies substantially between people, because scalp-to-cortex distance, skull thickness and gyral geometry all differ. This motivated E-field-based dosing, in which output is set by the field delivered to M1 at motor threshold rather than by %RMT ("Electric-field-based dosing for TMS", *Imaging Neuroscience* 2024, doi:10.1162/imag_a_00106).

Put the two together and a problem appears. A network-mapping analysis correlates a **connectivity map** with an **effect map** across stimulation sites. But sites differ systematically in how much field they receive — lateral, superficial targets receive more field than medial or sulcal ones at the same output. Scalp-to-cortex distance is itself spatially structured, and so is connectivity. A correlation between connectivity and effect can therefore be produced, in part or in whole, by dose. Almost no network-mapping paper measures the field, and almost none uses a spatial null that accounts for the smoothness of the maps being compared.

## The research gap

**What has been done.**

- Network mapping has been developed and prospectively validated, but with dose represented as at most a coil-to-cortex distance covariate (Fox et al., 2012; Weigand et al., 2018; Cash et al., 2021; Siddiqi et al., 2020, 2022).
- E-field dosing has been developed and validated as a way to equate stimulation strength across targets and people (*Imaging Neuroscience* 2024, doi:10.1162/imag_a_00106).
- Spatial null methodology is mature and available: the spin test (Alexander-Bloch et al., 2018, *NeuroImage*), variogram matching / BrainSMASH (Burt, Helmer, Shinn, Anticevic & Murray, 2020, *NeuroImage*), and the comparison showing these families are **not** interchangeable (Markello & Misic, 2021, *NeuroImage*).
- **Closest competing work:** a 2026 paper in the *Biological Psychiatry* family describes prospective development of a combined whole-brain connectivity and electric-field model for targeting (ScienceDirect S2667174326000807). This is the same instinct as the joint model here, and it means the bare idea "combine connectivity with E-field" is no longer novel on its own. **Re-check this paper's final scope at write-up and cite it as concurrent work.**

**What is specifically missing.** Three things, in descending order of how badly they are needed:

1. **A spatial-null audit of the network-mapping literature.** The field's central correlations are between two smooth maps. Published p-values are overwhelmingly parametric or based on permutations that do not preserve spatial autocorrelation. Nobody has taken a set of these comparisons, reproduced them on open data, and asked how many survive a spin test *and* a variogram-matched null. Our own synthetic check (`tests/test_tms_target.py::test_spatial_null_is_more_conservative_than_parametric`) shows the parametric test rejecting far above nominal rate on independent smooth maps — this is not a small correction.
2. **Dose as a confounder, not a covariate.** The question is not only "does adding E-field improve prediction" (which the 2026 work addresses) but "**how much of the existing connectivity-effect relationship is attributable to dose?**" That is a mediation/confounding decomposition, and it is answerable: with the field computed at every candidate site, connectivity's unique contribution can be separated from the part it shares with dose. If that shared part is large, a body of targeting recommendations needs re-derivation.
3. **Normative versus individualized, with dose held constant.** Individualized connectomes are noisier but subject-specific; normative ones are cleaner but generic. Every comparison to date has confounded this with dose, because an individualized analysis uses the subject's own anatomy — which changes the field too. Separating "individual connectivity helps" from "individual anatomy changes the field" requires computing both per subject, and it has not been done.

**Angle.** A joint model *plus* a confound audit *plus* rigorous nulls — evaluated against **measured, site-specific stimulation effects** from open concurrent TMS–EEG/fMRI datasets rather than clinical outcomes alone. Measured local effects are the right dependent variable for testing a dose model, because clinical outcome sits many steps downstream of the field.

## Research questions and hypotheses

1. **Null-model audit.**
   *H1:* Of the connectivity–effect map correlations we reproduce, ≥ 30% that are nominally significant at p < 0.05 parametrically fail to reach p < 0.05 under variogram-matched nulls; and spin and variogram nulls disagree on ≥ 15% of comparisons, consistent with Markello & Misic (2021).
2. **Dose confounding.**
   *H2:* Per-parcel E-field magnitude is itself correlated with seed-based connectivity to standard reference regions (|ρ| > 0.2, spatial null p < 0.05), establishing that dose is a confounder rather than an independent predictor.
3. **Joint model performance.**
   *H3:* The network-dose score (field-weighted connectivity, Eq. 1 in `scoring.py`) predicts measured stimulation effects better than connectivity-only scoring, with a positive paired difference in leave-one-subject-out Spearman ρ whose 95% cluster-bootstrap CI excludes 0.
4. **Thresholded versus linear dose response.**
   *H4:* A thresholded weighting `relu(E - E_th)` outperforms a linear one, and the fitted `E_th` falls in the 40–100 V/m range implied by the cortical-activation literature. *This is the test that makes dose non-trivial:* under a linear weighting a global output rescale cancels out of any correlation, so only a threshold makes absolute dose matter.
5. **Unique contribution of connectivity.**
   *H5:* After partialling out local and network dose, connectivity retains a statistically unique contribution (increment in cross-validated ρ > 0.05 with a CI excluding 0). If it does not, the honest conclusion is that a substantial part of network mapping's predictive value is dose, and we report that.
6. **Normative versus individualized.**
   *H6:* With dose held constant by using the same head model for both, individualized connectomes do **not** outperform the HCP normative connectome for n ≤ 30 min of rfMRI per subject — because normative SNR beats individual specificity at realistic scan durations. Falsifiable in either direction and interesting either way.
7. **Structural versus functional.**
   *H7:* Structural (tractography) network dose predicts effects at short latency (early TMS-evoked EEG components, < 50 ms) better than functional network dose, while functional wins for late components and BOLD — because early propagation is constrained by direct anatomical connections.

## Datasets

| Name | What is used | Size | Access level | URL |
|---|---|---|---|---|
| HCP S1200 group average | MSM-All group-average dense functional connectome (n = 1003, and the n = 812 r227 subset) and parcellated FC; the normative connectome | ~30–80 GB for dense products | **Free registration** + Open Access Data Use Terms | https://db.humanconnectome.org |
| HCP S1200 per-subject | rfMRI parcellated timeseries (`*Atlas_MSMAll_hp2000_clean.ptseries.nii`), T1w/T2w for head models, preprocessed diffusion for tractography; the individualized arm | ~1–3 TB for 100 subjects; far less if only parcellated timeseries | **Free registration** (Open Access); restricted family/behavioural variables need a separate Restricted Data agreement — **not needed here** | https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release |
| OpenNeuro `ds004024` | TMS-EEG + MRI/fMRI/DWI, paired associative stimulation and connectivity (Shirley Ryan AbilityLab). 13 participants, tasks `ccPAS`, `spTMS`, `rest`; BIDS 1.6.0. Provides measured site-specific TMS-evoked responses **with** the structural images needed to build each subject's head model | ~1 TB total — download by modality | **Open**, no registration | https://openneuro.org/datasets/ds004024 |
| OpenNeuro `ds005498` | Single-pulse concurrent TMS-fMRI; measured BOLD response to stimulation at known sites | Check the latest snapshot | **Open** | https://openneuro.org/datasets/ds005498 |
| OpenNeuro, further TMS/tDCS datasets | Screened by keyword search via the GraphQL API (`scripts/download_data.py --search-openneuro`); the catalogue grows, so the candidate list is generated rather than hard-coded | varies | **Open** | https://openneuro.org |
| HCP-MMP1 parcellation | 360-parcel cortical parcellation plus parcel centroids for the spatial nulls | < 100 MB | **Open** (Glasser et al. 2016 supplement / BALSA) | https://balsa.wustl.edu |
| SimNIBS `ernie` example | Reference head model for pipeline development before per-subject models exist | ~3 GB | **Open** with SimNIBS | https://simnibs.github.io/simnibs/build/html/dataset.html |

**No credentialed (CITI-training) data is required.** HCP Open Access needs registration and acceptance of data-use terms but no ethics certification. OpenNeuro datasets are fully open.

## Methods

**Step 1 — normative connectome.** Load the HCP S1200 group-average connectome (`connectome.load_parcellated_connectome` for parcellated, `dense_seed_connectivity` for row-wise access to the dense CIFTI — never load a `.dconn` into memory). Keep everything in Fisher-z space; averaging raw r is biased toward zero and the bias is spatially non-uniform, which distorts exactly the pattern under test. Build structural connectomes from HCP diffusion via MRtrix3 tractography, then `structural_to_weights` (log1p, symmetrize, normalize) because raw streamline counts span orders of magnitude and are distance-biased.

**Step 2 — per-subject head models and fields.** Run SimNIBS `charm` on each subject's T1w/T2w, then simulate the coil position and orientation recorded in the stimulation dataset. Read the resulting `.msh` with `efield.load_efield_msh(..., tissue_tags=(2,))` — restricting to grey matter, since field in skull and scalp would otherwise set the map's peak. Parcellate with `efield.parcellate_field` onto the same atlas as the connectome: field and connectivity must share a parcellation before any joint model or null.

**Step 3 — dose normalization.** `efield.normalize_to_motor_threshold` rescales each session so the 99th-percentile field in M1 equals a fixed reference value. Field solutions are linear in output, so this is exact. Report every analysis twice: with %RMT dosing (as the source studies did) and with E-field dosing. The difference between those two answers is itself a result.

**Step 4 — target scores.** `scoring.score_targets` computes four scores per candidate reference region: connectivity-only (peak-field parcel as seed — the fairest baseline, since that is what a careful connectivity-only study would use), network dose with linear weighting, network dose with thresholded weighting, and local dose alone.

**Step 5 — outcome extraction.** From the TMS–EEG datasets, per-session measured effects: TMS-evoked potential amplitude in defined windows (early < 50 ms, mid 50–150 ms, late > 150 ms) at sensor clusters and after source projection; from TMS–fMRI, the BOLD response amplitude per parcel. Preprocessing follows each dataset's published pipeline; the TMS artefact window is excised, not interpolated-and-analysed.

**Step 6 — nulls.** For every map comparison, generate 10 000 variogram-matched surrogates (`nulls.variogram_surrogates`, primary — works for volumetric and subcortical data, which the spin test cannot) and 10 000 spin surrogates (`nulls.spin_surrogates`, secondary, hemisphere-constrained). Generate surrogates **once** per map and reuse them across comparisons, so every test faces the same null realisations. Report the naive parametric p alongside, and the inflation factor between them. Also report `nulls.effective_dof`, which typically shows a 360-parcel map carrying only a few dozen independent observations.

**Step 7 — models and evaluation.** Ridge regression on standardised scores (`scoring.fit_score_model`) — ridge, not OLS, because the features are collinear by construction (network dose contains connectivity; local dose is part of network dose). Evaluation is **leave-one-subject-out** (`scoring.grouped_cv_predictions`): two sessions from one subject share anatomy, coil calibration and head position, so splitting within subject leaks the very thing the model must generalize across. The normative-versus-individualized comparison is paired over identical folds (`scoring.compare_normative_vs_individual`) with a group-level bootstrap.

**Baselines.** (a) Connectivity-only, the literature standard. (b) Local dose alone. (c) Euclidean distance from the stimulation site to the reference region — the trivial spatial baseline that any network model must beat, and one that is very rarely reported. (d) Scalp-to-cortex distance alone.

**Tooling.** numpy, scipy, pandas, nibabel, meshio, requests; SimNIBS or ROAST for fields; MRtrix3 for tractography; `brainsmash` / `neuromaps` as reference null implementations to cross-check ours; MNE-Python for the TMS–EEG arm.

## Evaluation and statistics

- **Primary outcome:** paired difference in leave-one-subject-out Spearman ρ between joint and connectivity-only scoring, with a subject-level bootstrap CI.
- **Audit outcome:** the fraction of reproduced comparisons whose significance changes under a spatial null, plus the median inflation factor `p_spatial / p_parametric`.
- **Validation scheme:** leave-one-subject-out within dataset; leave-one-*dataset*-out across datasets, which is the real test of transportability given how much TMS–EEG preprocessing varies between labs.
- **Leakage prevention.** Four specific hazards: (i) never split within subject; (ii) fit any threshold `E_th` inside the training fold only — fitting it on all data is the easiest way to manufacture H4; (iii) the normative connectome must not include subjects who also appear in the stimulation dataset (they do not overlap here, but check); (iv) parcel selection and any feature screening happen inside the fold.
- **Multiple comparisons:** the confirmatory family is H1–H7. Benjamini–Hochberg at q = 0.05 within the family. Per-parcel maps get FDR across parcels *on top of* the spatial null, since the null controls the map-level statistic, not per-parcel inference.
- **Nulls, three kinds:** (i) *spatial* — variogram and spin, as above; (ii) *dose-preserving* — permute connectivity while holding the field fixed, isolating connectivity's contribution; (iii) *site permutation* — shuffle which effect came from which stimulation site within subject, the null for the whole targeting claim.
- **Sensitivity analyses:** parcellation (HCP-MMP1 360 vs Schaefer 200 vs Schaefer 400); `E_th` grid; grey-matter-only versus whole-brain field; `p99` versus mean dose summary; Pearson versus Spearman; with and without dose normalization; tissue conductivity ±20%, since conductivity uncertainty is a known dominant error source in FEM field estimates.
- **Power.** With 13 subjects in `ds004024` and multiple sessions each, this is a small-*n* study. Treat effect sizes as estimates with wide intervals, pool across datasets where preprocessing permits, and pre-register the primary comparison. Do not chase H7 subcomponents at this n.

## Publishable angle

**Headline.** *"Re-analysing open TMS datasets with individualized E-field models, we find that a substantial fraction of connectivity–effect relationships does not survive a spatially constrained null, and that E-field magnitude accounts for a measurable share of the apparent connectivity effect. A joint connectivity × E-field target score improves prediction of measured stimulation effects over connectivity alone, and a normative connectome is not outperformed by individualized connectivity at realistic scan durations."*

Either direction of H5 is publishable, which is the mark of a well-posed question: if connectivity retains a unique contribution, network mapping is vindicated and now has a dose-corrected form; if it does not, a widely used targeting rationale needs revisiting.

**Target venues.**
- *Imaging Neuroscience* (MIT Press) — best fit: open access, explicitly receptive to methodological audits and null-model work.
- *Brain Stimulation* — the clinical-translation audience that acts on targeting recommendations.
- *NeuroImage* — for the null-model methodology if that becomes the paper's centre of gravity.
- *Human Brain Mapping* — solid fallback with tolerance for a long supplement.
- *OHBM* or *Brain Stimulation Conference* for the initial abstract.

**Follow-ups.** (1) Release the E-field maps and network-dose scores for all open TMS datasets as a reusable derivative — that artefact alone is a *Scientific Data* paper. (2) Prospective validation: pick targets by joint score versus connectivity-only score in the same subjects and compare measured effects. (3) Extend to tDCS/tACS, where montages are far less focal and dose weighting should matter more. (4) Compose with morphological susceptibility (see the companion `morphology-dependent-stimulation` project) for a three-level model: field → neuron → network.

## Risks, confounds and mitigations

| Risk | Why it bites | Mitigation |
|---|---|---|
| **Small n in open concurrent TMS datasets** | 13 subjects will not support a 7-hypothesis confirmatory family with narrow CIs | Pre-register a single primary comparison (H3); pool datasets; report CIs not just p-values; frame secondary hypotheses as exploratory. Use the OpenNeuro search to find datasets added since 2025 |
| **Coil position and orientation may be under-documented** | The field model is only as good as the recorded coil placement; a 1 cm error moves the peak substantially | Prefer datasets with neuronavigation logs or digitised coil transforms. Where only a scalp landmark is given, propagate the uncertainty: simulate a distribution of plausible placements and report score variance across it |
| **Conductivity uncertainty** | FEM field magnitudes carry ~±20% uncertainty from tissue conductivity alone | Sensitivity analysis over conductivity; lean on *relative* field patterns and on rank-based statistics rather than absolute V/m |
| **Our null implementations are not the reference ones** | A methodological audit must not itself be methodologically shaky | Cross-check against `brainsmash` and `neuromaps` on the same maps and report agreement; ship `tests/` demonstrating calibration; use the reference packages for the final published numbers |
| **Circularity in the connectivity-only baseline** | Seeding at the peak-field parcel already uses the field | Report both: seeding at the *nominal* target (what the original study used) and at the peak-field parcel. The gap between them is informative about how much "connectivity-only" already smuggles in dose |
| **The 2026 joint-model paper may pre-empt H3** | The constructive half of the project could be scooped | The audit (H1, H2, H5) and the normative-versus-individualized separation (H6) are not covered by it. Re-check at write-up and reposition as replication-plus-audit if needed |
| **TMS–EEG artefacts** | Early-component amplitudes are notoriously contaminated by muscle and decay artefacts | Follow each dataset's published pipeline; excise rather than interpolate the artefact window; pre-register the analysis windows; report results with and without the earliest window |
| **HCP is young and healthy** | A normative connectome from 22–35-year-olds may not describe an older clinical cohort | State the limitation; where an age-matched normative connectome exists (e.g. from an open aging cohort), repeat the primary analysis with it as a robustness check |
| Parcellation dependence | Results can hinge on parcel granularity | Three parcellations, reported in full, not just the best one |

## Milestones

- [ ] **M1 (month 1)** — HCP registration; group-average connectome downloaded and parcellated; OpenNeuro candidate search run and hand-screened; dataset inclusion table fixed
- [ ] **M2 (month 2)** — Parcellation and centroids in place; null models cross-checked against `brainsmash` / `neuromaps`; calibration figure (parametric versus spatial p on independent smooth maps)
- [ ] **M3 (month 4)** — SimNIBS `charm` head models for every subject of the primary dataset; fields simulated at recorded coil placements; parcellated field maps written
- [ ] **M4 (month 5)** — H2 answered: is dose spatially confounded with connectivity? This gates the rest of the project
- [ ] **M5 (month 7)** — TMS–EEG/fMRI outcomes extracted; target scores computed; H3 and H4 tested with leave-one-subject-out CV
- [ ] **M6 (month 9)** — Audit complete: reproduced comparisons re-tested under both null families; H1 and H5 answered
- [ ] **M7 (month 11)** — Individualized connectome arm; H6 and H7; leave-one-dataset-out transportability
- [ ] **M8 (month 12–14)** — Manuscript; derivatives (field maps, scores, surrogates) archived on OpenNeuro or Zenodo with the exact software versions

## Ethics and data-use notes

- **HCP Open Access** requires registration and acceptance of the WU-Minn Open Access Data Use Terms. Key obligations: do not attempt to re-identify participants, do not redistribute the data, and cite the HCP as specified. The **Restricted Data** tier (family structure, detailed ages, some clinical variables) needs a separate agreement and is **not used here** — avoid pulling restricted variables even incidentally.
- **OpenNeuro** datasets are released under CC0 or similar; they are de-identified and defaced, but defacing is imperfect. Do not attempt re-identification, and do not republish raw anatomical images.
- **Credentialed data rules.** This project uses none. For the wider programme: PhysioNet/MIMIC credentialed data may not be sent to third-party LLM APIs except as permitted by PhysioNet's responsible-use policy, and nothing in this repository should be mixed with such data. HCP and OpenNeuro data likewise should not be pasted into third-party services — keep analysis local.
- **AWS credentials** for the HCP S3 bucket are read from `HCP_AWS_ACCESS_KEY_ID` / `HCP_AWS_SECRET_ACCESS_KEY` environment variables. Never commit them; `.gitignore` excludes `.env` and `*credentials*`, but the real defence is not writing them to disk at all.
- **Never commit data.** `data/`, `derivatives/` and `outputs/` are gitignored, along with all imaging and mesh extensions. Commit the download commands, the dataset accession numbers and snapshot tags, and the manifest.
- **Clinical framing.** Nothing here is a validated clinical targeting tool. Any figure that ranks brain targets should be captioned as a research model, not a treatment recommendation.
