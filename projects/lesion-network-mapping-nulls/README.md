# lesion-network-mapping-nulls — Which null model should lesion network mapping believe? Anatomically realistic nulls on open stroke data

A head-to-head, simulation-anchored comparison of the null models now in dispute for lesion network mapping (symptom-label permutation, random synthetic lesions, shuffled lesion locations, territory- and volume-matched resampled lesions, connectome bootstrap, spatial-autocorrelation surrogates), run entirely on open lesion cohorts (ATLAS v2.0, Aphasia Recovery Cohort) and open normative connectomes, with a released per-cohort "lesion-prior bias atlas" and a leakage score for any published map.

## Status / difficulty / timeline / compute

- Status: design + starter code (lesion sampling and matching, parcel-level functional/structural lesion-network maps, six null families, Moran spectral randomization, a ground-truth simulation engine that returns type-I error and power).
- Difficulty: MSc-level methods project, 6-9 months. Strong PhD side-chapter if extended to multivariate prediction.
- Compute: CPU only. Voxel-to-parcel functional connectomes from 100 HCP-YA subjects need ~200 GB of downloads and ~2 CPU-days once; each lesion network map is then a matrix-row average (milliseconds). The full null multiverse (6 families × 10,000 null lesion sets × 3 cohorts × 2 connectomes) is ~2-4 CPU-days. Storage ~300 GB.

## Background

Lesion network mapping (LNM) seeds each patient's lesion in a normative connectome and asks which network the lesions of patients with a given symptom share (Boes et al., 2015, Brain; Fox, 2018, NEJM). The method has been applied to hundreds of symptoms. Its inference problem is the same one that voxel-based lesion-symptom mapping faced a decade earlier (Mah et al., 2014, Brain; Sperber and Karnath, 2017, Human Brain Mapping): lesions are not randomly placed. Stroke lesions follow arterial territories, so a set of lesions produces a structured network map even if no symptom was measured, and the "lesion prior" leaks into every symptom map (Salvalaggio et al., 2020, Brain, who found structural disconnection predicted deficits better than functional LNM; Bowren et al., 2022, Brain; Sperber, Griffis and Kasties, 2022, Brain Structure and Function).

In 2025-2026 the debate reached the top journals. A Nature Neuroscience paper ("Investigating the methodological foundation of lesion network mapping", 2025) reported that 70 of 78 published LNM maps failed a liberal significance criterion against a generative null of random synthetic lesions, with similar outcomes when lesion locations were shuffled. A 2026 Nature Neuroscience piece ("Null models for lesion network mapping") and a multicentre medRxiv study of 2,950 stroke patients ("...anatomical specificity in lesion network mapping: permutation-based inference...", 2026) argue instead that permuting symptom labels is the appropriate null, that it yields distinct and biologically plausible networks for different cognitive domains, and that it controls type-I error across 10,000 simulated null studies. A bioRxiv 2026 report ("Spatial bias in lesion network mapping is connectome-independent") adds that spatial bias maps built from 4,000,000 random permutations are cohort-specific and do not depend on which connectome is used. Metric choices for connectome-based lesion-symptom mapping were compared in stroke in Brain Communications (2024, 6(5): fcae313).

## The research gap

What has been done: each side has demonstrated its preferred null on its own cohort with its own definition of "valid". Random-lesion nulls test whether a map differs from what arbitrary lesions produce (a test of the connectome, not of the symptom); label permutation tests whether the map is specific to the symptom given this cohort's lesions (which conditions on, and therefore cannot detect, confounds carried by lesion location and volume). Nobody has evaluated the families side by side with (a) known ground truth, (b) realistic lesion anatomy, and (c) fully open data that others can re-run.

What is specifically missing:

1. No simulation study in which symptoms are generated from a known network under realistic, territory-structured lesion distributions and the type-I error, power and anatomical specificity of every null family are measured on the same cohorts. Existing simulations either use spheres (unrealistically easy) or permute labels within a real cohort (conditional on lesions by construction).
2. No intermediate, anatomically realistic null: resampling real lesions from a large pool (ATLAS v2.0) matched on volume, hemisphere and arterial territory, which preserves the lesion prior's shape without conditioning on the exact cohort.
3. No "prior leakage" statistic for published maps: the fraction of an LNM map's variance explained by the cohort's own lesion-prior map (the expected map under matched resampling).
4. No comparison of functional LNM vs structural disconnection mapping under identical nulls on the same open cohort with behaviour (Aphasia Recovery Cohort: 228 chronic left-hemisphere strokes with WAB-R scores).
5. No released bias atlases per cohort/connectome, although the 2026 preprint shows they are cohort-specific and therefore must be recomputed per study.

## Research questions / hypotheses

1. RQ1 (validity under ground truth). With symptom = β·(lesion-to-network-G connectivity) + γ·log(volume) + noise, what are the empirical type-I error (β = 0) and power (β > 0) of nulls N1-N6 at α = 0.05 (FWER by max statistic) for n = 50, 100, 228? H1: label permutation (N1) holds nominal error when γ = 0 but exceeds it when γ ≠ 0 and volume is not modelled; random spheres (N2) inflate false positives everywhere the connectome has hubs; territory/volume-matched resampling (N4) has near-nominal error in both cases at a modest power cost.
2. RQ2 (prior leakage in real maps). For WAB-R aphasia quotient, naming, fluency and comprehension in ARC, what fraction of each LNM map's variance is explained by the ARC lesion-prior map? H2: > 50% for classic sensitivity-style maps, < 30% for volume- and territory-adjusted regression maps.
3. RQ3 (bias-atlas stability). Do lesion-prior bias maps agree across connectomes (HCP-YA functional, HCP-842 structural) within a cohort and across cohorts (ATLAS sites, ARC)? H3: r > 0.8 across connectomes, r < 0.5 across cohorts, replicating the connectome-independence claim on open data.
4. RQ4 (structural vs functional). Under identical N4 nulls, structural disconnection maps have higher anatomical specificity (precision of significant parcels vs the truth in simulations; smaller prior leakage in ARC) than functional LNM. H4: specificity gain ≥ 0.15 in precision.
5. RQ5 (convergence). Adding lesion volume and territory covariates to a regression LNM makes N1 and N4 inferences converge (Jaccard of significant parcels > 0.7).

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| ATLAS v2.0 (Liew et al., 2022, Scientific Data) | Manually segmented chronic stroke lesion masks in MNI-152, site metadata; used as the lesion pool for matched resampling and as a behaviour-free cohort for bias atlases | 1,271 T1w scans; 655 with public masks (training split) | Free; INDI/NITRC Data Use Agreement; encrypted download | https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html |
| Aphasia Recovery Cohort, ARC (Gibson et al., 2024, Scientific Data) | T1w/T2w, expert lesion masks, WAB-R and other language scores; the behaviour cohort | 228 chronic left-hemisphere stroke | Open on OpenNeuro (accession ds004884 at time of writing; verify by searching "Aphasia Recovery Cohort") | https://openneuro.org/ |
| SOOP, Stroke Outcome Optimization Project (Absher et al., 2024, Scientific Data) | Acute clinical MRI (DWI/FLAIR) with lesion masks; optional acute-cohort replication | ~1,700 acute ischemic strokes | Imaging open on OpenNeuro (ds004889); behaviour on request to the authors | https://openneuro.org/datasets/ds004889 |
| HCP-YA S1200 (Van Essen et al., 2013, NeuroImage) | 100 unrelated subjects' FIX-cleaned rfMRI to build voxel-to-parcel functional connectomes | 100 subjects × 4 runs | Free registration; open-access data-use terms; AWS S3 or ConnectomeDB | https://db.humanconnectome.org/ |
| GSP 1000 (Holmes et al., 2015, Scientific Data) | Optional second functional connectome | 1,570 subjects | Free registration on Harvard Dataverse (DUA) | https://dataverse.harvard.edu/dataverse/GSP |
| HCP-842 tractography template (Yeh et al., 2018, NeuroImage) | Population-averaged structural connectome for disconnection maps (Lesion Quantification Toolkit, Griffis et al., 2021, NeuroImage: Clinical) | 1 template | Open download | https://brain.labsolver.org/hcp_template.html |
| Arterial territory atlas (Liu et al., 2023, Scientific Data) | Digital 3D MRI arterial territories (ACA/MCA/PCA, deep) used for territory matching | 1 atlas | Open (with the paper / NITRC) | https://www.nitrc.org/ (search "arterial territories atlas") |

## Methods

1. Lesion features (`lnm_nulls.lesion_sampling`). Resample masks to MNI 2 mm; volume, hemisphere, arterial-territory composition (fraction of voxels per territory), centroid. ARC and SOOP lesions registered to MNI with ANTs (SyN with lesion-mask cost-function masking); registration QC by visual check and Dice against a template ventricle mask.
2. Connectomes. Functional: voxel (2 mm GM) × parcel (Schaefer-400 + Tian S2) Fisher-z connectivity averaged over 100 HCP-YA subjects; structural: parcel-pair streamline counts of HCP-842 and per-voxel tract visitation for disconnection (LQT logic re-implemented at parcel level in `lnm_nulls.lnm.disconnection_map`).
3. Maps (`lnm_nulls.lnm`). Functional LNM = mean voxel-to-parcel connectivity over lesion voxels; classic sensitivity/specificity thresholding; regression LNM = per-parcel association between map value and symptom with lesion volume (log) and territory fractions as covariates; multivariate ridge prediction as a secondary outcome.
4. Null families (`lnm_nulls.nulls`): N1 symptom-label permutation (max-statistic FWER); N2 random spheres, volume-matched, inside the brain mask; N3 shuffled lesion locations (rigid translation of real lesions within hemisphere); N4 anatomically matched resampling from the ATLAS pool (volume ±20%, same hemisphere, same dominant territory); N5 connectome bootstrap (resample connectome subjects) for map uncertainty; N6 Moran spectral randomization of parcel maps (Wagner and Dray, 2015, Methods in Ecology and Evolution) for map-to-map correlations, cross-checked with BrainSMASH variogram surrogates (Burt et al., 2020, NeuroImage) and spin tests (Markello and Misic, 2021, NeuroImage).
5. Simulation engine (`lnm_nulls.simulation`). Toy and real-geometry versions: choose ground-truth network G; generate symptoms from lesion-to-G connectivity plus a volume term; run each null; record FPR, power, precision/recall of significant parcels; 500 studies per condition.
6. Bias atlases and leakage. Mean and SD of N4 null maps per cohort and connectome; prior-leakage R² of observed maps on the bias map; released as NIfTI/CSV with the code.

Libraries: numpy, scipy, pandas, statsmodels, scikit-learn; nibabel, nilearn; ANTsPy for registration; brainsmash, neuromaps (spin tests); optional Lesion Quantification Toolkit (MATLAB) for cross-checking disconnection maps.

## Evaluation and statistics

- Simulation outcomes: empirical FPR with Wilson 95% CI over 500 studies; power curves vs β and n; anatomical specificity as precision/recall/Dice of significant parcels vs G.
- Real data: FWER via max-statistic within each null family; FDR (q = 0.05) reported as a secondary; effect sizes as partial R² per parcel.
- Map comparisons: Spearman correlations with N6 (Moran) and spin-test p-values; 1,000 surrogates.
- Leakage prevention: all thresholds and matching tolerances fixed from simulations before touching ARC behaviour; ARC analyses pre-registered (OSF); no tuning on real symptoms.
- Multiple comparisons: five pre-registered contrasts (one per RQ) with Holm; parcel-level inference is already FWER-controlled.
- Robustness: repeat with Schaefer-1000, with GSP connectome, with 1 mm masks for a subset, and with lesion-volume-stratified subsamples.

## Publishable angle

Headline: "On open data with known ground truth, the null model changes which lesion networks survive: symptom permutation misses volume-driven false positives, random-lesion nulls overcall hubs, and anatomically matched resampling controls both; the cohort's own lesion prior explains up to X% of published-style maps." The paper ships bias atlases for ATLAS and ARC and a `leakage_r2()` one-liner that any LNM paper can report.

Target venues: Brain Communications; NeuroImage: Clinical; Human Brain Mapping; Brain (if the ARC structural-vs-functional result is strong).

Follow-ups: (i) apply the same nulls to TMS/DBS target-connectivity maps (related project in this repository: `tms-target-connectivity`); (ii) extend to multivariate LNM predictors (Bowren et al., 2022); (iii) acute cohort (SOOP) with NIHSS once behaviour is obtained; (iv) contribute nulls to neuromaps/BrainSMASH ecosystems.

## Risks, confounds and mitigations

- ARC is left-hemisphere and aphasia-only, so its lesion prior is narrow; ATLAS provides multi-site, bilateral lesions for the pool, and SOOP an acute cohort. Report generalization limits explicitly.
- Registration error of lesion masks into MNI space biases territory assignment; use cost-function masking, QC, and territory fractions rather than hard labels.
- Connectome choice (FIX-cleaned vs not; global signal regression) changes maps; run both and report the bias-atlas correlation across choices.
- Label permutation requires exchangeability; when symptom distributions differ by site, permute within site.
- The ATLAS test-split masks are hidden; only the 655 public masks form the pool. Volume matching with ±20% tolerance may fail for very large lesions; fall back to N3 for those and flag them.
- Debate sensitivity: cite both positions fairly and pre-register the criteria for "valid" (nominal FPR at known truth) so the outcome cannot be read as partisan.

## Milestones

- [ ] Sign the ATLAS DUA; download and decrypt ATLAS v2.0 masks; download ARC (OpenNeuro) and HCP-YA 100 unrelated subjects.
- [ ] Build voxel-to-parcel functional connectome and HCP-842 parcel disconnection tables.
- [ ] Lesion feature tables (volume, hemisphere, territory) for ATLAS and ARC; registration QC.
- [ ] Simulation study (RQ1) on toy geometry, then on real ATLAS lesions and the real connectome.
- [ ] Bias atlases and leakage R² (RQ2, RQ3); pre-register ARC analysis.
- [ ] ARC functional vs structural mapping under all nulls (RQ4, RQ5).
- [ ] Release atlases + code; manuscript.

## Ethics / data-use notes

- ATLAS v2.0 requires the INDI DUA; the decrypted data must not be redistributed. ARC and SOOP imaging are open (OpenNeuro) but derived per-patient lesion maps combined with behaviour should be shared only as group-level products. HCP-YA open-access terms forbid attempting re-identification and require the HCP acknowledgement.
- No credentials or data are committed; `data/` and `outputs/` are ignored by git.
- No third-party LLM services are used anywhere in the pipeline.
