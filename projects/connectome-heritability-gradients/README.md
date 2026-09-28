# Heritability atlas of cortical functional organisation: gradients, SC–FC coupling and dynamic states under one twin model

**Pitch.** Edge-level FC heritability is well mapped, and gradient loadings, SC–FC coupling and dynamic-state occupancy have each been shown to be heritable in isolation — but nobody has estimated them *jointly* in the same HCP twins with a shared, reliability-corrected, pipeline-robust twin model, asked whether they share genetic variance region by region, or tested with proper spatial and gene-set nulls whether the *heritability topography* (rather than the feature itself) is transcriptomically distinct.

| | |
|---|---|
| **Status** | Design + starter code; no results yet |
| **Difficulty / timeline** | MSc-level for the gradient arm (6 months); PhD-year-1 for the full multi-feature atlas incl. individual tractography (9–12 months) |
| **Compute** | Gradients/dynamic FC: workstation, ~1 h for 1,000 subjects from parcellated time series. Individual SC: MRtrix3 tractography ≈ 3 CPU-h × ~450 twins/siblings (cluster, ~1.5k CPU-h). Spins/bootstraps: minutes. |
| **Package** | `src/conn_h2` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # synthetic twins, block FC, spheres - no downloads
python scripts/download_data.py --sample        # parcellation files + HCP dry-run listing + restricted-data instructions
```

```python
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from conn_h2 import gradients, features, heritability, spatial_nulls

df = pd.read_csv("data/hcp/restricted.csv")                     # Subject, Family_ID, ZygosityGT, Mother_ID, Father_ID
pairs = heritability.build_twin_pairs(df); print(pairs.summary())
fc_list = [np.load(f"data/fc/{s}_schaefer400.npy") for s in df["Subject"]]   # (400, 400) Fisher-z per subject
loads, ecc = gradients.subject_gradient_features(fc_list, n_components=3)    # (n, 400, 3), (n, 400)
cov = df[["Age_in_Yrs", "sex", "mean_rms"]].to_numpy()
h2_g1 = heritability.heritability_map(loads[:, :, 0], pairs, method="ace", covariates=cov)
h2_ecc = heritability.heritability_map(ecc, pairs, method="falconer", covariates=cov, n_boot=1000)
sc_list = [np.loadtxt(f"data/sc/{s}_schaefer400_sift2.csv", delimiter=",") for s in df["Subject"]]
coupling = np.vstack([features.sc_fc_coupling(sc, fc) for sc, fc in zip(sc_list, fc_list)])
h2_cpl = heritability.heritability_map(coupling, pairs, method="ace", covariates=cov)

spins = spatial_nulls.spin_permutations(lh_centroids, rh_centroids, n_perm=10000)  # fsLR sphere centroids
print(spatial_nulls.spatial_correlation_test(h2_g1["h2"].to_numpy(), h2_cpl["h2"].to_numpy(), spins))
expr = spatial_nulls.fetch_expression("data/parcellations/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
print(spatial_nulls.random_gene_set_null(h2_g1["h2"].to_numpy(), expr, oligodendrocyte_markers, spins=spins))
```

Repository layout:

```
src/conn_h2/             gradients · features (SC-FC coupling, dynamic states) · heritability · spatial_nulls
scripts/download_data.py HCP rest/diffusion via S3, parcellations, abagen fetch, restricted-data instructions
data/README.md           acquisition, tractography recipe, surfaces for spins, ABCD
tests/test_conn_h2.py    simulated MZ/DZ twins with known h2, block FC, synthetic spheres
```

## Background

The principal functional gradient (Margulies et al., 2016, *PNAS*) orders cortex from unimodal to transmodal regions and organises where structure and function decouple (Vázquez-Rodríguez et al., 2019, *PNAS*; Baum et al., 2020, *PNAS*). Twin designs in HCP show that FC edges (Ge et al., 2017, *PNAS*; Colclough et al., 2017), individualised network topography (Anderson et al., 2021, *PNAS*), regional SC–FC coupling (Gu et al., 2021, *Nat Commun*), dynamic-state occupancy and transition trajectories (Vidaurre et al., 2017, *PNAS*; Jun et al., 2022, *NeuroImage*) and cortical microstructure-function coupling (Valk et al., 2022, *Nat Commun*) are all heritable. Since 2024 gradient heritability itself has been mapped: subcortico-cortical gradients (*Commun Biol* 2024), cortical sensorimotor–association gradient loadings (HCP twins, h² ≈ 0.57 after modelling intra-individual variance, 2025), functional–structural *gradient coupling* (*Nat Commun* 2026, HCP + ABCD, enriched for deep-layer excitatory-neuron genes), and GWAS of gradient loadings in > 30,000 UK Biobank participants with transcriptomic alignment (Wan et al., 2025, *medRxiv*).

## The research gap

**Done (2021–2026).** Univariate heritability of (i) gradient loadings, (ii) SC–FC coupling, (iii) dynamic-state temporal features, (iv) gradient coupling; transcriptomic correlations of group-level gradients and of gradient-coupling heritability.

**Still missing — the sharpened angle of this project.**
1. **One twin model, many features, same subjects.** The four feature families were estimated in different subsets, parcellations and estimators (ACE with OpenMx/APACE/SOLAR, Falconer, mixed models). Their heritability maps have never been compared region-by-region under an identical pipeline, so claims like "coupling is more heritable in unimodal cortex than gradients are" are not currently supported.
2. **Shared vs. distinct genetic variance.** Bivariate genetic correlations between gradient loading, eccentricity, SC–FC coupling and state occupancy at the same region are unreported (the 2025 HCP study found *distinct* genetics for gradient vs. microstructure vs. geodesic distance — coupling and dynamics were not included).
3. **Reliability ceiling.** Heritability is bounded by test–retest reliability; regional reliability of gradients/coupling is heterogeneous (e.g. low in limbic/insular cortex). A reliability-corrected atlas (h²/ICC using the HCP retest subset) has not been produced, and the 2025 result that modelling intra-individual variance raises h² from 0.37 to 0.57 suggests uncorrected maps mislocate "heritable" cortex.
4. **Pipeline multiverse of h².** Gradient parameters (sparsity, kernel, alignment, number of components, parcellation) change loadings; whether the heritability *map* is stable across these choices is unknown.
5. **Transcriptomics of the heritability map with correct nulls.** Prior gene-expression comparisons used the group gradient itself or single spin tests; comparing the *h² topography* against AHBA with both spatial-autocorrelation nulls and random-gene-set nulls (Fulcher et al., 2021, *Nat Commun*; Markello & Misic, 2021, *NeuroImage*) has not been done for gradients, coupling or dynamics.
6. **Age replication.** ABCD (9–11 y, ≈ 400+ twin pairs) allows a child vs. adult comparison of the same atlas.

## Research questions / hypotheses

1. **H1.** Gradient-1 loading heritability follows the unimodal→transmodal axis (higher in sensorimotor/visual cortex), whereas SC–FC coupling h² is highest in visual/subcortical-adjacent regions (Gu 2021) — the two maps correlate only moderately (ρ < 0.4, spin p > 0.05).
2. **H2.** Reliability correction (h²/ICC) *reorders* the map: ≥ 20 % of regions change tertile, and limbic/insular regions move up.
3. **H3.** Genetic correlations between gradient loading and coupling at the same region are weak (|rG| < 0.3) for most regions but strong in heteromodal association cortex — i.e. shared genetic factors where structure and function decouple.
4. **H4.** Dynamic-state fractional occupancy is heritable (h² ≈ 0.3–0.4, replicating Vidaurre/Jun) and its genetic variance is largely independent of static gradient loadings.
5. **H5.** The heritability map of gradient 1 correlates with the first principal component of AHBA expression (and with oligodendrocyte/excitatory neuron marker sets) beyond both spin and random-gene-set nulls; the coupling-h² map aligns with a different gene set (e.g. myelination genes).
6. **H6.** The h² topography is stable across the gradient multiverse (mean pairwise Spearman ρ > 0.8 across ≥ 24 pipeline variants) but *not* across estimators without reliability correction.
7. **H7.** The adult (HCP) and child (ABCD) heritability maps correlate (ρ > 0.5, spin p < 0.05) with systematically lower child h² in association cortex.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| HCP S1200 young adult | 4 rest runs (ICA-FIX, MSMAll), diffusion (structural connectomes), retest subset; **restricted**: Family_ID, ZygosityGT, Mother/Father IDs, exact age | ~1,000 with full rest; ≈ 130–150 MZ pairs, ≈ 70–90 DZ pairs (verify from export); 45 retest | Free registration (imaging via S3) + Restricted Data application | https://db.humanconnectome.org |
| Allen Human Brain Atlas | Regional microarray expression via abagen (6 donors) | ~15,000 genes × 400 parcels | Open | https://human.brain-map.org / `abagen` |
| Schaefer / Glasser parcellations, fsLR spheres | Parcellation, spin-test coordinates | – | Open | CBIG GitHub; HCP pipelines templates; `neuromaps` |
| ABCD (optional) | Baseline rest fMRI, twin zygosity (genotype), family IDs | ≈ 400–450 twin pairs | NDA DUC (credentialed) | https://nda.nih.gov/abcd |
| ENIGMA toolbox group SC (pilot only) | Consensus SC for coupling sanity checks | – | Open | https://enigma-toolbox.readthedocs.io |

## Methods

1. **Features (per subject, Schaefer-400; sensitivity Glasser-360, Schaefer-200/1000).**
   * Gradients: `conn_h2.gradients.subject_gradient_features` — 90 % row sparsity, normalised-angle kernel, diffusion map α = 0.5, Procrustes alignment to the group template from mean FC; outputs loadings G1–G3 and eccentricity (Bethlehem et al., 2020). Cross-check against BrainSpace (`use_brainspace=True`).
   * SC–FC coupling: `features.sc_fc_coupling` (Spearman over structurally connected pairs, log SC) and `features.multilinear_coupling` (adjusted R² from SC, shortest path, communicability).
   * Dynamic states: `features.dynamic_state_features` — sliding window 60 TR/step 10, k-means K = 4–6 on pooled windows, fractional occupancy, dwell time, transition probabilities. Optional HMM (hmmlearn) replication.
2. **Covariates.** Age, sex, mean RelativeRMS motion, (for SC) streamline count; residualised before twin modelling (`heritability.residualize_covariates`).
3. **Twin models.** `build_twin_pairs` from the restricted table (genotyped zygosity preferred). Estimators: Falconer (screen), DeFries–Fulker regression, maximum-likelihood ACE with AE/CE/E nested LRTs (`ace_ml`), pair-resampled bootstrap CIs (1,000). Non-twin siblings added as an extra relatedness class in a sensitivity ACE fit (OpenMx in R, exported pair tables). For repeat-measure correction, fit the Ge et al. (2017) model on the 4 runs (or use run-halves) and compare with `disattenuated_h2` using retest ICCs.
4. **Bivariate genetics.** `genetic_correlation_falconer` as a screen for every region-pair of features; confirmation with bivariate Cholesky ACE (OpenMx) for regions passing FDR.
5. **Multiverse of h².** Grid: sparsity {0.8, 0.9, 0.95} × kernel {normalised angle, cosine, none} × components {5, 10} × alignment {Procrustes, joint} × parcellation {Schaefer-200/400, Glasser} → 72 variants; heritability map per variant; stability = pairwise Spearman ρ and rank-tertile agreement.
6. **Transcriptomics.** `spatial_nulls.fetch_expression` (abagen, bidirectional mirroring, interpolate, matched normalisation); PC1 of expression; cell-type marker sets (Lake et al. 2018 / Seidlitz et al. 2020 gene lists); `random_gene_set_null` with 10,000 random sets **and** `spin_permutations` (10,000) — significance requires both p < 0.05 after BH-FDR.
7. **Replication.** ABCD baseline: identical pipeline, genotyped twins; compare maps with spin tests.

**Feature × estimator matrix** (every cell = one 400-region map with bootstrap CIs; primary cells in bold):

| Feature (per region) | Function | Falconer | DF regression | ML-ACE (+AE/CE/E LRT) | Retest ICC ceiling |
|---|---|---|---|---|---|
| Gradient-1/2/3 loading | `gradients.subject_gradient_features` | screen | sensitivity | **primary** | HCP retest (n = 45) |
| Eccentricity (G1–G3) | `gradients.gradient_dispersion` | screen | sensitivity | **primary** | HCP retest |
| SC–FC coupling (Spearman) | `features.sc_fc_coupling` | screen | sensitivity | **primary** | retest (needs 2 diffusion sessions: HCP retest has them) |
| Multilinear coupling R² | `features.multilinear_coupling` | screen | sensitivity | secondary | as above |
| State fractional occupancy (K = 4–6) | `features.dynamic_state_features` | screen | sensitivity | **primary** (per state, not per region) | run-halves split |
| Dwell time / transition probabilities | `features.state_metrics` | screen | – | secondary | run-halves split |
| Bivariate rG (G1 × coupling, G1 × FO, ecc × coupling) | `heritability.genetic_correlation_falconer` → OpenMx Cholesky | screen | – | **confirmatory** | – |

Sample sizes per cell (HCP S1200, verify from the restricted export): ≈ 130–150 MZ pairs, ≈ 70–90 DZ pairs, plus non-twin sibling pairs in the sensitivity ACE; ABCD ≈ 400–450 twin pairs for replication. A minimum of 60 pairs per zygosity is enforced before a cell is reported.

## Evaluation & statistics

* **Estimates:** h², c², e² with 95 % bootstrap CIs per region; ACE vs AE by LRT; regions reported as heritable if the AE h² CI excludes 0 and BH-FDR q < 0.05 across 400 regions.
* **Map comparisons:** Spearman ρ with spin permutation p (parcel centroids on the fsLR sphere, hemisphere-mirrored rotations); Moran-spectral or variogram nulls (brainsmash) as a second null family; report both.
* **Reliability ceiling:** ICC(2,1) from the retest subset per region/feature; h²/ICC with CIs by delta method + bootstrap; flag regions with ICC < 0.4 as uninterpretable.
* **Bivariate:** rG with bootstrap CIs; FDR across regions; confirm with OpenMx.
* **Gene sets:** two nulls (random gene sets, spins); ensemble-based enrichment rather than standard GSEA (Fulcher 2021).
* **Multiple comparisons:** BH-FDR within each map; family-wise for the small number of a-priori map-to-map comparisons (H1, H5, H7).
* **Leakage/dependence:** pairs are the unit of resampling; siblings are never used as independent observations; group gradient template is computed from *all* subjects (unsupervised, no leakage of zygosity).
* **Pre-registration** of H1–H7, parcellation and grid on OSF.

## Publishable angle

**Headline:** "A single reliability-corrected twin model shows that the heritable topographies of functional gradients, structure–function coupling and dynamic-state occupancy are largely *distinct*, converge only in heteromodal association cortex, and map onto different transcriptomic programmes — with the gradient heritability map tracking the expression PC1 beyond spatial and gene-set nulls." A secondary methodological result — that uncorrected h² maps are unstable across gradient pipelines while reliability-corrected maps are not — is itself citable.

**Venues:** *Nature Communications* (multi-feature atlas + genetics), *Imaging Neuroscience*, *Cerebral Cortex*, *Network Neuroscience* (methodological multiverse arm); OHBM abstract early.

**Follow-ups:** GWAS-based SNP heritability of the same features in UK Biobank (LDSC, comparison with twin h²); developmental change of the atlas in ABCD follow-ups; using the heritable regions as priors for polygenic-risk-to-brain mapping.

## Risks, confounds & mitigations

| Risk | Mitigation |
|---|---|
| Restricted-data approval delays | Pipeline is testable on synthetic twins (`tests/`) and on open PTN data; apply first week |
| Low power for c² and rG with ~150 MZ / ~80 DZ pairs | Report CIs, use AE models primarily, pool siblings in sensitivity ACE, replicate in ABCD |
| Head-motion heritability inflates FC h² | Motion as covariate; scrubbing sensitivity; report h² of motion itself |
| Gradient sign/order flips across subjects | Procrustes alignment to a group template; eccentricity is alignment-invariant as a second feature |
| Tractography cost and SC false positives | SIFT2 weighting, consensus thresholding (Betzel 2019); pilot with group SC; budget cluster hours early |
| AHBA covers mostly the left hemisphere; only 6 donors | Bidirectional mirroring; left-hemisphere-only sensitivity; donor leave-one-out stability |
| Gene-set enrichment false positives | Random-gene-set and spin nulls both required; no standard GSEA |
| Multiverse → cherry-picking | Full grid pre-registered; primary pipeline fixed a priori (BrainSpace defaults) |

## Milestones

- [ ] Restricted Data application; OSF pre-registration
- [ ] Download & parcellate rest fMRI for all twins/siblings; group template; per-subject gradients & eccentricity
- [ ] Dynamic-state features (k-means; HMM sensitivity)
- [ ] Tractography for twin subset (MRtrix3), SC–FC coupling and multilinear coupling
- [ ] Twin pairs; Falconer/DF/ACE maps with bootstrap CIs; retest ICC ceilings
- [ ] Multiverse stability of h² maps
- [ ] Bivariate genetic correlations; OpenMx confirmation
- [ ] abagen expression; spin + random-gene-set nulls; cell-type marker sets
- [ ] ABCD replication (optional)
- [ ] Manuscript, atlas release (parcel-level h² tables only, no subject data)

## Ethics / data-use notes

* Family structure, zygosity and exact age are HCP **restricted** data: keep them on approved storage, never in the repo, never in figures (no family IDs, no pair-level plots that could identify twins), never sent to third-party services or LLM APIs.
* Released outputs are parcel-level summary statistics (h², CIs, correlations) only.
* ABCD requires an NDA DUC; twin zygosity from genotypes is especially sensitive.
* AHBA data are open; cite Hawrylycz et al. (2012) and the abagen version.
* Heritability describes population variance in a specific cohort, not individual determinism; write results accordingly.
