# spatial-null-replication-audit

**Do published brain-map correlations survive the null model? A registered re-analysis of ~50 published cortical map-to-map associations under six spatial null families, with a per-claim null-robustness index and a smoothness-matched false-positive calibration that explains which claims flip and why.**

## Status / difficulty / timeline / compute

- Status: design + starter code (claim registry, from-scratch spin / Moran-spectral / variogram-matched surrogates on parcellated maps, audit driver, calibrated-FPR simulation, tests). No data shipped.
- Difficulty: MSc-level (methods audit; all maps are public summary maps, no subject-level processing). 5-8 months including the systematic claim search.
- Compute: a laptop. 50 claims x 6 nulls x 10,000 permutations on 100-1000-parcel maps runs in hours on one CPU; eigenstrapping on dense surfaces is the only step that benefits from 16+ GB RAM.

## Background

Comparing two brain maps (a cortical thickness map with a gene-expression map, a functional gradient with a receptor density map, a lesion-symptom map with a network map) is one of the most common analyses in systems neuroscience. Because both maps are spatially smooth, the ordinary parametric p-value for their correlation is meaningless, and a family of *spatial null models* has been built to fix this: spherical rotations of the surface ("spin test", Alexander-Bloch et al., 2018, NeuroImage; parcellated variants in Váša et al., 2018, Cerebral Cortex), variogram-matched surrogates (BrainSMASH, Burt et al., 2020, NeuroImage), Moran spectral randomisation (Wagner & Dray, 2015, Methods Ecol Evol; used in BrainSpace, Vos de Wael et al., 2020, Commun Biol), and, most recently, eigenstrapping (random rotation of geometric eigenmodes; Koussis et al., 2025, Imaging Neuroscience). Markello & Misic (2021, NeuroImage) benchmarked ten nulls and found that they disagree, sometimes by orders of magnitude in p, and that non-spatial nulls are badly anti-conservative. neuromaps (Markello et al., 2022, Nat Methods) standardised access to the maps and to the nulls.

Two 2024-2025 findings make an *audit of the published literature* timely. First, the eigenstrapping paper reports, in controlled simulations, that both the spin test and BrainSMASH surrogates fail to control the false-positive rate for strongly autocorrelated maps (BrainSMASH being worst), while eigenstrapping does. Second, "The effect of spherical projection on spin tests for brain maps" (Imaging Neuroscience, 2025) shows that spherical projection distorts inter-vertex distances, that spin-test false-positive rates rise in proportion to how much the rotated map's autocorrelation deviates from the original's, and that discarding deviant rotations restores control at the cost of power. So the two most cited nulls, used in hundreds of papers since 2018, are now known to be miscalibrated in a smoothness-dependent way. Nobody has gone back to the published claims to ask how many of them depended on that miscalibration.

## The research gap

What has been done:

- Method benchmarks on simulated or a handful of real maps: Markello & Misic (2021); Burt et al. (2020); Koussis et al. (2025); the spherical-projection paper (2025). All compare nulls to each other, not to the literature.
- Imaging-transcriptomics-specific audits: Fulcher, Arnatkeviciute & Fornito (2021, Nat Commun) showed gene-category enrichment results are dominated by spatial autocorrelation and within-category co-expression; Wei et al. (2022, Hum Brain Mapp) reported that among transcriptomic-neuroimaging associations passing a spatial null, a large share also passes for random gene sets, and only a small minority passes both spatial and gene-specificity nulls. These concern gene-set inference, not the general map-to-map correlation literature.
- Null-model reviews: Váša & Mišić (2022, Nat Rev Neurosci) catalogue nulls for network neuroscience and recommend reporting several.

What is specifically missing (our angle):

1. **A registered replication of published map-to-map claims** (not gene-set enrichments) under a *matrix* of nulls, including the two post-2024 corrections (eigenstrapping; projection-corrected spin). Output: for each claim, the reported p, the p under each null, and a null-robustness index.
2. **An explanation of flips**: a per-claim smoothness-matched calibration experiment (simulate independent maps with the same autocorrelation length and parcellation as the claim's maps, measure each null's actual FPR) so that a flip can be attributed to null miscalibration rather than to a marginal effect.
3. **Survey statistics of practice** in the same corpus: which null, how many permutations, parcellation, hemisphere handling, and whether the surrogate's autocorrelation matched the original (the diagnostic recommended by the 2025 spin paper).
4. **A reporting checklist and a re-usable claim registry** (`claims.csv` schema) so that future papers can be audited the same way.

## Research questions / hypotheses

1. **RQ1 (replication rate).** What fraction of published claims (reported p < 0.05 under the authors' null) remain p < 0.05 under (a) the same null re-run, (b) eigenstrapping, (c) projection-corrected spin, (d) Moran spectral randomisation, (e) variogram-matched surrogates, (f) all of b-e? H1: >= 90% replicate under (a); 60-75% under all of b-e; the loss concentrates in claims whose maps have long autocorrelation length (lambda above the corpus median) and whose reported p is in [0.001, 0.05].
2. **RQ2 (which null was used matters).** H2: claims originally tested with BrainSMASH flip more often under eigenstrapping than claims originally tested with the spin test, consistent with the simulation ranking in Koussis et al. (2025).
3. **RQ3 (calibration explains flips).** H3: for flipped claims, the smoothness-matched FPR of the original null exceeds 0.10; for non-flipped claims it is < 0.07.
4. **RQ4 (parcellation resolution).** H4: at fixed maps, p-values under every null become more conservative from 100 to 1000 parcels for spin-type nulls (fewer effective degrees of freedom captured) but not for spectral nulls; the null-robustness index is more stable for claims reported at >= 400 parcels.
5. **RQ5 (practice).** H5: fewer than 20% of papers report whether surrogate autocorrelation matched the original; fewer than half report the number of permutations and the hemisphere handling.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| neuromaps annotations (Markello et al., 2022) | the majority of published maps (PET receptor densities, gradients, myelin, evolutionary expansion, etc.) in fsaverage/fsLR/MNI | ~100 maps, < 1 GB | Open (OSF) | https://netneurolab.github.io/neuromaps/ |
| NeuroVault | unthresholded group maps used in specific claims | per claim, MB | Open REST API | https://neurovault.org/api/ |
| Allen Human Brain Atlas via abagen | regional gene-expression maps for the subset of claims that involve transcription | ~4 GB raw microarray | Open | https://abagen.readthedocs.io |
| ENIGMA Toolbox summary maps (Larivière et al., 2021, Nat Methods) | case-control cortical thickness/surface-area maps used in disorder-map claims | small | Open | https://enigma-toolbox.readthedocs.io |
| HCP S1200 group-average (myelin, thickness, curvature) | frequently used maps | ~1 GB | Free registration (HCP Open Access terms) | https://db.humanconnectome.org |
| Schaefer / Desikan-Killiany / Glasser parcellations, fsaverage & fsLR spheres | parcel centroids and spherical coordinates for spin tests | small | Open (CBIG GitHub; neuromaps) | https://github.com/ThomasYeoLab/CBIG |
| Claims corpus (built here) | 50 published map-to-map correlations, 2018-2025, with both maps publicly available | `data/claims/claims.csv` | Built from PubMed/E-utilities search + hand curation | https://eutils.ncbi.nlm.nih.gov |

Claim selection rule (pre-registered): PubMed/Europe PMC search for ("spin test" OR "spatial permutation" OR "BrainSMASH" OR "spatial autocorrelation-preserving") AND ("cortical map" OR "brain map" OR "gradient" OR "receptor"), 2018-2025; include a claim when (i) both maps are public in a standard space, (ii) a scalar correlation between two parcellated or vertex-wise cortical maps is the headline test, and (iii) the parcellation and null are stated. Sample the first 50 eligible claims ordered by citation count within year, stratified by null used (spin / BrainSMASH / other).

## Methods

Pipeline (modules in `src/spatial_null_audit/`):

1. **Claim registry** (`claims.py`). One row per claim: DOI, year, map A/B identifiers (neuromaps annotation id, NeuroVault image id or file), space, parcellation and resolution, hemisphere handling, reported r and p, null used, number of permutations, whether the surrogate-autocorrelation check was reported. `load_claims` validates the schema; `summarize_claims` yields the practice-survey statistics.
2. **Map preparation.** Fetch both maps with neuromaps, transform to the claim's space/resolution (`neuromaps.transforms`), parcellate with the claim's atlas (`neuromaps.parcellate.Parcellater`), drop the medial wall. Store parcel centroids on the sphere and geodesic (or Euclidean-on-sphere) distance matrices per parcellation.
3. **Null families** (`nulls.py`, all implemented from scratch on parcellated maps so the audit does not depend on any single package; cross-checked against neuromaps/BrainSMASH/eigenstrapping outputs on 5 claims):
   - `naive_permutation` (the anti-conservative reference),
   - `spin_surrogates`: random rotations of parcel centroids on the sphere, hemisphere-mirrored (Alexander-Bloch 2018), with either nearest-neighbour reassignment (duplicates allowed) or one-to-one greedy matching (Váša 2018);
   - `spin_surrogates(..., sa_tolerance=...)`: the projection-corrected variant that discards rotations whose surrogate autocorrelation deviates from the original's by more than a tolerance (the fix proposed in the 2025 Imaging Neuroscience paper);
   - `moran_surrogates`: Moran spectral randomisation (Wagner & Dray 2015 "singleton" scheme);
   - `variogram_surrogates`: variogram-matched surrogates (Burt 2020 scheme: permute, kernel-smooth at the bandwidth that best matches the empirical variogram, add fitted noise, rank-resample to the original values);
   - eigenstrapping via the `eigenstrapping` package on the dense surface (optional dependency; wrapper only).
4. **Audit driver** (`audit.py`). `audit_pair` computes the observed Pearson and Spearman correlations and the permutation p-value under each null (with the +1 correction), the surrogate-vs-original autocorrelation deviation, and the null-robustness index (fraction of null families with p < 0.05). `calibrated_fpr` simulates pairs of *independent* Gaussian random fields with the claim's autocorrelation length on the claim's parcellation and measures each null's empirical FPR.
5. **Resolution sweep.** Re-run each claim at 100/200/400/1000 Schaefer parcels where the maps allow.
6. **Meta-analysis of flips.** Logistic regression of flip (yes/no) on log autocorrelation length, original null family, parcel count, reported p bin, year; cluster-robust SEs by paper.

Tools: numpy/scipy (all nulls), neuromaps + nibabel (maps, transforms, parcellation), abagen (AHBA), eigenstrapping (optional), statsmodels (meta-regression), requests (NeuroVault, E-utilities).

## Evaluation & statistics

- Every null uses the same 10,000 surrogates per claim and the same seed policy (seed = hash of claim id) for reproducibility.
- p-values: two-sided, (b + 1)/(n_perm + 1). Correlations: Pearson as reported, Spearman as sensitivity.
- Replication is judged at the claim's own alpha; we also report the continuous p under each null and the robustness index.
- Calibration experiments: 1,000 simulated independent map pairs per (parcellation, lambda) cell; FPR with Wilson CIs; a null is "miscalibrated" for a cell when the CI excludes 0.05.
- Multiple comparisons: the 50 claims are not corrected (each is its own registered replication); the meta-regression uses Holm across its 5 predictors.
- Positive controls: 10 synthetic claims where map B = smoothed map A + noise (true association) must replicate under every null. Negative controls: 10 pairs of independent GRFs with matched smoothness must not.
- Autocorrelation diagnostics: exponential-variogram length lambda and Moran's I for every original and surrogate map; the mean absolute deviation of surrogate lambda from original lambda is reported per null and per claim (the 2025 spin paper's diagnostic).

## Publishable angle

Headline: "Of 50 published cortical map-to-map correlations, X replicate under their original null but only Y under the 2025 corrected nulls; flips are predicted by map smoothness and by which null was originally used, and are explained by the original null's measured false-positive rate on smoothness-matched simulations." Plus a practice survey and a checklist.

Target venues: Imaging Neuroscience (where the null-model debate is happening); NeuroImage; Nature Communications (meta-research); Network Neuroscience. OHBM abstract for the corpus.

Follow-ups: extend to subcortical and volumetric claims (where spins do not apply and MSR/eigenstrapping are the only options); a living registry accepting community-submitted claims; gene-set claims combined with the gene-specificity null of Fulcher et al. (2021).

## Related project

`imaging-transcriptomics-nulls` in this repository studies nulls for AHBA gene-expression associations specifically; this project audits the general map-to-map literature (receptors, gradients, disorder maps, evolutionary maps) and treats transcriptomic claims as one stratum. The two projects share no code.

## Risks, confounds & mitigations

- **Maps not exactly as used by the authors** (different release, smoothing). Mitigation: reproduce the reported r first; a claim whose r we cannot reproduce within +/-0.05 is flagged "map mismatch" and analysed separately.
- **Selection of claims biased toward high-profile papers.** Mitigation: stratified sampling by null type and year; sensitivity analysis on a random (non-citation-ordered) subsample.
- **Our from-scratch nulls could be wrong.** Mitigation: unit tests on synthetic fields; cross-check p-values against neuromaps/BrainSMASH/eigenstrapping reference implementations on 5 claims before running the audit.
- **Spin tests on MNI-centroid projections are approximate.** Mitigation: use true sphere coordinates from fsaverage/fsLR surfaces for all surface-based claims; MNI-centroid projection only for volumetric atlases, flagged as such.
- **Flip could reflect low power rather than miscalibration.** Mitigation: RQ3 calibration experiment distinguishes the two; report effect sizes with surrogate-based CIs.

## Milestones

- [ ] Pre-register claim selection rule and hypotheses (OSF).
- [ ] Build the claims corpus (PubMed search -> screening -> 50 claims with public maps).
- [ ] Fetch and parcellate maps; reproduce reported r for each claim.
- [ ] Validate nulls against reference implementations on 5 claims.
- [ ] Run the six-null audit at the original resolution; then the resolution sweep.
- [ ] Calibrated-FPR simulations per claim; flip meta-regression.
- [ ] Practice survey table; checklist; manuscript.

## Ethics / data-use notes

- All maps are public group-level summaries; no individual-level data.
- HCP group-average maps require accepting the HCP Open Access Data Use Terms (no redistribution of the raw files in this repo).
- This is a re-analysis of published claims: report results neutrally, by claim id, with the original authors' methods described accurately; share the full per-claim table so authors can check it.
- `data/` and `outputs/` are git-ignored; never commit maps.
