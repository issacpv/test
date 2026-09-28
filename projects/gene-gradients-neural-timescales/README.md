# gene-gradients-neural-timescales

**Which genes, and which cell-type compositions, predict single-neuron timescales and tuning along the mouse visual hierarchy? A composition-vs-within-type decomposition of gene-gradient predictions using Allen ISH, ABC-Atlas MERFISH and three independent Neuropixels surveys, with spatial and gene-ensemble nulls.**

## Status / difficulty / timeline / compute

- Status: design + starter code (timescale and tuning estimators, regional gene-map construction, PLS/ridge prediction with nested CV, composition decomposition, spatial and gene-ensemble nulls). No data is shipped.
- Difficulty: MSc-level to early PhD. Data are open and well curated (AllenSDK, ONE/IBL, ABC Atlas); the difficulty is in unbiased timescale estimation and in nulls that respect the hierarchy structure.
- Timeline: 6-9 months (2 months unit-level metrics on Visual Coding + Visual Behavior + IBL, 1 month gene maps from ISH/MERFISH, 2 months modelling and nulls, 1-2 months layer-resolved extension and writing).
- Compute: one workstation, 64 GB RAM, ~2 TB disk for the NWB files (Visual Coding Neuropixels ~ 58 sessions; Visual Behavior Neuropixels ~ 150 sessions; IBL brain-wide map ~ 700 insertions, streamed via ONE). No GPU needed.

## Background

Intrinsic neural timescales (the decay of spike-count autocorrelation) increase along cortical hierarchies in primates (Murray et al., 2014 Nature Neuroscience) and in mice (Rudelt et al., 2024 PLOS Computational Biology on the Allen Visual Coding Neuropixels survey; Siegle et al., 2021 Nature for the survey itself; "Hierarchical gradients of multiple timescales in the mammalian forebrain", PNAS 2024, shows multiple timescales rising along the hierarchy across species). In human cortex, Gao et al. (2020 eLife) linked ECoG timescales to T1w/T2w and to AHBA gene expression (ion-channel and synaptic genes). In macaque and human, Burt et al. (2018 Nature Neuroscience) described the transcriptomic hierarchy gradient. The Allen Mouse Brain Atlas ISH (Lein et al., 2007 Nature) and, since 2023, the Allen Brain Cell (ABC) Atlas MERFISH data (Yao et al., 2023 Nature; Zhang et al., 2023 Nature) give cell-resolved expression across the whole mouse brain, and Harris et al. (2019 Nature) provide an anatomical hierarchy score for cortical and thalamic areas.

Timescale estimation itself is fragile: exponential fits to spike-count autocorrelations are biased by finite windows and by multiple timescales (Zeraati, Engel & Levina, 2022 Nature Computational Science propose an approximate-Bayesian estimator; Spitmaan et al., 2020 PNAS show multiple timescales in single neurons).

## The research gap

What has been done (2023-2026):

- Shi, Zeraati, Levina & Engel (2025 bioRxiv, "Brain-wide organization of intrinsic timescales at single-neuron resolution") estimated timescales for >10,000 IBL neurons across 223 areas and showed with ridge regression that spatial gene-expression patterns (Allen ISH) predict timescale variation at a resolution finer than area boundaries. This is the closest work; it uses one dataset (IBL), one expression modality (ISH), one target (intrinsic timescale), and it does not separate cell-type composition from within-type expression, nor does it report gene-ensemble nulls for specific gene families.
- Rudelt et al. (2024 PLOS Comput Biol) and the PNAS 2024 multiple-timescales paper describe the hierarchy of timescales in mouse visual cortex without a molecular side.
- Gao et al. (2020 eLife) did genes-vs-timescales in human with AHBA (bulk microarray; six donors) and no cell-type resolution.
- Wei et al. (2023 Nature Communications) associated in-vitro, in-vivo and in-silico cell classes in mouse V1, connecting Neuropixels waveform/response classes to transcriptomic types, but not to regional gene gradients.

What is specifically missing (our angle):

1. **Targets beyond the intrinsic timescale**: response latency, adaptation index, temporal-frequency preference, orientation/direction selectivity, receptive-field size and the *information* timescale (Rudelt et al.) as a panel of tuning properties, each predicted from gene gradients. Do the same genes predict all of them (one hierarchy axis), or do specific gene families map onto specific properties?
2. **Composition vs within-type**: with MERFISH subclass proportions per area (and per layer), decompose each gene-gradient prediction into a part carried by cell-type composition (e.g., fraction of Pvalb vs Sst interneurons, L5 ET vs IT neurons) and a residual within-type expression part. Test whether, once composition is known, gene expression adds any predictive value.
3. **Layer resolution**: Visual Coding Neuropixels units have CSD-based layer assignments; MERFISH cells carry layer labels. A layer x area design tests whether the gene-timescale link holds within layers (where composition changes less) or is an across-layer composition effect.
4. **Triple replication**: the same model fit on Visual Coding units is tested on Visual Behavior Neuropixels (different mice, task) and on IBL visual-area units (different labs), with ISH- and MERFISH-derived gene maps as two expression modalities.
5. **Rigorous nulls**: 3-D variogram surrogates over area centroids, hierarchy-preserving permutations (permute areas within hierarchy strata), and gene-ensemble nulls matched on expression level and spatial autocorrelation for the candidate families (HCN, KCN*, SCN*, GRIN, GABR, synaptic vesicle genes).

## Research questions / hypotheses

1. **RQ1.** Which gene families predict which tuning property? H1: a single PLS component (the hierarchy/transcriptomic gradient) explains most of the between-area variance in intrinsic timescale and latency, but temporal-frequency preference and adaptation load on a second component enriched for HCN and Kv channel genes.
2. **RQ2.** H2: at area level, cell-type composition (MERFISH subclass proportions) explains >= 60% of the cross-area variance in median timescale; within-type gene expression adds < 10% out-of-sample R^2 once composition is included.
3. **RQ3.** H3: within layer (L2/3, L4, L5, L6 separately), the gene-timescale association weakens by at least half relative to the all-layer analysis, indicating a composition-driven effect; the exception is L5, where ET vs IT proportion drives a residual within-layer effect.
4. **RQ4 (replication).** H4: PLS models fit on Visual Coding generalise to Visual Behavior units with out-of-sample R^2 within 0.1 of in-sample; IBL visual-area units show the same rank order of areas.
5. **RQ5 (nulls).** H5: naive gene-set enrichment on the top-loading genes gives more than 10 significant GO categories at FDR 5%; after gene-ensemble nulls fewer than 3 remain, and ion-channel families survive.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| Allen Visual Coding - Neuropixels (Siegle et al., 2021) | Units in V1, LM, AL, RL, AM, PM, LGd, LP + hippocampus; spike times, stimuli (drifting gratings, natural movies, flashes), CSD-based layers, opto-tagging (Sst/Pvalb/Vip) | 58 sessions, ~100k units, ~1 TB NWB | Open (AllenSDK; AWS S3 `allen-brain-observatory`, no login) | https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html |
| Allen Visual Behavior - Neuropixels | Replication set (change-detection task; same areas) | ~150 sessions | Open (AllenSDK / S3) | https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html |
| IBL Brain-wide map (International Brain Laboratory, 2023 bioRxiv / 2025) | Replication set across labs; 200+ areas | ~700 insertions, ~600k units | Open (ONE API; AWS S3) | https://int-brain-lab.github.io/ONE/ |
| Allen Mouse Brain Atlas ISH (Lein et al., 2007) | Structure-level expression energy for ~20k genes | via API | Open | https://mouse.brain-map.org/ |
| ABC Atlas MERFISH (Yao et al., 2023; Zhang et al., 2023) | Cell-resolved expression + subclass + layer + CCF coordinates | ~4 M + ~9 M cells | Open (S3 bucket `allen-brain-cell-atlas`) | https://alleninstitute.github.io/abc_atlas_access/ |
| Allen CCFv3 + hierarchy scores (Harris et al., 2019 Nature, supplementary tables) | Area coordinates and anatomical hierarchy covariate | small | Open | https://atlas.brain-map.org/ |
| Allen Mouse Brain Connectivity Atlas | Feedforward/feedback connection weights (alternative hierarchy) | ~1 GB | Open (AllenSDK) | https://connectivity.brain-map.org/ |

## Methods

Pipeline (modules in `src/gene_timescales/`):

1. **Unit selection.** Quality metrics from AllenSDK (`isi_violations < 0.5`, `amplitude_cutoff < 0.1`, `presence_ratio > 0.9`), firing rate > 0.5 Hz in spontaneous epochs, area from CCF registration, layer from CSD (Visual Coding) or depth relative to L4 sink.
2. **Timescales** (`timescales.py`). Spike-count autocorrelation in spontaneous / grey-screen epochs and in natural-movie epochs; exponential fit with offset on lags 1-500 ms with bin 5 ms (`fit_exponential_timescale`); a two-timescale fit and the information timescale (Rudelt et al.) as secondary targets. Bias correction by matching the analysis window across areas and by reporting the aBC estimator (Zeraati et al., 2022; external package `abcTau`) as a sensitivity analysis.
3. **Tuning** (`timescales.py`). Orientation and direction selectivity index (drifting gratings), preferred temporal frequency, response latency (first bin exceeding baseline mean + 3 SD), adaptation index (late/early response ratio), receptive-field size (Gabor / flash mapping via AllenSDK receptive-field analysis).
4. **Area-level and layer-level targets.** Median of each property per area (and per area x layer), with bootstrap SE over units; per-unit models are fit with area as a random effect (statsmodels MixedLM) as a sensitivity analysis.
5. **Gene maps** (`gene_maps.py`). ISH: expression energy per CCF structure, coronal experiments averaged per gene, z-scored across areas. MERFISH: mean log-normalised expression per area (and per area x layer) from ABC cells; subclass proportion matrices. Genes filtered to those with reliable ISH (>= 2 experiments, correlated) and, for MERFISH, the 500/1,122 panel plus imputed genes as a sensitivity set.
6. **Prediction** (`gene_maps.pls_predict`, `ridge_predict`). PLS regression of a target (areas x 1) on genes (areas x G) with nested leave-one-area-out CV for the number of components; ridge as alternative. Gene loadings on the first components are the "gene gradient". Composition decomposition: fit target ~ composition (ridge), then residual target ~ genes; report the R^2 of each step (`gene_maps.composition_partial`).
7. **Nulls** (`nulls.py`). (a) 3-D variogram surrogates of the target over area centroids (CCF mm); (b) hierarchy-stratified permutations (areas permuted within tertiles of the Harris hierarchy score); (c) gene-ensemble nulls for family-level statistics with expression-level and Moran's-I matching; (d) BH-FDR across targets and families.

Tools: AllenSDK, ONE-api / ibllib, pynwb, numpy/scipy/pandas, scikit-learn (PLS, ridge), statsmodels, abagen-style conventions for expression normalisation.

## Evaluation & statistics

- Primary metric: out-of-sample R^2 (leave-one-area-out for area-level models; leave-one-session-out for unit-level mixed models). Report bootstrap CIs over units for the area medians and over areas for R^2.
- Nulls as above; p-values are empirical, with >= 5,000 surrogates/permutations.
- Multiple comparisons: BH-FDR across the tuning-property panel and the gene-family panel.
- Leakage: PLS component count and gene filtering are chosen inside the CV loop; Visual Behavior and IBL are touched only once, after the Visual Coding model is frozen (pre-registered).
- Timescale-estimation sensitivity: exponential-with-offset vs two-exponential vs aBC; bin size 2 vs 5 ms; spontaneous vs stimulus epochs. The paper reports whether the gene associations are stable across estimators, which is itself a result.
- Positive control: the Harris hierarchy score must be recoverable from gene expression (PLS R^2 > 0.5), replicating Fulcher et al. (2019 PNAS).

## Publishable angle

Headline: "Gene-expression gradients predict neural timescales and tuning across the mouse visual hierarchy, but most of the prediction is cell-type composition; a small set of channel genes carries a within-type effect that replicates across three Neuropixels surveys." This settles whether "genes predict timescales" is a statement about cell-type proportions or about molecular tuning within types, which matters for interpreting the human AHBA literature.

Target venues: Nature Communications; PLOS Computational Biology; eLife; Journal of Neuroscience; Cosyne / SfN abstracts.

Follow-ups: apply the frozen model to Patch-seq data (Gouwens et al., 2020 Cell) to test the within-type prediction at the single-cell level; extend to human ECoG (Gao et al., 2020) with the same decomposition using human MERFISH data as they appear; opto-tagged units as a direct composition check.

## Risks, confounds & mitigations

- Timescale estimates depend on firing rate and window length; mitigation: rate-matched subsampling, aBC estimator, and rate as a covariate.
- Area labels in Neuropixels data carry registration error (especially at area borders); mitigation: exclude units within 100 um of borders (CCF distance) and repeat with the finer per-unit coordinates (Shi et al. approach) instead of area means.
- Few areas (8-12 in the visual system) limit area-level power; mitigation: unit-level mixed models with CCF coordinates, and the IBL brain-wide set for a broad-area analysis.
- Gene co-expression makes "which genes" ill-posed; mitigation: report families and PLS components rather than single genes, with ensemble nulls.
- MERFISH composition and ISH expression come from different animals and ages; mitigation: use the MERFISH expression itself for a same-animal composition/within-type decomposition.
- Layer assignment noise; mitigation: use only units with confident CSD-derived layers and repeat with depth bins.

## Milestones

- [ ] AllenSDK cache set up; `--sample` downloads one Visual Coding session and computes unit-level timescales end to end
- [ ] Unit QC and tuning panel computed for all Visual Coding sessions; area x layer summary tables
- [ ] ISH and MERFISH gene maps per area (and per layer); composition matrices
- [ ] Positive control (hierarchy from genes) passes
- [ ] PLS/ridge models with nested CV; gene-family statistics with ensemble nulls
- [ ] Composition decomposition and within-layer analysis
- [ ] Pre-registration frozen; Visual Behavior and IBL replication run once
- [ ] Sensitivity analyses (timescale estimator, bins, epochs)
- [ ] Manuscript + code release

## Ethics / data-use notes

- All data are open, secondary-use animal data (Allen Institute Terms of Use; IBL data under CC-BY). Cite the primary papers and the data releases.
- No human or credentialed data are involved; nothing under `data/` is committed (see `.gitignore`).
- Report negative results for gene families that do not survive nulls; do not publish per-gene lists without the ensemble-null qualifier.
