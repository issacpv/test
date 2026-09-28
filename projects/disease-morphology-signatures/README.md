# disease-morphology-signatures

**A lab-effect-corrected meta-analysis of dendritic morphology in Alzheimer's-model, aging and epilepsy reconstructions on NeuroMorpho.org: are the reported "disease signatures" reproducible across archives, separable from normal aging, and large enough to matter electrotonically?**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (cohort builder for NeuroMorpho condition metadata, SWC morphometrics, random-effects meta-analysis with Hartung-Knapp CIs, passive-cable electrotonic calculator).
- Difficulty: MSc thesis to first PhD chapter. Statistics and cable theory; no GPU.
- Timeline: 6-9 months (1-2 months harvest + curation of conditions, 1 month morphometrics, 2 months meta-analytic modelling, 1-2 months electrotonic simulations, 1-2 months writing).
- Compute: a laptop. The condition-labelled subset of NeuroMorpho is a few thousand reconstructions (seconds per file); passive-cable solves are sparse linear systems of ~10^3-10^4 unknowns.

## Background

Dendritic structure is altered in Alzheimer's disease (AD) models, in normal aging and in chronic epilepsy, and each alteration has been argued to change excitability. Dendritic simplification of CA1 pyramidal neurons in APP/PS1 mice is functionally linked to hyperexcitability (Siskova et al., 2014, *Neuron*); rhesus-monkey prefrontal pyramidal neurons lose distal dendrite and spines with age and their electrotonic structure changes accordingly (Kabaso et al., 2009, *Cerebral Cortex*; Coskren et al., 2015, *Journal of Computational Neuroscience*); dentate granule cells born after status epilepticus show altered dendritic morphology that, in models, makes them less excitable individually but contributes to network hyperexcitability with mossy-fibre sprouting (Tejada et al., 2012, *PLoS ONE*; Tejada, Garcia-Cairasco & Roque, 2014, *PLoS Computational Biology*).

Almost every one of these claims rests on one laboratory, one model, one staining/reconstruction protocol and 10-40 cells per group. NeuroMorpho.org curates the `experiment_condition` of each deposited reconstruction (the 2024 update in *FASEB BioAdvances* reports more than 1,300 distinct experimental conditions across the archive; Ascoli, Donohue & Halavi, 2007, *Journal of Neuroscience*; Akram et al., 2018, *Scientific Data*), together with the archive (lab), reconstruction software, shrinkage correction, slice thickness and staining. That makes a cross-laboratory synthesis possible for the first time: most disease datasets were deposited *with their own controls*, so the disease contrast can be estimated within archive and then pooled across archives, the way clinical meta-analyses pool within-trial contrasts.

## The research gap

**What has been done**

- Single-lab morphological studies of AD models (APP/PS1, 3xTg-AD, 5xFAD, Tg2576), aging (rat, mouse, rhesus) and epilepsy (pilocarpine, kainate, kindling), many of which deposited SWC files on NeuroMorpho.org; the papers cited above are representative and their reconstructions are retrievable by `experiment_condition` and `reference_pmid`.
- Computational reuse of individual disease datasets to infer functional consequences (Coskren et al., 2015; Tejada et al., 2014) - one dataset at a time.
- Archive-wide mining of NeuroMorpho by metadata category (Polavaram, Gillette, Parekh & Ascoli, 2014, *Frontiers in Neuroanatomy*) that treats categories as fixed labels and does not contrast disease vs. control within archive.
- Batch/lab effects in NeuroMorpho have been recognised as a threat to cross-lab comparison (see the related project `neuromorpho-scaling-laws`, which models archive as a random effect for allometric scaling), but no disease-focused synthesis has used them.

**What is missing (checked against 2023-2026 literature; no cross-archive disease meta-analysis of neuronal reconstructions was found)**

1. No study has pooled the *within-archive* disease-vs-control morphological contrasts across NeuroMorpho archives to obtain a reproducible effect size per condition (AD model, aging, epilepsy) and per morphometric, with between-archive heterogeneity quantified.
2. No study has asked whether the AD-model morphological signature is *distinguishable from* the normal-aging signature after matching age and region, i.e. whether "AD morphology" is largely accelerated aging in the deposited data.
3. No study has propagated the pooled morphological effect sizes through passive cable models to obtain a pooled *electrotonic* effect size (input resistance, dendrite-to-soma attenuation, electrotonic length) with uncertainty that includes between-lab heterogeneity.
4. Reporting-bias tools standard in clinical meta-analysis (funnel asymmetry, small-study effects) have never been applied to neuromorphology, where deposited datasets are a biased sample of studies (positive results are more likely to be reconstructed and deposited).

## Research questions / hypotheses

1. **H1 (reproducibility).** For AD models, the pooled within-archive standardised difference in total dendritic length and branch-point number of hippocampal/cortical pyramidal neurons is negative (simplification) with a 95% Hartung-Knapp CI excluding zero; between-archive heterogeneity I^2 exceeds 50%. Test: REML random-effects meta-analysis per morphometric; Q-test and I^2 with CIs.
2. **H2 (aging vs. AD dissociation).** The vector of pooled effect sizes across morphometrics ("signature") for AD models is more similar to the aging signature than expected under a permutation null (cosine similarity > null 95th percentile), but the distal-dendrite (high Sholl radius) component differs. Test: signature cosine with within-archive label permutation; per-radius Sholl meta-regression with age as moderator.
3. **H3 (epilepsy is region-specific).** Epilepsy-model effects on dentate granule cells (basal dendrites, increased branching) have opposite sign to effects on CA1/cortical pyramidal cells (simplification); meta-regression with cell class as moderator explains a substantial share of heterogeneity (R^2_meta > 0.3).
4. **H4 (electrotonic consequence).** The pooled morphological changes, applied to matched control reconstructions via passive cable simulation, change somatic input resistance by more than 10% in AD models but not in epilepsy models; uncertainty intervals obtained by sampling from the meta-analytic posterior of effect sizes.
5. **H5 (small-study and protocol effects).** Effect sizes are larger in smaller-n datasets (Egger regression intercept != 0) and in datasets without shrinkage correction; "shrinkage not corrected" is a significant moderator of z-extent-dependent metrics.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| NeuroMorpho.org (v8.x) | Records whose `experiment_condition` matches AD-model / aging / epilepsy keywords plus same-archive controls; per-record `archive`, `reconstruction_software`, `shrinkage_corrected`, `slicing_thickness`, `stain`, `min_age`/`max_age`, `brain_region`, `cell_type`, `reference_pmid`; CNG.swc files; L-Measure morphometry | A few thousand condition-labelled reconstructions across tens of archives (exact counts are produced by `scripts/download_data.py --report`) | Open, no registration (REST API v1) | https://neuromorpho.org/api/ , https://neuromorpho.org/apiReference.html |
| Alzforum Research Models | Model-level covariates for AD models (transgene, promoter, onset of plaques/tangles) used as meta-regression moderators | ~200 models | Open (web) | https://www.alzforum.org/research-models |
| ModelDB (SenseLab) | Passive/active parameter sets published with the original disease-model papers (e.g. Coskren et al., 2015; Tejada et al., 2014) for the electrotonic step | a few entries | Open | https://modeldb.science |
| Allen Cell Types Database (optional) | Single-protocol, healthy adult mouse/human reconstructions as an external reference distribution for morphometrics | ~1,000 reconstructions | Open (allensdk) | https://celltypes.brain-map.org |

No credentialed data are involved.

## Methods

1. **Harvest and condition taxonomy** (`scripts/download_data.py`, `src/disease_morph/cohort.py`). Page through `/api/neuron/select` for mouse, rat, human and monkey; keep records with non-control `experiment_condition`; map the free-text condition to a controlled vocabulary (`ad_model`, `aging`, `epilepsy`, `control`, `other`) with an explicit keyword table that is committed with the code; attach same-archive controls (same species, region and cell class where possible). Log the mapping for every record so it can be audited.
2. **Inclusion / QC.** `physical_Integrity` containing "Dendrites Complete"; dendrites present in `structural_domains`; at least 5 cases and 5 controls per archive-contrast; duplicates across archives removed by `neuron_name`/`reference_pmid`.
3. **Morphometrics** (`src/disease_morph/swc.py`). Recomputed from CNG.swc so that the same definition applies to every archive: total dendritic length, branch points, tips, max branch order, mean branch length, Sholl profile at fixed radii (crossings per 20 um shell), maximum Euclidean and path extents, convex-hull volume, PCA effective dimensionality (to detect planar/z-compressed data), mean diameter. Apical and basal dendrites are separated where SWC type codes allow.
4. **Within-archive contrasts** (`src/disease_morph/meta.py`). For each archive x condition x morphometric: Hedges' g with small-sample correction and its variance; then REML random-effects pooling with Hartung-Knapp-Sidik-Jonkman intervals (recommended for few studies), Q, I^2 and tau^2; meta-regression (weighted least squares on g) with moderators: species, region, cell class, age, shrinkage correction, reconstruction software, staining method, n per group.
5. **Signatures.** Stack pooled g's over morphometrics into a signature vector per condition; cosine similarity between conditions; null by permuting condition labels within archive and recomputing the whole pipeline (1,000 permutations).
6. **Electrotonic propagation** (`src/disease_morph/passive.py`). Passive compartmental model of each reconstruction (Rm, Ra, Cm from the corresponding ModelDB entries or standard values 20 kOhm cm^2, 150 Ohm cm, 1 uF/cm^2): steady-state input resistance at the soma, voltage attenuation from each tip to the soma, electrotonic length. Pool the disease-control differences of these quantities exactly like the morphometrics, so the electrotonic effect size inherits between-lab heterogeneity.
7. **Tools**: `requests`, `numpy`, `scipy` (sparse solves, ConvexHull), `pandas`, `statsmodels` (meta-regression cross-check), optional `neurom` for morphometric cross-validation, optional `NEURON` for active-model follow-ups.

## Evaluation & statistics

- Primary estimands: pooled g per condition and morphometric with HKSJ 95% CI, tau^2 and I^2; signature cosine similarities with permutation p-values; pooled electrotonic effect sizes.
- Leakage prevention: all effects are within-archive, so no lab contributes both a case and a control from different protocols. Archives with the same PMID are merged. No supervised model is trained on pooled data except the leave-one-archive-out classifier used as a secondary reproducibility check (train on all-but-one archive's within-archive z-scored morphometrics, test on the held-out archive).
- Multiple comparisons: hypotheses H1-H4 pre-registered with one primary morphometric each (total length, distal Sholl area, branch points, input resistance); Benjamini-Hochberg across the remaining morphometrics.
- Nulls: (a) within-archive label permutation for signatures and pooled effects; (b) leave-one-archive-out influence analysis (does one lab drive the pooled effect?); (c) simulated meta-analyses with the observed archive sizes to calibrate I^2 under homogeneity.
- Sensitivity: L-Measure values from NeuroMorpho vs. own recomputation; including/excluding uncorrected-shrinkage archives; DL vs. REML tau^2; fixed-effect vs. random-effect models.

## Publishable angle

- **Headline**: "Across N archives and M reconstructions, AD-model neurons show a reproducible X% reduction in dendritic length (g = ..., I^2 = ...), of which Y% is shared with normal aging; epilepsy effects reverse sign between dentate granule and pyramidal cells; the pooled electrotonic consequence is a Z% increase in input resistance." Plus the first funnel-plot evidence on small-study effects in neuromorphology.
- Target venues: *Neurobiology of Aging*, *Alzheimer's & Dementia* (methods/meta-analysis section), *PLoS Computational Biology*, *Neuroinformatics*, *Epilepsia* (if the epilepsy result is the strongest).
- Follow-ups: the same design for spine-density metadata (see backlog item on spine mining), for human post-mortem AD reconstructions, and for active-conductance models of the pooled morphologies; a living meta-analysis that updates as NeuroMorpho grows.

## Risks, confounds & mitigations

- **Sparse identifying data**: some conditions may have only 2-3 archives with controls. Mitigation: HKSJ intervals and a minimum of 3 archives per pooled estimate; report single-archive conditions descriptively only.
- **Controls not matched** (different age, region or sex within an archive). Mitigation: restrict contrasts to same species/region/cell class; add age difference as a moderator; drop archives whose controls come from a different protocol (`reconstruction_software`, `stain`).
- **Condition text is free-form** ("APP/PS1 6 months", "aged 24 mo", "pilocarpine SE 30 d"). Mitigation: the keyword mapping is version-controlled and every unmatched string is listed for manual curation; report results by curated sub-condition where n allows.
- **Incomplete reconstructions in slices**: `physical_Integrity`, `slicing_thickness` and effective dimensionality as covariates; sensitivity on whole-cell in vivo data.
- **Publication/deposition bias**: Egger test and trim-and-fill are reported but interpreted cautiously with few archives.
- **API availability**: the client retries with backoff and caches every page; the analysis runs from local JSONL/parquet files.

## Milestones

- [ ] Harvest condition-labelled records for mouse/rat/human/monkey; audit the keyword mapping; produce the archive x condition count table.
- [ ] Freeze inclusion rules; build archive-level case-control tables.
- [ ] Recompute morphometrics from SWC; Bland-Altman against L-Measure.
- [ ] Random-effects meta-analysis per condition and morphometric (H1, H3); forest and funnel plots.
- [ ] Signature analysis and aging-vs-AD dissociation with permutation null (H2).
- [ ] Passive cable propagation and pooled electrotonic effects (H4).
- [ ] Moderator analyses (H5); leave-one-archive-out influence.
- [ ] Preprint, code and curated condition-mapping table released.

## Ethics / data-use notes

- NeuroMorpho.org data are open; cite NeuroMorpho.org and every depositing publication (`reference_pmid`/`reference_doi`) as required by its terms of use.
- Human reconstructions on NeuroMorpho are de-identified post-mortem or surgical material curated by the depositing labs; no identifiable data are handled.
- Animal data were generated under the original labs' approvals; this project performs secondary analysis only.
- Do not commit downloaded SWC files, metadata dumps or derived per-cell tables; `data/` and `outputs/` are git-ignored.
