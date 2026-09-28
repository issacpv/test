# Connectome-prediction fairness: a causal audit of subgroup failure in CPM / ridge-on-FC

**Pitch.** Brain-behaviour prediction models trained on the HCP majority demographic are less accurate — and wrong in a *different way* — for under-represented subgroups; this project measures the gap in error *structure*, decomposes it causally into sample-size, motion/scan-quality and label-reliability components, and tests whether group-aware training (balanced weighting, group-DRO) closes it without hurting anyone.

| | |
|---|---|
| **Status** | Design + starter code; no results yet |
| **Difficulty / timeline** | MSc thesis or 1st-year PhD project, 6–9 months (12 with ABCD replication) |
| **Compute** | Laptop-scale once FC matrices exist (n≈1,000 × 79,800 edges; ridge/CPM in seconds). Parcellating HCP dense time series needs ~1 TB scratch and a workstation (or use the PTN release: <20 GB). |
| **Package** | `src/cpm_fair` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                    # synthetic-data tests, no downloads
python scripts/download_data.py --sample     # AOMIC confounds for 2 subjects + HCP dry-run listing
```

```python
import sys; sys.path.insert(0, "src")
from cpm_fair import fc_loader, predictors, subgroup_metrics, reweighting, mediation

df = fc_loader.load_hcp_subject_table("data/hcp/unrestricted.csv", "data/hcp/restricted.csv")
X, ids = fc_loader.build_fc_dataset(
    df["Subject"], lambda s, run: f"data/hcp/{s}/{run}/{run}_Atlas_MSMAll_hp2000_clean.Schaefer400.ptseries.nii")
df = df.set_index("Subject").loc[ids]
groups = fc_loader.define_subgroups(df)                       # race/ethnicity label (restricted data)
y = df["CogTotalComp_Unadj"].to_numpy()
conf = df[["Age_in_Yrs", "Movement_RelativeRMS_mean"]].assign(sex=(df["Gender"] == "F").astype(int)).to_numpy()

preds = predictors.cross_val_predict_family(predictors.RidgeFC(), X, y, df["family"], stratify=groups, confounds=conf)
print(subgroup_metrics.subgroup_performance(preds["y"], preds["y_hat"], groups))
print(subgroup_metrics.cluster_bootstrap_gap(preds["y"], preds["y_hat"], groups, df["family"], reference="White"))
print(subgroup_metrics.error_structure(preds["y"], preds["y_hat"], groups, reference="White"))
print(reweighting.evaluate_training_schemes(X, y, groups, df["family"])["group_dro"])
print(mediation.decompose_gap(preds["y"], preds["y_hat"], groups, df["Movement_RelativeRMS_mean"],
                              reference="White", target="Black or African Am."))
```

Repository layout:

```
src/cpm_fair/            fc_loader · predictors · subgroup_metrics · reweighting · mediation
scripts/download_data.py HCP (S3, credentials from env), AOMIC (anonymous S3), ABCD instructions
data/README.md           step-by-step acquisition and expected layout
tests/test_cpm_fair.py   synthetic cohort with families, subgroups and motion
```

## Background

Connectome-based predictive modelling (CPM; Finn et al., 2015, *Nat Neurosci*; Shen et al., 2017, *Nat Protoc*) and ridge/kernel regression on vectorised functional connectivity (FC) are the workhorses of individual-differences neuroimaging. They are increasingly proposed as biomarkers, yet the samples they are trained on are demographically skewed (HCP S1200 is ~75% White; ABCD ~52% White). Li et al. (2022, *Sci Adv*) showed that FC models trained on HCP and ABCD predict cognition and other behaviours *worse for African American participants* than for White Americans, even when trained on African-American-only samples; Greene et al. (2022, *Nature*) showed that models fail for individuals who "defy sample stereotypes" — errors track how far a person's sociodemographic/clinical profile is from the sample's modal profile. Benkarim et al. (2022, *PLoS Biol*) showed prediction accuracy in clinical cohorts is interlocked with population heterogeneity, and Kopal, Uddin & Bzdok (2023, *Nat Methods*) and Dhamala, Yeo & Holmes (2025, *Nat Neurosci*) argue that diversity must be modelled rather than adjusted away. Ricard et al. (2023, *Nat Neurosci*) lay out the acquisition-side causes.

The engineering question a BME researcher can answer is *why*: is the gap a property of the brain-behaviour mapping (distribution shift), an artefact of fewer training subjects, a scan-quality confound (head motion differs between groups and destroys FC-behaviour associations; Siegel et al., 2017, *Cereb Cortex*), or a *label* problem (cognitive tests with lower reliability/validity in a subgroup cap attainable accuracy through attenuation)? Each cause implies a different fix.

## The research gap

**Done.**
* Li et al. 2022 — accuracy gaps (Pearson r) by race/ethnicity in HCP and ABCD, FC only, kernel ridge; no mechanism.
* Greene et al. 2022 — "stereotype" analysis: misclassified subjects deviate from sample-typical profiles (HCP, UCLA CNP, PNC).
* "When brain models aren't universal" (bioRxiv 2025, doi:10.1101/2025.11.12.688133) — ABCD benchmark across 80+ MRI phenotypes and four training strategies: structural MRI most biased, task-fMRI least; **balanced subsampling removed the gap without accuracy loss**; gains from adding minority subjects plateau at balance.
* "Supervised domain adaptation mitigates cross-ethnicity prediction gaps" (bioRxiv 2026) — same cohort; balanced weighting ≈ best and stable from n=10 target subjects.
* Rosenblatt et al. (2024, *Nat Commun*) — leakage (including family leakage) inflates CPM accuracy, so audits must use family-aware CV.

**Missing (and what this project targets).**
1. **No causal decomposition.** All prior work reports the gap; none partitions it into sample size vs. motion/scan quality vs. label reliability vs. genuine mapping shift. A model-agnostic decomposition (matched-n retraining, motion-matched evaluation, attenuation ceilings, then residual) has not been reported.
2. **Error structure, not just r.** Gaps have been reported as differences in correlation; calibration slope, systematic bias (mean residual), residual variance and regression-toward-the-majority-mean have not been quantified per subgroup. These matter clinically: a model can have equal r but systematically over-predict a subgroup.
3. **Adults + cross-dataset transport.** The 2025–2026 benchmarks are ABCD (9–11-year-olds). Whether the same fixes work in adults (HCP) and transport across datasets (HCP → AOMIC/HBN/NKI) is untested.
4. **Group-DRO / worst-group objectives** (Sagawa et al., 2020, *ICLR*) have not been evaluated for connectome regression; balanced weighting is the only reweighting scheme tested so far.
5. **Intersectionality and SES.** Race × sex × SES cells and continuous SES (income/education) have not been audited jointly.

## Research questions / hypotheses

1. **H1 (gap exists in adults, family-aware).** Under family-aware 10×5-fold CV on HCP S1200, ridge-on-FC and CPM predicting `CogTotalComp_Unadj` show lower r and higher RMSE for Black/African American, Hispanic/Latino and low-SES subgroups than for White participants (gap in r ≥ 0.10; family-cluster bootstrap CI excluding 0).
2. **H2 (error structure differs).** The minority-group residuals show (a) a non-zero mean residual (systematic over-prediction toward the majority mean), (b) a calibration slope < the majority's and (c) larger residual variance (Levene p < 0.05), even where r gaps are small.
3. **H3 (motion mediates part of the gap).** Mean framewise displacement/relative RMS mediates a *minority* share (10–40%) of the |error| difference; motion-matched evaluation shrinks but does not eliminate the gap.
4. **H4 (sample size explains less than half).** Training the majority model at the minority's n reproduces < 50% of the gap.
5. **H5 (label reliability caps accuracy).** Test–retest reliability of NIH Toolbox composites (HCP retest subset, n≈45; ABCD 2-year) differs by subgroup by < 0.05 → the attenuation component is small; if larger, the "gap" is partly a measurement, not a model, problem.
6. **H6 (fixes).** Balanced subsampling / inverse-frequency weighting and group-DRO reduce the worst-group gap by ≥ 50% with ≤ 0.02 loss in overall r; group-DRO yields the smallest worst-group RMSE.
7. **H7 (transport).** Fixes trained on HCP transfer to AOMIC (sex, education) and to HBN/NKI (race/ethnicity) with the same ordering of schemes.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| HCP S1200 (young adult) | 4 × 15-min rest fMRI (ICA-FIX, MSMAll), task fMRI (optional), NIH Toolbox cognition, sex, age; **restricted**: Family_ID, Race, Ethnicity, SSAGA income/education, exact age; motion (RelativeRMS) | 1,003 with 4 rest runs; ~1,200 with behaviour | Free registration (open) + Restricted Data application (DUA) | https://db.humanconnectome.org ; S3 `s3://hcp-openaccess` |
| HCP retest subset | Repeat sessions for reliability of FC and cognition | 45 | Same as above | same |
| AOMIC-PIOP1 / PIOP2 (ds002785 / ds002790) | Rest fMRI with fMRIPrep derivatives, Raven IQ, sex, education category | 216 / 226 | Open (CC0) | https://openneuro.org/datasets/ds002785 |
| UCLA CNP (ds000030) | Rest fMRI, WAIS/other tests, sex, diagnosis; **no race column** → sex/QC audit only | 272 | Open | https://openneuro.org/datasets/ds000030 |
| Healthy Brain Network | Rest fMRI, WISC-V, race/ethnicity, household income | > 2,000 | DUA | https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/ |
| NKI-Rockland Enhanced | Rest fMRI, cognitive battery, race/ethnicity, lifespan | > 1,000 | DUA (NITRC) | http://fcon_1000.projects.nitrc.org/indi/enhanced/ |
| ABCD (optional) | Rest fMRI, NIH Toolbox, race/ethnicity, income, twin/family IDs | ~11,800 | NDA DUC (credentialed) | https://nda.nih.gov/abcd |

## Methods

1. **FC features.** Schaefer-400 (7-network) parcellation of MSMAll-aligned, ICA-FIX-cleaned rest runs (`wb_command -cifti-parcellate`), Pearson → Fisher-z, average of 4 runs; 79,800 edges. Sensitivity: Glasser-360, HCP PTN ICA-300 node time series, task-fMRI FC (WM, language).
2. **Targets.** `CogTotalComp_Unadj` (primary), `CogFluidComp_Unadj`, `PMAT24_A_CR`, plus a "non-cognitive" negative control (e.g. `PSQI_Score`).
3. **Subgroups.** `cpm_fair.fc_loader.define_subgroups`: Hispanic/Latino → own category; race categories with n < 30 collapsed. SES index = mean z of income level and years of education. Intersectional cells race × sex.
4. **Models.** (a) CPM (`CPMRegressor`, p < 0.01 edge selection inside the fold, positive/negative networks); (b) standardised ridge with inner LOO-CV alpha (`RidgeFC`); (c) kernel ridge (linear kernel) as in Li et al. for comparability.
5. **Validation.** `FamilyKFold`: families never split; folds stratified by subgroup; 10 repeats × 5 folds; confounds (age, sex, mean RelativeRMS, optionally ICV) residualised with *training-fold* coefficients (`residualize`). Never residualise race.
6. **Fairness metrics** (`subgroup_metrics`): per-group r, R², MAE, RMSE, bias, residual SD, calibration slope/intercept; `error_structure` (residual-vs-truth slope, Welch, Levene, KS); `accuracy_at_matched_n` to compare groups at equal n.
7. **Uncertainty & significance.** Family-cluster bootstrap CIs for every gap; family-level label-permutation null (`permutation_gap_test`, 5,000 permutations) — the null preserves group sizes and family structure.
8. **Causal decomposition** (`mediation`): (i) `matched_n_reference_performance` (train majority at minority n, 50 draws); (ii) `mediation_by_motion` on |error| with age/sex covariates and bootstrap CI; (iii) `motion_matched_subsample` evaluation (nearest-neighbour on RelativeRMS, caliper 0.02 mm); (iv) `attenuation_ceiling` from group-specific test–retest ICCs (HCP retest; ABCD baseline→2-year); (v) `decompose_gap` assembles the additive table and reports order sensitivity (all 3! orders).
9. **Fixes** (`reweighting`): ERM baseline; inverse-frequency weights; balanced subsampling (upper bound per the 2025 ABCD benchmark); `GroupDRORidge` (exponentiated-gradient group weights + closed-form weighted ridge); optionally group-specific intercepts (a "group-aware" model that uses the label at test time — report separately because it is not deployable without the label).
10. **Cross-dataset transport.** Train on HCP, test zero-shot on AOMIC (harmonised parcellation; sex and education gaps) and on HBN/NKI (race/ethnicity), then fine-tune with balanced weighting (n = 10, 30, 100 target subjects).

**Pre-registered analysis grid** (primary endpoint in bold; everything else is sensitivity or secondary):

| Factor | Levels |
|---|---|
| Target | **CogTotalComp_Unadj**; CogFluidComp_Unadj; PMAT24_A_CR; PSQI_Score (negative control) |
| Features | **Schaefer-400 rest FC (4 runs)**; Glasser-360; PTN ICA-300; task FC (WM, language) |
| Model | **ridge (RidgeFC)**; CPM p < 0.01; kernel ridge (Li et al. 2022 replication) |
| Subgroup axis | **race (White vs. Black/African American)**; Hispanic/Latino; sex; SES tertile; race × sex cells (n ≥ 30) |
| Training scheme | **ERM**; inverse-frequency; balanced subsample; group-DRO; group-specific intercept (label-at-test, reported separately) |
| Confounds (in-fold) | **age, sex, mean RelativeRMS**; + ICV; none |
| CV | **10 × 5-fold family-stratified**; 100 repeats for the primary endpoint |
| Decomposition | matched-n (50 draws); motion-matching (200 draws, caliper 0.02 mm); attenuation ceiling (HCP retest ICC; ABCD 2-yr ICC) |

Primary-endpoint tests: 1 target × 1 feature set × 1 model × 1 axis × 5 schemes = 5 gaps (Holm-corrected); the remaining ~700 cells of the grid are reported as heat-maps with CIs, not tested individually.

## Evaluation & statistics

* **Primary endpoint:** gap in Pearson r (and RMSE) between reference and each subgroup, with 95% family-cluster bootstrap CI and permutation p; Holm correction across (targets × subgroups × models).
* **Error structure:** bias and calibration slope per group with bootstrap CIs; Levene/KS for variance and distribution shape.
* **Leakage control:** family-aware folds; confound regression and feature selection inside folds; hyper-parameters chosen by inner CV; test folds stratified by subgroup so every fold contains minority subjects.
* **Multiple comparisons:** Holm within each family of tests; the decomposition components are descriptive (bootstrap CIs, no p-values).
* **Null models:** label permutation at the family level; "shuffled-race" negative control (assign race labels randomly → gap should vanish); negative-control target (PSQI) should show no consistent gap pattern.
* **Sample-size floors:** subgroups < 30 are reported but not tested; matched-n analyses use the smallest subgroup's n.
* **Pre-registration:** hypotheses H1–H7 and the analysis grid registered on OSF before touching restricted data.

## Publishable angle

**Headline result:** "Most of the racial/ethnic gap in connectome-based cognition prediction in adults is *not* explained by sample size or head motion; it is a shift in the brain–behaviour mapping that balanced weighting and group-DRO remove at no cost to majority accuracy — but the models remain mis-calibrated (systematic over-prediction toward the majority mean) unless calibrated per group." Whichever way the decomposition falls, the result is informative: if motion dominates, the fix is acquisition/denoising; if sample size dominates, the fix is recruitment; if reliability dominates, the fix is measurement.

**Target venues:** *Nature Human Behaviour* (methods/audit papers with fairness angle), *Imaging Neuroscience*, *Science Advances* (as direct follow-up to Li et al. 2022), *NeuroImage*; conference: MICCAI FAIMI workshop or OHBM.

**Follow-ups:** the same decomposition for psychiatric classification (ABIDE, HBN diagnoses); extension to structural connectivity and multimodal stacking; a "fairness card" reporting standard for BWAS models.

## Risks, confounds & mitigations

| Risk | Mitigation |
|---|---|
| Restricted-data approval delay (family/race) | Start with AOMIC sex/education audit and HCP PTN data; family-aware code is ready; apply on day 1 |
| Small subgroup n (HCP Black ≈ 150, Hispanic ≈ 100) → noisy gaps | Cluster bootstrap; matched-n comparisons; pool across 10 CV repeats; replicate in HBN/ABCD |
| Motion is both confound and mediator | Report both "residualised" and "un-residualised" analyses; mediation model treats motion as mediator explicitly |
| Race is a social construct; risk of biological essentialism | Analyse race as a proxy for structural exposures (SES, scanner site distribution), report SES jointly, follow Ricard et al. 2023 guidance in writing |
| Multiple targets/models → false positives | Pre-registered primary endpoint; Holm |
| Scanner/site effects in multi-site replications | Include site as covariate; ComBat only inside folds |
| Reliability estimates from n=45 retest are imprecise | Use ABCD (n > 5,000 with 2-year follow-up) for reliability by subgroup; report CIs |

## Milestones

- [ ] ConnectomeDB registration; Restricted Data application; OSF pre-registration
- [ ] Download & parcellate 1,003 HCP subjects (or PTN node time series); build FC dataset
- [ ] Reproduce Li et al. 2022 gap with kernel ridge under family-aware CV (sanity benchmark)
- [ ] Error-structure tables (bias, calibration, variance) for race, ethnicity, sex, SES, intersections
- [ ] Mediation-by-motion and motion-matched evaluation
- [ ] Matched-n retraining and attenuation ceilings (HCP retest ICCs)
- [ ] Group-DRO / reweighting / balanced-subsample comparison
- [ ] AOMIC transport (sex, education); HBN or NKI transport (race/ethnicity); optional ABCD
- [ ] Manuscript + released "fairness card" template and code

## Ethics / data-use notes

* HCP restricted data (family structure, race/ethnicity, exact age, income) is governed by the Restricted Data Use Terms: do not redistribute, do not attempt re-identification, do not publish family-level identifiers; report subgroup results only for cells with n ≥ 10.
* ABCD data may only be used under an approved NDA DUC and must not leave approved environments; HBN/NKI require DUAs.
* Never upload subject-level data (including derived FC matrices or residual tables with demographic labels) to third-party services or LLM APIs.
* Race/ethnicity are treated as social categories used solely to audit model behaviour; language in outputs follows Ricard et al. (2023) and the APA guidance on reporting race.
* No data files are committed (see `.gitignore`); credentials live in environment variables only.
