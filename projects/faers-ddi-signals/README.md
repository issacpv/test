# faers-ddi-signals: sex-stratified drug–drug interaction signal detection in FAERS with mechanism-pooled shrinkage

**One-sentence pitch.** Build the first sex-stratified, mechanism-aware drug–drug interaction (DDI) signal resource from FAERS 2015–2026 by comparing the classical DDI statistics (Ω shrinkage, interaction ROR, additive/multiplicative contrasts) with an empirical-Bayes hierarchical model that pools strength across drug pairs sharing a pharmacokinetic mechanism, and evaluate every method against curated clinical DDI lists *and* the reporter-flagged "interacting" role code that FAERS already contains but nobody uses as a label.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc thesis to early-PhD. 6–9 months for the methods-comparison + sex-stratified atlas paper; 12 months with the mechanism-pooled model and a released resource.
- Compute: CPU only. The full openFDA drug/event bulk is ~20 M reports (~10 GB zipped); pair enumeration over a curated 300-drug panel (~45 000 pairs × ~2 000 events) fits in a 64 GB workstation with DuckDB/pyarrow. The `--sample` mode and all tests run on a laptop.

## Background

Roughly one in six adverse-event reports in FAERS lists two or more suspect drugs, and clinically important DDIs (CYP3A4 inhibition of statins, QT-prolonging combinations, serotonergic combinations, anticoagulant + antiplatelet bleeding) are among the most preventable causes of serious drug harm. Spontaneous-report DDI detection has a mature statistical toolkit: the logistic-regression interaction term (van Puijenbroek et al., 1999, *Br J Clin Pharmacol*), additive vs multiplicative expected-count models (Thakrar, Grundschober & Doessegger, 2007, *Br J Clin Pharmacol*), empirical-Bayes MGPS for higher-order itemsets (Almenoff et al., 2003, *Pharmacoepidemiol Drug Saf*; DuMouchel, 1999, *Am Stat*), and the shrinkage observed-to-expected ratio Ω (Norén, Sundberg, Bate & Edwards, 2008, *Stat Med*). TWOSIDES (Tatonetti et al., 2012, *Sci Transl Med*) turned AERS into a public DDI resource, but with data ending in 2009, pooled sexes and propensity-matched single-drug backgrounds.

Sex is the obvious missing stratifier. Women file more reports and experience more adverse drug reactions (Watson et al., 2019, *EClinicalMedicine*; Zucker & Prendergast, 2020, *Biol Sex Differ*), CYP3A4 activity and QT interval differ by sex, and yet DDI signal detection has treated sex, at best, as a sensitivity analysis.

## The research gap

**What has been done.**

- Methods: Ω shrinkage (Norén et al., 2008), interaction ROR / logistic interaction (van Puijenbroek et al., 1999), additive/multiplicative models (Thakrar et al., 2007), MGPS itemsets (Almenoff et al., 2003), systematic evaluation of DDI reporting patterns in VigiBase (Strandell et al., 2011, *Drug Saf*). A simulation-based comparison of DDI detection methods appeared in *PLOS ONE* in 2024 (doi:10.1371/journal.pone.0300268); a 2024 *Journal of Biomedical Informatics* paper discovered clinical DDIs with known pharmacokinetic mechanisms by combining spontaneous reports with EHR data; a 2025 arXiv preprint (2504.00646) proposes detection of high-order (3+ drug) interactions from individual case safety reports.
- Resources: TWOSIDES/OFFSIDES (Tatonetti et al., 2012), AEOLUS standardised FAERS (Banda et al., 2016, *Sci Data*), the DiAna drug-name dictionary for FAERS (Fusaroli et al., 2024, *Drug Saf*), the READUS-PV reporting guideline for disproportionality analyses (Fusaroli et al., 2024, *Drug Saf*).
- Practice: a wave of 2025–2026 single-drug FAERS papers (many in *Frontiers in Pharmacology*) now bolt on an "IOR + Ω" DDI section with age/sex stratification as a sensitivity analysis, always for one index drug.

**What is specifically missing.**

1. **No sex-stratified DDI signal atlas.** Every DDI resource pools sexes. Whether the Ω / IOR of, e.g., (clarithromycin, simvastatin) → rhabdomyolysis, or (citalopram, ondansetron) → QT prolongation differs between women and men in FAERS has not been tested at scale with a sex-specific background and a formal three-way interaction test.
2. **Reporter-flagged interactions are an unused label.** FAERS role code 3 ("interacting"; `DRUG.ROLE_COD = 'I'` in the ASCII files) records the reporter's judgement that a drug interacted. It has never been used to benchmark DDI statistics, although it is the only *FAERS-native* positive label available at pair level.
3. **Mechanism pooling.** DDI statistics are computed pair by pair, so a rarely co-prescribed (strong CYP3A4 inhibitor, sensitive substrate) pair cannot borrow strength from the dozens of pairs sharing the same mechanism. A hierarchical model with a mechanism-level mean is the natural fix and has not been reported for spontaneous-report DDI detection.
4. **Methods compared only in simulation or on one drug.** The 2024 *PLOS ONE* simulation and the single-drug papers do not tell us which statistic ranks best against clinical reference sets on the modern (2015–2026) FAERS.

## Research questions / hypotheses

1. **H1 (method ranking).** Against the ONC high-priority DDI list and the CYP3A4 mechanism set, the mechanism-pooled posterior mean of log-IOR achieves higher AUROC than Ω025, IOR lower bound and RERI (difference ≥ 0.05 AUROC, pair-level bootstrap CI excluding 0), with the largest gain for pairs with n111 < 20.
2. **H2 (no-interaction model matters).** Ω computed under additive, multiplicative, independence and max expected-rate models produces rankings with Kendall τ < 0.8 among the top 500 pairs; the max model is the most specific and the multiplicative the most sensitive against reference sets.
3. **H3 (reporter-flagged label).** Pairs flagged "interacting" on ≥ 3 reports are recovered with recall ≥ 0.6 at a 5 % FDR by IOR-based screening; they are enriched for pharmacokinetic (CYP/P-gp) rather than pharmacodynamic mechanisms.
4. **H4 (sex-specific DDIs).** After BH correction across all (pair, event) triples with n111 ≥ 3 in both sexes, ≥ 5 % of significant DDI signals show a sex × interaction term with ratio-of-IOR > 1.5 or < 0.67; QT-class events (torsade de pointes, electrocardiogram QT prolonged) show stronger interaction signals in women, statin-myopathy interactions do not differ by sex.
5. **H5 (sex-specific background).** Sex-stratified Ω/IOR computed with a same-sex background disagree with the pooled estimate in ≥ 10 % of signals; the disagreement is predicted by the sex imbalance of the two drugs' reporting populations (Simpson-type confounding).

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| openFDA drug/event (FAERS) | reports 2015-01 → 2026-06: drugs with role codes, `openfda.generic_name`, sex, age, reactions, seriousness | ~20 M reports total; ~12 M in window | open (free API key optional) | https://open.fda.gov/apis/drug/event/ |
| FAERS quarterly ASCII | `CASEID`/`CASEVERSION` deduplication, `ROLE_COD` = I | ~1.5 GB/yr zipped | open | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| ONC high-priority DDI list | clinical positives (Phansalkar et al., 2012, *JAMIA*) | ~15 interaction classes | open (supplement) | https://doi.org/10.1136/amiajnl-2012-000935 |
| CredibleMeds QTdrugs | QT-risk categories | ~300 drugs | free registration | https://www.crediblemeds.org/ |
| FDA DDI tables (substrates/inhibitors/inducers) | mechanism sets per CYP/transporter | ~200 drugs | open | https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers |
| TWOSIDES | legacy comparator | 868 k pair-event signals | open | https://nsides.io/ |
| DiAna dictionary / RxNorm | drug-name harmonisation | — | open | Fusaroli et al., 2024; https://rxnav.nlm.nih.gov/ |
| MedDRA | PT → HLT/SOC grouping | — | licence (free academic) | https://www.meddra.org/ |

## Methods

1. **Extraction and deduplication (`faers_ddi.openfda`).** Stream openFDA bulk partitions through `flatten_record`, keep the latest `CASEVERSION` per `CASEID` from the ASCII files, drop literature duplicates by (drugs, reactions, sex, age, country, event date) hashing. Harmonise names with DiAna/RxNorm; keep the role of each drug.
2. **Drug panel and pair space (`faers_ddi.ddi_tables.candidate_pairs`).** Panel = the 300 most-reported substances ∪ every drug in the ONC, CredibleMeds-Known and FDA CYP tables. Unordered pairs co-reported ≥ 20 times; events = MedDRA PTs with ≥ 100 reports, plus PT groups (HLT) for the QT, bleeding, serotonin-syndrome and myopathy classes.
3. **Counting.** Report-level n_ijk cells (each report counted once; role = any, with suspect-only sensitivity analysis). Sex-stratified tables restrict all eight cells to one sex (`sex_stratified_tables`).
4. **Statistics (`faers_ddi.ddi_stats`).** Ω and Ω025 under the four no-interaction models; IOR with LRT; RERI on the odds scale; three-way sex interaction (ratio of IORs with LRT). Multiplicity by BH within each family.
5. **Mechanism pooling.** Assign each pair to mechanism groups (CYP3A4-inhibitor×substrate, CYP2D6, P-gp, QT+QT, serotonergic+serotonergic, anticoagulant+antiplatelet, "none"). Empirical-Bayes normal–normal shrinkage of log-IOR within group (`hierarchical_shrinkage`; DerSimonian–Laird τ²). Extension: full Bayesian model in PyMC/NumPyro with group-level means, pair random effects and sex offsets (not in the starter code).
6. **Evaluation (`faers_ddi.reference_sets`).** Positives from ONC / CredibleMeds / FDA mechanism cross-products with event-class matching; negatives = pairs of drugs in unrelated classes with no listed interaction in DrugBank; partial positives from role code 3. AUROC, AP, precision@k with pair-level bootstrap.
7. **Baselines.** TWOSIDES scores for the overlapping pairs; single-drug ROR of the more toxic drug (to confirm the interaction term adds information).

## Evaluation & statistics

- Primary endpoint: AUROC of each score against the ONC + mechanism reference set (H1); secondary: AP, recall of role-code-3 pairs at 5 % FDR (H3).
- Sex effects: LRT on the three-way term, BH at q ≤ 0.05 within event class; report ratio of IORs with 95 % CI; require n111 ≥ 3 in both sexes.
- Confounding checks: stratify by age band and reporter type (`primarysource.qualification`); re-run with suspect-only roles; exclude reports where either drug is the only suspect (indication channelling proxy).
- Nulls: permutation of sex labels within (pair, event) strata → null distribution of the three-way statistic; permutation of drug B labels across reports → empirical null for Ω/IOR to calibrate thresholds; negative-control pairs (e.g., topical + ophthalmic drugs) should yield no signals.
- Leakage: reference sets are compiled from labels/guidelines and never from FAERS-derived resources (TWOSIDES is a comparator, not a label). Role-code-3 labels are used only for recall.
- Reporting follows READUS-PV (Fusaroli et al., 2024).

## Publishable angle

- **Headline result.** "A mechanism-pooled hierarchical model outperforms Ω and IOR for detecting DDIs in FAERS, and X % of DDI signals differ by sex, including stronger QT-class interaction signals in women" — released with a sex-stratified DDI signal table (a 2015–2026 successor to TWOSIDES).
- **Venues.** *Drug Safety*; *Clinical Pharmacology & Therapeutics*; *Pharmacoepidemiology and Drug Safety*; *Journal of Biomedical Informatics* (methods/resource); *JAMIA* for the resource release.
- **Follow-ups.** Three-drug interactions (extension of the 2025 high-order-interaction preprint) with the same hierarchical prior; validation of the top sex-specific signals in EHR data (see related project `faers-signal-ehr-validation`); label-mining to check which sex-specific interactions are already in the `drug_interactions` section of SPL labels (see `unlabeled-adverse-event-mining`).

## Risks, confounds & mitigations

- **Confounding by indication / co-prescription.** Pairs co-prescribed for one condition (e.g., HIV regimens) create event enrichment without interaction. Mitigation: indication-stratified backgrounds using `drugindication`; sensitivity analysis excluding fixed-dose combinations and same-class pairs.
- **Sex imbalance in the drug-specific reporting population** (Simpson's paradox). Mitigation: always compute sex-specific backgrounds; report pooled vs stratified discordance (H5).
- **Masking and competition bias** when one drug dominates an event. Mitigation: remove reports of the dominant drug from the background (masking-unmasking analysis).
- **Duplicates and stimulated reporting** (litigation waves, media). Mitigation: ASCII-based deduplication; exclude lawyer-reported cases in sensitivity; interrupted time-series flags around FDA safety communications (borrow from `faers-reporting-bias`).
- **Name harmonisation errors** (≈15–20 % of drug entries lack `openfda` fields). Mitigation: DiAna/RxNorm mapping; report coverage; restrict the primary analysis to mapped entries.
- **Reference-set incompleteness**: absence from ONC ≠ no interaction. Mitigation: only use explicit negatives (unrelated classes, DrugBank-negative), and report recall-only metrics for partial labels.
- **Ω no-interaction model ambiguity.** Four models are implemented; H2 makes the choice an explicit result rather than an assumption.

## Milestones

- [ ] Bulk FAERS pull, deduplication, name harmonisation; coverage report
- [ ] Reference sets assembled (ONC, CredibleMeds, FDA CYP, role-code-3 partial labels, negatives)
- [ ] Pair/event tables for the 300-drug panel (pooled, female, male)
- [ ] Ω (4 models), IOR, RERI screens with BH; method-ranking figure (H1, H2)
- [ ] Mechanism-pooled hierarchical model (EB, then full Bayesian) (H1)
- [ ] Sex-specific interaction screen and QT/statin/bleeding class analyses (H4, H5)
- [ ] Permutation nulls, confounding sensitivity analyses
- [ ] Resource release (parquet + web table), READUS-PV-compliant manuscript

## Ethics / data-use notes

- FAERS/openFDA data are public and de-identified; no IRB is normally required for secondary analysis, but check local policy. Do not attempt re-identification (e.g., via free-text narratives, which openFDA does not expose).
- Signals are hypotheses, not causal claims; the manuscript must state this per READUS-PV and FDA's openFDA disclaimer.
- Never commit downloaded data or the API key; `data/` is git-ignored and the key is read from `OPENFDA_API_KEY`.
- MedDRA terms are licensed; do not redistribute the hierarchy, only PT names as they appear in FAERS.

## Related projects in this repository

`faers-reporting-bias` (reporter-type and stimulated-reporting adjustment), `faers-signal-ehr-validation` (EHR validation of signals), `unlabeled-adverse-event-mining` (label expectedness). This project is self-contained and does not import from them.

## Quick start (starter code)

```bash
cd projects/faers-ddi-signals
pip install -r requirements.txt
export OPENFDA_API_KEY=...                      # optional
python scripts/download_data.py --sample        # ~30 requests, data/raw/*.jsonl
PYTHONPATH=src python -m pytest -q tests        # synthetic-data tests
```

```python
from faers_ddi import read_jsonl, index_reports, candidate_pairs, pair_event_table, screen, sex_stratified_tables, sex_specific_interaction
reports = read_jsonl("data/raw/pair_clarithromycin_simvastatin.jsonl") + read_jsonl("data/raw/drug_simvastatin.jsonl")
tab = pair_event_table(reports, [("CLARITHROMYCIN", "SIMVASTATIN")], min_n111=3)
res = screen(tab, omega_model="independence")          # omega, omega025, ior, reri, q_ior
st = sex_stratified_tables(reports, [("CLARITHROMYCIN", "SIMVASTATIN")], min_n111=3)
```

Module map: `openfda.py` (client, `flatten_record` with per-role drug lists), `ddi_tables.py` (n_ijk cells, pair enumeration, sex strata), `ddi_stats.py` (Omega with four no-interaction models, IOR, RERI, three-way sex interaction, empirical-Bayes hierarchical shrinkage, BH, `screen`), `reference_sets.py` (curated/mechanism/reporter-flagged sets, AUROC/AP evaluation).

## Key variables and cohort definitions

| Variable | Definition | Source field(s) |
|---|---|---|
| Report | one FAERS case (latest version per `CASEID`; openFDA `safetyreportid` otherwise) | `safetyreportid`, ASCII `CASEID`/`CASEVERSION` |
| Drug A / B present | harmonised generic name listed in any role (primary); suspect-only (sensitivity) | `patient.drug.openfda.generic_name`, `medicinalproduct`, `drugcharacterization` |
| Interacting flag | reporter coded the drug as interacting (partial positive label) | `drugcharacterization = 3` / ASCII `ROLE_COD = I` |
| Event | MedDRA PT (primary) or HLT class for QT, bleeding, serotonin syndrome, myopathy | `patient.reaction.reactionmeddrapt` |
| Sex stratum | female / male; unknown excluded from stratified analyses | `patient.patientsex` (2 / 1) |
| Age band | < 18, 18–64, ≥ 65 (unit-converted) | `patientonsetage`, `patientonsetageunit` |
| Reporter type | physician / pharmacist / other HCP / lawyer / consumer | `primarysource.qualification` |
| Window | receipt date 2015-01-01 to 2026-06-30 | `receivedate` |
| Mechanism group | CYP3A4-inh × substrate, CYP2D6, P-gp, QT+QT, serotonergic, anticoagulant+antiplatelet, none | FDA DDI tables, CredibleMeds, ONC list |
| n111 minimum | 3 reports on A+B+E (pooled); 3 in each sex for the sex screen | — |
