# drug-shortage-ae-patterns

**Do drug shortages leave a measurable footprint in FAERS? A staggered difference-in-differences panel of US shortage episodes (2012-2025) testing whether shortage onset raises medication-error and dosing-related adverse-event reports for the shortage drug and for its therapeutic substitutes.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (openFDA client with count/paginated queries, shortage-episode and drug-month panel builder, Poisson event-study and Callaway-Sant'Anna-style estimators with cluster bootstrap and placebo permutation).
- Difficulty: MSc thesis / first PhD paper. Causal-inference and pharmacovigilance heavy; no GPU.
- Timeline: 6-9 months (1-2 months exposure list curation + name harmonisation, 1 month FAERS counts, 2-3 months estimation and robustness, 1-2 months writing).
- Compute: laptop. The design uses openFDA `count` queries (a few thousand small HTTP calls) rather than downloading all ~25M FAERS reports; the optional full-panel rebuild from FAERS quarterly ASCII files needs ~30 GB disk and a few CPU-hours.

## Background

US drug shortages are chronic and worsening: ASHP counted an all-time high of active shortages in early 2024, dominated by sterile injectables (chemotherapy, anaesthetics, IV fluids, ADHD stimulants, GLP-1 agonists). The standard mitigation is substitution: a different concentration, vial size, salt, route, or a different molecule in the same class. Substitution is exactly the situation in which handling errors occur: unfamiliar concentration (heparin, insulin), unfamiliar dosing units (semaglutide "clicks" vs mg, compounded vials), and unfamiliar drugs for the indication (norepinephrine -> phenylephrine). Case reports and surveys (McLaughlin et al., 2013, J Manag Care Pharm; Fox, Sweet & Jensen, 2014, Mayo Clin Proc) document these harms, and single-shortage outcome studies exist (Vail et al., 2017, JAMA, norepinephrine shortage and septic-shock mortality; Metzger, Billett & Link, 2012, NEJM, mechlorethamine in paediatric Hodgkin lymphoma). What is missing is a population-scale, multi-episode estimate of how much excess *error-type* adverse-event reporting a shortage causes, for whom, and whether it reverses when the shortage resolves.

FAERS is the only public data source that (a) codes medication errors as MedDRA terms (HLGT "Medication errors and other product use errors and issues": "Wrong drug administered", "Incorrect dose administered", "Product substitution issue", "Wrong product administered", "Accidental overdose", "Incorrect route of product administration", ...), (b) covers every marketed drug, and (c) has a monthly time axis long enough to span hundreds of shortage episodes. Since 2024 openFDA also exposes the FDA Drug Shortage Database as an API, giving machine-readable exposure timing.

## The research gap

**What has been done**

- A single-drug synthetic-control study of the 2017 hurricane-driven heparin shortage found medication-error reports in FAERS rose by roughly 150% for heparin and roughly 110% for the substitute enoxaparin (Manufacturing & Service Operations Management, 2024, doi:10.1287/msom.2023.0297). This is the closest precedent and shows the signal is detectable, but it is one episode with one hand-picked substitute.
- FAERS analyses of GLP-1 receptor agonists (2024-2025, e.g. Frontiers in Pharmacology 2025; tirzepatide FAERS 2022-2025 analyses) report that dosing and administration-error reports rose sharply from late 2022, coinciding with the semaglutide/tirzepatide shortage and compounding era; these are descriptive, not comparative.
- The shortage-outcomes literature (Vail et al., 2017; Metzger et al., 2012; Hantel and colleagues' oncology shortage work) is single-episode and uses institutional or claims data, not spontaneous reports.
- Methodologically, the difference-in-differences literature now has estimators robust to staggered adoption and heterogeneous effects (Callaway & Sant'Anna, 2021, J Econometrics; Sun & Abraham, 2021, J Econometrics; Goodman-Bacon, 2021, J Econometrics; de Chaisemartin & D'Haultfoeuille, 2020, Am Econ Rev), which have not been applied to pharmacovigilance count panels.
- Journals now treat FAERS disproportionality papers with suspicion because of paper-mill flooding (Retraction Watch, Sept 2025; Frontiers in Pharmacology author guidelines 2025). A design with an external exposure, a pre-registered panel and causal estimators is precisely what distinguishes a credible FAERS study.

**What is missing (checked against 2023-2026 literature)**

1. No study treats the *set* of shortage episodes as staggered treatments and estimates an average and heterogeneous effect on error-type reporting with modern DiD estimators, event-study dynamics (anticipation, onset, resolution) and cluster-robust inference.
2. No study systematically defines substitutes (same Established Pharmacologic Class and route, from openFDA `pharm_class_epc`) and estimates spillover effects on substitute drugs' error and overdose reporting.
3. No study uses within-FAERS negative-control outcomes and the total-report denominator to separate substitution errors from stimulated reporting (media attention to a shortage could raise reporting of everything).
4. No study exploits the shortage-list fields (reason: manufacturing/quality vs demand; presentation: injectable vs oral; therapeutic category) as effect modifiers, nor validates exposure timing against the MedDRA product-supply terms reported in FAERS itself.

## Research questions / hypotheses

1. **H1 (index-drug effect).** Shortage onset increases the monthly rate of medication-error reports (as a share of all reports naming the drug) for the index drug. Pre-registered effect size of interest: incidence-rate ratio (IRR) >= 1.2 over months 0-6 after onset relative to not-yet-shortage months, with no pre-trend (event-study leads jointly null).
2. **H2 (substitute spillover).** Drugs in the same EPC class and route as the index drug show an increase in medication-error and dosing-error reports (PTs "Incorrect dose administered", "Accidental overdose", "Overdose", "Underdose", "Wrong technique in product usage process") after the index shortage onset. Prediction: IRR in [1.1, 1.5], larger when the index drug has few substitutes.
3. **H3 (heterogeneity).** Effects are larger for sterile injectables than oral solids, for narrow-therapeutic-index drugs (curated list: heparin, insulin, digoxin, warfarin, opioids, vasopressors, chemotherapy, anticonvulsants), and for shortages whose listed reason is manufacturing/quality problems (abrupt) rather than increased demand (gradual).
4. **H4 (reversal).** Reports return toward baseline within 6 months of the shortage being marked resolved (symmetric event study around resolution).
5. **H5 (specificity / exposure validity).** (a) Negative-control outcomes unrelated to handling (e.g. PTs "Alopecia", "Rash", "Nausea") show IRR ~ 1 after onset; (b) the PTs in the MedDRA product supply/availability group rise sharply at onset (exposure validation); (c) all-cause report counts rise less than error-type counts (ratio-of-IRR > 1), which argues against pure stimulated reporting.
6. **H6 (serious outcomes).** Error reports during shortage months are not less likely to be serious (hospitalisation/death) than error reports outside shortage months, i.e. the excess is not confined to trivial near-misses.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| openFDA Drug Shortages API (FDA CDER Drug Shortage Database) | Exposure: generic name, presentation, status (current/resolved/discontinued), initial posting date, update dates, shortage reason, therapeutic category, company | ~hundreds of current + resolved records; history back to roughly 2012 on the FDA web database | Open (optional `OPENFDA_API_KEY` raises rate limits) | https://api.fda.gov/drug/shortages.json ; https://www.accessdata.fda.gov/scripts/drugshortages/ |
| ASHP / University of Utah Drug Information Service shortage list | Secondary exposure list (broader than FDA's: includes shortages FDA does not list), start dates, resolution dates | ~1,500+ episodes since 2011 | Public web pages; bulk research extracts by request to ASHP/UUDIS (respect terms of use) | https://www.ashp.org/drug-shortages |
| FAERS via openFDA `drug/event` | Outcome counts: reports per drug per month; medication-error, dosing, supply-issue and negative-control PTs; seriousness; reporter type | ~25M reports (2004-), monthly | Open | https://api.fda.gov/drug/event.json |
| FAERS quarterly ASCII/XML files | Optional exact rebuild of the panel with de-duplication by case ID and MedDRA version control | ~1-2 GB/quarter | Open | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| openFDA `drug/ndc` and `drug/label` (`openfda.pharm_class_epc`, `route`, `dosage_form`) | Substitute definition (same EPC + route), injectable flag | ~100k NDC products | Open | https://api.fda.gov/drug/ndc.json |
| RxNorm API (NLM) | Ingredient normalisation (salt stripping, brand -> ingredient) | - | Open | https://rxnav.nlm.nih.gov |
| CMS Medicaid State Drug Utilization Data (SDUD) | Optional utilisation denominator (prescriptions per quarter per NDC) to test whether shortage months truly had fewer dispensings | ~quarterly files, 2012- | Open | https://data.medicaid.gov |

No credentialed data are involved.

## Methods

1. **Exposure table** (`scripts/download_data.py --shortages`; `src/shortage_ae/shortage_panel.py`): pull all shortage records from openFDA; parse `initial_posting_date` (onset), status and `update_date` (resolution when status = Resolved); collapse multiple presentations of one ingredient into one episode per ingredient (earliest onset, latest resolution); normalise names with `normalize_name` (salt/dosage-form stripping) and, optionally, RxNorm. Merge the ASHP list for a sensitivity analysis with the broader exposure set.
2. **Outcome counts** (`openfda_client.OpenFDAClient.count`): for each ingredient in the union of (shortage drugs, their substitutes, never-shortage control drugs matched on EPC and report volume), query `count=receivedate` with search filters for (a) all reports naming the drug as suspect, (b) reports with any medication-error PT, (c) dosing-error PTs, (d) supply-issue PTs, (e) negative-control PTs, (f) serious reports. Aggregate daily counts to months. This needs one call per (drug, outcome group) and avoids the 25,000-record `skip` ceiling.
3. **Panel** (`build_panel`): drug x month long table with `n_total`, `n_error`, ..., `treated` (in-shortage indicator), `cohort` (first onset month), `rel_time` (months since onset), `resolved_rel_time`, and modifiers (injectable, NTI, reason, category).
4. **Estimation** (`staggered_did.py`):
   - Event-study Poisson regression with drug and month fixed effects, offset log(`n_total`) (share of reports that are errors) or no offset (raw counts), leads/lags -12..+12 binned at the ends, reference month -1. Clustered standard errors by drug.
   - Callaway-Sant'Anna-style group-time ATTs on log rates using not-yet-treated drugs as controls, aggregated to dynamic (event-time) and overall effects; drug-cluster bootstrap CIs.
   - Placebo permutation: reassign onset dates across drugs (preserving the calendar distribution) to obtain a null for each aggregate.
   - Substitute spillover: same design with the substitute's outcome and the index drug's onset as treatment; drugs that are both index and substitute are handled by excluding overlapping windows.
5. **Validation of exposure timing**: change-point detection (PELT via `ruptures`, optional) on the supply-issue PT series per drug; compare detected change points with listed onset dates (median lag, share within +/-2 months).
6. **Denominator sensitivity**: repeat with Medicaid SDUD prescriptions as an offset for the subset of drugs with reliable NDC mapping.
7. **Tools**: `requests`, `pandas`, `numpy`, `statsmodels` (GLM Poisson, cluster-robust covariance), `scipy`; optional `ruptures`, `linearmodels`.

## Evaluation & statistics

- Primary estimand: dynamic ATT on log monthly error-report rate (offset by all reports of the drug), horizons 0-6 months after onset; secondary: horizons 7-12, resolution horizons, substitute spillover.
- Inference: cluster-robust (drug) SEs; drug-cluster bootstrap (999 resamples) for CS aggregates; placebo permutation p-values (999 permutations) for the overall ATT.
- Pre-trend test: joint Wald test on leads -12..-2; report the event-study plot with simultaneous confidence bands (sup-t).
- Leakage/contamination: drugs that are substitutes of a shortage drug are removed from the never-treated control pool; drugs with overlapping/repeated episodes use the first episode for the main analysis and all episodes in a sensitivity analysis.
- Multiple comparisons: H1-H4 are confirmatory with a fixed order (H1 first); modifiers and PT-level breakdowns use Benjamini-Hochberg (q = 0.05).
- Nulls: placebo onset permutation; negative-control outcomes; "placebo drugs" (matched never-shortage drugs assigned the index drug's onset).
- Robustness: FAERS quarterly-file rebuild with case de-duplication; exclude the 2019-2021 pandemic months; exclude GLP-1s (dominant recent episode); exclude reports from lawyers; vary the error PT set (HLGT vs curated PT list); ASHP vs FDA exposure list.

## Publishable angle

- **Headline**: "Across N shortage episodes (2012-2025), shortage onset increased medication-error reporting for the index drug by X% (95% CI) and for same-class substitutes by Y%, concentrated in sterile injectables and narrow-therapeutic-index drugs; effects reversed within Z months of resolution and were absent for negative-control outcomes." A second, methodological contribution is a reusable staggered-DiD template for external-exposure pharmacovigilance panels.
- Target venues: *Drug Safety*; *Pharmacoepidemiology and Drug Safety*; *JAMA Network Open* or *JAMA Health Forum* (policy angle); *Clinical Pharmacology & Therapeutics*; *Health Affairs Scholar*.
- Follow-ups: (i) link episodes to openFDA `drug/enforcement` recalls to separate quality-driven from demand-driven shortages; (ii) heterogeneous effects with causal forests on drug/episode features; (iii) NLP on FAERS narratives (only available in the full FAERS files by FOIA, not openFDA) to classify substitution mechanisms; (iv) European replication with EudraVigilance line listings and EMA shortage catalogue.

## Risks, confounds & mitigations

- **Stimulated reporting** (media, FDA communications) could raise all reports during shortages. Mitigation: total-report offset, negative-control outcomes, ratio-of-IRR test (H5c), and exclusion of reports citing literature/media sources.
- **Exposure misclassification**: FDA and ASHP lists differ in coverage and timing; onset date on the list may lag the actual supply disruption. Mitigation: two lists, change-point validation against supply-issue PTs, and window sensitivity (+/-2 months).
- **Name harmonisation**: FAERS `medicinalproduct` is free text; openFDA's `openfda.generic_name` covers most but not all reports. Mitigation: RxNorm normalisation; report coverage rates; sensitivity with brand + generic search.
- **Staggered-treatment bias**: two-way FE Poisson with heterogeneous effects can be biased (Goodman-Bacon, 2021). Mitigation: CS-style estimator is primary; TWFE is secondary.
- **Repeated / overlapping shortages** of the same ingredient. Mitigation: first-episode analysis; episode-level clustering.
- **Low counts** for rare drugs. Mitigation: pre-specify minimum baseline volume (>= 20 reports/month median) for the main analysis; pooled rare-drug analysis as secondary.
- **openFDA limits**: `skip` caps at 25,000 records; count queries return at most 1,000 buckets. Mitigation: use count queries by date and split large record pulls by date windows (implemented).
- **MedDRA version drift** (PT renames). Mitigation: fix PT lists at the analysis date; use HLGT-level grouping from the FAERS ASCII rebuild.

## Milestones

- [ ] Freeze exposure list (FDA + ASHP), harmonised ingredient names, episode table with modifiers.
- [ ] Define substitute sets from EPC + route; define matched never-shortage controls.
- [ ] Pull all count series (monthly) for drugs x outcome groups; QC coverage.
- [ ] Pre-register hypotheses, PT lists, windows and estimators (OSF).
- [ ] Event-study and CS estimates for H1; placebo permutation; pre-trend checks.
- [ ] Spillover (H2), heterogeneity (H3), resolution (H4), specificity (H5), seriousness (H6).
- [ ] FAERS quarterly-file rebuild robustness; Medicaid denominator sensitivity.
- [ ] Preprint, code + derived monthly panel release.

## Ethics / data-use notes

- FAERS and shortage data are public and de-identified. Do not attempt to re-identify patients or reporters; do not combine with narratives obtained by FOIA in ways that could identify individuals.
- openFDA terms: cite openFDA and FDA as the data source; the API key (`OPENFDA_API_KEY`) is personal and must stay in the environment, never in code or commits.
- ASHP shortage pages are copyrighted; bulk extraction should follow their terms or use a data agreement.
- Never commit downloaded data; `data/` is git-ignored. Pre-register the analysis to distinguish this work from the flood of low-quality FAERS disproportionality papers.

## Quick start

```bash
pip install -r requirements.txt
export OPENFDA_API_KEY=...                      # optional
python scripts/download_data.py --sample         # 100 shortage records + count series for 5 demo drugs
PYTHONPATH=src pytest -q tests                   # synthetic-data tests (no network)
```

```python
import pandas as pd
from shortage_ae import (OpenFDAClient, build_faers_search, MEDICATION_ERROR_PTS, episodes_from_openfda,
                         daily_counts_to_monthly, build_panel, callaway_santanna_att, aggregate_att)
from shortage_ae.openfda_client import records_to_frame

client = OpenFDAClient()
episodes = episodes_from_openfda(records_to_frame(client.shortage_records()))
months = pd.date_range("2012-01-01", "2025-06-01", freq="MS")
counts = {}
for drug in ["heparin", "norepinephrine", "lorazepam"]:
    counts[drug] = {
        "all": daily_counts_to_monthly(client.count_by_date("drug/event", build_faers_search(drug)), months),
        "error": daily_counts_to_monthly(client.count_by_date("drug/event", build_faers_search(drug, MEDICATION_ERROR_PTS)), months),
    }
panel = build_panel(episodes, counts, months)
att = callaway_santanna_att(panel, horizons=range(0, 13), n_boot=500)
print(aggregate_att(att, range(0, 7)))           # IRR over months 0-6 after shortage onset
```

## Pre-specified operational definitions

| Item | Definition (frozen before estimation) |
|---|---|
| Unit | ingredient (normalised generic name) x calendar month, 2012-01 to the last complete FAERS quarter |
| Exposure onset | month of the earliest `initial_posting_date` across presentations of the ingredient on the FDA list; ASHP "first reported" date in the sensitivity list |
| Exposure end | month of the last `update_date` with status Resolved when all presentations are resolved; else censored (still current) |
| Index outcome | reports with the ingredient as primary suspect (`drugcharacterization:1`) and >= 1 PT in `MEDICATION_ERROR_PTS` |
| Spillover outcome | same, for each substitute (same EPC class and route, from openFDA `drug/ndc`), with `DOSING_ERROR_PTS` |
| Denominator | all primary-suspect reports of the ingredient in the month (`n_all`); Medicaid SDUD prescriptions in the sensitivity analysis |
| Controls | never-shortage ingredients matched 2:1 on EPC class and median monthly report volume (2012-2015), excluding substitutes |
| Estimator | Callaway-Sant'Anna group-time ATT on log rates with not-yet-treated controls, dynamic aggregation; TWFE Poisson event study as secondary |
| Inference | drug-cluster bootstrap (999), placebo onset permutation (999), joint pre-trend Wald test on leads -12..-2 |
| Modifiers | injectable (presentation text), narrow-therapeutic-index list, listed shortage reason, therapeutic category, number of substitutes |
| Minimum volume | median >= 20 primary-suspect reports/month over the pre-period for the main analysis; pooled rare-drug analysis otherwise |
| Exclusions | reports with `primarysource.qualification` = lawyer; reports flagged as literature; duplicates by `safetyreportid` |

Effect sizes are reported as IRRs with 95% CIs on both the log-rate (CS) and count (Poisson) scales, and every estimate is accompanied by its placebo-permutation p-value.

## Related projects

- `faers-reporting-bias` (reporter-type and stimulated-reporting bias models; the interrupted-time-series helpers there are complementary to the staggered-DiD estimators here).
- `unlabeled-adverse-event-mining` (label-vs-FAERS comparisons; substitution errors are by definition not in the substitute's label).
