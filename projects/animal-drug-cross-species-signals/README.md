# animal-drug-cross-species-signals

**Dogs and cats as real-world pharmacovigilance sentinels: cross-species concordance, discordance and lead-lag of adverse-event signals for the active ingredients shared between openFDA's veterinary (CVM) and human (FAERS) spontaneous-reporting systems.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (openFDA animal/veterinary client with date-windowed paging and flattening, ingredient normalisation, VeDDRA/MedDRA organ-system harmonisation layer with fuzzy PT matching, BCPNN signal tables per species, concordance/lead-lag statistics with ingredient-level bootstrap, mg/kg dose parsing).
- Difficulty: MSc thesis to first PhD paper (pharmacovigilance + comparative pharmacology + terminology work). No GPU.
- Timeline: 6-9 months (2 months harvest and harmonisation, 2 months signal tables and validation sets, 2 months modelling, 1-2 months writing).
- Compute: laptop. The CVM database is ~1.3M reports (a few GB of JSON); FAERS is used via count queries.

## Background

openFDA exposes the FDA Center for Veterinary Medicine adverse-event database (`animalandveterinary/event`): about 1.3 million reports from 1987 to 2025, roughly 985k in dogs and 148k in cats, with species, breed, sex, age, weight, active ingredients with dose, route, VeDDRA-coded reactions, outcomes and reporter type. A large share of the ingredients are human drugs used under AMDUCA extra-label provisions or approved for both species: NSAIDs (meloxicam), anticonvulsants (phenobarbital, levetiracetam, zonisamide, gabapentin), psychotropics (fluoxetine, clomipramine, trazodone), endocrine drugs (levothyroxine, methimazole, insulin), cardiovascular drugs (enalapril, amlodipine, furosemide, spironolactone), immunosuppressants (cyclosporine, azathioprine, mycophenolate), antimicrobials, antifungals, chemotherapeutics (doxorubicin, vincristine, cyclophosphamide, chlorambucil), opioids and sedatives. Some classes went the other way: the JAK inhibitor oclacitinib (dogs, 2013) parallels tofacitinib; the anti-NGF antibodies bedinvetmab and frunevetmab (2022-2023) parallel tanezumab, whose human programme was halted over joint safety, and musculoskeletal adverse events in dogs receiving bedinvetmab are now a live regulatory topic (commentaries, 2025).

Concordance between animal and human toxicity has been studied for *preclinical* laboratory studies (Olson et al., 2000, Regul Toxicol Pharmacol, ~71% concordance; Clark & Steger-Hartmann, 2018, Regul Toxicol Pharmacol; Monticello et al., 2017, Toxicol Appl Pharmacol, the IQ consortium database). Whether *real-world* spontaneous reports in companion animals, with chronic dosing, comorbidity and owner-reported outcomes, carry information about human risks (or vice versa) has not been quantified, even though both data sources are open and cover the same molecules.

## The research gap

**What has been done**

- Descriptive veterinary pharmacovigilance: narrative reviews of veterinary PV situations (2024, PMC11600972); species-specific analyses (gentamicin ototoxicity in dogs; isoxazoline neurological events); bedinvetmab musculoskeletal-event commentaries (2025).
- Human misuse of veterinary drugs analysed in FAERS (Toxics, 2024, 21 drugs, ~39k events): a human-side view of shared products, not a cross-species comparison.
- Machine-learning safety profiling on the CVM database (arXiv 2510.01520, 2025) and unsupervised pattern analysis of Japanese veterinary toxicology data with a cross-species risk-assessment framing (arXiv, 2026): single-system analyses.
- Terminology: VeDDRA (EMA/VICH) and MedDRA are separate dictionaries; practitioners query FAERS and the CVM database separately and no open, validated crosswalk exists (industry commentary on the human/veterinary data-standards divide).
- Preclinical-to-clinical concordance literature (Olson 2000; Clark & Steger-Hartmann 2018) uses controlled toxicology studies in healthy young animals, not spontaneous reports.

**What is missing (checked against 2023-2026 literature)**

1. A quantitative, ingredient-wide comparison of disproportionality signals between CVM (dog, cat) and FAERS (human) at a harmonised organ-system level, with a released VeDDRA-MedDRA organ-system crosswalk and PT-level mapping table.
2. Tests of whether known species-specific pharmacology (feline glucuronidation deficiency and acetaminophen; canine MDR1 and macrocyclic lactones; canine NSAID gastro-intestinal susceptibility) is recovered as *discordance*, and shared pharmacodynamics (sedation with gabapentinoids, hypoglycaemia with insulin, bleeding with anticoagulants) as *concordance* - i.e. positive controls for both directions.
3. A lead-lag ("sentinel") analysis on quarterly signal trajectories: for mechanism-matched pairs, does the first sustained signal in one species precede the other, and by how much?
4. Use of the CVM dose and body-weight fields to compute mg/kg exposure and test whether allometric dose differences explain exposure-driven discordance.

## Research questions / hypotheses

1. **H1 (concordance).** Across shared ingredient x organ-system cells (ingredients with >= 100 reports in each system), BCPNN IC in dogs and IC in humans are positively correlated (Spearman rho >= 0.3, ingredient-cluster bootstrap CI excluding 0), and binary signal flags (IC025 > 0) agree beyond chance (Cohen's kappa > 0.2).
2. **H2 (mechanism dependence).** Concordance is higher for pharmacodynamically mediated organ systems (gastro-intestinal, nervous system/behavioural, cardiovascular, metabolic) than for immune/idiosyncratic ones (skin/hypersensitivity, hepatic, haematological): test the difference of rho by organ-system group with a permutation test.
3. **H3 (positive controls of discordance).** Pre-specified species-specific pairs (cat-acetaminophen-haematological; dog-ivermectin-neurological; dog-NSAID-GI ulceration; cat-permethrin-neurological) show signal in the animal species but not (or weaker) in humans; pre-specified shared-pharmacology pairs signal in both. Report as a 2 x 2 validation with exact tests.
4. **H4 (lead-lag).** Among mechanism-matched pairs signalling in both systems, the animal signal precedes the human signal by a median >= 4 quarters when the animal product was marketed first or concurrently, and the reverse when the human product was marketed first (trajectory analysis with first-sustained-signal quarters).
5. **H5 (dose scaling).** In dog reports with parsable dose and weight, reported mg/kg relative to the labelled mg/kg predicts seriousness (OR per doubling >= 1.3); for exposure-driven organ systems, allometrically scaled human vs dog dose ratios explain part of the IC difference (interaction term p < 0.05).
6. **H6 (robustness).** Concordance is stable when excluding manufacturer-submitted reports, "lack of expected effectiveness" reports, and reports coded before VeDDRA v3 (term-set change).

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| openFDA Animal & Veterinary Adverse Events (`animalandveterinary/event`) | All fields: `animal.species/breed/gender/age/weight`, `drug[].active_ingredients[].name/dose`, `drug[].route/brand_name/used_according_to_label`, `reaction[].veddra_term_name/code/version`, `outcome[]`, `serious_ae`, `primary_reporter`, `original_receive_date`, `onset_date` | ~1.3M reports (1987-2025) | Open (optional `OPENFDA_API_KEY`) | https://api.fda.gov/animalandveterinary/event.json |
| FAERS via openFDA `drug/event` | Human counts per ingredient x PT x quarter (count queries) | ~25M reports | Open | https://api.fda.gov/drug/event.json |
| VeDDRA term lists (EMA / VICH) | VeDDRA LLT/PT/HLT/SOC hierarchy for organ-system grouping | ~3k terms | Open (EMA publishes the combined list) | https://www.ema.europa.eu (search "VeDDRA") |
| MedDRA | PT -> SOC hierarchy for the human side | ~25k PTs | Licence (free for academic/non-profit via MSSO) | https://www.meddra.org |
| Animal Drugs @ FDA / Green Book | Veterinary approval dates, species, labelled doses | ~1k applications | Open (downloadable data files) | https://animaldrugsatfda.fda.gov ; https://www.fda.gov/animal-veterinary/products/approved-animal-drug-products-green-book |
| openFDA `drug/drugsfda` | Human approval dates and application numbers | ~25k | Open | https://api.fda.gov/drug/drugsfda.json |
| RxNorm API | Ingredient normalisation | - | Open | https://rxnav.nlm.nih.gov |
| DrugBank (optional) | Targets/mechanism classes for H2 grouping | - | Free academic licence | https://go.drugbank.com |

No credentialed data are involved.

## Methods

1. **Harvest** (`scripts/download_data.py`): page the CVM endpoint by `original_receive_date` windows (30-day windows keep every window under the 25,000-record `skip` ceiling), flatten one row per report-drug-reaction (`xspecies_pv.openfda_animal.flatten_animal_record`), store JSONL/parquet. FAERS counts per ingredient x PT x quarter with count queries.
2. **Ingredient harmonisation** (`term_mapping.normalize_ingredient` + RxNorm): strip salts/esters, map brands to ingredients; shared set = ingredients with >= 100 reports in both systems (expected ~150).
3. **Reaction harmonisation** (`term_mapping.py`): both VeDDRA and MedDRA terms are assigned to ~22 harmonised organ-system buckets by a rule-based keyword classifier (dictionary-licence independent), then refined with the official SOC hierarchies where available; PT-level fuzzy candidates (`suggest_pt_matches`) are produced for manual review of the top 300 VeDDRA terms (two reviewers, kappa reported). Deliverable: an open crosswalk table.
4. **Signal tables** (`cross_species_signals.signal_table`): per species, per ingredient x bucket cumulative 2x2 table and BCPNN IC/IC025 (Norén et al., 2013); FAERS uses primary-suspect drugs only; CVM uses all listed actives (sensitivity: first-listed only).
5. **Concordance** (`concordance`): Spearman rho on IC, kappa on flags, per-bucket breakdown; ingredient-cluster bootstrap and label-permutation null (shuffle ingredient identity in one species).
6. **Lead-lag** (`lead_lag`): quarterly cumulative IC per species; first sustained IC025 > 0 (2 consecutive quarters); difference in quarters; mixed model with mechanism-match and marketing-order covariates.
7. **Dose analysis** (`mg_per_kg`): parse dose numerator/unit and animal weight; compare with labelled dose from the Green Book; logistic regression of `serious_ae` on log dose ratio with ingredient random intercepts (statsmodels `BinomialBayesMixedGLM` or GEE).
8. **Tools**: `requests`, `pandas`, `numpy`, `scipy`, `statsmodels`, `rapidfuzz` (optional; difflib fallback), `pyarrow`.

## Evaluation & statistics

- Primary estimands: Spearman rho and kappa between species (H1) with 95% ingredient-cluster bootstrap CIs; permutation p-values (999 shuffles of ingredient labels within the human table).
- Positive/negative control sets (H3) pre-registered before running signal tables; report sensitivity/specificity of "animal signal predicts human signal" against the control sets.
- Multiple comparisons: bucket-level and ingredient-level tests with Benjamini-Hochberg (q = 0.05); H1-H4 confirmatory in fixed order.
- Leakage: harmonisation dictionaries are built from term lists, never from the outcome (no tuning of the crosswalk on concordance); reviewers are blind to signal values.
- Denominators: each system's own totals; sensitivity with per-ingredient report-volume matching (only ingredient x bucket cells with >= 5 reports in both).
- Time: signals computed on data received up to each quarter end; VeDDRA version changes recorded as covariates.

## Publishable angle

- **Headline**: "Across N shared ingredients, canine and human spontaneous-report signals agree at rho = X (kappa = Y) at the organ-system level; concordance is concentrated in pharmacodynamic organ systems, known species-specific toxicities are recovered as discordance, and in mechanism-matched pairs the earlier-marketed species' signal precedes the other by a median of Z quarters" - plus the first open VeDDRA-MedDRA organ-system crosswalk with a reviewed PT-level mapping.
- Target venues: *Drug Safety*; *Journal of Veterinary Pharmacology and Therapeutics*; *Pharmacoepidemiology and Drug Safety*; *Frontiers in Veterinary Science* (Veterinary Pharmacology); *One Health* (Elsevier).
- Follow-ups: cats and horses as separate sentinel species; EudraVigilance Veterinary replication; drug-drug interaction concordance (TWOSIDES-style) across species; anti-NGF and JAK-inhibitor case studies with trajectory modelling; breed as a genetic stratifier (MDR1 breeds).

## Risks, confounds & mitigations

- **Terminology mismatch**: VeDDRA and MedDRA granularity differ; owner-reported animal signs (vomiting, lethargy, anorexia) dominate. Mitigation: organ-system level primary analysis; PT-level secondary with reviewed mapping; report term-coverage rates.
- **Reporting structure**: CVM reports are mostly manufacturer-forwarded; FAERS has consumer/lawyer reports. Mitigation: reporter-type stratification; H6 exclusions.
- **Indication and co-medication confounding**: animals receive different indications (e.g. fluoxetine for separation anxiety). Mitigation: bucket-level (not PT-level) primary analysis; indication field where present; masking-adjusted IC as sensitivity.
- **Denominator artefacts**: veterinary products with few actives vs human polypharmacy. Mitigation: first-listed-active sensitivity; volume-matched cells.
- **Label-driven stimulated reporting** (e.g. isoxazoline 2018 alert; bedinvetmab 2024-2025). Mitigation: trajectory analysis dates first signal before alerts; sensitivity excluding alert-affected quarters.
- **Dose parsing quality**: free-text units; missing weights. Mitigation: strict parser with unit whitelist; report coverage; treat as exploratory (H5).
- **API limits**: 25,000-record skip ceiling and 1,000-bucket counts. Mitigation: date-window paging (implemented) and count queries.

## Milestones

- [ ] Full CVM harvest and flattening; FAERS count series for candidate ingredients.
- [ ] Ingredient normalisation; shared-ingredient list frozen; approval dates joined.
- [ ] Organ-system crosswalk and PT mapping reviewed (kappa); release v0.
- [ ] Pre-register control sets, hypotheses and thresholds.
- [ ] Signal tables and concordance (H1-H3) with bootstrap/permutation.
- [ ] Lead-lag trajectories (H4); dose analysis (H5); robustness (H6).
- [ ] Preprint; crosswalk + signal tables released.

## Ethics / data-use notes

- Both databases are public and de-identified; owners/veterinarians must not be re-identified. Keep `OPENFDA_API_KEY` in the environment only.
- MedDRA is licensed: do not redistribute the dictionary; release only PT names as they appear in public data and the organ-system crosswalk built from public term lists.
- Cite FDA CVM, openFDA and EMA VeDDRA; spontaneous-report caveats (no incidence, reporting biases) must be stated in any publication.
- Never commit downloaded data (`data/` is git-ignored).

## Quick start

```bash
pip install -r requirements.txt
export OPENFDA_API_KEY=...                       # optional
python scripts/download_data.py --sample          # ~500 dog/cat reports (one month) + demo FAERS counts
PYTHONPATH=src pytest -q tests                    # synthetic-data tests (no network)
```

```python
import pandas as pd
from xspecies_pv import (assign_organ_system, normalize_ingredient, signal_table, concordance,
                         permutation_null_rho, cumulative_ic_by_quarter, lead_lag, mg_per_kg)

animal = pd.read_parquet("data/animal/flat/animal_events.parquet")
dog = animal[animal["species"] == "Dog"].assign(
    ingredient=lambda d: d["active_ingredient"].map(normalize_ingredient),
    bucket=lambda d: d["veddra_term"].map(assign_organ_system))
human = pd.read_parquet("outputs/faers_long.parquet").assign(     # report_id, ingredient, pt, quarter
    bucket=lambda d: d["pt"].map(assign_organ_system))
t_dog, t_hum = signal_table(dog), signal_table(human)
res = concordance(t_dog, t_hum, min_a=3, n_boot=1000)
print(res["rho"], res["rho_ci"], res["kappa"], permutation_null_rho(t_dog, t_hum)["p_value"])
print(lead_lag(cumulative_ic_by_quarter(dog, "meloxicam", "gastrointestinal"),
               cumulative_ic_by_quarter(human, "meloxicam", "gastrointestinal")))
```

## Pre-specified operational definitions

| Item | Definition (frozen before signal tables are computed) |
|---|---|
| Shared ingredient | normalised active ingredient with >= 100 reports in the CVM database (dog or cat) and >= 100 primary-suspect FAERS reports, 2008-2025 |
| Animal exposure | every active ingredient listed on the report (primary); first-listed only (sensitivity) |
| Human exposure | primary suspect only (`drugcharacterization:1`) |
| Reaction unit | harmonised organ-system bucket (22 classes, `ORGAN_SYSTEMS`); PT-level secondary analysis restricted to reviewed VeDDRA-MedDRA pairs |
| Cell | ingredient x bucket, one count per distinct report; cells with a >= 3 in both species enter the concordance |
| Signal | BCPNN IC025 > 0 and a >= 3 (Norén et al. 2013 shrinkage), computed against each species' own totals |
| Concordance | Spearman rho on IC and Cohen's kappa on signal flags; ingredient-cluster bootstrap (1,000) and ingredient-label permutation (999) |
| Positive controls (discordant) | cat-acetaminophen-haematological; dog-ivermectin-nervous; dog-NSAID-gastrointestinal (ulceration PTs); cat-permethrin-nervous |
| Positive controls (concordant) | gabapentin-nervous (sedation/ataxia); insulin-endocrine_metabolic (hypoglycaemia); anticoagulant-cardiovascular (bleeding); levothyroxine-cardiovascular |
| Lead-lag | first quarter of two consecutive IC025 > 0 quarters per species; difference in quarters; mechanism match from shared EPC class |
| Dose analysis | mg/kg from `active_ingredients.dose` and `animal.weight` (unit whitelist); ratio to labelled mg/kg (Green Book); logistic model of `serious_ae` with ingredient random intercepts |
| Exclusions (H6) | manufacturer-only reports; lack-of-efficacy reports; reports coded before VeDDRA v3; FAERS lawyer/literature reports |

## Related projects

- `faers-reporting-bias` (reporter-type bias models reusable for the human side).
- `faers-signal-ehr-validation` (validation of FAERS signals; the animal side has no EHR analogue yet).
