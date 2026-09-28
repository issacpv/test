# caers-supplement-signals: ingredient-level, cross-system safety signals for dietary supplements from CAERS (openFDA food/event) and FAERS

**One-sentence pitch.** Turn the FDA's under-used CFSAN Adverse Event Reporting System (CAERS) into an ingredient-level pharmacovigilance resource by linking verbatim supplement product names to ingredients (lexicon + NIH DSLD), then test which ingredient–event signals replicate between CAERS and the supplement reports hidden inside FAERS, how they perform against hepatotoxicity and adulteration reference sets, and how the 2007 mandatory serious-reporting rule and media episodes shaped the data.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are committed.
- Difficulty: MSc-level (pharmacoepidemiology + record linkage); 6–8 months to a first paper; 9–12 months including DSLD linkage validation and the cross-system study.
- Compute: CPU only. CAERS is small (on the order of 10⁵ reports since 2004, tens of thousands for supplements); FAERS supplement-named reports are a few hundred thousand. A laptop is sufficient; DSLD lookups are the slow step (thousands of distinct product names, rate-limited API).

## Background

Dietary supplements are used by more than half of US adults, are regulated as foods (DSHEA 1994) without pre-market safety review, and cause an estimated 23 000 emergency-department visits per year (Geller et al., 2015, *N Engl J Med*). Herbal and dietary supplements account for a rising share of drug-induced liver injury in the DILIN network (Navarro et al., 2014, *Hepatology*; Navarro et al., 2017, *Hepatology*), and hundreds of products have been found adulterated with pharmaceutical ingredients (Tucker et al., 2018, *JAMA Netw Open*; Cohen, 2018, *JAMA Netw Open*). Since December 2007 the Dietary Supplement and Nonprescription Drug Consumer Protection Act requires manufacturers to submit *serious* adverse-event reports to FDA; these, plus voluntary MedWatch reports, form CAERS, which FDA began publishing in 2016 and which openFDA exposes as the `food/event` endpoint. CAERS records product names verbatim, with an "industry code" (54 = dietary supplements), MedDRA-coded reactions, outcomes, and consumer age/sex — but no ingredient coding, no reporter type, and no narrative.

## The research gap

**What has been done.**

- Descriptive analysis of CAERS dietary-supplement reports 2004–2013 (Timbo et al., 2018, *Annals of Pharmacotherapy*).
- A 2026 preprint applying four disproportionality methods, CUSUM temporal detection and demographic stratification to ~49 000 unique CAERS supplement reports 2004–2025 at the *product-name* level, reporting hepatotoxicity clustering in herbal/botanical and weight-loss categories and preliminary hepatic-enzyme signals for a few branded products (Authorea, 2026, doi:10.22541/au.177383914.49860394/v1).
- CAERS-based disproportionality for cosmetics (facial skincare adverse-event atlas, 2025/2026) — same endpoint, different industry codes.
- Supplement hepatotoxicity reference knowledge in LiverTox (NCBI Bookshelf) and DILIN case series; FDA's Tainted Products Marketed as Dietary Supplements list; FDA warning letters.
- NIH ODS Dietary Supplement Label Database (DSLD): ~150 000 product labels with full ingredient lists and an open API — never linked to CAERS.

**What is specifically missing.**

1. **Ingredient-level signals.** All CAERS analyses are at product-name level ("AG1", "Hydroxycut"), so the same botanical sold under hundreds of brands is never aggregated, and multi-ingredient products cannot be decomposed. DSLD linkage makes ingredient-level 2×2 tables possible for the first time.
2. **Cross-system replication.** Supplements are also reported to FAERS as "drugs" (verbatim names such as TURMERIC, KRATOM, ASHWAGANDHA); whether CAERS and FAERS agree on ingredient-level signals — and which system detects what earlier — is unknown.
3. **Performance evaluation.** No CAERS method has been evaluated against a reference set (AUROC of ROR/PRR/IC for known herbal hepatotoxins vs negative controls; detection of adulterated products before FDA action).
4. **Structural features of the data.** The 2007 mandatory-reporting rule (manufacturer-submitted serious cases) and product-specific media episodes (e.g., Hydroxycut 2009 recall, OxyElite Pro 2013 hepatitis cluster) create level shifts and stimulated reporting that existing analyses treat as signals rather than modelling them.
5. **Denominator-aware and demographic analyses.** NHANES supplement-use modules give sex- and age-specific prevalence of use for major ingredients, enabling crude reporting rates rather than proportions.

## Research questions / hypotheses

1. **H1 (linkage).** Lexicon + DSLD linkage assigns at least one ingredient to ≥ 70 % of CAERS supplement reports (≥ 85 % of reports for the 200 most-reported product names); manual validation of 300 linked products shows ≥ 90 % ingredient precision.
2. **H2 (ingredient signals).** Ingredient-level screening within the supplement stratum yields ≥ 30 ingredient–PT signals (ROR lower bound > 1, BH q ≤ 0.05, a ≥ 3), among them known hepatotoxins (green tea extract, kava, kratom, ashwagandha, turmeric, garcinia, red yeast rice, niacin, anabolic/SARM products) with hepatic PTs; product-level screening misses at least a third of these because counts are split across brands.
3. **H3 (reference performance).** Against a LiverTox/DILIN-derived hepatotoxicity reference set (positives) and nutrient/probiotic controls (negatives), ingredient-level IC/ROR achieve AUROC ≥ 0.80, exceeding product-level AUROC by ≥ 0.10.
4. **H4 (cross-system).** CAERS and FAERS ingredient-level log-RORs correlate (Spearman ρ ≥ 0.5) for pairs reported ≥ 3 times in both; FAERS captures more serious/hospitalised cases and physician-coded PTs, CAERS more consumer-type PTs; for ≥ 3 ingredients CAERS shows the signal ≥ 12 months before FAERS (or vice versa), quantified by CUSUM first-alarm dates.
5. **H5 (structural breaks).** The December 2007 rule produced a level increase in serious supplement reports (ITS level ratio > 1.5) with no change in non-serious reports; media episodes (Hydroxycut 2009, OxyElite Pro 2013) produce transient stimulated reporting for the named products *and* for unrelated weight-loss products (spill-over).
6. **H6 (adulteration).** Products later listed on FDA's tainted-products list show sexual-enhancement, weight-loss and bodybuilding category signals (hypotension, tachycardia, hepatic PTs) in CAERS before the FDA listing date for ≥ 25 % of listed products with ≥ 5 reports.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| openFDA food/event (CAERS) | all reports 2004–2026: products (name, role, industry code), reactions, outcomes, consumer age/sex, dates | ~10⁵ reports; supplements ~5 × 10⁴ | open (free API key optional) | https://open.fda.gov/apis/food/event/ |
| FDA CAERS quarterly CSV/XLSX | same fields from the source system (cross-check; "CAERS Created Date") | ~50 MB | open | https://www.fda.gov/food/compliance-enforcement-food/cfsan-adverse-event-reporting-system-caers |
| openFDA drug/event (FAERS) | reports with supplement-like verbatim product names; seriousness, reporter type, PTs | ~few × 10⁵ of ~2 × 10⁷ | open | https://open.fda.gov/apis/drug/event/ |
| NIH ODS DSLD (labels + API) | brand → ingredient lists | ~150 k labels | open (API; full export) | https://dsld.od.nih.gov/ |
| LiverTox (NCBI Bookshelf) | herbal/dietary-supplement hepatotoxicity monographs → positive reference set | ~100 HDS monographs | open | https://www.ncbi.nlm.nih.gov/books/NBK547852/ |
| FDA Tainted Products Marketed as Dietary Supplements | adulterated product list with dates and hidden ingredients | ~1 000 entries | open | https://www.fda.gov/drugs/medication-health-fraud/tainted-products-marketed-dietary-supplements-cder |
| NHANES DSQ (Dietary Supplement Use) | prevalence of use by ingredient, sex, age | ~10 k participants/cycle | open | https://wwwn.cdc.gov/nchs/nhanes/ |
| MedDRA | PT → HLT/SOC grouping (hepatic, cardiovascular panels) | — | licence (free academic) | https://www.meddra.org/ |

## Methods

1. **Extraction (`caers_signals.openfda`, `normalize.flatten_caers`).** Pull all CAERS reports (bulk partitions); normalise product strings (case, dose/size tokens, punctuation); code outcomes into serious / death / hospitalisation / medically attended; restrict the supplement stratum to industry code 54 with a Suspect role.
2. **Ingredient linkage (`normalize.map_ingredients`, `dsld_search`).** Stage 1: curated regex lexicon (~55 canonical ingredients/classes) applied to product strings. Stage 2: DSLD API lookup of the top-N product names (cached), taking the best-matching label's ingredient list; map DSLD ingredient names onto the canonical set with the same lexicon. Report linkage coverage and validate on 300 hand-checked products (H1). Keep both mapping routes for sensitivity analyses.
3. **Signal screen (`signals.screen`).** For every (ingredient, PT) with a ≥ 3: ROR with Wald CI, PRR with the Evans criterion, BCPNN IC with shrinkage; BH across pairs; two backgrounds (supplement stratum only; all CAERS). Product-level screen for comparison (H2). Hepatic, cardiovascular and psychiatric PT panels via MedDRA HLTs.
4. **Reference evaluation (`signals.reference_evaluation`).** Positives: LiverTox HDS monographs with likelihood scores A–C paired with hepatic PTs; negatives: nutrient/probiotic ingredients paired with hepatic PTs and hepatotoxins paired with unrelated PTs. AUROC/AP with bootstrap CIs, ingredient- vs product-level (H3).
5. **Cross-system (`normalize.flatten_faers_for_supplements`, `signals.cross_system_concordance`).** Apply the lexicon to FAERS verbatim names; ingredient-level screen in FAERS; Spearman/kappa/Jaccard concordance; CUSUM first-alarm dates per system (H4).
6. **Time (`signals.monthly_series`, `segmented_poisson_its`, `poisson_cusum`).** Segmented Poisson ITS at 2007-12 for serious vs non-serious supplement reports; product- and category-level ITS around media episodes with spill-over tests (H5); CUSUM for emerging ingredient signals with expected rates from the pre-period.
7. **Adulteration (H6).** Link tainted-products list to CAERS product names (fuzzy matching, rapidfuzz), compute category signals before listing date.
8. **Denominators.** NHANES DSQ prevalence by ingredient × sex × age → crude reporting rates for the ten most-used ingredients; sex-stratified RORs with sex-specific backgrounds.

## Evaluation & statistics

- Primary endpoints: linkage coverage/precision (H1); number of ingredient-level signals and their overlap with product-level signals (H2); AUROC ingredient- vs product-level with bootstrap CI (H3); Spearman ρ with CI (H4); ITS level ratios with quasi-Poisson SEs (H5).
- Multiplicity: BH within each screen; Holm across the six hypotheses' primary tests.
- Nulls: permutation of ingredient labels across reports within calendar year (null for ROR/IC counts); placebo break dates for ITS; shuffled reference labels for AUROC.
- Duplicates: CAERS contains manufacturer and consumer versions of the same event; deduplicate on (date_started, age, sex, product string, reaction set) and report sensitivity to the rule.
- Bias controls: separate mandatory (post-2007, serious) from voluntary reports where outcomes allow; exclude the media-episode windows in sensitivity analyses; report both backgrounds.
- Reporting per READUS-PV (Fusaroli et al., 2024, *Drug Saf*) adapted to CAERS.

## Publishable angle

- **Headline result.** "Linking CAERS to DSLD reveals ingredient-level supplement safety signals invisible at the product level; they replicate in FAERS with ρ ≈ X, achieve AUROC ≈ Y against known hepatotoxins, and CAERS detected Z adulterated-product categories before FDA action." Delivered with an open ingredient-level signal table and the linkage code.
- **Venues.** *Drug Safety*; *Pharmacoepidemiology and Drug Safety*; *JAMA Network Open* (adulteration/regulatory angle); *Clinical Gastroenterology and Hepatology* or *Hepatology Communications* (hepatotoxicity sub-study); *Journal of Dietary Supplements*.
- **Follow-ups.** Cosmetics (industry code 53) with the same pipeline; NLP on FDA warning letters as an additional reference set; comparison with poison-centre (NPDS) annual report categories.

## Risks, confounds & mitigations

- **Product-name ambiguity** (brand families with many formulations; reformulations over time, e.g., Hydroxycut pre/post 2009). Mitigation: DSLD label dates; time-restricted mapping; sensitivity with lexicon-only mapping.
- **Multi-ingredient products** dilute ingredient attribution. Mitigation: proportional attribution sensitivity; restrict a primary analysis to single-ingredient products.
- **No reporter type in CAERS.** Mitigation: infer manufacturer vs consumer source from seriousness × period and from the quarterly FDA files where a source field exists; treat as a limitation.
- **Small counts** for most ingredients. Mitigation: IC shrinkage; minimum a ≥ 3; hierarchical pooling within ingredient classes as an extension.
- **Reference-set circularity** (LiverTox monographs cite case reports that may be in CAERS/FAERS). Mitigation: restrict positives to ingredients with DILIN-adjudicated cases or mechanistic evidence; report both.
- **Duplicate and stimulated reporting** around recalls. Mitigation: deduplication rule; ITS modelling; exclusion windows.

## Milestones

- [ ] Full CAERS pull; normalisation; seriousness coding; deduplication rule
- [ ] Lexicon + DSLD linkage; coverage and 300-product validation (H1)
- [ ] Ingredient- vs product-level screens with two backgrounds (H2)
- [ ] Reference sets (LiverTox, controls, tainted list); AUROC evaluation (H3, H6)
- [ ] FAERS supplement extraction; concordance and CUSUM timing (H4)
- [ ] ITS for 2007 rule and media episodes; spill-over analysis (H5)
- [ ] NHANES denominators; sex-stratified rates
- [ ] Manuscript, released signal table and linkage code

## Ethics / data-use notes

- CAERS, FAERS, DSLD, LiverTox and NHANES are public; CAERS has no narratives and no identifiers beyond age/sex. Secondary-use exemption typical; confirm locally.
- Ingredient-level signals are hypotheses; brand-level statements must carry the openFDA disclaimer and note that report counts reflect reporting, not incidence.
- Never commit downloaded data; `data/` is git-ignored; API key from `OPENFDA_API_KEY`. DSLD API terms of use apply to bulk lookups (cache results; respect rate limits).

## Related projects in this repository

`faers-reporting-bias` (stimulated-reporting ITS, reporter-type adjustment), `unlabeled-adverse-event-mining` (label expectedness — not applicable to supplements, which have no adverse-reaction labelling; this contrast is itself a discussion point), `faers-ddi-signals` (supplement–drug interaction pairs as a follow-up). This project is self-contained.

## Quick start (starter code)

```bash
cd projects/caers-supplement-signals
pip install -r requirements.txt
export OPENFDA_API_KEY=...                      # optional
python scripts/download_data.py --sample --per-query 1000
PYTHONPATH=src python -m pytest -q tests
```

```python
from caers_signals import read_jsonl, screen, serious_fraction_model, monthly_series, segmented_poisson_its, cross_system_concordance
supp = [r for r in read_jsonl("data/raw/caers_supplements.jsonl") if r["supplement"]]
tab = screen(supp, exposure_key="ingredients", min_a=3)             # ROR/PRR/IC per (ingredient, PT), BH q
hep = tab[tab["event"].isin(["HEPATOTOXICITY", "HEPATITIS", "JAUNDICE"])]
sf = serious_fraction_model(supp, "KRATOM")                          # OR of serious outcome
s = monthly_series(supp, mask=lambda r: r["serious"])
its = segmented_poisson_its(s.values, change_index=list(s.index).index(s.index[s.index >= "2007-12-01"][0]))
faers = screen(read_jsonl("data/raw/faers_kratom.jsonl") + read_jsonl("data/raw/faers_turmeric.jsonl"), min_a=3)
conc = cross_system_concordance(tab, faers)
```

Module map: `openfda.py` (client; works for `food/event` and `drug/event`), `normalize.py` (CAERS flattening, product-string normalisation, 55-entry ingredient lexicon, outcome coding, DSLD lookup with cache, FAERS supplement flattening), `signals.py` (ROR/PRR/IC screen, seriousness model, monthly series, segmented Poisson ITS, Poisson CUSUM, cross-system concordance, reference-set AUROC, starter hepatotoxicity reference).

## Key variables and cohort definitions

| Variable | Definition | Source field(s) |
|---|---|---|
| Supplement report | ≥ 1 product with industry code 54 in Suspect role | `products.industry_code`, `products.role` |
| Product string | upper-cased `name_brand` with dose/size tokens and punctuation removed | `products.name_brand` |
| Ingredient | canonical lexicon class (stage 1) and/or DSLD label ingredients mapped to the lexicon (stage 2) | derived; DSLD API |
| Event | MedDRA PT as coded by CFSAN | `reactions[]` |
| Serious | any of death, life-threatening, hospitalization, disability, congenital anomaly, required intervention, other serious/important medical event | `outcomes[]` |
| Medically attended | ER visit or health-care-provider visit | `outcomes[]` |
| Sex / age | Female / Male; age in years from `age` + `age_unit` | `consumer.*` |
| Period | `date_created` month; mandatory-reporting break 2007-12; media windows (Hydroxycut 2009-05, OxyElite Pro 2013-10) | `date_created` |
| Background | supplement stratum only (primary); all CAERS (secondary) | derived |
| FAERS supplement report | any verbatim drug name matching the lexicon | `patient.drug.medicinalproduct`, `openfda.generic_name` |
| Reference positives / negatives | LiverTox HDS ingredients × hepatic PTs; nutrients/probiotics × hepatic PTs, hepatotoxins × unrelated PTs | `reference/*.csv` |
