# ais-plasticity-real-morphologies

**Population-scale modelling of axon-initial-segment (AIS) plasticity across hundreds of real NeuroMorpho.org reconstructions: how much AIS relocation or elongation does each somatodendritic tree "demand" to hold excitability at a set point, does resistive-coupling theory predict the simulations, and does the predicted morphology-AIS covariation match measured AIS geometry?**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (SWC loader, frequency-domain passive "dendritic load" calculator, resistive-coupling AIS theory with a plasticity-demand solver, NEURON model-builder stub).
- Difficulty: MSc/PhD-chapter level; requires comfort with cable theory and NEURON. No GPU.
- Timeline: 9-12 months (1 month morphology harvest and QC, 2 months passive load metrics and theory, 3-4 months NEURON simulations across morphologies and channel sets, 2 months comparison with empirical AIS data, 2 months writing).
- Compute: a workstation (16-32 cores). Each NEURON simulation (one morphology x one AIS geometry x one channel set, 200 ms) takes seconds; a full grid (500 morphologies x 100 AIS geometries x 3 channel sets) is ~150k runs, a day or two on 32 cores. The passive/theory pass runs on a laptop.

## Background

Action potentials initiate in the AIS, and neurons move or resize their AIS to regulate excitability: distal relocation with chronic depolarisation in hippocampal cultures (Grubb & Burrone, 2010, *Nature*), elongation after auditory deprivation in chick nucleus magnocellularis (Kuba, Oichi & Ohmori, 2010, *Nature*), rapid shortening after whisker stimulation in mouse barrel cortex (Jamann et al., 2021, *Nature Communications*), and relocation within hours after M-current block (Lezmy et al., 2017, *PNAS*). Reviews: Kole & Brette, 2018, *Current Opinion in Neurobiology*; Jenkins & Bender, 2025, *Physiological Reviews*.

What these structural changes do to excitability depends on the somatodendritic tree they are attached to. Gulledge & Bravo (2016, *eNeuro*) showed with a handful of reconstructed neurons that the optimal AIS length and the sign of the position effect depend on neuron size; Hamada et al. (2016, *PNAS*) found that AIS location covaries with dendritic size in vivo so as to normalise the somatic action potential; Goethals & Brette (2020, *eLife*) derived a resistive-coupling theory in which the somatic voltage threshold depends logarithmically on the axial resistance between soma and AIS and on the total AIS sodium conductance, with the surprising prediction that moving the AIS *away* from a large soma *lowers* the threshold; Fekete et al. (2021, *PNAS*) confirmed the axial-resistance dependence experimentally. Verbist, Salvade & Giugliano (2020, *PLoS Computational Biology*) showed that AIS location also sets the bandwidth of spike-initiation dynamics. Recent population anatomy has measured AIS geometry in thousands of hippocampal pyramidal neurons and related it to proximal dendritic geometry (Cerebral Cortex, 2025, "Diversity of axon initial segment geometry in the mouse hippocampus"), and a 2024 preprint explores the variability of AIS geometry and its functional impact.

## The research gap

**What has been done**

- Theory (Brette, 2013, *PLoS Computational Biology*; Goethals & Brette, 2020) and simulations on one or a few morphologies (Gulledge & Bravo, 2016; Hamada et al., 2016; Verbist et al., 2020).
- ModelDB hosts the biophysical AIS models these papers used (search ModelDB for Hu et al., 2009, *Nature Neuroscience*; Hallermann et al., 2012, *Nature Neuroscience*; Kole et al., 2008, *Nature Neuroscience*; Gulledge & Bravo, 2016); each was tuned on its own morphology.
- NeuroMorpho.org supplies >250,000 reconstructions with soma and dendrites (axons often truncated), and NeuroMorpho-derived somatodendritic trees have been used before with synthetic AIS/axons attached (Gulledge & Bravo, 2016), but never at population scale.
- Empirical AIS-versus-dendrite covariation exists for a few cell classes (Hamada et al., 2016; the 2025 hippocampal population study).

**What is missing (checked against 2023-2026 literature)**

1. No study has computed, across hundreds of real reconstructions within and across cell classes, the AIS geometry (distance from soma, length, channel density) required to hold a functional set point (threshold, rheobase, somatic AP amplitude) constant - the "plasticity demand" imposed by morphology.
2. The resistive-coupling prediction that threshold is set by soma-AIS axial resistance *and largely independent of dendritic load* has not been tested against the full range of dendritic loads in a database; nor has the complementary prediction that somatic AP amplitude and onset rapidness *do* depend on load (which is what Hamada et al. showed the AIS compensates for).
3. Whether empirically measured AIS-dendrite covariation (Hamada 2016; 2025 hippocampal data) matches the homeostatic prediction (AIS placed where the set point is met) has not been quantified with a model that includes the real morphologies.
4. Sensitivity of all of the above to the channel model (ModelDB variants differ in Nav1.2/Nav1.6 distribution, Kv1/Kv7 placement) is unknown; conclusions drawn from one channel set may not transfer.

Related projects in this repository: `neuromorpho-scaling-laws` (batch effects in NeuroMorpho morphometrics; its lab-effect model is reused here as a covariate source) and `morphology-dependent-stimulation` (morphology-dependent responses to extracellular stimulation). This project is self-contained.

## Research questions / hypotheses

1. **H1 (load independence of threshold).** Across morphologies with a fixed AIS (distance 30 um, length 40 um, fixed Na density), the simulated somatic voltage threshold varies by < 3 mV across a 10-fold range of high-frequency somatic capacitive load, whereas rheobase varies > 5-fold. Test: mixed model of threshold and rheobase on log load with cell class as random intercept.
2. **H2 (plasticity demand scales with load).** The AIS distance needed to restore somatic AP amplitude (or maximal dV/dt) to a class-specific set point increases with the effective somatic load; the relation is log-linear as predicted by resistive-coupling theory. Test: fit demand = a + b log(load); compare slope with the theoretical value from `ais_theory`.
3. **H3 (theory predicts simulation).** The Goethals-Brette threshold formula, parameterised only by axial resistance, AIS conductance and k_a, predicts NEURON-simulated thresholds across morphologies with R^2 > 0.8 for point-like AIS and degrades systematically for extended AIS - quantify the deviation as a function of AIS length.
4. **H4 (empirical covariation is homeostatic).** The slope of measured AIS distance vs. dendritic size in Hamada et al. (2016) and the hippocampal population data is within the CI of the slope predicted by the plasticity-demand model for the same cell class; it is not predicted by a model with a constant AIS.
5. **H5 (channel-set robustness).** Conclusions H1-H3 hold (same sign, overlapping CIs) across at least three ModelDB channel sets; channel set explains less variance in plasticity demand than morphology does.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| NeuroMorpho.org (v8.x) | Somatodendritic reconstructions of principal cells (neocortical L2/3 and L5 pyramidal, CA1/CA3 pyramidal, dentate granule) and interneurons, with archive/software/shrinkage metadata; CNG.swc | target 500-2,000 reconstructions after QC | Open, no registration (REST API) | https://neuromorpho.org/api/ |
| ModelDB (SenseLab) | Published AIS/axon channel models (.mod files and hoc templates) from Hu et al. 2009, Hallermann et al. 2012, Gulledge & Bravo 2016 and Goethals & Brette 2020 | 4-6 entries | Open (download or git) | https://modeldb.science |
| Allen Cell Types Database | Mouse and human reconstructions *with matched* electrophysiology (rheobase, threshold, AP amplitude, dV/dt) for calibrating set points and validating simulated features | ~1,000 cells | Open (allensdk) | https://celltypes.brain-map.org |
| Empirical AIS geometry tables | AIS distance/length vs. dendritic size from Hamada et al. 2016 (supplement) and the 2025 hippocampal population study (supplement or repository named in the paper) | hundreds to thousands of cells | Open (paper supplements) | see cited papers |
| MICrONS / cortical EM (optional) | EM-reconstructed neurons with visible AIS for cross-checking AIS geometry against dendritic size in the same cells | thousands | Open (CAVE / BossDB) | https://www.microns-explorer.org |

No credentialed data are involved.

## Methods

1. **Morphology harvest** (`scripts/download_data.py`). NeuroMorpho REST API: filter to `physical_Integrity` "Dendrites Complete" (axon may be absent), `protocol` in vitro/in vivo, cell classes above, species mouse/rat/human; download CNG.swc; keep archive, software and shrinkage metadata.
2. **Passive dendritic load** (`src/ais_plasticity/dendritic_load.py`). Frequency-domain passive solution of the tree (sparse complex solve) to obtain: steady-state input conductance at the soma, effective capacitance at 500-2,000 Hz (what the AIS spike must charge), total membrane area, and the somatic transfer impedance seen from a virtual AIS at distance Delta. This is the morphology summary used by the theory.
3. **Theory** (`src/ais_plasticity/ais_theory.py`). Resistive-coupling threshold as a function of AIS distance, length and Na conductance following Goethals & Brette (2020); axial resistance from axon diameter and Ra; a solver that returns the AIS distance/length required to reach a target threshold or a target somatic AP amplitude for a given load ("plasticity demand"); sensitivity to k_a and g_Na.
4. **NEURON simulations** (`src/ais_plasticity/neuron_builder.py`). Attach a standard axon (hillock 10 um tapering, AIS of variable start/length, myelinated axon) to each somatodendritic tree; insert channel sets from ModelDB; measure voltage threshold (dV/dt criterion), rheobase, somatic AP amplitude, maximal dV/dt and onset rapidness (phase-plane slope at threshold). Grid: AIS start 0-60 um, length 10-60 um, Na density x0.5-2.
5. **Plasticity demand.** For each morphology and channel set, find the AIS geometry closest to the class reference geometry that restores the set point (bisection on distance, then on length); report demand as (Delta distance, Delta length) and as "unreachable" when no geometry meets it.
6. **Empirical comparison.** Regress measured AIS distance on dendritic size (from the published tables) and compare with the demand model's prediction for the same cell class; use Allen Cell Types matched ephys to check simulated rheobase/threshold distributions.
7. **Tools**: `numpy`, `scipy` (sparse complex solves, root finding), `pandas`, `statsmodels` (mixed models), `neuron` (>= 8.2, Python API) for step 4, `requests`.

## Evaluation & statistics

- Primary estimands: coefficient of variation of threshold vs. rheobase across load (H1), slope of demand on log load with CI from cluster bootstrap over archives (H2), theory-vs-simulation R^2 and calibration slope (H3), predicted vs. observed empirical covariation slope with CIs (H4), variance decomposition morphology vs. channel set (H5, nested ANOVA / mixed model).
- Confounds: archive/software/shrinkage as random effects or covariates (shrinkage changes diameters and therefore load); results reported with and without shrinkage-corrected subsets. Morphologies with truncated dendrites are flagged by NeuroMorpho integrity fields and excluded from H2/H4.
- Leakage: the set points are defined on Allen Cell Types data (independent of NeuroMorpho morphologies); no fitting across morphologies except the reported regressions.
- Multiple comparisons: five pre-registered hypotheses; secondary per-class analyses BH-corrected.
- Nulls: (a) shuffled morphology-load pairing (does the demand relation survive when load is permuted across cells?); (b) synthetic ball-and-stick controls spanning the same load range to separate "load" from "shape"; (c) simulations with the AIS held constant as the non-homeostatic reference for H4.
- Numerical checks: passive solutions validated against the analytic Rall ball-and-stick solution (unit test); NEURON time-step convergence (dt 25 vs 5 us) on a subset.

## Publishable angle

- **Headline**: "Across N real morphologies, the AIS geometry required to hold excitability constant spans X um of relocation and Y um of length, scales log-linearly with somatodendritic load as resistive-coupling theory predicts, and reproduces the AIS-dendrite covariation measured in vivo; results are robust to the choice of channel model." Secondary: a public table of load metrics and plasticity demand per NeuroMorpho neuron.
- Target venues: *eLife*, *PLoS Computational Biology*, *Journal of Neuroscience*, *Journal of Neurophysiology*.
- Follow-ups: disease morphologies (see `disease-morphology-signatures`) - does dendritic simplification in AD models change plasticity demand?; extension to axo-axonic (chandelier) input placement; use of EM-measured AIS geometry (MICrONS) with the same cells' dendrites.

## Risks, confounds & mitigations

- **Axons are absent or truncated in most NeuroMorpho files.** This project deliberately attaches a standard synthetic axon; the biological axon diameter is a free parameter that is swept and reported.
- **Soma representation** in SWC (single point vs. contour) changes somatic area. Mitigation: standardise soma as a sphere of equivalent area; sensitivity analysis with +-20% somatic area.
- **Channel-set tuning is morphology-specific**: each ModelDB model was tuned on one cell. Mitigation: treat channel set as a factor (H5), rescale Na density to match a reference threshold on the reference morphology only.
- **Theory regime**: resistive coupling assumes a large soma relative to the AIS; small neurons (granule cells) may violate it - report deviation explicitly.
- **Shrinkage** affects diameters and therefore both load and axial resistance; use `shrinkage_corrected` metadata.
- **Compute**: prune the AIS grid adaptively (bisection rather than full grid) once H1-H3 are established.

## Milestones

- [ ] Harvest and QC 500+ morphologies across 5 cell classes; passive load table.
- [ ] Theory module calibrated on the reference morphology; unit tests vs. Rall analytic solutions.
- [ ] NEURON builder validated on Gulledge & Bravo (2016) morphologies (reproduce their optimal-AIS-length result).
- [ ] Fixed-AIS sweep across morphologies (H1); theory vs. simulation (H3).
- [ ] Plasticity-demand solver across morphologies and channel sets (H2, H5).
- [ ] Comparison with empirical AIS-dendrite covariation (H4).
- [ ] Preprint, code and per-neuron demand table released.

## Quick start

```bash
cd projects/ais-plasticity-real-morphologies
pip install -r requirements.txt
python scripts/download_data.py --sample        # 20 mouse pyramidal records, 5 SWC files
PYTHONPATH=src pytest -q                        # passive-load, theory and trace-feature tests (no NEURON needed)
```

Passive load and plasticity demand for a folder of reconstructions (no NEURON required):

```python
from pathlib import Path
import pandas as pd
from ais_plasticity.swc import read_swc, ball_and_stick
from ais_plasticity.dendritic_load import load_metrics
from ais_plasticity.ais_theory import AISParams, plasticity_demand

ref = load_metrics(ball_and_stick(r_soma=10, diam=2, length=400))   # or a chosen reference reconstruction
rows = {}
for f in Path("data/swc").rglob("*.CNG.swc"):
    load = load_metrics(read_swc(f))
    rows[f.stem] = {**load, **plasticity_demand(load, ref, AISParams())}
table = pd.DataFrame.from_dict(rows, orient="index")
print(table[["c_eff_1000hz_pf", "input_conductance_ns", "delta_distance_um", "delta_length_um"]].describe())
```

The NEURON step (`neuron_builder.build_model`, `run_step`, `rheobase`, `spike_features`) needs `pip install neuron`
and compiled ModelDB mechanisms (`nrnivmodl` in the folder holding the `.mod` files).

## Pre-registered analysis table

| # | Unit of analysis | Primary outcome | Estimand / test | Decision rule | Confirmatory / exploratory |
|---|---|---|---|---|---|
| H1 | reconstruction (fixed AIS 30/40 um) | somatic voltage threshold; rheobase | mixed model on log C_eff(1 kHz), archive random intercept; CV of threshold vs. rheobase | threshold range < 3 mV over a 10x load range; rheobase ratio > 5 | confirmatory |
| H2 | reconstruction | AIS distance restoring AP amplitude set point | slope of demand on log load, cluster bootstrap over archives | slope CI excludes 0; sign as predicted by theory | confirmatory |
| H3 | reconstruction x AIS geometry | simulated threshold | R^2 and calibration slope of theory vs. NEURON; deviation vs. AIS length | R^2 > 0.8 for point-like AIS | confirmatory |
| H4 | cell class | AIS distance vs. dendritic size slope | predicted (demand model) vs. measured slope, CIs | measured slope within predicted CI; constant-AIS model rejected | confirmatory |
| H5 | reconstruction x channel set | plasticity demand | nested ANOVA / mixed model: morphology vs. channel-set variance | morphology variance share > channel-set share | confirmatory |
| S1 | reconstruction | onset rapidness, max dV/dt | descriptive vs. load | - | exploratory |

Set points are defined on Allen Cell Types matched electrophysiology per class (median rheobase, threshold, AP amplitude) before any NeuroMorpho morphology is simulated.

## Key references

- Grubb MS, Burrone J (2010) Activity-dependent relocation of the axon initial segment fine-tunes neuronal excitability. *Nature*.
- Kuba H, Oichi Y, Ohmori H (2010) Presynaptic activity regulates Na+ channel distribution at the axon initial segment. *Nature*.
- Brette R (2013) Sharpness of spike initiation in neurons explained by compartmentalization. *PLoS Comput Biol*.
- Gulledge AT, Bravo JJ (2016) Neuron morphology influences axon initial segment plasticity. *eNeuro*.
- Hamada MS, Goethals S, de Vries SI, Brette R, Kole MHP (2016) Covariation of axon initial segment location and dendritic tree normalizes the somatic action potential. *PNAS*.
- Kole MHP, Brette R (2018) The electrical significance of axon location diversity. *Curr Opin Neurobiol*.
- Goethals S, Brette R (2020) Theoretical relation between axon initial segment geometry and excitability. *eLife*.
- Verbist C, Salvade MG, Giugliano M (2020) The location of the axon initial segment affects the bandwidth of spike initiation dynamics. *PLoS Comput Biol*.
- Fekete A et al. (2021) Neural excitability increases with axonal resistance between soma and axon initial segment. *PNAS*.
- Jamann N et al. (2021) Sensory input drives rapid homeostatic scaling of the axon initial segment in mouse barrel cortex. *Nat Commun*.
- Lezmy J et al. (2017) M-current inhibition rapidly induces a unique CK2-dependent plasticity of the axon initial segment. *PNAS*.
- Hu W et al. (2009) Distinct contributions of Nav1.6 and Nav1.2 in action potential initiation and backpropagation. *Nat Neurosci*.
- Hallermann S, de Kock CPJ, Stuart GJ, Kole MHP (2012) State and location dependence of action potential metabolic cost in cortical pyramidal neurons. *Nat Neurosci*.
- Jenkins PM, Bender KJ (2025) Axon initial segment structure and function in health and disease. *Physiol Rev*.
- "Diversity of axon initial segment geometry in the mouse hippocampus" (2025) *Cereb Cortex* (population AIS geometry dataset used for H4).

## Ethics / data-use notes

- NeuroMorpho.org and ModelDB data are open; cite each depositing publication and ModelDB accession in any output.
- Allen Cell Types data are under the Allen Institute terms of use (citation required).
- Human reconstructions are de-identified; no identifiable data are handled.
- Do not commit SWC files, ModelDB downloads or simulation outputs; `data/` and `outputs/` are git-ignored.
