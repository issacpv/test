# celltype-composition-mri-contrast

**Which cell types drive regional MRI contrast in the mouse brain? An interpretable, spatially-null-controlled regression of T1-, T2-, T2*-, magnetisation-transfer- and diffusion-derived regional signals from open mouse MRI atlases on *directly measured* MERFISH cell-type densities from the Allen Brain Cell (ABC) Atlas, with cross-atlas (ex vivo vs. in vivo) replication and a gene-program vs. cell-count model comparison.**

## Status / difficulty / timeline / compute

- Status: proposal + working starter code (ABC-Atlas cell-table to regional density conversion, NIfTI/NRRD regional contrast extraction, ridge/PLS regression with leave-region-out CV and relative-importance decomposition, Moran spectral randomisation spatial nulls).
- Difficulty: MSc thesis to early-PhD chapter; data engineering + spatial statistics. No GPU.
- Timeline: 6-9 months (2 months data assembly and registration checks, 1 month density and contrast tables, 2 months modelling and nulls, 1 month replication across atlases, 1-2 months writing).
- Compute: laptop for regional analyses (a few hundred regions); a workstation with 64 GB RAM for voxel-level (25-50 um) density maps from ~4 million MERFISH cells. Storage: ~20 GB (ABC Atlas MERFISH metadata + CCF; MRI atlases are < 5 GB).

## Background

MRI contrast in grey matter reflects a mixture of myelin, iron, cell and neurite density, and water compartmentation, but the mapping from cellular composition to contrast is mostly inferred from histology of a few regions or from lesion models (e.g. cuprizone demyelination validations of myelin-sensitive MRI). Two developments make a brain-wide, cell-type-resolved answer possible in the mouse. First, the ABC Atlas provides a whole-brain MERFISH dataset of about 4 million segmented cells assigned to a taxonomy of classes, subclasses and clusters and registered to the Allen Common Coordinate Framework (Yao et al., 2023, *Nature*; Zhang et al., 2023, *Nature*). Second, several high-resolution multi-contrast mouse MRI atlases exist in or near CCF space: the ex vivo DSURQE template and labels from the Mouse Imaging Centre (Dorr et al., 2008, *NeuroImage*, and later releases), the Australian Mouse Brain Mapping Consortium (AMBMC) ex vivo atlases (Ullmann et al., 2013, *NeuroImage*), and in vivo diffusion-tensor atlases from the Zhang laboratory (Wu et al., 2013, *NeuroImage*), all mappable to CCFv3 with the ANTsX mouse-brain tools (*Nature Communications*, 2025) or the Allen registration pipelines.

Earlier attempts to relate cellular composition to MRI used *inferred* densities: Fulcher, Murray, Zerbi & Wang (2019, *PNAS*) related the T1w:T2w ratio across mouse cortical areas to gene expression and to cell densities from the Blue Brain Cell Atlas (Ero et al., 2018, *Frontiers in Neuroinformatics*), which are estimated from Nissl and ISH rather than counted. Deep-learning "virtual histology" from multi-contrast MRI (Liang et al., 2022, *eLife*) and a 2025 preprint that trains deep networks to predict MERFISH cell types from high-resolution diffusion MRI ("High-resolution MRI-guided whole mouse brain cell type atlas using deep learning", bioRxiv, November 2025) go in the *inverse* direction (MRI to histology/cell type) with black-box models, and the 2025 preprint reports that MRI resolution limits per-cell classification.

## The research gap

**What has been done**

- Cortex-only, single-contrast (T1w:T2w) associations with ISH-inferred cell densities (Fulcher et al., 2019).
- Histological validation of individual quantitative contrasts against single stains in small regions (myelin vs. MPF/MT, NODDI indices vs. NeuN/LFB density in mouse; several 2016-2019 papers).
- Inverse deep-learning models predicting histology or cell types from MRI (Liang et al., 2022; 2025 preprint), which do not report which cell types *explain* which contrast and cannot separate cell counts from cell-type gene programs.
- Human imaging-transcriptomics with deconvolved cell-type proportions (Allen Human Brain Atlas based), which inherit the limits of bulk microarray deconvolution.

**What is missing (checked against 2023-2026 literature)**

1. No forward, interpretable model relates *counted* MERFISH cell-type densities (class / subclass level, including oligodendrocytes, OPCs, astrocytes, microglia, vascular and neuronal subclasses) to multiple MRI contrasts across the whole mouse brain.
2. Spatial autocorrelation has not been handled in mouse cell-density-vs-MRI comparisons; region-level correlations without spatially-informed nulls are inflated (well documented for human brain maps: Burt et al., 2020, *NeuroImage*; Markello & Misic, 2021, *NeuroImage*). Spin tests do not apply to a 3-D volume, so a volume-appropriate null (Moran spectral randomisation or variogram-matched surrogates) is required.
3. No study has asked whether contrast is better explained by *how many* cells of each type are present or by the *expression of specific gene programs* (myelin genes, iron-handling genes) measured in the same MERFISH panel, i.e. whether cell-type composition is sufficient.
4. Ex vivo (fixed, gadolinium-doped) and in vivo atlases have different contrast mechanisms; no study has tested whether cell-type explanations replicate across atlases, which is the minimal requirement for a claim about biology rather than about one dataset.

Related project in this repository: `imaging-transcriptomics-nulls` (spatial nulls for human brain-map correlations); this project is the mouse, cell-type-resolved, multi-contrast counterpart and is self-contained.

## Research questions / hypotheses

1. **H1 (oligodendrocyte dominance for myelin-sensitive contrasts).** Oligodendrocyte density explains the largest share of between-region variance in T1w:T2w, MT and FA, with relative importance > 40% in a ridge model of ~300 CCF regions; neuronal subclass densities add < 10%. Test: relative-importance decomposition with bootstrap CIs; spatial-null p-values.
2. **H2 (neuron density vs. diffusion).** Total neuron density (and the excitatory/inhibitory ratio) explains mean diffusivity and NODDI-style neurite density indices better than glial densities do; the association survives the Moran spectral randomisation null (p_spatial < 0.01).
3. **H3 (cell counts vs. gene programs).** Adding MERFISH-panel myelin (e.g. *Mbp*, *Plp1*, *Mog* if present in the panel) and iron-related gene expression to the cell-density model improves out-of-region prediction of T2*/QSM-like contrast by more than 10% R^2, but not T1w:T2w; i.e. counts are sufficient for myelin-driven contrast but not for iron-driven contrast.
4. **H4 (cross-atlas replication).** The ranking of cell classes by relative importance is concordant between the ex vivo DSURQE-based contrasts and the in vivo Zhang-lab DTI atlas (Kendall's tau > 0.6), with expected exceptions for contrasts affected by fixation (T2).
5. **H5 (resolution).** Regional (structure-level) models explain more variance than voxel-level (50 um) models after accounting for partial-volume mixing, and the difference is smaller for myelin-driven contrasts; this quantifies the resolution limit reported by the 2025 preprint from the forward side.

## Datasets

| Name | What is used | Size | Access | URL |
|---|---|---|---|---|
| Allen Brain Cell (ABC) Atlas - whole mouse brain MERFISH (Yao et al., 2023) | Cell metadata (cell id, CCF x/y/z, class, subclass, cluster, parcellation labels) and the per-cell expression matrix for the ~500-gene panel | ~4 million cells; metadata ~1 GB, expression ~10 GB | Open (AWS S3 public bucket; `abc_atlas_access` Python package) | https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas ; https://github.com/AllenInstitute/abc_atlas_access |
| Allen CCFv3 annotation and template (Wang et al., 2020, *Cell*) | Region labels, hierarchy, average template for registration | 10-100 um NRRD | Open | http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ |
| DSURQE template and labels (Mouse Imaging Centre, Toronto) | Ex vivo T2-weighted average template at 40 um and its labels; used both as a contrast and as a registration bridge | ~1 GB | Open (HTTP repository) | https://wiki.mouseimaging.ca/display/MICePub/Mouse+Brain+Atlases |
| AMBMC atlases (Ullmann et al., 2013 and related) | Ex vivo 15-30 um T2*-weighted template | ~1 GB | Open (download page) | https://imaging.org.au/AMBMC |
| Zhang-lab in vivo DTI atlas (Wu et al., 2013) and multi-contrast atlases | In vivo FA, MD, T2 and related maps | < 1 GB | Open (distribution named in the papers; contact authors if links move) | see cited papers |
| Blue Brain Cell Atlas (Ero et al., 2018) | ISH-inferred neuron/glia densities as a comparison baseline for H1 | ~100 MB | Open | https://bbp.epfl.ch/nexus/cell-atlas |
| Allen ISH (allensdk) | Gene-level expression grids for genes missing from the MERFISH panel (H3 sensitivity) | 200 um grids | Open | https://mouse.brain-map.org |

No credentialed data are involved.

## Methods

1. **Cell densities** (`src/celltype_mri/abc_atlas.py`). Load the ABC Atlas cell metadata (CCF coordinates in mm + class/subclass labels) with `abc_atlas_access`; assign each cell to a CCF structure by indexing the annotation volume; compute per-structure counts and densities (cells/mm^3, using structure volumes from the annotation), per class and subclass; also voxel-level density maps at 50-100 um for H5. Report section coverage (MERFISH is coronal sections with gaps) and adjust densities by sampled volume.
2. **Regional contrasts** (`src/celltype_mri/mri.py`). Load each atlas (NIfTI/NRRD via nibabel / pynrrd); register to CCF with ANTs (ANTsX mouse pipelines) or use the CCF-registered versions when provided; extract per-structure mean/median of each contrast; z-score within atlas; T1w:T2w ratio when both exist.
3. **Regression** (`src/celltype_mri/regress.py`). Standardised ridge and PLS regression of each contrast on log-densities of classes/subclasses; leave-one-region-out and leave-one-major-division-out CV; relative importance via Shapley-style (LMG) decomposition on the ridge fit; nested model comparison (densities vs. densities + gene programs).
4. **Spatial nulls** (`src/celltype_mri/spatial_nulls.py`). Moran spectral randomisation (Wagner & Dray, 2015) with a distance-based weight matrix built from structure centroids, producing surrogate contrast maps with matched Moran's I; permutation p-values for every coefficient and R^2. Variogram-matched surrogates (BrainSMASH-style) as a second null.
5. **Replication** (`src/celltype_mri/regress.py`). Fit the same models per atlas; compare importance rankings (Kendall's tau) and coefficients (concordance correlation).
6. **Tools**: `abc_atlas_access`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `nibabel`, `pynrrd`, `antspyx` (registration), `statsmodels`.

## Evaluation & statistics

- Primary estimands: per-contrast R^2 (leave-region-out), relative importance per cell class with bootstrap CIs, spatial-null p-values, cross-atlas Kendall's tau.
- Leakage: region-level CV holds out whole regions (and whole major divisions in the stricter scheme); the gene-program model is compared on the same folds; no atlas is used both to select regions and to evaluate.
- Multiple comparisons: five pre-registered hypotheses; subclass-level importance tests BH-corrected within contrast.
- Nulls: Moran spectral randomisation (1,000 surrogates) and variogram-matched surrogates; naive permutation reported for comparison to show inflation.
- Sensitivity: class vs. subclass resolution; density vs. proportion (compositional) parameterisation (centred log-ratio); including/excluding white-matter and ventricular structures; MERFISH section-coverage weights; registration jitter (re-run with 50 um random displacements).

## Publishable angle

- **Headline**: "Across N CCF regions and M MRI contrasts, counted oligodendrocyte density explains X% of T1w:T2w and FA variance, neuron density explains Y% of MD, and a Z-gene iron/myelin program adds W% for T2*-weighted contrast; results replicate between ex vivo and in vivo atlases and survive spatial nulls." Plus a public table of regional cell-type densities aligned with regional MRI contrasts, a resource for interpreting mouse MRI phenotypes (e.g. from mutant-line imaging consortia).
- Target venues: *NeuroImage*, *Magnetic Resonance in Medicine*, *Nature Communications* (if replication and gene-program results are strong), *Imaging Neuroscience*.
- Follow-ups: apply the fitted forward model to interpret MRI changes in disease models with cell-type resolution; extension to human with spatial transcriptomics as it becomes brain-wide; using the model as a prior for cell-type-resolved MRI segmentation.

## Risks, confounds & mitigations

- **Registration error** between MRI atlases and CCF (different fixation, shrinkage). Mitigation: region-level analysis is tolerant; registration jitter sensitivity; exclude thin structures.
- **MERFISH sampling** (coronal sections with gaps, section-level QC drop-outs). Mitigation: densities normalised by sampled volume per structure; exclude structures with < 200 cells.
- **Compositional data**: densities of all classes sum to total cell density; use centred log-ratio transforms as a sensitivity analysis.
- **Ex vivo contrast** depends on fixation and contrast agent; treat atlases separately, replicate rather than pool.
- **Collinearity** between cell classes (oligodendrocytes and white-matter fraction). Mitigation: ridge + relative importance; report partial R^2; exclude fibre tracts in a sensitivity analysis.
- **Panel limits**: the MERFISH panel may lack some myelin/iron genes; use Allen ISH grids as a fallback for H3.

## Milestones

- [ ] ABC Atlas metadata download; per-structure density table (class and subclass) with coverage QC.
- [ ] MRI atlases collected and registered to CCF; regional contrast table per atlas.
- [ ] Ridge/PLS models with leave-region-out CV; relative importance (H1, H2).
- [ ] Spatial nulls implemented and calibrated on synthetic maps; p-values for all models.
- [ ] Gene-program comparison (H3).
- [ ] Cross-atlas replication (H4); voxel-level models (H5).
- [ ] Preprint, aligned density-contrast table and code released.

## Quick start

```bash
cd projects/celltype-composition-mri-contrast
pip install -r requirements.txt
python scripts/download_data.py --ccf --structures --resolution 25   # annotation + ontology (plain HTTP)
python scripts/download_data.py --abc --sample                       # MERFISH cell metadata (needs abc-atlas-access)
python scripts/download_data.py --dsurqe                             # ex vivo template + labels
PYTHONPATH=src pytest -q                                             # densities, contrasts, regression, spatial nulls
```

Regional model with spatial nulls once the tables exist:

```python
import numpy as np, pandas as pd, nrrd
from celltype_mri.abc_atlas import assign_structures, sampled_volume_mm3, densities_by_structure, log_density
from celltype_mri.mri import load_volume, contrast_table, structure_centroids, align_tables
from celltype_mri.regress import fit_cv, relative_importance
from celltype_mri.spatial_nulls import distance_weights, spatial_pvalue

ann, _ = nrrd.read("data/ccf/annotation_25.nrrd")
cells = pd.read_parquet("data/abc_atlas/tables/cells.parquet")
cells["structure_id"] = assign_structures(cells[["x_ccf", "y_ccf", "z_ccf"]].to_numpy(), ann, 25.0)
vol = sampled_volume_mm3(ann, 25.0)
dens = log_density(densities_by_structure(cells, "structure_id", "class", vol, min_cells=200))
contr = contrast_table({"t2w": load_volume("data/mri/dsurqe/t2w_ccf25.nii.gz")}, ann, dens.index)
X, Y = align_tables(dens, contr)
res = fit_cv(X, Y["t2w"], "ridge", cv="loo")
imp = relative_importance(X, Y["t2w"])
W = distance_weights(structure_centroids(ann, X.index, 25.0).to_numpy(), k=10)
print(res.r2_cv, imp.sort_values(ascending=False).head())
print(spatial_pvalue(lambda y: fit_cv(X, pd.Series(y, index=X.index)).r2_cv, Y["t2w"].to_numpy(), W, n_surr=500))
```

## Pre-registered analysis table

| # | Unit of analysis | Primary outcome | Estimand / test | Decision rule | Confirmatory / exploratory |
|---|---|---|---|---|---|
| H1 | CCF structure (~300) | T1w:T2w, MT, FA | LMG relative importance of class densities; MSR spatial p | oligodendrocyte share > 40%, p_spatial < 0.01 | confirmatory |
| H2 | CCF structure | MD, neurite-density index | ridge leave-region-out R^2; partial R^2 for neuron density | neuron density partial R^2 > glial; p_spatial < 0.01 | confirmatory |
| H3 | CCF structure | T2*/QSM-like vs. T1w:T2w | nested CV comparison: densities vs. densities + gene programs | delta R^2 > 0.10 for T2*, not for T1w:T2w | confirmatory |
| H4 | atlas (ex vivo vs. in vivo) | importance ranking | Kendall's tau, Lin's CCC | tau > 0.6 | confirmatory |
| H5 | voxel (50 um) vs. structure | R^2 | partial-volume-adjusted comparison | structure R^2 > voxel R^2; gap smaller for myelin-driven contrasts | confirmatory |
| S1 | structure | subclass-level importance | BH within contrast | descriptive | exploratory |

Cell classes entering the primary model (class level of the ABC taxonomy, aggregated to <= 12 predictors): glutamatergic neurons (cortical + subcortical grouped), GABAergic neurons, other neurons (e.g. dopaminergic/serotonergic grouped), oligodendrocytes, OPCs, astrocytes/ependymal, microglia/immune, vascular.

## Key references

- Yao Z et al. (2023) A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain. *Nature*.
- Zhang M et al. (2023) Molecularly defined and spatially resolved cell atlas of the whole mouse brain. *Nature*.
- Wang Q et al. (2020) The Allen Mouse Brain Common Coordinate Framework: a 3D reference atlas. *Cell*.
- Dorr AE, Lerch JP, Spring S, Kabani N, Henkelman RM (2008) High resolution three-dimensional brain atlas using an average magnetic resonance image of 40 adult C57Bl/6J mice. *NeuroImage*.
- Ullmann JFP, Watson C, Janke AL, Kurniawan ND, Reutens DC (2013) A segmentation protocol and MRI atlas of the C57BL/6J mouse neocortex. *NeuroImage* (AMBMC).
- Wu D et al. (2013) In vivo high-resolution diffusion tensor imaging of the mouse brain. *NeuroImage*.
- Fulcher BD, Murray JD, Zerbi V, Wang X-J (2019) Multimodal gradients across mouse cortex. *PNAS*.
- Ero C, Gewaltig M-O, Keller D, Markram H (2018) A cell atlas for the mouse brain. *Front Neuroinform*.
- Liang Z et al. (2022) Virtual mouse brain histology from multi-contrast MRI via deep learning. *eLife*.
- "High-resolution MRI guided whole mouse brain cell type atlas using deep learning" (2025) *bioRxiv* (inverse-direction deep-learning model; motivates H5).
- Burt JB, Helmer M, Shinn M, Anticevic A, Murray JD (2020) Generative modeling of brain maps with spatial autocorrelation. *NeuroImage*.
- Markello RD, Misic B (2021) Comparing spatial null models for brain maps. *NeuroImage*.
- Wagner HH, Dray S (2015) Generating spatially constrained null models for irregularly spaced data using Moran spectral randomization methods. *Methods Ecol Evol*.
- Tustison NJ et al. (2025) The ANTsX ecosystem for mapping the mouse brain. *Nat Commun*.

## Ethics / data-use notes

- All data are open animal atlases; cite the ABC Atlas papers, the Allen Institute terms of use, and each MRI atlas paper.
- No human or identifiable data are involved.
- Do not commit atlas volumes or cell tables; `data/` and `outputs/` are git-ignored.
