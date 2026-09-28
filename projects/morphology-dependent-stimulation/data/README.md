# Data acquisition

Nothing in this directory is committed (see `.gitignore`). Everything below is
reproducible from the commands here plus the manifest that
`scripts/download_data.py` writes.

Record the **retrieval date** for every source. NeuroMorpho.Org and the Allen
Cell Types Database both grow over time, so a cohort is only reproducible if the
query string and the date are both written down.

## Expected directory layout

```
data/
├── neuromorpho/
│   ├── manifest.json                 # written by scripts/download_data.py
│   ├── metadata/
│   │   ├── species_human__brain_region_neocortex__cell_type_pyramidal_interneuron.json
│   │   ├── species_mouse__brain_region_neocortex__cell_type_pyramidal_interneuron.json
│   │   └── species_human__brain_region_neocortex__experiment_condition_....json
│   └── swc/
│       ├── <neuron_name>.CNG.swc     # one file per reconstruction
│       └── ...
├── allen/
│   ├── manifest.json                 # allensdk CellTypesCache manifest
│   ├── specimens/<specimen_id>/reconstruction.swc
│   └── models/<neuronal_model_id>/   # all-active biophysical model package
├── head_models/
│   ├── simnibs_examples/ernie/m2m_ernie/ernie.msh
│   └── mida/                         # manual download, free registration
└── reference/
    └── aberra2019/                   # optional: Zenodo model neurons with full axons
```

## 1. NeuroMorpho.Org (open, no registration)

The only bulk download the script performs. The API is documented at
https://neuromorpho.org/api.jsp and the field reference at
https://neuromorpho.org/apiReference.html .

```bash
# Check the plan without touching the network
python scripts/download_data.py --dry-run

# Small sample first - 15 reconstructions per cohort, ~1 minute
python scripts/download_data.py --sample --out data

# Full open cohorts (human cortical, mouse cortical)
python scripts/download_data.py --out data

# Everything including the disease/aging arm
python scripts/download_data.py --all --out data

# Metadata only, e.g. to size a cohort before committing to the download
python scripts/download_data.py --metadata-only --cohort human
```

What the script does, in API terms:

```
GET https://neuromorpho.org/api/health
GET https://neuromorpho.org/api/neuron/select?q=species:human&q=brain_region:neocortex\
        &q=cell_type:pyramidal,interneuron&page=0&size=100
    -> {"_embedded": {"neuronResources": [...]}, "page": {"totalPages": N, ...}}
    (iterate page=0..N-1; comma-separated values within one q are ORed,
     separate q parameters are ANDed; size is capped at 500 by the server)

GET https://neuromorpho.org/dableFiles/<archive-lowercased-nospaces>/CNG%20version/<neuron_name>.CNG.swc
```

Use the **CNG version** of each file, not the original submission: the CNG
pipeline re-roots, de-duplicates and unit-checks the morphology, which matters
when a sweep must run unattended over files from dozens of labs.

If the `dableFiles` path 404s for a record, the archive directory name is the
likely culprit (a few archives have unusual capitalisation or punctuation). Cross-
check against the record's own fields, and fall back to the per-neuron endpoint:

```bash
curl -s "https://neuromorpho.org/api/neuron/name/<neuron_name>" | python -m json.tool
```

Be polite: the default 0.5 s inter-request delay in `NeuroMorphoClient` exists
because this is a shared academic service. Do not parallelize the crawl.

### Inclusion criteria applied after download

Applied in analysis code, not at download time, so the exclusions are auditable:

- ≥ 200 SWC nodes and ≥ 500 µm total neurite length
- a single connected component after root repair
- < 5% duplicate coordinates
- `archive` recorded (it is the grouping variable for every statistical model)
- axon completeness (`frac_axon_nodes`) recorded and used to stratify — **not** to
  exclude, since most NeuroMorpho cortical reconstructions lack full axons

## 2. Allen Cell Types Database (open, no registration)

Used for (a) systematically sampled human and mouse morphologies as a check on
NeuroMorpho's selection bias, and (b) all-active biophysical model packages for
the NEURON calibration arm.

```bash
pip install allensdk    # consider a separate venv; it pins numpy/pandas tightly
```

```python
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.biophysical_api import BiophysicalApi

ctc = CellTypesCache(manifest_file="data/allen/manifest.json")

# Human cells (~413 available); use ['Mus musculus'] for mouse (~1920)
cells = ctc.get_cells(species=["Homo Sapiens"])
for cell in cells:
    if cell.get("reconstruction_type"):          # not every cell has a morphology
        ctc.get_reconstruction(cell["id"])       # -> SWC under data/allen/
ctc.get_ephys_features()
ctc.get_morphology_features()

# One biophysical model package, ready to run under NEURON.
# Find model ids at https://celltypes.brain-map.org with
# Models -> 'Biophysical - all active' checked.
BiophysicalApi().cache_data(<neuronal_model_id>, working_directory="data/allen/models/<id>")
```

The model packages ship `.mod` files; compile once per machine with `nrnivmodl`
in the package directory before using `morph_stim.neuron_driver` with them.

## 3. Head models (only for the field-direction prior, step 5)

**SimNIBS example dataset** (open, ships with SimNIBS ≥ 4):

```bash
pip install simnibs          # or the platform installer, which is the supported route
# The example dataset ('ernie' subject + MNI152 head + a spherical model) is
# listed under 'Datasets' in the docs:
#   https://simnibs.github.io/simnibs/build/html/dataset.html
# -> data/head_models/simnibs_examples/ernie/m2m_ernie/ernie.msh
```

To build a head model from your own T1/T2 instead:

```bash
charm ernie org/ernie_T1.nii.gz org/ernie_T2.nii.gz   # creates m2m_ernie/
```

**MIDA** (free registration with the IT'IS Foundation): download the NIfTI label
volume manually from
https://itis.swiss/virtual-population/regional-human-models/mida-model/ into
`data/head_models/mida/`. Not redistributable — do not mirror it into this repo.

## 4. Optional reference morphologies with complete axons

The Aberra et al. TMS model neurons are archived on Zenodo
(https://zenodo.org/records/2488573) and are the best available bound on how much
the missing-axon problem biases NeuroMorpho-derived thresholds. Download manually
into `data/reference/aberra2019/`.

## Credentialed sources

This project uses none. If the programme is later extended with credentialed
data, read credentials from environment variables and never from a file in the
repository:

```bash
export PHYSIONET_USER=...      # example pattern only; not used by this project
export PHYSIONET_PASS=...
```

## Provenance checklist before analysis

- [ ] `data/neuromorpho/manifest.json` exists and its counts match the SWC file count
- [ ] Retrieval date recorded in the lab notebook and in the manuscript methods
- [ ] Per-archive counts tabulated, with the human/mouse crosstab, so the
      species-versus-archive confound is visible from the start
- [ ] Axon-completeness distribution plotted before any threshold is computed
- [ ] Source publications for each archive pulled via
      `GET https://neuromorpho.org/api/literature/...` for the citation list
