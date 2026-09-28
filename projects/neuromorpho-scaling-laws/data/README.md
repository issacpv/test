# Data acquisition: neuromorpho-scaling-laws

Everything here is open access; no registration is required. Nothing in `data/` is committed.

## 1. NeuroMorpho.org REST API v1

Base URL: `https://neuromorpho.org/api` (reference: https://neuromorpho.org/apiReference.html, how-to: https://neuromorpho.org/api.jsp).

Endpoints used:

| Endpoint | Purpose |
|---|---|
| `GET /neuron?page=P&size=S` | all neurons, paginated (`size` ≤ 500, `page` starts at 0) |
| `GET /neuron/select?q=field:value&fq=field2:value2&page=P&size=S` | filtered listing (GET form) |
| `POST /neuron/select?page=P&size=S` with JSON body `{"species": ["mouse"], "brain_region": ["neocortex"]}` | filtered listing (POST form, multiple values per field) |
| `GET /neuron/fields` and `GET /neuron/fields/{field}` | list of queryable fields / distinct values of a field |
| `GET /neuron/name/{neuron_name}` , `GET /neuron/id/{neuron_id}` | one record |
| `GET /morphometry/name/{neuron_name}` , `POST /morphometry/select` | NeuroMorpho's precomputed L-Measure summary morphometrics |
| `https://neuromorpho.org/dableFiles/{archive_lowercase}/CNG version/{neuron_name}.CNG.swc` | standardised SWC file (space in path must be URL-encoded) |

Responses are Spring-HATEOAS style: `{"_embedded": {"neuronResources": [...]}, "page": {"size", "totalElements", "totalPages", "number"}}`.

Batch-effect metadata fields returned per neuron (verify against `/neuron/fields`): `archive`, `reconstruction_software`, `shrinkage_reported`, `shrinkage_corrected`, `protocol`, `slicing_thickness`, `slicing_direction`, `objective_type`, `magnification`, `stain`, `original_format`, `physical_Integrity`, `structural_domains`, `experiment_condition`, `deposition_date`, `upload_date`; biology fields: `species`, `scientific_name`, `strain`, `brain_region` (list), `cell_type` (list), `age_classification`, `min_age`, `max_age`, `gender`, `domain`; provenance: `reference_pmid`, `reference_doi`.

### Commands

```bash
cd projects/neuromorpho-scaling-laws
pip install -r requirements.txt

# quick smoke test: ~25 neurons for each of a few species + 5 SWC files each
python scripts/download_data.py --sample

# full metadata harvest for selected species (metadata only, no SWC)
python scripts/download_data.py --species mouse rat human monkey cat rabbit --metadata-only

# everything (all species), with L-Measure morphometry and SWC files
python scripts/download_data.py --all --morphometry --swc --workers 4
```

Tips:
- If TLS verification fails on neuromorpho.org (their certificate chain has been intermittently misconfigured), set `NEUROMORPHO_VERIFY_SSL=0` for the session and note it in your lab notebook. Never disable verification globally.
- The script is resumable: existing JSONL pages and SWC files are skipped.
- Use `--max-pages` during development.

## 2. TimeTree divergence times

Go to http://timetree.org, use "Get divergence time for a pair of taxa" or upload a species list (one Latin name per line, e.g. `Mus musculus`, `Rattus norvegicus`, `Homo sapiens`, ...) to obtain a Newick tree with branch lengths in Myr. Save it as `data/phylo/timetree_species.nwk`. `nm_scaling.phylo_gls.newick_to_covariance` converts it to a Brownian covariance matrix. The `scientific_name` field of NeuroMorpho records gives the Latin name for matching.

## 3. Species-level covariates

Brain mass and neuron counts per species: take the supplementary tables of Herculano-Houzel and colleagues (e.g. PNAS 2006/2007 rodent and primate scaling papers, and later reviews). Store as `data/species_covariates.csv` with columns `scientific_name, brain_mass_g, n_neurons, source`.

## 4. Optional: Allen Cell Types Database (single-protocol validation set)

```bash
pip install allensdk
python - <<'EOF'
from allensdk.core.cell_types_cache import CellTypesCache
ctc = CellTypesCache(manifest_file='data/allen/manifest.json')
cells = ctc.get_cells(require_reconstruction=True)
print(len(cells))
for c in cells[:5]:
    ctc.get_reconstruction(c['id'])  # downloads SWC into the cache
EOF
```

## Expected layout

```
data/
  README.md
  metadata/
    neurons_mouse.jsonl          # one JSON record per neuron, as returned by the API
    neurons_rat.jsonl
    ...
    neurons.parquet              # flattened table (built by the script)
    morphometry.parquet          # L-Measure summary per neuron_name (optional)
  swc/
    <archive>/<neuron_name>.CNG.swc
  phylo/
    timetree_species.nwk
  species_covariates.csv
  allen/                         # optional
```
