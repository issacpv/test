# Data acquisition

## 1. Open-access literature via Europe PMC (open, no key)

REST API: https://europepmc.org/RestfulWebService

- Search: `GET https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=<q>&format=json&pageSize=100&cursorMark=*`
- Full text: `GET https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML`

```bash
python scripts/download_data.py --europepmc --sample                    # 20 papers
python scripts/download_data.py --europepmc --max-papers 3000 \
   --query '("spine density" OR "dendritic spine") AND ("per micrometer" OR "per µm" OR "spines/µm" OR "per 10 µm" OR "per 100 µm") AND OPEN_ACCESS:Y AND HAS_FT:Y'
```

Outputs `data/europepmc/hits.jsonl` (search results) and `data/europepmc/xml/PMC*.xml`.
Respect article licences: the curated table you distribute should contain values + citations, not full text.

## 2. PubMed E-utilities (open) for abstract-level screening of non-OA papers

`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=...&retmode=json` then
`esummary.fcgi`/`efetch.fcgi`; add `&api_key=$NCBI_API_KEY` for higher rate limits (optional, free).

## 3. NeuroMorpho.org (open REST API)

```bash
python scripts/download_data.py --neuromorpho --sample                            # a few species, small pages
python scripts/download_data.py --neuromorpho --species mouse rat human --swc    # full metadata + SWC
```

Outputs `data/neuromorpho/metadata/neurons_<species>.jsonl`, `data/neuromorpho/swc/<archive>/<name>.CNG.swc`.
The audit (`spine_mining.swc_spines.audit_file`) is then run over every SWC.

## 4. EM connectomes with exhaustive spine counts (tables only; no image volumes needed)

- **MICrONS** (mouse V1, cubic millimetre): register for a CAVE token at https://global.daf-apis.com/auth/ (free),
  then `pip install caveclient` and query `client.materialize.synapse_query(...)` and the skeleton
  tables (`minnie65_public`). Tutorial: https://www.microns-explorer.org
- **H01** (human temporal cortex): https://h01-release.storage.googleapis.com/landing.html ; skeletons and
  synapse tables are public on Google Cloud (BigQuery / GCS); `pip install cloud-volume` for the segmentation.
- **Kasthuri 2015** (mouse S1): https://bossdb.org (project `kasthuri2015`); `pip install intern`.

Store derived per-dendrite counts under `data/em/<dataset>/dendrite_segments.parquet` with columns
`cell_id, compartment, distance_bin_um, length_um, n_spine_synapses, n_shaft_synapses`.

## 5. Human spine dataset (J Neurophysiol 2025)

Follow the paper's data-availability statement; place any released tables under `data/human_spines/`.

## Expected layout

```
data/
  README.md
  europepmc/
    hits.jsonl
    xml/PMC1234567.xml
    text/PMC1234567.txt            (extracted body text + captions)
  neuromorpho/
    metadata/neurons_<species>.jsonl
    metadata/neurons.parquet
    swc/<archive>/<name>.CNG.swc
    audit.parquet                  (one row per SWC: spine flags and densities)
  em/<dataset>/dendrite_segments.parquet
  human_spines/
  curated/
    statements_raw.parquet         (all extracted statements with provenance)
    annotations_gold.csv           (manual validation set)
    spine_density_curated.csv      (released table: value, unit-normalised density, method, species, region,
                                    cell type, compartment, n, SD, PMCID, sentence id)
```

Nothing under `data/` is committed.
