# Data acquisition

Nothing in this folder is committed except this file. Expected total: 150-300 GB for the sessions used.

## Expected layout

```
data/
  allen/visual-coding/
    cache/sessions.csv probes.csv channels.csv units.csv
    session_<id>/session_<id>.nwb                 # stimulus tables, units, running, pupil (~2.9 GB)
    session_<id>/probe_<probe_id>_lfp.nwb         # LFP + flash-evoked CSD (~2 GB each)
    session_<id>/session_<id>_analysis_metrics.csv
  allen/visual-behavior/
    behavior_ecephys_sessions/<id>/ecephys_session_<id>.nwb  probe_probeA_lfp.nwb ...
  dandi/000166/                                   # Senzai et al. 2019 mouse V1 laminar (open)
  manifests/                                      # probe/layer manifests (safe to commit)
```

## 1. Allen Visual Coding Neuropixels (open; S3 `allen-brain-observatory`)

Verified S3 layout (us-west-2, no credentials):

```
visual-coding-neuropixels/ecephys-cache/sessions.csv | probes.csv | channels.csv | units.csv | manifest.json
visual-coding-neuropixels/ecephys-cache/session_715093703/session_715093703.nwb                 (2.86 GB)
visual-coding-neuropixels/ecephys-cache/session_715093703/probe_810755797_lfp.nwb              (~2 GB)
visual-coding-neuropixels/ecephys-cache/session_715093703/session_715093703_analysis_metrics.csv
```

```bash
python scripts/download_data.py --cache                       # the four CSVs
python scripts/download_data.py --sessions 5                  # session NWBs for the first 5 brain_observatory_1.1 sessions
python scripts/download_data.py --session 715093703 --lfp     # + all probe LFP NWBs of one session
python scripts/download_data.py --sample                      # CSVs + analysis_metrics.csv of one session (small)
```

Or with the Allen SDK (handles the same bucket via its manifest):

```python
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
cache = EcephysProjectCache.from_warehouse(manifest="data/allen/visual-coding/manifest.json")
session = cache.get_session_data(715093703)
csd = session.get_current_source_density(810755797)      # xarray: (virtual_channel_index x time), vertical_position coord
units = cache.get_units()                                  # includes structure acronyms and peak channels
```

Useful columns: `channels.csv` -> `probe_vertical_position`, `probe_horizontal_position`,
`ecephys_structure_acronym`, CCF coordinates; `probes.csv` -> `surface_channel_index`, `sampling_rate`,
`lfp_sampling_rate`; `units.csv` -> `peak_channel_id`, QC metrics (`isi_violations`, `amplitude_cutoff`,
`presence_ratio`), waveform metrics.

## 2. Allen Visual Behavior Neuropixels (open; S3 `visual-behavior-neuropixels-data`)

```
visual-behavior-neuropixels/behavior_ecephys_sessions/<session_id>/ecephys_session_<session_id>.nwb
visual-behavior-neuropixels/behavior_ecephys_sessions/<session_id>/probe_probeA_lfp.nwb ... probeF
```

```bash
python scripts/download_data.py --vbn-session 1043752325
```

or `VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=...)` from `allensdk`.

## 3. DANDI:000166 (open) - Senzai, Fernandez-Ruiz & Buzsaki 2019

```bash
pip install dandi
dandi download DANDI:000166 --output-dir data/dandi          # 19 NWB files, ~787 GB
```

Stream a single file instead of downloading (see the DANDI docs on `fsspec` + `h5py`). The electrode table
carries depth; the paper's laminar landmarks (depth of maximal spike power, CSD sink/source) are recomputed
with `lamlat.layers`.
