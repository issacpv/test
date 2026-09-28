# Data acquisition

Nothing in this folder is committed except this file. All four datasets are on public AWS S3 buckets (no
credentials). Expected total: 0.5-2 TB depending on how many experiments are cached.

## Expected layout

```
data/
  visual-coding-2p/
    cell_specimens.json  experiment_containers.json  manifest.json
    ophys_experiment_analysis/<experiment_id>_three_session_<A|B|C>_analysis.h5
    ophys_experiment_data/<experiment_id>.nwb            # optional (dF/F, events, stimulus tables)
  visual-coding-neuropixels/
    ecephys-cache/sessions.csv units.csv channels.csv probes.csv brain_observatory_1.1_analysis_metrics.csv
    ecephys-cache/session_<id>/session_<id>.nwb          # optional
  visual-behavior-ophys/
    behavior_ophys_experiments/behavior_ophys_experiment_<id>.nwb
  visual-behavior-neuropixels/
    behavior_ecephys_sessions/<id>/ecephys_session_<id>.nwb
  manifests/                                              # experiment/session selections (safe to commit)
```

## 1. Visual Coding 2-photon (open; S3 `allen-brain-observatory`, prefix `visual-coding-2p/`)

Verified keys: `visual-coding-2p/cell_specimens.json`, `visual-coding-2p/experiment_containers.json`,
`visual-coding-2p/manifest.json`, `visual-coding-2p/ophys_experiment_analysis/<id>_three_session_<X>_analysis.h5`.

```bash
python scripts/download_data.py --dataset vc2p --sample          # json tables + 5 analysis h5 files
python scripts/download_data.py --dataset vc2p --analysis        # all analysis h5 files (~50 GB)
```

or with the Allen SDK:

```python
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file="data/visual-coding-2p/manifest.json")
cells = boc.get_cell_specimens()          # per-cell tuning metrics (OSI, DSI, pref TF/SF, sparseness, p-values)
exps = boc.get_ophys_experiments(stimuli=["drifting_gratings"])
ds = boc.get_ophys_experiment_data(exps[0]["id"])    # dF/F traces, stimulus tables, running speed
```

## 2. Visual Coding Neuropixels (open; same bucket, prefix `visual-coding-neuropixels/ecephys-cache/`)

```bash
python scripts/download_data.py --dataset vcnpx --sample     # CSVs + brain_observatory_1.1_analysis_metrics.csv
python scripts/download_data.py --dataset vcnpx --session 715093703
```

`brain_observatory_1.1_analysis_metrics.csv` contains the released per-unit tuning metrics (OSI/DSI, pref
TF/SF, lifetime sparseness, responsiveness p-values) computed by the Allen pipeline - the starting point for
matching the 2P `cell_specimens` metrics. The DANDI mirrors are DANDI:000021 (brain_observatory_1.1) and
DANDI:000022 (functional_connectivity).

## 3. Visual Behavior 2-photon (open; S3 `visual-behavior-ophys-data`, prefix `visual-behavior-ophys/`)

Verified keys: `visual-behavior-ophys/behavior_ophys_experiments/behavior_ophys_experiment_<id>.nwb`.

```bash
python scripts/download_data.py --dataset vbo --sample       # first 3 experiment NWBs
```

or `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=...)` from `allensdk`, which also provides the
`ophys_experiment_table` (Cre line, imaging depth, area, session type).

## 4. Visual Behavior Neuropixels (open; S3 `visual-behavior-neuropixels-data`, prefix `visual-behavior-neuropixels/`)

Verified keys: `visual-behavior-neuropixels/behavior_ecephys_sessions/<id>/ecephys_session_<id>.nwb` and
`probe_probe<A-F>_lfp.nwb`.

```bash
python scripts/download_data.py --dataset vbn --sample       # first session NWB (without LFP)
```

## 5. Ground-truth forward-model priors

Huang et al. (2021, eLife 10.7554/eLife.51675) released simultaneous loose-seal + imaging data for GCaMP6s/6f
transgenic lines; follow the paper's data-availability statement. Used only to set priors on the forward
model's kernel and nonlinearity parameters.
