# Data acquisition

Nothing in this folder is committed except this file. Raw AP-band data are very large; extract fixed time
windows with HTTP byte-range requests instead of downloading whole files.

## Expected layout

```
data/
  allen/
    cache/                              # visual-coding-neuropixels/ecephys-cache/*.csv (+ optional NWB)
      sessions.csv probes.csv channels.csv units.csv
      session_<id>/session_<id>.nwb     # stimulus tables + released Kilosort-2 units (optional, ~2 GB each)
    raw/<session_id>/<probe_id>/
      spike_band.dat                    # full file (~197 GB) OR
      spike_band_t<start>-<end>s.dat    # windowed extraction (int16, 384 ch, 30 kHz)
      channel_states.npy event_timestamps.npy
      geometry_np1.csv                  # written by scripts/download_data.py
  dandi/000034/                         # SpikeInterface-paper recordings (open)
  ibl/<eid>/raw_ephys_data/probe00/     # _spikeglx_ephysData_g0_t0.imec0.ap.cbin/.ch/.meta via ONE
  manifests/raw_probes.csv              # produced by --list (safe to commit)
```

## 1. Allen Visual Coding Neuropixels raw data (open, AWS S3, no credentials)

Bucket `allen-brain-observatory` (us-west-2). Verified layout:

```
visual-coding-neuropixels/raw-data/<session_id>/<probe_id>/spike_band.dat     (int16, 384 x 30 kHz)
visual-coding-neuropixels/raw-data/<session_id>/<probe_id>/lfp_band.dat       (int16, 384 x 2.5 kHz)
visual-coding-neuropixels/raw-data/<session_id>/<probe_id>/channel_states.npy
visual-coding-neuropixels/raw-data/<session_id>/<probe_id>/event_timestamps.npy
visual-coding-neuropixels/ecephys-cache/sessions.csv | probes.csv | channels.csv | units.csv
```

```bash
# manifest of all raw probes with sizes (paginated S3 listing)
python scripts/download_data.py --list
# cache CSVs (sessions/probes/channels; add --units for units.csv)
python scripts/download_data.py --cache
# a 2-second slice of one probe (byte-range request, ~46 MB) + geometry file: end-to-end smoke test
python scripts/download_data.py --sample
# a 40-minute window of one probe (~55 GB)
python scripts/download_data.py --session 715093703 --probe 810755797 --t-start 600 --t-end 3000
```

Equivalent with the AWS CLI (no sign-in needed):

```bash
aws s3 ls --no-sign-request s3://allen-brain-observatory/visual-coding-neuropixels/raw-data/715093703/
aws s3 cp --no-sign-request s3://allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/probes.csv data/allen/cache/
```

Notes:

- Sample/probe timing: `event_timestamps.npy` and `channel_states.npy` are the Open Ephys sync events for the
  probe; the NWB session file (`session_<id>.nwb`, via `allensdk`'s `EcephysProjectCache`) has the stimulus
  tables in the master clock. Align windows to stimulus blocks using the session NWB and the probe's
  sampling-rate/offset from `probes.csv` (`sampling_rate` column) - the released `channels.csv` /
  `units.csv` give the probe-to-master time alignment used by Allen.
- Gain: Neuropixels 1.0 AP band at gain 500 -> 2.34375 uV/bit (1.2 V range / 1024 / 500). See
  `sortverse.raw_io.to_microvolts`.
- The `sessions.csv` header is `id,date_of_acquisition,isi_experiment_id,published_at,specimen_id,session_type,age_in_days,sex,genotype,has_nwb`;
  `probes.csv` has `id,air_channel_index,ecephys_session_id,lfp_sampling_rate,lfp_temporal_subsampling_factor,name,phase,sampling_rate,surface_channel_index,has_lfp_data`.

## 2. DANDI:000034 (open)

```bash
pip install dandi
dandi download DANDI:000034 --output-dir data/dandi        # 6 files, ~74 GB
```

These are the recordings analysed in Buccino et al. (2020, eLife); use them to reproduce the published
agreement analysis as a sanity check of the harness before running on Allen data.

## 3. IBL raw AP data (open, ONE API)

```bash
pip install ONE-api
python - <<'EOF'
from one.api import ONE
one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
eids = one.search(atlas_acronym="VISp", datasets=["raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin"])
print(len(eids), eids[:3])
# one.load_dataset(eid, "raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin", download_only=True)
EOF
```

The compressed `.cbin` needs `ibl-neuropixel` (`spikeglx.Reader`) or SpikeInterface's `read_cbin_ibl` to
decompress. Dataset names may differ per probe/session; list with `one.list_datasets(eid)`.

## 4. Sorter containers

Install SpikeInterface with docker/singularity support. Kilosort 2.5 and 3 need the MATLAB runtime images
provided by SpikeInterface; Kilosort 4 is `pip install kilosort`. Record container digests and versions in
`outputs/manifest.json` for every run.
