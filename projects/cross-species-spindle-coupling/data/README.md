# Data acquisition

Nothing in this folder is committed except this file. Total volume if every session is cached: ~1.4 TB;
streaming a few LFP channels per session needs < 50 GB.

## Expected layout

```
data/
  dandi/
    000041/                      # Watson et al. 2016, rat frontal cortex sleep (open)
      dandiset.yaml  assets.yaml
      sub-BWRat17/sub-BWRat17_ses-BWRat17-121912_ecephys.nwb ...
    000978/                      # W-track rats, CA1 + PFC with sleep epochs (open)
    000166/                      # Senzai et al. 2019, mouse V1 laminar (open)
    000044/                      # Grosmark & Buzsaki 2016, rat CA1 rest (open)
  mni-ieeg-atlas/                # MNI Open iEEG Atlas (open after terms of use)
    <as downloaded: per-region .edf/.mat segments + channel table>
  sleep-edfx/                    # PhysioNet sleep-edfx 1.0.0 (open)
    sleep-cassette/SC4001E0-PSG.edf SC4001EC-Hypnogram.edf ...
  nsrr/cfs/ nsrr/mesa/           # optional, DUA
  manifests/                     # channel/session selections (safe to commit)
```

## 1. DANDI (open)

Metadata and asset lists are public. Two routes:

```bash
# (a) official CLI (recommended for whole files)
pip install dandi
dandi download DANDI:000041 --output-dir data/dandi            # whole dandiset (~155 GB)
dandi download https://dandiarchive.org/dandiset/000041/0.250624.0419/files?location=sub-BWRat17 \
    --output-dir data/dandi/000041                              # one subject

# (b) this repo's script: lists assets via the DANDI REST API (paginated), falls back to the public
#     S3 mirror of the dandiset metadata if the API is unreachable, and can fetch the smallest asset
python scripts/download_data.py --dataset dandi --dandiset 000041 --list
python scripts/download_data.py --dataset dandi --dandiset 000041 --sample     # smallest NWB (~1.5 GB)
```

Streaming instead of downloading (preferred for the >100 GB dandisets):

```python
import fsspec, h5py, pynwb
url = "https://dandiarchive.s3.amazonaws.com/blobs/<asset blob path>"   # from the asset's contentUrl
with fsspec.open(url, "rb") as f, h5py.File(f, "r") as h5:
    io = pynwb.NWBHDF5IO(file=h5, load_namespaces=True)
    nwb = io.read()
    lfp = nwb.processing["ecephys"]["LFP"]["ElectricalSeries"]   # names vary per dandiset
```

Dandiset-specific notes:

- **000041** (rat frontal cortex, 11 subjects, 22 sessions; NWB names `sub-<rat>_ses-<rat>-<date>_ecephys.nwb`). LFP and units; sleep states (WAKE/NREM/REM) are provided in the original Buzsaki-lab format and, in the NWB conversion, as interval tables. Verify the interval table names with `nwb.intervals.keys()` and `nwb.processing.keys()`.
- **000978** (8 rats, W-track, CA1 + PFC LFP, interleaved sleep). Use the sleep epochs; PFC channels for SO/spindles, CA1 for ripples.
- **000166** (mouse V1 laminar, 19 files). Electrode table gives depth; use the CSD landmarks from the paper (L4 sink) for laminar analysis.
- **000044** (4 rats, CA1, pre/post rest). Rest epochs contain NREM; cortical channels are limited - use for ripple-spindle nesting.

## 2. MNI Open iEEG Atlas (open, terms of use)

1. Visit https://mni-open-ieegatlas.research.mcgill.ca/ , accept the terms of use, and download the sleep atlas package (wake, NREM and REM segments per channel, with region labels and MNI coordinates).
2. Unpack to `data/mni-ieeg-atlas/`. Record the atlas version in `data/manifests/mni_version.txt`.
3. Use only channels labelled as belonging to normal (non-epileptogenic) regions; keep the region table for the frontal-vs-other analysis.

## 3. Sleep-EDF Expanded (open, PhysioNet)

```bash
python scripts/download_data.py --dataset sleep-edfx --sample   # SC4001/SC4002 PSG + hypnogram
python scripts/download_data.py --dataset sleep-edfx            # all 197 recordings (~8 GB)
# equivalent: wget -r -N -c -np -nH --cut-dirs=3 -P data/sleep-edfx https://physionet.org/files/sleep-edfx/1.0.0/
```

## 4. NSRR CFS / MESA (optional, DUA)

Request access at https://sleepdata.org, export `NSRR_TOKEN`, then `gem install nsrr` and
`nsrr download cfs/polysomnography/edfs --token=$NSRR_TOKEN` from `data/nsrr/`. See the NSRR docs for
the annotation XML format (profusion).

## 5. MASS SS2 (optional)

Register at http://ceams-carsm.ca/mass/ and sign the agreement; SS2 contains expert spindle annotations
used to calibrate the percentile thresholds in `xspindle.detect.detect_spindles`.
