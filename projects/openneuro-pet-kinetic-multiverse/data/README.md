# Data acquisition — openneuro-pet-kinetic-multiverse

All primary data are open PET-BIDS datasets on OpenNeuro (CC0). Nothing under `data/` is committed.

## 1. Discover PET datasets (small, scripted)

```bash
pip install requests
python scripts/download_data.py --discover            # → data/index/pet_datasets.csv
python scripts/download_data.py --discover --sample   # first 5 PET datasets
```

The script pages the OpenNeuro GraphQL API (`https://openneuro.org/crn/graphql`) with `modality: "pet"`, then fetches, per dataset, `dataset_description.json`, `participants.tsv` and every `*_pet.json` sidecar to record `TracerName`, `TracerRadionuclide`, number of frames, `FrameDuration`, `ImageDecayCorrected`, `InjectedRadioactivity` and whether `*_blood.tsv` files exist. It classifies datasets as *dynamic* (frames > 1) and *reference-region tracer* (keyword list in `pet_multiverse.tacs.REFERENCE_TRACERS`, editable).

The PET portal listing is also browsable at https://openneuro.org/pet.

## 2. Download selected datasets (large)

```bash
# openneuro-py
pip install openneuro-py
openneuro-py download --dataset ds004869 --target-dir data/openneuro/ds004869

# DataLad
datalad install https://github.com/OpenNeuroDatasets/ds004869.git data/openneuro/ds004869
datalad get data/openneuro/ds004869

# S3 (anonymous)
aws s3 sync --no-sign-request s3://openneuro.org/ds004869 data/openneuro/ds004869
```

`python scripts/download_data.py --fetch ds004869` wraps the S3 route (requires `boto3`) and by default fetches `pet/`, `anat/` and top-level metadata only.

Layout:

```
data/openneuro/<dsid>/sub-01/[ses-01/]pet/sub-01_pet.nii.gz, sub-01_pet.json, sub-01_recording-*_blood.tsv
data/openneuro/<dsid>/sub-01/[ses-01/]anat/sub-01_T1w.nii.gz
```

## 3. Regional TACs

Preferred: PETPrep (https://github.com/nipreps/petprep) or petsurfer-bids (https://github.com/freesurfer/petsurfer-bids) — both write BIDS derivatives with `*_desc-<atlas>_tacs.tsv` (one column per region, plus `frame_start`/`frame_end` or `FrameTimesStart` in the sidecar). Example:

```bash
docker run --rm -v $PWD/data/openneuro/ds004869:/data:ro -v $PWD/data/derivatives/ds004869:/out \
    nipreps/petprep:latest /data /out participant --fs-license-file /opt/license.txt
```

Fallback: `pet_multiverse.tacs.extract_tacs(pet_4d, label_img, labels)` computes mean TACs from a 4D NIfTI and a label volume in the same space (e.g., FreeSurfer `aparc+aseg` coregistered to PET).

Store TACs as `data/tacs/<dsid>/sub-XX[_ses-YY]_tacs.tsv` with columns `frame_start_min, frame_end_min, <region>...`.

## 4. Optional replication set

Cimbi database (NRU Copenhagen): apply at https://nru.dk/index.php/cimbi-database. Data are delivered under a DUA; place TACs under `data/tacs/cimbi/` and never commit them.

## 5. Sample mode without network

`python scripts/download_data.py --simulate` writes simulated test-retest TACs for 12 subjects × 2 sessions × 6 regions (`data/sample/`) using the SRTM forward model with realistic frame timing, so the multiverse runner can be exercised offline.
