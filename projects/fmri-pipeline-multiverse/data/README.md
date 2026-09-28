# Data acquisition — `fmri-pipeline-multiverse`

All datasets are open (OpenNeuro, CC0). Nothing under `data/` is committed. Layout after download:

```
data/
├── ds001734/                              # NARPS
│   ├── participants.tsv, task-MGT_bold.json
│   ├── sub-001/func/sub-001_task-MGT_run-01_events.tsv
│   └── derivatives/fmriprep/sub-001/func/
│       ├── sub-001_task-MGT_run-01_desc-confounds_regressors.tsv   (fMRIPrep 1.1.4 naming)
│       └── sub-001_task-MGT_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
├── ds002785/                              # AOMIC-PIOP1 (fMRIPrep 1.3.2 shipped)
├── ds002790/                              # AOMIC-PIOP2
├── ds000030/                              # UCLA CNP (fMRIPrep shipped)
├── ds000117/, ds000228/                   # verify derivatives first (--check)
├── parcellations/                         # Schaefer 200/400 in MNI152NLin2009cAsym (templateflow)
└── rerun/<dataset>/fmriprep-<version>/    # your own fMRIPrep re-runs for the version arm
```

## 1. Check which datasets ship fMRIPrep derivatives

```bash
python scripts/download_data.py --check
```
This queries the OpenNeuro GraphQL API (`https://openneuro.org/crn/graphql`) for a `derivatives/fmriprep*` directory
in the latest snapshot. The registry in `src/fmri_multiverse/fetch.py` marks NARPS (ds001734), UCLA CNP (ds000030)
and the AOMIC datasets (ds002785 / ds002790 / ds003097) as shipping fMRIPrep outputs; ds000117 and ds000228 are
marked *verify*. The **OpenNeuroDerivatives** GitHub organisation (<https://github.com/OpenNeuroDerivatives>) hosts
additional fMRIPrep/MRIQC derivative datasets (`<dsid>-fmriprep`) that can be installed with DataLad.

## 2. Download (three equivalent routes)

```bash
# openneuro-py (resumable, glob filters)
pip install openneuro-py
openneuro-py download --dataset ds002785 --include participants.tsv --include "sub-*/func/*events.tsv" \
    --include "derivatives/fmriprep/sub-*/func/*task-workingmemory*desc-confounds_*.tsv" \
    --include "derivatives/fmriprep/sub-*/func/*task-workingmemory*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

# AWS CLI, anonymous
aws s3 sync --no-sign-request s3://openneuro.org/ds001734 data/ds001734 --exclude "*" \
    --include "participants.tsv" --include "*events.tsv" --include "derivatives/fmriprep/*desc-confounds_*.tsv"

# DataLad
datalad install -s https://github.com/OpenNeuroDatasets/ds001734.git data/ds001734
datalad get data/ds001734/derivatives/fmriprep/sub-001

# or through the project script
python scripts/download_data.py --dataset ds002785 --task workingmemory --bold --n-subjects 50
```

Approximate sizes: confounds + events for all subjects < 1 GB per dataset; preprocessed BOLD in MNI space
~0.5–1 GB per run (NARPS: 108 subjects × 4 runs ≈ 300 GB; AOMIC-PIOP1 workingmemory ≈ 60 GB).

## 3. Parcellations

Schaefer 2018 (200 / 400 parcels, 7 networks) in `MNI152NLin2009cAsym` from TemplateFlow:
```python
from templateflow import api as tf
tf.get("MNI152NLin2009cAsym", atlas="Schaefer2018", desc="200Parcels7Networks", resolution=2)
```
Parcel time series are extracted with `nilearn.maskers.NiftiLabelsMasker` (no smoothing for the parcel arm; smoothing
is a voxel-wise-arm factor).

## 4. fMRIPrep version arm (your own compute)

Re-run a subset (e.g. 30 subjects of NARPS and PIOP1) with two fMRIPrep releases via containers:
```bash
docker run --rm -v $PWD/data/ds001734:/data:ro -v $PWD/data/rerun/ds001734/fmriprep-20.2.7:/out \
    -v $FS_LICENSE:/opt/freesurfer/license.txt nipreps/fmriprep:20.2.7 /data /out participant \
    --participant-label 001 --output-spaces MNI152NLin2009cAsym:res-2 --use-aroma --fs-no-reconall
docker run ... nipreps/fmriprep:24.1.1 ... (AROMA via fmripost-aroma in >= 24)
```
Budget: ~8–16 CPU-h per subject; a 30-subject × 2-version arm ≈ 600–1,000 CPU-h.

## 5. Ground-truth results

* NARPS: nine hypotheses, per-team decisions and consensus maps are on OpenNeuro (ds001734 derivatives and the
  NARPS results repository referenced in Botvinik-Nezer et al., 2020) and NeuroVault collection *NARPS*.
* AOMIC: group-level contrast maps are included in each dataset's `derivatives/` (see Snoek et al., 2021).
* CNP: task descriptions and expected contrasts in Poldrack et al., 2016; NeuroVault collection *CNP*.
Store the published effect locations in `fetch.DATASETS[...].findings` before running the multiverse (a-priori ROIs).
