# Data acquisition — layer-fmri-7t-reproducibility

All data live under `data/` (git-ignored).

## 1. OpenNeuro laminar datasets (open)

Curated accessions (verify on OpenNeuro; new ones appear regularly):

| Accession | Content |
|---|---|
| ds003216 | Kenshu: whole-brain layer-fMRI VASO + BOLD, 7T, one participant, 6 sessions, audio movie |
| ds001547 | Layer VASO in the visual system (V1), 7T |

```bash
python scripts/download_data.py --list                                   # curated list
python scripts/download_data.py --search laminar layer VASO "cortical depth"   # query OpenNeuro for more
python scripts/download_data.py --dataset ds001547 --sample              # one subject: bold/vaso + anat + json/tsv
python scripts/download_data.py --dataset ds003216 --include 'ses-01' --include 'anat'
# alternatives: openneuro-py download --dataset ds003216 ; aws s3 sync --no-sign-request s3://openneuro.org/ds003216 ...
```

Expected layout (BIDS):

```
data/openneuro/ds003216/sub-01/ses-01/func/sub-01_ses-01_task-movie_acq-vaso_run-01_bold.nii.gz   (names vary)
data/openneuro/ds003216/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz
data/openneuro/ds001547/sub-*/func/*.nii.gz
```

## 2. Donders Repository datasets (free registration; data-use agreement)

Kok et al. (2016, Current Biology) and Lawrence et al. (2019, eLife) 7T V1 layer data are hosted on the Donders Repository (https://data.donders.ru.nl/). Search for the paper title, accept the data-use agreement, and download the collection with the repository's WebDAV interface (`curl -u <user>` or `rclone`). Place under `data/donders/<collection>/`.

## 3. Author-shared datasets

- MPI CBS replication of Finn et al. (2019): see https://www.cbs.mpg.de/neurophysics/finn-et-al-replication for the data statement; request access if not yet public. Place under `data/mpi_cbs_finn_replication/`.
- Finn et al. (2019) original data: check the paper's data-availability statement / OpenNeuro.

## 4. Tools

- LayNii (https://github.com/layerfMRI/LAYNII): `LN2_LAYERS -rim rim.nii -nr_layers 6 -equivol` produces equidistant and equivolume layer files used as the reference implementation for `layerfmri_repro.layering`.
- The rim file (1 = WM boundary, 2 = pial boundary, 3 = GM) is generated from FreeSurfer/CAT12 segmentations in upsampled EPI space.

## Derived products

```
outputs/
  layers/<dataset>/<sub>_<ses>_<roi>_depth-equidist.nii.gz, *_depth-equivol.nii.gz
  profiles/<dataset>/<sub>_<ses>_<run>_<roi>_<spec>.tsv     # one row per depth bin
  multiverse/<paradigm>_specification_curve.parquet
  reliability/<dataset>_icc.csv, variance_components.csv
  degradation/<dataset>_accuracy_vs_tsnr.csv
```
