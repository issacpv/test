# Fixed split manifests

One CSV per (task, domain): `manifests/<task>/<domain>_splits.csv` with columns `group,split` where `group` is the leakage unit (subject / patient / stay id, already public in the source dataset) and `split` is `train`, `val` or `test`. These files contain **no data**, only identifiers, and are the only benchmark artefact that is versioned: a leaderboard entry is valid only if it names the manifest version it used.

Generate them with `python scripts/download_data.py --make-manifests` once the caches exist, or with `bioshift.adapters.write_split_manifest`. Where an official split exists (PTB-XL folds 1-8/9/10; TUSZ train/dev/eval; EchoNet-Dynamic `FileList.csv` split column) the manifest reproduces it instead of drawing a random one.
