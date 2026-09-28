#!/usr/bin/env python3
"""Optional 3D-CNN arm: MONAI DenseNet121 on skull-stripped, MNI-registered T1 volumes.

Input: a CSV with columns ``subject, path, amyloid_pos`` (and optionally ``cog_status``) produced from
:func:`amyloid_mri.oasis_tables.build_subject_table` after registration (e.g. ANTs to MNI152 at
1.5 mm, then cropped to 96×112×96). Splits are grouped by subject. Out-of-fold predictions are saved
so that they can be evaluated with :mod:`amyloid_mri.models` and :mod:`amyloid_mri.decision_curve`
exactly like the ROI models.

Usage
-----
    python scripts/train_cnn.py --table outputs/cnn_table.csv --out outputs/cnn --folds 5 --epochs 40

    # label-permutation null (repeat with different --seed and --permute-labels)
    python scripts/train_cnn.py --table ... --permute-labels --seed 1

Requires: torch, monai, nibabel (see requirements.txt, commented block).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from monai.data import DataLoader, Dataset
    from monai.networks.nets import DenseNet121
    from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged, RandAffined, RandFlipd,
                                  ScaleIntensityd, SpatialPadd, CenterSpatialCropd, EnsureTyped)
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The CNN arm needs torch, monai and nibabel: pip install torch monai nibabel") from exc

from sklearn.model_selection import StratifiedGroupKFold


def make_transforms(train: bool, roi=(96, 112, 96)):
    t = [LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"), ScaleIntensityd(keys="image"),
         SpatialPadd(keys="image", spatial_size=roi), CenterSpatialCropd(keys="image", roi_size=roi)]
    if train:
        t += [RandFlipd(keys="image", prob=0.5, spatial_axis=0),
              RandAffined(keys="image", prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.05, 0.05, 0.05))]
    t += [EnsureTyped(keys=["image", "label"])]
    return Compose(t)


def run_fold(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray, args: argparse.Namespace, device) -> np.ndarray:
    recs = lambda idx: [{"image": r.path, "label": np.float32(r.amyloid_pos)} for r in df.iloc[idx].itertuples()]
    dl_tr = DataLoader(Dataset(recs(tr), make_transforms(True)), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers)
    dl_te = DataLoader(Dataset(recs(te), make_transforms(False)), batch_size=args.batch_size, num_workers=args.workers)
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_weight = torch.tensor([(1 - df.amyloid_pos.iloc[tr].mean()) / df.amyloid_pos.iloc[tr].mean()], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(args.epochs):
        model.train()
        for batch in dl_tr:
            x, yb = batch["image"].to(device), batch["label"].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for batch in dl_te:
            out.append(torch.sigmoid(model(batch["image"].to(device))).cpu().numpy().ravel())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute-labels", action="store_true", help="label-permutation null run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = pd.read_csv(args.table).dropna(subset=["path", "amyloid_pos"]).reset_index(drop=True)
    y = df.amyloid_pos.astype(int).to_numpy()
    if args.permute_labels:
        # permute at the subject level so that within-subject sessions keep a common (wrong) label
        rng = np.random.default_rng(args.seed)
        subj = df.subject.unique()
        perm = dict(zip(subj, rng.permutation(df.groupby("subject").amyloid_pos.first().loc[subj].to_numpy())))
        df["amyloid_pos"] = df.subject.map(perm).astype(float)
        y = df.amyloid_pos.astype(int).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(df), np.nan)
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, te) in enumerate(cv.split(df, y, df.subject)):
        print(f"fold {k}: train {len(tr)} test {len(te)}")
        oof[te] = run_fold(df, tr, te, args, device)
    df["p_cnn"] = oof
    df.to_csv(out_dir / f"oof_seed{args.seed}{'_perm' if args.permute_labels else ''}.csv", index=False)
    with open(out_dir / "args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)


if __name__ == "__main__":
    main()
