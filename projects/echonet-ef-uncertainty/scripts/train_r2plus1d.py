#!/usr/bin/env python
"""Optional: R(2+1)D-18 EF regression in the EchoNet-Dynamic style (requires torch + torchvision).

Trains on FileList.csv/Videos of one source, writes per-video predictions for
TRAIN/VAL/TEST (and any extra target FileLists given with --target) to CSV so
that the conformal harness (echo_uq.conformal) can be run without a GPU.

A 5-clip test-time average and the across-clip standard deviation are stored as
``pred`` and ``sigma`` (heuristic scale for locally adaptive conformal scores).

Example
-------
python scripts/train_r2plus1d.py --root data/echonet-dynamic --epochs 20 --out outputs/echonet_r2plus1d
python scripts/train_r2plus1d.py --root data/echonet-dynamic --weights outputs/echonet_r2plus1d/best.pt \
    --target data/echonet-pediatric/A4C --target data/camus_as_filelist --predict-only
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echo_uq.video import load_video, sample_clip, sample_clips, to_tensor  # noqa: E402


def read_filelist(root: Path) -> list[dict]:
    with open(root / "FileList.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        fn = r["FileName"]
        r["path"] = str(root / "Videos" / (fn if fn.lower().endswith((".avi", ".mp4", ".npz")) else fn + ".avi"))
    return rows


def load_frames(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        return np.load(path)["frames"]
    return load_video(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--target", type=Path, action="append", default=[], help="extra FileList roots to predict on")
    ap.add_argument("--out", type=Path, default=Path("outputs/r2plus1d"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--period", type=int, default=2)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    import torch
    import torchvision

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.video.r2plus1d_18(weights="KINETICS400_V1" if not args.weights else None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    rows = read_filelist(args.root)
    train = [r for r in rows if r["Split"].upper() == "TRAIN"]
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.predict_only:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.1)
        rng = np.random.default_rng(0)
        best = float("inf")
        for ep in range(args.epochs):
            model.train()
            rng.shuffle(train)
            losses = []
            for s in range(0, len(train), args.batch):
                batch = train[s:s + args.batch]
                x = torch.tensor(np.stack([to_tensor(sample_clip(load_frames(r["path"]), args.length, args.period, rng=rng)) for r in batch])).to(device)
                y = torch.tensor([float(r["EF"]) for r in batch], dtype=torch.float32, device=device)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(x).squeeze(1), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()
            val_mae = evaluate(model, [r for r in rows if r["Split"].upper() == "VAL"], args, device)
            print(f"epoch {ep + 1}: train mse {np.mean(losses):.2f}  val mae {val_mae:.2f}")
            if val_mae < best:
                best = val_mae
                torch.save(model.state_dict(), args.out / "best.pt")
        model.load_state_dict(torch.load(args.out / "best.pt", map_location=device))

    predict(model, rows, args.out / f"pred_{args.root.name}.csv", args, device)
    for t in args.target:
        predict(model, read_filelist(t), args.out / f"pred_{t.name}.csv", args, device)


def evaluate(model, rows: list[dict], args, device: str) -> float:
    import torch

    model.eval()
    err = []
    with torch.no_grad():
        for r in rows:
            clips = torch.tensor(np.stack([to_tensor(c) for c in sample_clips(load_frames(r["path"]), 3, args.length, args.period)])).to(device)
            err.append(abs(model(clips).mean().item() - float(r["EF"])))
    return float(np.mean(err)) if err else float("nan")


def predict(model, rows: list[dict], out_csv: Path, args, device: str) -> None:
    import torch

    model.eval()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "Split", "EF", "pred", "sigma"])
        with torch.no_grad():
            for r in rows:
                clips = torch.tensor(np.stack([to_tensor(c) for c in sample_clips(load_frames(r["path"]), 5, args.length, args.period)])).to(device)
                p = model(clips).squeeze(1).cpu().numpy()
                w.writerow([r["FileName"], r.get("Split", ""), r.get("EF", ""), float(p.mean()), float(p.std() + 1e-3)])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
