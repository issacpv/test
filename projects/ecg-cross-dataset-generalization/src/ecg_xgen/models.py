"""Baseline models.

* :class:`HandcraftedFeatures` + :func:`fit_logistic_baseline` - a transparent,
  dependency-light lower bound (rate/rhythm statistics, QRS-width proxy,
  lead-wise spectral band powers).  Runs anywhere numpy/scipy/sklearn run.
* :class:`ResNet1D` - a 1D ResNet-18-style network for (12, 5000) inputs.
  Imported lazily; requires PyTorch.  ``train_resnet`` is a minimal loop that
  the full experiments extend (AdamW, cosine schedule, early stopping on the
  *source* validation set only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as sps

from .loaders import TARGET_FS


@dataclass
class HandcraftedFeatures:
    """Compute a fixed-length feature vector per harmonised ECG (12, n) array."""

    fs: int = TARGET_FS
    bands: tuple[tuple[float, float], ...] = ((0.5, 4.0), (4.0, 15.0), (15.0, 40.0))

    @property
    def names(self) -> list[str]:
        n = ["hr_mean", "rr_std", "rr_cv", "rr_rmssd", "pnn50", "qrs_width_ms", "n_beats"]
        for lead in range(12):
            for lo, hi in self.bands:
                n.append(f"bp_l{lead}_{lo:g}-{hi:g}")
            n.append(f"amp_l{lead}")
        return n

    def r_peaks(self, x: np.ndarray) -> np.ndarray:
        """Detect R peaks on lead II (falls back to the highest-energy lead) via band-passed energy envelope."""
        lead = x[1] if np.abs(x[1]).sum() > 0 else x[np.argmax(np.abs(x).sum(axis=1))]
        nyq = 0.5 * self.fs
        sos = sps.butter(2, [5 / nyq, 20 / nyq], btype="band", output="sos")
        env = sps.sosfiltfilt(sos, lead.astype(np.float64)) ** 2
        env = np.convolve(env, np.ones(int(0.12 * self.fs)) / int(0.12 * self.fs), mode="same")
        thr = 0.3 * np.max(env) if np.max(env) > 0 else 1.0
        peaks, _ = sps.find_peaks(env, height=thr, distance=int(0.25 * self.fs))
        return peaks

    def qrs_width_ms(self, x: np.ndarray, peaks: np.ndarray) -> float:
        """Median width (ms) of the absolute-slope burst around each R peak on lead II."""
        if peaks.size == 0:
            return float("nan")
        lead = x[1]
        d = np.abs(np.gradient(lead))
        widths = []
        w = int(0.1 * self.fs)
        for p in peaks:
            seg = d[max(0, p - w):p + w]
            if seg.size == 0:
                continue
            above = seg > 0.2 * seg.max()
            widths.append(above.sum() / self.fs * 1000.0)
        return float(np.median(widths)) if widths else float("nan")

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        peaks = self.r_peaks(x)
        rr = np.diff(peaks) / self.fs if peaks.size > 1 else np.array([np.nan])
        hr = 60.0 / np.nanmean(rr) if np.isfinite(np.nanmean(rr)) and np.nanmean(rr) > 0 else np.nan
        drr = np.diff(rr) if rr.size > 1 else np.array([np.nan])
        feats = [
            hr,
            float(np.nanstd(rr)),
            float(np.nanstd(rr) / np.nanmean(rr)) if np.nanmean(rr) > 0 else np.nan,
            float(np.sqrt(np.nanmean(drr ** 2))),
            float(np.nanmean(np.abs(drr) > 0.05)) if drr.size else np.nan,
            self.qrs_width_ms(x, peaks),
            float(peaks.size),
        ]
        for lead in range(12):
            f, p = sps.welch(x[lead], fs=self.fs, nperseg=min(1024, x.shape[1]))
            total = np.trapezoid(p, f) + 1e-12
            for lo, hi in self.bands:
                m = (f >= lo) & (f < hi)
                feats.append(float(np.trapezoid(p[m], f[m]) / total))
            feats.append(float(np.percentile(np.abs(x[lead]), 99)))
        return np.asarray(feats, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """``X``: (n, 12, samples) -> (n, n_features); NaNs are imputed by the column median."""
        F = np.stack([self.transform_one(x) for x in X])
        med = np.nanmedian(F, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        return np.where(np.isfinite(F), F, med)


@dataclass
class OvRLogistic:
    """One-vs-rest logistic regression over multi-hot labels; classes without positives yield 0 probability."""

    scaler: Any
    models: dict[int, Any]
    n_classes: int

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(F)
        P = np.zeros((F.shape[0], self.n_classes), dtype=np.float32)
        for j, m in self.models.items():
            P[:, j] = m.predict_proba(Z)[:, 1]
        return P


def fit_logistic_baseline(F: np.ndarray, Y: np.ndarray, C: float = 1.0, seed: int = 0) -> OvRLogistic:
    """Fit one L2-logistic model per harmonised class on standardised handcrafted features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Y = np.asarray(Y)
    sc = StandardScaler().fit(F)
    Z = sc.transform(F)
    models: dict[int, Any] = {}
    for j in range(Y.shape[1]):
        if 0 < Y[:, j].sum() < len(Y):  # need both classes present
            models[j] = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Z, Y[:, j])
    return OvRLogistic(sc, models, Y.shape[1])


def predict_proba_full(model: OvRLogistic, F: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Probability matrix (n, n_classes); kept for API symmetry with the deep models."""
    P = model.predict_proba(F)
    if n_classes is not None and n_classes != P.shape[1]:
        raise ValueError("n_classes does not match the fitted model")
    return P


# ----------------------------------------------------------------------------
# Optional PyTorch 1D-ResNet
# ----------------------------------------------------------------------------
def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for ResNet1D: pip install torch") from exc
    return torch, nn


def build_resnet1d(n_classes: int, in_channels: int = 12, widths: tuple[int, ...] = (64, 128, 256, 512), blocks_per_stage: int = 2):
    """Return an ``nn.Module`` 1D ResNet-18-style classifier (logits output)."""
    torch, nn = _torch()

    class BasicBlock(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(cout)
            self.conv2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False)
            self.bn2 = nn.BatchNorm1d(cout)
            self.short = nn.Sequential()
            if stride != 1 or cin != cout:
                self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu(out + self.short(x))

    class ResNet1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, widths[0], 15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(widths[0]), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1),
            )
            layers = []
            cin = widths[0]
            for i, w in enumerate(widths):
                for b in range(blocks_per_stage):
                    layers.append(BasicBlock(cin, w, stride=2 if (b == 0 and i > 0) else 1))
                    cin = w
            self.layers = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(cin, n_classes)

        def embed(self, x):
            return self.pool(self.layers(self.stem(x))).flatten(1)

        def forward(self, x):
            return self.head(self.embed(x))

    return ResNet1D()


def train_resnet(model, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
                 epochs: int = 30, lr: float = 1e-3, batch_size: int = 64, device: str | None = None, patience: int = 5):
    """Minimal BCE training loop with early stopping on source-validation macro AUROC. Returns the best model."""
    torch, nn = _torch()
    from sklearn.metrics import roc_auc_score

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    best, best_state, bad = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), batch_size):
            idx = perm[s:s + batch_size]
            xb, yb = Xt[idx].to(device), Yt[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(torch.cat([model(Xv[s:s + 256].to(device)).cpu() for s in range(0, len(Xv), 256)])).numpy()
        aucs = [roc_auc_score(Y_val[:, j], pv[:, j]) for j in range(Y_val.shape[1]) if 0 < Y_val[:, j].sum() < len(Y_val)]
        score = float(np.mean(aucs)) if aucs else -np.inf
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
