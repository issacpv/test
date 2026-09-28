"""Baselines: Moens-Korteweg PTT model, feature ridge, small 1D-CNN and a physics-DL hybrid.

Moens-Korteweg + exponential pressure-stiffness law (Hughes et al., 1979):

    PWV^2 = E h / (rho d),   E = E0 exp(alpha P)   ==>   P = (1/alpha) ln(rho d L^2 / (E0 h)) - (2/alpha) ln(PTT)

so a population model is linear in ln(PTT):  P = a + b ln(PTT) with b = -2/alpha < 0
(Mukkamala et al., 2015 IEEE TBME). The inverse-square variant P = a + b / PTT^2 follows from a
linear stiffness law. Both are implemented; both admit a one-point per-subject calibration in
which only the intercept ``a`` (vessel geometry / E0) is re-estimated while ``b`` (the
pressure-stiffness exponent) stays population-level.

The CNN / hybrid need PyTorch; they are defined only if it imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------------------
# Moens-Korteweg PTT model
# ----------------------------------------------------------------------------------------------


@dataclass
class MoensKortewegPTT:
    """P = a + b * g(PTT), g = ln (exponential stiffness law) or 1/PTT^2 (linear law)."""

    form: Literal["log", "inverse_square"] = "log"
    a: float = field(default=np.nan)
    b: float = field(default=np.nan)
    ptt_range_s: tuple[float, float] = (0.08, 0.8)

    def _g(self, ptt: np.ndarray) -> np.ndarray:
        ptt = np.clip(np.asarray(ptt, float), *self.ptt_range_s)
        return np.log(ptt) if self.form == "log" else 1.0 / ptt ** 2

    def fit(self, ptt: np.ndarray, bp: np.ndarray, subject_ids: Sequence | None = None) -> "MoensKortewegPTT":
        """Least-squares fit. With ``subject_ids``, fit the slope on subject-centred data (a
        within-subject slope, the physiologically meaningful one) and the intercept on the means,
        which avoids the slope being dominated by between-subject differences in geometry."""
        g, y = self._g(ptt), np.asarray(bp, float)
        m = np.isfinite(g) & np.isfinite(y)
        g, y = g[m], y[m]
        if subject_ids is not None:
            s = np.asarray(subject_ids)[m]
            df = pd.DataFrame({"s": s, "g": g, "y": y})
            gc = df["g"] - df.groupby("s")["g"].transform("mean")
            yc = df["y"] - df.groupby("s")["y"].transform("mean")
            denom = float(np.sum(gc ** 2))
            self.b = float(np.sum(gc * yc) / denom) if denom > 0 else 0.0
            self.a = float(np.mean(y) - self.b * np.mean(g))
        else:
            X = np.column_stack([np.ones_like(g), g])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.a, self.b = float(coef[0]), float(coef[1])
        return self

    def predict(self, ptt: np.ndarray, a: float | np.ndarray | None = None) -> np.ndarray:
        a = self.a if a is None else a
        return a + self.b * self._g(ptt)

    def calibrate_intercepts(self, ptt_cal: np.ndarray, bp_cal: np.ndarray, subject_cal: Sequence) -> dict:
        """One-point (or few-point) per-subject calibration of the intercept only."""
        g = self._g(ptt_cal)
        df = pd.DataFrame({"s": np.asarray(subject_cal), "r": np.asarray(bp_cal, float) - self.b * g})
        return df.groupby("s")["r"].mean().to_dict()

    def predict_calibrated(self, ptt: np.ndarray, subject_ids: Sequence, intercepts: dict) -> np.ndarray:
        a = np.array([intercepts.get(s, self.a) for s in np.asarray(subject_ids)], float)
        return self.predict(ptt, a)

    @property
    def alpha(self) -> float:
        """Pressure-stiffness exponent (1/mmHg) implied by the log form: b = -2/alpha."""
        return float(-2.0 / self.b) if self.form == "log" and self.b != 0 else np.nan


# ----------------------------------------------------------------------------------------------
# feature ridge
# ----------------------------------------------------------------------------------------------

DEFAULT_FEATURES = ("pat_foot", "pat_peak", "pat_slope", "hr", "ppg_amp", "ppg_rise_t", "ppg_width50", "pat_foot_sd", "hr_sd")


def make_feature_ridge(alpha: float = 1.0) -> Pipeline:
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit_feature_ridge(df: pd.DataFrame, target: str = "sbp", features: Sequence[str] = DEFAULT_FEATURES,
                      alpha: float = 1.0) -> tuple[Pipeline, list[str]]:
    """Fit ridge on the hand-crafted features (rows with NaN in any feature are dropped)."""
    cols = [c for c in features if c in df]
    d = df.dropna(subset=cols + [target])
    model = make_feature_ridge(alpha).fit(d[cols].to_numpy(float), d[target].to_numpy(float))
    return model, cols


# ----------------------------------------------------------------------------------------------
# CNN and hybrid (PyTorch optional)
# ----------------------------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when torch is installed
    import torch
    from torch import nn

    class _ResBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, k: int = 7, stride: int = 2):
            super().__init__()
            self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2)
            self.bn1 = nn.BatchNorm1d(c_out)
            self.conv2 = nn.Conv1d(c_out, c_out, k, padding=k // 2)
            self.bn2 = nn.BatchNorm1d(c_out)
            self.skip = nn.Conv1d(c_in, c_out, 1, stride=stride)
            self.act = nn.ReLU()

        def forward(self, x):
            h = self.act(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return self.act(h + self.skip(x))

    class SmallCNN1D(nn.Module):
        """Raw ECG+PPG (2 x 1250 samples at 125 Hz) -> (SBP, DBP). ~150k parameters."""

        def __init__(self, in_channels: int = 2, widths: tuple[int, ...] = (16, 32, 64), n_out: int = 2):
            super().__init__()
            layers, c = [], in_channels
            for w in widths:
                layers.append(_ResBlock(c, w))
                c = w
            self.body = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, 64), nn.ReLU(), nn.Linear(64, n_out))

        def forward(self, x):
            return self.head(self.body(x))

    class HybridPTTCNN(nn.Module):
        """Physics term a + b ln(PAT) (trainable a, b initialised from a MoensKortewegPTT fit)
        plus a CNN residual on the raw waveforms. The residual is L2-penalised in the training
        loop so that the model prefers the physiological explanation when it suffices."""

        def __init__(self, a0: float, b0: float, cnn: nn.Module | None = None):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([a0, a0 * 0.65], dtype=torch.float32))  # SBP, DBP intercepts
            self.b = nn.Parameter(torch.tensor([b0, b0 * 0.6], dtype=torch.float32))
            self.cnn = cnn or SmallCNN1D()

        def forward(self, x, log_pat):
            phys = self.a[None, :] + self.b[None, :] * log_pat[:, None]
            resid = self.cnn(x)
            return phys + resid, resid

    def train_cnn(model: nn.Module, X: np.ndarray, Y: np.ndarray, log_pat: np.ndarray | None = None,
                  X_val: np.ndarray | None = None, Y_val: np.ndarray | None = None, log_pat_val: np.ndarray | None = None,
                  epochs: int = 20, lr: float = 1e-3, batch_size: int = 256, resid_l2: float = 1e-3,
                  device: str | None = None, seed: int = 0) -> dict:
        """Minimal training loop (Adam, MSE, early stopping on validation MAE). Targets should be
        standardised by the caller with *training-subject* statistics only."""
        torch.manual_seed(seed)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        Lt = torch.tensor(log_pat, dtype=torch.float32) if log_pat is not None else None
        n = len(Xt)
        best, best_state, hist = np.inf, None, []
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
                if isinstance(model, HybridPTTCNN):
                    out, resid = model(xb, Lt[idx].to(dev))
                    loss = nn.functional.mse_loss(out, yb) + resid_l2 * resid.pow(2).mean()
                else:
                    loss = nn.functional.mse_loss(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            row = {"epoch": ep, "train_loss": tot / n}
            if X_val is not None:
                pv = predict_cnn(model, X_val, log_pat_val, device=dev)
                row["val_mae"] = float(np.mean(np.abs(pv - Y_val)))
                if row["val_mae"] < best:
                    best, best_state = row["val_mae"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist.append(row)
        if best_state is not None:
            model.load_state_dict(best_state)
        return {"history": hist, "best_val_mae": best}

    def predict_cnn(model: nn.Module, X: np.ndarray, log_pat: np.ndarray | None = None, device: str | None = None,
                    batch_size: int = 1024) -> np.ndarray:
        dev = device or next(model.parameters()).device
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(dev)
                if isinstance(model, HybridPTTCNN):
                    lb = torch.tensor(log_pat[i:i + batch_size], dtype=torch.float32).to(dev)
                    outs.append(model(xb, lb)[0].cpu().numpy())
                else:
                    outs.append(model(xb).cpu().numpy())
        return np.vstack(outs)

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


def windows_to_tensor(ecg: np.ndarray, ppg: np.ndarray) -> np.ndarray:
    """Stack per-window standardised ECG and PPG into (n, 2, L) for the CNN."""
    def z(x):
        x = np.asarray(x, float)
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)

    return np.stack([z(ecg), z(ppg)], axis=1).astype(np.float32)
