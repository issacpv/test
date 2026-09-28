"""Video I/O, clip sampling and synthetic beating-LV videos.

Real data: EchoNet AVI files (112x112, ~50 fps) are read with OpenCV (fallback:
imageio).  CAMUS ``.mhd`` sequences can be read with SimpleITK (optional).
Synthetic data: :func:`synthetic_lv_video` renders a prolate-ellipsoid LV whose
cavity is a 2-D ellipse with time-varying semi-axes, returning frames, masks
and the analytic EF - used by the tests and by ``scripts/download_data.py --sample``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Per-channel statistics computed over the EchoNet-Dynamic training split are
# used by the reference implementation; recompute them with `compute_mean_std`.
DEFAULT_MEAN = 32.0
DEFAULT_STD = 50.0


def load_video(path: Path | str, grayscale: bool = True) -> np.ndarray:
    """Read an AVI/MP4 into a uint8 array of shape (T, H, W) (grey) or (T, H, W, 3)."""
    path = str(path)
    frames: list[np.ndarray] = []
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise OSError(f"cannot open {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
    except ImportError:  # pragma: no cover
        import imageio.v3 as iio

        arr = iio.imread(path, plugin="pyav")
        frames = [f.mean(axis=-1).astype(np.uint8) if grayscale and f.ndim == 3 else f for f in arr]
    if not frames:
        raise OSError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.uint8)


def video_fps(path: Path | str) -> float:
    """Frames per second reported by the container (EchoNet FileList.csv also stores FPS)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def resize_frames(frames: np.ndarray, size: int = 112) -> np.ndarray:
    """Nearest/area resize of (T, H, W[, C]) frames to (T, size, size[, C]) without external deps if square."""
    T, H, W = frames.shape[:3]
    if H == size and W == size:
        return frames
    try:
        import cv2

        out = [cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames]
        return np.stack(out)
    except ImportError:  # pragma: no cover
        ys = (np.arange(size) * H / size).astype(int)
        xs = (np.arange(size) * W / size).astype(int)
        return frames[:, ys][:, :, xs]


def sample_clip(frames: np.ndarray, length: int = 32, period: int = 2, start: int | None = None,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Take ``length`` frames spaced ``period`` apart (EchoNet recipe), zero-padding short videos."""
    T = frames.shape[0]
    need = (length - 1) * period + 1
    if T < need:
        pad = np.zeros((need - T,) + frames.shape[1:], dtype=frames.dtype)
        frames = np.concatenate([frames, pad])
        T = frames.shape[0]
    if start is None:
        rng = rng or np.random.default_rng(0)
        start = int(rng.integers(0, T - need + 1))
    idx = start + period * np.arange(length)
    return frames[idx]


def sample_clips(frames: np.ndarray, n_clips: int = 5, length: int = 32, period: int = 2) -> np.ndarray:
    """Evenly spaced test-time clips -> (n_clips, length, H, W[, C]); average predictions over clips."""
    T = frames.shape[0]
    need = (length - 1) * period + 1
    starts = np.linspace(0, max(T - need, 0), n_clips).astype(int)
    return np.stack([sample_clip(frames, length, period, int(s)) for s in starts])


def to_tensor(clip: np.ndarray, mean: float = DEFAULT_MEAN, std: float = DEFAULT_STD) -> np.ndarray:
    """(T, H, W) or (T, H, W, C) uint8 clip -> float32 (C, T, H, W) normalised, 3 channels (grey replicated)."""
    x = clip.astype(np.float32)
    if x.ndim == 3:
        x = np.repeat(x[..., None], 3, axis=-1)
    x = (x - mean) / std
    return np.transpose(x, (3, 0, 1, 2)).astype(np.float32)


def compute_mean_std(paths: Sequence[Path | str], max_videos: int = 200) -> tuple[float, float]:
    """Global intensity mean/std over a sample of training videos (call once per source, never on targets)."""
    s, s2, n = 0.0, 0.0, 0
    for p in list(paths)[:max_videos]:
        v = load_video(p).astype(np.float64)
        s += v.sum()
        s2 += (v ** 2).sum()
        n += v.size
    mean = s / n
    return float(mean), float(np.sqrt(s2 / n - mean ** 2))


# ----------------------------------------------------------------------------
# Synthetic beating LV
# ----------------------------------------------------------------------------
@dataclass
class SyntheticEcho:
    frames: np.ndarray        # (T, H, W) uint8
    masks: np.ndarray         # (T, H, W) bool  LV cavity
    ef_true: float            # analytic EF from ellipsoid volumes
    edv_true: float           # 4/3 pi a b^2 at end-diastole (pixel^3)
    esv_true: float
    ed_frame: int
    es_frame: int
    fps: float


def ellipse_mask(size: int, cx: float, cy: float, a: float, b: float, theta: float) -> np.ndarray:
    """Boolean mask of an ellipse with semi-axes ``a`` (along ``theta``) and ``b``."""
    yy, xx = np.mgrid[0:size, 0:size]
    x, y = xx - cx, yy - cy
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def synthetic_lv_video(n_frames: int = 64, size: int = 112, ef: float = 0.55, a_ed: float = 38.0, b_ed: float = 20.0,
                       theta: float = 0.35, fps: float = 50.0, heart_rate: float = 70.0, rr_jitter: float = 0.0,
                       noise: float = 12.0, seed: int = 0) -> SyntheticEcho:
    """Render a beating prolate-ellipsoid LV (2-D ellipse cavity) with analytic EF.

    The ES semi-axes are scaled so that the ellipsoid volume ratio equals ``1 - ef``
    (long axis shortens by 10 %, short axis absorbs the rest).  ``rr_jitter``
    (fraction of the RR interval) makes the rhythm irregular, which is the
    proxy used for arrhythmia in the hard-case analyses.
    """
    rng = np.random.default_rng(seed)
    cx, cy = size / 2 + 4, size / 2 + 6
    a_es = 0.9 * a_ed
    b_es = b_ed * np.sqrt((1 - ef) * a_ed / a_es)
    frames = np.zeros((n_frames, size, size), dtype=np.uint8)
    masks = np.zeros((n_frames, size, size), dtype=bool)
    rr = fps * 60.0 / heart_rate
    # phase accumulates with per-beat jitter
    phase = np.zeros(n_frames)
    t, beat_len = 0.0, rr * (1 + rng.normal(0, rr_jitter)) if rr_jitter else rr
    for i in range(n_frames):
        if t >= beat_len:
            t -= beat_len
            beat_len = rr * (1 + rng.normal(0, rr_jitter)) if rr_jitter else rr
        phase[i] = t / beat_len
        t += 1.0
    sector = ellipse_mask(size, size / 2, size * 0.1, size * 0.95, size * 0.95, 0.0)
    for i in range(n_frames):
        s = 0.5 - 0.5 * np.cos(2 * np.pi * phase[i])  # 0 at ED, 1 at ES
        a = a_ed + (a_es - a_ed) * s
        b = b_ed + (b_es - b_ed) * s
        cav = ellipse_mask(size, cx, cy, a, b, theta)
        wall = ellipse_mask(size, cx, cy, a + 7, b + 7, theta) & ~cav
        img = np.full((size, size), 25.0)
        img[wall] = 170.0
        img[cav] = 40.0
        img += rng.normal(0, noise, (size, size))
        img[~sector] = 0.0
        frames[i] = np.clip(img, 0, 255).astype(np.uint8)
        masks[i] = cav & sector
    edv = 4 / 3 * np.pi * a_ed * b_ed ** 2
    esv = 4 / 3 * np.pi * a_es * b_es ** 2
    ed_frame = int(np.argmin(np.abs(phase - 0.0)))
    es_frame = int(np.argmin(np.abs(phase - 0.5)))
    return SyntheticEcho(frames, masks, float(1 - esv / edv), float(edv), float(esv), ed_frame, es_frame, fps)


def video_descriptors(frames: np.ndarray) -> dict[str, float]:
    """Cheap image-level descriptors used for shift cards and density-ratio weights.

    Keys: mean/std intensity inside the imaging sector, fraction of sector pixels,
    fraction of near-black pixels, temporal intensity variation, frame count, edge density.
    """
    f = frames.astype(np.float32)
    sector = f.max(axis=0) > 5
    inside = f[:, sector] if sector.any() else f.reshape(f.shape[0], -1)
    gy, gx = np.gradient(f.mean(axis=0))
    edge = np.sqrt(gx ** 2 + gy ** 2)
    return {
        "mean_intensity": float(inside.mean()),
        "std_intensity": float(inside.std()),
        "sector_fraction": float(sector.mean()),
        "dark_fraction": float((inside < 20).mean()),
        "temporal_var": float(f.std(axis=0)[sector].mean()) if sector.any() else float(f.std(axis=0).mean()),
        "n_frames": float(frames.shape[0]),
        "edge_density": float((edge > 20).mean()),
        "height": float(frames.shape[1]),
        "width": float(frames.shape[2]),
    }


def descriptor_matrix(videos: Sequence[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Stack :func:`video_descriptors` over videos -> (n, d) float array and key names."""
    rows = [video_descriptors(v) for v in videos]
    keys = list(rows[0].keys())
    return np.array([[r[k] for k in keys] for r in rows], dtype=np.float64), keys
