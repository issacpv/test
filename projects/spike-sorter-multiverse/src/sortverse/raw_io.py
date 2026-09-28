"""Raw AP-band I/O for the Allen Visual Coding Neuropixels ``spike_band.dat`` files and generic int16 binaries.

Layout (verified on the public bucket): ``visual-coding-neuropixels/raw-data/<session>/<probe>/spike_band.dat``
is a flat int16 array of shape (n_samples, 384), channel-interleaved, at ~30 kHz. Because the file is flat,
any time window maps to one contiguous byte range, which is what HTTP Range requests need.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

RAW_BUCKET = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com"
RAW_PREFIX = "visual-coding-neuropixels/raw-data"
FS_AP = 30000.0
N_CHANNELS = 384
ITEMSIZE = 2  # int16
NP1_GAIN_UV_PER_BIT = 1.2 / 1024 / 500 * 1e6  # 1.2 V ADC range, 10 bit, gain 500 -> 2.34375 uV/bit


def raw_url(session_id: str, probe_id: str, filename: str = "spike_band.dat") -> str:
    return f"{RAW_BUCKET}/{RAW_PREFIX}/{session_id}/{probe_id}/{filename}"


def byte_range_for_window(t_start: float, t_end: float, fs: float = FS_AP, n_channels: int = N_CHANNELS,
                          itemsize: int = ITEMSIZE) -> Tuple[int, int]:
    """Inclusive byte range [b0, b1] covering samples in [t_start, t_end) seconds of a flat int16 file."""
    if t_end <= t_start:
        raise ValueError("t_end must be > t_start")
    s0 = int(np.floor(t_start * fs))
    s1 = int(np.ceil(t_end * fs))
    frame = n_channels * itemsize
    return s0 * frame, s1 * frame - 1


def n_samples_in_file(size_bytes: int, n_channels: int = N_CHANNELS, itemsize: int = ITEMSIZE) -> int:
    return size_bytes // (n_channels * itemsize)


def read_segment(path: Path, t_start: float, t_end: float, fs: float = FS_AP, n_channels: int = N_CHANNELS,
                 dtype=np.int16, file_offset_s: float = 0.0) -> np.ndarray:
    """Memory-mapped read of [t_start, t_end) seconds from a flat binary. ``file_offset_s`` is the time of the
    file's first sample (use it for windowed extractions written by scripts/download_data.py).

    Returns an (n_samples, n_channels) array (a copy, dtype preserved)."""
    path = Path(path)
    n_total = n_samples_in_file(path.stat().st_size, n_channels, np.dtype(dtype).itemsize)
    mm = np.memmap(path, dtype=dtype, mode="r", shape=(n_total, n_channels))
    s0 = int(round((t_start - file_offset_s) * fs))
    s1 = int(round((t_end - file_offset_s) * fs))
    if s0 < 0 or s1 > n_total or s1 <= s0:
        raise ValueError(f"window [{t_start}, {t_end}) s outside file ({n_total / fs:.2f} s from {file_offset_s} s)")
    return np.array(mm[s0:s1])


def to_microvolts(x: np.ndarray, gain_uv_per_bit: float = NP1_GAIN_UV_PER_BIT) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) * np.float32(gain_uv_per_bit)


def neuropixels1_geometry(n_channels: int = N_CHANNELS) -> Tuple[np.ndarray, np.ndarray]:
    """Neuropixels 1.0 staggered checkerboard site positions (um): x cycles 43, 11, 59, 27; y = 20 * (i // 2)."""
    i = np.arange(n_channels)
    x = np.array([43.0, 11.0, 59.0, 27.0])[i % 4]
    y = 20.0 * (i // 2)
    return x, y


def write_binary_segment(x: np.ndarray, path: Path) -> Path:
    """Write an (n_samples, n_channels) int16 array as a flat binary usable by SpikeInterface/Kilosort."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(x, dtype=np.int16).tofile(path)
    return path


def to_spikeinterface(path: Path, fs: float = FS_AP, n_channels: int = N_CHANNELS,
                      gain_uv_per_bit: float = NP1_GAIN_UV_PER_BIT):
    """Wrap a flat int16 binary as a SpikeInterface recording with Neuropixels 1.0 geometry (optional dep)."""
    import probeinterface as pi  # type: ignore
    import spikeinterface as si  # type: ignore

    rec = si.read_binary(str(path), sampling_frequency=fs, num_channels=n_channels, dtype="int16",
                         gain_to_uV=gain_uv_per_bit, offset_to_uV=0.0)
    x, y = neuropixels1_geometry(n_channels)
    probe = pi.Probe(ndim=2)
    probe.set_contacts(positions=np.column_stack([x, y]), shapes="square", shape_params={"width": 12})
    probe.set_device_channel_indices(np.arange(n_channels))
    return rec.set_probe(probe)


def synthetic_recording(duration_s: float, fs: float, n_channels: int, rng: np.random.Generator,
                        noise_uv: float = 10.0) -> np.ndarray:
    """Gaussian-noise int16 recording for I/O tests (no spikes)."""
    x = rng.normal(0.0, noise_uv / NP1_GAIN_UV_PER_BIT, size=(int(duration_s * fs), n_channels))
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)
