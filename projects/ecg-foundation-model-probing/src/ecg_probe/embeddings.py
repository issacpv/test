"""Frozen-model embedding extraction for 12-lead ECGs.

A model registry hides each backbone behind a common interface::

    encoder = get_encoder("mock", n_layers=4, dim=256)
    layers = encoder.encode(x)          # x: (12, 5000) float32
    # layers: {"layer0": vec, ..., "final": vec}

The ``MockEncoder`` is deterministic (seeded random projections of simple
waveform statistics) so the probing / erasure / fairness pipeline and its tests
run without downloading any real weights. Real backbones (ECG-FM, HuBERT-ECG,
ECGFounder) implement the same ``Encoder`` protocol in ``_torch_backends`` once
weights and torch are available.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

N_LEADS = 12
N_SAMPLES = 5000  # 10 s @ 500 Hz


@runtime_checkable
class Encoder(Protocol):
    def encode(self, x: np.ndarray) -> dict[str, np.ndarray]:
        ...


def waveform_features(x: np.ndarray) -> np.ndarray:
    """Cheap per-lead summary features used as the mock encoder's input.

    Returns a flat vector of per-lead [mean, std, range, dominant-band power,
    zero-crossing rate]. These carry real (if coarse) age/sex-linked information
    (amplitudes, rate), which is what makes the mock a meaningful floor.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != N_LEADS:
        raise ValueError(f"expected ({N_LEADS}, T) array, got {x.shape}")
    feats = []
    for lead in x:
        lead = lead - lead.mean()
        fft = np.abs(np.fft.rfft(lead))
        band = fft[1:20].sum() / (fft.sum() + 1e-9)  # low-frequency power fraction
        zcr = np.mean(np.abs(np.diff(np.sign(lead)))) / 2.0
        feats.extend([lead.mean(), lead.std(), lead.ptp(), band, zcr])
    return np.asarray(feats, dtype=float)


class MockEncoder:
    """Deterministic multi-layer encoder for testing the probing pipeline.

    Each 'layer' is a fixed random linear map of the waveform features passed
    through a tanh nonlinearity; deeper layers mix more features, mimicking the
    way task-relevant/demographic information concentrates with depth.
    """

    def __init__(self, n_layers: int = 4, dim: int = 128, seed: int = 0) -> None:
        self.n_layers = n_layers
        self.dim = dim
        rng = np.random.default_rng(seed)
        in_dim = N_LEADS * 5
        self._W = [rng.normal(0, 1 / np.sqrt(in_dim), size=(in_dim, dim)) for _ in range(n_layers)]

    def encode(self, x: np.ndarray) -> dict[str, np.ndarray]:
        f = waveform_features(x)
        out: dict[str, np.ndarray] = {}
        h = f
        for i, W in enumerate(self._W):
            h = np.tanh(f @ W + 0.1 * i)
            out[f"layer{i}"] = h.copy()
        out["final"] = h.copy()
        return out


_REGISTRY = {"mock": MockEncoder}


def get_encoder(name: str = "mock", **kwargs) -> Encoder:
    """Instantiate an encoder by name. Only 'mock' ships here; real backbones
    are registered in ``_torch_backends`` when torch + weights are present."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder '{name}'; available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def extract_matrix(encoder: Encoder, X: np.ndarray, layer: str = "final") -> np.ndarray:
    """Stack embeddings for a batch of ECGs X (N, 12, 5000) at one layer -> (N, dim)."""
    return np.vstack([encoder.encode(x)[layer] for x in X])
