"""Tabular synthetic-data generators with a common ``fit`` / ``sample`` interface.

* ``GaussianCopulaGenerator``     empirical marginals + Gaussian copula (numpy only)
* ``IndependentMarginalsGenerator`` samples each column independently (negative control)
* ``BootstrapJitterGenerator``    resample rows and add small noise (privacy-leaky positive control)
* ``SDVAdapter`` / ``SynthcityAdapter``  wrappers around CTGAN/TVAE and DDPM (import-guarded)

All generators take a DataFrame of numeric columns; binary columns (values in {0, 1}) are detected
automatically and sampled as binaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "BaseGenerator",
    "GaussianCopulaGenerator",
    "IndependentMarginalsGenerator",
    "BootstrapJitterGenerator",
    "SDVAdapter",
    "SynthcityAdapter",
    "detect_binary_columns",
]


def detect_binary_columns(df: pd.DataFrame) -> List[str]:
    """Columns whose non-null values are all in {0, 1}."""
    out = []
    for c in df.columns:
        v = df[c].dropna().unique()
        if len(v) <= 2 and set(np.asarray(v, dtype=float)).issubset({0.0, 1.0}):
            out.append(c)
    return out


class BaseGenerator:
    """Interface: ``fit(df)`` then ``sample(n, rng)`` -> DataFrame with the same columns."""

    columns: List[str]

    def fit(self, df: pd.DataFrame) -> "BaseGenerator":  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class IndependentMarginalsGenerator(BaseGenerator):
    """Negative control: preserves every marginal, destroys all dependence."""

    def fit(self, df: pd.DataFrame) -> "IndependentMarginalsGenerator":
        self.columns = list(df.columns)
        self._values = {c: df[c].to_numpy(dtype=float) for c in self.columns}
        return self

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        rng = rng or np.random.default_rng(0)
        return pd.DataFrame({c: rng.choice(self._values[c], size=n, replace=True) for c in self.columns})


class BootstrapJitterGenerator(BaseGenerator):
    """Positive control for utility (and a privacy worst case): resampled real rows plus noise."""

    def __init__(self, noise_frac: float = 0.05) -> None:
        self.noise_frac = noise_frac

    def fit(self, df: pd.DataFrame) -> "BootstrapJitterGenerator":
        self.columns = list(df.columns)
        self._data = df.to_numpy(dtype=float)
        self._binary = np.array([c in detect_binary_columns(df) for c in self.columns])
        self._sd = np.nanstd(self._data, axis=0)
        return self

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        rng = rng or np.random.default_rng(0)
        idx = rng.integers(0, self._data.shape[0], size=n)
        out = self._data[idx].copy()
        noise = rng.standard_normal(out.shape) * self._sd * self.noise_frac
        noise[:, self._binary] = 0.0
        return pd.DataFrame(out + noise, columns=self.columns)


class GaussianCopulaGenerator(BaseGenerator):
    """Gaussian copula with empirical marginals.

    Fit: map each column to normal scores via (average-tied) ranks, estimate the score correlation
    matrix (shrunk toward identity by ``shrinkage`` for stability). Sample: draw multivariate normal
    scores, invert through the empirical quantile function (numeric) or threshold at the empirical
    prevalence (binary). This is the classical baseline (e.g. SDV's GaussianCopula).
    """

    def __init__(self, shrinkage: float = 0.01) -> None:
        self.shrinkage = shrinkage

    def fit(self, df: pd.DataFrame) -> "GaussianCopulaGenerator":
        self.columns = list(df.columns)
        data = df.to_numpy(dtype=float)
        n, p = data.shape
        self._binary = np.array([c in detect_binary_columns(df) for c in self.columns])
        self._sorted = [np.sort(data[:, j]) for j in range(p)]
        self._prev = np.nanmean(data, axis=0)
        scores = np.empty_like(data)
        for j in range(p):
            r = stats.rankdata(data[:, j], method="average")
            scores[:, j] = stats.norm.ppf(r / (n + 1))
        corr = np.corrcoef(scores, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        self.corr_ = (1 - self.shrinkage) * corr + self.shrinkage * np.eye(p)
        return self

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        rng = rng or np.random.default_rng(0)
        p = len(self.columns)
        z = rng.multivariate_normal(np.zeros(p), self.corr_, size=n, method="cholesky")
        u = stats.norm.cdf(z)
        out = np.empty((n, p))
        for j in range(p):
            if self._binary[j]:
                out[:, j] = (u[:, j] > 1 - self._prev[j]).astype(float)
            else:
                out[:, j] = np.quantile(self._sorted[j], u[:, j], method="linear")
        return pd.DataFrame(out, columns=self.columns)


class SDVAdapter(BaseGenerator):
    """Wrapper for SDV single-table synthesizers (``ctgan`` | ``tvae`` | ``copula``)."""

    def __init__(self, model: str = "ctgan", epochs: int = 300, **kwargs: object) -> None:
        self.model = model
        self.epochs = epochs
        self.kwargs = kwargs

    def fit(self, df: pd.DataFrame) -> "SDVAdapter":
        try:
            from sdv.metadata import SingleTableMetadata  # type: ignore
            from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pip install sdv to use SDVAdapter") from exc
        self.columns = list(df.columns)
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)
        cls = {"ctgan": CTGANSynthesizer, "tvae": TVAESynthesizer, "copula": GaussianCopulaSynthesizer}[self.model]
        if self.model in ("ctgan", "tvae"):
            self._synth = cls(meta, epochs=self.epochs, **self.kwargs)
        else:
            self._synth = cls(meta, **self.kwargs)
        self._synth.fit(df)
        return self

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        return self._synth.sample(num_rows=n)[self.columns]


class SynthcityAdapter(BaseGenerator):
    """Wrapper for synthcity plugins (default ``ddpm``)."""

    def __init__(self, plugin: str = "ddpm", **kwargs: object) -> None:
        self.plugin = plugin
        self.kwargs = kwargs

    def fit(self, df: pd.DataFrame) -> "SynthcityAdapter":
        try:
            from synthcity.plugins import Plugins  # type: ignore
            from synthcity.plugins.core.dataloader import GenericDataLoader  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pip install synthcity to use SynthcityAdapter") from exc
        self.columns = list(df.columns)
        self._model = Plugins().get(self.plugin, **self.kwargs)
        self._model.fit(GenericDataLoader(df))
        return self

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        return self._model.generate(count=n).dataframe()[self.columns]


def make_generator(name: str, **kwargs: object) -> BaseGenerator:
    """Factory by short name: copula | independent | bootstrap | ctgan | tvae | ddpm."""
    table: Dict[str, object] = {
        "copula": lambda: GaussianCopulaGenerator(**kwargs),
        "independent": lambda: IndependentMarginalsGenerator(),
        "bootstrap": lambda: BootstrapJitterGenerator(**kwargs),
        "ctgan": lambda: SDVAdapter("ctgan", **kwargs),
        "tvae": lambda: SDVAdapter("tvae", **kwargs),
        "ddpm": lambda: SynthcityAdapter("ddpm", **kwargs),
    }
    if name not in table:
        raise ValueError(f"unknown generator {name!r}")
    return table[name]()  # type: ignore[operator]
