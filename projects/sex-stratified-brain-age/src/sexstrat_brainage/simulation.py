"""Lifespan cohort simulator with known ground truth.

The generative model has a latent *biological age* ``bio = age + true_delta``
where ``true_delta`` is the quantity a brain-age model should recover and an
outcome (e.g., cognition) depends on it. Sex can differ in (i) head size only,
(ii) ageing slope, (iii) both, or (iv) nothing. Regional volumes scale with
head size (allometric exponent < 1), so a model that ignores head size will
attribute the sex difference in volumes to age.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIONS_THK = ["superiorfrontal", "precentral", "postcentral", "superiorparietal", "inferiorparietal",
               "precuneus", "lateraloccipital", "superiortemporal", "middletemporal", "inferiortemporal",
               "fusiform", "entorhinal", "parahippocampal", "insula", "rostralmiddlefrontal", "caudalmiddlefrontal"]
REGIONS_VOL = ["Left-Hippocampus", "Right-Hippocampus", "Left-Thalamus", "Right-Thalamus", "Left-Putamen",
               "Right-Putamen", "Left-Caudate", "Right-Caudate", "Left-Amygdala", "Right-Amygdala",
               "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "CortexVol", "CerebralWhiteMatterVol"]


@dataclass
class SimConfig:
    head_size_ratio: float = 1.10       # male/female TIV ratio
    slope_ratio: float = 1.00           # male/female ageing slope ratio (1 = no difference)
    delta_sd: float = 4.0               # SD of true biological-age deviation (years)
    outcome_beta: float = -0.4          # outcome ~ beta * true_delta (standardized units)
    noise_thk: float = 0.08
    noise_vol_frac: float = 0.05
    allometry: float = 0.8              # regional volume ∝ TIV**allometry
    female_frac: float = 0.5
    age_min: float = 18.0
    age_max: float = 90.0


def simulate_lifespan_cohort(n: int = 800, seed: int = 0, cfg: SimConfig | None = None) -> pd.DataFrame:
    """Return a FreeSurfer-style table with columns thk_*, vol_*, eTIV, age, sex, true_delta, outcome."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    sex = (rng.uniform(size=n) > cfg.female_frac).astype(int)  # 1 = male
    age = rng.uniform(cfg.age_min, cfg.age_max, n)
    true_delta = rng.normal(0, cfg.delta_sd, n)
    bio = age + true_delta
    tiv = rng.normal(1.40e6, 0.10e6, n) * np.where(sex == 1, cfg.head_size_ratio, 1.0)
    slope_mult = np.where(sex == 1, cfg.slope_ratio, 1.0)

    df = pd.DataFrame({"age": age, "sex": sex, "true_delta": true_delta, "eTIV": tiv})
    for i, r in enumerate(REGIONS_THK):
        base = 2.4 + 0.4 * ((i % 5) / 4)
        slope = (0.006 + 0.004 * ((i * 7) % 5) / 4) * slope_mult
        for hemi in ("lh", "rh"):
            df[f"thk_{hemi}_{r}"] = base - slope * (bio - 50) + rng.normal(0, cfg.noise_thk, n)
    for i, r in enumerate(REGIONS_VOL):
        base = {"CortexVol": 480e3, "CerebralWhiteMatterVol": 450e3}.get(r, 4000 + 800 * (i % 6))
        ventricle = "Ventricle" in r
        slope = (0.004 + 0.003 * ((i * 3) % 4) / 3) * slope_mult
        scale = (tiv / 1.40e6) ** cfg.allometry
        if ventricle:
            vol = base * scale * np.exp(0.035 * slope_mult * (bio - 50) / 1.0)
        else:
            vol = base * scale * (1 - slope * (bio - 50))
        df[f"vol_{r}"] = vol * (1 + rng.normal(0, cfg.noise_vol_frac, n))
    df["mean_thickness"] = df[[c for c in df.columns if c.startswith("thk_")]].mean(axis=1)
    df["outcome"] = cfg.outcome_beta * (true_delta / cfg.delta_sd) - 0.02 * (age - 50) + rng.normal(0, 0.9, n)
    df["group"] = [f"sub-{i:05d}" for i in range(n)]
    df["cohort"] = "sim"
    return df


def scenario(name: str) -> SimConfig:
    """Named ground-truth scenarios used in the simulation study."""
    if name == "none":
        return SimConfig(head_size_ratio=1.0, slope_ratio=1.0)
    if name == "head_size_only":
        return SimConfig(head_size_ratio=1.10, slope_ratio=1.0)
    if name == "slope_only":
        return SimConfig(head_size_ratio=1.0, slope_ratio=1.15)
    if name == "both":
        return SimConfig(head_size_ratio=1.10, slope_ratio=1.15)
    raise ValueError(name)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("thk_", "vol_")) or c in ("eTIV", "mean_thickness")]


def recovery_error(res: pd.DataFrame, df: pd.DataFrame) -> dict[str, float]:
    """How well does the modelled delta recover the true delta (overall and by sex)?"""
    d = res["delta"].to_numpy()
    t = df.loc[res.index, "true_delta"].to_numpy()
    s = res["sex"].to_numpy()
    out = {"corr_all": float(np.corrcoef(d, t)[0, 1])}
    for k, name in ((0, "female"), (1, "male")):
        out[f"corr_{name}"] = float(np.corrcoef(d[s == k], t[s == k])[0, 1])
    return out
