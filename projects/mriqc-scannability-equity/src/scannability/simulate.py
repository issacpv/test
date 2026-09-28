"""Synthetic multi-dataset corpus with IQMs that depend on age, group, motion and dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_corpus(n_datasets: int = 20, seed: int = 0, mean_n: int = 60) -> pd.DataFrame:
    """Simulate datasets with distinct populations/protocols and participant-level IQMs.

    Ground truth built in:
    * motion (latent) is U-shaped in age, higher in clinical groups and in males;
    * fd_mean depends on motion plus a dataset offset (protocol);
    * cnr declines with age (atrophy) and in children *independently of motion* — the
      biology-IQM coupling — and drops with motion; cjv is the inverse;
    * efc rises with motion and slightly with age.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_datasets):
        ds = f"ds{d:06d}"
        kind = rng.choice(["adult_control", "lifespan", "pediatric", "clinical_psych", "clinical_neuro"],
                          p=[0.3, 0.2, 0.15, 0.2, 0.15])
        n = max(10, int(rng.normal(mean_n, mean_n / 3)))
        ds_offset_fd = rng.normal(0, 0.05)
        ds_offset_cnr = rng.normal(0, 0.3)
        if kind == "adult_control":
            age = rng.uniform(18, 40, n)
            group = np.array(["control"] * n)
        elif kind == "lifespan":
            age = rng.uniform(8, 88, n)
            group = np.array(["control"] * n)
        elif kind == "pediatric":
            age = rng.uniform(5, 17, n)
            group = rng.choice(["control", "neurodevelopmental"], n, p=[0.5, 0.5])
        elif kind == "clinical_psych":
            age = rng.uniform(18, 60, n)
            group = rng.choice(["control", "psychiatric"], n, p=[0.45, 0.55])
        else:
            age = rng.uniform(40, 85, n)
            group = rng.choice(["control", "neurological"], n, p=[0.4, 0.6])
        sex = rng.choice(["F", "M"], n)
        motion = (0.12 + 0.0004 * (age - 30) ** 2 / 10 + 0.05 * (group != "control") + 0.02 * (sex == "M")
                  + rng.gamma(2.0, 0.03, n))
        fd_mean = np.clip(motion + ds_offset_fd + rng.normal(0, 0.03, n), 0.02, None)
        fd_perc = np.clip(100 * (fd_mean - 0.1) * 1.5 + rng.normal(0, 5, n), 0, 100)
        tsnr = np.clip(60 - 80 * (fd_mean - 0.15) + rng.normal(0, 8, n), 5, None)
        atrophy = np.clip((age - 55) / 30, 0, None) ** 1.5 + np.clip((12 - age) / 12, 0, None)
        cnr = np.clip(3.8 - 0.8 * atrophy - 3.0 * (fd_mean - 0.15) + ds_offset_cnr + rng.normal(0, 0.25, n), 0.5, None)
        cjv = np.clip(0.38 + 0.08 * atrophy + 0.5 * (fd_mean - 0.15) - 0.05 * ds_offset_cnr + rng.normal(0, 0.03, n), 0.2, None)
        efc = np.clip(0.45 + 0.4 * (fd_mean - 0.15) + 0.01 * atrophy + rng.normal(0, 0.03, n), 0.3, 0.9)
        rows.append(pd.DataFrame({
            "dataset": ds, "participant_id": [f"sub-{i:03d}" for i in range(n)], "age": age.round(1), "sex": sex,
            "group": group, "fd_mean": fd_mean, "fd_perc": fd_perc, "tsnr": tsnr, "cnr": cnr, "cjv": cjv, "efc": efc,
            "true_motion": motion, "dataset_kind": kind,
        }))
    return pd.concat(rows, ignore_index=True)
