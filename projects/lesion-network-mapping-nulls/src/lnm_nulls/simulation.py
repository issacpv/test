"""Ground-truth simulation engine for comparing LNM null families.

A toy brain (ellipsoid on a small grid) is divided into block parcels and three arterial
territories per hemisphere. A distance-decaying parcel connectome is generated; each voxel's
connectivity is its parcel's row plus voxel noise. Lesions are drawn from a pool of blobs
whose placement follows the territory prior (most lesions in "MCA"). Symptoms are generated
from a known network G plus an optional lesion-volume term; every null family is run and
type-I error / power / precision / recall are returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lesion_sampling as ls
from . import lnm, nulls


@dataclass
class ToyBrain:
    shape: tuple[int, int, int]
    brain_mask: np.ndarray
    parcels: np.ndarray            # int labels 1..P (0 outside)
    territory: np.ndarray          # int labels 1..6 (0 outside)
    fc_vp: np.ndarray              # (n_voxels_total, P) voxel-to-parcel connectivity (flat grid order)
    parcel_coords: np.ndarray      # (P, 3)
    n_parcels: int
    territory_prior: np.ndarray = field(default=None)


def make_toy_brain(shape=(24, 28, 20), block=(6, 7, 5), fc_length: float = 6.0, voxel_noise: float = 0.1,
                   rng: np.random.Generator | None = None) -> ToyBrain:
    rng = np.random.default_rng(0) if rng is None else rng
    g = np.indices(shape).astype(float)
    c = (np.array(shape) - 1) / 2.0
    ell = sum(((g[i] - c[i]) / (0.5 * shape[i])) ** 2 for i in range(3))
    brain = ell <= 1.0
    # block parcels
    pb = (g[0] // block[0]) * 1000 + (g[1] // block[1]) * 100 + (g[2] // block[2])
    labels = np.zeros(shape, int)
    uniq = np.unique(pb[brain])
    for k, u in enumerate(uniq, start=1):
        labels[(pb == u) & brain] = k
    P = len(uniq)
    coords = np.array([np.argwhere(labels == k).mean(0) for k in range(1, P + 1)])
    # territories: 3 anterior-posterior slabs per hemisphere (1-3 left, 4-6 right)
    terr = np.zeros(shape, int)
    slab = np.minimum((g[1] / shape[1] * 3).astype(int), 2)
    terr[brain] = (slab + 1 + 3 * (g[0] >= shape[0] / 2).astype(int))[brain]
    # parcel connectome with spatial decay + hub structure, then voxel rows
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    fc_pp = np.exp(-d / fc_length) + 0.05 * rng.standard_normal((P, P))
    fc_pp = (fc_pp + fc_pp.T) / 2
    np.fill_diagonal(fc_pp, 1.0)
    fc_vp = np.zeros((np.prod(shape), P))
    flat_labels = labels.ravel()
    for k in range(1, P + 1):
        rows = np.flatnonzero(flat_labels == k)
        fc_vp[rows] = fc_pp[k - 1] + voxel_noise * rng.standard_normal((rows.size, P))
    prior = np.array([0.15, 0.55, 0.30, 0.15, 0.55, 0.30])  # MCA-like middle slab dominates
    return ToyBrain(shape, brain, labels, terr, fc_vp, coords, P, prior / prior.sum())


def sample_pool(brain: ToyBrain, n: int, rng: np.random.Generator, radius_range=(2.0, 4.5),
                hemisphere: int | None = None) -> list[np.ndarray]:
    """Blob lesions placed according to the territory prior (optionally one hemisphere)."""
    out = []
    terrs = np.arange(1, 7)
    prior = brain.territory_prior.copy()
    if hemisphere == 0:
        prior[3:] = 0
    elif hemisphere == 1:
        prior[:3] = 0
    prior /= prior.sum()
    for _ in range(n):
        t = rng.choice(terrs, p=prior)
        coords = np.argwhere(brain.territory == t)
        c = coords[rng.integers(coords.shape[0])]
        r = rng.uniform(*radius_range)
        s = ls.sphere_lesion(brain.shape, tuple(c + rng.normal(0, 0.5, 3)), r) & brain.brain_mask
        # make it irregular: drop a random half-space slab
        if rng.random() < 0.5:
            axis = rng.integers(3)
            cut = np.indices(brain.shape)[axis] > c[axis] + rng.integers(-1, 2)
            s &= ~cut | (rng.random(brain.shape) < 0.5)
        if s.sum() < 5:
            s = ls.sphere_lesion(brain.shape, tuple(c), 2.0) & brain.brain_mask
        out.append(s)
    return out


def generate_symptoms(maps: np.ndarray, volumes: np.ndarray, network: np.ndarray, beta: float, gamma: float,
                      noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    """score = beta * mean map value in ``network`` parcels + gamma * log(volume) + noise (all standardised)."""
    x = maps[:, network].mean(axis=1)
    x = (x - x.mean()) / (x.std() + 1e-12)
    v = np.log(volumes.astype(float))
    v = (v - v.mean()) / (v.std() + 1e-12)
    return beta * x + gamma * v + noise_sd * rng.standard_normal(maps.shape[0])


@dataclass
class StudyOutcome:
    family: str
    any_significant: bool
    n_significant: int
    precision: float
    recall: float


def evaluate_families(brain: ToyBrain, lesions: list[np.ndarray], scores: np.ndarray, network: np.ndarray,
                      pool: list[np.ndarray], n_perm: int, rng: np.random.Generator,
                      families=("N1_label", "N2_sphere", "N3_shuffle", "N4_matched"), adjust_volume: bool = False,
                      alpha: float = 0.05) -> list[StudyOutcome]:
    """Run the requested null families on one simulated study and score them against ``network``."""
    maps = lnm.lesion_network_maps(lesions, brain.fc_vp)
    vols = np.array([m.sum() for m in lesions])
    cov = np.log(vols)[:, None] if adjust_volume else None
    feats_t = [ls.lesion_features(m, brain.territory, 6) for m in lesions]
    feats_p = [ls.lesion_features(m, brain.territory, 6) for m in pool]
    pool_maps = lnm.lesion_network_maps(pool, brain.fc_vp)
    truth = np.zeros(brain.n_parcels, bool)
    truth[network] = True
    out = []
    for fam in families:
        if fam == "N1_label":
            res = nulls.label_permutation_null(maps, scores, cov, n_perm, rng)
        elif fam == "N2_sphere":
            res = nulls.synthetic_lesion_null(lesions, scores, cov, brain.fc_vp, brain.brain_mask, n_perm, rng, "sphere")
        elif fam == "N3_shuffle":
            res = nulls.synthetic_lesion_null(lesions, scores, cov, brain.fc_vp, brain.brain_mask, n_perm, rng, "shuffle")
        elif fam == "N4_matched":
            res = nulls.matched_resampling_null(lesions, scores, cov, brain.fc_vp, pool, feats_p, feats_t, n_perm, rng,
                                                pool_maps=pool_maps)
        else:
            raise ValueError(fam)
        sig = res.significant(alpha)
        tp = np.sum(sig & truth)
        prec = tp / sig.sum() if sig.sum() else float("nan")
        rec = tp / truth.sum()
        out.append(StudyOutcome(fam, bool(sig.any()), int(sig.sum()), float(prec), float(rec)))
    return out


def run_simulation(n_studies: int, n_patients: int, beta: float, gamma: float, n_perm: int = 99,
                   pool_size: int = 150, noise_sd: float = 1.0, adjust_volume: bool = False,
                   seed: int = 0, families=("N1_label", "N2_sphere", "N3_shuffle", "N4_matched")) -> dict[str, dict[str, float]]:
    """Repeat simulated studies; return per-family family-wise positive rate (= FPR when beta=0,
    power otherwise) plus mean precision/recall."""
    rng = np.random.default_rng(seed)
    brain = make_toy_brain(rng=rng)
    pool = sample_pool(brain, pool_size, rng)
    # ground-truth network: a contiguous cluster of parcels
    center = rng.integers(brain.n_parcels)
    d = np.linalg.norm(brain.parcel_coords - brain.parcel_coords[center], axis=1)
    network = np.argsort(d)[:4]
    tallies: dict[str, list[StudyOutcome]] = {f: [] for f in families}
    for _ in range(n_studies):
        idx = rng.choice(len(pool), n_patients, replace=False)
        lesions = [pool[i] for i in idx]
        maps = lnm.lesion_network_maps(lesions, brain.fc_vp)
        vols = np.array([m.sum() for m in lesions])
        scores = generate_symptoms(maps, vols, network, beta, gamma, noise_sd, rng)
        for o in evaluate_families(brain, lesions, scores, network, pool, n_perm, rng, families, adjust_volume):
            tallies[o.family].append(o)
    summary = {}
    for f, outs in tallies.items():
        precisions = [o.precision for o in outs if np.isfinite(o.precision)]
        summary[f] = {
            "positive_rate": float(np.mean([o.any_significant for o in outs])),
            "mean_precision": float(np.mean(precisions)) if precisions else float("nan"),
            "mean_recall": float(np.mean([o.recall for o in outs])),
        }
    return summary
