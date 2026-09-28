"""Single-channel and spatiotemporal features of extracellular spike waveforms."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats


def peak_channel(wf: np.ndarray) -> int:
    """Index of the site with the largest negative deflection (trough)."""
    return int(np.argmin(wf.min(axis=1)))


def single_channel_features(v: np.ndarray, t_ms: np.ndarray) -> Dict[str, float]:
    """Trough-to-peak, half-width, peak/trough ratio, repolarisation and recovery slopes.

    Follows the conventions of Jia et al. (2019) / AllenSDK ecephys quality metrics:
    trough = global minimum; peak = maximum after the trough; half-width at half trough
    amplitude; repolarisation slope = linear slope from the trough to 30% recovery;
    recovery slope = slope after the peak.
    """
    v = np.asarray(v, float)
    dt = t_ms[1] - t_ms[0]
    i_tr = int(np.argmin(v))
    after = v[i_tr:]
    i_pk = i_tr + int(np.argmax(after))
    trough, peak = v[i_tr], v[i_pk]
    feats = {"trough_uV": float(trough), "peak_uV": float(peak),
             "trough_to_peak_ms": float((i_pk - i_tr) * dt),
             "peak_trough_ratio": float(abs(peak) / max(abs(trough), 1e-9))}
    half = trough / 2.0
    below = np.nonzero(v <= half)[0]
    feats["half_width_ms"] = float((below.max() - below.min() + 1) * dt) if len(below) else float("nan")
    # repolarisation slope: from trough to 30 % recovery point
    target = trough * 0.7
    rec = np.nonzero(after >= target)[0]
    if len(rec) > 1:
        j = rec[0]
        feats["repolarization_slope_uV_per_ms"] = float((v[i_tr + j] - trough) / max(j * dt, dt))
    else:
        feats["repolarization_slope_uV_per_ms"] = float("nan")
    tail = v[i_pk:]
    if len(tail) > 3:
        n = min(len(tail), int(round(0.5 / dt)) + 1)
        feats["recovery_slope_uV_per_ms"] = float(stats.linregress(np.arange(n) * dt, tail[:n])[0])
    else:
        feats["recovery_slope_uV_per_ms"] = float("nan")
    return feats


def spatial_features(wf: np.ndarray, site_xyz: np.ndarray, t_ms: np.ndarray, footprint_frac: float = 0.12,
                     max_sites_velocity: int = 12) -> Dict[str, float]:
    """Footprint, spread, decay exponent, above/below asymmetry and propagation velocity.

    - footprint_um: vertical extent (max - min depth) of sites whose trough amplitude exceeds
      ``footprint_frac`` of the peak-site amplitude (Ye et al., 2025 use 12 %).
    - n_sites_footprint: number of such sites.
    - spread_um: amplitude-weighted RMS distance of sites from the peak site.
    - decay_exponent: slope of log|amp| vs log(distance) over sites 10-150 um from the peak.
    - asymmetry: (sum |amp| above - sum |amp| below) / total, "above" = smaller depth (toward pia).
    - velocity_above/below_um_per_ms: slope of distance vs trough latency for sites along the shank.
    """
    wf = np.asarray(wf, float)
    xyz = np.asarray(site_xyz, float)
    amp = -wf.min(axis=1)  # trough amplitude, positive
    ipk = int(np.argmax(amp))
    a_max = amp[ipk]
    d = np.linalg.norm(xyz - xyz[ipk], axis=1)
    dy = xyz[:, 1] - xyz[ipk, 1]
    m = amp >= footprint_frac * a_max
    feats: Dict[str, float] = {
        "peak_site": float(ipk),
        "amp_max_uV": float(a_max),
        "n_sites_footprint": float(m.sum()),
        "footprint_um": float(xyz[m, 1].max() - xyz[m, 1].min()) if m.any() else 0.0,
        "spread_um": float(np.sqrt(np.sum(amp * d ** 2) / max(amp.sum(), 1e-9))),
    }
    sel = (d >= 10) & (d <= 150) & (amp > 0)
    if sel.sum() >= 3:
        feats["decay_exponent"] = float(stats.linregress(np.log(d[sel]), np.log(amp[sel]))[0])
    else:
        feats["decay_exponent"] = float("nan")
    above, below = amp[dy < 0].sum(), amp[dy > 0].sum()
    feats["asymmetry_above_below"] = float((above - below) / max(above + below, 1e-9))
    dt = t_ms[1] - t_ms[0]
    lat = np.argmin(wf, axis=1) * dt
    for name, mask in (("above", dy < 0), ("below", dy > 0)):
        idx = np.nonzero(mask & m)[0]
        idx = idx[np.argsort(np.abs(dy[idx]))][:max_sites_velocity]
        if len(idx) >= 3 and np.std(lat[idx] - lat[ipk]) > 0:
            slope = stats.linregress(lat[idx] - lat[ipk], np.abs(dy[idx]))[0]
            feats[f"velocity_{name}_um_per_ms"] = float(slope)
        else:
            feats[f"velocity_{name}_um_per_ms"] = float("nan")
    return feats


def waveform_feature_vector(wf: np.ndarray, site_xyz: np.ndarray, t_ms: np.ndarray) -> Dict[str, float]:
    """All features on the peak channel plus spatial features."""
    ipk = peak_channel(wf)
    out = single_channel_features(wf[ipk], t_ms)
    out.update(spatial_features(wf, site_xyz, t_ms))
    return out


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1-D energy distance between two samples (for simulated-vs-observed feature comparisons)."""
    return float(stats.energy_distance(np.asarray(a, float), np.asarray(b, float)))


def permutation_energy_test(a: np.ndarray, b: np.ndarray, n_perm: int = 1000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = energy_distance(a, b)
    pooled = np.concatenate([a, b])
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pooled)
        null[i] = energy_distance(p[: len(a)], p[len(a):])
    return {"energy_distance": obs, "p": float((np.sum(null >= obs) + 1) / (n_perm + 1))}
