"""Synthetic-data tests for nm_scaling (no network, no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nm_scaling import swc as swcmod  # noqa: E402
from nm_scaling.api_client import flatten_record, neurons_to_dataframe, NeuroMorphoClient  # noqa: E402
from nm_scaling.mixed_models import (  # noqa: E402
    fit_scaling_mixed_model,
    prepare_allometric_frame,
    species_effect_attenuation,
    wald_test_exponent,
    variance_partition,
)
from nm_scaling.phylo_gls import newick_to_covariance, pgls, phylogenetic_signal, divergence_times_to_covariance  # noqa: E402


# ---------------------------------------------------------------------- helpers
def random_binary_tree_swc(n_bif: int = 30, seed: int = 0, planar: bool = False) -> str:
    """Grow a random binary dendritic tree and return SWC text."""
    rng = np.random.default_rng(seed)
    rows = [(1, 1, 0.0, 0.0, 0.0, 5.0, -1)]  # soma
    xyz = {1: np.zeros(3)}
    tips = []
    # one stem
    nid = 2
    pos = np.array([10.0, 0.0, 0.0])
    rows.append((nid, 3, *pos, 1.0, 1))
    xyz[nid] = pos
    tips.append(nid)
    nid += 1
    for _ in range(n_bif):
        parent = tips.pop(rng.integers(len(tips)))
        for _ in range(2):
            step = rng.normal(size=3) * 15.0
            if planar:
                step[2] = 0.0
            pos = xyz[parent] + step
            rows.append((nid, 3, *pos, 0.8, parent))
            xyz[nid] = pos
            tips.append(nid)
            nid += 1
    return "# synthetic\n" + "\n".join(" ".join(f"{v}" for v in r) for r in rows)


# ---------------------------------------------------------------------- SWC
def test_parse_and_morphometrics_consistency():
    tree = swcmod.parse_swc_text(random_binary_tree_swc(n_bif=25, seed=1))
    assert len(tree) == 2 + 2 * 25
    m = swcmod.morphometrics(tree)
    seg = swcmod.segment_lengths(tree)
    assert m["total_length"] == pytest.approx(seg.sum(), rel=1e-9)
    assert m["n_branch_points"] == 25
    assert m["n_tips"] == 26  # binary tree: tips = bifurcations + 1
    assert m["n_stems"] == 1
    assert 1.0 <= m["effective_dimension"] <= 3.0
    assert m["hull_volume"] > 0


def test_planar_tree_has_low_dimension_and_zero_hull_volume():
    tree = swcmod.parse_swc_text(random_binary_tree_swc(n_bif=25, seed=2, planar=True))
    m = swcmod.morphometrics(tree)
    assert m["effective_dimension"] < 2.2
    assert m["planarity"] == pytest.approx(0.0, abs=1e-9)
    assert m["hull_volume"] == 0.0
    assert m["hull_area_2d"] > 0


def test_sholl_profile_simple_line():
    # straight dendrite along x: nodes at 10, 20, 30 um; one crossing at any radius < 30
    text = "1 1 0 0 0 5 -1\n2 3 10 0 0 1 1\n3 3 20 0 0 1 2\n4 3 30 0 0 1 3\n"
    tree = swcmod.parse_swc_text(text)
    prof = swcmod.sholl_profile(tree, radii=[5, 15, 25, 35])
    assert prof.tolist() == [1, 1, 1, 0]
    orders = swcmod.branch_orders(tree)
    assert orders.max() == 0


def test_branch_orders_binary():
    text = ("1 1 0 0 0 5 -1\n2 3 10 0 0 1 1\n3 3 20 5 0 1 2\n4 3 20 -5 0 1 2\n"
            "5 3 30 8 0 1 3\n6 3 30 2 0 1 3\n")
    tree = swcmod.parse_swc_text(text)
    orders = swcmod.branch_orders(tree)
    assert orders.tolist() == [0, 0, 1, 1, 2, 2]


def test_wiring_law_prediction_exponents():
    n = np.array([10.0, 80.0])
    v = np.array([1.0, 1.0])
    pred = swcmod.wiring_law_prediction(n, v, dim=3)
    assert pred[1] / pred[0] == pytest.approx(8 ** (2 / 3))


# ---------------------------------------------------------------------- API helpers
def test_flatten_record_and_dataframe():
    rec = {"neuron_name": "n1", "archive": "LabA", "species": "mouse", "brain_region": ["neocortex", "layer 5"],
           "cell_type": ["principal cell", "pyramidal"], "slicing_thickness": "300 um", "shrinkage_corrected": "Corrected",
           "_page": 0}
    flat = flatten_record(rec)
    assert flat["brain_region_0"] == "neocortex" and flat["n_brain_region"] == 2 and "_page" not in flat
    df = neurons_to_dataframe([rec])
    assert df.loc[0, "slicing_thickness"] == 300
    assert bool(df.loc[0, "shrinkage_corrected_flag"]) is True
    url = NeuroMorphoClient.swc_url("LabA", "n1")
    assert url.endswith("/laba/CNG%20version/n1.CNG.swc")


# ---------------------------------------------------------------------- mixed models
def _synthetic_allometry(n_labs=12, cells_per_lab=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    species_of_lab = {}
    for lab in range(n_labs):
        # labs 0-3 have two species (identifying subset), the rest one species
        sp_list = ["mouse", "rat"] if lab < 4 else [["mouse", "rat", "human"][lab % 3]]
        species_of_lab[lab] = sp_list
        lab_eff = rng.normal(0, 0.25)
        for _ in range(cells_per_lab):
            sp = sp_list[rng.integers(len(sp_list))]
            n_bp = rng.integers(10, 200)
            vol = rng.lognormal(12, 0.5)
            sp_eff = {"mouse": 0.0, "rat": 0.1, "human": 0.3}[sp]
            log_L = 1.0 + (2 / 3) * np.log(n_bp) + (1 / 3) * np.log(vol) + sp_eff + lab_eff + rng.normal(0, 0.1)
            rows.append({"archive": f"lab{lab}", "species": sp, "n_branch_points": n_bp, "hull_volume": vol,
                         "total_length": np.exp(log_L)})
    return pd.DataFrame(rows)


def test_mixed_model_recovers_exponents():
    df = prepare_allometric_frame(_synthetic_allometry())
    fit = fit_scaling_mixed_model(df, fixed=("log_n", "log_V"), categorical=("species",), group="archive")
    assert abs(fit.params["log_n"] - 2 / 3) < 0.03
    assert abs(fit.params["log_V"] - 1 / 3) < 0.03
    assert fit.lab_variance > 0.01
    res = wald_test_exponent(fit, "log_n", 2 / 3)
    assert res["p"] > 0.01


def test_species_attenuation_and_variance_partition():
    df = prepare_allometric_frame(_synthetic_allometry(seed=3))
    out = species_effect_attenuation(df, "log_L", covariates=("log_n", "log_V"))
    assert out["n_labs_multispecies"] == 4
    assert 0 <= out["lab_variance"]
    assert np.isfinite(out["attenuation"])
    vp = variance_partition(df, "log_L", factors=("species", "archive"), numeric_covariates=("log_n", "log_V"))
    assert set(vp["factor"]) == {"species", "archive"}
    assert (vp["semi_partial_r2"] >= -1e-9).all()


# ---------------------------------------------------------------------- phylogenetics
def test_newick_covariance():
    names, C = newick_to_covariance("((A:1,B:1):1,C:2);")
    i = {n: k for k, n in enumerate(names)}
    assert C[i["A"], i["A"]] == pytest.approx(2.0)
    assert C[i["A"], i["B"]] == pytest.approx(1.0)
    assert C[i["A"], i["C"]] == pytest.approx(0.0)
    C2 = divergence_times_to_covariance(["A", "B", "C"], {("A", "B"): 1.0, ("A", "C"): 2.0, ("B", "C"): 2.0})
    assert np.allclose(C2, C[np.ix_([i["A"], i["B"], i["C"]], [i["A"], i["B"], i["C"]])])


def test_pgls_recovers_slope_under_brownian_motion():
    rng = np.random.default_rng(0)
    # balanced 16-tip tree
    def build(depth, prefix):
        if depth == 0:
            return prefix
        return f"({build(depth - 1, prefix + 'a')}:1,{build(depth - 1, prefix + 'b')}:1)"
    names, C = newick_to_covariance(build(4, "t") + ";")
    L = np.linalg.cholesky(C)
    x = L @ rng.normal(size=len(names))
    y = 0.5 + 1.5 * x + L @ rng.normal(size=len(names)) * 0.3
    fit = pgls(y, x, C, names=["x"])
    assert abs(fit.params[1] - 1.5) < 0.3
    assert fit.lam > 0.5
    sig = phylogenetic_signal(y, C)
    assert 0 <= sig["lambda"] <= 1 and sig["p"] < 0.5
