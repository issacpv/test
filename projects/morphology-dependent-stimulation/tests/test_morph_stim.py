"""Synthetic-data tests for the morph_stim core.

These exercise the physics the whole project rests on, against cases with known
analytic answers:

* a straight sealed-end cable in a uniform field polarizes antisymmetrically,
  with a peak proportional to the electrotonic length constant;
* polarization is exactly linear in field amplitude (this is what licenses the
  three-solve orientation sweep);
* a field perpendicular to a straight cable produces no polarization;
* threshold scales inversely with peak polarization, and the waveform low-pass
  correction orders the modalities as tDCS < tACS < TMS < DBS in efficiency.

No network access and no NEURON required.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morph_stim.popstats import (  # noqa: E402
    bootstrap_ratio_ci,
    class_separability,
    prepare_frame,
    variance_components,
)
from morph_stim.swc_cable import CableModel, PassiveParams, SWCNeuron, load_swc  # noqa: E402
from morph_stim.thresholds import (  # noqa: E402
    MODALITIES,
    Waveform,
    fibonacci_directions,
    orientation_sweep,
    sweep_population,
    threshold_for_directions,
)


# --------------------------------------------------------------------- fixtures
def straight_cable(n: int = 201, spacing_um: float = 5.0, radius_um: float = 0.5,
                   axis: np.ndarray | None = None, name: str = "synth_cable") -> SWCNeuron:
    """A straight unbranched cable with the soma at one end."""
    axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    xyz = np.outer(np.arange(n) * spacing_um, axis)
    return SWCNeuron(
        sample_id=np.arange(1, n + 1),
        xyz_um=xyz,
        radius_um=np.full(n, radius_um),
        type_id=np.r_[1, np.full(n - 1, 2)],  # soma then axon
        parent=np.r_[-1, np.arange(n - 1)],
        name=name,
        metadata={"species": "synthetic", "cell_type": "pyramidal", "archive": "synth"},
    )


def branched_cell(name: str = "synth_branched", n_branch: int = 6,
                  branch_len: int = 40, seed: int = 0) -> SWCNeuron:
    """A soma with an apical trunk plus randomly oriented terminal branches."""
    rng = np.random.default_rng(seed)
    xyz = [np.zeros(3)]
    parent = [-1]
    type_id = [1]
    trunk_len = 60
    for i in range(trunk_len):
        xyz.append(np.array([0.0, 0.0, (i + 1) * 10.0]))
        parent.append(len(xyz) - 2)
        type_id.append(4)  # apical dendrite
    trunk_top = len(xyz) - 1
    for _ in range(n_branch):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        prev = trunk_top
        base = xyz[trunk_top]
        for j in range(branch_len):
            xyz.append(base + direction * (j + 1) * 8.0)
            parent.append(prev)
            type_id.append(2)  # axon-typed terminals so the default site works
            prev = len(xyz) - 1
    xyz_arr = np.array(xyz)
    return SWCNeuron(
        sample_id=np.arange(1, len(xyz) + 1),
        xyz_um=xyz_arr,
        radius_um=np.full(len(xyz), 0.6),
        type_id=np.array(type_id),
        parent=np.array(parent),
        name=name,
        metadata={"species": "synthetic", "cell_type": "pyramidal", "archive": "synth"},
    )


# ------------------------------------------------------------------ cable model
def test_uniform_field_polarizes_antisymmetrically() -> None:
    model = CableModel(straight_cable())
    v = model.steady_state_polarization(np.array([0.0, 0.0, 1.0]))
    assert v[0] * v[-1] < 0, "the two sealed ends must polarize with opposite sign"
    # Antisymmetry is approximate, not exact: the root node is the soma and is
    # given a sphere's membrane area rather than a segment's, so the two ends are
    # not electrically identical. 2% is the size of that deliberate asymmetry.
    assert v[0] == pytest.approx(-v[-1], rel=0.02)
    # the middle of a near-symmetric cable is essentially unpolarized
    assert abs(v[len(v) // 2]) < 0.01 * abs(v[0])


def test_polarization_is_linear_in_field() -> None:
    model = CableModel(straight_cable())
    e = np.array([0.0, 0.0, 1.0])
    v1 = model.steady_state_polarization(e)
    v3 = model.steady_state_polarization(3.0 * e)
    assert np.allclose(v3, 3.0 * v1, rtol=1e-8, atol=1e-12)


def test_perpendicular_field_does_not_polarize_a_straight_cable() -> None:
    model = CableModel(straight_cable())
    v = model.steady_state_polarization(np.array([1.0, 0.0, 0.0]))
    assert np.abs(v).max() < 1e-9


def test_polarization_length_matches_analytic_length_constant() -> None:
    """Peak polarization of a long cable approaches lambda * E.

    For a semi-infinite cable, lambda = sqrt(d*Rm/(4*Ra)). Our cable is long
    relative to lambda, so max|v| / |E| should land within ~15% of lambda.
    """
    params = PassiveParams(Ra=150.0, Rm=30_000.0)
    radius_um, spacing_um, n = 0.5, 10.0, 601
    model = CableModel(straight_cable(n=n, spacing_um=spacing_um, radius_um=radius_um), params)
    d_cm = 2.0 * radius_um / 1e4
    lambda_um = math.sqrt(d_cm * params.Rm / (4.0 * params.Ra)) * 1e4
    got = model.polarization_length_um(np.array([0.0, 0.0, 1.0]))
    assert got == pytest.approx(lambda_um, rel=0.15), f"got {got:.1f} um, expected ~{lambda_um:.1f}"


def test_activating_function_is_concentrated_at_terminals() -> None:
    model = CableModel(straight_cable())
    af = np.abs(model.activating_function(np.array([0.0, 0.0, 1.0])))
    interior = af[2:-2]
    assert af[0] > 100 * (interior.max() + 1e-30)
    assert af[-1] > 100 * (interior.max() + 1e-30)


def test_sensitivity_matrix_reproduces_direct_solves() -> None:
    model = CableModel(branched_cell())
    d = np.array([0.3, -0.5, 0.81])
    d /= np.linalg.norm(d)
    direct = model.steady_state_polarization(d)
    via_basis = model.polarization_per_unit_field(d[None, :])[0]
    assert np.allclose(direct, via_basis, rtol=1e-8, atol=1e-10)


def test_summary_descriptors_are_sane() -> None:
    nrn = straight_cable(n=101, spacing_um=5.0)
    s = CableModel(nrn).summary()
    assert s["n_nodes"] == 101
    assert s["total_length_um"] == pytest.approx(100 * 5.0, rel=1e-6)
    assert s["n_terminals"] == 1
    assert s["has_axon"] is True


def test_load_swc_roundtrip(tmp_path: Path) -> None:
    content = """# a comment
# another
1 1 0 0 0 1.0 -1
2 3 0 0 10 0.5 1
3 3 0 0 20 0.5 2

7 3 0 0 30 0.5 3
"""
    path = tmp_path / "toy.CNG.swc"
    path.write_text(content, encoding="utf-8")
    nrn = load_swc(path, species="human")
    assert nrn.n_nodes == 4
    assert nrn.name == "toy"
    assert nrn.metadata["species"] == "human"
    # non-contiguous id 7 must still resolve its parent
    assert nrn.parent.tolist() == [-1, 0, 1, 2]
    assert nrn.total_length_um() == pytest.approx(30.0)


def test_load_swc_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.swc"
    path.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no parsable SWC rows"):
        load_swc(path)


# -------------------------------------------------------------------- waveforms
def test_waveform_attenuation_ordering() -> None:
    tau = PassiveParams().tau_m_ms
    dc = MODALITIES["tDCS"].attenuation(tau)
    ac10 = MODALITIES["tACS_10Hz"].attenuation(tau)
    ac1k = MODALITIES["tACS_1kHz"].attenuation(tau)
    tms = MODALITIES["TMS"].attenuation(tau)
    dbs = MODALITIES["DBS"].attenuation(tau)
    assert dc == 1.0
    assert dc > ac10 > ac1k > 0
    assert dc > tms > dbs > 0


def test_waveform_validation() -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        Waveform("bad", "sinusoid").attenuation(30.0)
    with pytest.raises(ValueError, match="pulse_width_ms"):
        Waveform("bad", "pulse").attenuation(30.0)
    with pytest.raises(ValueError, match="tau_m_ms must be positive"):
        MODALITIES["tDCS"].attenuation(0.0)


# ------------------------------------------------------------------- thresholds
def test_threshold_inversely_proportional_to_delta_v() -> None:
    model = CableModel(branched_cell())
    d = fibonacci_directions(16)
    t1 = threshold_for_directions(model, d, delta_v_th=10.0)
    t2 = threshold_for_directions(model, d, delta_v_th=20.0)
    assert np.allclose(t2, 2.0 * t1, rtol=1e-8)


def test_threshold_is_higher_for_briefer_pulses() -> None:
    model = CableModel(branched_cell())
    d = fibonacci_directions(16)
    t_dc = threshold_for_directions(model, d, waveform="tDCS")
    t_tms = threshold_for_directions(model, d, waveform="TMS")
    t_dbs = threshold_for_directions(model, d, waveform="DBS")
    assert np.all(t_tms > t_dc)
    assert np.all(t_dbs > t_tms)


def test_bidirectional_threshold_never_exceeds_unidirectional() -> None:
    model = CableModel(branched_cell())
    d = fibonacci_directions(32)
    uni = threshold_for_directions(model, d, bidirectional=False)
    bi = threshold_for_directions(model, d, bidirectional=True)
    assert np.all(bi <= uni + 1e-12)


def test_fibonacci_directions_are_unit_vectors() -> None:
    d = fibonacci_directions(64)
    assert d.shape == (64, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-10)
    hemi = fibonacci_directions(64, hemisphere=True)
    assert np.all(hemi[:, 2] >= -1e-12)
    with pytest.raises(ValueError):
        fibonacci_directions(0)


def test_orientation_sweep_keys_and_anisotropy() -> None:
    model = CableModel(straight_cable())
    res = orientation_sweep(model, waveform="tDCS", n_directions=64)
    for key in (
        "threshold_min_Vpm",
        "threshold_max_Vpm",
        "best_direction",
        "anisotropy_ratio",
        "threshold_along_column_Vpm",
        "column_alignment",
        "polarization_length_um",
    ):
        assert key in res
    assert res["threshold_min_Vpm"] <= res["threshold_median_Vpm"] <= res["threshold_max_Vpm"]
    # A straight cable is maximally anisotropic: perpendicular fields never fire it.
    assert res["anisotropy_ratio"] > 10.0
    # ...and its easiest direction is its own axis.
    assert res["column_alignment"] > 0.9


def test_sweep_population_is_tidy() -> None:
    models = [CableModel(branched_cell(name=f"cell{i}", seed=i)) for i in range(4)]
    rows = sweep_population(models, waveforms=("tDCS", "TMS"), n_directions=32)
    assert len(rows) == 8
    assert {r["waveform"] for r in rows} == {"tDCS", "TMS"}
    assert all("total_length_um" in r and "species" in r for r in rows)


def test_rotating_the_neuron_rotates_the_best_direction() -> None:
    """Thresholds must be rotation-equivariant: the physics has no preferred axis."""
    axis_a = np.array([0.0, 0.0, 1.0])
    axis_b = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    a = orientation_sweep(CableModel(straight_cable(axis=axis_a)), n_directions=256)
    b = orientation_sweep(CableModel(straight_cable(axis=axis_b)), n_directions=256)
    assert a["threshold_min_Vpm"] == pytest.approx(b["threshold_min_Vpm"], rel=0.02)
    assert abs(np.dot(b["best_direction"], axis_b)) > 0.9


# -------------------------------------------------------------------- popstats
def _synthetic_rows(n_per: int = 30, seed: int = 1) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for species, base in (("human", 2.4), ("mouse", 2.1)):
        for cell_type, bump in (("pyramidal", 0.0), ("interneuron", 0.15)):
            for archive in ("labA", "labB", "labC"):
                offset = {"labA": -0.05, "labB": 0.0, "labC": 0.05}[archive]
                for k in range(n_per):
                    log_t = base + bump + offset + rng.normal(0, 0.18)
                    rows.append(
                        {
                            "name": f"{species}_{cell_type}_{archive}_{k}",
                            "waveform": "tDCS",
                            "threshold_min_Vpm": float(10**log_t),
                            "total_length_um": float(np.exp(rng.normal(8.5, 0.4))),
                            "species": species,
                            "cell_type": cell_type,
                            "archive": archive,
                            "brain_region": "neocortex",
                            "layer": "layer 5",
                        }
                    )
    return rows


def test_prepare_frame_builds_expected_columns() -> None:
    df = prepare_frame(_synthetic_rows(5))
    for col in ("threshold", "log10_threshold", "log10_total_length", "cell_class"):
        assert col in df.columns
    assert set(df["cell_class"]) == {"pyramidal", "interneuron"}
    assert np.isfinite(df["log10_threshold"]).all()


def test_prepare_frame_drops_infinite_thresholds() -> None:
    rows = _synthetic_rows(3)
    rows.append({**rows[0], "name": "dead", "threshold_min_Vpm": float("inf")})
    df = prepare_frame(rows)
    assert "dead" not in set(df["name"])


def test_prepare_frame_validates_input() -> None:
    with pytest.raises(ValueError, match="no rows"):
        prepare_frame([])
    with pytest.raises(ValueError, match="not in"):
        prepare_frame(_synthetic_rows(2), threshold_col="nope")


def test_variance_components_recovers_planted_structure() -> None:
    df = prepare_frame(_synthetic_rows(40))
    vc = variance_components(df).set_index("factor")
    # species and cell_class were planted with real offsets, archive with a tiny one
    assert vc.loc["species", "eta_squared"] > vc.loc["archive", "eta_squared"]
    assert 0.0 <= vc.loc["archive", "eta_squared"] < 0.1


def test_bootstrap_ratio_ci_brackets_the_truth() -> None:
    df = prepare_frame(_synthetic_rows(40))
    out = bootstrap_ratio_ci(df, n_boot=400, seed=3)
    truth = 10**0.3  # human base was 0.3 dex above mouse
    assert out["ci_low"] < out["ratio"] < out["ci_high"]
    assert out["ci_low"] < truth < out["ci_high"]
    assert out["n_clusters"] == 3


def test_bootstrap_ratio_ci_rejects_missing_group() -> None:
    df = prepare_frame(_synthetic_rows(5))
    with pytest.raises(ValueError, match="empty group"):
        bootstrap_ratio_ci(df, numerator="rat")


def test_class_separability_reports_overlap() -> None:
    df = prepare_frame(_synthetic_rows(40))
    sep = class_separability(df)
    assert len(sep) == 1
    row = sep.iloc[0]
    assert 0.0 <= row["auc"] <= 1.0
    # A 0.15 dex shift against 0.18 dex noise gives real but far from perfect
    # separation. The sign depends on which class lands in column a, so test the
    # magnitude of the departure from chance and require d and auc to agree.
    assert 0.05 < abs(row["auc"] - 0.5) < 0.45
    assert np.sign(row["cohens_d"]) == np.sign(row["auc"] - 0.5)


def test_mixed_model_recovers_species_effect() -> None:
    statsmodels = pytest.importorskip("statsmodels")
    assert statsmodels is not None
    from morph_stim.popstats import fit_threshold_mixed_model

    df = prepare_frame(_synthetic_rows(40))
    res = fit_threshold_mixed_model(df, fixed="species + cell_class + log10_total_length")
    assert res.n_obs == len(df)
    assert res.n_groups == 3
    names = [str(i) for i in res.coefficients.index]
    species_term = [n for n in names if n.startswith("species")]
    assert species_term, f"no species coefficient in {names}"
    # human is the reference level alphabetically, so mouse should be ~-0.3 dex
    est = float(res.coefficients.loc[species_term[0], "estimate"])
    assert est == pytest.approx(-0.3, abs=0.1)
    tbl = res.summary_table()
    assert "fold_change" in tbl.columns


def test_mixed_model_requires_enough_rows() -> None:
    pytest.importorskip("statsmodels")
    from morph_stim.popstats import fit_threshold_mixed_model

    df = prepare_frame(_synthetic_rows(1)).head(4)
    with pytest.raises(ValueError, match=">=10 complete rows"):
        fit_threshold_mixed_model(df)


# --------------------------------------------------------------- neuron driver
def test_neuron_driver_degrades_gracefully() -> None:
    from morph_stim import neuron_driver

    assert isinstance(neuron_driver.neuron_available(), bool)
    if not neuron_driver.neuron_available():
        with pytest.raises(RuntimeError, match="NEURON is not available"):
            neuron_driver.neuron_threshold(straight_cable(n=20), np.array([0.0, 0.0, 1.0]))


def test_calibration_report_on_perfect_agreement() -> None:
    from morph_stim.neuron_driver import calibration_report

    a = [100.0, 200.0, 400.0, 800.0]
    out = calibration_report(a, [2.0 * x for x in a])
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["log_slope"] == pytest.approx(1.0, abs=1e-6)
    assert out["median_fold_error"] == pytest.approx(2.0, rel=1e-6)
    with pytest.raises(ValueError, match=">=3 finite pairs"):
        calibration_report([1.0, float("nan")], [1.0, 2.0])
