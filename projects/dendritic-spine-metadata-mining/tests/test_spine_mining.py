"""Tests with synthetic SWC trees, synthetic text/XML and a simulated literature table (no network)."""
import numpy as np
import pytest

from spine_mining.europepmc import xml_to_text
from spine_mining.meta_regression import meta_regression, method_r2_permutation, prepare_effects, simulate_literature
from spine_mining.swc_spines import audit_tree, detect_spines, read_swc
from spine_mining.text_extraction import extract_statements, normalise_density, tag_context


def _dendrite_with_spines(n_dend=60, step=2.0, spine_every=4, spine_len=1.2, custom_type=False):
    """Straight basal dendrite along x with short side protrusions (spines)."""
    lines = ["1 1 0 0 0 5 -1"]
    idx = 2
    prev = 1
    dend_ids = []
    for i in range(1, n_dend + 1):
        lines.append(f"{idx} 3 {i * step:.2f} 0 0 0.6 {prev}")
        dend_ids.append(idx)
        prev = idx
        idx += 1
    for k, d in enumerate(dend_ids):
        if k % spine_every == 0 and 0 < k < n_dend - 1:
            t = 5 if custom_type else 3
            lines.append(f"{idx} {t} {(k + 1) * step:.2f} {spine_len:.2f} 0 0.3 {d}")
            idx += 1
    return "\n".join(lines) + "\n"


def test_detect_spines_geometry_and_custom_type():
    swc = read_swc(_dendrite_with_spines())
    det = detect_spines(swc)
    assert det["by_geometry"].sum() == 14 and det["by_type"].sum() == 0
    rec = audit_tree(swc)
    assert rec["n_spines"] == 14
    assert rec["dendrite_length_um"] == pytest.approx(120.0, rel=1e-6)
    assert rec["spine_density_per_um"] == pytest.approx(14 / 120.0, rel=1e-6)
    assert rec["has_spine_annotation"]
    swc2 = read_swc(_dendrite_with_spines(custom_type=True))
    det2 = detect_spines(swc2)
    assert det2["by_type"].sum() == 14 and det2["spine"].sum() == 14
    # a dendrite without protrusions has no spines and the tip is not miscounted
    swc3 = read_swc(_dendrite_with_spines(spine_every=10 ** 6))
    assert audit_tree(swc3)["n_spines"] == 0


def test_extract_statements_units_and_context():
    text = ("In control mice, spine density on basal dendrites of layer 5 pyramidal neurons was 1.8 ± 0.3 spines/µm "
            "(n = 12 cells) as measured with Golgi-Cox staining. In the human temporal cortex, density reached "
            "12.4 spines per 10 μm on apical oblique branches imaged by confocal microscopy. "
            "Surface density was 2.1 spines/µm2. Values ranged from 8 to 14 per 100 µm in the distal tuft.")
    st = extract_statements(text)
    vals = {round(s.density_per_um, 3) if s.density_per_um is not None else None for s in st}
    assert 1.8 in vals and 1.24 in vals and None in vals
    first = [s for s in st if s.value == 1.8][0]
    assert first.uncertainty == 0.3 and first.n_reported == 12 and first.n_unit == "cells"
    assert "mouse" in first.tags["species"] and "golgi" in first.tags["method"] and "basal" in first.tags["compartment"]
    rng_stmt = [s for s in st if s.value_high is not None][0]
    assert rng_stmt.density_per_um == pytest.approx(0.08) and rng_stmt.density_high_per_um == pytest.approx(0.14)
    surface = [s for s in st if s.is_surface][0]
    assert surface.density_per_um is None
    assert normalise_density(12.4, 10, False) == pytest.approx(1.24)
    tags = tag_context("apical oblique dendrites of CA1 pyramidal cells in rat")
    assert tags["compartment"] == ["apical_oblique"] and "hippocampus_ca1" in tags["region"] and "rat" in tags["species"]


def test_no_statement_without_spine_context():
    assert extract_statements("The synapse density was 1.2 per µm in this sample.") == []


def test_xml_to_text_body_tables_and_captions():
    xml = ("<article><front><article-meta/></front><body><sec><title>Results</title>"
           "<p>Spine density was 1.5 spines/µm.</p><table-wrap><caption><p>Table 1. Densities</p></caption>"
           "<table><tr><th>Group</th><th>spines per 10 µm</th></tr><tr><td>Control</td><td>15.2</td></tr></table></table-wrap>"
           "</sec></body></article>")
    txt = xml_to_text(xml)
    assert "Spine density was 1.5 spines/µm." in txt
    assert "Control | 15.2" in txt and "Group | spines per 10 µm" in txt
    assert "Table 1. Densities" in txt
    assert xml_to_text("<not xml") == ""


def test_meta_regression_recovers_method_effects():
    df = prepare_effects(simulate_literature(n_studies=240, n_labs=40, seed=1))
    res = meta_regression(df, ["species", "compartment", "method"], cluster="lab")
    coef = res["coef"]
    # reference level is alphabetical: em; golgi and fluorescence are relative to em
    assert coef["method_golgi"] == pytest.approx(-0.8, abs=0.2)
    assert coef["method_fluorescence"] == pytest.approx(-0.3, abs=0.2)
    # species reference level is 'human' (alphabetical); mouse and rat sit ~0.25 log units lower
    assert coef["species_mouse"] < -0.1 and coef["species_rat"] < -0.1
    assert res["p"]["method_golgi"] < 0.001
    assert 0 <= res["I2"] <= 1
    assert res["R2_between"] > 0.3
    perm = method_r2_permutation(df, ["method"], ["species", "compartment"], n_perm=20, seed=0)
    assert perm["observed_R2_method"] > perm["null_mean"] and perm["p_perm"] < 0.1
