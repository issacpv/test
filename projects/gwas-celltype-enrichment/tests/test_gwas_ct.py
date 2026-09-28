"""Synthetic-data tests for gwas_ct core functions (no downloads)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwas_ct import abc_loader, enrichment, magma_io, scdrs_lite, specificity  # noqa: E402


def make_synthetic(n_types: int = 5, n_genes: int = 400, cells_per_type: int = 60, n_marker: int = 20, seed: int = 0):
    """Cells x genes matrix where each cell type over-expresses its own block of marker genes."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([f"T{i}" for i in range(n_types)], cells_per_type)
    X = rng.gamma(2.0, 0.5, size=(len(labels), n_genes))
    markers = {}
    n_marker = min(n_marker, n_genes // n_types)
    for i in range(n_types):
        block = np.arange(i * n_marker, (i + 1) * n_marker)
        X[np.ix_(labels == f"T{i}", block)] += 3.0
        markers[f"T{i}"] = [f"G{j}" for j in block]
    genes = [f"G{j}" for j in range(n_genes)]
    return X, labels, genes, markers


def test_pseudobulk_and_specificity_recover_markers():
    X, labels, genes, markers = make_synthetic()
    pb = abc_loader.pseudobulk(X, labels, genes)
    assert pb.shape == (5, 400)
    spec = specificity.specificity_matrix(pb)
    assert np.allclose(spec.sum(axis=0), 1.0)
    top = specificity.top_decile_genes(spec, "T2")
    assert len(set(top) & set(markers["T2"])) >= 18  # top decile = 40 genes, 20 planted markers


def test_pseudobulk_sparse_matches_dense():
    from scipy import sparse
    X, labels, genes, _ = make_synthetic(n_types=3, n_genes=50, cells_per_type=20)
    d = abc_loader.pseudobulk(X, labels, genes)
    s = abc_loader.pseudobulk(sparse.csr_matrix(X), labels, genes)
    assert np.allclose(d.to_numpy(), s.to_numpy())


def test_hierarchy_aggregation_is_cell_weighted():
    pb = pd.DataFrame([[1.0, 0.0], [3.0, 0.0]], index=["c1", "c2"], columns=["g1", "g2"])
    agg = specificity.aggregate_hierarchy(pb, {"c1": "S", "c2": "S"}, n_cells={"c1": 1, "c2": 3})
    assert agg.loc["S", "g1"] == pytest.approx(2.5)


def test_specificity_quantiles_range():
    X, labels, genes, _ = make_synthetic(n_types=3, n_genes=100, cells_per_type=10)
    spec = specificity.specificity_matrix(abc_loader.pseudobulk(X, labels, genes))
    q = specificity.specificity_quantiles(spec, n_bins=40)
    assert q.min().min() >= 0 and q.max().max() <= 40
    assert (q.loc["T0"] == 40).sum() >= 1


def test_bootstrap_enrichment_detects_planted_signal():
    X, labels, genes, markers = make_synthetic()
    rng = np.random.default_rng(1)
    gene_z = pd.Series(rng.normal(0, 1, len(genes)), index=genes)
    gene_z.loc[markers["T1"]] += 2.0  # GWAS signal in T1 markers
    spec = specificity.specificity_matrix(abc_loader.pseudobulk(X, labels, genes))
    sets = specificity.all_top_decile_sets(spec)
    tab = enrichment.enrichment_table(gene_z, sets, n_boot=2000, seed=0)
    assert tab.index[0] == "T1"
    assert tab.loc["T1", "p"] < 0.01
    assert tab.loc["T3", "p"] > 0.05
    # matched version runs and agrees on the winner
    tab_m = enrichment.enrichment_table(gene_z, sets, n_boot=500, seed=0, match_on=pd.Series(rng.random(400), index=genes))
    assert tab_m.index[0] == "T1"


def test_ewce_bootstrap_detects_planted_signal():
    X, labels, genes, markers = make_synthetic()
    spec = specificity.specificity_matrix(abc_loader.pseudobulk(X, labels, genes))
    hits = markers["T4"] + [f"G{j}" for j in range(300, 330)]
    res = enrichment.ewce_bootstrap(spec, hits, n_boot=2000, expr_level=pd.Series(X.mean(axis=0), index=genes))
    assert res.index[0] == "T4"
    assert res.loc["T4", "p"] < 0.01


def test_bh_fdr_and_cauchy():
    p = np.array([0.001, 0.01, 0.5, np.nan])
    q = enrichment.bh_fdr(p)
    assert np.isnan(q[3]) and q[0] <= q[1] <= q[2] and q[0] == pytest.approx(0.003)
    assert enrichment.cauchy_combination([0.01, 0.02]) < 0.05
    assert 0.4 < enrichment.cauchy_combination([0.5, 0.5]) < 0.6


def test_scdrs_lite_scores_planted_cells_higher():
    X, labels, genes, markers = make_synthetic(cells_per_type=40)
    weights = {g: 5.0 for g in markers["T0"]}
    scores, ctrl = scdrs_lite.score_cells(X, genes, weights, n_ctrl=30, n_bins=10, seed=0)
    assert scores.shape[0] == X.shape[0] and ctrl.shape == (X.shape[0], 30)
    assoc = scdrs_lite.group_association(scores["norm_score"].to_numpy(), ctrl, labels, min_cells=10)
    assert assoc.index[0] == "T0"
    assert assoc.loc["T0", "assoc_mcp"] < 0.1


def test_magma_files_roundtrip(tmp_path):
    sets = {"T0 neurons": ["A", "B", "C"], "T1": ["D"]}
    p = magma_io.write_magma_geneset_file(sets, tmp_path / "sets.txt", min_genes=2)
    lines = p.read_text().strip().splitlines()
    assert lines == ["T0_neurons A B C"]
    covar = pd.DataFrame({"T0": [1, 2], "T1": [3, 4]}, index=["A", "B"])
    p2 = magma_io.write_magma_covar_file(covar, tmp_path / "covar.tsv", id_map={"A": "11", "B": "22"})
    df = pd.read_csv(p2, sep="\t", index_col=0)
    assert list(df.index) == [11, 22] and "Average" in df.columns
    gsa = tmp_path / "x.gsa.out"
    gsa.write_text("# comment\nVARIABLE TYPE NGENES BETA BETA_STD SE P\nT0 SET 10 0.2 0.01 0.05 1e-4\n")
    out = magma_io.read_magma_gsa_out(gsa)
    assert out.loc[0, "P"] == pytest.approx(1e-4)
    cmds = magma_io.magma_commands("magma", "gwas.txt", tmp_path / "out", "genes.loc", "g1000", 10000,
                                   geneset_file=p, covar_file=p2)
    assert len(cmds) == 4 and "--gene-covar" in cmds[-1]


def test_manifest_parsing(tmp_path):
    manifest = {
        "version": "test",
        "directory_listing": {"DS": {}},
        "file_listing": {
            "DS": {
                "metadata": {"cell_metadata": {"files": {"csv": {"url": "u", "relative_path": "metadata/DS/cell_metadata.csv",
                                                                   "size": 1, "file_hash": "x"}}}},
                "expression_matrices": {"DS-1": {"log2": {"files": {"h5ad": {"url": "u2", "relative_path": "e/DS-1-log2.h5ad",
                                                                              "size": 2, "file_hash": "y"}}},
                                                 "raw": {"files": {"h5ad": {"url": "u3", "relative_path": "e/DS-1-raw.h5ad",
                                                                             "size": 3, "file_hash": "z"}}}}},
            }
        },
    }
    m = abc_loader.AbcManifest(manifest)
    assert m.datasets() == ["DS"]
    assert m.get("DS", "metadata", "cell_metadata").ext == "csv"
    assert m.get("DS", "expression_matrices", "DS-1").variant == "log2"
    with pytest.raises(KeyError):
        m.get("DS", "metadata", "nope")


def test_merfish_domain_labels_and_spatial_specificity():
    meta = pd.DataFrame({"subclass": ["A", "A", "B", "B"], "parcellation_division": ["CTX", "TH", "CTX", "unassigned"]},
                        index=[f"c{i}" for i in range(4)])
    lab = abc_loader.merfish_domain_labels(meta, "parcellation_division", "subclass")
    assert list(lab) == ["A|CTX", "A|TH", "B|CTX"]
    pb = pd.DataFrame([[2.0, 1.0], [2.0, 3.0], [1.0, 1.0]], index=["A|CTX", "A|TH", "B|CTX"], columns=["g1", "g2"])
    spec_all, within = specificity.spatial_specificity(pb)
    assert within.loc["A|CTX", "g2"] == pytest.approx(0.25)
    assert np.allclose(spec_all.sum(axis=0), 1.0)
