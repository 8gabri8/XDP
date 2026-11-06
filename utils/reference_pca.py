"""Factors of cells from a reference: PCA fitted on reference cells and projection of other cells onto it,
cNMF usages on spectra trained elsewhere, dorsal-ventral gradient score.

Used by script/HD_Handsaker/DEG/PCA_on_BICAN.ipynb (fit on BICAN with its own gene names, save)
and script/HD_Handsaker/DEG/Handsaker_DEG_repeats_NEBULA.ipynb (load or fit, project HD cells).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from scipy.optimize import nnls


def fit_pca(adata, n_hvg, n_pcs, target_sum, seed):
    """PCA on the raw counts in adata.X (cells x genes).

    Steps: normalize_total + log1p -> n_hvg highly variable genes -> z-score each HVG -> PCA.

    Returns a small AnnData (no X) with everything needed to project other cells:
        var_names                genes used for the library size normalisation
        var["highly_variable"]   genes in the PCA
        var["mean"], var["std"]  z-scoring of each HVG
        varm["PCs"]              loadings (0 for genes that are not HVG)
        obsm["X_pca"]            scores of the reference cells, obs = their metadata
        uns["variance_ratio"], uns["target_sum"]
    """
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat")

    hvg = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(hvg)                                   # z-score; saves var["mean"], var["std"]
    sc.pp.pca(hvg, n_comps=n_pcs, random_state=seed)

    pca = ad.AnnData(obs=adata.obs.copy(), var=pd.DataFrame(index=adata.var_names))
    pca.var["highly_variable"] = a.var["highly_variable"].values
    pca.var["mean"] = hvg.var["mean"].reindex(pca.var_names).fillna(0).values
    pca.var["std"] = hvg.var["std"].reindex(pca.var_names).fillna(1).values
    pca.varm["PCs"] = pd.DataFrame(hvg.varm["PCs"], index=hvg.var_names).reindex(pca.var_names).fillna(0).values
    pca.obsm["X_pca"] = hvg.obsm["X_pca"]
    pca.uns["variance_ratio"] = hvg.uns["pca"]["variance_ratio"]
    pca.uns["target_sum"] = target_sum
    return pca


def project_pca(adata, pca, match_on=None):
    """PC scores of the cells in adata (raw counts in X) on a PCA made by fit_pca.

    Same steps as the fit: normalize_total + log1p over the PCA genes, z-score with the reference
    mean / std, multiply by the loadings.

    match_on: column of pca.var whose values are adata.var_names (e.g. "gene_id" for Ensembl IDs);
              None = match pca.var_names to adata.var_names.
    PCA genes missing in adata are counted as 0 counts; how many are found is printed.
    """
    pca_genes = pd.Index(pca.var_names if match_on is None else pca.var[match_on].astype(str))
    hvg = pca.var["highly_variable"].values
    cols = adata.var_names.get_indexer(pca_genes)          # -1: gene not in adata
    found = cols >= 0
    print(f"  {found.sum()}/{len(found)} PCA genes found, {(found & hvg).sum()}/{hvg.sum()} HVGs")

    # counts over the PCA genes, in the PCA gene order; missing genes -> 0
    X = sp.csr_matrix(adata.X, dtype=np.float32)[:, np.where(found, cols, 0)]
    X = X @ sp.diags(found.astype(np.float32))
    a = ad.AnnData(X)
    sc.pp.normalize_total(a, target_sum=float(pca.uns["target_sum"]))
    sc.pp.log1p(a)

    hvg = pca.var["highly_variable"].values
    x = sp.csr_matrix(a.X[:, hvg])
    mean = pca.var["mean"].values[hvg]
    std = pca.var["std"].values[hvg]
    loadings = pca.varm["PCs"][hvg]

    # ((x - mean) / std) @ loadings, written so that x stays sparse
    scores = x @ (loadings / std[:, None]) - (mean / std) @ loadings
    names = [f"PC_{i + 1}" for i in range(loadings.shape[1])]
    return pd.DataFrame(np.asarray(scores), index=adata.obs_names, columns=names)


def knee_point(variance_ratio):
    """Number of PCs up to the knee of the scree curve: the PC farthest from the straight line
    joining the first and the last PC (curve scaled to 0..1 on both axes)."""
    y = np.asarray(variance_ratio) / variance_ratio[0]
    x = np.linspace(0, 1, len(y))
    distance = np.abs(y - (y[0] + (y[-1] - y[0]) * x))
    return int(np.argmax(distance)) + 1


def cnmf_usage(adata, spectra, reference, normalise):
    """GEP usages of the cells in adata (raw counts in X) on cNMF spectra (DataFrame GEPs x genes)
    trained on `reference` (raw counts in X).

    As cNMF: each gene is divided by its SD in the reference counts, then each cell is fitted on the
    spectra by non-negative least squares. normalise: usages as fractions per cell (sum to 1).
    Spectra genes missing in adata or in the reference are left out (number printed).
    """
    genes = spectra.columns[spectra.columns.isin(adata.var_names) & spectra.columns.isin(reference.var_names)]
    print(f"  {len(genes)}/{spectra.shape[1]} spectra genes found")

    R = sp.csr_matrix(reference[:, genes].X, dtype=np.float64)
    mean = np.asarray(R.mean(axis=0)).ravel()
    sd = np.sqrt(np.asarray(R.multiply(R).mean(axis=0)).ravel() - mean ** 2)
    sd[sd == 0] = 1

    X = sp.csr_matrix(adata[:, genes].X, dtype=np.float64).toarray() / sd
    G = spectra[genes].values.T                                 # genes x GEPs
    usage = np.array([nnls(G, x)[0] for x in X])
    if normalise:
        usage = usage / np.maximum(usage.sum(axis=1, keepdims=True), 1e-12)
    return pd.DataFrame(usage, index=adata.obs_names, columns=spectra.index)


def gradient_score(adata, weights, target_sum):
    """Dorsal-ventral position score of each cell in adata (raw counts in X), as in NucSeq_DEG_factors:
    normalize_total + log1p, z-score each gradient gene over these cells, sum weighted by the gene's
    spatial rho (weights: Series gene -> rho, genes matched to adata.var_names; number found printed).
    """
    genes = weights.index[weights.index.isin(adata.var_names)]
    print(f"  {len(genes)}/{len(weights)} gradient genes found")

    X = sp.csr_matrix(adata.X, dtype=np.float64)
    counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
    L = sp.diags(target_sum / np.maximum(counts_per_cell, 1)) @ X[:, adata.var_names.get_indexer(genes)]
    L = np.log1p(L.toarray())
    L = (L - L.mean(axis=0)) / (L.std(axis=0) + 1e-9)          # z-score per gene
    return pd.Series(L @ weights[genes].values, index=adata.obs_names)
