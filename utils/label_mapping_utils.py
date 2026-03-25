import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

import scvi
import harmonypy



def preprocess_common_pipeline(adata, umap_name="", leiden_name="", resolutions=[0.1, 0.5, 0.07], integrate_with_harmony=False, integrate_with_scvi=False, batch_key=None, categorical_covariate_keys=None):

    SEED =42
    
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()

    # Cehck that all abtcjes have enight cells, otheriw use usail hvg method
    print("Running HVG...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", 
                                     layer="counts", batch_key=batch_key)
        print(f"HVG computed with batch_key='{batch_key}'")
    except ValueError as e:
        print(f"WARNING: HVG with batch_key failed ({e}), retrying without batch_key")
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", 
                                     layer="counts", batch_key=None)

    if integrate_with_scvi:
        import scvi
        # Train scVI
        adata_hvg = adata[:, adata.var.highly_variable].copy()  # temp copy, HVG only
        scvi.model.SCVI.setup_anndata(adata_hvg, layer="counts", batch_key=batch_key, categorical_covariate_keys=categorical_covariate_keys)
        model = scvi.model.SCVI(adata_hvg, n_layers=2, n_latent=30)
        model.train(batch_size=512, max_epochs=50)# datasplitter_kwargs={'num_workers': 16})

        loss_key = "elbo_train" if "elbo_train" in model.history else "train_loss_epoch"
        plt.plot(model.history[loss_key]); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

        adata.obsm["X_scvi"] = model.get_latent_representation()
        sc.pp.neighbors(adata, use_rep="X_scvi", random_state=SEED)
        sc.tl.umap(adata, key_added=f"umap{umap_name}", random_state=SEED)

    elif integrate_with_harmony:
        print("Running Harmony...")
        import harmonypy
        if batch_key is None:
            raise ValueError("batch_key must be provided for Harmony integration.")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50, random_state=SEED)
        # Run Harmony on the PCA embedding
        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, vars_use=batch_key, max_iter_harmony=30)
        adata.obsm["X_pca_harmony"] = ho.Z_corr
        sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_pcs=30, random_state=SEED)
        sc.tl.umap(adata, key_added=f"umap{umap_name}", random_state=SEED)
        
    else:
        sc.pp.scale(adata, max_value=10) #z-score normalization
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50, random_state=SEED)
        sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=30, random_state=SEED) # from X_pca
        sc.tl.umap(adata, key_added=f"umap{umap_name}", random_state=SEED) # uese neighbors

    for r in resolutions:
        sc.tl.leiden(adata, resolution=r, key_added=f"leiden_{r}{leiden_name}", flavor="igraph", n_iterations=2, random_state=SEED) # use neighbors
    leiden_cluster_names = [f"leiden_{r}{leiden_name}" for r in resolutions]

    # ATTENTIUON
    adata.X = adata.layers["counts"].copy()  # restore original log1p-normalized data in adata.X
    
    return leiden_cluster_names

def which_clusters_are_split(adata, CLUSTERS_GROUP_ALL_CELLS_COL, MMC_ANNOTATION_LEVEL, CLEAN_THRESHOLD = 90):

    ct_per_cluster = (
        adata.obs
        .groupby([CLUSTERS_GROUP_ALL_CELLS_COL, MMC_ANNOTATION_LEVEL])
        .size() # eaw counts
        .unstack(fill_value=0) 
    )
    ct_per_cluster_pct = ct_per_cluster.div(ct_per_cluster.sum(axis=1), axis=0) * 100 # cal pct

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(ct_per_cluster_pct, annot=True, fmt=".0f", cmap="Blues",
                cbar_kws={"label": "% of cluster"}, ax=ax)
    ax.set_title(f"{CLUSTERS_GROUP_ALL_CELLS_COL} x {MMC_ANNOTATION_LEVEL}  (% of cluster)")
    ax.set_xlabel("MMC label"); ax.set_ylabel("Cluster")
    plt.tight_layout(); plt.show()

    # for each clutser applu ruels to say if it ok or need ot investigate
    #CLEAN_THRESHOLD = 90   # % to call a cluster "clean"

    cluster_status = {}
    for cluster, row in ct_per_cluster_pct.iterrows():
        top_label  = row.idxmax()
        top_pct    = row.max()
        if top_pct >= CLEAN_THRESHOLD:
            status = "clean"
        else:
            status = "INVESTIGATE"   # mixed → check marker genes manually
        cluster_status[cluster] = {"top_label": top_label, "top_pct": top_pct, "status": status}

    cluster_status_df = pd.DataFrame(cluster_status).T
    display(cluster_status_df)


def plot_to_select(adata, GROUP_LABELS, CLUSTERS_ALL_CELLS_COL, CLUSTERS_GROUP_ALL_CELLS):
    fig, ax = plt.subplots(1, 3, figsize=(20,5))
    sc.pl.umap(adata, color="Subclass_name", groups=GROUP_LABELS, ax=ax[0], show=False, legend_loc="on data")
    sc.pl.umap(adata, color=CLUSTERS_ALL_CELLS_COL, ax=ax[1], show=False, legend_loc="on data")
    sc.pl.umap(adata, color=CLUSTERS_ALL_CELLS_COL, groups=CLUSTERS_GROUP_ALL_CELLS, ax=ax[2], show=False, legend_loc="on data")
    plt.tight_layout(); plt.show()

def plot_to_decide(adata_nn, NAME, ADATA_NN_LEIDEN_COL_FOR_MAPPING, GROUP_LABELS):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sc.pl.embedding(adata_nn, basis=f"umap{NAME}", color=ADATA_NN_LEIDEN_COL_FOR_MAPPING, legend_loc="on data", ax=axes[0], show=False,legend_fontsize=20,)
    sc.pl.embedding(adata_nn, basis=f"umap{NAME}", color="Group_name", groups=GROUP_LABELS, legend_loc="on data", ax=axes[1], show=False)
    plt.tight_layout()
    plt.show()

# mmc mapping from subcalss to broad category (ussed for finding easylt clusters names)
subclass_cell_types_mapping = {
    "MSN": [
        "STR D1 MSN", "STR D2 MSN", "STR Hybrid MSN",
    ],
    "non_MSN_neuron": [
        "ACx MEIS2 GABA", "CN Cholinergic GABA", "CN GABA-Glut",
        "CN LAMP5-CXCL14 GABA", "CN LAMP5-LHX6 GABA", "CN LHX8 GABA",
        "CN MEIS2 GABA", "CN ONECUT1 GABA", "CN ST18 GABA", "CN VIP GABA",
        "F GABA", "F Glut", "F M GATA3 GABA", "F M Glut", "M Dopa",
        "OT Granular GABA", "SN PAX7 GABA", "STR SST-CHODL GABA","STR RSPO2 GABA"
    ],
    "glia": [
        "Astrocyte", "COP", "Ependymal", "Oligodendrocyte", "OPC"
    ],
    "vascular": [
        "Endo", "Pericyte", "SMC", "VLMC"
    ],
    "immune": [
        "Lymphocyte", "Macrophage", "Microglia"
    ],
}

# reverse map: cell type → broad category (useful for adata.obs)
celltype_to_broad = {
    cell: broad
    for broad, cells in subclass_cell_types_mapping.items()
    for cell in cells
}

markers_ct = {
    "Oligodendrocytes":     ["MBP", "MOG", "PLP1", "CNP", "OLIG2"],
    "Astrocytes":           ["GFAP", "AQP4", "ALDH1L1", "SLC1A2"],
    
    "Endothelial_BBB":      ["CLDN5", "OCLN", "SLC2A1", "MFSD2A"],  
    "Pericytes":            ["ABCC9", "RGS5", "KCNJ8", "IFITM1"],  
    "Fibroblasts":          ["PDGFRA", "DCN", "LUM", "COL1A1"],
    "SmoothMuscleCells":    ["MYH11", "CNN1", "MYOCD", "TAGLN"],

    "Microglia":            ["P2RY12", "TMEM119", "FCRLS", "SALL1"],
    "Monocytes":            ["CCR2", "LY6C2", "S100A8", "S100A9", "CD300E"],
    "Macrophages":          ["MRC1", "LYVE1", "CD163", "LGALS3", "CD200R1"],

    "Tcells":               ["CD3D", "CD3E", "CD8A", "GZMB"],
    "Bcells":               ["CD19", "MS4A1", "CD79A", "CD79B"],
    "NKcells":              ["NCAM1", "KLRD1", "NKG7", "GNLY", "KLRB1", "FCGR3A"],  # ← added

    "Proliferating":        ["MKI67", "TOP2A", "PCNA", "CDK1"],
    "DendriticCells":       ["CD1C", "FCER1A", "CLEC4C", "IRF7"],
    "Dissociation_Stress":  ["FOS", "JUN", "DUSP1", "HIF1A"],
}
