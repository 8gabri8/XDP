import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import random
import numpy as np

import scvi
import harmonypy

import decoupler as dc

import math

custom_palette = ["#E63946","#2196F3","#4CAF50","#FF9800","#9C27B0","#00BCD4","#FFEB3B","#F06292","#8BC34A","#FF5722","#3F51B5","#009688","#795548","#607D8B","#E91E63","#CDDC39","#FF1744","#00E676","#D500F9","#FFAB40","#1DE9B6","#FF6D00","#6200EA","#76FF03","#FF4081","#00B0FF","#FFD740","#69F0AE","#F50057","#40C4FF","#B2FF59","#EA80FC","#CCFF90","#84FFFF","#FF6E40","#18FFFF","#E040FB","#AEEA00","#FF9E80","#80D8FF","#CFD8DC"]



def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Makes CUDA ops deterministic (can slow things down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import scvi
        scvi.settings.seed = seed  # ← critical for scVI
    except ImportError:
        pass


def preprocess_common_pipeline(adata, umap_name="", leiden_name="", resolutions=[0.1, 0.5, 0.07], integrate_with_harmony=False, integrate_with_scvi=False, batch_key=None, categorical_covariate_keys=None):

    SEED =42
    set_global_seeds(SEED)
    
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

def plot_to_decide(adata_nn, NAME, ADATA_NN_LEIDEN_COL_FOR_MAPPING, GROUP_LABELS, palette=custom_palette):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sc.pl.embedding(adata_nn, basis=f"umap{NAME}", color=ADATA_NN_LEIDEN_COL_FOR_MAPPING, legend_loc="on data", ax=axes[0], show=False,legend_fontsize=20, palette=palette)
    sc.pl.embedding(adata_nn, basis=f"umap{NAME}", color="Group_name", groups=GROUP_LABELS, legend_loc="on data", ax=axes[1], show=False, palette=palette)
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


def plot_umap_spatial(adata, color, name="", palette=custom_palette, gene_symbols="gene_symbol", ncols=1, **kwargs):

    import math

    sym_to_id = dict(zip(adata.var[gene_symbols], adata.var_names))
    id_to_sym = dict(zip(adata.var_names, adata.var[gene_symbols]))

    def resolve(c):
        if isinstance(c, list):
            return [sym_to_id.get(g, g) for g in c]
        return sym_to_id.get(c, c)

    color_resolved = [c for item in color for c in (resolve(item) if isinstance(item, list) else [resolve(item)])]
    color_labels   = [id_to_sym.get(c, c) for c in color_resolved]

    n = len(color_resolved)
    nrows = math.ceil(n / ncols)

    # each "slot" is 2 subplots wide (umap + spatial), so total cols = ncols * 2
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(6 * ncols * 2, 4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols * 2)

    for i, (c, label) in enumerate(zip(color_resolved, color_labels)):
        row = i // ncols
        col = (i % ncols) * 2  # umap at col, spatial at col+1
        sc.pl.embedding(adata, basis=f"umap{name}", color=[c], ax=axes[row, col],   show=False, palette=palette, title=label, **kwargs)
        sc.pl.embedding(adata, basis="spatial",     color=[c], ax=axes[row, col+1], show=False, palette=palette, title=label, **kwargs)

    # hide unused axes
    for j in range(i + 1, nrows * ncols):
        row, col = j // ncols, (j % ncols) * 2
        axes[row, col].set_visible(False)
        axes[row, col + 1].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_umap_spatial_samples(adata, color, SAMPLE_VARIABLE,CLUSTERS, name="", palette=custom_palette, gene_symbols="gene_symbol", ncols=1, **kwargs):
    for s in sorted(adata.obs[SAMPLE_VARIABLE].unique()):
        fig, axes = plt.subplots(2, len(CLUSTERS), figsize=(len(CLUSTERS) * 3, 5), squeeze=False)
        for (ax0, ax1), c in zip(axes.T, CLUSTERS):
            sc.pl.embedding(adata[adata.obs[SAMPLE_VARIABLE] == s], basis="spatial", color=color, groups=[c], palette=custom_palette, size=20, ax=ax0, show=False,title=c, **kwargs)
            sc.pl.embedding(adata[adata.obs[SAMPLE_VARIABLE] == s], basis=f"umap{name}", color=color, groups=[c], palette=custom_palette, size=20, ax=ax1, show=False,title=c, **kwargs)
        plt.suptitle(f"Sample: {s}"); plt.tight_layout(); plt.show()

def aucell(adata_nn, gene_list, name):
    name_to_id = dict(zip(adata_nn.var["gene_symbol"], adata_nn.var_names))
    mapped   = [name_to_id[g] for g in gene_list if g in name_to_id]
    print(f"Mapped {len(mapped)}/{len(gene_list)} genes for {name}")
    net = pd.DataFrame({"source": name, "target": mapped, "weight": 1.0})
    dc.mt.aucell(data=adata_nn, net=net, layer="counts", raw=False, verbose=False, tmin=3)
    score_col = f"{name}"
    adata_nn.obs[score_col] = adata_nn.obsm["score_aucell"][name].values
    del adata_nn.obsm["score_aucell"]


handsaker_genes = [
    "PDE10A", "PPP2R2B", "PPP3CA", "PHACTR1", "RYR3", "KCNA4", "KCNAB1", "KCND2", "KCNH1", "KCNK1", "KCNQ5", "KCTD1",
    "HOXA", "HOXB", "HOXC", "HOXD", "HOXB2",
    "HOTAIR", "HOTTIP", "HOTAIRM1",
    "FOXD1", "LHX9", "ONECUT1", "POU4F2", "SHOX2", "SIX1", "TBX5", "TLX2", "ZIC4",
    "SLC17A6", "SLC17A7", "SLC6A5", "SLC1A2", "CALB2", "KCNC2",
    "CDKN2A", "CDKN2B",
]

prc2_genes = [
    "POU4F2", "TWIST1", "ALDH1A2", "IRX5", "CALCR", "CDKN2A", "SKOR1", "PITX2",
    "HOXC8", "WT1", "SFMBT2", "POU4F1", "TAL1", "PNMT", "DSP", "CDHR1", "SPP1",
    "HOXD8", "NKX6-1", "HOXC4", "RIPK4", "LECT1", "ALDH1A7",
    "PYGL", "DMRT3", "TFAP2C", "FST", "GPR101", "RUNX3", "ONECUT2", "BARX1",
    "AMN", "TMEM30B", "IRX3", "CCND1", "IRX2", "S100A10", "FAM196B", "LGALS3",
    "SIX6", "COX6B2", "TBX2", "CD44", "ASB4", "PTPLAD2", "SCARF2", "VGLL2",
    "BNC2", "LYPD1", "GATA3", "OTOF", "HOXD10", "HOXA9", "HOXC10", "SLC6A5",
    "LBX1", "HOXA5", "TBX20", "HAND2", "HOXD9", "HOXB7", "HOXB8", "MNX1",
    "DMRTA2", "FOXF1", "SIM2", "GATA6", "SIX2", "LUM", "GATA2", "PMAIP1",
    "HOXA10", "BHLHE22", "FOXD1", "PITX1", "PAX1", "HMX1", "TBX3",
    "FOXA1", "EBF3", "SIM1", "HOXB13", "EN1",
    "HOXD13", "LHX5", "HOXA7", "HOXC5", "NEUROD1", "TFAP2A",
    "HOXB5", "SHOX2", "TFAP2B", "NEUROD2", "TLX1", "TNNT2",
    "VAX2", "HOXD11", "CDKN1A", "NEUROG2",
    "EOMES", "TPBG", "SSTR3", "DLK1", "PRDM8",
    "HOXA4", "NRN1", "VSX2", "GFI1",
    "ADCYAP1", "LHX1", "NKX3-1", "LHX9", "ZIC2", "HOXB3",
    "IGFBP3", "FBN1", "CDKN2B", "HOXB2", "HEYL", "NKX2-5",
    "TSHZ2", "FOXD2", "DLX4", "SIX1",
    "HOXB6", "LMX1B", "CBX4",
    "PAX3", "SFRP2",
    "GATA4", "ONECUT1", "DAB1",
    "NTF3", "GRHL1", "HOXA3", "GREB1L", "CPA6", "DMRT2",
    "HOXD3", "ALX1",
]

xdp_genes = [
    "TAF1", "OGT", "OGA", "ACRC", "CXCR3", "NONO", "ING2", "NLGN3", "GJB1", "ZNF261", "ITGB1BP2",
    "TBP", "TAF2", "TAF3", "TAF4", "TAF5", "TAF6", "TAF7", "TAF8", "TAF9", "TAF10", "TAF11", "TAF12", "TAF13",
    "TP53", "MDM2", "RB1", "AR", "JUN", "CCND1", "CCNA2",
    "GTF2B", "GTF2A1", "GTF2A2", "GTF2E1", "GTF2E2", "GTF2F1", "GTF2F2", "GTF2H1", "GTF2H2", "GTF2H3", "GTF2H4", "GTF2H5", "TCEA1", "ERCC2", "ERCC3",
    "KMT2A", "ASH2L", "HCFC1", "WDR5", "RBBP5",
    "DRD2", "DRD1", "PDE10A", "GNAL",
    "SYP", "SNAP25", "RAB26", "RAB27B", "RAB8B", "DDC", "GCH1", "SLC18A2", "SYT7", "DLG4", "SNAP91", "DBH", "SYTL2", "SLC5A3", "EFNB1",
    "RELA", "NFKB1", "NFKB2", "IKBKB", "IKBKG", "TNFAIP6", "IL8",
    "MSH3", "PMS2",
    "SRSF2", "DHX15", "DHX9", "FUS", "GEMIN5", "HRNPU", "NOVA1", "PAPOLA", "PRPF6", "SART3", "SRSF5", "RALY", "U2AF1L5",
    "BMS1", "TRMT5",
    "ZNF91", "TRIM28", "ZNF93", "SETDB1", "DNMT3A", "DNMT3B",
    "L1RE1",
    "UPF1", "UPF2", "UPF3B",
    "KMT5C", "CIITA",
    "TAF1L",
    "TIA1", "ZNF772", "SMG7", "MOV10L1", "EXO1", "SSH", "GLIA1", "GLI3",
]

# corr > 0.2
spatial_genes = [
    'TESPA1', 'ZNF385B', 'AL138701.2', 'AL139379.1', 'PLEKHA7', 'EPB41', 'PDE10A', 'LARGE1',
    'GLI3', 'MERTK', 'SRGAP1', 'PLCB1', 'OPCML', 'VLDLR-AS1', 'KCNT1', 'MME', 'ARHGAP10',
    'ELL2', 'SCARB1', 'RYR3', 'TTLL11', 'SLC24A2', 'TRPC6', 'AL158077.2', 'COL25A1', 'ENTPD3',
    'GPRIN3', 'SPOCK1', 'EFR3B', 'PCSK6', 'ENSG00000287923', 'CNR1', 'ENSG00000286780', 'RGS9',
    'SAMD5', 'DGKB', 'IVNS1ABP', 'AL033504.1', 'LDLRAD4', 'RGS7BP', 'LINC00632', 'RBMS1',
    'AFF3', 'PRKCB', 'SMPD3', 'SYNDIG1', 'FAM135B', 'LINC02822', 'TCF7L1', 'ADCY1', 'STXBP6',
    'KCND2', 'CDH6', 'AL450332.1', 'FUT9', 'PLEK', 'ENSG00000227681', 'STRN', 'SASH1',
    'AL445123.2', 'DACH1', 'NFATC2', 'RIC8B', 'PTPN7', 'GNG7', 'EXTL2', 'YARS', 'THSD4',
    'SCARA3', 'ROBO2', 'EPHA5', 'CHRM3', 'SMAD3', 'ENSG00000286685', 'DLC1', 'ONECUT2',
    'GUCY1A2', 'DPP6', 'JCAD', 'ADCY5', 'LINC02137', 'NIN', 'RTTN', 'EXPH5', 'GNAO1',
    'ATP2B1', 'SLC9A1', 'HS3ST5', 'KIAA1217', 'SCN4B', 'GRM3', 'PRKD1', 'IQCJ-SCHIP1',
    'SLC38A11', 'LOXHD1', 'SIK3', 'AC062028.1', 'AC004551.1', 'IQCA1', 'CD226', 'PCAT1',
    'ITPR1', 'RCAN2', 'CCDC39', 'RASD2', 'WFDC2', 'MSI2', 'GRID2IP', 'ABHD12B', 'ADAM12',
    'STRIT1', 'ENSG00000287976', 'PPP2R2C', 'PLA2R1', 'ST3GAL6', 'KCTD1', 'CPNE8', 'CAMKK2',
    'CCDC3', 'SEC14L5', 'AL033523.1', 'PLCL2', 'INPP5A', 'AC013643.2', 'ATXN1', 'WIPF3',
    'TANC2', 'RHOBTB3', 'FGF14', 'LPL', 'HS3ST4', 'GRIN2A', 'RCSD1', 'GPR63',
    'ENSG00000289953', 'NCKAP5', 'CACNA2D3', 'PDE8B', 'KCNK2', 'KLHDC8A', 'SCN9A', 'FAT2',
    'MPDZ', 'WASHC4', 'DNAH14', 'MYRIP', 'PEA15', 'GABRA3', 'ANO4', 'LINC01484',
    'AC013287.1', 'ABI3BP', 'SORL1', 'AC005999.2', 'ZBTB20', 'MAP2', 'LINGO2', 'CDH12',
    'SLC35F1', 'SHC2', 'PGM2L1', 'CREG2', 'PSD3', 'SPEF2', 'LRRK1', 'CRYM', 'AC009899.1',
    'NPFFR1', 'PDE1C', 'TMEFF2', 'PCLO', 'NAV3', 'ST6GALNAC5', 'OCA2', 'SFT2D2', 'GABRB1',
    'PPFIA2', 'RGS6', 'EPB41L4B', 'ATP2B4', 'GNAI1', 'KCTD4', 'AK5', 'OPRM1', 'BRINP3',
    'LINC01902', 'KIAA2012', 'VSNL1', 'GABRA5', 'ERC2', 'PALMD', 'DGKK', 'P2RY1', 'ABLIM1',
    'ZNF83', 'ROBO1', 'GALNTL6', 'GRIA4', 'CADM1', 'HCN1', 'GDA', 'PRKG1', 'ARHGAP6',
]

INTERESTING_GENES = list(dict.fromkeys(handsaker_genes + prc2_genes + xdp_genes))