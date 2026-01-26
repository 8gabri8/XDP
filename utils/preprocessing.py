

import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from diptest import diptest
from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix

import rpy2.robjects as ro
r = ro.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def plot_qc_metrics(adata, pct_intronic_col="pct_intronic", Class_bootstrapping_probability_col="Class_bootstrapping_probability"):

    fig, axes = plt.subplots(2,3, figsize=(15, 10))

    axes = axes.flatten()

    # --- 1. Total UMI counts ---
        # GOOD
            # Single, smooth peak (unimodal distribution)
            # Peak centered around 8-10 on log scale (roughly 3,000-20,000 UMIs)
            # Narrow spread
        # BAD
            # Two peaks (bimodal): Suggests two populations - one could be empty droplets or debris
            # Very wide spread: Indicates high variability in sequencing depth between cells
            # Long left tail: Presence of empty droplets or broken cells
            # peask less than 7
    x = adata.obs["total_counts"]
    x = x[x > 0]
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    axes[0].clear()
    sns.histplot(x, bins=bins, kde=False, ax=axes[0])
    axes[0].set_title("Total UMI counts per cell")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_xlabel("log1p total counts")
    axes[0].set_xscale("log")


    # --- 2. Mitochondrial counts ---
        # GOOD
            # Most cells clustered at lower values
            # Smooth, single-peaked shape
        # BAD
            # Heavy upper tail: Many dying/stressed cells
            # Bimodal (two bumps): Two populations with different health states
    sns.histplot(adata.obs["pct_counts_mt"], bins=100, kde=False, ax=axes[1])
    axes[1].set_title("Mitochondrial percentage per cell")
    axes[1].set_ylabel("Number of cells")
    axes[1].set_xlabel("Percentage mito counts")
    axes[1].set_yscale("log")

    # --- 3. Scatter: counts vs genes, colored by mito ---
        # GOOD
            # Strong positive correlation: Diagonal line from bottom-left to top-right: More total counts = more genes detected
        # BAD
            #  Cells in bottom-left corner (low counts, low genes) --< Empty droplets or debris

    # Sort data so highest MT% values are plotted last (on top)
    sorted_idx = adata.obs["pct_counts_mt"].argsort()

    sns.scatterplot(
        x=adata.obs["log1p_total_counts"].iloc[sorted_idx],
        y=adata.obs["log1p_n_genes_by_counts"].iloc[sorted_idx],
        hue=adata.obs["pct_counts_mt"].iloc[sorted_idx],
        ax=axes[2],
        s=3,
        linewidth=0,
        palette="viridis"
    )
    axes[2].set_title("Counts vs Genes Colored by % MT")
    axes[2].set_xlabel("log1p(total counts)")
    axes[2].set_ylabel("log1p(n genes)")

    # --- 4. % introns ---
        # GOOD

        # BAD
            # having 2 peaks

    sns.histplot(adata.obs[pct_intronic_col], bins=100, kde=False, ax=axes[3])
    axes[3].set_title("Total percnet introns per cell")
    axes[3].set_ylabel("Number of cells")
    axes[3].set_xlabel("Percent intornic")

    # --- 5. n genes ---
        # GOOD

        # BAD
    x = adata.obs["n_genes_by_counts"]
    x = x[x > 0]
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    sns.histplot(x, bins=bins, kde=False, ax=axes[4])
    axes[4].set_title("Total genes per cell")
    axes[4].set_ylabel("Number of cells")
    axes[4].set_xlabel("Total counts")
    axes[4].set_xscale("log")

    # --- 6. mapmycell probs ---
        # GOOD

    if Class_bootstrapping_probability_col:
            # BAD
        axes[5].hist(
            adata.obs[Class_bootstrapping_probability_col].dropna(),
            bins=100, log=True
        )
        axes[5].set_title("Bootstrapping probability")

    #plt.tight_layout()
    plt.show()


def preprocess(adata, n_pcs_elbow=30, n_hvg=3000, hvg_batch_key=None, hvg_layer="counts", save_raw_counts=False, verbose=False):

    print("Expect .X is raw counts!")

    if save_raw_counts:
        adata.layers["counts"] = adata.X.copy()

    # Normalize & Transform --> X amtrix changes
    print("Normalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()

    # HVG
    print("Findingn HVGs...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3', batch_key=hvg_batch_key, layer=hvg_layer) #needs raw cpount if falvour=seurat3
        # do not remove not hvg right now! 
    if verbose:
        sc.pl.highly_variable_genes(adata, show=True)
            # BAD: nif there is not clear seapration

    # scale
        # ATTENTION: now X stores nroalised counts --> not used for all stat test alter
    print("Scaling...")
    sc.pp.scale(adata, max_value=10) #z-score normalization

    # PCA
    print("Calculating PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=80) # ATTENTION: # Uses HVGs only (if calculted, even if the adat has all geens)
    if verbose:
        sc.pl.pca_variance_ratio(adata, log=True, n_pcs=80, show=True)

    # neighbours (for umap and leiden)
    print("Calculating neighbors...")
    sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=n_pcs_elbow) # from X_pca

    # umap
    print("Calculating Umap...")
    sc.tl.umap(adata) # uese neighbors

    # clustering
    print("Clutering...")
    sc.tl.leiden(adata, resolution=0.1, key_added="leiden_1", flavor="igraph", n_iterations=2) # use neighbors
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden_2", flavor="igraph", n_iterations=2)
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden_3", flavor="igraph", n_iterations=2)
    sc.tl.leiden(adata, resolution=2.0, key_added="leiden_4", flavor="igraph", n_iterations=2)

    print("Preprocessing done.")

def reprocess_subset(adata, old_umap_name="", filter_genes=True):

    # bring back counts to X
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    if "X_umap" in adata.obsm:
        adata.obsm[f"{old_umap_name}_old_umap"] = adata.obsm["X_umap"].copy()

    # Filter genes
    if filter_genes:
        print(f"   Genes before filtering: {adata.n_vars}")
        sc.pp.filter_genes(adata, min_cells=3)
        print(f"   Genes after filtering: {adata.n_vars}")

    # reprocess
    preprocess(adata, n_pcs_elbow=30, n_hvg=3000, verbose=False)

    # Calculate QC
    adata.X = adata.layers["counts"].copy()     # bring back counts to X
    adata.var["mt"] = adata.var.gene_symbol.str.startswith("MT-") # mitochondrial genes, ATTNETION: the names are in "gene_symbol"
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

    adata.X = csr_matrix(adata.shape)


def show_top_pc_genes(adata, n_pcs=3, top_n=10, n_top_abs_genes_PC1=1):

    pcs_to_show = [f'PC{i}' for i in range(1, n_pcs + 1)]

    # Get loadings for first n_pcs
    loadings = pd.DataFrame(
        adata.varm['PCs'][:, :n_pcs],                # First n_pcs PCs (all genes x n_pcs)
        index=adata.var['gene_symbol'],              # Use gene symbols (keeps your original choice)
        columns=[f'PC{i}' for i in range(1, n_pcs + 1)]
    )

    # # Get top genes for each PC (print)
    # pcs_to_show = [f'PC{i}' for i in range(1, n_pcs + 1)]
    # for pc in pcs_to_show:
    #     print(f"\n{pc} - Top Positive Genes:")
    #     print(loadings[pc].nlargest(top_n))

    #     print(f"\n{pc} - Top Negative Genes:")
    #     print(loadings[pc].nsmallest(top_n))

    # Plot
    fig, axes = plt.subplots(1, n_pcs, figsize=(4 * n_pcs, 5))
    # if only one axis (n_pcs==1), make it indexable
    if n_pcs == 1:
        axes = [axes]

    for i, pc in enumerate(pcs_to_show):
        top_genes = pd.concat([
            loadings[pc].nlargest(top_n),
            loadings[pc].nsmallest(top_n)
        ]).sort_values()

        top_genes.plot(kind='barh', ax=axes[i], color='steelblue')
        axes[i].set_title(f'{pc} Top Genes')
        axes[i].set_xlabel('Loading')

    plt.tight_layout()
    plt.show()

    # PCs are capturing real biological variation: copy PC values into adata.obs
    for i in range(n_pcs):
        adata.obs[f'PC{i+1}'] = adata.obsm['X_pca'][:, i]
    # Plot UMAP colored by those PCs (keeps the original call but with the selected PCs)
    sc.pl.umap(adata, color=[f'PC{i+1}' for i in range(n_pcs)], cmap='RdBu_r')

    # Get top genes for PC1
    pc1_top_positive = loadings['PC1'].nlargest(n_top_abs_genes_PC1).index.tolist()
    pc1_top_negative = loadings['PC1'].nsmallest(n_top_abs_genes_PC1).index.tolist()

    return pc1_top_positive + pc1_top_negative


def plot_cell_type(adata, adata_spatial, ANNOTATION_LEVEL):

    subclasses = adata.obs[ANNOTATION_LEVEL].value_counts().index.to_list()
    n_subclasses = len(subclasses)

    # Calculate rows (ceiling division)
    n_rows = int(np.ceil(n_subclasses / 2))
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()  # Flatten to 1D for easy indexing

    for i, s in enumerate(subclasses):
        # Create highlight column
        adata.obs["temp_highlight"] = adata.obs[ANNOTATION_LEVEL].apply(
            lambda x: s if x == s else 'Other'
        )
        adata_spatial.obs["temp_highlight"] = adata_spatial.obs[ANNOTATION_LEVEL].apply(
            lambda x: s if x == s else 'Other'
        )
        
        # Spatial plot (left side: columns 0, 2, 4, ...)
        sc.pl.embedding(
            adata_spatial, 
            basis="spatial", 
            color="temp_highlight", 
            palette={'Other': 'lightgray', s: 'red'}, 
            na_color='#CCCCCC',  # Different shade to avoid conflict
            ax=axes[i*2], 
            show=False, 
            legend_loc=None, 
            size=10,
            title=f"{s} (spatial)"
        )
        
        # UMAP plot (right side: columns 1, 3, 5, ...)
        sc.pl.embedding(
            adata, 
            basis="X_umap", 
            color="temp_highlight", 
            palette={'Other': 'lightgray', s: 'red'}, 
            na_color='#CCCCCC',  # Different shade to avoid conflict
            ax=axes[i*2 + 1], 
            show=False, 
            legend_loc=None,
            title=f"{s} (UMAP)"
        )

    # Hide unused subplots
    for j in range(i*2 + 2, len(axes)):
        axes[j].axis('off')

    # Clean up temp column from BOTH objects
    adata.obs.drop('temp_highlight', axis=1, inplace=True)
    adata_spatial.obs.drop('temp_highlight', axis=1, inplace=True)  # Added this line

    plt.tight_layout()
    plt.show()


def plot_cell_type_distribution(adata, annotation_levels=["Class_name"], dropna=False):

    for ann in annotation_levels:
        vc = adata.obs[ann].value_counts(dropna=dropna).sort_values(ascending=False)
        vp = vc / vc.sum() * 100  # percentages

        fig, axes = plt.subplots(2,1, figsize=(12,8))

        # # log scale
        # vc.plot(kind="bar", logy=True, ax=axes[0])
        # axes[0].set_title(f"{ann} counts (log y)")
        # axes[0].set_xlabel("")
        # axes[0].set_ylabel("Counts")

        # linear scale
        vc.plot(kind="bar", logy=False, ax=axes[0])
        axes[0].set_title(f"{ann} counts (linear y)")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Counts")

        # percentages
        vp.plot(kind="bar", ax=axes[1])
        axes[1].set_title(f"{ann} percentages")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Percentage (%)")

        # rotate all x-labels
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)

        plt.tight_layout()
        plt.show()

def plot_cell_type_distribution_per_cluster(adata, ANNOTATION_LEVEL, CLUSTER_LEVEL):

    # Convert to string to avoid categorical issues, then replace NaN
    annotation_with_nan_label = adata.obs[ANNOTATION_LEVEL].astype(str).replace('nan', 'Not_Annotated')
    
    # Calculate cell type composition including "Not_Annotated"
    cluster_composition = pd.crosstab(
        adata.obs[CLUSTER_LEVEL],
        annotation_with_nan_label
    )
    
    # Move 'Not_Annotated' to the end if it exists
    if 'Not_Annotated' in cluster_composition.columns:
        cols = [col for col in cluster_composition.columns if col != 'Not_Annotated']
        cols.append('Not_Annotated')
        cluster_composition = cluster_composition[cols]

    # Convert to percentages
    cluster_composition_pct = cluster_composition.div(
        cluster_composition.sum(axis=1), axis=0
    ) * 100

    # Generate enough distinct colors
    n_types = len(cluster_composition.columns)
    
    # Combine multiple colormaps to get enough distinct colors
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    if n_types <= 20:
        colors = [cm.tab20(i) for i in range(n_types)]
    else:
        # Combine tab20, tab20b, tab20c, and Set3 for more colors
        colors = []
        colors.extend([cm.tab20(i) for i in range(20)])
        if n_types > 20:
            colors.extend([cm.tab20b(i) for i in range(min(20, n_types - 20))])
        if n_types > 40:
            colors.extend([cm.tab20c(i) for i in range(min(20, n_types - 40))])
        if n_types > 60:
            colors.extend([cm.Set3(i) for i in range(min(12, n_types - 60))])
        if n_types > 72:
            # For very large numbers, use hsv colormap
            colors.extend([cm.hsv(i/max(n_types-72, 1)) for i in range(n_types - 72)])
    
    # Make Not_Annotated gray if it exists
    if 'Not_Annotated' in cluster_composition.columns:
        colors[-1] = (0.7, 0.7, 0.7, 1.0)  # Gray color

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    # Plot percentages
    cluster_composition_pct.plot(
        kind='bar', 
        stacked=True,
        ax=axes[0],
        color=colors
    )
    axes[0].set_title(f'Cell Type Composition per {CLUSTER_LEVEL} (Percentage)')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].legend(title=ANNOTATION_LEVEL, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Plot counts
    cluster_composition.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        color=colors
    )
    axes[1].set_title(f'Cell Type Composition per {CLUSTER_LEVEL} (Counts)')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of cells')
    axes[1].legend(title=ANNOTATION_LEVEL, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def plot_cluster_and_get_valley(ax, scores, cluster, gmm, is_bimodal, p_value, score_col):
    """Plot histogram, Gaussians, and threshold if bimodal"""

    ################################

    def find_valley_threshold(scores, gmm):
        """Find the minimum (valley) between two Gaussian peaks"""

        # build the 2 gaussians
        means = np.sort(gmm.means_.flatten())
        stds = np.sqrt(gmm.covariances_.flatten()[np.argsort(gmm.means_.flatten())])
        weights = gmm.weights_[np.argsort(gmm.means_.flatten())]
        
        # Evaluate mixture PDF between the two means
        x = np.linspace(means[0], means[1], 2000)
        pdf = (weights[0] * norm.pdf(x, means[0], stds[0]) + 
            weights[1] * norm.pdf(x, means[1], stds[1]))
        
        return x[np.argmin(pdf)] # find valley as lowest porint bryween the 2 gaussianns (summed)

    ################################

    # Histogram
    ax.hist(scores, bins=50, density=True, alpha=0.6, color='gray', edgecolor='black')
    
    # Fit Gaussians
    means = np.sort(gmm.means_.flatten())
    stds = np.sqrt(gmm.covariances_.flatten()[np.argsort(gmm.means_.flatten())])
    weights = gmm.weights_[np.argsort(gmm.means_.flatten())]
    
    x = np.linspace(scores.min(), scores.max(), 1000)
    
    # Plot individual Gaussians
    for i in range(2):
        ax.plot(x, weights[i] * norm.pdf(x, means[i], stds[i]), 
                '--', linewidth=2, alpha=0.7)
    
    # Plot mixture
    pdf_total = sum(weights[i] * norm.pdf(x, means[i], stds[i]) for i in range(2))
    ax.plot(x, pdf_total, 'k-', linewidth=2, alpha=0.5)
    
    # Threshold if bimodal
    threshold = None
    if is_bimodal:
        threshold = find_valley_threshold(scores, gmm)
        ax.axvline(threshold, color='red', linewidth=3, label=f'Thr: {threshold:.3f}')
        ax.legend(fontsize=8)
    
    # Title with p-value
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    title = f"Cluster {cluster} {'✓ BIMODAL' if is_bimodal else ''}\np = {p_value:.4f} {sig}"
    ax.set_title(title, fontweight='bold', color='red' if is_bimodal else 'black', fontsize=10)
    ax.set_xlabel(score_col, fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    
    return threshold


def simple_wilcoxon_on_leiden(adata, leiden_col="leiden_1", leiden_cluster="0", n_genes_gsea=None):

    # Run rank_genes_groups for ALL clusters (each vs rest)
    sc.tl.rank_genes_groups(
        adata, 
        groupby=leiden_col,  # Compare each leiden cluster vs rest
        method='wilcoxon', 
        corr_method="benjamini-hochberg",
        
        groups=[leiden_cluster],  
        reference='rest',  # Default is 'rest'
        
        use_raw=False, 
        layer="log1p_norm",
        pts=True,
        
        key_added=F"leiden_{leiden_cluster}_DEGs",
        rankby_abs=False
    )


    # Get results for this cluster
    deg_df = sc.get.rank_genes_groups_df(
        adata, 
        group=[leiden_cluster], 
        key=F"leiden_{leiden_cluster}_DEGs"
    )

    # Filter DEGs
    deg_filtered = deg_df[
        (deg_df.pvals_adj < 0.05) &
        #(np.abs(deg_df.logfoldchanges) > 1) &
        (deg_df.logfoldchanges > 1) &
        (deg_df['pct_nz_group'] > 0.25) &  # only test genes that are detected in a minimum fraction  IN cluster
        (deg_df['pct_nz_reference'] < 0.20)  # only test genes NOT expressed outside
    ].copy()
        
    # Mapo gene
    gene_symbol_map = adata.var["gene_symbol"].to_dict()
    deg_filtered["gene_symbol"] = deg_filtered["names"].map(gene_symbol_map)

    print(f"\nTotal DEGs across all clusters: {len(deg_filtered)}")
    display(deg_filtered.head(20))
    deg_genes = deg_filtered.gene_symbol.to_list()
    print(deg_genes)



    import gseapy as gp

    # Get gene list (use gene symbols)
    if n_genes_gsea is not None:
        deg_genes = deg_genes[:n_genes_gsea]
    print(f"Running enrichment on {len(deg_genes)} genes...")

    # Run Enrichr
    enr = gp.enrichr(
        gene_list=deg_genes,
        gene_sets=['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'KEGG_2021_Human', 'MSigDB_Hallmark_2020'],
        organism='human',
        outdir=None,  # Don't save files
    )

    # Display top results
    print("\n=== Top Enriched Pathways ===")
    enriched_paths_df = enr.results[enr.results["Adjusted P-value"] <= 0.05]
    display(enriched_paths_df)

    return deg_filtered, deg_genes, enriched_paths_df



def calculate_and_plot_markers(adata, makers_dict, ncol=3):
    
    print("Assure that adata.var.index has same gene name of makers...")
    for cell_type, genes in tqdm(makers_dict.items()):
        # Only use genes that exist in your data
        #genes_available = [g for g in genes if g in query.var.gene_symbol.to_list()]
        
        if len(genes) > 0:
            #print(f"  {cell_type}: {len(genes)} markers")
            sc.tl.score_genes(adata, genes, score_name=f'{cell_type}_score', use_raw=False, layer="log1p_norm") # normalised scores (not scaled)
        else:
            print(f"  ⚠ {cell_type}: No markers found!")

    # plot umaps
    score_cols = [f'{ct}_score' for ct in makers_dict.keys() if f'{ct}_score' in adata.obs]
    sc.pl.umap(adata, color=score_cols, cmap='Reds', ncols=ncol)
    sc.pl.embedding(adata, basis="spatial", color=score_cols, ncols=ncol, size=40, cmap="Reds")

def wilcoxon_between_two_clusters(
    adata, 
    leiden_col="leiden_1", 
    cluster_A="0", 
    cluster_B="1",
    n_genes_gsea=None,
    pval_cutoff=0.05,
    logfc_cutoff=1.0,
    pct_cutoff=0.25
):
    """
    Compare DEGs between two specific leiden clusters using Wilcoxon test.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    leiden_col : str
        Column name in adata.obs containing cluster assignments
    cluster_A : str
        First cluster ID (will be tested as "group")
    cluster_B : str
        Second cluster ID (will be tested as "reference")
    n_genes_gsea : int, optional
        Number of top genes to use for enrichment (None = use all)
    pval_cutoff : float
        Adjusted p-value threshold (default: 0.05)
    logfc_cutoff : float
        Log fold change threshold (default: 1.0)
    pct_cutoff : float
        Minimum fraction of cells expressing gene (default: 0.25)
    
    Returns:
    --------
    dict with DEG results and enrichment
    """
    
    print(f"\n{'='*60}")
    print(f"Comparing Cluster {cluster_A} vs Cluster {cluster_B}")
    print(f"{'='*60}\n")
    
    # Run rank_genes_groups comparing cluster_A vs cluster_B
    sc.tl.rank_genes_groups(
        adata, 
        groupby=leiden_col,
        method='wilcoxon', 
        corr_method="benjamini-hochberg",
        groups=[cluster_A],      # Test this cluster
        reference=cluster_B,     # Against this cluster
        use_raw=False, 
        layer="log1p_norm",
        pts=True,
        key_added=f"DEG_{cluster_A}_vs_{cluster_B}",
        rankby_abs=False
    )

    # Get results
    deg_df = sc.get.rank_genes_groups_df(
        adata, 
        group=cluster_A,
        key=f"DEG_{cluster_A}_vs_{cluster_B}"
    )

    # Map gene symbols
    if "gene_symbol" in adata.var.columns:
        gene_symbol_map = adata.var["gene_symbol"].to_dict()
        deg_df["gene_symbol"] = deg_df["names"].map(gene_symbol_map)
    else:
        deg_df["gene_symbol"] = deg_df["names"]

    print(f"Total genes tested: {len(deg_df)}")
    print(f"Columns available: {deg_df.columns.tolist()}")

    # Filter for UPREGULATED genes in cluster_A vs cluster_B
    # Note: When comparing two clusters, we only have pct_nz_group, not pct_nz_reference
    deg_up_A = deg_df[
        (deg_df.pvals_adj < pval_cutoff) &
        (deg_df.logfoldchanges > logfc_cutoff) &
        (deg_df['pct_nz_group'] > pct_cutoff)  # Only filter on group percentage
    ].copy().sort_values('logfoldchanges', ascending=False)
    
    # Filter for DOWNREGULATED genes (higher in cluster_B)
    deg_up_B = deg_df[
        (deg_df.pvals_adj < pval_cutoff) &
        (deg_df.logfoldchanges < -logfc_cutoff) &
        (deg_df['pct_nz_group'] < (1 - pct_cutoff))  # Low expression in cluster_A
    ].copy().sort_values('logfoldchanges', ascending=True)

    print(f"\nGenes UP in cluster {cluster_A}: {len(deg_up_A)}")
    print(f"Genes UP in cluster {cluster_B}: {len(deg_up_B)}")
    
    if len(deg_up_A) > 0:
        print(f"\nTop genes UP in cluster {cluster_A}:")
        print("="*60)
        display(deg_up_A.head(20))
    
    if len(deg_up_B) > 0:
        print(f"\nTop genes UP in cluster {cluster_B}:")
        print("="*60)
        display(deg_up_B.head(20))

    # Prepare gene lists
    deg_genes_up = deg_up_A.gene_symbol.dropna().to_list()
    deg_genes_down = deg_up_B.gene_symbol.dropna().to_list()
    
    print(f"\nCluster {cluster_A} signature genes ({len(deg_genes_up)}):")
    print(deg_genes_up[:50])
    print(f"\nCluster {cluster_B} signature genes ({len(deg_genes_down)}):")
    print(deg_genes_down[:50])

    # Run enrichment analysis
    import gseapy as gp
    
    enrichment_results = {}
    
    # Enrichment for cluster_A genes
    if len(deg_genes_up) > 0:
        genes_for_enrichment = deg_genes_up[:n_genes_gsea] if n_genes_gsea else deg_genes_up
        print(f"\n{'='*60}")
        print(f"Running enrichment for Cluster {cluster_A} ({len(genes_for_enrichment)} genes)...")
        print(f"{'='*60}")
        
        try:
            enr_up = gp.enrichr(
                gene_list=genes_for_enrichment,
                gene_sets=[
                    'GO_Biological_Process_2023', 
                    'GO_Cellular_Component_2023', 
                    'KEGG_2021_Human', 
                    'MSigDB_Hallmark_2020'
                ],
                organism='human',
                outdir=None,
            )
            enriched_up = enr_up.results[enr_up.results["Adjusted P-value"] <= 0.05]
            if len(enriched_up) > 0:
                print(f"\nTop pathways enriched in Cluster {cluster_A}:")
                display(enriched_up.head(10))
            else:
                print(f"No significant pathways found for Cluster {cluster_A}")
            enrichment_results[f'cluster_{cluster_A}'] = enriched_up
        except Exception as e:
            print(f"Enrichment failed for cluster {cluster_A}: {e}")
            enrichment_results[f'cluster_{cluster_A}'] = None
    
    # Enrichment for cluster_B genes
    if len(deg_genes_down) > 0:
        genes_for_enrichment = deg_genes_down[:n_genes_gsea] if n_genes_gsea else deg_genes_down
        print(f"\n{'='*60}")
        print(f"Running enrichment for Cluster {cluster_B} ({len(genes_for_enrichment)} genes)...")
        print(f"{'='*60}")
        
        try:
            enr_down = gp.enrichr(
                gene_list=genes_for_enrichment,
                gene_sets=[
                    'GO_Biological_Process_2023', 
                    'GO_Cellular_Component_2023', 
                    'KEGG_2021_Human', 
                    'MSigDB_Hallmark_2020'
                ],
                organism='human',
                outdir=None,
            )
            enriched_down = enr_down.results[enr_down.results["Adjusted P-value"] <= 0.05]
            if len(enriched_down) > 0:
                print(f"\nTop pathways enriched in Cluster {cluster_B}:")
                display(enriched_down.head(10))
            else:
                print(f"No significant pathways found for Cluster {cluster_B}")
            enrichment_results[f'cluster_{cluster_B}'] = enriched_down
        except Exception as e:
            print(f"Enrichment failed for cluster {cluster_B}: {e}")
            enrichment_results[f'cluster_{cluster_B}'] = None

    # Return results
    return {
        'deg_up_in_A': deg_up_A,
        'deg_up_in_B': deg_up_B,
        'genes_up_in_A': deg_genes_up,
        'genes_up_in_B': deg_genes_down,
        'enrichment': enrichment_results,
        'all_degs': deg_df
    }



def plot_spatial_umap_per_type(adata, ANNOTATION_LEVEL):

    subclasses = adata.obs[ANNOTATION_LEVEL].value_counts().index.to_list()
    n_subclasses = len(subclasses)

    # Calculate rows (ceiling division)
    n_rows = int(np.ceil(n_subclasses / 2))
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()  # Flatten to 1D for easy indexing

    for i, s in enumerate(subclasses):
        # Create highlight column
        adata.obs["temp_highlight"] = adata.obs[ANNOTATION_LEVEL].apply(
            lambda x: s if x == s else 'Other'
        )
        
        # Spatial plot (left side: columns 0, 2, 4, ...)
        sc.pl.embedding(
            adata, 
            basis="spatial", 
            color="temp_highlight", 
            palette={'Other': 'lightgray', s: 'red'}, 
            na_color='#CCCCCC',  # Different shade to avoid conflict
            ax=axes[i*2], 
            show=False, 
            legend_loc=None, 
            size=10,
            title=f"{s} (spatial)"
        )
        
        # UMAP plot (right side: columns 1, 3, 5, ...)
        sc.pl.embedding(
            adata, 
            basis="X_umap", 
            color="temp_highlight", 
            palette={'Other': 'lightgray', s: 'red'}, 
            na_color='#CCCCCC',  # Different shade to avoid conflict
            ax=axes[i*2 + 1], 
            show=False, 
            legend_loc=None,
            title=f"{s} (UMAP)"
        )

    # Hide unused subplots
    for j in range(i*2 + 2, len(axes)):
        axes[j].axis('off')

    # Clean up temp column from BOTH objects
    adata.obs.drop('temp_highlight', axis=1, inplace=True)

    plt.tight_layout()
    plt.show()



def scrub(adata):
    import scrublet as scr

    counts = adata.layers['counts'].copy()

    scrub = scr.Scrublet(
        counts,
        #expected_doublet_rate=predicted_total_doublet_rate_by_oligo,  # Use prior estimate 
        sim_doublet_ratio=2
    )

    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2,
        min_cells=3,
        min_gene_variability_pctl=85,
        n_prin_comps=30
    )

    scrub.plot_histogram()

    # store in adata
    adata.obs['scrublet_score'] = doublet_scores
    adata.obs['is_scrublet_doublet'] = predicted_doublets

    # calculate
    total_doublet_rate_by_scrublet = predicted_doublets.sum() / len(predicted_doublets)
    print(total_doublet_rate_by_scrublet)

    # Plot overlap
    # sc.pl.umap(adata, color='is_oligo_doublet', show=False, 
    #            title=f'Oligo doublets ({predicted_total_doublet_rate_by_oligo*100:.1f}%)')
    sc.pl.umap(adata, color='is_scrublet_doublet', show=False,
            title=f'Scrublet doublets ({total_doublet_rate_by_scrublet*100:.1f}%)')


def plot_embedding_grid(adata, bases, colors, highlight_groups=None, size=10, figsize=None):
    
    import matplotlib.pyplot as plt
    
    n_rows = len(bases)
    n_cols = len(colors)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6*n_cols, 5*n_rows)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single row/col case
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Plot each combination
    for i, basis in enumerate(bases):
        for j, color in enumerate(colors):
            
            # Get groups to highlight (if any)
            groups = highlight_groups.get(color, None) if highlight_groups else None
            
            # Plot
            sc.pl.embedding(
                adata,
                basis=basis,
                color=color,
                groups=groups,
                size=size,
                ax=axs[i, j],
                show=False,
                legend_loc="lower right"
            )
            
            # Title
            axs[i, j].set_title(f"{basis} - {color}", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
