

import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from diptest import diptest


def plot_qc_metrics(adata):

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
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
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
    sns.scatterplot(
        x=adata.obs["log1p_total_counts"],
        y=adata.obs["log1p_n_genes_by_counts"],
        hue=adata.obs["pct_counts_mt"],
        ax=axes[2],
        s=30,        # point size
        linewidth=0, 
        palette="viridis"
    )
    axes[2].set_title("Counts vs GenesColored by % MT")

    # --- 4. % introns ---
        # GOOD

        # BAD
            # having 2 peaks

    sns.histplot(adata.obs["pct_intronic"], bins=100, kde=False, ax=axes[3])
    axes[3].set_title("Total percnet introns per cell")
    axes[3].set_ylabel("Number of cells")
    axes[3].set_xlabel("Percent intornic")

    # --- 5. n genes ---
        # GOOD

        # BAD
    x = adata.obs["n_genes_by_counts"]
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    sns.histplot(x, bins=bins, kde=False, ax=axes[4])
    axes[4].set_title("Total genes per cell")
    axes[4].set_ylabel("Number of cells")
    axes[4].set_xlabel("Total counts")
    axes[4].set_xscale("log")

    # --- 6. mapmycell probs ---
        # GOOD

        # BAD
    axes[5].hist(
        adata.obs['Class_bootstrapping_probability'].dropna(),
        bins=100, log=True
    )
    axes[5].set_title("Bootstrapping probability")

    #plt.tight_layout()
    plt.show()


def preprocess(adata, n_pcs_elbow=30, n_hvg=3000):

    # Normalize & Transform --> X amtrix changes
    print("Normalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()

    # HVG
    print("Findingn HCGs...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    sc.pl.highly_variable_genes(adata, show=True)
        # BAD: nif there is not clear seapration

    # scale
        # ATTENTION: now X stores nroalised counts --> not used for all stat test alter
    print("Scaling...")
    sc.pp.scale(adata, max_value=10) #z-score normalization

    # PCA
    print("Calculating PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=80) # ATTENTION: # Uses HVGs only (if calculted, even if the adat has all geens)
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs=80, show=True)

    # neighbours (for umap and leiden)
    print("Calculating neightbors...")
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

def show_top_pc_genes(adata, n_pcs=3, top_n=10):

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


def plot_cell_type_distribution(adata, annotation_levels=["Class_name"]):

    for ann in annotation_levels:
        vc = adata.obs[ann].value_counts(dropna=False).sort_values(ascending=False)
        vp = vc / vc.sum() * 100  # percentages

        fig, axes = plt.subplots(1, 3, figsize=(25, 6))

        # log scale
        vc.plot(kind="bar", logy=True, ax=axes[0])
        axes[0].set_title(f"{ann} counts (log y)")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Counts")

        # linear scale
        vc.plot(kind="bar", logy=False, ax=axes[1])
        axes[1].set_title(f"{ann} counts (linear y)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Counts")

        # percentages
        vp.plot(kind="bar", ax=axes[2])
        axes[2].set_title(f"{ann} percentages")
        axes[2].set_xlabel("")
        axes[2].set_ylabel("Percentage (%)")

        # rotate all x-labels
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

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




















