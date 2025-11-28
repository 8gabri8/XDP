

import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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


def plot_cell_type(adata, ANNOTATION_LEVEL):

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
            ax=axes[i*2 + 1], 
            show=False, 
            legend_loc=None,
            title=f"{s} (UMAP)"
        )

    # Hide unused subplots
    for j in range(i*2 + 2, len(axes)):
        axes[j].axis('off')

    # Clean up temp column
    adata.obs.drop('temp_highlight', axis=1, inplace=True)

    plt.tight_layout()
    plt.show()


def plot_cell_type_distribution(adata, annotation_levels=["Class_name"]):

    for ann in annotation_levels:
        vc = adata.obs[ann].value_counts().sort_values(ascending=False)
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























