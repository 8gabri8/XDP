import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import subprocess

from scipy import sparse
from anndata import AnnData
from tqdm import tqdm

import rpy2.robjects as ro
r = ro.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import decoupler as dc
import pertpy as pt


def convert_df_to_fig(df, title):
    import matplotlib.pyplot as plt
    
    df = df.round(5)
    df = df.reset_index()
    
    fig, ax = plt.subplots(figsize=(10, len(df)*0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.title(title, fontsize=12, pad=20)
    plt.tight_layout()
    
    return fig  # Returns matplotlib figure

def check_corr_cov_in_design(adata, DEG_FORMULA, corr_thr=0.7, split=" + "):
    
    # Get all variables from formula
    vars_in_formula = DEG_FORMULA[2:].split(split)
    print(DEG_FORMULA)
    print(vars_in_formula)

    # Extract data
    df = adata.obs[vars_in_formula].copy()

    # Encode categorical as numbers
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = pd.Categorical(df[col]).codes

    # Calculate correlation
    corr = df.corr()

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True)
    plt.title('Variable Correlations')
    plt.tight_layout()
    plt.show()

    # Find high correlations
    print(f"\n High correlations (|r| > {corr_thr}):")
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if abs(corr.iloc[i, j]) > corr_thr:
                print(f"{corr.index[i]} <-> {corr.columns[j]}: r = {corr.iloc[i, j]:.3f}")

def pseudobulk(adata, SAMPLE_VARIABLE, COV_FOR_PSEUDOBULK, GROUP_DEG_COL, COVARIATES_FOR_DEG, layer="counts", INTERESTING_COV=[], CONTRAST_VARIABLE=None, 
               MIN_CELLS_PER_PSUDOCELL=10,MIN_COUNTS_PER_PSEUDOCELL=1000, filter_genes=False,
               MIN_COUNTS=10,LARGE_N=10, MIN_TOTAL_COUNTS=15, MIN_PROP_BY_EXPR=0.5,MIN_PROP_BY_PROP=0.1,MIN_SMPLS=2, 
               ):
    
    # Pusdobulk 
    # comnination (donor_id x Group_name x zone)
    print("\nPsudobulking")
    adata_pb_all = dc.pp.pseudobulk(
        adata=adata,
        sample_col=SAMPLE_VARIABLE, # Creates ONE pseudobulk per unique value in this column
        groups_col=[*COV_FOR_PSEUDOBULK], # Would create separate pseudobulks for each combination (together with sample_col)
        layer=layer, # use .X with raw counts
        skip_checks=True,
        mode="sum",
    )

    # Col with all rpeusdbulk name concatemated
    adata_pb_all.obs["pseudobulk_group"] = (
        adata_pb_all.obs[[SAMPLE_VARIABLE] + COV_FOR_PSEUDOBULK]
        .astype(str)          # make sure all values are strings
        .agg("-".join, axis=1)  # join cell-wise
    )
    adata_pb_all.obs[GROUP_DEG_COL] = (
        adata_pb_all.obs[COV_FOR_PSEUDOBULK]
        .astype(str)          # make sure all values are strings
        .agg("-".join, axis=1)  # join cell-wise
    )

    # Check row counts in .X
    print(adata_pb_all.X)

    ##################à

    # Add covarite aggrgeated values
    print("\nAdding metadata")
    agg_dict = {}

    for cov in [*COVARIATES_FOR_DEG, *INTERESTING_COV]:
        if pd.api.types.is_numeric_dtype(adata.obs[cov]):
            agg_dict[cov] = "mean"
        else:
            agg_dict[cov] = "first"
    print(agg_dict)

    celltype_zone_metadata = adata.obs.groupby([SAMPLE_VARIABLE, *COV_FOR_PSEUDOBULK]).agg(
        agg_dict
    )

    # Reset index to turn groupby keys into columns 
    celltype_zone_metadata = celltype_zone_metadata.reset_index()
    #print(celltype_zone_metadata)

    # Drop old covariate columns to avoid duplicates, then merge
    adata_pb_all.obs = (
        adata_pb_all.obs
        .drop(columns=list(agg_dict.keys()), errors='ignore')  # Drop only the aggregated covariates
        .merge(
            celltype_zone_metadata,
            on=[SAMPLE_VARIABLE, *COV_FOR_PSEUDOBULK],  # Join on pseudobulk keys
            how='left'
        )
    )
    adata_pb_all.obs.set_index("pseudobulk_group", inplace=True)

    print(adata_pb_all.obs.head(3))
    print(adata_pb_all)

    ##################à

    # QC filters

    # Filer bad pseudocells
    dc.pl.filter_samples(
        adata=adata_pb_all,
        groupby=None,
        min_cells=MIN_CELLS_PER_PSUDOCELL,
        min_counts=MIN_COUNTS_PER_PSEUDOCELL,
        figsize=(5, 5))
    dc.pp.filter_samples(adata_pb_all, min_cells=MIN_CELLS_PER_PSUDOCELL, min_counts=MIN_COUNTS_PER_PSEUDOCELL)

    # Filter genes
    if filter_genes:
        dc.pp.filter_by_expr(adata=adata_pb_all, group=GROUP_DEG_COL,min_count=MIN_COUNTS, large_n=LARGE_N, min_total_count=MIN_TOTAL_COUNTS, min_prop=MIN_PROP_BY_EXPR)
        dc.pp.filter_by_prop(adata=adata_pb_all,min_prop=MIN_PROP_BY_PROP,min_smpls=MIN_SMPLS)

    print("\nAfter Filterign:")
    print(adata_pb_all)

    ########################


    # Normalise
    adata_pb_all.layers["counts"] = adata_pb_all.X.copy()
    import scipy.sparse as sp
    adata_pb_all.layers["scaled"] = np.zeros(adata_pb_all.shape) #sp.csr_matrix(adata_pb_all.shape)

    for group in adata_pb_all.obs[GROUP_DEG_COL].unique():
        print("Normalising: ", group)

        mask = adata_pb_all.obs[GROUP_DEG_COL] == group
        adata_pb_tmp = adata_pb_all[mask].copy()

        sc.pp.normalize_total(adata_pb_tmp, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata_pb_tmp)
        sc.pp.scale(adata_pb_tmp, max_value=10) #z-score normalization

        # Write back to main object
        adata_pb_all.layers["scaled"][mask] = adata_pb_tmp.X

    return adata_pb_all

def DEG_deseq2_edgeR(
    adata_pb_all,             # AnnData object containing pseudobulk data
    psuedobulk_group_for_deg, # current cell type to process
    psuedobulk_group_for_deg_col, # relative col name
    design_formula,           # Design formula for test
    save_folder,              # Folder where to save results
    SAMPLE_VARIABLE,          # column in adata.obs identifying samples/donors
    CONTRAST_VARIABLE,        # column in adata.obs defining groups for DEG (e.g., disease/control)
    MIN_CELLS_PER_PSUDOCELL,  # minimum cells per pseudobulk sample
    MIN_COUNTS_PER_PSEUDOCELL,# minimum counts per pseudobulk sample
    MIN_PSEUDOCELL_PER_GROUP, # minimum pseudobulks per group to run DEG
    MIN_COUNTS,               # min count filter for genes
    LARGE_N,                  # min samples expressing gene
    MIN_TOTAL_COUNTS,         # total counts across samples
    MIN_PROP_BY_EXPR,         # min fraction of samples a gene is expressed in
    MIN_PROP_BY_PROP,         # min fraction of cells expressing gene
    MIN_SMPLS,                # min number of samples expressing gene
    CONTRAST_BASELINE,        # baseline level for DEG contrast
    CONTRAST_STIM,            # stimulated/condition group for DEG contrast
    ALPHA_MULTIPLE_TEST,      # adjusted p-value threshold
    LOGFC_THR,                # log fold change threshold
    PVAL_THR,                 # p-value threshold for volcano plots
    N_TOP_GENES_TO_NAME,      # how many top genes to label in volcano
    GENE_COL_NAME="gene",     # column name for gene in DEG results
    LOGFC_COL_NAME="log_fc",  # column name for log fold change
    ADJ_P_VAL_COL_NAME="adj_p_value", # column name for adjusted p-value
    calculate_umap=False, 
    split= " + "
):
    
    # Prepare for capturing figures
    figures = []  # Store all figures here
    # Save orinal show
    original_show = plt.show
    # Create dummy show that does nothing
    def dummy_show(*args, **kwargs):
        pass  # Do nothing - don't display or clear
    # Replace show
    plt.show = dummy_show

    print(f"""
            #########################
            ### Processing psudobulk group: {psuedobulk_group_for_deg}
            #########################
            """)

    # Subset to current cimbination
        # ATTENTION: index must contain group name
    adata_pb_tmp = adata_pb_all[adata_pb_all.obs[psuedobulk_group_for_deg_col] == psuedobulk_group_for_deg].copy()


    #############################

    # Chnage design matrix to remove cov with None
    vars_in_formula = design_formula[2:].split(split)
    for var in vars_in_formula:
        if (adata_pb_all.obs[var].isna().sum() > 0):
            design_formula = design_formula.replace(f" + {var}", "")
            print(f"ATTENTION: {var} removed")
    print(f"\nCorrected dsign Matrix: {design_formula}")

     #############################

    # Check covaritne correlation
    check_corr_cov_in_design(adata_pb_all, design_formula, corr_thr=0.7, split=split)
    figures.append(plt.gcf())

    #############################

    # Umap this cell type

    if calculate_umap:
        ...
        # TODO

    #############################

    # Filter sample/pesudocells

    # Plot psuedocells
    dc.pl.filter_samples(
        adata=adata_pb_tmp,
        groupby=[CONTRAST_VARIABLE],
        min_cells=MIN_CELLS_PER_PSUDOCELL,
        min_counts=MIN_COUNTS_PER_PSEUDOCELL,
        figsize=(5, 5),
    )
    figures.append(plt.gcf())

    # Filer bad pseudocells
    dc.pp.filter_samples(adata_pb_tmp, min_cells=MIN_CELLS_PER_PSUDOCELL, min_counts=MIN_COUNTS_PER_PSEUDOCELL)

    # Plot how many remainin
    # dc.pl.obsbar(adata=adata_pb_tmp, y=CT_FOR_DEG_VARIABLE, hue=CONTRAST_VARIABLE, figsize=(5, 2))
    # figures.append(plt.gcf())

    # Stop if we dont have enoght sample per group
    group_counts = adata_pb_tmp.obs[CONTRAST_VARIABLE].value_counts()
    print(f"Samples per group: {dict(group_counts)}")

    if any(group_counts < MIN_PSEUDOCELL_PER_GROUP):
        print(f"   Insufficient samples: {dict(group_counts)}")
        print(f"   Need ≥{MIN_PSEUDOCELL_PER_GROUP} per group. Skipping this combination.")
        return  # Skip this cell_type × zone

    #############################

    # Variance Exploration

    # Store raw counts in layers
    adata_pb_tmp.layers["counts"] = adata_pb_tmp.X.copy()
    # Normalize, scale and compute pca
    sc.pp.normalize_total(adata_pb_tmp, target_sum=1e4)
    sc.pp.log1p(adata_pb_tmp)
    sc.pp.scale(adata_pb_tmp, max_value=10)
    sc.tl.pca(adata_pb_tmp)
    # Return raw counts to X
    dc.pp.swap_layer(adata=adata_pb_tmp, key="counts", inplace=True)

    # one-way ANOVA --> Which PCA components are most strongly associated with differences between states
    # High F-statistic / low p-value --> That PCA axis strongly separates healthy vs diseased.
    dc.tl.rankby_obsm(adata_pb_tmp, key="X_pca", 
                        obs_keys=[CONTRAST_VARIABLE]) # Only categoridal covariates
    sc.pl.pca_variance_ratio(adata_pb_tmp)
    figures.append(plt.gcf())
    dc.pl.obsm(adata=adata_pb_tmp, nvar=5, titles=["PC scores", "Adjusted p-values"], figsize=(5, 5))
    figures.append(plt.gcf())

    # PCA plot colored by state
    sc.pl.pca(
        adata_pb_tmp,
        color=[SAMPLE_VARIABLE, CONTRAST_VARIABLE],
        ncols=3,
        size=300,
        frameon=True,
        components=["1,2"]
    )
    figures.append(plt.gcf())

    # Gens drivin PC1
    loadings = adata_pb_tmp.varm["PCs"]
    pc1 = loadings[:, 0]
    genes = adata_pb_tmp.var_names
    df_pc1 = pd.DataFrame({
        "gene": genes,
        "loading": pc1
    }).sort_values("loading", ascending=False)
    #display(df_pc1)
    figures.append(convert_df_to_fig(pd.concat([df_pc1[0:10], df_pc1[-10:]]), "PC1 gene loadings"))

    #############################

    # Filter noisy/low wxpressed genes (cell type level)

    # Plot (not filter yet)
        # genes in upper-rught quadrant are kept
    dc.pl.filter_by_expr(
        # TEST: A gene is kept if it 
        #   - has at least "min_count" counts in at least "large_n" samples AND
        #   - its total count across all samples is ≥ "min_total_count" AND
        #   - is expressed in at least "min_prop" fraction of samples
        adata=adata_pb_tmp,
        group=CONTRAST_VARIABLE, # Column in obs defining biological groups (e.g. disease / condition)

        min_count=MIN_COUNTS, 
        large_n=LARGE_N, 
        min_total_count=MIN_TOTAL_COUNTS, 

        min_prop=MIN_PROP_BY_EXPR, # gene must be expressed in at least this fraction of samples
    )
    figures.append(plt.gcf())

    # Plot (not filter yet)
        # genes on the right are kept
    dc.pl.filter_by_prop(
        # TEST: A gene is kept if it
        #   - is detected in ≥ "min_prop" of cells
        #   - in at least "min_smpls" samples.
        adata=adata_pb_tmp,
        min_prop=MIN_PROP_BY_PROP,
        min_smpls=MIN_SMPLS,
    )
    figures.append(plt.gcf())

    # Apply filters
    dc.pp.filter_by_expr(adata=adata_pb_tmp, group=CONTRAST_VARIABLE,min_count=MIN_COUNTS, large_n=LARGE_N, min_total_count=MIN_TOTAL_COUNTS, min_prop=MIN_PROP_BY_EXPR)
    dc.pp.filter_by_prop(adata=adata_pb_tmp,min_prop=MIN_PROP_BY_PROP,min_smpls=MIN_SMPLS)


    #############################

    # # Check Multi-collinaarty covariates

    # # Extract covariate data
    # df = adata_pb_tmp.obs[COVARIATES_FOR_DEG].copy()
    
    # # Calculate correlation matrix (Pearson by default)
    # corr_matrix = df.corr()
    
    # # Create plot
    # fig, ax = plt.subplots(figsize=(8, 6))
    
    # # Heatmap with annotations
    # sns.heatmap(
    #     corr_matrix,
    #     annot=True,           # Show correlation values
    #     fmt='.2f',            # 2 decimal places
    #     cmap='coolwarm',      # Red=positive, Blue=negative
    #     vmin=-1, vmax=1,      # Correlation range
    #     center=0,             # Center colormap at 0
    #     square=True,          # Square cells
    #     cbar_kws={'label': 'Pearson Correlation'},
    #     ax=ax
    # )
    
    # plt.title('Covariate Correlations', fontsize=14, pad=20)
    # plt.tight_layout()

    # figures.append(fig)

    #############################

    # Create Design formual

    print(f"\nDesign formula: {design_formula}\n")

    #############################

    # Perform DEG: EdgeR

    edgr = pt.tl.EdgeR(
            adata_pb_tmp, 
            design_formula,
            layer=None)

    edgr.fit()

    contrast = edgr.contrast(
        column=CONTRAST_VARIABLE, 
        baseline=CONTRAST_BASELINE, 
        group_to_compare=CONTRAST_STIM)

    # Get desidered constartc
    res_df_edgeR = (
        edgr.test_contrasts(
            contrast,
            adjust_method='BH', 
            alpha=ALPHA_MULTIPLE_TEST
        )
        .rename(columns={"variable": GENE_COL_NAME, "log_fc":LOGFC_COL_NAME, "adj_p_value":ADJ_P_VAL_COL_NAME})
        .sort_values(LOGFC_COL_NAME)
    )

    # add greaidne gene col
    res_df_edgeR["method"] = "edgeR"
    res_df_edgeR["is_significant"] = (res_df_edgeR[ADJ_P_VAL_COL_NAME] <= ALPHA_MULTIPLE_TEST) & (np.abs(res_df_edgeR[LOGFC_COL_NAME])) >= LOGFC_THR

    # Save df significative
    res_df_edgeR_sig = res_df_edgeR[res_df_edgeR["is_significant"]]
    figures.append(convert_df_to_fig(pd.concat([res_df_edgeR_sig[0:10], res_df_edgeR_sig[-10:]])[[GENE_COL_NAME, LOGFC_COL_NAME, ADJ_P_VAL_COL_NAME]], "EdgeR Results Filtered"))

    #display(res_df_edgeR_sig)

    fig = edgr.plot_volcano(
                    res_df_edgeR, 
                    log2fc_thresh=LOGFC_THR,
                    pval_thresh=PVAL_THR,
                    log2fc_col=LOGFC_COL_NAME,
                    pvalue_col=ADJ_P_VAL_COL_NAME,
                    symbol_col=GENE_COL_NAME,
                    to_label=N_TOP_GENES_TO_NAME,
                    return_fig=True,
                    )
    figures.append(fig)


    #############################

    # Perform DEG: Deseq2

    pds2 = pt.tl.PyDESeq2(
            adata_pb_tmp, 
            design_formula,
            layer=None)

    pds2.fit()

    contrast = pds2.contrast(
        column=CONTRAST_VARIABLE, 
        baseline=CONTRAST_BASELINE, 
        group_to_compare=CONTRAST_STIM)

    # Get desidered constartc
    res_df_pds2 = (
        pds2.test_contrasts(
            contrast,
            alpha=ALPHA_MULTIPLE_TEST
        )
        .rename(columns={"variable": GENE_COL_NAME, "log_fc":LOGFC_COL_NAME, "adj_p_value":ADJ_P_VAL_COL_NAME})
        .sort_values(LOGFC_COL_NAME)
    )

    # add greaidne gene col
    res_df_pds2["method"] = "Deseq2"
    res_df_pds2["is_significant"] = (res_df_pds2[ADJ_P_VAL_COL_NAME] <= ALPHA_MULTIPLE_TEST) & (np.abs(res_df_pds2[LOGFC_COL_NAME])) >= LOGFC_THR

    # Save df significative
    res_df_pds2_sig = res_df_pds2[res_df_pds2["is_significant"]]
    figures.append(convert_df_to_fig(pd.concat([res_df_pds2_sig[0:10], res_df_pds2_sig[-10:]])[[GENE_COL_NAME, LOGFC_COL_NAME, ADJ_P_VAL_COL_NAME]], "Deseq2 Results Filtered"))

    #display(res_df_pds2_sig)

    fig = pds2.plot_volcano(
                    res_df_pds2, 
                    log2fc_thresh=LOGFC_THR,
                    pval_thresh=PVAL_THR,
                    log2fc_col=LOGFC_COL_NAME,
                    pvalue_col=ADJ_P_VAL_COL_NAME,
                    symbol_col=GENE_COL_NAME,
                    to_label=N_TOP_GENES_TO_NAME,
                    return_fig=True,
                    )
    figures.append(fig)

    
    #############################

    # Merge df and save

    output_file = f'{save_folder}/deg_df.csv'
    df_merged = pd.concat([res_df_edgeR, res_df_pds2])
    df_merged.to_csv(output_file)

    #############################

    # Save pdf

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(f'{save_folder}/images.pdf') as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    figures.clear()  # Clear the list



    # Report oringla show
    plt.show = original_show

    return df_merged

def run_nebula_parallel_script(
    path_qs: str,
    path_nebula_script: str,
    id_col: str = "donor_id",
    covs: list = None,
    offset_col: str = "nCount_RNA",
    n_folds: int = 8,
    n_cores: int = 16,
    save_tmp: bool = False,
    suffix: str = None,
):
    """
    Run NEBULA analysis in parallel using the bash orchestration script.
    
    Returns:
    --------
    str : Path to the combined results file
    """
    
    # Build command
    cmd = [
        "bash",
        path_nebula_script,
        "--path", path_qs,
        "--id-col", id_col,
        "--offset-col", offset_col,
        "--n-folds", str(n_folds),
        "--n-cores", str(n_cores),
        "--save-tmp", "1" if save_tmp else "0"
    ]
    
    # Add covariates if provided
    if covs:
        cmd.extend(["--covs", ",".join(covs)])
    
    # Add suffix if provided
    if suffix:
        cmd.extend(["--suffix", suffix])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the script
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        # Parse output to get final results path
        for line in result.stdout.split('\n'):
            if 'Final combined results:' in line:
                final_path = line.split('Final combined results:')[1].strip()
                return final_path
        
        return None
    
    except subprocess.CalledProcessError as e:
        print("❌ ERROR running NEBULA script!")
        print(f"\nCommand: {' '.join(cmd)}")
        print(f"\nSTDOUT:\n{e.stdout}")  # ADD THIS!
        print(f"\nSTDERR:\n{e.stderr}")  # ADD THIS!
        raise




















###########################################
###########################################à
############################################
################################################à

def run_mast_deg(
    adata,
    condition_col,
    contrast,  # [test_group, reference_group]
    replicate_col, # sample (if only one sample this will be consatnt)
    cell_type_col=None,
    cell_type=None,
    additional_covariates=None,  
    library_col=None,
    min_cells_per_group=5,
    min_freq_per_gene=0.1,
    min_cells=5,
    fdr_cutoff=0.05
):
    """
    Run MAST differential expression with random effects and covariates.
    """
    
    print(f"\n{'='*60}")
    print(f"Running MAST for: {cell_type}")
    print(f"Contrast: {contrast[0]} vs {contrast[1]} (reference)")
    print(f"{'='*60}\n")
    
    # ============================================
    # 1. PREPARE DATA
    # ============================================

    # Subset adata
    if cell_type_col:
        adata_subset = adata[
            (adata.obs[cell_type_col] == cell_type) &
            (adata.obs[condition_col].isin(contrast)) # select 2 LOD regions
        ].copy()

    # Optional: cell type name crrection
    #adata_subset.obs[cell_type] = [ct.replace(" ", "_") for ct in adata_subset.obs[cell_type]]
    
    # Remove unsued cat
    adata_subset.obs[condition_col] = adata_subset.obs[condition_col].cat.remove_unused_categories()
    
    # Print how many cell for each category in contrast
    counts = adata_subset.obs[condition_col].value_counts()
    print(f"Cell counts: {counts.to_dict()}")
    
    # Chekc if we have ogu cells in each at
    if len(counts) < 2 or any(counts[g] < min_cells_per_group for g in contrast):
        print(f"ERROR: Insufficient cells")
        return pd.DataFrame()

    # Remvoe genes not expressed in % cells
        #  Effective threshold = max(min_cells, min_freq * n_cells)
    sc.pp.filter_genes(adata_subset, min_cells=min_cells)
    min_cells_by_freq = int(min_freq_per_gene * adata_subset.n_obs)
    sc.pp.filter_genes(adata_subset, min_cells=min_cells_by_freq)

    # Calculte QC for cov (log1p_n_genes_by_counts) --> ATTENTION Need raw counts in .X
        # don't use it afterwards
    adata_subset.layers["log1p_norm_all_cells"] = adata_subset.layers["log1p_norm"].copy() # ATTENTION: DEG on log normadata ormailsed using all cell types
    adata_subset.X = adata_subset.layers["counts"].copy()
    adata_subset.var["mt"] = adata_subset.var.gene_symbol.str.startswith("MT-") # mitochondrial genes, ATTNETION: the names are in "gene_symbol"
    sc.pp.calculate_qc_metrics(adata_subset, qc_vars=['mt'], inplace=True)

    # ============================================
    # 0. CHECK CORR
    # ============================================

    # 2. Calculate correlation between MT and Total Counts per group
        # Include other covariates if:
            # they are technical
            # AND not perfectly collinear with CDR
    display(adata_subset.obs[additional_covariates + ["log1p_n_genes_by_counts"]].corr())
    
    # ============================================
    # 2. PREPARE DATA
    # ============================================

    ### COUNTS

    # First, transfer the expression matrix and obs --> # Transpose: genes x cells
    counts_df = pd.DataFrame(
        adata_subset.layers["log1p_norm_all_cells"].toarray(), # ATTNETION: log norm data from all cells
        index=adata_subset.obs_names,
        columns=adata_subset.var_names
    ).T # Transpose: genes x cells
    
    ### METADATA

    # Prepare metadata - start with condition and replicate
    metadata_cols = [condition_col, replicate_col, "log1p_n_genes_by_counts"] # add cov later

     # Add Optional lib cov (not scaled)
    if library_col is not None:
        if library_col in adata_subset.obs.columns:
            metadata_cols.append(library_col)
        else:
            print(f"WARNING: library_col '{library_col}' not found, skipping")
            library_col = None  # Set to None so R doesn't try to use it
    
    # Add additional covariates if specified
        # ATTENTION: z-score them to be mroe interpretable
    if additional_covariates:
        for cov in additional_covariates:
            if cov not in adata_subset.obs.columns:
                print(f"WARNING: Covariate '{cov}' not found in obs, skipping")
            else:
                metadata_cols.append(cov)
    
    metadata = adata_subset.obs[metadata_cols].copy()

    # THEN scale the additional covariates
    if additional_covariates:
        for cov in additional_covariates:
            if cov in metadata.columns:
                metadata[cov] = (metadata[cov] - metadata[cov].mean()) / metadata[cov].std()
    # Scale n_genes_by_counts (CDR)
    metadata["log1p_n_genes_by_counts"] = (metadata["log1p_n_genes_by_counts"] - metadata["log1p_n_genes_by_counts"].mean()) / metadata["log1p_n_genes_by_counts"].std()

    display(metadata.head(3))

    ### FORMULA

    # Build covariate formula
    covariate_formula = "log1p_n_genes_by_counts" # ATTENTION: alwayw have number of genes wiht postive count --< CELLULAR DETECTION RATE (CDR)
    if additional_covariates:
        valid_covs = [c for c in additional_covariates if c in metadata.columns]
        if valid_covs:
            covariate_formula += " + " + " + ".join(valid_covs)
    #print(f"Formula covariates: {covariate_formula}")

    # Build model formula
        # label(contrast) and replicate(sample) deifned in R code
    # Check number of unique replicates
    n_replicates = adata_subset.obs[replicate_col].nunique()

    if n_replicates > 1:
        random_effects = "(1 | replicate)"
        
        # Add library as random effect if provided
        if library_col is not None:
            n_libraries = adata_subset.obs[library_col].nunique()
            if n_libraries > 1:
                # Decide: crossed or nested
                # Use crossed if libraries are shared across samples
                #random_effects += " + (1 | library)"   
                # OR use nested if libraries are unique per sample:
                random_effects = "(1 | replicate/library)"
        
        model_formula = "~ label + " + covariate_formula + " + " + random_effects
        r_method = "glmer"
        fit_args = "list(nAGQ = 0)"
    else:
        # ATTENTION: No replicates - use library as fixed effect if available
        if library_col is not None:
            model_formula = "~ label + library + " + covariate_formula
        else:
            model_formula = "~ label + " + covariate_formula
        r_method = "glm"
        fit_args = "list()"

    print("Model Formula", model_formula)

    # ============================================
    # 3. MOVE DATA INTO R 
    # ============================================

    ro.globalenv['contrast_ref'] = contrast[1]
    ro.globalenv['contrast_test'] = contrast[0]
    ro.globalenv['condition_col'] = condition_col
    ro.globalenv['replicate_col'] = replicate_col
    ro.globalenv['fdr_cutoff'] = fdr_cutoff
    ro.globalenv['model_formula'] = model_formula
    ro.globalenv['r_method'] = r_method
    ro.globalenv['fit_args_str'] = fit_args
    ro.globalenv['library_col'] = library_col if library_col else ro.NULL
    ro.globalenv['has_library_col'] = library_col is not None

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['metadata_r'] = metadata
        ro.globalenv['counts_matrix'] = counts_df # alredy stranposed
        ro.globalenv['cell_metadata'] = metadata # SCALED VERSION
        ro.globalenv['gene_metadata'] = adata_subset.var
    
    # ============================================
    # 4. RUN MAST IN R 
    # ============================================

    # Load Libraries
    r('library(MAST)')
    r('library(SingleCellExperiment)')
    
    
    # Define fucntion to run MAST (as in https://www.sc-best-practices.org/conditions/differential_gene_expression.html#single-cell-specific)
    r('''
    
    # Define function to call in R
    find_de_MAST_RE <- function(){

        # ========================================
        # 1. CREATE MAST OBJECT
        # ========================================

        # First create SingleCellExperiment
        sce <- SingleCellExperiment(
            assays = list(X = as.matrix(counts_matrix)),
            colData = cell_metadata,
            rowData = gene_metadata
        )

        # Then convert to MAST SingleCellAssay
        sca <- SceToSingleCellAssay(sce, class = "SingleCellAssay")
      
        print(sca)
        
        # ========================================
        # 3. SET UP FACTORS WITH REFERENCE LEVELS
        # ========================================
        # Convert condition column to factor
        label <- factor(colData(sca)[[condition_col]])
        
        # Set reference level (baseline for comparison)
        # E.g., if contrast_ref = "intact", then "degenerated" will be compared to "intact"
        label <- relevel(label, contrast_ref)
        colData(sca)$label <- label
        
        # Convert replicate/donor to factor for random effects
        replicate <- factor(colData(sca)[[replicate_col]])
        colData(sca)$replicate <- replicate
      
        # Add library factor if provided
        if (!is.null(library_col) && has_library_col) {
            library_factor <- factor(colData(sca)[[library_col]])
            colData(sca)$library <- library_factor
        }
        
        # ========================================
        # 6. FIT HURDLE MODEL
        # ========================================
        # MAST uses a two-part hurdle model:
        #   - Discrete component: logistic regression for P(expression > 0)
        #   - Continuous component: linear regression for log(expression | expression > 0)
        zlmCond <- zlm(
            formula = as.formula(model_formula),
            sca = sca,
            method = r_method,             # Generalized linear mixed effects model
            ebayes = FALSE,                # Don't use empirical Bayes (for speed)
            strictConvergence = FALSE,     # Allow approximate convergence
            fitArgsD = eval(parse(text = fit_args_str))  # Parse the string      # Use Laplace approximation (faster)
        )
        
        # ========================================
        # 7. LIKELIHOOD RATIO TEST
        # ========================================
        # Test if the test group differs from reference group
        
        # Perform likelihood ratio test comparing:
        #   Full model (with group) vs Reduced model (without group)
        summaryCond <- summary(zlmCond, doLRT = paste0('label', contrast_test))
        
        # ========================================
        # 8. EXTRACT RESULTS
        # ========================================
        # Get data.table with results
        summaryDt <- summaryCond$datatable
        
        # Merge two components:
        # - 'H' (Hurdle): Combined p-value from discrete + continuous test
        # - 'logFC': Log fold change
        result <- merge(
            # P-values from hurdle test (tests BOTH detection AND expression level)
            summaryDt[contrast == paste0('label', contrast_test) & component == 'H', .(primerid, `Pr(>Chisq)`)],
            # Log fold changes
            summaryDt[contrast == paste0('label', contrast_test) & component == 'logFC', .(primerid, coef)],
            by = 'primerid'
        )
        
        # ========================================
        # 9. CONVERT AND FILTER
        # ========================================
        # MAST uses natural logarithm so we convert the coefficients to log2 base to be comparable to edgeR
        # Convert natural log (ln) to log2 for compatibility with other tools
        # log2(FC) = ln(FC) / ln(2)
        result[, coef := coef / log(2)]
        
        # Multiple testing correction using Benjamini-Hochberg FDR
        result[, FDR := p.adjust(`Pr(>Chisq)`, 'fdr')]
        
        # Filter: keep only genes with FDR < threshold
        result <- result[result$FDR < fdr_cutoff, drop = FALSE]
        
        # Remove any rows with NA values
        result <- stats::na.omit(as.data.frame(result))

        # print(paste("Rows after merge:", nrow(result)))
        # print(paste("Columns after merge:", ncol(result)))
        # print(result)
        
        return(result)
    }
    ''')

    # call funtion
    r('result <- find_de_MAST_RE()')
    
    # ============================================
    # 4. GET RESULTS AND PROCESS THEM
    # ============================================
    result = r['result']
    result = pd.DataFrame(result)

    # display(result)
    # print(f"Shape: {result.shape}")
    # print(f"Columns: {result.columns.tolist()}")

    result = result.T
    
    if len(result) == 0:
        print("No significant results")
        return pd.DataFrame()

    result.columns = ['gene', 'pval', 'log2FC', 'FDR']

    gene_map = adata_subset.var["gene_symbol"].to_dict()
    result["gene_symbol"] = result["gene"].map(gene_map)

    result["direction"] = result["log2FC"].apply(
        lambda x: f"UP in {contrast[0]}" if x > 0 else f"DOWN in {contrast[0]}"
    )

    result = result.sort_values('FDR')
    
    print(f"\nRESULTS: {len(result)} DEGs (FDR < {fdr_cutoff})")
    n_up = (result['direction'] == f'UP in {contrast[0]}').sum()
    n_down = (result['direction'] == f'DOWN in {contrast[0]}').sum()
    print(f"↑ {n_up} UP / ↓ {n_down} DOWN")

    result["cell_type"] = cell_type
    
    return result


def run_edger_analysis(
    adata_pb,
    contrast_variable,
    contrast_test,
    contrast_baseline,
    covariates=None,
    alpha=0.05,
    lfc_threshold=0.5
):
    """
    Run edgeR differential expression analysis on pseudobulk data.
    
    Parameters:
    -----------
    adata_pb : AnnData
        Pseudobulk AnnData object with raw counts in .X
    contrast_variable : str
        Column name in .obs for the main comparison (e.g., 'condition')
    contrast_test : str
        Test level (e.g., 'diseased')
    contrast_baseline : str
        Baseline/reference level (e.g., 'healthy')
    covariates : list, optional
        List of covariate column names to include in model
    alpha : float
        FDR threshold for significance (default: 0.05)
    lfc_threshold : float
        Log2 fold change threshold for filtering (default: 0.5)
        
    Returns:
    --------
    dict with keys:
        - 'results_df': DataFrame with all genes and statistics
        - 'sig_results': DataFrame with significant genes only
        - 'summary': Dict with summary statistics
    """
    
    try:
        # Check if edgeR is available
        r('suppressPackageStartupMessages(library(edgeR))')
    except Exception as e:
        raise ImportError(f"edgeR not installed in R: {e}")
    
    try:
        # ==========================================
        # 1. PREPARE DATA
        # ==========================================
        
        # Set reference level
        adata_pb.obs[contrast_variable] = adata_pb.obs[contrast_variable].astype('category')
        adata_pb.obs[contrast_variable] = adata_pb.obs[contrast_variable].cat.reorder_categories(
            [contrast_baseline] + [c for c in adata_pb.obs[contrast_variable].cat.categories 
                                   if c != contrast_baseline]
        )
        
        # Build design formula
        design_formula = f"~ {contrast_variable}"
        if covariates:
            for cov in covariates:
                if adata_pb.obs[cov].dtype == 'object':
                    adata_pb.obs[cov] = adata_pb.obs[cov].astype('category')
            design_formula += " + " + " + ".join(covariates)
        
        # Extract counts (genes × samples)
        counts = adata_pb.X
        if hasattr(counts, 'toarray'):
            counts = counts.toarray()
        
        counts_df = pd.DataFrame(
            counts.T,
            index=adata_pb.var_names,
            columns=adata_pb.obs_names
        )
        
        # Extract metadata
        metadata_cols = [contrast_variable] + (covariates if covariates else [])
        sample_metadata = adata_pb.obs[metadata_cols].copy()
        
        print(f"Design: {design_formula}")
        print(f"Counts: {counts_df.shape[0]} genes × {counts_df.shape[1]} samples")
        print(f"Groups: {sample_metadata[contrast_variable].value_counts().to_dict()}")
        
        # ==========================================
        # 2. TRANSFER DATA TO R
        # ==========================================
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_counts = ro.conversion.py2rpy(counts_df)
            r_metadata = ro.conversion.py2rpy(sample_metadata)
        
        ro.globalenv['counts_matrix'] = r_counts
        ro.globalenv['sample_metadata'] = r_metadata
        ro.globalenv['design_formula'] = design_formula
        ro.globalenv['contrast_var'] = contrast_variable
        ro.globalenv['contrast_test'] = contrast_test
        
        # ==========================================
        # 3. RUN edgeR IN R
        # ==========================================
        
        r('''
            suppressPackageStartupMessages(library(edgeR))
            
            # Create DGEList
            dge <- DGEList(counts = counts_matrix, samples = sample_metadata)
            
            # Design matrix
            design <- model.matrix(as.formula(design_formula), data = sample_metadata)
            
            # Filter genes
            keep <- filterByExpr(dge, design = design)
            dge <- dge[keep, , keep.lib.sizes = FALSE]
            
            cat("Genes after filtering:", sum(keep), "\n")
            
            # Normalize (TMM)
            dge <- calcNormFactors(dge, method = "TMM")
            
            # Estimate dispersion
            dge <- estimateDisp(dge, design, robust = TRUE)
            
            cat("Common dispersion:", dge$common.dispersion, "\n")
            
            # Fit quasi-likelihood GLM
            fit <- glmQLFit(dge, design, robust = TRUE)
            
            # Find coefficient
            contrast_coef <- paste0(contrast_var, contrast_test)
            if (!(contrast_coef %in% colnames(design))) {
                possible <- grep(contrast_test, colnames(design), value = TRUE)
                if (length(possible) > 0) {
                    contrast_coef <- possible[1]
                }
            }
            
            cat("Testing coefficient:", contrast_coef, "\n")
            
            # Perform QL F-test
            qlf <- glmQLFTest(fit, coef = contrast_coef)
            
            # Extract results
            results_table <- as.data.frame(topTags(qlf, n = Inf, sort.by = "none"))
            results_table$gene <- rownames(results_table)
        ''')
        
        # ==========================================
        # 4. GET RESULTS BACK TO PYTHON
        # ==========================================
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            results_df = ro.conversion.rpy2py(ro.globalenv['results_table'])
        
        # Rename columns to standard format
        results_df = results_df.rename(columns={
            'logFC': 'log2FoldChange',
            'logCPM': 'baseMean',
            'PValue': 'pvalue',
            'FDR': 'padj'
        })
        
        # ==========================================
        # 5. FILTER AND SUMMARIZE
        # ==========================================
        
        # Filter significant results
        sig = (results_df['padj'] <= alpha) & (np.abs(results_df['log2FoldChange']) >= lfc_threshold)
        sig_results = results_df[sig].sort_values(by='log2FoldChange', ascending=False)
        
        # Calculate summary statistics
        n_up = ((results_df['log2FoldChange'] > lfc_threshold) & (results_df['padj'] < alpha)).sum()
        n_down = ((results_df['log2FoldChange'] < -lfc_threshold) & (results_df['padj'] < alpha)).sum()
        
        summary = {
            'total_genes': len(results_df),
            'significant_genes': sig.sum(),
            'up_regulated': n_up,
            'down_regulated': n_down,
            'alpha': alpha,
            'lfc_threshold': lfc_threshold
        }
        
        # print(f"\n{'='*50}")
        # print(f"edgeR Summary")
        # print(f"{'='*50}")
        # print(f"Total genes tested: {summary['total_genes']}")
        # print(f"Significant DEGs: {summary['significant_genes']}")
        # print(f"  Up-regulated: {summary['up_regulated']}")
        # print(f"  Down-regulated: {summary['down_regulated']}")
        
        # ==========================================
        # 6. RETURN RESULTS
        # ==========================================
        
        return {
            'results_df': results_df,
            'sig_results': sig_results,
            'summary': summary
        }
        
    except Exception as e:
        print(f"❌ edgeR failed: {e}")
        import traceback
        traceback.print_exc()
        return None