import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import sparse
from anndata import AnnData
from tqdm import tqdm

import rpy2.robjects as ro
r = ro.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter



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