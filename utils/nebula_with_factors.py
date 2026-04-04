import scanpy as sc
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import rpy2.robjects as ro

from scipy.optimize import nnls
from joblib import Parallel, delayed

import DEG as DEG
import preprocessing

def calculate_pca(adata, calculate_or_project, saving_folder, variance_threshold=None, N_PCS_TO_USE=None, df_stats=None, df_loadings=None):
    
    # Common steps
    print("Normalising")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if calculate_or_project == "calculate":
        # Find highly variable genes on HEALTHY only
        print("HVG")
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        hvg = adata.var_names[adata.var['highly_variable']].tolist()  # Get as ordered list
        adata = adata[:, hvg].copy() # Subset with 01 Index
    elif calculate_or_project == "project":
        hvg = df_loadings.index.tolist()
        # Subset and REORDER to match loadings
        adata = adata[:, hvg].copy() # Subset wiht gene names
        assert all(adata.var_names == df_loadings.index), "Gene order mismatch!"

    # Ectarct dense matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # Scale using healthy statistics
    print("Scaling")
    if calculate_or_project == "calculate":
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X)

        mean_train = scaler.mean_
        var_train = scaler.var_

        df_stats = pd.DataFrame({
            "mean_train": mean_train,
            "var_train": var_train
        }, index=adata.var_names)
        df_stats.to_csv(f"{saving_folder}/healthy_stats.csv")
        
    elif calculate_or_project == "project":
        df_stats = df_stats.reindex(adata.var_names) # FORCE SAME ORDER
        mean_train = df_stats["mean_train"].values
        var_train = df_stats["var_train"].values

        X_scaled = (X - mean_train) / np.sqrt(var_train)


    # PCA 
    MAX_PCS_TO_COMPUTE = 50
    if calculate_or_project == "calculate":
        print("PCA on Healthy")
        pca = PCA(n_components=MAX_PCS_TO_COMPUTE, random_state=42, svd_solver="full")
        pcs = pca.fit_transform(X_scaled)

        if variance_threshold is not None:
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            N_PCS_TO_USE = int(np.searchsorted(cumvar, variance_threshold) + 1)
            N_PCS_TO_USE = min(N_PCS_TO_USE, MAX_PCS_TO_COMPUTE)  # ← clamp
            print(f"Auto-selected {N_PCS_TO_USE} PCs "
                f"({cumvar[N_PCS_TO_USE-1]*100:.1f}% variance explained)")
        elif N_PCS_TO_USE is None:
            # ELBOW LOGIC
            var_explained = pca.explained_variance_ratio_
            cumvar = np.cumsum(var_explained)

            # # Elbow = point of maximum curvature (second derivative)
            # deltas = np.diff(var_explained)        # first derivative
            # curvature = np.diff(deltas)            # second derivative
            # N_PCS_TO_USE = int(np.argmax(curvature) + 2)  # +2: argmax is 0-based on double-diff array

            # KNEE
            x = np.arange(len(var_explained))
            x_norm = x / x.max()
            y_norm = var_explained / var_explained.max()
            distances = np.abs(y_norm - (y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm))
            N_PCS_TO_USE = int(np.argmax(distances) + 1)

            print(f"Auto-selected {N_PCS_TO_USE} PCs via elbow "
                f"({cumvar[N_PCS_TO_USE-1]*100:.1f}% variance explained using {MAX_PCS_TO_COMPUTE})")

        # Save pcs
        df_healthy = pd.DataFrame(
            pcs[:, :N_PCS_TO_USE],
            columns=[f'PC_{i+1}' for i in range(N_PCS_TO_USE)]
        )
        df_healthy.insert(0, 'barcode', adata.obs_names.values)
        df_healthy.to_csv(f"{saving_folder}/healthy_pcs.csv", index=False)

        # Extract loadings (genes × PCs)
        loadings = pca.components_.T  # Shape: (n_genes, n_components)
        # save loadings
        df_loadings = pd.DataFrame(
            loadings[:, :N_PCS_TO_USE],
            index= adata.var_names,
            columns=[f'PC_{i+1}' for i in range(N_PCS_TO_USE)]
        )
        df_loadings.to_csv(f"{saving_folder}/gene_loadings.csv")

    elif calculate_or_project == "project":
        print("PCA on Diseased")
        loadings = df_loadings.values # Shape: (n_genes, n_components)
        pcs = X_scaled @ loadings  # apply ppject wiht laodings
        num_pcs = pcs.shape[1]
        # save pcs
        df_diseased = pd.DataFrame(
            pcs,
            columns=[f'PC_{i+1}' for i in range(num_pcs)]
        )
        df_diseased.insert(0, 'barcode', adata.obs_names.values)
        df_diseased.to_csv(f"{saving_folder}/diseased_pcs.csv", index=False)


    # # Plot PCs of a fingle sample
    # sample_id = adata.obs[SAMPLE_VARIABLE].unique()[0]
    # adata_s = adata[adata.obs[SAMPLE_VARIABLE] == sample_id].to_memory().copy()
    # adata_s.obs = adata_s.obs.drop(columns=[c for c in adata_s.obs.columns if "PC_" in c])
    # adata_s.obs = adata_s.obs.merge(
    #     df_healthy if calculate_or_project == "calculate" else df_diseased,
    #     left_index=True,
    #     right_on="barcode",
    #     how="left", 
    #     validate="one_to_one"
    # )
    # pcs = [f'PC_{i+1}' for i in range(N_PCS_TO_USE)]
    # if "spatial" in adata.obsm.keys():
    #     adata_s.obsm["spatial"] = adata_s.obs[["x", "y"]].values  # or .to_numpy()
    #     sc.pl.embedding(adata_s, basis="spatial", color=["ct_for_deg", "Group_name", *pcs])
    # del adata_s

    # SAVE PCS + DIAGBOSTICS
    if calculate_or_project == "calculate":

        # Elbow plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-')
        plt.axvline(x=N_PCS_TO_USE, color='red', linestyle='--', label=f'Selected: PC {N_PCS_TO_USE}')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('Scree Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{saving_folder}/elbow_plot.png", dpi=150)
        #plt.show()

        # Top geens for each PC
        def get_top_genes_per_pc(pca, gene_names, n_top=20, n_pcs=6):
            """
            Create a simple DataFrame showing top N genes for each PC
            
            Returns DataFrame with columns: PC_1, PC_1_loading, PC_2, PC_2_loading, etc.
            """
            loadings = pca.components_.T  # (n_genes, n_PCs)
            
            result = {}
            
            for i in range(min(n_pcs, loadings.shape[1])):
                pc_name = f'PC_{i+1}'
                
                # Get absolute loadings
                abs_loadings = np.abs(loadings[:, i])
                
                # Get top N indices
                top_indices = np.argsort(abs_loadings)[::-1][:n_top]
                
                # Get gene names and actual loadings (with sign)
                top_genes = gene_names[top_indices]
                top_loadings = loadings[top_indices, i]
                
                # Add to result dictionary
                result[pc_name] = top_genes
                result[f'{pc_name}_loading'] = top_loadings
            
            # Create DataFrame
            df = pd.DataFrame(result)
            
            return df

        N_TOP_GENES = 20
        df_top_genes = get_top_genes_per_pc(
            pca, 
            gene_names=adata.var_names.values,
            n_top=N_TOP_GENES,
            n_pcs=N_PCS_TO_USE
        )
        #display(df_top_genes)
        df_top_genes.to_csv(f"{saving_folder}/{N_TOP_GENES}_top_genes_per_pc.csv")

    print("Done.")
    
    return {
        'pcs': pcs,
        'df_stats': df_stats, 
        'df_loadings': df_loadings,
        'df_pcs': df_healthy if calculate_or_project == "calculate" else df_diseased
    }

def project_cnmf_geps(adata, healthy_or_diseased, saving_folder, df_spectra, reference_hvg, reference_gene_stds, normalize=True):
    """
    Project cells onto pre-trained cNMF GEPs using NNLS.
    Mirrors calculate_pca() interface: saves healthy_geps.csv or diseased_geps.csv.

    Parameters
    ----------
    adata               : AnnData with raw counts in .X
    healthy_or_diseased : "healthy" or "diseased"
    saving_folder       : where to save the output CSV
    df_spectra          : DataFrame (K x genes), loaded from *.spectra.k_K.dt_D.consensus.txt
    reference_hvg       : list of gene names (HVGs from cNMF training, defines gene order)
    reference_gene_stds : np.array of per-gene stds from healthy training data
    normalize           : L1-normalize usages per cell (sum to 1)
    """
    assert healthy_or_diseased in ("healthy", "diseased")

    # Subset and reorder to match spectra gene order
    adata = adata[:, reference_hvg].copy()
    assert list(adata.var_names) == reference_hvg, "Gene order mismatch with spectra"

    # get raw counts (convert to dense)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # Variance-normalize using healthy gene stds (transfer to diseased too)
    X_norm = X / reference_gene_stds[np.newaxis, :]

    # NNLS projection: solve min ||spectra.T @ u - x|| for each cell
    G = df_spectra.values.T  # Shape: (genes, K)

    def solve_one(cell_expr):
        return nnls(G, cell_expr)[0]

    usages = np.array(
        Parallel(n_jobs=-1, backend="loky")(
            delayed(solve_one)(X_norm[i]) for i in range(X_norm.shape[0])
        )
    )

    # L1-normalize per cell (to sum to 1 per cell)
    if normalize:
        row_sums = usages.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        usages = usages / row_sums

    K = usages.shape[1] #numne of GEPs
    # ATTNETION Drop last GEP: usages sum to 1 per cell → perfect collinearity with K covariates
    gep_names = [f"GEP_{i+1}" for i in range(K - 1)]

    df_out = pd.DataFrame(usages[:, :K-1], columns=gep_names)
    df_out.insert(0, "barcode", adata.obs_names.values)
    df_out.to_csv(f"{saving_folder}/{healthy_or_diseased}_geps.csv", index=False)

    print(f"Saved {healthy_or_diseased} GEP projections: {df_out.shape[0]} cells, {len(gep_names)} GEPs")
    return df_out

def filter_genes_for_nebula(adata_ct, sample_col, condition_col, min_cells_per_sample=10, min_counts=50):

    print("ATTENTION: .X must be raw!")

    # ── 1. Drop donors with too few cells ─────────────────────────────────────
    cells_per_sample = adata_ct.obs[sample_col].value_counts()
    keep_samples     = cells_per_sample[cells_per_sample >= min_cells_per_sample].index
    drop_samples     = cells_per_sample[cells_per_sample <  min_cells_per_sample].index

    if len(drop_samples):
        # Build a summary: donor → (n_cells, condition)
        donor_meta = (
            adata_ct.obs[[sample_col, condition_col]]
            .drop_duplicates(subset=sample_col)
            .set_index(sample_col)
        )
        print(f"Dropping {len(drop_samples)} donor(s) with < {min_cells_per_sample} cells:")
        for donor in sorted(drop_samples.tolist()):
            n      = cells_per_sample[donor]
            status = donor_meta.loc[donor, condition_col]
            print(f"  {donor:30s}  n_cells={n:4d}  condition={status}")

        adata_ct = adata_ct[adata_ct.obs[sample_col].isin(keep_samples)].copy()

    n_samples      = adata_ct.obs[sample_col].nunique()
    n_genes_before = adata_ct.n_vars

    # ── 2. Gene filtering ──────────────────────────────────────────────────────
    min_cells = n_samples * min_cells_per_sample
    sc.pp.filter_genes(adata_ct, min_cells=min_cells)
    sc.pp.filter_genes(adata_ct, min_counts=min_counts)

    n_genes_after = adata_ct.n_vars
    print(f"Retained donors : {n_samples}")
    print(f"Filtered genes  : {n_genes_before} -> {n_genes_after} "
          f"(min_cells={min_cells}, min_counts={min_counts})")

    return adata_ct

def run_nebula_with_factors(
    adata,  
    ADATA_PATH_QS, SAVE_HEALTHY_PCA_RESULT, PARALLEL_NEBULA_SCRIPT_PATH,
    CONTRAST_VARIABLE,CONTRAST_BASELINE, CONTRAST_STIM,
    SAMPLE_VARIABLE,LIBRARY_SIZE_COL,COVARIATES_FOR_DEG,
    N_PCS_TO_USE=None, variance_threshold=None, delete_tmp_files=True,TYPE_DEG=None,
    MIN_SAMPLES_PER_CONDITION=4,     MIN_NUMBER_GENES_PER_ANALYSIS = 3000, CNMF_SPECTRA_PATH=None
):
    ###################
    # Filter low quality genes
    ###################      

    print(np.array_equal(adata.X.data, adata.X.data.astype(int)) if hasattr(adata.X, 'data') else np.array_equal(adata.X, adata.X.astype(int)))
    adata = filter_genes_for_nebula(adata, SAMPLE_VARIABLE, CONTRAST_VARIABLE, min_cells_per_sample=10, min_counts=50)


    ###################
    # Check if we can run Nebula
    ###################  

    # chek nin number fo samples
    condition_sample_counts = (
        adata.obs[[SAMPLE_VARIABLE, CONTRAST_VARIABLE]]
        .drop_duplicates()
        .groupby(CONTRAST_VARIABLE)[SAMPLE_VARIABLE]
        .count()
        .reindex([CONTRAST_BASELINE, CONTRAST_STIM], fill_value=0)  # ← fix: force both conditions
            # otherwise if one cosntion has 0 samples, it will be missing from the count and we won't detect the problem
    )

    failed = condition_sample_counts[condition_sample_counts < MIN_SAMPLES_PER_CONDITION]
    n_samples = condition_sample_counts.count()
    if (len(failed)) or (n_samples == 0):
        print(f"\nERROR: Insufficient donors per condition (threshold={MIN_SAMPLES_PER_CONDITION}):")
        for cond, n in condition_sample_counts.items():
            flag = " ← BELOW THRESHOLD" if n < MIN_SAMPLES_PER_CONDITION else ""
            print(f"  {cond:20s}  n_donors={n}{flag}")
        print("Skipping this cell type — NEBULA random effects will be unreliable.\n")
        return None

    # chekc number remianing genes
    n_genes_after = adata.n_vars
    if n_genes_after <= MIN_NUMBER_GENES_PER_ANALYSIS:
        print(f"ERROR: Too few genes filtered out after gene filtering ")
        return None
    
    ###################
    # Calculate Factors for Healthy
    ###################

    print("Running Factor calculation on Healthy...")

    adata_healthy = adata[adata.obs[CONTRAST_VARIABLE] == CONTRAST_BASELINE].to_memory().copy()

    adata_healthy.X = adata_healthy.layers["counts"].copy()

    if (TYPE_DEG == "PCA_nebula") | (TYPE_DEG == "baseline_nebula"):
        results_healthy = calculate_pca(
            adata=adata_healthy,
            calculate_or_project="calculate",
            variance_threshold=variance_threshold,
            N_PCS_TO_USE=N_PCS_TO_USE,
            saving_folder=SAVE_HEALTHY_PCA_RESULT, 
        )
    elif (TYPE_DEG == "cNMF_nebula"):
        # Save h5ad to run cNF on AUGER
        if not os.path.exists(CNMF_SPECTRA_PATH):
            adata_healthy = adata[adata.obs[CONTRAST_VARIABLE] == CONTRAST_BASELINE].to_memory().copy()
            adata_healthy.X = adata_healthy.layers["counts"].astype(np.float64).copy()
            adata_healthy.write_h5ad(f"{SAVE_HEALTHY_PCA_RESULT}/healthy_cNMF.h5ad")
            print(f"Saved healthy h5ad. Run cNMF on remote server, then re-run.")
            print("NOW RUN cNMF on AUGER using this h5ad, then save the resulting GEP loadings and scores in the same folder, and re-run this function to continue with projection/merging/nebula steps.")
            return None
        
        # if cnmf files ecists --> project H and D
        df_spectra = pd.read_csv(CNMF_SPECTRA_PATH, sep="\t", index_col=0)
        reference_hvg = df_spectra.columns.tolist()
        gene_stds = np.load(CNMF_SPECTRA_PATH.replace("spectra", "gene_stds").replace(".txt", ".npy"))  # adjust path as needed

        adata_healthy = adata[adata.obs[CONTRAST_VARIABLE] == CONTRAST_BASELINE].to_memory().copy()
        adata_healthy.X = adata_healthy.layers["counts"].copy()
        df_healthy = project_cnmf_geps(adata_healthy, "healthy", SAVE_HEALTHY_PCA_RESULT, df_spectra, reference_hvg, gene_stds)

    df_stats_healthy = pd.read_csv(f"{SAVE_HEALTHY_PCA_RESULT}/healthy_stats.csv", index_col=0)
    df_loadings_healthy = pd.read_csv(f"{SAVE_HEALTHY_PCA_RESULT}/gene_loadings.csv", index_col=0)

    ###################
    # Calculate Factors for Diseased
    ###################

    print("Running Factor calculation on Diseased...")

    adata_diseased = adata[adata.obs[CONTRAST_VARIABLE] == CONTRAST_STIM].to_memory().copy()

    adata_diseased.X = adata_diseased.layers["counts"].copy()

    if (TYPE_DEG == "PCA_nebula") | (TYPE_DEG == "baseline_nebula"):
        results = calculate_pca(
            adata=adata_diseased,
            calculate_or_project="project",
            saving_folder=SAVE_HEALTHY_PCA_RESULT, 
            df_loadings=df_loadings_healthy,
            df_stats=df_stats_healthy
        )
    elif (TYPE_DEG == "cNMF_nebula"):
        df_diseased = project_cnmf_geps(adata_diseased, "diseased", SAVE_HEALTHY_PCA_RESULT, df_spectra, reference_hvg, gene_stds)

    ###################
    # Merge
    ###################

    # Merge with Healthy (alredy claculted --> SAME for all)    
    df_pcs_healthy = pd.read_csv(f"{SAVE_HEALTHY_PCA_RESULT}/healthy_pcs.csv")
    df_pcs_diseased = pd.read_csv(f"{SAVE_HEALTHY_PCA_RESULT}/diseased_pcs.csv")

    df_all = pd.concat([df_pcs_healthy, df_pcs_diseased], axis=0)
    df_all.to_csv(f"{SAVE_HEALTHY_PCA_RESULT}/pcs_all.csv", index=False)

    ###################
    # Save h5ad with these PCs
    ###################

    print("Saving h5ad file...")

    pcs = [c for c in df_all.columns if c.startswith("PC_")]
    for pc in pcs:
        if pc in adata.obs.columns.tolist():
            adata.obs = adata.obs.drop(columns=[pc])
    if "barcode" in adata.obs.columns:
        adata.obs = adata.obs.drop(columns=["barcode"])
    adata.obs = adata.obs.merge(df_all, how="left", left_index=True, right_on="barcode", validate="one_to_one")
    adata.obs.index = adata.obs["barcode"]  # restore original index
    adata.obs = adata.obs.drop(columns=["barcode"])

    # Sanity checks before saving
    assert adata.obs.shape[0] == adata.X.shape[0], \
        f"obs rows ({adata.obs.shape[0]}) != X rows ({adata.X.shape[0]})"
    assert adata.obs.index.equals(pd.Index(adata.obs_names)), "Index mismatch after merge"
    assert not adata.obs[pcs].isna().any().any(), "Some cells missing PC values after merge"
    assert adata.obs_names.is_unique, "Duplicate barcodes after merge"
    print(f"adata intact: {adata.shape[0]} cells, {adata.shape[1]} genes")

    covs = [CONTRAST_VARIABLE, *COVARIATES_FOR_DEG, *pcs]
    # ATTENTION: interaction term is not a real col, so do the ehck only on asubset of col
    covs_for_nan_check = [c for c in covs if ":" not in c and "*" not in c]
    mask = adata.obs[covs_for_nan_check].isna().any(axis=1)
    print(f"ATTENTION: Dropping {mask.sum()} cells with NaN covariates: {adata.obs_names[mask].tolist()}")
    adata = adata[~mask].copy()

    ADATA_PATH_H5AD = f"{os.path.splitext(ADATA_PATH_QS)[0]}.h5ad"
    adata.write_h5ad(ADATA_PATH_H5AD)

    DEG.check_corr_cov_in_design(adata, vars_in_formula = [CONTRAST_VARIABLE, *COVARIATES_FOR_DEG, *pcs, LIBRARY_SIZE_COL], corr_thr=0.7, split=" + ")

    ###################
    # Create qs file
    ###################

    print("Saving qs file...")

    import tempfile
    tmp_dir = tempfile.mkdtemp(dir="/tmp")
    adata.X = adata.layers["counts"].copy() # ATTENTION
    print("\tSaving files for R coversion...")
    preprocessing.save_files_for_R_conversion(adata, tmp_dir)
    print("\tBuilding R obj...")
    preprocessing.build_qs_from_python_files(tmp_dir, saving_path=ADATA_PATH_QS, contrast_var=CONTRAST_VARIABLE, reference_level=CONTRAST_BASELINE)

    ###################
    # Run Nebula
    ###################

    print("Running Nebula...")

    if TYPE_DEG == "PCA_nebula":
        # all the PCA calculate/project/merge/save h5ad code stays here
        pcs = [c for c in df_all.columns if c.startswith("PC_")]
    elif TYPE_DEG == "baseline_nebula":
        pcs = []  # no PCs as covariates
    elif TYPE_DEG == "gradient_score_nebula":
        pcs = ["gradient_score"]  # use gradient score as covariate
    elif TYPE_DEG == "cNMF_nebula":
        pcs = [c for c in df_all.columns if c.startswith("GEP_")]  # use cNMF gene expression programs as covariates
    else:
        raise ValueError(f"Unknown TYPE_DEG: {TYPE_DEG}")

    # Define PCs cov
    # Run Nebula
    nebula_result_path = DEG.run_nebula_parallel_script(
        path_qs = ADATA_PATH_QS,
        path_nebula_script = PARALLEL_NEBULA_SCRIPT_PATH,
        id_col = SAMPLE_VARIABLE,
        covs = [CONTRAST_VARIABLE, *COVARIATES_FOR_DEG, *pcs], # wihtout LIBRARY_SIZE_COL, with constrst vaibale too
        offset_col = LIBRARY_SIZE_COL,
        n_folds = 40, # (MORE gene chunks = LESS RAM per chunk)
        n_cores = 40, # (FEWER parallel jobs = LESS total RAM)
        save_tmp = False,
        suffix = None,
    )

    # save csv
    save_csv_path = f"{SAVE_HEALTHY_PCA_RESULT}/deg_results.csv"
    ro.r(f'''
        library(qs)
        obj <- qread("{nebula_result_path}")
        df <- obj$summary
        write.csv(df, "{save_csv_path}", row.names = FALSE)
    ''')

    ###################
    # Delete tmp files
    ###################

    if delete_tmp_files:
        os.remove(ADATA_PATH_H5AD)
        os.remove(ADATA_PATH_QS)
        shutil.rmtree(f"{SAVE_HEALTHY_PCA_RESULT}/de_results")