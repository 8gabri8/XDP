import re
import numpy as np
import pandas as pd
import scipy.sparse
from statsmodels.stats.multitest import multipletests
import scanpy as sc


# # ---------------------------------------------------------------------------
# # Main function
# # ---------------------------------------------------------------------------

# def find_depressed_genes(
#     adata_healthy,                        # AnnData, healthy donors only
#     donor_col      = "donor_id",          # obs column: donor identifier
#     age_col        = "age",               # obs column: donor age
#     umi_col        = "n_counts",          # obs column: per-cell total counts
#     covariate_cols = None,                # extra regression covariates
#     threshold      = 0.02,               # max DR in young donors (e.g. 0.02 = 2%)
#     min_cells      = 30,                  # min cells per donor to be included
#     n_perm         = 10_000,             # permutations for the FWL test
#     fdr_alpha      = 0.05,               # BH FDR threshold
#     seed           = 42,
# ):

#     EXCLUDE_RE = re.compile(r"^MT-|^RPS\d|^RPL\d|^HB[^P]")

#     rng       = np.random.RandomState(seed)
#     var_names = list(adata_healthy.var_names)
#     obs       = adata_healthy.obs
#     counts = adata_healthy.X

#     # check that x is raw counta
#     assert (counts[:100,:100].toarray() % 1 == 0).all(), "counts must be integers"

#     print(f"  Input: {counts.shape[0]:,} cells, {obs[donor_col].nunique()} donors")

#     # ------------------------------------------------------------------
#     # Step 1 — drop donors with too few cells
#     #
#     # A donor with very few cells has unreliable detection rates: a gene
#     # seen in 2 out of 10 cells looks like DR=20% but is just sampling noise.
#     # ------------------------------------------------------------------

#     donors        = obs[donor_col].values
#     unique_donors = np.unique(donors)

#     donor_ncells = {d: int((donors == d).sum()) for d in unique_donors}
#     valid_donors = sorted(d for d, n in donor_ncells.items() if n >= min_cells)
#     n_dropped    = len(unique_donors) - len(valid_donors)

#     print(f"  Donors kept: {len(valid_donors)}  "
#           f"(dropped {n_dropped} with < {min_cells} cells)")
#     if not valid_donors:
#         raise ValueError("No donors pass the minimum cell count filter.")

#     # ------------------------------------------------------------------
#     # Step 2 — downsample all donors to the same number of cells
#     #
#     # Without this, a donor with 5,000 cells will always show higher DR
#     # than one with 200 cells for the same gene, simply because of more
#     # sampling draws. Equalising cell counts makes DR comparable across donors.
#     #
#     # Strategy: find the minimum cell count, then for each donor randomly
#     # select exactly that many cells. Indices are decided here and reused
#     # when loading the count matrix — same two-pass approach as the original.
#     # ------------------------------------------------------------------
#     min_n = min(donor_ncells[d] for d in valid_donors)
#     print(f"  Downsampling all donors to {min_n} cells")

#     donor_keep = {}
#     for d in valid_donors:
#         total = donor_ncells[d]
#         if total == min_n:
#             donor_keep[d] = set(range(total))           # keep all, no shuffle
#         else:
#             donor_keep[d] = set(
#                 rng.choice(total, size=min_n, replace=False).tolist()
#             )

#     # ------------------------------------------------------------------
#     # Step 3 — compute per-gene detection rate (DR) per donor
#     #
#     # DR = fraction of downsampled cells where the gene has >= 1 count.
#     # This is binary (detected / not detected), not expression level.
#     #
#     # Also collect donor-level metadata for the regression:
#     #   - age, covariates: taken from the first cell (donor-level, not
#     #     cell-level, so every cell has the same value for this donor)
#     #   - mean_log_umi: average of log(total counts) across ALL cells of
#     #     the donor, not just the subsample. This is a property of the donor's
#     #     sequencing depth, independent of which cells were sampled.
#     # ------------------------------------------------------------------
#     n_genes = len(var_names)

#     # detect_matrix[g, j] = DR of gene g in donor j
#     detect_matrix = np.zeros((n_genes, len(valid_donors)), dtype=np.float32)
#     meta_records  = []

#     for j, d in enumerate(valid_donors):
#         all_idx    = np.where(donors == d)[0]           # all cell indices for d
#         local_keep = sorted(donor_keep[d])              # which of those to keep
#         selected   = all_idx[local_keep]                # global indices of kept cells

#         # Count how many kept cells have >= 1 count per gene, then normalise
#         detected = np.asarray(
#             (counts[selected, :] > 0).sum(axis=0)
#         ).ravel()
#         detect_matrix[:, j] = detected / min_n          # DR in [0, 1]

#         # Donor-level metadata (same value for all cells of this donor)
#         first  = obs.iloc[all_idx[0]]
#         record = {
#             "donor_id":     d,
#             "age":          first[age_col],
#             # log of raw counts, no +1 — matches the original exactly
#             "mean_log_umi": np.log(
#                 obs.iloc[all_idx][umi_col].values.astype(float)
#             ).mean(),
#         }
#         for col in covariate_cols:
#             record[col] = first[col]

#         meta_records.append(record)

#     donor_meta = pd.DataFrame(meta_records)

#     # ------------------------------------------------------------------
#     # Step 4 — define Young (Y) and Old (O) donor groups by age quartile
#     #
#     # Y = bottom 25th percentile of age among valid donors
#     # O = top 75th percentile of age among valid donors
#     #
#     # These are used ONLY to compute mean DR for candidate gene selection
#     # (steps 5-6). The permutation test in step 7 uses ALL donors to
#     # estimate the age slope across the full age gradient.
#     # ------------------------------------------------------------------
#     ages       = donor_meta["age"].values
#     q25, q75   = np.percentile(ages, [25, 75])
#     young_mask = ages <= q25
#     old_mask   = ages >= q75

#     print(f"  Young (age <= {q25:.0f}): {young_mask.sum()} donors")
#     print(f"  Old   (age >= {q75:.0f}): {old_mask.sum()} donors")

#     # Mean DR across young / old donors for each gene
#     young_detect = detect_matrix[:, young_mask].mean(axis=1)  # shape: (n_genes,)
#     old_detect   = detect_matrix[:, old_mask].mean(axis=1)    # shape: (n_genes,)

#     # ------------------------------------------------------------------
#     # Step 5 — select candidate genes
#     #
#     # A gene is a candidate if:
#     #   (a) mean DR in young donors <= threshold  →  it is silent in youth
#     #   (b) it is not a mitochondrial / ribosomal / hemoglobin gene
#     #
#     # Note: whether old_detect > threshold (exceeds_in_old) is saved as
#     # an annotation in the output table but is NOT a hard filter. The
#     # permutation test is the only statistical gate.
#     # ------------------------------------------------------------------
#     exclude_mask   = np.array([bool(EXCLUDE_RE.match(g)) for g in var_names])
#     candidate_mask = (young_detect <= threshold) & ~exclude_mask
#     n_cand         = int(candidate_mask.sum())

#     print(f"\n  Threshold: {threshold*100:.1f}%")
#     print(f"  Candidate genes: {n_cand}")
#     if n_cand == 0:
#         raise ValueError(
#             f"No candidates found at threshold={threshold}. "
#             "Try raising the threshold."
#         )

#     # Subset detection matrix and gene names to candidates only
#     detect_cand = detect_matrix[candidate_mask, :]              # (n_cand, n_donors)
#     cand_names  = [g for g, m in zip(var_names, candidate_mask) if m]

#     # ------------------------------------------------------------------
#     # Step 6 — FWL permutation test
#     #
#     # For each candidate gene, estimate the age slope (beta) on detection
#     # rate after controlling for covariates, and test whether it is
#     # significantly positive via permutation.
#     #
#     # See _run_permutation_test for full details.
#     # ------------------------------------------------------------------
#     print(f"  Running permutation test ({n_perm:,} permutations)...")
#     perm_res = _run_permutation_test(
#         detect_cand, donor_meta, covariate_cols, n_perm, seed
#     )

#     # ------------------------------------------------------------------
#     # Step 7 — assemble results and apply FDR threshold
#     # ------------------------------------------------------------------
#     results_df = pd.DataFrame({
#         "gene":           cand_names,
#         "young_detect":   young_detect[candidate_mask],
#         "old_detect":     old_detect[candidate_mask],
#         "exceeds_in_old": old_detect[candidate_mask] > threshold,
#         "obs_beta":       perm_res["obs_beta"].values,
#         "perm_pval":      perm_res["perm_pval"].values,
#         "perm_padj":      perm_res["perm_padj"].values,
#         "n_donors":       len(valid_donors),
#     })

#     sig_mask  = results_df["perm_padj"] < fdr_alpha
#     gene_list = results_df.loc[sig_mask, "gene"].tolist()

#     print(f"  Positive beta: {int((results_df['obs_beta'] > 0).sum())} / {n_cand}")
#     print(f"  Significant (padj < {fdr_alpha}): {sig_mask.sum()}")
#     print(f"  Final G: {len(gene_list)} genes")

#     return gene_list, results_df, donor_meta


# ---------------------------------------------------------------------------
# FWL permutation test
# ---------------------------------------------------------------------------

# def _run_permutation_test(
#     detect_matrix,    # (n_genes, n_donors) detection rates, candidates only
#     meta,             # per-donor DataFrame with age, mean_log_umi, covariates
#     covariate_cols,   # list of extra covariate column names
#     n_perm = 10_000,
#     seed   = 42,
# ):
#     """
#     Test whether each gene's detection rate increases with age, using the
#     Frisch-Waugh-Lovell (FWL) theorem + permutation for inference.

#     WHY FWL?
#     Running one OLS per gene per permutation would cost n_genes * n_perm
#     full matrix solves. FWL reduces this to:
#       - one QR decomposition (shared across all genes and permutations)
#       - one matrix residualization of all genes (done once)
#       - n_perm cheap vector dot products
#     The result is mathematically identical to per-gene OLS.

#     HOW IT WORKS (three steps):
#       1. Residualize age on covariates → age_resid
#          (the part of age that covariates cannot explain)
#       2. Residualize all gene detection rates on the same covariates → detect_resid
#          (the part of DR that covariates cannot explain)
#       3. The OLS slope of detect_resid on age_resid is the age coefficient,
#          identical to what full OLS would give.

#     PERMUTATION NULL:
#       Shuffle age labels 10,000 times. For each shuffle, re-residualize the
#       permuted age vector and recompute the slope. This builds a null
#       distribution of slopes under "no age effect". The one-sided p-value is
#       the fraction of null slopes >= the observed slope.

#     detect_resid is NOT re-computed for each permutation — it stays fixed.
#     Only age is shuffled, because the null hypothesis is specifically that
#     age labels are arbitrary, not that detection rates are arbitrary.
#     """

#     rng = np.random.RandomState(seed)
#     n_genes, n_donors = detect_matrix.shape

#     # ------------------------------------------------------------------
#     # Build covariate matrix X
#     #
#     # Always includes: intercept, logUMI (linear), logUMI² (quadratic).
#     # logUMI² matters because DR saturates at high depth — the relationship
#     # is not linear.
#     # Plus any user-specified covariates (default: sex, PC1–PC5).
#     # ------------------------------------------------------------------
#     log_umi = meta["mean_log_umi"].values
#     X_cov = np.column_stack([
#         np.ones(n_donors),                                          # intercept
#         log_umi,                                                    # logUMI
#         log_umi ** 2,                                               # logUMI²
#         *[meta[col].values.astype(float) for col in covariate_cols],
#     ])

#     # QR decomposition of X: X = QR, Q has orthonormal columns.
#     # Projection onto column space of X is simply Q @ Q.T,
#     # avoiding the numerically unstable (X.T @ X)^{-1} inverse.
#     Q, _ = np.linalg.qr(X_cov, mode="reduced")   # Q shape: (n_donors, k)

#     # ------------------------------------------------------------------
#     # FWL step 1 — residualize age on covariates (done once)
#     #
#     #   age_resid = age - Q (Q.T age)
#     #
#     # age_resid is the part of age that covariates cannot predict.
#     # This is the "clean" age signal used in all regressions.
#     # ------------------------------------------------------------------
#     age_vec   = meta["age"].values.astype(float)
#     age_resid = age_vec - Q @ (Q.T @ age_vec)     # shape: (n_donors,)
#     ss_age    = float(np.sum(age_resid ** 2))      # scalar denominator, reused below

#     if ss_age < 1e-12:
#         raise ValueError(
#             "No residual age variance after removing covariates. "
#             "Check for collinearity between age and the covariate columns."
#         )

#     # ------------------------------------------------------------------
#     # FWL step 2 — residualize all genes on covariates (done once)
#     #
#     #   detect_resid = Y - Q (Q.T Y.T).T        [shape: n_genes x n_donors]
#     #
#     # Each row is one gene's detection rates with all covariate effects removed.
#     # This is the expensive step but is paid only once, not once per permutation.
#     # ------------------------------------------------------------------
#     detect_resid = detect_matrix - (Q @ (Q.T @ detect_matrix.T)).T

#     # ------------------------------------------------------------------
#     # FWL step 3 — observed age beta for every gene (one matrix multiply)
#     #
#     #   beta_g = (detect_resid[g] · age_resid) / ||age_resid||²
#     #
#     # Vectorised across all genes: detect_resid @ age_resid / ss_age
#     # Positive beta = detection rate increases with age = de-repression.
#     # ------------------------------------------------------------------
#     obs_betas = (detect_resid @ age_resid) / ss_age   # shape: (n_genes,)

#     # ------------------------------------------------------------------
#     # Permutation null distribution
#     #
#     # Shuffle age labels n_perm times. All permutations are done at once
#     # as matrix operations:
#     #
#     #   perm_indices   : (n_donors, n_perm)  — each column = one permutation
#     #   age_perm_mat   : (n_donors, n_perm)  — shuffled age vectors
#     #   age_resid_perm : (n_donors, n_perm)  — residualized shuffled ages
#     #   beta_perm_mat  : (n_genes,  n_perm)  — all null betas
#     #
#     # detect_resid stays fixed — only age is shuffled (see docstring above).
#     # ------------------------------------------------------------------
#     perm_indices   = np.column_stack(
#         [rng.permutation(n_donors) for _ in range(n_perm)]
#     )                                                       # (n_donors, n_perm)
#     age_perm_mat   = age_vec[perm_indices]                  # (n_donors, n_perm)
#     age_resid_perm = age_perm_mat - Q @ (Q.T @ age_perm_mat)  # residualize
#     ss_age_perm    = np.sum(age_resid_perm ** 2, axis=0)   # (n_perm,) one ss per perm

#     beta_perm_mat  = (
#         (detect_resid @ age_resid_perm) / ss_age_perm[np.newaxis, :]
#     )                                                       # (n_genes, n_perm)

#     # ------------------------------------------------------------------
#     # One-sided p-value per gene
#     #
#     #   p_g = (1 + #{perms where beta_perm >= obs_beta}) / (1 + n_perm)
#     #
#     # One-sided: de-repression means DR goes UP with age.
#     # The +1 in numerator and denominator prevents p=0 (which would be
#     # dishonest given a finite number of permutations).
#     # ------------------------------------------------------------------
#     perm_counts = np.sum(
#         beta_perm_mat >= obs_betas[:, np.newaxis], axis=1
#     )                                                       # (n_genes,)
#     perm_pvals  = (1 + perm_counts) / (1 + n_perm)

#     # BH FDR correction across all candidate genes jointly
#     _, perm_padjs, _, _ = multipletests(perm_pvals, method="fdr_bh")

#     return pd.DataFrame({
#         "obs_beta":  obs_betas,
#         "perm_pval": perm_pvals,
#         "perm_padj": perm_padjs,
#     })


def find_depressed_genes(
    adata_healthy,
    donor_col      = "donor_id",
    age_col        = "age",
    tot_UMI_per_cell_col        = "n_counts", # total number of UMI counts per cell
    threshold      = 0.02,
    min_cells      = 30,
    n_perm         = 10_000,
    fdr_alpha       = 0.05,
    seed           = 42,
):
    EXCLUDE_RE = re.compile(r"^MT-|^RPS\d|^RPL\d|^HB[^P]")
    rng = np.random.RandomState(seed)

    adata = adata_healthy.copy()
    assert (adata.X[:100, :100].toarray() % 1 == 0).all(), "X must be raw counts"

    # --- Step 1: drop donors with too few CELLS ---
    cell_counts = adata.obs[donor_col].value_counts()
    valid_donors = cell_counts[cell_counts >= min_cells].index.tolist()
    adata = adata[adata.obs[donor_col].isin(valid_donors)].copy()
    print(f"  Donors kept: {len(valid_donors)}  (dropped {cell_counts.shape[0] - len(valid_donors)} with < {min_cells} cells)")

    # --- Step 2: downsample donors to same number of cells ---
    min_n = cell_counts[valid_donors].min()
    print(f"  Downsampling to {min_n} cells per donor")

    adata_sub = adata[adata.obs.groupby(donor_col).sample(n=min_n, random_state=seed).index].copy()

    print(f"  After downsampling: {adata_sub.obs[donor_col].value_counts()} cells, {adata_sub.n_vars} genes, {adata_sub.obs[donor_col].nunique()} donors")

    # --- Step 3: compute per-donor detection rate --- (samples x genes)
    detect_matrix = np.zeros((len(valid_donors), adata_sub.n_vars))

    for j, d in enumerate(valid_donors):
        cells = adata_sub[adata_sub.obs[donor_col] == d].X
        detect_matrix[j] = np.asarray((cells > 0).mean(axis=0)).ravel()

    detect_df = pd.DataFrame(detect_matrix, index=valid_donors, columns=adata_sub.var_names)

    # compute donor-level metadata

    donor_meta = adata.obs.groupby(donor_col).agg(
        age        = (age_col, 'first'),
        mean_log_umi  = (tot_UMI_per_cell_col, lambda x: np.log(x.astype(float)).mean()),
        mean_log_umi2 = (tot_UMI_per_cell_col, lambda x: (np.log(x.astype(float))**2).mean()),
    )
    donor_meta = donor_meta.loc[valid_donors] # reindex to match detect_df
    detect_df = detect_df.join(donor_meta)

    # --- Step 4: young / old groups ---
    ages = donor_meta["age"].values
    q25, q75 = np.percentile(ages, [25, 75])
    young_mask = ages <= q25
    old_mask   = ages >= q75
    print(f"  Young (≤{q25:.0f}): {young_mask.sum()}  |  Old (≥{q75:.0f}): {old_mask.sum()}")

    # average detection rate across young (or old) donors for each gene.
    young_detect = detect_matrix[young_mask, :].mean(axis=0)
    old_detect   = detect_matrix[old_mask,   :].mean(axis=0)

    # --- Step 5: candidate genes ---
    exclude_mask   = np.array([bool(EXCLUDE_RE.match(g)) for g in adata.var_names])
    candidate_mask = (young_detect <= threshold) & ~exclude_mask
    cand_names     = adata.var_names[candidate_mask].tolist()
    detect_cand = detect_matrix[:, candidate_mask].T
    print(f"  Candidates: {len(cand_names)}")

    # --- Step 6: permutation test ---
    print(f"  Running permutation test ({n_perm:,} perms)...")

    # NO age
    X_cov = np.column_stack([
        np.ones(len(donor_meta)),
        donor_meta[["mean_log_umi", "mean_log_umi2"]].values
    ])
    age_vec=donor_meta["age"].values.astype(float)

    perm_res = permutation_test_on_age_FWL(
                    Y=detect_cand,    # (n_genes, n_donors) detection rates, candidates only
                    X_cov=X_cov.astype(float),
                    age_vec=age_vec,
                    n_perm = 10_000,
                    seed   = 42,
                )

    # --- Step 7: results ---
    results_df = pd.DataFrame({
        "gene":           cand_names,
        "young_detect":   young_detect[candidate_mask],
        "old_detect":     old_detect[candidate_mask],
        "overall_detect": detect_cand.mean(axis=1),
        "exceeds_in_old": old_detect[candidate_mask] > threshold,
        "obs_beta":       perm_res["obs_beta"].values,
        "perm_pval":      perm_res["perm_pval"].values,
        "perm_padj":      perm_res["perm_padj"].values,
    })

    sig_mask  = results_df["perm_padj"] < fdr_alpha
    gene_list = results_df.loc[sig_mask, "gene"].tolist()
    print(f"  Significant (padj < {fdr_alpha}): {sig_mask.sum()}  →  {len(gene_list)} genes")

    return gene_list, results_df, donor_meta

def compute_scores_and_cov(adata, gene_set, donor_col="donor_id", age_col="age",
                            umi_col="n_counts", condition_col=None, cov_cols=None,
                            score_method="gene_count", percentile=90):
    """
    Score each donor on expression of gene_set G and collect donor-level covariates.
    Returns one row per donor, ready for test_scores().

    Always returned: donor_id, age, age_scaled, median_umi, median_umi_scaled, score_S
    Optional:        condition  (if condition_col given)
                     cov_cols   (any list of .obs columns, e.g. ["sex","PC1",...,"PC5"])
                                 → for each column: takes the first value per donor
                                   (assumes the column is donor-level, same for all cells)

    score_method:
      "gene_count"   — fraction of G genes detected per cell (count > 0)
      "count_ratio"  — G counts / total counts per cell
      "scanpy_score" — sc.tl.score_genes: mean(G) minus random background (fast)
      "ucell"        — AUCell rank-based score (requires decoupler, slowest)
    percentile:
      how to collapse cells → donor (default: 90th percentile)
    """

    adata = adata.copy()
    assert (adata.X[:100, :100].toarray() % 1 == 0).all(), "X must be raw counts"

    if cov_cols is None:
        cov_cols = []

    # keep only G genes present in adata
    gene_set = [g for g in gene_set if g in adata.var_names]
    if not gene_set:
        raise ValueError("No genes in G found in adata.var_names.")

    # --- per-cell score ---------------------------------------------------
    gene_idx = [i for i, g in enumerate(adata.var_names) if g in set(gene_set)]
    X_sub    = scipy.sparse.csr_matrix(adata.X[:, gene_idx])

    if score_method == "gene_count":
        cell_scores = np.asarray((X_sub > 0).sum(axis=1)).ravel() / len(gene_set)

    elif score_method == "count_ratio":
        X_all      = scipy.sparse.csr_matrix(adata.X)
        g_counts   = np.asarray(X_sub.sum(axis=1)).ravel().astype(float)
        tot_counts = np.asarray(X_all.sum(axis=1)).ravel().astype(float)
        tot_counts[tot_counts == 0] = 1
        cell_scores = g_counts / tot_counts

    elif score_method == "scanpy_score":
        raw_X = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.score_genes(adata, gene_list=gene_set, score_name="_score")
        adata.X = raw_X
        cell_scores = adata.obs["_score"].values

    elif score_method == "ucell":
        import decoupler as dc
        raw_X = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        net = pd.DataFrame({"source": "G", "target": gene_set, "weight": 1.0})
        dc.mt.aucell(data=adata, net=net, 
                     n_up=adata.n_vars // 100, # ATTNETION: this is a heuristic parameter that may need tuning
                     layer=None, raw=False, verbose=False)
        adata.X = raw_X
        cell_scores = adata.obsm["score_aucell"]["G"].values  

    else:
        raise ValueError(f"Unknown score_method '{score_method}'. "
                         "Choose: 'gene_count', 'count_ratio', 'scanpy_score', 'ucell'.")

    # --- aggregate cells → donors -----------------------------------------
    
    adata.obs["_score"] = cell_scores

    # DICT TO EPXLAIN HWO TO AGGRGEATE EACHVARIABLE:
    agg_dict = {
        "age":            (age_col,   "first"),
        "score_S":        ("_score",  lambda x: np.percentile(x, percentile)),
        "median_log_umi": (umi_col,   lambda x: np.log(x.median())),
    }
    if condition_col:
        agg_dict["condition"] = (condition_col, "first")
    for col in cov_cols:
        agg_dict[col] = (col, "first")

    df = adata.obs.groupby(donor_col, observed=True).agg(**agg_dict).reset_index()

    # NORMLAISE
    # scale continuous variables — ready to use directly in test_scores()
    df["age_scaled"]        = (df["age"] - df["age"].mean()) / df["age"].std()
    df["median_log_umi_scaled"] = (df["median_log_umi"] - df["median_log_umi"].mean()) / df["median_log_umi"].std()
    # only for cross-zone comparison
    df["score_S_scaled"] = (df["score_S"] - df["score_S"].mean()) / df["score_S"].std()

    return df



def permutation_test_on_age_FWL(
    Y,    # (n_genes, n_donors) detection rates, candidates only
    X_cov, 
    age_vec,
    n_perm = 10_000,
    seed   = 42,
):

    rng = np.random.RandomState(seed)
    n_genes, n_donors = Y.shape

    # Calcultae: QR decomposition of X. WHY?
    # 1) Projection onto column space of X is simply Q @ Q.T --> easiaer when clacuting close form of OLS
    # 2) avoiding the numerically unstable (X.T @ X)^{-1} inverse.
    Q, _ = np.linalg.qr(X_cov, mode="reduced")   # Q shape: (n_donors, k)

    # ------------------------------------------------------------------
    # FWL step 1 — residualize age on covariates (done once)
    #
    #   (M age_vec) = age_resid = (I -QQ.T) age = age - Q (Q.T age)
    #
    # age_resid is the part of age that covariates cannot predict.
    # ------------------------------------------------------------------
    age_resid = age_vec - Q @ (Q.T @ age_vec)     # shape: (n_donors,) --> (M . age_vec) = a~
    ss_age    = float(np.sum(age_resid ** 2))      # scalar denominator --> ||a~||²

    if ss_age < 1e-12:
        raise ValueError(
            "No residual age variance after removing covariates. "
            "Check for collinearity between age and the covariate columns."
        )

    # ------------------------------------------------------------------
    # FWL step 2 — residualize all genes on covariates (done once)
    #
    #   detect_resid = Y - Q (Q.T Y.T).T        [shape: n_genes x n_donors]
    #
    # Each row is one gene's detection rates with all covariate effects removed.
    # This is the expensive step but is paid ONLY once
    # ------------------------------------------------------------------
    Y_res = Y - Y @ Q @ Q.T

    # ------------------------------------------------------------------
    # FWL step 3 — observed age beta for every gene (one matrix multiply)
    #
    #   beta_g = (Y_res[g] · age_resid) / ||age_resid||²
    #
    # Vectorised across all genes: Y_res @ age_resid / ss_age
    # ------------------------------------------------------------------
    obs_betas = (Y_res @ age_resid) / ss_age   # shape: (n_genes,)

    # ------------------------------------------------------------------
    # Permutation null distribution
    #
    # Shuffle age labels n_perm times
    #
    # Need to claculte same vlaues used before but for each eprmutation
    #
    # Once i have them nice clsed form to calcualte beta_age for each permutation
    # ------------------------------------------------------------------

    # Create permutation matrix
    P = np.column_stack([rng.permutation(n_donors) for _ in range(n_perm)]) # (n_donors, n_perm)

    # Create permuted age matrix
    age_perm_mat = age_vec[P] # (n_donors, n_perm)

    # Calculate residualized age for each permutation (column)
    # age_resid_perm = M age_perm_mat = (I - QQ.T) age_perm_mat = age_perm_mat - Q (Q.T age_perm_mat)
    age_resid_perm = age_perm_mat - Q @ (Q.T @ age_perm_mat) # (n_donors, n_perm)

    # Calcualte module (sum of squares) for each permuted age vector (column)
    ss_age_perm = np.sum(age_resid_perm ** 2, axis=0)   # (n_perm,) one ss per perm

    # Closed compact form for all permutations beta_age:
    beta_perm_mat  = (Y_res @ age_resid_perm) / ss_age_perm[np.newaxis, :] # (n_genes, n_perm)
    
    # ------------------------------------------------------------------
    # One-sided p-value per gene
    #
    #   p_g = (1 + #{perms where beta_perm >= obs_beta}) / (1 + n_perm)
    #
    # One-sided: de-repression means DR goes UP with age.
    # The +1 in numerator and denominator prevents p=0 
    # ------------------------------------------------------------------

    # Check how many permuted betas are greater than or equal to the observed beta for each gene
    perm_counts = np.sum(beta_perm_mat >= obs_betas[:, np.newaxis], axis=1) # (n_genes,)
    
    # Compute one-sided p-values
    perm_pvals  = (1 + perm_counts) / (1 + n_perm)

    # BH FDR correction across all candidate genes jointly
    _, perm_padjs, _, _ = multipletests(perm_pvals, method="fdr_bh")

    return pd.DataFrame({
        "obs_beta":  obs_betas,
        "perm_pval": perm_pvals,
        "perm_padj": perm_padjs,
    })


def polycomb_enrichment_cmh(
    results_df,       # output of permutation_test_on_age_FWL — one row per candidate gene
    polycomb_genes,   # set/list of polycomb target gene names
    fdr_alpha = 0.05, # significance threshold applied to perm_padj
    n_bins    = 20,   # number of expression-level strata
):

    from statsmodels.stats.contingency_tables import StratifiedTable

    # results_df --> deifne the unverse for the analyss = gene that have low DR in Yooung

    # ── Step 1: define the two binary labels for every candidate gene ──────
    # S_g = 1 if gene g is significant (perm_padj < alpha)
    # P_g = 1 if gene g is in the polycomb reference set
    pc_set  = set(polycomb_genes)
    is_sig  = (results_df["perm_padj"] < fdr_alpha).values  # genes sincat by OLS + permutation test
    is_pc   = results_df["gene"].isin(pc_set).values        # genes in polycomb reference set

    # Quick counts for reporting
    n_cand    = len(results_df)
    n_sig     = is_sig.sum()
    n_pc_cand = is_pc.sum()
    overlap   = (is_sig & is_pc).sum()

    # Edge case: no significant genes → nothing to test
    if n_sig == 0:
        return {"odds_ratio": np.nan, "pval": 1.0, "overlap": 0,
                "n_sig": 0, "n_pc_cands": n_pc_cand,
                "n_candidates": n_cand, "n_strata": 0}

    # ── Step 2: bin candidates by young_detect (expression in young donors) ─
    # np.percentile with linspace(0,100, n_bins+1) gives n_bins+1 quantile
    # edges that divide the genes into n_bins groups of equal size.
    # np.unique removes duplicate edges (can happen when many genes share
    # the same detection rate, e.g. exactly 0).
    expr        = results_df["overall_detect"].values                   # mean DR (across donor)
    bin_edges   = np.unique(np.percentile(expr, np.linspace(0, 100, n_bins + 1)))
    bin_edges[-1] += 1e-10                                           # make upper edge inclusive
    bin_idx     = np.clip(np.digitize(expr, bin_edges) - 1, 0, len(bin_edges) - 2)                     # (n_genes,) int

    # ── Step 3: build one 2×2 contingency table per expression bin ─────────
    #
    # Within bin b the table is:
    #
    #              is_pc=1   is_pc=0
    #   is_sig=1  [  a_b       b_b  ]
    #   is_sig=0  [  c_b       d_b  ]
    #
    # A stratum is only valid if both the significant row (a+b > 0)
    # and the non-significant row (c+d > 0) are non-empty — otherwise
    # the stratum carries no information about the odds ratio.
    tables = []
    for b in range(len(bin_edges) - 1):
        in_bin = bin_idx == b
        a  = int((in_bin &  is_sig &  is_pc).sum())
        bv = int((in_bin &  is_sig & ~is_pc).sum())
        c  = int((in_bin & ~is_sig &  is_pc).sum())
        d  = int((in_bin & ~is_sig & ~is_pc).sum())
        if (a + bv) > 0 and (c + d) > 0:        # both rows non-empty
            tables.append([[a, bv], [c, d]])

    # ── Step 4: fallback to plain Fisher if too few valid strata ───────────
    # CMH requires at least 2 strata to be meaningful. With fewer we fall
    # back to an unstratified Fisher's exact test on the overall 2×2 table.
    if len(tables) < 2:
        a  = int(overlap)
        bv = int(n_sig    - overlap)
        c  = int(n_pc_cand - overlap)
        d  = int(n_cand - a - bv - c)
        or_val, pval = fisher_exact([[a, bv], [c, d]], alternative="greater")
        return {"odds_ratio": or_val, "pval": pval, "overlap": int(overlap),
                "n_sig": int(n_sig), "n_pc_cands": int(n_pc_cand),
                "n_candidates": n_cand, "n_strata": len(tables)}

    # ── Step 5: CMH test ────────────────────────────────────────────────────
    #
    # StratifiedTable expects shape (2, 2, K) — two axes for the table,
    # one axis for strata. np.array(tables) has shape (K, 2, 2), so we
    # transpose axes (1, 2, 0) to get (2, 2, K).
    st = StratifiedTable(np.array(tables).transpose(1, 2, 0))

    # Mantel-Haenszel pooled OR:
    #
    #        Σ_b  (a_b * d_b) / n_b
    #  ψ̂ = ─────────────────────────
    #        Σ_b  (b_b * c_b) / n_b
    #
    # Each stratum contributes proportionally to its size n_b.
    # ψ̂ > 1 means significant genes are more likely to be polycomb targets
    # than equally-expressed non-significant genes.
    pooled_or  = st.oddsratio_pooled # magnitude of the enrichment 

    # CMH χ² test of H₀: ψ = 1 in all strata.
    # This is a two-sided test — we convert to one-sided below.
    cmh_result = st.test_null_odds()

    # ── Step 6: convert to one-sided p-value ───────────────────────────────
    # The biological hypothesis is directional: enrichment (OR > 1).
    # A two-sided test wastes power on the depletion direction.
    # If OR >= 1 (correct direction): p_one = p_two / 2
    # If OR <  1 (wrong direction):   p_one = 1 - p_two / 2  (penalised)
    if pooled_or >= 1:
        pval_one = cmh_result.pvalue / 2
    else:
        pval_one = 1 - cmh_result.pvalue / 2

    return {
        "odds_ratio":   pooled_or,
        "pval":         pval_one,
        "overlap":      int(overlap),
        "n_sig":        int(n_sig),
        "n_pc_cands":   int(n_pc_cand),
        "n_candidates": n_cand,
        "n_strata":     len(tables),
    }