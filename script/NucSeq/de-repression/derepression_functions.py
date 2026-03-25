import re
import numpy as np
import pandas as pd
import scipy.sparse
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact
import scanpy as sc


def find_depressed_genes(
    adata_healthy,
    donor_col      = "donor_id",
    age_col        = "age",
    tot_UMI_per_cell_col        = "n_counts", # total number of UMI counts per cell
    covariate_cols       = None,
    threshold      = 0.02,
    min_cells      = 30,
    n_perm         = 10_000,
    fdr_alpha       = 0.05,
    seed           = 42,
):
    EXCLUDE_RE = re.compile(r"^MT-|^RPS\d|^RPL\d|^HB[^P]")

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

    #print(f"  After downsampling: {adata_sub.obs[donor_col].value_counts()} cells, {adata_sub.n_vars} genes, {adata_sub.obs[donor_col].nunique()} donors")

    # --- Step 3: compute per-donor detection rate --- (samples x genes)
    detect_matrix = np.zeros((len(valid_donors), adata_sub.n_vars))

    for j, d in enumerate(valid_donors):
        cells = adata_sub[adata_sub.obs[donor_col] == d].X
        detect_matrix[j] = np.asarray((cells > 0).mean(axis=0)).ravel()

    # compute donor-level metadata

    donor_meta = adata.obs.groupby(donor_col).agg(
        age        = (age_col, 'first'),
        mean_log_umi  = (tot_UMI_per_cell_col, lambda x: np.log(x.astype(float)).mean()),
        mean_log_umi2 = (tot_UMI_per_cell_col, lambda x: (np.log(x.astype(float))**2).mean()),
    )

    donor_meta = donor_meta.loc[valid_donors] # reindex to match detect_df

    if covariate_cols:
        cov_meta = adata.obs.groupby(donor_col)[covariate_cols].first()
        donor_meta = donor_meta.join(cov_meta)

    # --- Step 4: young / old groups ---
    ages = donor_meta["age"].values
    q25, q75 = np.percentile(ages, [25, 75])
    young_mask = ages <= q25
    old_mask   = ages >= q75
    print("Subsampling adata to Young and Old quartiles only...")
    quartile_mask = young_mask | old_mask # samples to reatain for the analysis
    detect_matrix = detect_matrix[quartile_mask, :] # subset to young + old samples only
    donor_meta = donor_meta[quartile_mask] # subset to young + old samples only
    # attention need to recompute young_mask and old_mask on the subset of samples only (otherwise they will be misaligned with the detect_matrix and donor_meta)
    young_mask    = young_mask[quartile_mask]   # recompute on the subset
    old_mask      = old_mask[quartile_mask]     # recompute on the subset
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
        np.ones(len(donor_meta)), # interecept
        donor_meta[["mean_log_umi", "mean_log_umi2"]].values, # mean log UMI and its square
        *(donor_meta[c].values.astype(float) for c in (covariate_cols or [])), # add all other covs
    ])
    
    # age vecs as binary
    age_vec = old_mask.astype(float)   # 0 = young quartile, 1 = old quartile
    #age_vec=donor_meta["age"].values.astype(float)

    perm_res = permutation_test_on_age_FWL(
                    Y=detect_cand,    # (n_genes, n_donors) detection rates, candidates only
                    X_cov=X_cov.astype(float),
                    age_vec=age_vec,
                    n_perm = n_perm,
                    seed   = seed,
                )
    
    # for logFC: "half a detected cell" in the downsampled pool,
    eps = 1 / (2 * min_n)   # half a cell in the downsampled pool

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
        "logFC_detect":   np.log2(
                        (old_detect[candidate_mask]   + eps) /
                        (young_detect[candidate_mask] + eps)
                    ),
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
        #fraction of gene set with count > 0
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
        net = pd.DataFrame({"source": "G", "target": gene_set, "weight": 1.0})
        dc.mt.aucell(data=adata, net=net, 
                     n_up = max(100, int(adata.n_vars * 0.2)),   
                        # 20%, min 100 genes, 
                        # # ATTNETION: this is a heuristic parameter that may need tuning
                     layer=None, raw=False, verbose=False)
        cell_scores = adata.obsm["score_aucell"]["G"].values  
        print(f"  UCell NaNs: {np.isnan(cell_scores).sum()} / {len(cell_scores)}")

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
    overlap       = (is_sig & is_pc).sum()
    overlap_genes = results_df.loc[is_sig & is_pc, "gene"].tolist()

    # Edge case: no significant genes → nothing to test
    if n_sig == 0:
        return {"odds_ratio": np.nan, "pval": 1.0, "overlap": 0,
                "overlap_genes": [],
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
                "overlap_genes": overlap_genes,
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
        "overlap_genes": overlap_genes,
        "n_sig":        int(n_sig),
        "n_pc_cands":   int(n_pc_cand),
        "n_candidates": n_cand,
        "n_strata":     len(tables),
    }