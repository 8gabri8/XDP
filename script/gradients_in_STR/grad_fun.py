
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, t as t_dist, norm
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc


def process_adata(adata):
    """Restore raw counts, keep protein-coding genes, normalize + log1p.
    Adds a "log1p_norm" layer used by every correlation/plotting function below."""
    adata = adata[:, ~adata.var_names.str.startswith("ENS")].copy()  # keep only protein coding genes
    adata.X = adata.layers["counts"].copy()  # restore raw counts before normalization

    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)  # necessary!!!
    sc.pp.log1p(adata)  # not as much as corr is rank based

    adata.layers["log1p_norm"] = adata.X.copy()
    return adata

def calc_spatial_corr_multi_donor(adata, metric_key="r", donor_col="donor_id", min_cells=20):

    X_full = adata.layers["log1p_norm"]
    X_full = X_full.toarray() if sparse.issparse(X_full) else np.asarray(X_full, dtype=float)

    donor_rhos  = {}   # donor -> array (n_genes,)
    donor_ncells = {}  # donor -> int

    for donor in adata.obs[donor_col].unique():

        mask = (adata.obs[donor_col] == donor).values
        r    = adata.obs.loc[mask, metric_key].values.astype(float)
        valid = ~np.isnan(r)

        if valid.sum() < min_cells:
            print(f"  Skipping {donor}: only {valid.sum()} valid cells")
            continue

        X = X_full[mask][valid]
        r = r[valid]

        Xr = rankdata(X, axis=0).astype(float)
        yr = rankdata(r).astype(float)
        Xr -= Xr.mean(axis=0)
        yr -= yr.mean()

        denom = np.sqrt((Xr**2).sum(axis=0) * (yr**2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (Xr.T @ yr) / denom

        rho = np.where(denom == 0, 0.0, rho)   # constant gene -> rho = 0, no warning

        donor_rhos[donor]   = rho
        donor_ncells[donor] = valid.sum()

    if len(donor_rhos) == 0:
        raise ValueError("Need at least 1 donor with data")

    genes      = adata.var_names
    donor_list = list(donor_rhos.keys())

    # --- donors x genes raw correlation matrix ---
    rho_matrix = np.array([donor_rhos[d] for d in donor_list])          # (n_donors, n_genes)
    df_raw     = pd.DataFrame(rho_matrix, index=donor_list, columns=genes)

    if len(donor_rhos) == 1:
        # single donor: no cross-donor variance to run Fisher-z + t-test over,
        # so test this donor's rho per gene via the Spearman t-approximation instead
        rho_combined = rho_matrix[0]
        n = donor_ncells[donor_list[0]]
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = rho_combined * np.sqrt((n - 2) / (1 - rho_combined**2))
        pval = 2 * t_dist.sf(np.abs(tstat), df=n - 2)
        pval = np.where(np.abs(rho_combined) >= 1, 0.0, pval)   # |rho|=1 -> tstat undefined, treat as significant
        tau2 = np.full_like(rho_combined, np.nan)
    else:
        # --- Random-effects (DerSimonian-Laird) meta-analysis on Fisher z ---
        z_matrix = np.arctanh(np.clip(rho_matrix, -0.9999, 0.9999))     # (n_donors, n_genes)
        n_cells  = np.array([donor_ncells[d] for d in donor_list], dtype=float)
        w_fixed  = np.clip(n_cells - 3, 1e-6, None)                     # per-donor scalar weight

        z_fixed = (w_fixed[:, None] * z_matrix).sum(axis=0) / w_fixed.sum()
        Q       = (w_fixed[:, None] * (z_matrix - z_fixed) ** 2).sum(axis=0)

        k    = len(donor_list)
        c    = w_fixed.sum() - (w_fixed ** 2).sum() / w_fixed.sum()
        tau2 = np.clip((Q - (k - 1)) / c, 0, None)   # DerSimonian-Laird between-donor variance

        w_re   = 1.0 / (1.0 / w_fixed[:, None] + tau2[None, :])         # (n_donors, n_genes)
        z_mean = (w_re * z_matrix).sum(axis=0) / w_re.sum(axis=0)
        se     = np.sqrt(1.0 / w_re.sum(axis=0))

        rho_combined = np.tanh(z_mean)
        zstat = z_mean / se
        pval  = 2 * norm.sf(np.abs(zstat))

    df_summary = pd.DataFrame({
        "rho"     : rho_combined,
        "pval"    : pval,
        "tau2"    : tau2,
        "z_mean"  : z_mean if len(donor_rhos) > 1 else np.arctanh(np.clip(rho_combined, -0.9999, 0.9999)),
        "se"      : se if len(donor_rhos) > 1 else np.sqrt(1/(donor_ncells[donor_list[0]] - 3)),
        "n_donors": len(donor_rhos),
    }, index=genes)
    df_summary["padj"] = multipletests(df_summary["pval"].fillna(1), method="fdr_bh")[1]
    df_summary = df_summary.sort_values("rho", key=np.abs, ascending=False)

    return df_summary, df_raw

def gene_expr_stats(adata, ct, metric_key="r", donor_col="donor_id", min_cells=20, frac=0.25):
    adata_ct = adata[adata.obs["ct"] == ct]

    counts = adata_ct.layers["counts"]
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)

    X_full = adata_ct.layers["log1p_norm"]
    X_full = X_full.toarray() if sparse.issparse(X_full) else np.asarray(X_full, dtype=float)

    donor_fc = {}
    for donor in adata_ct.obs[donor_col].unique():
        mask  = (adata_ct.obs[donor_col] == donor).values
        r     = adata_ct.obs.loc[mask, metric_key].values.astype(float)
        valid = ~np.isnan(r)
        if valid.sum() < min_cells:
            continue
        X = X_full[mask][valid]
        r = r[valid]

        lo_thr, hi_thr = np.quantile(r, [frac, 1 - frac])
        donor_fc[donor] = X[r >= hi_thr].mean(axis=0) - X[r <= lo_thr].mean(axis=0)

    fc_matrix = np.array(list(donor_fc.values()))  # (n_donors, n_genes)

    return pd.DataFrame({
        "pct_expr":  (counts > 0).mean(axis=0),   # fraction of cells with count > 0
        "mean_expr": counts.mean(axis=0),
        "log2fc":    fc_matrix.mean(axis=0),      # mean across donors of (high-pole - low-pole) log1p_norm expr
    }, index=adata.var_names)

def classify_gradient_agreement(df, rho_x, rho_y, padj_x, padj_y, pexpr_x, pexpr_y,
                                 mexpr_x=None, mexpr_y=None,
                                 threshold=0.2, threshold_specific=0.4, threshold_only=0.1,
                                 padj_thr=0.05, min_pct_expr=0.20, min_mean_expr=0.10,
                                 label_x="x", label_y="y"):
    """Label each gene as strongly_agree / strongly_disagree / only_<x> / only_<y> / other,
    based on whether its spatial-gradient rho clears `threshold` (plus significance and
    expression gates) in each of two conditions being compared."""
    cols = [rho_x, rho_y, padj_x, padj_y, pexpr_x, pexpr_y]
    if mexpr_x is not None:
        cols += [mexpr_x, mexpr_y]
    df = df[cols].dropna(subset=[rho_x, rho_y]).copy()

    x, y = df[rho_x], df[rho_y]
    sig_x, sig_y = df[padj_x] < padj_thr, df[padj_y] < padj_thr
    expr_x = df[pexpr_x] >= min_pct_expr
    expr_y = df[pexpr_y] >= min_pct_expr
    if mexpr_x is not None:
        expr_x = expr_x & (df[mexpr_x] >= min_mean_expr)
        expr_y = expr_y & (df[mexpr_y] >= min_mean_expr)

    has_x, has_y   = (x.abs() >= threshold) & sig_x & expr_x, (y.abs() >= threshold) & sig_y & expr_y
    flat_x, flat_y = x.abs() < threshold_only, y.abs() < threshold_only

    both_strong        = has_x & has_y
    strongly_agree     = both_strong & (np.sign(x) == np.sign(y))
    strongly_disagree  = both_strong & (np.sign(x) != np.sign(y))
    only_x             = (x.abs() >= threshold_specific) & sig_x & expr_x & flat_y
    only_y             = (y.abs() >= threshold_specific) & sig_y & expr_y & flat_x

    only_x_label, only_y_label = f"only_{label_x}", f"only_{label_y}"
    df["type_gene"] = "other"
    df.loc[strongly_agree,    "type_gene"] = "strongly_agree"
    df.loc[strongly_disagree, "type_gene"] = "strongly_disagree"
    df.loc[only_x,  "type_gene"] = only_x_label
    df.loc[only_y,  "type_gene"] = only_y_label
    return df


def plot_gradient_agreement(df, rho_x, rho_y, threshold=0.2, threshold_only=0.1,
                             label_x="x", label_y="y", xlabel=None, ylabel=None, title=None,
                             height=6, annotate_genes=False, n_annotate=10, dpi=None):
    """Jointplot + summary tables for a df already labeled by `classify_gradient_agreement`.

    height         : passed straight to sns.jointplot - controls the figure size.
    annotate_genes : if True, label points with their gene name - same genes, same sort
                     order, same `n_annotate` cap per category as the strongly_disagree /
                     only_x / only_y tables printed below (strongly_agree isn't printed as
                     its own table below, so it isn't annotated either).
    dpi            : resolution of the rendered figure - jointplot has no dpi= kwarg of its
                     own, so this is set on the figure after it's built.
    """
    only_x_label, only_y_label = f"only_{label_x}", f"only_{label_y}"
    palette = {
        "strongly_agree":    "firebrick",
        "strongly_disagree": "steelblue",
        only_x_label:        "darkorange",
        only_y_label:        "seagreen",
        "other":             "lightgray",
    }

    # legend labels: "type_gene" -> "gene type"; underscores -> spaces; first letter only
    # capitalized (plain .capitalize() would lowercase cell-type names like "Matrix")
    pretty = lambda s: (s.replace("_", " ")[:1].upper() + s.replace("_", " ")[1:]) if s else s
    df = df.copy()
    df["_gene_type"] = df["type_gene"].map(pretty)
    palette_pretty = {pretty(k): v for k, v in palette.items()}

    g = sns.jointplot(
        data=df, x=rho_x, y=rho_y,
        hue="_gene_type", palette=palette_pretty,
        alpha=0.4, height=height,
        marginal_kws=dict(fill=True, alpha=0.3),
        ylim=(-1, 1), xlim=(-1, 1)
    )
    if dpi is not None:
        g.figure.set_dpi(dpi)
    for val in [-threshold, threshold]:
        g.ax_joint.axhline(val, color="k",    lw=0.5, ls="--", alpha=0.3)
        g.ax_joint.axvline(val, color="k",    lw=0.5, ls="--", alpha=0.3)
    for val in [-threshold_only, threshold_only]:
        g.ax_joint.axhline(val, color="gray", lw=0.5, ls=":",  alpha=0.4)
        g.ax_joint.axvline(val, color="gray", lw=0.5, ls=":",  alpha=0.4)
    g.ax_joint.axhline(0, color="k", lw=0.7, alpha=0.4)
    g.ax_joint.axvline(0, color="k", lw=0.7, alpha=0.4)
    g.ax_joint.set_xlabel(xlabel or f"{label_x}  ρ")
    g.ax_joint.set_ylabel(ylabel or f"{label_y}  ρ")
    g.figure.suptitle(title or f"{label_x}  vs  {label_y}", y=1.02, fontsize=12)
    if g.ax_joint.legend_ is not None:
        g.ax_joint.legend_.set_title("Gene type")

    if annotate_genes and n_annotate > 0:
        to_annotate = pd.concat([
            df[df["type_gene"] == "strongly_disagree"]
              .sort_values([rho_x, rho_y], key=np.abs, ascending=False).head(n_annotate),
            df[df["type_gene"] == only_x_label]
              .sort_values(rho_x, key=np.abs, ascending=False).head(n_annotate),
            df[df["type_gene"] == only_y_label]
              .sort_values(rho_y, key=np.abs, ascending=False).head(n_annotate),
        ])
        for gene, row in to_annotate.iterrows():
            g.ax_joint.annotate(str(gene), (row[rho_x], row[rho_y]), fontsize=7,
                                 xytext=(3, 3), textcoords="offset points")

    plt.show()

    display(df["type_gene"].value_counts().rename(title or f"{label_x} vs {label_y}"))
    print("strongly_disagree")
    display(df[df["type_gene"] == "strongly_disagree"]
              .sort_values([rho_x, rho_y], key=np.abs, ascending=False))
    print(only_x_label)
    display(df[df["type_gene"] == only_x_label]
              .sort_values(rho_x, key=np.abs, ascending=False))
    print(only_y_label)
    display(df[df["type_gene"] == only_y_label]
              .sort_values(rho_y, key=np.abs, ascending=False))
    
def plot_gene_gradient_shift_combined(df, genes, cell_types=("Matrix", "Patch"), padj_thr=0.05, offset=0.18):
    genes = [g for g in genes if g in df.index]

    value_to_size = lambda x, vmin, vmax: 20 + 300 * (x - vmin) / (vmax - vmin)
    COLOR_LINE   = "#8a8a86"
    COLOR_HEALTH = "green"
    COLOR_XDP    = "red"

    max_change = pd.concat(
        [(df.loc[genes, f"rho_{ct}"] - df.loc[genes, f"rho_{ct}_xdp"]).abs() for ct in cell_types],
        axis=1
    ).max(axis=1)
    order = max_change.sort_values(ascending=True).index
    base_y = np.arange(len(order))

    mean_cols = [f"mean_expr_{ct}" for ct in cell_types] + [f"mean_expr_{ct}_xdp" for ct in cell_types]
    all_means = df.loc[order, mean_cols].values.flatten()
    vmin, vmax = np.nanmin(all_means), np.nanmax(all_means)

    ct_marker   = {cell_types[0]: "o", cell_types[1]: "s"}
    ct_y_offset = {cell_types[0]: +offset, cell_types[1]: -offset}

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(genes) + 1.8))

    for i, yi in enumerate(base_y):
        if i % 2 == 1:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f7f7f5", zorder=0)
    ax.axvline(0, color="#8a8a86", lw=0.7, zorder=1)

    for ct in cell_types:
        y = base_y + ct_y_offset[ct]
        marker = ct_marker[ct]

        rho_h_o = df.loc[order, f"rho_{ct}"].values
        rho_x_o = df.loc[order, f"rho_{ct}_xdp"].values
        h_sig   = (df.loc[order, f"padj_{ct}"].fillna(1) < padj_thr).values
        x_sig   = (df.loc[order, f"padj_{ct}_xdp"].fillna(1) < padj_thr).values
        size_h  = value_to_size(df.loc[order, f"mean_expr_{ct}"].values, vmin, vmax)
        size_x  = value_to_size(df.loc[order, f"mean_expr_{ct}_xdp"].values, vmin, vmax)

        for i in range(len(order)):
            ax.plot([rho_h_o[i], rho_x_o[i]], [y[i], y[i]], color=COLOR_LINE, lw=1.3, zorder=2)

        ax.scatter(rho_h_o[h_sig],  y[h_sig],  s=size_h[h_sig],  marker=marker, color=COLOR_HEALTH, zorder=3)
        ax.scatter(rho_h_o[~h_sig], y[~h_sig], s=size_h[~h_sig], marker=marker,
                   facecolors="white", edgecolors=COLOR_HEALTH, linewidths=1.4, zorder=3)
        ax.scatter(rho_x_o[x_sig],  y[x_sig],  s=size_x[x_sig],  marker=marker, color=COLOR_XDP, zorder=3)
        ax.scatter(rho_x_o[~x_sig], y[~x_sig], s=size_x[~x_sig], marker=marker,
                   facecolors="white", edgecolors=COLOR_XDP, linewidths=1.4, zorder=3)

    ax.set_yticks(base_y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("ρ (spatial gradient)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#ececeb", lw=0.6, zorder=0)

    cond_handles = [
        plt.scatter([], [], s=70, color=COLOR_HEALTH, label="health (sig.)"),
        plt.scatter([], [], s=70, facecolors="white", edgecolors=COLOR_HEALTH, linewidths=1.4, label="health (n.s.)"),
        plt.scatter([], [], s=70, color=COLOR_XDP, label="XDP (sig.)"),
        plt.scatter([], [], s=70, facecolors="white", edgecolors=COLOR_XDP, linewidths=1.4, label="XDP (n.s.)"),
    ]
    shape_handles = [plt.scatter([], [], s=70, marker=ct_marker[ct], color="#52514e", label=ct) for ct in cell_types]

    leg1 = fig.legend(handles=cond_handles, loc="upper left", bbox_to_anchor=(0.0, 1.10),
                       ncol=4, frameon=False, fontsize=9)
    fig.add_artist(leg1)
    fig.legend(handles=shape_handles, loc="upper left", bbox_to_anchor=(0.05, 1.03),
               ncol=2, frameon=False, fontsize=9, title="cell type", title_fontsize=8)

    ref_vals  = np.array([vmin, (vmin + vmax) / 2, vmax])
    ref_sizes = value_to_size(ref_vals, vmin, vmax)
    size_handles = [plt.scatter([], [], s=s, facecolors="none", edgecolors="#52514e", linewidths=1.2) for s in ref_sizes]
    size_labels = [f"{v:.0f}" for v in ref_vals]
    fig.legend(size_handles, size_labels, loc="upper right", bbox_to_anchor=(1.0, 1.03),
               title="mean expr.", frameon=False, fontsize=8, title_fontsize=8,
               labelspacing=1.6, ncol=3, handletextpad=0.3, columnspacing=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.show()


def permutation_test_donor_level(adata, genes, ct, donor_col="donor_id", metric_key="r",
                                  min_cells=20, n_perm=2000, seed=0, layer="log1p_norm",
                                  ci_level=0.95):
    """Empirical null for the meta-analyzed rho, built by permuting `r` within each donor
    (all n_perm permutations batched per donor into one matmul instead of a python loop)."""
    rng = np.random.default_rng(seed)
    adata_ct = adata[adata.obs["ct"] == ct]

    X_full = adata_ct[:, genes].layers[layer]
    X_full = X_full.toarray() if sparse.issparse(X_full) else np.asarray(X_full, dtype=float)
    r_full = adata_ct.obs[metric_key].values.astype(float)
    donors_full = adata_ct.obs[donor_col].values

    donor_data = {}
    for donor in pd.unique(donors_full):
        mask = (donors_full == donor) & ~np.isnan(r_full)
        n = mask.sum()
        if n < min_cells:
            continue
        Xr = rankdata(X_full[mask], axis=0).astype(float)
        Xr -= Xr.mean(axis=0)
        yr0 = rankdata(r_full[mask]).astype(float)          # ranked ONCE - permutations just shuffle this
        yr0 -= yr0.mean()
        donor_data[donor] = dict(Xr=Xr, Xr_ss=(Xr**2).sum(axis=0), yr0=yr0, yr0_ss=(yr0**2).sum(), n=n)

    n_cells   = np.array([d["n"] for d in donor_data.values()], dtype=float)
    n_donors  = len(donor_data)
    n_genes   = len(genes)

    # observed rho per donor (unpermuted yr0)
    rho_obs_list = []
    for d in donor_data.values():
        denom = np.sqrt(d["Xr_ss"] * d["yr0_ss"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_obs_list.append(np.where(denom == 0, 0.0, (d["Xr"].T @ d["yr0"]) / denom))
    rho_obs_matrix = np.array(rho_obs_list)                  # (n_donors, n_genes)

    # ALL permutations per donor at once - one batched matmul instead of a python loop over n_perm
    rho_null = np.empty((n_donors, n_genes, n_perm))
    for di, d in enumerate(donor_data.values()):
        Y = np.broadcast_to(d["yr0"], (n_perm, d["n"])).copy()
        Y = rng.permuted(Y, axis=1)                          # each of the n_perm rows shuffled independently
        denom = np.sqrt(d["Xr_ss"][:, None] * d["yr0_ss"])   # invariant across permutations - computed once
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_null[di] = np.where(denom == 0, 0.0, (d["Xr"].T @ Y.T) / denom)

    def combined_z(rho_matrix, has_perm_axis=False):
        z = np.arctanh(np.clip(rho_matrix, -0.9999, 0.9999))
        w_fixed = np.clip(n_cells - 3, 1e-6, None)
        w_b = w_fixed.reshape(-1, 1, 1) if has_perm_axis else w_fixed.reshape(-1, 1)
        z_fixed = (w_b * z).sum(axis=0) / w_fixed.sum()
        Q = (w_b * (z - z_fixed) ** 2).sum(axis=0)
        c = w_fixed.sum() - (w_fixed**2).sum() / w_fixed.sum()
        tau2 = np.clip((Q - (n_donors - 1)) / c, 0, None) if n_donors > 1 else np.zeros_like(Q)
        w_re = 1.0 / (1.0 / w_b + tau2[None, ...])
        z_mean = (w_re * z).sum(axis=0) / w_re.sum(axis=0)
        se = np.sqrt(1.0 / w_re.sum(axis=0))
        return z_mean, se

    z_obs, se_obs = combined_z(rho_obs_matrix)                # (n_genes,)
    z_null, _     = combined_z(rho_null, has_perm_axis=True)  # (n_genes, n_perm)

    zcrit = norm.ppf(0.5 + ci_level / 2)
    ci_lo = np.tanh(z_obs - zcrit * se_obs)
    ci_hi = np.tanh(z_obs + zcrit * se_obs)

    pval_emp  = np.maximum((np.abs(z_null) >= np.abs(z_obs)[:, None]).mean(axis=1), 1.0 / n_perm)
    pval_bonf = np.minimum(pval_emp * n_genes, 1.0)

    print("Min p-value testable:", 1.0 / n_perm)

    return pd.DataFrame({
        f"z_obs_{ct}": z_obs,
        f"rho_{ct}": np.tanh(z_obs),
        f"ci_lo_{ct}": ci_lo, f"ci_hi_{ct}": ci_hi,
        f"pval_{ct}": pval_emp, f"padj_{ct}": pval_bonf,
    }, index=genes)


def plot_gene_gradient_shift_permutation(df, genes, expr_df, cell_types=("Matrix", "Patch"),
                                          padj_thr=0.05, offset=0.18, sort_col="rho_Patch_matched"):
    genes = [g for g in genes if g in df.index]

    value_to_size = lambda x, vmin, vmax: 20 + 300 * (x - vmin) / (vmax - vmin)
    COLOR_LINE, COLOR_HEALTH, COLOR_XDP = "#8a8a86", "green", "red"

    order = df.loc[genes, sort_col].sort_values(ascending=True).index   # <-- sort by whatever column you pass
    base_y = np.arange(len(order))

    mean_cols = [f"mean_expr_{ct}" for ct in cell_types] + [f"mean_expr_{ct}_xdp" for ct in cell_types]
    all_means = expr_df.loc[order, mean_cols].values.flatten()
    vmin, vmax = np.nanmin(all_means), np.nanmax(all_means)

    ct_marker   = {cell_types[0]: "o", cell_types[1]: "s"}
    ct_y_offset = {cell_types[0]: +offset, cell_types[1]: -offset}

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(genes) + 1.8))
    for i, yi in enumerate(base_y):
        if i % 2 == 1:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f7f7f5", zorder=0)
    ax.axvline(0, color="#8a8a86", lw=0.7, zorder=1)

    for ct in cell_types:
        y = base_y + ct_y_offset[ct]
        marker = ct_marker[ct]

        rho_h = df.loc[order, f"rho_{ct}_matched"].values
        rho_x = df.loc[order, f"rho_{ct}_xdp"].values
        h_sig = (df.loc[order, f"padj_{ct}_matched"] < padj_thr).values
        x_sig = (df.loc[order, f"padj_{ct}_xdp"] < padj_thr).values
        size_h = value_to_size(expr_df.loc[order, f"mean_expr_{ct}"].values, vmin, vmax)
        size_x = value_to_size(expr_df.loc[order, f"mean_expr_{ct}_xdp"].values, vmin, vmax)

        err_h = np.abs(np.vstack([rho_h - df.loc[order, f"ci_lo_{ct}_matched"].values,
                                   df.loc[order, f"ci_hi_{ct}_matched"].values - rho_h]))
        err_x = np.abs(np.vstack([rho_x - df.loc[order, f"ci_lo_{ct}_xdp"].values,
                                   df.loc[order, f"ci_hi_{ct}_xdp"].values - rho_x]))

        ax.errorbar(rho_h, y, xerr=err_h, fmt="none", ecolor=COLOR_HEALTH, elinewidth=1.0, capsize=2, zorder=3)
        ax.errorbar(rho_x, y, xerr=err_x, fmt="none", ecolor=COLOR_XDP,    elinewidth=1.0, capsize=2, zorder=3)

        ax.scatter(rho_h[h_sig],  y[h_sig],  s=size_h[h_sig],  marker=marker, color=COLOR_HEALTH, zorder=4)
        ax.scatter(rho_h[~h_sig], y[~h_sig], s=size_h[~h_sig], marker=marker,
                   facecolors="white", edgecolors=COLOR_HEALTH, linewidths=1.4, zorder=4)
        ax.scatter(rho_x[x_sig],  y[x_sig],  s=size_x[x_sig],  marker=marker, color=COLOR_XDP, zorder=4)
        ax.scatter(rho_x[~x_sig], y[~x_sig], s=size_x[~x_sig], marker=marker,
                   facecolors="white", edgecolors=COLOR_XDP, linewidths=1.4, zorder=4)

    ax.set_yticks(base_y); ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(-0.7, len(order) - 0.3); ax.set_xlim(-1, 1)
    ax.set_xlabel("ρ (spatial gradient), with 95% CI  ·  dot size = mean expr.")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#ececeb", lw=0.6, zorder=0)

    cond_handles = [
        plt.scatter([], [], s=70, color=COLOR_HEALTH, label="health, matched (sig.)"),
        plt.scatter([], [], s=70, facecolors="white", edgecolors=COLOR_HEALTH, linewidths=1.4, label="health, matched (n.s.)"),
        plt.scatter([], [], s=70, color=COLOR_XDP, label="XDP (sig.)"),
        plt.scatter([], [], s=70, facecolors="white", edgecolors=COLOR_XDP, linewidths=1.4, label="XDP (n.s.)"),
    ]
    shape_handles = [plt.scatter([], [], s=70, marker=ct_marker[ct], color="#52514e", label=ct) for ct in cell_types]
    leg1 = fig.legend(handles=cond_handles, loc="upper left", bbox_to_anchor=(0.0, 1.10), ncol=4, frameon=False, fontsize=9)
    fig.add_artist(leg1)
    fig.legend(handles=shape_handles, loc="upper left", bbox_to_anchor=(0.05, 1.03), ncol=2, frameon=False, fontsize=9, title="cell type", title_fontsize=8)

    ref_vals  = np.array([vmin, (vmin + vmax) / 2, vmax])
    ref_sizes = value_to_size(ref_vals, vmin, vmax)
    size_handles = [plt.scatter([], [], s=s, facecolors="none", edgecolors="#52514e", linewidths=1.2) for s in ref_sizes]
    fig.legend(size_handles, [f"{v:.0f}" for v in ref_vals], loc="upper right", bbox_to_anchor=(1.0, 1.03),
               title="mean expr.", frameon=False, fontsize=8, title_fontsize=8, labelspacing=1.6, ncol=3,
               handletextpad=0.3, columnspacing=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.show()


def combine_donor_z(rho_per_donor, n_per_donor):
    """DerSimonian-Laird random-effects combination of per-donor Spearman rho."""
    rho_per_donor = np.asarray(rho_per_donor, dtype=float)
    n_per_donor   = np.asarray(n_per_donor, dtype=float)

    z = np.arctanh(np.clip(rho_per_donor, -0.9999, 0.9999))
    w_fixed = np.clip(n_per_donor - 3, 1e-6, None)

    z_fixed = (w_fixed * z).sum() / w_fixed.sum()
    Q = (w_fixed * (z - z_fixed) ** 2).sum()

    k = len(rho_per_donor)
    if k > 1:
        c = w_fixed.sum() - (w_fixed ** 2).sum() / w_fixed.sum()
        tau2 = max((Q - (k - 1)) / c, 0.0)
    else:
        tau2 = 0.0

    w_re = 1.0 / (1.0 / w_fixed + tau2)
    z_mean = (w_re * z).sum() / w_re.sum()
    se = np.sqrt(1.0 / w_re.sum())

    return dict(rho=np.tanh(z_mean), z=z_mean, se=se,
                pval=2 * norm.sf(np.abs(z_mean / se)), tau2=tau2, n_donors=k)


def donor_rho_for_gene(adata, ct, gene, donor_col="donor_id", metric_key="r", min_cells=20):
    """Per-donor Spearman rho for one gene within a given cell type - no range filtering,
    operates on whatever cells `adata` already contains."""
    adata_ct = adata[adata.obs["ct"] == ct]
    x_all = adata_ct[:, gene].layers["log1p_norm"]
    x_all = x_all.toarray().ravel() if sparse.issparse(x_all) else np.asarray(x_all).ravel()
    r_all = adata_ct.obs[metric_key].values.astype(float)
    donors_all = adata_ct.obs[donor_col].values

    rhos, ns, donors_used = [], [], []
    for donor in pd.unique(donors_all):
        mask = (donors_all == donor) & ~np.isnan(r_all)
        n = mask.sum()
        if n < min_cells:
            continue
        rho, _ = spearmanr(x_all[mask], r_all[mask])
        rhos.append(rho); ns.append(n); donors_used.append(donor)

    return np.array(rhos), np.array(ns), donors_used


def compare_health_vs_xdp_donor_level(adata_matched, xdp_adata_sub, ct, gene,
                                       donor_col="donor_id", metric_key="r", min_cells=20):
    """Compare donor-level gene~r correlation between a caller-prepared, already-matched
    health dataset and XDP. Matching (range/density/whatever) is the caller's job - this
    only runs the donor-level DL combination + difference test on what it's given."""
    rho_xdp_d, n_xdp_d, donors_xdp = donor_rho_for_gene(xdp_adata_sub, ct, gene, donor_col, metric_key, min_cells)
    xdp_combined = combine_donor_z(rho_xdp_d, n_xdp_d)

    rho_h_d, n_h_d, donors_h = donor_rho_for_gene(adata_matched, ct, gene, donor_col, metric_key, min_cells)
    health_combined = combine_donor_z(rho_h_d, n_h_d)

    z_stat = (health_combined["z"] - xdp_combined["z"]) / np.sqrt(health_combined["se"]**2 + xdp_combined["se"]**2)
    p_diff = 2 * norm.sf(np.abs(z_stat))

    return dict(p_diff=p_diff,
                health_restricted=health_combined, health_donors_used=donors_h,
                xdp=xdp_combined, xdp_donors_used=donors_xdp)


def compare_health_vs_xdp(adata, xdp_adata_sub, ct, gene, metric_key="r"):
    """"is rho_health - rho_xdp bigger than what noise alone would produce," not two
    independent tests that each ask "is this one rho nonzero." Cell-pooled (not donor-level),
    with health explicitly restricted to the r-range XDP actually covers."""
    # 1. what r-range (and n) do surviving XDP cells of this type actually cover?
    xdp_sub = xdp_adata_sub[xdp_adata_sub.obs["ct"] == ct]
    x_xdp = xdp_sub[:, gene].layers["log1p_norm"]
    x_xdp = x_xdp.toarray().ravel() if hasattr(x_xdp, "toarray") else np.asarray(x_xdp).ravel()
    r_xdp = xdp_sub.obs[metric_key].values.astype(float)
    rho_xdp, _ = spearmanr(x_xdp, r_xdp)
    n_xdp = len(r_xdp)

    # 2. health, restricted to that SAME r-window - removes the "we lost the dorsal cells" confound
    r_lo, r_hi = r_xdp.min(), r_xdp.max()
    h_sub = adata[(adata.obs["ct"] == ct) & (adata.obs[metric_key] >= r_lo) & (adata.obs[metric_key] <= r_hi)]
    x_h = h_sub[:, gene].layers["log1p_norm"]
    x_h = x_h.toarray().ravel() if hasattr(x_h, "toarray") else np.asarray(x_h).ravel()
    r_h = h_sub.obs[metric_key].values.astype(float)
    rho_h_restricted, _ = spearmanr(x_h, r_h)
    n_h = len(r_h)

    # 3. test whether the two rhos (health-restricted vs xdp) actually differ, given their own n's
    z_h, z_x = np.arctanh(rho_h_restricted), np.arctanh(rho_xdp)
    se_diff = np.sqrt(1/(n_h - 3) + 1/(n_xdp - 3))
    z_stat = (z_h - z_x) / se_diff
    p_diff = 2 * norm.sf(np.abs(z_stat))

    return dict(rho_health_restricted=rho_h_restricted, n_health_restricted=n_h,
                rho_xdp=rho_xdp, n_xdp=n_xdp, p_diff=p_diff)


def qc_vs_r_corr(adata, ct, qc_cols=("nCount_RNA", "nFeature_RNA", "pct_mt"),
                  metric_key="r", donor_col="donor_id", min_cells=20):
    """Does a QC metric (depth, mito%, etc.) itself have a gradient along `r`?
    If so, every gene inherits a bit of that correlation after normalization -
    compare per-donor to see if XDP stands out vs. the healthy donors."""
    adata_ct = adata[adata.obs["ct"] == ct]
    rows = []
    for donor in adata_ct.obs[donor_col].unique():
        obs = adata_ct.obs.loc[adata_ct.obs[donor_col] == donor]
        r = obs[metric_key].astype(float)
        valid = r.notna()
        if valid.sum() < min_cells:
            continue
        row = {"donor": donor, "n_cells": int(valid.sum())}
        for col in qc_cols:
            row[col], _ = spearmanr(obs.loc[valid, col], r[valid])
        rows.append(row)
    return pd.DataFrame(rows).set_index("donor")


# --- significance as a SATURATING gate in [0,1], not an unbounded multiplier ---
def confidence(padj, cap=50.0):
    """cap = the -log10(padj) beyond which extra significance stops mattering."""
    padj = padj.fillna(1.0).clip(lower=1e-300, upper=1.0)   # upper=1 so padj>1 can't flip the sign
    return (-np.log10(padj) / cap).clip(upper=1.0)

def soft_expr_weight(pct_expr, midpoint=0.2, steepness=40):
    return 1.0 / (1.0 + np.exp(-steepness * (pct_expr.fillna(0.0) - midpoint)))

def signed_evidence(rho, padj, pct_expr):
    """Signed effect size, gated by (capped) significance and expression. Stays ~[-1,1]."""
    return rho.fillna(0.0) * confidence(padj) * soft_expr_weight(pct_expr)

def expr_along_r(gene, ct, adata, n_bins=20, donor_col="donor_id", conditions=None):
    """For one gene/cell type, bin cells by r into n_bins and plot, per condition:
       - mean log1p_norm expression per bin (top)
       - fraction of cells with count > 0 per bin (bottom)
    Each donor is binned separately then averaged, so one donor with lots of cells
    can't dominate the curve; thin lines show the individual donors.

    `conditions` defaults to the three standard datasets (health / health_matched / XDP)
    resolved from globals at call time, so this works regardless of where those objects
    are (re)built relative to this cell.
    """
    if conditions is None:
        conditions = (("health", adata, "steelblue"),
                      ("health_matched", adata_matched, "green"),
                      ("XDP", xdp_adata_sub, "firebrick"))

    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for cond_name, ad, color in conditions:
        ad_ct = ad[ad.obs["ct"] == ct]
        if gene not in ad_ct.var_names:
            print(f"{gene} not found in {cond_name} var_names for {ct}")
            continue

        X_norm = ad_ct[:, gene].layers["log1p_norm"]
        X_norm = X_norm.toarray().flatten() if sparse.issparse(X_norm) else np.asarray(X_norm).flatten()

        counts = ad_ct[:, gene].layers["counts"] if "counts" in ad_ct.layers else None
        if counts is not None:
            counts = counts.toarray().flatten() if sparse.issparse(counts) else np.asarray(counts).flatten()

        r      = ad_ct.obs["r"].values.astype(float)
        donors = ad_ct.obs[donor_col].values

        valid = ~np.isnan(r)
        X_norm, r, donors = X_norm[valid], r[valid], donors[valid]
        if counts is not None:
            counts = counts[valid]

        bin_idx = np.clip(np.digitize(r, bins) - 1, 0, n_bins - 1)

        mean_curves = []
        pct_curves  = []
        for donor in np.unique(donors):
            dmask = donors == donor
            mean_row = np.full(n_bins, np.nan)
            pct_row  = np.full(n_bins, np.nan)
            for b in range(n_bins):
                sel = dmask & (bin_idx == b)
                if sel.sum() == 0:
                    continue
                mean_row[b] = X_norm[sel].mean()
                pct_row[b]  = (counts[sel] > 0).mean() if counts is not None else np.nan
            mean_curves.append(mean_row)
            pct_curves.append(pct_row)

        mean_curves = np.array(mean_curves)
        pct_curves  = np.array(pct_curves)

        mean_avg = np.nanmean(mean_curves, axis=0)
        mean_sem = np.nanstd(mean_curves, axis=0) / np.sqrt(np.maximum((~np.isnan(mean_curves)).sum(axis=0), 1))
        ax_mean.plot(bin_centers, mean_avg, color=color, lw=2, label=cond_name)
        ax_mean.fill_between(bin_centers, mean_avg - mean_sem, mean_avg + mean_sem, color=color, alpha=0.2)
        for row in mean_curves:
            ax_mean.plot(bin_centers, row, color=color, lw=0.5, alpha=0.3)

        if counts is not None:
            pct_avg = np.nanmean(pct_curves, axis=0)
            ax_pct.plot(bin_centers, pct_avg, color=color, lw=2, label=cond_name)
            for row in pct_curves:
                ax_pct.plot(bin_centers, row, color=color, lw=0.5, alpha=0.3)

    ax_mean.set_ylabel("mean log1p_norm expr")
    ax_mean.set_title(f"{gene}  -  {ct}")
    ax_mean.legend()
    ax_pct.set_ylabel("fraction cells expr > 0")
    ax_pct.set_xlabel("r  (0 -> 1 along the spline)")
    ax_pct.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def calc_gene_profile_curves(adata, ct, genes, n_bins=20, donor_col="donor_id", metric_key="r",
                              min_cells=20, min_cells_per_bin=10, layer="log1p_norm"):
    """Per-DONOR (not donor-averaged) binned expression profile along the spline, for each
    gene: the binned mean log1p_norm expression curve, its 1st and 2nd derivative along `r`,
    and expression-weighted moments of `r` that summarize the curve's shape.

    `min_cells_per_bin` NaNs out any bin averaged over fewer than that many cells (default
    10) - without this, a bin with e.g. 1-2 cells (common for lowly-expressed genes) can
    spike/crash the curve at a single point and inflate skew_r/kurt_r from pure sampling
    noise, not real localized biology. This is the cell-count analogue of the
    min_donors_per_bin reliability gate in calc_gradient_spread_scores.

    Unlike `calc_gradient_spread_scores` (which pools donors into one reliability-gated
    curve), this keeps every donor's curve separate so donor-to-donor consistency of the
    shape itself can be inspected/plotted.

    Derivatives use `np.gradient` (central differences) rather than `np.diff`, so
    mean_expr/deriv1/deriv2 all stay the same length and share the same bin/`r` axis.

    Moments treat the (non-negative) binned expression curve as a mass distribution over
    `r` and ask where that mass sits:
      mean_r - expression-weighted centroid along r (WHERE expression is concentrated)
      var_r  - spread of that mass around the centroid
      skew_r - asymmetry (>0: mass skewed toward high r / ventral; <0: toward low r / dorsal)
      kurt_r - excess kurtosis (>0: peaked/concentrated in a narrow band; <0: flat/spread out)

    Returns
    -------
    df_curves  : long-format DataFrame, one row per (donor, gene, bin) - columns
                 donor, gene, bin, r, mean_expr, deriv1, deriv2
    df_moments : one row per (donor, gene), indexed ["donor", "gene"] - columns
                 mean_r_<ct>, var_r_<ct>, skew_r_<ct>, kurt_r_<ct> - suffixed with `ct` so
                 moments from different cell types can be merged side by side without
                 colliding (e.g. into a df_corr-style table with per-ct columns).
    """
    adata_ct = adata[adata.obs["ct"] == ct]
    genes = [g for g in genes if g in adata_ct.var_names]

    X_full = adata_ct[:, genes].layers[layer]
    X_full = X_full.toarray() if sparse.issparse(X_full) else np.asarray(X_full, dtype=float)
    r_full = adata_ct.obs[metric_key].values.astype(float)
    donors_full = adata_ct.obs[donor_col].values

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    curve_rows, moment_rows = [], []
    for donor in pd.unique(donors_full):
        mask = (donors_full == donor) & ~np.isnan(r_full)
        if mask.sum() < min_cells:
            continue
        bin_idx = np.clip(np.digitize(r_full[mask], bins) - 1, 0, n_bins - 1)
        Xd = X_full[mask]

        curve = np.full((n_bins, len(genes)), np.nan)
        for b in range(n_bins):
            sel = bin_idx == b
            if sel.sum() >= min_cells_per_bin:
                curve[b] = Xd[sel].mean(axis=0)

        deriv1 = np.gradient(curve, bin_centers, axis=0)
        deriv2 = np.gradient(deriv1, bin_centers, axis=0)

        for gi, gene in enumerate(genes):
            curve_rows.extend(
                (donor, gene, b, bin_centers[b], curve[b, gi], deriv1[b, gi], deriv2[b, gi])
                for b in range(n_bins)
            )

            w = curve[:, gi]
            valid = ~np.isnan(w)
            if valid.sum() < 3 or np.nansum(w[valid]) <= 0:
                moment_rows.append((donor, gene, np.nan, np.nan, np.nan, np.nan))
                continue

            rv, wv = bin_centers[valid], w[valid]
            mean_r = np.average(rv, weights=wv)
            var_r  = np.average((rv - mean_r) ** 2, weights=wv)
            std_r  = np.sqrt(var_r)
            if std_r > 0:
                skew_r = np.average((rv - mean_r) ** 3, weights=wv) / std_r**3
                kurt_r = np.average((rv - mean_r) ** 4, weights=wv) / std_r**4 - 3
            else:
                skew_r = kurt_r = np.nan
            moment_rows.append((donor, gene, mean_r, var_r, skew_r, kurt_r))

    moment_cols = [f"mean_r_{ct}", f"var_r_{ct}", f"skew_r_{ct}", f"kurt_r_{ct}"]

    df_curves = pd.DataFrame(
        curve_rows, columns=["donor", "gene", "bin", "r", "mean_expr", "deriv1", "deriv2"]
    )
    df_moments = pd.DataFrame(
        moment_rows, columns=["donor", "gene", *moment_cols]
    ).set_index(["donor", "gene"])

    return df_curves, df_moments


def plot_gene_profile_curves(df_curves, gene, ct=None, donor_col="donor"):
    """Per-donor mean-expression / 1st-derivative / 2nd-derivative curves along r, for one
    gene (output of `calc_gene_profile_curves`). Thin lines = individual donors, bold =
    across-donor average - visual counterpart of the mean_r/var_r/skew_r/kurt_r moments."""
    df_gene = df_curves[df_curves["gene"] == gene]
    if df_gene.empty:
        print(f"{gene} not found in df_curves")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for donor in df_gene[donor_col].unique():
        d = df_gene[df_gene[donor_col] == donor].sort_values("r")
        ax1.plot(d["r"], d["mean_expr"], color="steelblue", lw=0.6, alpha=0.35)
        ax2.plot(d["r"], d["deriv1"],   color="firebrick", lw=0.6, alpha=0.35)
        ax3.plot(d["r"], d["deriv2"],   color="seagreen",  lw=0.6, alpha=0.35)

    avg = df_gene.groupby("r")[["mean_expr", "deriv1", "deriv2"]].mean()
    ax1.plot(avg.index, avg["mean_expr"], color="steelblue", lw=2.5, label="donor mean")
    ax2.plot(avg.index, avg["deriv1"],   color="firebrick",  lw=2.5)
    ax3.plot(avg.index, avg["deriv2"],   color="seagreen",   lw=2.5)

    for ax in (ax2, ax3):
        ax.axhline(0, color="black", lw=0.7, alpha=0.5)

    ax1.set_ylabel("mean log1p_norm expr")
    ax1.set_title(f"{gene}" + (f"  -  {ct}" if ct else ""))
    ax1.legend()
    ax2.set_ylabel("1st derivative\n(d expr / d r)")
    ax3.set_ylabel("2nd derivative\n(d² expr / d r²)")
    ax3.set_xlabel("r  (0 -> 1 along the spline)")
    plt.tight_layout()
    plt.show()


def plot_gene_profile_moments(df_moments, genes=None):
    """One row per moment (mean_r / var_r / skew_r / kurt_r), genes on the x-axis, one point
    per donor (output of `calc_gene_profile_curves`) - quick look at how consistent a gene's
    along-r shape is across donors. Red dash = across-donor mean for that gene."""
    df = df_moments.reset_index()
    genes = df["gene"].unique().tolist() if genes is None else [g for g in genes if g in df["gene"].unique()]
    df = df[df["gene"].isin(genes)]

    moment_cols = ["mean_r", "var_r", "skew_r", "kurt_r"]
    fig, axes = plt.subplots(len(moment_cols), 1, sharex=True,
                              figsize=(max(0.6 * len(genes), 3) + 2, 3 * len(moment_cols)))
    for ax, m in zip(axes, moment_cols):
        sns.stripplot(data=df, x="gene", y=m, order=genes, ax=ax,
                       color="steelblue", alpha=0.5, jitter=0.15, zorder=1)
        gene_mean = df.groupby("gene")[m].mean().reindex(genes)
        ax.scatter(range(len(genes)), gene_mean.values, color="firebrick",
                   marker="_", s=400, linewidths=2, zorder=2)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_ylabel(m)
        ax.set_xlabel("")

    axes[-1].set_xticks(range(len(genes)))
    axes[-1].set_xticklabels(genes, rotation=45, ha="right")
    fig.suptitle("Per-donor expression-weighted moments of r", y=1.0)
    plt.tight_layout()
    plt.show()