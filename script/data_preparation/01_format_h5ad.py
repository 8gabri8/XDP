import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from scipy.io import mmwrite
from pathlib import Path

# Load data
FILE_PATH = "/home/gdallagl/myworkdir/XDP/data/XDP/disease/diseased_1/diseased_1_adata.h5ad"
print("Reading adata...")
adata = sc.read_h5ad(FILE_PATH)

# Create output directory
output_dir = Path(FILE_PATH).parent / "seurat_export"
output_dir.mkdir(exist_ok=True)
print(f"Exporting to: {output_dir}")

# 1. Export counts matrix as Matrix Market format (FAST for sparse matrices)
print("Exporting counts matrix (Matrix Market format)...")
counts = adata.layers['counts']

# Transpose for R (genes = rows, cells = cols)
if issparse(counts):
    counts_t = counts.T.tocoo()  # Transpose and convert to COO format
else:
    from scipy.sparse import coo_matrix
    counts_t = coo_matrix(counts.T)

# Save as .mtx (Matrix Market format - R can read natively)
mmwrite(output_dir / "counts.mtx", counts_t)
print(f"  Counts shape: {counts_t.shape} (genes × cells)")

# Save gene names (rows)
with open(output_dir / "genes.txt", 'w') as f:
    f.write('\n'.join(adata.var_names))

# Save cell names (columns)
with open(output_dir / "barcodes.txt", 'w') as f:
    f.write('\n'.join(adata.obs_names))

print(f"  Matrix saved efficiently as sparse format")

# 2. Export var (gene metadata)
print("Exporting var (gene metadata)...")
var_cols = ['feature_types', 'genome', 'gene_symbol', 'highly_variable']
var_cols_present = [col for col in var_cols if col in adata.var.columns]
var_df = adata.var[var_cols_present].copy()
var_df.to_csv(output_dir / "var.csv")
print(f"  Var columns: {list(var_df.columns)}")

# 3. Export obs (cell metadata)
print("Exporting obs (cell metadata)...")
obs_cols = ['x', 'y', 'pct_intronic', 'is_cell', 'dbscan_clusters', 'dbscan_score', 'has_spatial', 
            'library', 'library_name',
            'Neighborhood_name', 'Class_name', 'Subclass_name', 'Group_name',
            'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 
            'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt']

obs_cols_present = [col for col in obs_cols if col in adata.obs.columns]
obs_df = adata.obs[obs_cols_present].copy()

# Convert categorical to string
for col in obs_df.columns:
    if pd.api.types.is_categorical_dtype(obs_df[col]):
        obs_df[col] = obs_df[col].astype(str)

obs_df.to_csv(output_dir / "obs.csv")
print(f"  Obs columns: {list(obs_df.columns)}")

# 4. Export UMAP (small, fast)
print("Exporting UMAP...")
np.savetxt(output_dir / "umap.txt", adata.obsm['X_umap'])
print(f"  UMAP shape: {adata.obsm['X_umap'].shape}")

# 5. Export spatial if exists
if 'spatial' in adata.obsm:
    print("Exporting spatial...")
    np.savetxt(output_dir / "spatial.txt", adata.obsm['spatial'])
    print(f"  Spatial shape: {adata.obsm['spatial'].shape}")

print("\n✓ Export complete!")
print(f"Files saved to: {output_dir}")
print("\nFiles created:")
print("  - counts.mtx      (sparse matrix - FAST!)")
print("  - genes.txt       (gene names)")
print("  - barcodes.txt    (cell names)")
print("  - var.csv         (gene metadata)")
print("  - obs.csv         (cell metadata)")
print("  - umap.txt        (UMAP coordinates)")
if 'spatial' in adata.obsm:
    print("  - spatial.txt     (spatial coordinates)")

