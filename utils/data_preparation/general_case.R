library(qs)
library(Matrix)

q = qread(path.qs)

# 1. Extract raw counts (sparse matrix)
counts = q@assays[["RNA"]]@layers[["counts"]]

# 2. Extract metadata
metadata <- q@meta.data

# 3. Get gene and cell names
genes <- rownames(q[["RNA"]]) 
cells <- colnames(q)

# 4. Save everything
# Sparse matrix (memory efficient!)
writeMM(counts, file = "counts_matrix.mtx")

# Gene names
write.csv(data.frame(gene = genes), file = "genes.csv", row.names = FALSE)

# Cell barcodes
write.csv(data.frame(barcode = cells), file = "barcodes.csv", row.names = FALSE)

# Metadata (keep cell barcodes as rownames)
write.csv(metadata, file = "metadata.csv", row.names = TRUE)



# import scanpy as sc
# import pandas as pd
# from scipy.io import mmread

# # Read sparse matrix (transpose because mtx is transposed)
# counts = mmread('counts_matrix.mtx').T.tocsr()

# # Read annotations
# genes = pd.read_csv('genes.csv')
# barcodes = pd.read_csv('barcodes.csv')
# metadata = pd.read_csv('metadata.csv', index_col=0)

# # Create AnnData
# adata = sc.AnnData(X=counts)
# adata.var_names = genes['gene'].values
# adata.obs_names = barcodes['barcode'].values
# adata.obs = metadata

# # Optional: Add dimensional reductions
# adata.obsm['X_pca'] = pd.read_csv('pca.csv', index_col=0).values
# adata.obsm['X_umap'] = pd.read_csv('umap.csv', index_col=0).values

# # Save as h5ad
# adata.write('data.h5ad')

# print(adata)






