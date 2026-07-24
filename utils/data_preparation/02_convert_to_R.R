library(Seurat)
library(Matrix)
library(ggplot2)


# Set paths
input_dir <- "/home/gdallagl/myworkdir/XDP/data/XDP/disease/diseased_1/seurat_export"
output_file <- "/home/gdallagl/myworkdir/XDP/data/XDP/disease/diseased_1/diseased_1_seurat.rds"

cat("Loading data from:", input_dir, "\n")

# 1. Read sparse matrix (FAST!)
cat("Reading sparse counts matrix...\n")
counts <- readMM(file.path(input_dir, "counts.mtx"))

# Read gene and cell names
genes <- readLines(file.path(input_dir, "genes.txt"))
barcodes <- readLines(file.path(input_dir, "barcodes.txt"))

# Assign names
rownames(counts) <- genes
colnames(counts) <- barcodes

cat("  Counts dimensions:", dim(counts), "\n")
cat("  Matrix class:", class(counts), "\n")

# 2. Read metadata
cat("Reading cell metadata...\n")
metadata <- read.csv(
  file.path(input_dir, "obs.csv"),
  row.names = 1,
  check.names = FALSE
)
cat("  Metadata dimensions:", dim(metadata), "\n")

# 3. Read var (gene metadata)
cat("Reading gene metadata...\n")
gene_metadata <- read.csv(
  file.path(input_dir, "var.csv"),
  row.names = 1,
  check.names = FALSE
)
cat("  Gene metadata dimensions:", dim(gene_metadata), "\n")

# 4. Read UMAP
cat("Reading UMAP...\n")
umap_coords <- read.table(file.path(input_dir, "umap.txt"))
rownames(umap_coords) <- barcodes
colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
cat("  UMAP dimensions:", dim(umap_coords), "\n")

# 5. Read spatial if exists
spatial_file <- file.path(input_dir, "spatial.txt")
if (file.exists(spatial_file)) {
  cat("Reading spatial...\n")
  spatial_coords <- read.table(spatial_file)
  rownames(spatial_coords) <- barcodes
  colnames(spatial_coords) <- c("spatial_1", "spatial_2")
  cat("  Spatial dimensions:", dim(spatial_coords), "\n")
}

# 6. Create Seurat object
cat("\nCreating Seurat object...\n")
seurat_obj <- CreateSeuratObject(
  counts = counts,
  meta.data = metadata,
  project = "XDP_diseased_1"
)

# 7. Add gene metadata (Seurat v5 compatible way)
cat("Adding gene metadata...\n")
# In Seurat v5, add gene metadata to the var slot directly
for (col in colnames(gene_metadata)) {
  seurat_obj[["RNA"]][[col]] <- gene_metadata[[col]][rownames(seurat_obj)]
}

# 8. Add UMAP
cat("Adding UMAP...\n")
seurat_obj[["umap"]] <- CreateDimReducObject(
  embeddings = as.matrix(umap_coords),
  key = "UMAP_",
  assay = "RNA"
)

# 9. Add spatial if exists
if (exists("spatial_coords")) {
  cat("Adding spatial...\n")
  seurat_obj[["spatial"]] <- CreateDimReducObject(
    embeddings = as.matrix(spatial_coords),
    key = "spatial_",
    assay = "RNA"
  )
}

# 10. Summary
cat("\n=== Seurat Object Summary ===\n")
print(seurat_obj)
cat("\nAssays:", names(seurat_obj@assays), "\n")
cat("Reductions:", names(seurat_obj@reductions), "\n")
cat("Metadata columns:", ncol(seurat_obj@meta.data), "\n")

# Check gene metadata was added
cat("\nGene metadata columns in RNA assay:\n")
print(colnames(seurat_obj[["RNA"]][[]][]))

# 11. Save
cat("\nSaving Seurat object...\n")
saveRDS(seurat_obj, output_file)

cat("\n✓ Conversion complete!\n")
cat("Seurat object saved as:", output_file, "\n")

# Quick verification
cat("\n=== Quick Verification ===\n")
if ("Class_name" %in% colnames(seurat_obj@meta.data)) {
  cat("Cell types found:", length(unique(seurat_obj$Class_name)), "\n")
  cat("Cell type counts:\n")
  print(table(seurat_obj$Class_name))
}

# Check dimensions
cat("Genes:", nrow(seurat_obj), "\nCells:", ncol(seurat_obj), "\n\n")

# Extract data and check
umap_df <- as.data.frame(seurat_obj[["umap"]]@cell.embeddings)
umap_df$Class_name <- seurat_obj$Class_name

cat("UMAP data dimensions:", dim(umap_df), "\n")
cat("First few rows:\n")
print(head(umap_df))
cat("\nClass_name levels:", length(unique(umap_df$Class_name)), "\n")

# Simple plot saved directly
png("/home/gdallagl/myworkdir/XDP/data/XDP/disease/diseased_1/seurat_export/umap_test.png", 
    width = 12, height = 10, units = "in", res = 300)

# Create color palette
colors <- rainbow(length(unique(umap_df$Class_name)))
names(colors) <- unique(umap_df$Class_name)
plot_colors <- colors[umap_df$Class_name]

# Simple scatter
plot(umap_df$UMAP_1, umap_df$UMAP_2, 
     col = plot_colors, 
     pch = 20, cex = 0.5,
     xlab = "UMAP_1", ylab = "UMAP_2",
     main = "UMAP by Cell Type")

dev.off()

cat("\n✓ Saved: umap_test.png\n")

# Spatial
spatial_df <- as.data.frame(seurat_obj[["spatial"]]@cell.embeddings)
spatial_df$Class_name <- seurat_obj$Class_name

png("/home/gdallagl/myworkdir/XDP/data/XDP/disease/diseased_1/seurat_export/spatial_test.png", 
    width = 14, height = 10, units = "in", res = 300)

plot_colors <- colors[spatial_df$Class_name]
plot(spatial_df$spatial_1, spatial_df$spatial_2, 
     col = plot_colors, 
     pch = 20, cex = 0.5,
     xlab = "Spatial_1", ylab = "Spatial_2",
     main = "Spatial by Cell Type",
     asp = 1)

dev.off()

cat("✓ Saved: spatial_test.png\n")