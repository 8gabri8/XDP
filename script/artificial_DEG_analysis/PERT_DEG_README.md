# Folders Structure
```text
my-project/
├── artificial_DEG_analysis/
│   ├── analysis_<METHOD>.ipynb          # Notebook for a specific DEG method
│   └── ...                              # Additional analysis notebooks
│
├── data/
│   └── XDP/
│       └── artificial_bican/
│           ├── all_samples/             # Data common to ALL gene sets
│           │   ├── depleted_<CELLTYPE>/ # Cell-type–specific deletion (common across gene sets)
│           │   │   ├── depletion_<CELLTYPE>_all.h5ad
│           │   │   │   # Perturbation applied, no depletion
│           │   │   ├── depletion_<CELLTYPE>_healthy.h5ad
│           │   │   │   # Perturbation applied, healthy cells only (no depletion by definition)
│           │   │   └── <METHOD>/         # Method-specific covariance calculation
│           │   │       └── <METHOD>_all.csv
│           │   │           # Variables calculated for all cells
│           │   │
│           │   ├── zonated_objs_combined_with_md.h5ad
│           │   │   # All samples BEFORE perturbation (no perturbation, no depletion)
│           │   └── gradient_scores.csv
│           │       # Gradient scores for all cells
│           │
│           └── geneset_<ID>/             # A specific perturbed gene set
│               ├── genes.csv             # List of perturbed genes
│               │
│               ├── initial_perturbed/    # Perturbed, NOT depleted
│               │   ├── <sample>.qs       # Original file
│               │   ├── <sample>.h5ad     # Converted version
│               │   └── ...               # Additional conversion artifacts
│               │
│               └── depleted_versions/    # Versions with depleted cells (ATTENTION: they coudl have been a subset of cell types)
│                   └── depleted_<PCT>_<TARGET>/
│                       ├── <sample>.qs
│                       ├── <sample>.h5ad
│                       ├── <sample>_pseudobulked.h5ad
│                       └── <METHOD>/
│                           └── <METHOD>_results.csv
│                               # DEG results for this method

```

# How to run

1. Go to `gs://macosko_data/ferris/bican/simulations/variable_subject_perturbation/spn_type/split_001/geneset_001/` and copy `genes.txt` and `zonated_objs_combined_with_md__combined__rep_001__ventral_matrix_keep_1.0.qs` (data perturbed wiht no depletion) here.

```bash
GENESET=004

FOLDER=/home/gdallagl/myworkdir/XDP/data/XDP/artificial_bican/geneset_${GENESET}/initial_perturbed
mkdir -p "$FOLDER"

gsutil cp \
  gs://macosko_data/ferris/bican/simulations/variable_subject_perturbation/spn_type/split_001/geneset_${GENESET}/genes.csv \
  gs://macosko_data/ferris/bican/simulations/variable_subject_perturbation/spn_type/split_001/geneset_${GENESET}/zonated_objs_combined_with_md__combined__rep_001__dorsal_matrix_keep_1.0.qs \
  "$FOLDER"
```

2. Convert .qs into .h5ad with [notebook](./XDP/script/artificial_DEG_analysis/prepare_data/01_ArtificialBican_qs_conversion.ipynb)


3. Create dataset with specific depletion using [notebook](./XDP/script/artificial_DEG_analysis/prepare_data/02_ArtificialBican_create_depletions_and_gradient-score.ipynb)
    - pertubation (geneset) --> fix for this gene set
    - each datset will have differt %depletion ONLY in DISEASED cells
    - ATTENTION:
        - the dataset coudl be made by only a subset of celltypes (ex onyl matrix cells)
        - the depletion logic could change (in general based on gradient-score)


4. Run specific analysis...
