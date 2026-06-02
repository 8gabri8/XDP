import os
from pathlib import Path

"""
    **Example output structure:**
    SlideTags_single_libraries/
    └── sample_01/
        ├── images/
        ├── seurat_export/
        ├── notebooks/
        └── bcl_run_123/
            ├── lib1/
            │   ├── adata/
            │   ├── cellbender/
            │   ├── map_my_cell/
            │   └── slide_tags/
            ├── lib2/
            │   ├── adata/
            │   ├── cellbender/
            │   ├── map_my_cell/
            │   └── slide_tags/
            └── lib3/
                ├── adata/
                ├── cellbender/
                ├── map_my_cell/
                └── slide_tags/
"""

# Configuration variables
BASE_PATH = "/home/gdallagl/myworkdir/XDP/data/XDP"
SAMPLE_ID = "sample_09"
BCL = "240911_SL-EXD_0362_A22FLV2LT4"
LIBRARIES = ["SI-TT-E1", "SI-TT-E2", "SI-TT-E3", "SI-TT-E4", "SI-TT-E5", "SI-TT-E6", "SI-TT-E7", "SI-TT-E8"]  # List of library names (e.g., SI-TT-B10, SI-TT-B11, SI-TT-B12)

# Create base directory structure
base_path = Path(BASE_PATH) / "SlideTags_single_libraries" / SAMPLE_ID

# Create top-level directories
directories = [
    base_path / "images",
    base_path / "seurat_export",
    base_path / "notebooks",
    base_path / "adata",
    base_path / "zoning",

]

for dir_path in directories:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {dir_path}")

if BCL != "":
    # Create BCL directory with library subdirectories
    bcl_path = base_path / BCL

    for library in LIBRARIES:
        lib_path = bcl_path / library
        
        # Create subdirectories for each library
        subdirs = ["adata", "cellbender", "map_my_cell", "slide_tags", "dropsift"]
        
        for subdir in subdirs:
            subdir_path = lib_path / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {subdir_path}")

print("\nDirectory structure created successfully!")

