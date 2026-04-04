import os
from pathlib import Path

"""
    **Example output structure:**
    diseased/
    └── sample_001/
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
DISEASE_TYPE = "diseased" # diseased / healthy
SAMPLE_ID = "sample_04"
BCL = "260313_SL-EXA_0758_B23GHNHLT4"
LIBRARIES = ["SI-TT-C6", "SI-TT-C7", "SI-TT-C8", "SI-TT-C9"]  # List of library names (e.g., SI-TT-B10, SI-TT-B11, SI-TT-B12)

# Create base directory structure
base_path = Path(BASE_PATH) / DISEASE_TYPE / SAMPLE_ID

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

