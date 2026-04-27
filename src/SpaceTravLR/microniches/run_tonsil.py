from __future__ import annotations

import logging
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from SpaceTravLR.microniches.niche_utils import (
    identify_microniches, visualize_niches
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# Configuration for Tonsil dataset
FEATHER_DIR  = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/runs/tf_lr_tfl__full_2"
H5AD_PATH    = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil.h5ad"
OUT_DIR      = "/tmp/tonsil_func_31genes"

TARGET_GENES = {
    "BCL6", "AICDA", "PAX5", "IRF4", "PRDM1", "FOXO1", "BACH2", "MYBL1",
    "CXCR4", "CXCR5", "SELL", "CD83", "CD86",
    "IL21", "IL7", "IL7R", "IL4", "IL2RA", "ICOS", "PDCD1",
    "CXCL13", "LTB", "LTB4R",
    "FAS", "IL6", "IL6R",
    "CD74", "HLA-DRA", "HLA-DRB1",
    "CXCL12", "CD28",
}

# ──────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load h5ad
    log.info("Loading h5ad …")
    adata = anndata.read_h5ad(H5AD_PATH)
    
    # 2. Run the niche identification pipeline
    # We use the modular identify_microniches function from niche_utils
    z, niche_labels = identify_microniches(
        adata=adata,
        feather_dir=FEATHER_DIR,
        target_genes=TARGET_GENES,
        output_dir=OUT_DIR,
        tfh_cell_type="T_follicular_helper",
        cell_type_col="cell_type",
        ref_annot_col="cell_type_2",
        n_workers=16
    )

    # 3. Visualization
    log.info("Generating plots …")
    visualize_niches(
        adata=adata,
        niche_labels=niche_labels,
        z=z,
        output_dir=OUT_DIR,
        cell_type_col="cell_type",
        ref_annot_col="cell_type_2",
        compute_umap=True
    )
    
    # Optional: Tonsil-specific GC zone reporting
    gc_mask = (adata.obs["cell_type"] == "B_germinal_center")
    if gc_mask.any():
        gc_df = pd.DataFrame({
            "niche": adata.obs["functional_niche"][gc_mask], 
            "ct2": adata.obs["cell_type_2"][gc_mask]
        })
        log.info("\nGC B cells: niche × zone distribution")
        log.info(pd.crosstab(gc_df["niche"], gc_df["ct2"]).to_string())

    log.info(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
