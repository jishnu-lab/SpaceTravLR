import scanpy as sc
import anndata
import pandas as pd
import numpy as np

def cluster_embeddings(z, resolutions=(0.2, 0.5, 0.8)):
    """
    Cluster embeddings using Leiden.
    Returns a dictionary of labels indexed by resolution.
    """
    adata = anndata.AnnData(X=z)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    
    results = {}
    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
        results[res] = adata.obs[f'leiden_{res}'].values
        
    return results
