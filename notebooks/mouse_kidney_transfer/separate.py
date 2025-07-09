import scanpy as sc 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

print("Loading h5ad file...")
adata_st = sc.read_h5ad('stardist/cdata.h5ad')
print(f"Loaded data with shape: {adata_st.shape}")

print("Processing spatial coordinates...")
coords = adata_st.obsm['spatial']
coords = pd.DataFrame(coords, columns = ['x', 'y'], index=adata_st.obs.index)
coords['object_id'] = adata_st.obs['object_id']
coords = coords.groupby('object_id').mean()
print(f"Processed coordinates for {len(coords)} objects")

print("Processing gene expression data...")
gex_df = adata_st.to_df(layer=None)
gex_df['object_id'] = adata_st.obs['object_id']

print("Cleaning up memory...")
import gc
del adata_st
gc.collect()

print("Aggregating gene expression by object...")
gex_df = gex_df.groupby('object_id').sum()
print(f"Aggregated expression data shape: {gex_df.shape}")

print("Saving results...")
gex_df.to_parquet('cdata_KO_gex.parquet')
coords.to_parquet('cdata_KO_coords.parquet')
print("Done!")

gex_df = pd.read_parquet('cdata_KO_gex.parquet')
coords = pd.read_parquet('cdata_KO_coords.parquet')

adata_st = sc.AnnData(
    X = gex_df.values,
    var = pd.DataFrame(index = gex_df.columns),
    obs = pd.DataFrame(index = gex_df.index),
    obsm = {'spatial': coords.reindex(gex_df.index, axis=0).values}
)

adata_st
import sys
sys.path.append('../../src')
from spaceoracle.tools.utils import scale_adata

adata_st = scale_adata(adata_st, cell_size=10)
adata_st.write_h5ad(f'mouse_kidney_visiumHD.h5ad')