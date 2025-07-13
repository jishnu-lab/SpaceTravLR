import scanpy as sc 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# print("Loading h5ad file...")
# # adata_st = sc.read_h5ad('stardist/cdata.h5ad')
# adata_st = sc.read_h5ad('stardist/cdata_KO.h5ad')
# print(f"Loaded data with shape: {adata_st.shape}")

# print("Processing spatial coordinates...")
# coords = adata_st.obsm['spatial']
# coords = pd.DataFrame(coords, columns = ['x', 'y'], index=adata_st.obs.index)
# coords['object_id'] = adata_st.obs['object_id']
# coords = coords.groupby('object_id').mean()
# print(f"Processed coordinates for {len(coords)} objects")

# print("Processing gene expression data...")
# gex_df = adata_st.to_df(layer=None)
# gex_df['object_id'] = adata_st.obs['object_id']

# print("Cleaning up memory...")
# import gc
# del adata_st
# gc.collect()

# print("Aggregating gene expression by object...")
# gex_df = gex_df.groupby('object_id').sum()
# print(f"Aggregated expression data shape: {gex_df.shape}")

# print("Saving results...")
# gex_df.to_parquet('cdata_KO_gex.parquet')
# coords.to_parquet('cdata_KO_coords.parquet')
# print("Done!")

gex_df = pd.read_parquet('cdata_KO_gex.parquet')
coords = pd.read_parquet('cdata_KO_coords.parquet')


adata_st = sc.AnnData(
    X = gex_df.values,
    var = pd.DataFrame(index = gex_df.columns),
    obs = pd.DataFrame(index = gex_df.index),
    obsm = {'spatial': coords.reindex(gex_df.index, axis=0).values}
)

plt.scatter(coords['x'], coords['y'], s=1, alpha=0.5)
plt.gca().set_aspect('equal')

import numpy as np
import easydict

data = np.load(f'KO_shapes.npz')

shape_4 = easydict.EasyDict({'data': data['shape_4']})
shape_3 = easydict.EasyDict({'data': data['shape_3']})
shape_2 = easydict.EasyDict({'data': data['shape_2']})
shape_1 = easydict.EasyDict({'data': data['shape_1']})


from shapely.geometry import Point, Polygon

def points_in_polygon(polygon_coords, points):
    polygon = Polygon(polygon_coords)
    return np.array([pt for pt in points if polygon.contains(Point(pt))])

adata_coords = {}
coords = adata_st.obsm['spatial']

# data = [shape_1.data, shape_2.data, shape_3.data]
data = [shape_1.data, shape_2.data, shape_3.data, shape_4.data]

for i, polygon_coords in enumerate(data):
    selected_points = points_in_polygon(polygon_coords[0], coords)
    adata_coords[i+1] = selected_points

# takes around 10 min

cell_lookup = pd.DataFrame(
    adata_st.obs_names, index=[f'{x}_{y}'for x, y in adata_st.obsm['spatial']])

adata_1_coords = [f'{x}_{y}' for x, y in adata_coords[1]]
adata_2_coords = [f'{x}_{y}' for x, y in adata_coords[2]]
adata_3_coords = [f'{x}_{y}' for x, y in adata_coords[3]]

adata_1 = adata_st[cell_lookup.loc[adata_1_coords].values.flatten(), :].copy()
adata_2 = adata_st[cell_lookup.loc[adata_2_coords].values.flatten(), :].copy()
adata_3 = adata_st[cell_lookup.loc[adata_3_coords].values.flatten(), :].copy()

adata_4_coords = [f'{x}_{y}' for x, y in adata_coords[4]]
adata_4 = adata_st[cell_lookup.loc[adata_4_coords].values.flatten(), :].copy()


import sys
sys.path.append('../../src')
from spaceoracle.tools.utils import scale_adata

adata_1 = scale_adata(adata_1, cell_size=10)
adata_2 = scale_adata(adata_2, cell_size=10)
adata_3 = scale_adata(adata_3, cell_size=10)
adata_4 = scale_adata(adata_4, cell_size=10)

tmp_dir = '/ix3/djishnu/alw399/SpaceOracle/data/visiumHD_lymph'

adata_1.write_h5ad(f'{tmp_dir}/KO_adata_1.h5ad')
adata_2.write_h5ad(f'{tmp_dir}/KO_adata_2.h5ad')
adata_3.write_h5ad(f'{tmp_dir}/KO_adata_3.h5ad')
adata_4.write_h5ad(f'{tmp_dir}/KO_adata_4.h5ad')

print('Finished all')